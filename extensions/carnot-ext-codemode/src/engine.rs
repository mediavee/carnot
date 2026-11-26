//! Isolated `QuickJS` async execution engine for `carnot-ext-codemode`.
//!
//! Runs each script inside a fresh `rquickjs::AsyncRuntime` on a dedicated worker thread with:
//! - Hard memory limit (`set_memory_limit`) and stack limit (`set_max_stack_size`)
//! - Cooperative interrupt handler (`set_interrupt_handler`) killing tight loops on timeout or cancel
//! - True multi-threaded concurrency for `Promise.all` via `tokio::runtime::Handle::spawn`
//! - Transactional session `store()`/`load()` committed only when the script succeeds

use crate::{CodeModeConfig, CodeModeHost};
use carnot_core::{ToolOutput, ToolResultBlock, ToolSpec};
use rquickjs::{
    AsyncContext, AsyncRuntime, CatchResultExt, CaughtError, Function, Promise, prelude::Async,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

const MAX_STORE_ENTRY_BYTES: usize = 256 * 1024;
const EXIT_SENTINEL: &str = "__CARNOT_CODEMODE_EXIT__";

const PRELUDE: &str = r#"
(() => {
  const fmt = (v) => {
    if (typeof v === "string") return v;
    if (v instanceof Error) {
      return v.stack ? `${v.name}: ${v.message}\n${v.stack}` : `${v.name}: ${v.message}`;
    }
    try {
      const s = JSON.stringify(v);
      return s === undefined ? String(v) : s;
    } catch (_) {
      return String(v);
    }
  };
  globalThis.__carnotFmt = fmt;

  const logFn = (level) => (...args) => __log(level, args.map(fmt).join(" "));
  globalThis.console = {
    log: logFn("log"),
    info: logFn("info"),
    warn: logFn("warn"),
    error: logFn("error"),
    debug: logFn("debug"),
  };

  globalThis.text = (v) => {
    if (v !== undefined) __text(fmt(v));
  };

  globalThis.exit = (v) => {
    if (v !== undefined) __text(fmt(v));
    __exit();
    // Backward branch triggers QuickJS's interrupt handler immediately, raising an uncatchable
    // internal interruption that `try/catch` blocks cannot swallow.
    while (true) {}
  };

  const call = async (name, args) => {
    const payload = args === undefined ? "{}" : JSON.stringify(args);
    const raw = await __callTool(String(name), payload);
    const r = JSON.parse(raw);
    if (!r.ok) throw new Error(`${name}: ${r.error}`);
    return r.value;
  };
  globalThis.callTool = call;
  globalThis.tools = new Proxy({}, {
    get(_, name) {
      if (typeof name !== "string" || name === "then" || name === "toJSON") return undefined;
      return (args) => call(name, args);
    },
    has(_, name) {
      return typeof name === "string" && __hasTool(name);
    },
    ownKeys() {
      return JSON.parse(__listTools()).map((t) => t.name);
    },
    getOwnPropertyDescriptor(_, name) {
      return typeof name === "string" && __hasTool(name)
        ? { enumerable: true, configurable: true }
        : undefined;
    },
  });

  globalThis.listTools = (filter) => {
    const all = JSON.parse(__listTools());
    if (!filter) return all;
    const f = String(filter).toLowerCase();
    return all.filter(
      (t) => t.name.toLowerCase().includes(f) || t.description.toLowerCase().includes(f)
    );
  };

  globalThis.searchTools = (query, limit) => {
    const lim = typeof limit === "number" && limit > 0 ? Math.floor(limit) : 8;
    return JSON.parse(__searchTools(String(query ?? ""), lim));
  };

  globalThis.describeTool = (name) => {
    const raw = __describeTool(String(name));
    return raw === "" ? undefined : JSON.parse(raw);
  };

  Object.defineProperty(globalThis, "ALL_TOOLS", {
    get() {
      return globalThis.listTools();
    },
    enumerable: true,
  });

  globalThis.store = (key, value) => {
    const k = String(key);
    if (value === undefined) {
      __storeSet(k, "", true);
      return undefined;
    }
    const serialized = JSON.stringify(value);
    if (serialized === undefined) {
      __storeSet(k, "", true);
      return undefined;
    }
    const err = __storeSet(k, serialized, false);
    if (err !== "") throw new Error(err);
    return value;
  };

  globalThis.load = (key) => {
    const raw = __storeGet(String(key));
    return raw === "" ? undefined : JSON.parse(raw);
  };

  globalThis.storedKeys = () => JSON.parse(__storeKeys());

  // Built-in pure-sandbox stdlib: path, base64, blake3, CSV/TSV, globMatch, Read helpers.
  const normPath = (p) => {
    const s = String(p ?? "").replace(/\\/g, "/");
    if (!s) return ".";
    const isAbs = s.startsWith("/");
    const parts = s.split("/");
    const out = [];
    for (const seg of parts) {
      if (!seg || seg === ".") continue;
      if (seg === "..") {
        if (out.length && out[out.length - 1] !== "..") out.pop();
        else if (!isAbs) out.push("..");
      } else {
        out.push(seg);
      }
    }
    const res = (isAbs ? "/" : "") + out.join("/");
    return res || (isAbs ? "/" : ".");
  };
  globalThis.path = Object.freeze({
    normalize: normPath,
    isAbsolute: (p) => String(p ?? "").startsWith("/"),
    join: (...parts) => normPath(parts.filter((x) => x != null && String(x) !== "").join("/")),
    dirname: (p) => {
      const n = normPath(p).replace(/\/+$/, "");
      if (!n || n === "/") return n || ".";
      const idx = n.lastIndexOf("/");
      return idx < 0 ? "." : idx === 0 ? "/" : n.slice(0, idx);
    },
    basename: (p, ext) => {
      const n = normPath(p).replace(/\/+$/, "");
      const idx = n.lastIndexOf("/");
      let b = idx < 0 ? n : n.slice(idx + 1);
      if (ext && b.endsWith(ext)) b = b.slice(0, b.length - ext.length);
      return b;
    },
    extname: (p) => {
      const b = globalThis.path.basename(p);
      const idx = b.lastIndexOf(".");
      return idx <= 0 ? "" : b.slice(idx);
    },
  });

  globalThis.base64Encode = (s) => __base64Encode(String(s ?? ""));
  globalThis.base64Decode = (s) => {
    const r = JSON.parse(__base64Decode(String(s ?? "")));
    if (!r.ok) throw new Error(r.error);
    return r.value;
  };
  globalThis.blake3 = (s) => __blake3(String(s ?? ""));

  globalThis.parseCsv = (input, delimiter = ",") => {
    const src = String(input ?? "");
    const rows = [];
    let row = [];
    let field = "";
    let inQuotes = false;
    for (let i = 0; i < src.length; i++) {
      const ch = src[i];
      if (inQuotes) {
        if (ch === '"') {
          if (src[i + 1] === '"') { field += '"'; i++; }
          else inQuotes = false;
        } else {
          field += ch;
        }
      } else if (ch === '"') {
        inQuotes = true;
      } else if (ch === delimiter) {
        row.push(field);
        field = "";
      } else if (ch === "\n" || ch === "\r") {
        if (ch === "\r" && src[i + 1] === "\n") i++;
        row.push(field);
        if (row.length > 1 || row[0] !== "") rows.push(row);
        row = [];
        field = "";
      } else {
        field += ch;
      }
    }
    if (field !== "" || row.length > 0) {
      row.push(field);
      rows.push(row);
    }
    return rows;
  };
  globalThis.parseTsv = (input) => globalThis.parseCsv(input, "\t");

  globalThis.globMatch = (pattern, text) => {
    const p = String(pattern ?? "");
    const t = String(text ?? "");
    const compile = (src) => {
      let re = "";
      for (let i = 0; i < src.length; i++) {
        const c = src[i];
        if (c === "*") {
          if (src[i + 1] === "*") {
            if (src[i + 2] === "/") { re += "(?:.*/)?"; i += 2; }
            else { re += ".*"; i += 1; }
          } else {
            re += "[^/]*";
          }
        } else if (c === "?") {
          re += "[^/]";
        } else if (c === "{") {
          const end = src.indexOf("}", i + 1);
          if (end > i) {
            const alts = src.slice(i + 1, end).split(",").map(compile);
            re += "(?:" + alts.join("|") + ")";
            i = end;
          } else {
            re += "\\{";
          }
        } else {
          re += c.replace(/[.+^$|()[\]\\]/g, "\\$&");
        }
      }
      return re;
    };
    return new RegExp("^" + compile(p) + "$").test(t);
  };

  globalThis.stripLineNumbers = (raw) =>
    String(raw ?? "")
      .split("\n")
      .map((l) => l.replace(/^\d+\t/, ""))
      .join("\n");

  globalThis.readLines = async (filePath, opts) => {
    const raw = await call("Read", Object.assign({ file_path: String(filePath) }, opts || {}));
    return globalThis.stripLineNumbers(raw).split("\n");
  };

  globalThis.readJson = async (filePath) => {
    const raw = await call("Read", { file_path: String(filePath) });
    return JSON.parse(globalThis.stripLineNumbers(raw));
  };

  // Standard Web pure-data built-ins (zero I/O, spec-compliant):
  // Preserve QuickJS-ng's native Latin-1 `btoa`/`atob` if present; otherwise provide a strict Latin-1 fallback.
  if (typeof globalThis.btoa !== "function") {
    globalThis.btoa = (s) => {
      const str = String(s);
      const res = JSON.parse(__btoaLatin1(str));
      if (!res.ok) throw new Error(res.error);
      return res.value;
    };
  }
  if (typeof globalThis.atob !== "function") {
    globalThis.atob = (s) => {
      const res = JSON.parse(__atobLatin1(String(s)));
      if (!res.ok) throw new Error(res.error);
      return res.value;
    };
  }

  globalThis.structuredClone = (value) => {
    const seen = new WeakMap();
    const clone = (v) => {
      if (v === null || typeof v !== "object") return v;
      if (seen.has(v)) return seen.get(v);
      if (v instanceof Date) return new Date(v.getTime());
      if (v instanceof RegExp) return new RegExp(v.source, v.flags);
      if (v instanceof ArrayBuffer) return v.slice(0);
      if (ArrayBuffer.isView(v)) {
        const buf = v.buffer.slice(v.byteOffset, v.byteOffset + v.byteLength);
        return new v.constructor(buf);
      }
      if (v instanceof Map) {
        const out = new Map();
        seen.set(v, out);
        for (const [k, val] of v.entries()) out.set(clone(k), clone(val));
        return out;
      }
      if (v instanceof Set) {
        const out = new Set();
        seen.set(v, out);
        for (const val of v.values()) out.add(clone(val));
        return out;
      }
      if (Array.isArray(v)) {
        const out = new Array(v.length);
        seen.set(v, out);
        for (let i = 0; i < v.length; i++) out[i] = clone(v[i]);
        return out;
      }
      const out = {};
      seen.set(v, out);
      for (const k of Object.keys(v)) out[k] = clone(v[k]);
      return out;
    };
    return clone(value);
  };

  globalThis.crypto = Object.freeze({
    randomUUID: () => __randomUUID(),
  });

  globalThis.TextEncoder = class TextEncoder {
    encode(input = "") {
      const s = String(input);
      const wellFormed = typeof s.toWellFormed === "function" ? s.toWellFormed() : s;
      return new Uint8Array(__utf8Encode(wellFormed));
    }
  };

  globalThis.TextDecoder = class TextDecoder {
    decode(bytes) {
      if (!bytes) return "";
      const view = bytes instanceof Uint8Array
        ? bytes
        : ArrayBuffer.isView(bytes)
          ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)
          : bytes instanceof ArrayBuffer
            ? new Uint8Array(bytes)
            : new Uint8Array(bytes);
      return __utf8Decode(Array.from(view));
    }
  };

  globalThis.URLSearchParams = class URLSearchParams {
    constructor(init) {
      this._entries = [];
      if (typeof init === "string") {
        const raw = init.startsWith("?") ? init.slice(1) : init;
        if (raw) {
          for (const part of raw.split("&")) {
            if (!part) continue;
            const eq = part.indexOf("=");
            const k = decodeURIComponent((eq >= 0 ? part.slice(0, eq) : part).replace(/\+/g, " "));
            const v = eq >= 0 ? decodeURIComponent(part.slice(eq + 1).replace(/\+/g, " ")) : "";
            this._entries.push([k, v]);
          }
        }
      } else if (Array.isArray(init)) {
        for (const [k, v] of init) this._entries.push([String(k), String(v)]);
      } else if (init && typeof init === "object") {
        for (const [k, v] of Object.entries(init)) this._entries.push([k, String(v)]);
      }
    }
    get(k) { const hit = this._entries.find(([key]) => key === String(k)); return hit ? hit[1] : null; }
    getAll(k) { return this._entries.filter(([key]) => key === String(k)).map(([, v]) => v); }
    has(k) { return this._entries.some(([key]) => key === String(k)); }
    append(k, v) { this._entries.push([String(k), String(v)]); }
    set(k, v) {
      const ks = String(k), vs = String(v);
      let set = false;
      this._entries = this._entries.filter(([key]) => key !== ks || (!set && (set = true)));
      if (set) { const idx = this._entries.findIndex(([key]) => key === ks); this._entries[idx][1] = vs; }
      else this._entries.push([ks, vs]);
    }
    delete(k) { const ks = String(k); this._entries = this._entries.filter(([key]) => key !== ks); }
    entries() { return this._entries[Symbol.iterator](); }
    keys() { return this._entries.map(([k]) => k)[Symbol.iterator](); }
    values() { return this._entries.map(([, v]) => v)[Symbol.iterator](); }
    [Symbol.iterator]() { return this.entries(); }
    toString() {
      return this._entries.map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join("&");
    }
  };

  globalThis.URL = class URL {
    constructor(input, base) {
      const res = JSON.parse(__parseUrl(String(input), base === undefined ? "" : String(base)));
      if (!res.ok) throw new TypeError(res.error);
      this.protocol = res.protocol;
      this.username = res.username;
      this.password = res.password;
      this.host = res.host;
      this.hostname = res.hostname;
      this.port = res.port;
      this.origin = res.origin;
      this.pathname = res.pathname;
      this.search = res.search;
      this.hash = res.hash;
      this.searchParams = new globalThis.URLSearchParams(res.params);
      this._baseHref = res.href;
    }
    get href() {
      const q = this.searchParams.toString();
      const auth = this.username ? (this.password ? `${this.username}:${this.password}@` : `${this.username}@`) : "";
      if (this.host) {
        return `${this.protocol}//${auth}${this.host}${this.pathname}${q ? "?" + q : ""}${this.hash}`;
      }
      return this._baseHref;
    }
    toString() { return this.href; }
  };
})();
"#;

#[derive(Debug)]
pub(crate) struct RunOutput {
    pub(crate) ok: bool,
    pub(crate) return_value: Option<String>,
    pub(crate) text_items: Vec<String>,
    pub(crate) console_lines: Vec<String>,
    pub(crate) logs_truncated: bool,
    pub(crate) tool_calls: usize,
    pub(crate) tool_names_called: Vec<String>,
    pub(crate) error: Option<String>,
    pub(crate) duration_ms: u64,
    pub(crate) next_store: Option<BTreeMap<String, Value>>,
}

/// Converts a tool name (e.g. `mcp__dev-radius__search`) into a valid JS identifier
/// (`mcp__dev_radius__search`) so `tools.<ident>(args)` resolves cleanly.
pub(crate) fn sanitize_js_ident(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (idx, ch) in name.chars().enumerate() {
        let valid = if idx == 0 {
            ch.is_ascii_alphabetic() || ch == '_' || ch == '$'
        } else {
            ch.is_ascii_alphanumeric() || ch == '_' || ch == '$'
        };
        out.push(if valid { ch } else { '_' });
    }
    if out.is_empty() { "_".into() } else { out }
}

/// Strips a single outer markdown code fence (```js ... ```) if a model wraps the snippet.
pub(crate) fn strip_outer_code_fence(code: &str) -> &str {
    let trimmed = code.trim();
    if let Some(rest) = trimmed.strip_prefix("```")
        && let Some(nl) = rest.find('\n')
        && let Some(body) = rest[nl + 1..].strip_suffix("```")
    {
        return body.trim();
    }
    trimmed
}

fn is_forbidden_inner_tool(name: &str) -> bool {
    matches!(name, "CodeMode" | "CallDeferredTool")
}

fn is_core_text_tool(name: &str) -> bool {
    matches!(name, "Read" | "Edit" | "Write" | "Grep" | "Bash")
}

/// Converts a [`ToolOutput`] into a JS-ready JSON payload for `await tools.<name>(args)`.
///
/// - Failed calls become `{ ok: false, error }`.
/// - Image results (e.g. `Read` on PNG/JPEG) become `{ ok: true, value: { type: "image", ... } }`.
/// - Non-core tools (MCP / extensions) whose text is a JSON object or array auto-parse to JS objects/arrays.
/// - Core text tools (`Read`, `Edit`, `Write`, `Grep`, `Bash`) always return strings.
fn encode_tool_output_for_js(tool_name: &str, out: &ToolOutput) -> String {
    if out.is_error {
        return json!({ "ok": false, "error": out.text_only() }).to_string();
    }
    for block in &out.blocks {
        if let ToolResultBlock::Image {
            media_type,
            data_base64,
        } = block
        {
            return json!({
                "ok": true,
                "value": {
                    "type": "image",
                    "media_type": media_type,
                    "data_base64": data_base64,
                    "note": out.text_only(),
                }
            })
            .to_string();
        }
    }
    let text = out.text_only();
    let trimmed = text.trim();
    if !is_core_text_tool(tool_name)
        && ((trimmed.starts_with('{') && trimmed.ends_with('}'))
            || (trimmed.starts_with('[') && trimmed.ends_with(']')))
        && let Ok(parsed) = serde_json::from_str::<Value>(trimmed)
    {
        return json!({ "ok": true, "value": parsed }).to_string();
    }
    json!({ "ok": true, "value": text }).to_string()
}

/// Simple token-overlap relevance search over tool specs (mirrors Carnot's deferred tool search).
fn rank_tools(specs: &[ToolSpec], query: &str, limit: usize) -> Vec<Value> {
    let q_lower = query.to_ascii_lowercase();
    let tokens: Vec<&str> = q_lower
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
        .collect();

    let mut scored: Vec<(usize, &ToolSpec)> = specs
        .iter()
        .filter_map(|spec| {
            if tokens.is_empty() {
                return Some((1, spec));
            }
            let name_l = spec.name.to_ascii_lowercase();
            let desc_l = spec.description.to_ascii_lowercase();
            let mut score = 0usize;
            if name_l == q_lower {
                score += 100;
            } else if name_l.contains(&q_lower) {
                score += 40;
            }
            for tok in &tokens {
                if name_l.contains(tok) {
                    score += 10;
                }
                if desc_l.contains(tok) {
                    score += 3;
                }
            }
            (score > 0).then_some((score, spec))
        })
        .collect();

    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
    scored
        .into_iter()
        .take(limit.max(1))
        .map(|(_, s)| {
            json!({
                "name": s.name,
                "js_name": sanitize_js_ident(&s.name),
                "description": s.description,
                "deferred": s.defer_loading,
            })
        })
        .collect()
}

struct SharedState {
    host: Arc<dyn CodeModeHost>,
    outer: tokio::runtime::Handle,
    parent_call_id: String,
    child_cancel: CancellationToken,
    cancelled_flag: Arc<AtomicBool>,
    timed_out: Arc<AtomicBool>,
    exited: Arc<AtomicBool>,
    timeout: Duration,
    concurrency: Arc<tokio::sync::Semaphore>,
    specs: Vec<ToolSpec>,
    /// Lookup map from exact name AND sanitized JS identifier -> canonical tool name.
    name_map: HashMap<String, String>,
    store: Mutex<BTreeMap<String, Value>>,
    text_items: Mutex<(Vec<String>, usize, bool)>,
    logs: Mutex<(Vec<String>, usize, bool)>,
    return_value: Mutex<Option<String>>,
    tool_calls: AtomicUsize,
    tool_names_called: Mutex<Vec<String>>,
    config: CodeModeConfig,
}

impl SharedState {
    fn resolve_tool_name(&self, raw: &str) -> Option<String> {
        self.name_map.get(raw).cloned()
    }

    /// Returns up to `max` fuzzy-matched tool names (case-insensitive exact/prefix/substring +
    /// bounded Levenshtein distance) so mistyped `tools.<name>` calls get immediate suggestions.
    fn suggest_tool_names(&self, raw: &str, max: usize) -> Vec<String> {
        let q = raw.to_ascii_lowercase();
        let mut scored: Vec<(usize, String)> = self
            .specs
            .iter()
            .filter(|s| !is_forbidden_inner_tool(&s.name))
            .filter_map(|s| {
                let js_name = sanitize_js_ident(&s.name);
                let low = js_name.to_ascii_lowercase();
                let score = if low == q {
                    0
                } else if low.starts_with(&q) || q.starts_with(&low) {
                    1
                } else if low.contains(&q) || q.contains(&low) {
                    2
                } else {
                    let threshold = (q.len().max(low.len()) / 2).clamp(2, 5);
                    2 + bounded_edit_distance(&q, &low, threshold)?
                };
                Some((score, js_name))
            })
            .collect();
        scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        scored.truncate(max);
        scored.into_iter().map(|(_, n)| n).collect()
    }

    fn push_log(&self, level: &str, msg: String) {
        let mut g = self
            .logs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (lines, bytes, truncated) = &mut *g;
        if *truncated {
            return;
        }
        let formatted = if level == "log" || level == "info" {
            msg
        } else {
            format!("[{level}] {msg}")
        };
        if bytes.saturating_add(formatted.len()) > self.config.max_log_bytes {
            *truncated = true;
            return;
        }
        *bytes += formatted.len() + 1;
        lines.push(formatted);
    }

    fn push_text(&self, msg: String) {
        let mut g = self
            .text_items
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (items, bytes, truncated) = &mut *g;
        if *truncated {
            return;
        }
        // Allow up to 64x preview cap in memory so CAS spill receives full lossless data.
        let hard_cap = self
            .config
            .max_output_bytes
            .saturating_mul(64)
            .max(1_048_576);
        if bytes.saturating_add(msg.len()) > hard_cap {
            *truncated = true;
            return;
        }
        *bytes += msg.len() + 1;
        items.push(msg);
    }
}

fn format_caught_error(err: CaughtError<'_>) -> String {
    match err {
        CaughtError::Exception(ex) => {
            let msg = ex.message().unwrap_or_default();
            let stack = ex.stack().unwrap_or_default();
            if stack.trim().is_empty() {
                if msg.is_empty() {
                    "JavaScript exception".to_string()
                } else {
                    msg
                }
            } else if msg.is_empty() {
                stack
            } else {
                format!("{msg}\n{stack}")
            }
        }
        CaughtError::Value(val) => format!("Thrown value: {val:?}"),
        CaughtError::Error(e) => e.to_string(),
    }
}

fn run_on_worker_thread(
    code: &str,
    shared: &Arc<SharedState>,
    deadline: Instant,
) -> std::result::Result<(), String> {
    let rt_tokio = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .map_err(|e| format!("failed to build CodeMode local runtime: {e}"))?;

    rt_tokio.block_on(async move {
        let js_rt = AsyncRuntime::new()
            .map_err(|e| format!("failed to create QuickJS runtime: {e}"))?;
        js_rt
            .set_memory_limit(shared.config.memory_limit_bytes)
            .await;
        js_rt
            .set_max_stack_size(shared.config.max_stack_size_bytes)
            .await;

        let cancel_flag = Arc::clone(&shared.cancelled_flag);
        let timed_out_flag = Arc::clone(&shared.timed_out);
        let exited_flag = Arc::clone(&shared.exited);
        js_rt
            .set_interrupt_handler(Some(Box::new(move || {
                if exited_flag.load(Ordering::Relaxed) || cancel_flag.load(Ordering::Relaxed) {
                    return true;
                }
                if Instant::now() >= deadline {
                    timed_out_flag.store(true, Ordering::Relaxed);
                    return true;
                }
                false
            })))
            .await;

        let js_ctx = AsyncContext::full(&js_rt)
            .await
            .map_err(|e| format!("failed to create QuickJS context: {e}"))?;

        let eval_res: std::result::Result<(), String> = tokio::select! {
            biased;
            () = shared.child_cancel.cancelled() => {
                Err(if shared.timed_out.load(Ordering::Relaxed) {
                    format!("CodeMode timed out after {}ms", shared.timeout.as_millis())
                } else {
                    "CodeMode cancelled".to_string()
                })
            }
            res = js_ctx.async_with(async |ctx| {
            let globals = ctx.globals();

            // __exit(): marks clean early termination so even `try { exit(); } catch {}` cannot swallow it.
            let s_exit = Arc::clone(shared);
            globals
                .set(
                    "__exit",
                    Function::new(ctx.clone(), move || {
                        s_exit.exited.store(true, Ordering::SeqCst);
                    })
                    .map_err(|e| format!("bind __exit: {e}"))?,
                )
                .map_err(|e| format!("set __exit: {e}"))?;

            // __log(level, msg)
            let s_log = Arc::clone(shared);
            globals
                .set(
                    "__log",
                    Function::new(ctx.clone(), move |level: String, msg: String| {
                        s_log.push_log(&level, msg);
                    })
                    .map_err(|e| format!("bind __log: {e}"))?,
                )
                .map_err(|e| format!("set __log: {e}"))?;

            // __text(msg)
            let s_text = Arc::clone(shared);
            globals
                .set(
                    "__text",
                    Function::new(ctx.clone(), move |msg: String| {
                        s_text.push_text(msg);
                    })
                    .map_err(|e| format!("bind __text: {e}"))?,
                )
                .map_err(|e| format!("set __text: {e}"))?;

            // __setReturn(msg)
            let s_ret = Arc::clone(shared);
            globals
                .set(
                    "__setReturn",
                    Function::new(ctx.clone(), move |msg: String| {
                        *s_ret
                            .return_value
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(msg);
                    })
                    .map_err(|e| format!("bind __setReturn: {e}"))?,
                )
                .map_err(|e| format!("set __setReturn: {e}"))?;

            // __hasTool(name)
            let s_has = Arc::clone(shared);
            globals
                .set(
                    "__hasTool",
                    Function::new(ctx.clone(), move |name: String| -> bool {
                        s_has.resolve_tool_name(&name).is_some()
                    })
                    .map_err(|e| format!("bind __hasTool: {e}"))?,
                )
                .map_err(|e| format!("set __hasTool: {e}"))?;

            // __listTools()
            let list_json = serde_json::to_string(
                &shared
                    .specs
                    .iter()
                    .map(|s| {
                        json!({
                            "name": s.name,
                            "js_name": sanitize_js_ident(&s.name),
                            "description": s.description,
                            "deferred": s.defer_loading,
                        })
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap_or_else(|_| "[]".to_string());
            globals
                .set(
                    "__listTools",
                    Function::new(ctx.clone(), move || list_json.clone())
                        .map_err(|e| format!("bind __listTools: {e}"))?,
                )
                .map_err(|e| format!("set __listTools: {e}"))?;

            // __searchTools(query, limit)
            let s_search = Arc::clone(shared);
            globals
                .set(
                    "__searchTools",
                    Function::new(ctx.clone(), move |query: String, limit: usize| -> String {
                        let ranked = rank_tools(&s_search.specs, &query, limit);
                        serde_json::to_string(&ranked).unwrap_or_else(|_| "[]".to_string())
                    })
                    .map_err(|e| format!("bind __searchTools: {e}"))?,
                )
                .map_err(|e| format!("set __searchTools: {e}"))?;

            // __describeTool(name)
            let s_desc = Arc::clone(shared);
            globals
                .set(
                    "__describeTool",
                    Function::new(ctx.clone(), move |name: String| -> String {
                        let Some(canon) = s_desc.resolve_tool_name(&name) else {
                            return String::new();
                        };
                        let Some(spec) = s_desc.specs.iter().find(|s| s.name == canon) else {
                            return String::new();
                        };
                        json!({
                            "name": spec.name,
                            "js_name": sanitize_js_ident(&spec.name),
                            "description": spec.description,
                            "input_schema": spec.input_schema,
                            "deferred": spec.defer_loading,
                        })
                        .to_string()
                    })
                    .map_err(|e| format!("bind __describeTool: {e}"))?,
                )
                .map_err(|e| format!("set __describeTool: {e}"))?;

            // __storeSet(key, val_json, is_delete) -> "" on ok, error string on cap violation
            let s_sset = Arc::clone(shared);
            globals
                .set(
                    "__storeSet",
                    Function::new(
                        ctx.clone(),
                        move |key: String, val_json: String, is_delete: bool| -> String {
                            let mut st = s_sset
                                .store
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            if is_delete {
                                st.remove(&key);
                                return String::new();
                            }
                            if val_json.len() > MAX_STORE_ENTRY_BYTES {
                                return format!(
                                    "store('{key}') value exceeds {MAX_STORE_ENTRY_BYTES} bytes limit"
                                );
                            }
                            let Ok(parsed) = serde_json::from_str::<Value>(&val_json) else {
                                return format!("store('{key}') value is not valid JSON");
                            };
                            let current_total: usize = st
                                .iter()
                                .filter(|(k, _)| k.as_str() != key.as_str())
                                .map(|(k, v)| k.len() + v.to_string().len())
                                .sum();
                            if current_total + key.len() + val_json.len()
                                > s_sset.config.max_store_bytes
                            {
                                return format!(
                                    "store exceeds total session cap ({} bytes)",
                                    s_sset.config.max_store_bytes
                                );
                            }
                            st.insert(key, parsed);
                            String::new()
                        },
                    )
                    .map_err(|e| format!("bind __storeSet: {e}"))?,
                )
                .map_err(|e| format!("set __storeSet: {e}"))?;

            // __storeGet(key) -> json string or ""
            let s_sget = Arc::clone(shared);
            globals
                .set(
                    "__storeGet",
                    Function::new(ctx.clone(), move |key: String| -> String {
                        s_sget
                            .store
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .get(&key)
                            .map(Value::to_string)
                            .unwrap_or_default()
                    })
                    .map_err(|e| format!("bind __storeGet: {e}"))?,
                )
                .map_err(|e| format!("set __storeGet: {e}"))?;

            // __storeKeys() -> json array
            let s_skeys = Arc::clone(shared);
            globals
                .set(
                    "__storeKeys",
                    Function::new(ctx.clone(), move || -> String {
                        let keys: Vec<String> = s_skeys
                            .store
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .keys()
                            .cloned()
                            .collect();
                        serde_json::to_string(&keys).unwrap_or_else(|_| "[]".to_string())
                    })
                    .map_err(|e| format!("bind __storeKeys: {e}"))?,
                )
                .map_err(|e| format!("set __storeKeys: {e}"))?;

            // __base64Encode / __base64Decode / __blake3 (native safe stdlib helpers)
            globals
                .set(
                    "__base64Encode",
                    Function::new(ctx.clone(), |s: String| -> String {
                        use base64::Engine as _;
                        base64::engine::general_purpose::STANDARD.encode(s.as_bytes())
                    })
                    .map_err(|e| format!("bind __base64Encode: {e}"))?,
                )
                .map_err(|e| format!("set __base64Encode: {e}"))?;

            globals
                .set(
                    "__base64Decode",
                    Function::new(ctx.clone(), |s: String| -> String {
                        use base64::Engine as _;
                        match base64::engine::general_purpose::STANDARD.decode(s.trim()) {
                            Ok(bytes) => match String::from_utf8(bytes) {
                                Ok(text) => json!({ "ok": true, "value": text }).to_string(),
                                Err(e) => {
                                    json!({ "ok": false, "error": format!("invalid UTF-8: {e}") })
                                        .to_string()
                                }
                            },
                            Err(e) => {
                                json!({ "ok": false, "error": format!("invalid base64: {e}") })
                                    .to_string()
                            }
                        }
                    })
                    .map_err(|e| format!("bind __base64Decode: {e}"))?,
                )
                .map_err(|e| format!("set __base64Decode: {e}"))?;

            globals
                .set(
                    "__blake3",
                    Function::new(ctx.clone(), |s: String| -> String {
                        blake3::hash(s.as_bytes()).to_hex().to_string()
                    })
                    .map_err(|e| format!("bind __blake3: {e}"))?,
                )
                .map_err(|e| format!("set __blake3: {e}"))?;

            globals
                .set(
                    "__randomUUID",
                    Function::new(ctx.clone(), || -> String {
                        uuid::Uuid::new_v4().to_string()
                    })
                    .map_err(|e| format!("bind __randomUUID: {e}"))?,
                )
                .map_err(|e| format!("set __randomUUID: {e}"))?;

            globals
                .set(
                    "__btoaLatin1",
                    Function::new(ctx.clone(), |s: String| -> String {
                        use base64::Engine as _;
                        let mut bytes = Vec::with_capacity(s.len());
                        for ch in s.chars() {
                            let cp = ch as u32;
                            if cp > 0xFF {
                                return json!({
                                    "ok": false,
                                    "error": format!("InvalidCharacterError: btoa() only accepts Latin-1 characters (got U+{cp:04X})")
                                })
                                .to_string();
                            }
                            bytes.push(cp as u8);
                        }
                        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
                        json!({ "ok": true, "value": encoded }).to_string()
                    })
                    .map_err(|e| format!("bind __btoaLatin1: {e}"))?,
                )
                .map_err(|e| format!("set __btoaLatin1: {e}"))?;

            globals
                .set(
                    "__atobLatin1",
                    Function::new(ctx.clone(), |s: String| -> String {
                        use base64::Engine as _;
                        // Per WHATWG forgiving-base64, strip ASCII whitespace before decoding.
                        let compact: String = s
                            .chars()
                            .filter(|c| !matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0C'))
                            .collect();
                        match base64::engine::general_purpose::STANDARD.decode(&compact) {
                            Ok(bytes) => {
                                let latin1: String = bytes.into_iter().map(char::from).collect();
                                json!({ "ok": true, "value": latin1 }).to_string()
                            }
                            Err(e) => json!({
                                "ok": false,
                                "error": format!("InvalidCharacterError: invalid base64: {e}")
                            })
                            .to_string(),
                        }
                    })
                    .map_err(|e| format!("bind __atobLatin1: {e}"))?,
                )
                .map_err(|e| format!("set __atobLatin1: {e}"))?;

            globals
                .set(
                    "__utf8Encode",
                    Function::new(ctx.clone(), |s: String| -> Vec<u8> { s.into_bytes() })
                        .map_err(|e| format!("bind __utf8Encode: {e}"))?,
                )
                .map_err(|e| format!("set __utf8Encode: {e}"))?;

            globals
                .set(
                    "__utf8Decode",
                    Function::new(ctx.clone(), |bytes: Vec<u8>| -> String {
                        String::from_utf8_lossy(&bytes).into_owned()
                    })
                    .map_err(|e| format!("bind __utf8Decode: {e}"))?,
                )
                .map_err(|e| format!("set __utf8Decode: {e}"))?;

            globals
                .set(
                    "__parseUrl",
                    Function::new(ctx.clone(), |input: String, base: String| -> String {
                        let parsed = if base.is_empty() {
                            url::Url::parse(&input)
                        } else {
                            url::Url::parse(&base).and_then(|b| b.join(&input))
                        };
                        match parsed {
                            Ok(u) => {
                                let protocol = format!("{}:", u.scheme());
                                let hostname = u.host_str().unwrap_or("").to_string();
                                let port = u.port().map_or_else(String::new, |p| p.to_string());
                                let host = if port.is_empty() {
                                    hostname.clone()
                                } else {
                                    format!("{hostname}:{port}")
                                };
                                let origin = u.origin().ascii_serialization();
                                let search = u
                                    .query()
                                    .map_or_else(String::new, |q| format!("?{q}"));
                                let hash = u
                                    .fragment()
                                    .map_or_else(String::new, |f| format!("#{f}"));
                                let params: Vec<(String, String)> = u
                                    .query_pairs()
                                    .map(|(k, v)| (k.into_owned(), v.into_owned()))
                                    .collect();
                                json!({
                                    "ok": true,
                                    "href": u.as_str(),
                                    "origin": origin,
                                    "protocol": protocol,
                                    "username": u.username(),
                                    "password": u.password().unwrap_or(""),
                                    "host": host,
                                    "hostname": hostname,
                                    "port": port,
                                    "pathname": u.path(),
                                    "search": search,
                                    "hash": hash,
                                    "params": params,
                                })
                                .to_string()
                            }
                            Err(e) => {
                                json!({ "ok": false, "error": format!("Invalid URL: {e}") })
                                    .to_string()
                            }
                        }
                    })
                    .map_err(|e| format!("bind __parseUrl: {e}"))?,
                )
                .map_err(|e| format!("set __parseUrl: {e}"))?;

            // async __callTool(name, args_json) -> json `{ ok, value | error }`
            let s_call = Arc::clone(shared);
            globals
                .set(
                    "__callTool",
                    Function::new(
                        ctx.clone(),
                        Async(move |name: String, args_json: String| {
                            let s = Arc::clone(&s_call);
                            async move {
                                if s.cancelled_flag.load(Ordering::Relaxed)
                                    || s.child_cancel.is_cancelled()
                                {
                                    return json!({
                                        "ok": false,
                                        "error": "CodeMode execution cancelled or timed out"
                                    })
                                    .to_string();
                                }
                                if is_forbidden_inner_tool(&name) {
                                    return json!({
                                        "ok": false,
                                        "error": format!("cannot invoke `{name}` recursively inside CodeMode")
                                    })
                                    .to_string();
                                }
                                let Some(canon) = s.resolve_tool_name(&name) else {
                                    let suggestions = s.suggest_tool_names(&name, 4);
                                    let hint = if suggestions.is_empty() {
                                        "call listTools() or searchTools(query) to inspect available tools".to_string()
                                    } else {
                                        format!(
                                            "did you mean: {}? (or call listTools() / searchTools(query))",
                                            suggestions.join(", ")
                                        )
                                    };
                                    return json!({
                                        "ok": false,
                                        "error": format!("unknown tool `{name}`; {hint}"),
                                        "suggestions": suggestions,
                                    })
                                    .to_string();
                                };
                                let seq = s.tool_calls.fetch_add(1, Ordering::SeqCst) + 1;
                                if seq > s.config.max_tool_calls {
                                    s.tool_calls.fetch_sub(1, Ordering::SeqCst);
                                    return json!({
                                        "ok": false,
                                        "error": format!(
                                            "CodeMode exceeded max_tool_calls limit ({})",
                                            s.config.max_tool_calls
                                        )
                                    })
                                    .to_string();
                                }
                                let args: Value = match serde_json::from_str(&args_json) {
                                    Ok(v) => v,
                                    Err(e) => {
                                        return json!({
                                            "ok": false,
                                            "error": format!("invalid JSON arguments for `{canon}`: {e}")
                                        })
                                        .to_string();
                                    }
                                };
                                {
                                    let mut names = s
                                        .tool_names_called
                                        .lock()
                                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                                    names.push(canon.clone());
                                }
                                let inner_call_id =
                                    format!("{}:cm:{seq}", s.parent_call_id);
                                let host = Arc::clone(&s.host);
                                let cancel = s.child_cancel.clone();
                                let sem = Arc::clone(&s.concurrency);
                                let target_name = canon.clone();
                                // Dispatch on the outer multi-threaded Tokio runtime with bounded
                                // concurrency so `Promise.all([...])` runs tools in parallel
                                // without flooding the host or permission broker.
                                let jh = s.outer.spawn(async move {
                                    let _permit = tokio::select! {
                                        biased;
                                        () = cancel.cancelled() => {
                                            return Err(carnot_core::CarnotError::Cancelled);
                                        }
                                        p = sem.acquire_owned() => match p {
                                            Ok(p) => p,
                                            Err(_) => return Err(carnot_core::CarnotError::Cancelled),
                                        }
                                    };
                                    host.call_tool(inner_call_id, target_name, args, cancel)
                                        .await
                                });
                                match jh.await {
                                    Ok(Ok(out)) => encode_tool_output_for_js(&canon, &out),
                                    Ok(Err(e)) => json!({
                                        "ok": false,
                                        "error": e.to_string()
                                    })
                                    .to_string(),
                                    Err(e) => json!({
                                        "ok": false,
                                        "error": format!("inner tool task panicked: {e}")
                                    })
                                    .to_string(),
                                }
                            }
                        }),
                    )
                    .map_err(|e| format!("bind __callTool: {e}"))?,
                )
                .map_err(|e| format!("set __callTool: {e}"))?;

            // Install JS prelude
            ctx.eval::<(), _>(PRELUDE.as_bytes())
                .catch(&ctx)
                .map_err(|e| format!("CodeMode prelude error: {}", format_caught_error(e)))?;

            // Execute user async snippet
            let body = strip_outer_code_fence(code);
            let wrapped = format!(
                "(async () => {{\n{body}\n}})().then(\n\
                 (v) => {{ if (v !== undefined) __setReturn(globalThis.__carnotFmt(v)); }},\n\
                 (e) => {{ if (e && e.message === \"{EXIT_SENTINEL}\") return; throw e; }}\n\
                 )"
            );
            let promise: Promise<'_> = ctx
                .eval(wrapped.as_bytes())
                .catch(&ctx)
                .map_err(format_caught_error)?;

            promise
                .into_future::<()>()
                .await
                .catch(&ctx)
                .map_err(format_caught_error)?;

            Ok(())
            }) => res,
        };

        if !shared.child_cancel.is_cancelled() && !shared.exited.load(Ordering::Relaxed) {
            let _ = tokio::time::timeout(Duration::from_millis(50), js_rt.idle()).await;
        }
        if shared.exited.load(Ordering::Relaxed) {
            Ok(())
        } else {
            eval_res
        }
    })
}

pub(crate) async fn execute_script(
    code: String,
    parent_call_id: String,
    host: Arc<dyn CodeModeHost>,
    initial_store: BTreeMap<String, Value>,
    config: CodeModeConfig,
    timeout: Duration,
    cancel: CancellationToken,
) -> RunOutput {
    let started = Instant::now();
    let deadline = started + timeout;

    let specs: Vec<ToolSpec> = host
        .list_tools()
        .into_iter()
        .filter(|s| !is_forbidden_inner_tool(&s.name))
        .collect();

    let mut name_map = HashMap::with_capacity(specs.len() * 2);
    for spec in &specs {
        name_map.insert(spec.name.clone(), spec.name.clone());
        let sanitized = sanitize_js_ident(&spec.name);
        name_map
            .entry(sanitized)
            .or_insert_with(|| spec.name.clone());
    }

    let child_cancel = cancel.child_token();
    let cancelled_flag = Arc::new(AtomicBool::new(false));
    let timed_out = Arc::new(AtomicBool::new(false));
    let exited = Arc::new(AtomicBool::new(false));
    let max_conc = config.max_concurrent_tool_calls.max(1);

    let shared = Arc::new(SharedState {
        host,
        outer: tokio::runtime::Handle::current(),
        parent_call_id,
        child_cancel: child_cancel.clone(),
        cancelled_flag: Arc::clone(&cancelled_flag),
        timed_out: Arc::clone(&timed_out),
        exited: Arc::clone(&exited),
        timeout,
        concurrency: Arc::new(tokio::sync::Semaphore::new(max_conc)),
        specs,
        name_map,
        store: Mutex::new(initial_store),
        text_items: Mutex::new((Vec::new(), 0, false)),
        logs: Mutex::new((Vec::new(), 0, false)),
        return_value: Mutex::new(None),
        tool_calls: AtomicUsize::new(0),
        tool_names_called: Mutex::new(Vec::new()),
        config,
    });

    // Watchdog cancels QuickJS interrupt handler + worker select! + in-flight inner tool calls.
    let wd_done = CancellationToken::new();
    let wd_done_clone = wd_done.clone();
    let wd_cancel = cancel.clone();
    let wd_child_cancel = child_cancel.clone();
    let wd_flag = Arc::clone(&cancelled_flag);
    let wd_timed_out = Arc::clone(&timed_out);
    let watchdog = tokio::spawn(async move {
        tokio::select! {
            () = wd_done_clone.cancelled() => {}
            () = wd_cancel.cancelled() => {
                wd_flag.store(true, Ordering::SeqCst);
                wd_child_cancel.cancel();
            }
            () = tokio::time::sleep(timeout) => {
                wd_timed_out.store(true, Ordering::SeqCst);
                wd_flag.store(true, Ordering::SeqCst);
                wd_child_cancel.cancel();
            }
        }
    });

    let (tx, mut rx) = tokio::sync::oneshot::channel();
    let worker_shared = Arc::clone(&shared);
    let spawn_res = std::thread::Builder::new()
        .name("carnot-codemode".into())
        .spawn(move || {
            let res = run_on_worker_thread(&code, &worker_shared, deadline);
            let _ = tx.send(res);
        });

    let run_result = match spawn_res {
        Ok(_handle) => tokio::select! {
            res = &mut rx => res.unwrap_or_else(|_| {
                Err("CodeMode worker thread terminated unexpectedly".into())
            }),
            () = child_cancel.cancelled() => {
                // Give the worker thread a short grace window to drop QuickJS cleanly after wakeup.
                if let Ok(Ok(res)) = tokio::time::timeout(Duration::from_millis(250), rx).await {
                    res
                } else if timed_out.load(Ordering::Relaxed) {
                    Err(format!("CodeMode timed out after {}ms", timeout.as_millis()))
                } else {
                    Err("CodeMode cancelled".into())
                }
            }
        },
        Err(e) => Err(format!("failed to spawn CodeMode worker thread: {e}")),
    };

    // Stop watchdog and cancel any unawaited in-flight inner tool calls.
    wd_done.cancel();
    child_cancel.cancel();
    let _ = watchdog.await;

    let duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let (text_items, _, text_trunc) = {
        let mut g = shared
            .text_items
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let items = std::mem::take(&mut g.0);
        (items, g.1, g.2)
    };
    let (console_lines, _, logs_trunc) = {
        let mut g = shared
            .logs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let lines = std::mem::take(&mut g.0);
        (lines, g.1, g.2)
    };
    let return_value = shared
        .return_value
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    let tool_calls = shared.tool_calls.load(Ordering::Relaxed);
    let tool_names_called = {
        let mut g = shared
            .tool_names_called
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut *g)
    };

    match run_result {
        Ok(()) => {
            let next_store = shared
                .store
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            RunOutput {
                ok: true,
                return_value,
                text_items,
                console_lines,
                logs_truncated: logs_trunc || text_trunc,
                tool_calls,
                tool_names_called,
                error: None,
                duration_ms,
                next_store: Some(next_store),
            }
        }
        Err(err) => {
            let error_msg = if timed_out.load(Ordering::Relaxed) {
                format!("CodeMode timed out after {}ms", timeout.as_millis())
            } else if cancel.is_cancelled() {
                "CodeMode cancelled".to_string()
            } else {
                err
            };
            RunOutput {
                ok: false,
                return_value,
                text_items,
                console_lines,
                logs_truncated: logs_trunc || text_trunc,
                tool_calls,
                tool_names_called,
                error: Some(error_msg),
                duration_ms,
                next_store: None,
            }
        }
    }
}

/// Bounded two-row Wagner-Fischer Levenshtein distance with length cap and row-minimum early exit
/// so adversarial/huge unknown tool names can never stall synchronous Rust execution.
fn bounded_edit_distance(a: &str, b: &str, max_dist: usize) -> Option<usize> {
    const MAX_LEN: usize = 128;
    let a_chars: Vec<char> = a.chars().take(MAX_LEN + 1).collect();
    let b_chars: Vec<char> = b.chars().take(MAX_LEN + 1).collect();
    if a_chars.len() > MAX_LEN || b_chars.len() > MAX_LEN {
        return None;
    }
    if a_chars.len().abs_diff(b_chars.len()) > max_dist {
        return None;
    }
    if a_chars.is_empty() {
        return Some(b_chars.len());
    }
    if b_chars.is_empty() {
        return Some(a_chars.len());
    }
    let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
    let mut cur = vec![0usize; b_chars.len() + 1];
    for (i, &ca) in a_chars.iter().enumerate() {
        cur[0] = i + 1;
        let mut row_min = cur[0];
        for (j, &cb) in b_chars.iter().enumerate() {
            let cost = usize::from(ca != cb);
            let v = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
            cur[j + 1] = v;
            row_min = row_min.min(v);
        }
        if row_min > max_dist {
            return None;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let d = prev[b_chars.len()];
    (d <= max_dist).then_some(d)
}
