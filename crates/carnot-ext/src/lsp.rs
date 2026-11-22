//! Optional Tier-1 LSP Diagnostics Delta & Deferred Semantic Navigation (`LspDiagnosticsModule` + `LspQuery`).
//!
//! Complements Carnot's <0.5 ms Tier-0 tree-sitter syntax check with full type/borrow/symbol
//! intelligence from `rust-analyzer`, `typescript-language-server`, `pyright-langserver`, `gopls`,
//! `clangd` (C/C++), or `zls` (Zig):
//!   - Lazy-spawns per language on first `Read`/`Edit`/`Write`/`LspQuery` (opt-in for trusted workspaces!)
//!   - Shared `LspServerPool` between passive post-Edit diagnostics AND active deferred `LspQuery`
//!     (`definition`, `references`, `hover`, `diagnostics` — `defer_loading: true`, 0 static prefix cost!)
//!   - Exact byte-counted `Content-Length:` JSON-RPC 2.0 framing
//!   - Answers server-initiated `workspace/configuration`, `client/registerCapability`,
//!     `window/workDoneProgress/create` so servers never deadlock
//!   - Deduplicates diagnostics by `(code, message, line_text)` (immune to shifted line numbers!)
//!   - Strict latency budget (default 400 ms): emits 0 extra tokens on clean edits or timeouts!

use crate::{BoxFuture, Module};
use carnot_core::{CarnotError, Result, ToolEffects, ToolOutput, ToolResultBlock, ToolSpec};
use carnot_sandbox::apply_safe_env;
use carnot_tools::{Tool, files::resolve_confined_path};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{Mutex, Notify, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LspLang {
    Rust,
    TypeScript,
    Python,
    Go,
    Cpp,
    Zig,
}

impl LspLang {
    fn from_path(path: &Path) -> Option<Self> {
        match path.extension().and_then(|e| e.to_str())? {
            "rs" => Some(Self::Rust),
            "ts" | "tsx" | "js" | "jsx" | "mts" | "cts" => Some(Self::TypeScript),
            "py" | "pyi" => Some(Self::Python),
            "go" => Some(Self::Go),
            "c" | "h" | "cc" | "cpp" | "cxx" | "hpp" | "hxx" => Some(Self::Cpp),
            "zig" => Some(Self::Zig),
            _ => None,
        }
    }

    fn language_id(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::TypeScript => "typescript",
            Self::Python => "python",
            Self::Go => "go",
            Self::Cpp => "cpp",
            Self::Zig => "zig",
        }
    }

    fn default_command(self) -> (&'static str, &'static [&'static str]) {
        match self {
            Self::Rust => ("rust-analyzer", &[]),
            Self::TypeScript => ("typescript-language-server", &["--stdio"]),
            Self::Python => ("pyright-langserver", &["--stdio"]),
            Self::Go => ("gopls", &[]),
            Self::Cpp => ("clangd", &[]),
            Self::Zig => ("zls", &[]),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DiagFingerprint {
    code: String,
    message: String,
    line_text: String,
}

#[derive(Debug, Clone)]
struct RawDiag {
    line: u32,
    code: String,
    message: String,
}

struct LspConn {
    write_tx: mpsc::Sender<Vec<u8>>,
    pending: Arc<StdMutex<HashMap<u64, oneshot::Sender<Value>>>>,
    diagnostics: Arc<StdMutex<HashMap<String, Vec<RawDiag>>>>,
    diag_notify: Arc<Notify>,
    doc_versions: Arc<StdMutex<HashMap<String, i32>>>,
    next_id: AtomicU64,
    _child: Arc<Mutex<tokio::process::Child>>,
}

impl std::fmt::Debug for LspConn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LspConn").finish()
    }
}

fn frame_lsp_message(v: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(v).unwrap_or_default();
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    let mut out = Vec::with_capacity(header.len() + body.len());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&body);
    out
}

impl LspConn {
    async fn spawn(bin: &str, args: &[String], workspace_root: &Path) -> Option<Arc<Self>> {
        let mut cmd = Command::new(bin);
        cmd.args(args)
            .current_dir(workspace_root)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        apply_safe_env(&mut cmd, &BTreeMap::new());
        let mut child = cmd.spawn().ok()?;
        let mut stdin = child.stdin.take()?;
        let stdout = child.stdout.take()?;

        let (write_tx, mut write_rx) = mpsc::channel::<Vec<u8>>(128);
        let pending: Arc<StdMutex<HashMap<u64, oneshot::Sender<Value>>>> =
            Arc::new(StdMutex::new(HashMap::new()));
        let diagnostics: Arc<StdMutex<HashMap<String, Vec<RawDiag>>>> =
            Arc::new(StdMutex::new(HashMap::new()));
        let diag_notify = Arc::new(Notify::new());

        tokio::spawn(async move {
            while let Some(bytes) = write_rx.recv().await {
                if stdin.write_all(&bytes).await.is_err() || stdin.flush().await.is_err() {
                    break;
                }
            }
        });

        let p_r = pending.clone();
        let d_r = diagnostics.clone();
        let n_r = diag_notify.clone();
        let w_r = write_tx.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut content_len: Option<usize> = None;
                let mut line = String::new();
                loop {
                    line.clear();
                    let Ok(n) = reader.read_line(&mut line).await else {
                        return;
                    };
                    if n == 0 {
                        return;
                    }
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        break;
                    }
                    if let Some((k, v)) = trimmed.split_once(':')
                        && k.eq_ignore_ascii_case("content-length")
                    {
                        content_len = v.trim().parse().ok();
                    }
                }
                let Some(len) = content_len else {
                    continue;
                };
                if len > 16 * 1024 * 1024 {
                    return;
                }
                let mut body = vec![0u8; len];
                if reader.read_exact(&mut body).await.is_err() {
                    return;
                }
                let Ok(v) = serde_json::from_slice::<Value>(&body) else {
                    continue;
                };

                // Answer server-initiated LSP requests so server never deadlocks!
                if v.get("method").is_some() && v.get("id").is_some() {
                    let srv_id = v.get("id").cloned().unwrap_or(Value::Null);
                    let m = v.get("method").and_then(Value::as_str).unwrap_or("");
                    let result = match m {
                        "workspace/configuration" => {
                            let count = v
                                .pointer("/params/items")
                                .and_then(Value::as_array)
                                .map(|a| a.len().max(1))
                                .unwrap_or(1);
                            Value::Array(vec![json!({}); count])
                        }
                        "client/registerCapability" | "window/workDoneProgress/create" => {
                            Value::Null
                        }
                        _ => Value::Null,
                    };
                    let reply = json!({"jsonrpc": "2.0", "id": srv_id, "result": result});
                    let _ = w_r.send(frame_lsp_message(&reply)).await;
                    continue;
                }

                if v.get("method").and_then(Value::as_str)
                    == Some("textDocument/publishDiagnostics")
                    && let Some(params) = v.get("params")
                    && let Some(uri) = params.get("uri").and_then(Value::as_str)
                {
                    let mut errs = Vec::new();
                    if let Some(arr) = params.get("diagnostics").and_then(Value::as_array) {
                        for d in arr {
                            let sev = d.get("severity").and_then(Value::as_u64).unwrap_or(1);
                            if sev != 1 {
                                continue; // Errors only — zero warning noise!
                            }
                            let ln = d
                                .pointer("/range/start/line")
                                .and_then(Value::as_u64)
                                .unwrap_or(0) as u32;
                            let code = d
                                .get("code")
                                .map(|c| {
                                    c.as_str()
                                        .map(str::to_string)
                                        .unwrap_or_else(|| c.to_string())
                                })
                                .unwrap_or_default();
                            let message = d
                                .get("message")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .lines()
                                .next()
                                .unwrap_or("")
                                .to_string();
                            errs.push(RawDiag {
                                line: ln,
                                code,
                                message,
                            });
                        }
                    }
                    if let Ok(mut map) = d_r.lock() {
                        map.insert(uri.to_string(), errs);
                    }
                    n_r.notify_waiters();
                    continue;
                }

                if let Some(id) = v.get("id").and_then(Value::as_u64)
                    && let Some(tx) = p_r.lock().ok().and_then(|mut m| m.remove(&id))
                {
                    let _ = tx.send(v.get("result").cloned().unwrap_or(Value::Null));
                }
            }
        });

        let conn = Arc::new(Self {
            write_tx,
            pending,
            diagnostics,
            diag_notify,
            doc_versions: Arc::new(StdMutex::new(HashMap::new())),
            next_id: AtomicU64::new(1),
            _child: Arc::new(Mutex::new(child)),
        });

        // Initialize handshake (best-effort 3s budget)
        let root_uri = format!("file://{}", workspace_root.display());
        let init_ok = conn
            .request(
                "initialize",
                json!({
                    "processId": std::process::id(),
                    "rootUri": root_uri,
                    "capabilities": {
                        "textDocument": {
                            "publishDiagnostics": { "relatedInformation": false },
                            "definition": { "linkSupport": true },
                            "references": {},
                            "hover": { "contentFormat": ["markdown", "plaintext"] }
                        }
                    }
                }),
                Duration::from_secs(3),
            )
            .await
            .is_some();
        if !init_ok {
            return None;
        }
        conn.notify("initialized", json!({})).await;
        Some(conn)
    }

    async fn notify(&self, method: &str, params: Value) {
        let msg = json!({"jsonrpc": "2.0", "method": method, "params": params});
        let _ = self.write_tx.send(frame_lsp_message(&msg)).await;
    }

    async fn request(&self, method: &str, params: Value, timeout: Duration) -> Option<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        if let Ok(mut m) = self.pending.lock() {
            m.insert(id, tx);
        }
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        if self.write_tx.send(frame_lsp_message(&msg)).await.is_err() {
            return None;
        }
        let res = tokio::time::timeout(timeout, rx).await.ok()?.ok();
        if let Ok(mut m) = self.pending.lock() {
            m.remove(&id);
        }
        res
    }

    async fn sync_document(&self, path: &Path, lang: LspLang, text: &str) -> String {
        let uri = format!("file://{}", path.display());
        let ver = {
            let mut m = self.doc_versions.lock().expect("lock");
            let v = m.entry(uri.clone()).or_insert(0);
            *v += 1;
            *v
        };
        if ver == 1 {
            self.notify(
                "textDocument/didOpen",
                json!({
                    "textDocument": {
                        "uri": uri,
                        "languageId": lang.language_id(),
                        "version": ver,
                        "text": text,
                    }
                }),
            )
            .await;
        } else {
            self.notify(
                "textDocument/didChange",
                json!({
                    "textDocument": { "uri": uri, "version": ver },
                    "contentChanges": [{ "text": text }],
                }),
            )
            .await;
        }
        uri
    }
}

/// Shared lazy LSP server pool used by both passive `LspDiagnosticsModule` and active `LspQueryTool`.
#[derive(Debug)]
struct LspServerPool {
    workspace_root: PathBuf,
    servers: Mutex<HashMap<LspLang, Option<Arc<LspConn>>>>,
    custom_commands: HashMap<LspLang, (String, Vec<String>)>,
}

impl LspServerPool {
    fn new(
        workspace_root: PathBuf,
        custom_commands: HashMap<LspLang, (String, Vec<String>)>,
    ) -> Self {
        Self {
            workspace_root,
            servers: Mutex::new(HashMap::new()),
            custom_commands,
        }
    }

    async fn get_server(&self, lang: LspLang) -> Option<Arc<LspConn>> {
        let mut map = self.servers.lock().await;
        if let Some(cached) = map.get(&lang) {
            return cached.clone();
        }
        let (bin, args): (String, Vec<String>) = match self.custom_commands.get(&lang) {
            Some((b, a)) => (b.clone(), a.clone()),
            None => {
                let (b, a) = lang.default_command();
                (b.to_string(), a.iter().map(|s| (*s).to_string()).collect())
            }
        };
        let spawned = LspConn::spawn(&bin, &args, &self.workspace_root).await;
        map.insert(lang, spawned.clone());
        spawned
    }
}

fn format_lsp_locations(val: &Value, workspace_root: &Path, max_items: usize) -> Vec<String> {
    let items: Vec<&Value> = match val {
        Value::Array(arr) => arr.iter().collect(),
        Value::Object(_) => vec![val],
        _ => Vec::new(),
    };
    let ws_canon = workspace_root
        .canonicalize()
        .unwrap_or_else(|_| workspace_root.to_path_buf());
    let mut out = Vec::new();
    for item in items.into_iter().take(max_items) {
        let uri = item
            .get("uri")
            .or_else(|| item.get("targetUri"))
            .and_then(Value::as_str);
        let range = item
            .get("range")
            .or_else(|| item.get("targetSelectionRange"))
            .or_else(|| item.get("targetRange"));
        let (Some(uri), Some(range)) = (uri, range) else {
            continue;
        };
        let line0 = range
            .pointer("/start/line")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let col0 = range
            .pointer("/start/character")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let raw_path = uri.strip_prefix("file://").unwrap_or(uri);
        let path = PathBuf::from(raw_path);
        let disp = path
            .strip_prefix(&ws_canon)
            .or_else(|_| path.strip_prefix(workspace_root))
            .unwrap_or(&path)
            .display()
            .to_string();
        let preview = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| s.lines().nth(line0).map(|l| l.trim().to_string()))
            .filter(|l| !l.is_empty());
        match preview {
            Some(line_txt) => out.push(format!("{disp}:{}:{}  {line_txt}", line0 + 1, col0 + 1)),
            None => out.push(format!("{disp}:{}:{}", line0 + 1, col0 + 1)),
        }
    }
    out
}

fn format_hover_contents(val: &Value) -> Option<String> {
    let contents = val.get("contents")?;
    match contents {
        Value::String(s) => (!s.trim().is_empty()).then(|| s.trim().to_string()),
        Value::Object(map) => map
            .get("value")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        Value::Array(arr) => {
            let parts: Vec<String> = arr
                .iter()
                .filter_map(|item| match item {
                    Value::String(s) => Some(s.trim().to_string()),
                    Value::Object(map) => map
                        .get("value")
                        .and_then(Value::as_str)
                        .map(|s| s.trim().to_string()),
                    _ => None,
                })
                .filter(|s| !s.is_empty())
                .collect();
            (!parts.is_empty()).then(|| parts.join("\n---\n"))
        }
        _ => None,
    }
}

/// Deferred active semantic navigation tool (`LspQuery`, `defer_loading: true` => 0 static prefix tokens).
#[derive(Debug)]
struct LspQueryTool {
    pool: Arc<LspServerPool>,
}

impl Tool for LspQueryTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "LspQuery".into(),
            description: "Query active language server for semantic `definition`, `references`, `hover` (type/docs), or `diagnostics` at a 1-based `(file_path, line, column)`.".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["action", "file_path"],
                "properties": {
                    "action": { "type": "string", "enum": ["definition", "references", "hover", "diagnostics"] },
                    "file_path": { "type": "string" },
                    "line": { "type": "integer", "description": "1-based line number (required for definition/references/hover)" },
                    "column": { "type": "integer", "description": "1-based column number (default 1)" }
                }
            }),
            defer_loading: true,
        }
    }

    fn effects(&self, _input: &Value) -> ToolEffects {
        ToolEffects::global_read()
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let Some(action) = input.get("action").and_then(Value::as_str) else {
                return Ok(ToolOutput::err_text("LspQuery requires `action`"));
            };
            let Some(raw_fp) = input.get("file_path").and_then(Value::as_str) else {
                return Ok(ToolOutput::err_text("LspQuery requires `file_path`"));
            };
            let path = match resolve_confined_path(&self.pool.workspace_root, &[], raw_fp) {
                Ok(p) => p,
                Err(msg) => return Ok(ToolOutput::err_text(msg)),
            };
            let Some(lang) = LspLang::from_path(&path) else {
                return Ok(ToolOutput::err_text(format!(
                    "No LSP configured for file extension of `{raw_fp}`"
                )));
            };
            let text = match tokio::fs::read_to_string(&path).await {
                Ok(t) => t,
                Err(e) => return Ok(ToolOutput::err_text(format!("Cannot read `{raw_fp}`: {e}"))),
            };
            let Some(conn) = self.pool.get_server(lang).await else {
                return Ok(ToolOutput::err_text(format!(
                    "LSP server `{}` is not installed or failed to start",
                    lang.default_command().0
                )));
            };

            let uri = conn.sync_document(&path, lang, &text).await;
            if action == "diagnostics" {
                let diags = conn
                    .diagnostics
                    .lock()
                    .ok()
                    .and_then(|m| m.get(&uri).cloned())
                    .unwrap_or_default();
                if diags.is_empty() {
                    return Ok(ToolOutput::ok_text(format!(
                        "No LSP error diagnostics reported for {raw_fp}."
                    )));
                }
                let lines: Vec<String> = diags
                    .iter()
                    .take(25)
                    .map(|d| format!("L{}: [{}] {}", d.line + 1, d.code, d.message))
                    .collect();
                return Ok(ToolOutput::ok_text(lines.join("\n")));
            }

            let Some(line1) = input
                .get("line")
                .and_then(Value::as_u64)
                .filter(|&l| l >= 1)
            else {
                return Ok(ToolOutput::err_text(format!(
                    "LspQuery action `{action}` requires 1-based `line` (>= 1)"
                )));
            };
            let col1 = input
                .get("column")
                .and_then(Value::as_u64)
                .unwrap_or(1)
                .max(1);
            let pos = json!({ "line": line1 - 1, "character": col1 - 1 });

            let (method, params) = match action {
                "definition" => (
                    "textDocument/definition",
                    json!({ "textDocument": { "uri": uri }, "position": pos }),
                ),
                "references" => (
                    "textDocument/references",
                    json!({
                        "textDocument": { "uri": uri },
                        "position": pos,
                        "context": { "includeDeclaration": true }
                    }),
                ),
                "hover" => (
                    "textDocument/hover",
                    json!({ "textDocument": { "uri": uri }, "position": pos }),
                ),
                other => {
                    return Ok(ToolOutput::err_text(format!(
                        "Unknown LspQuery action `{other}`"
                    )));
                }
            };

            let resp = tokio::select! {
                _ = cancel.cancelled() => return Err(CarnotError::Cancelled),
                r = conn.request(method, params, Duration::from_secs(5)) => r,
            };
            let Some(val) = resp else {
                return Ok(ToolOutput::err_text(format!(
                    "LSP `{method}` timed out or returned no response"
                )));
            };

            match action {
                "definition" | "references" => {
                    let locs = format_lsp_locations(&val, &self.pool.workspace_root, 30);
                    if locs.is_empty() {
                        Ok(ToolOutput::ok_text(format!(
                            "No {action} found at {raw_fp}:{line1}:{col1}."
                        )))
                    } else {
                        Ok(ToolOutput::ok_text(locs.join("\n")))
                    }
                }
                "hover" => match format_hover_contents(&val) {
                    Some(h) => Ok(ToolOutput::ok_text(h)),
                    None => Ok(ToolOutput::ok_text(format!(
                        "No hover info at {raw_fp}:{line1}:{col1}."
                    ))),
                },
                _ => unreachable!(),
            }
        })
    }
}

/// Optional Tier-1 LSP Diagnostics & Semantic Query module.
/// Emits ONLY newly introduced errors after `Edit`/`Write` and registers deferred `LspQuery`.
#[derive(Debug)]
pub struct LspDiagnosticsModule {
    pool: Arc<LspServerPool>,
    budget: Duration,
    baseline: StdMutex<HashMap<String, HashSet<DiagFingerprint>>>,
    query_tool: Arc<dyn Tool>,
}

impl LspDiagnosticsModule {
    pub fn new(workspace_root: PathBuf, budget_ms: u64) -> Self {
        Self::with_custom_commands(workspace_root, budget_ms, HashMap::new())
    }

    /// Creates an `LspDiagnosticsModule` with optional custom language-server binary overrides.
    pub fn with_custom_commands(
        workspace_root: PathBuf,
        budget_ms: u64,
        custom_commands: HashMap<LspLang, (String, Vec<String>)>,
    ) -> Self {
        let pool = Arc::new(LspServerPool::new(workspace_root, custom_commands));
        let query_tool = Arc::new(LspQueryTool {
            pool: Arc::clone(&pool),
        });
        Self {
            pool,
            budget: Duration::from_millis(budget_ms.clamp(50, 5_000)),
            baseline: StdMutex::new(HashMap::new()),
            query_tool,
        }
    }
}

fn fingerprint_diags(diags: &[RawDiag], file_text: &str) -> Vec<(RawDiag, DiagFingerprint)> {
    let lines: Vec<&str> = file_text.lines().collect();
    diags
        .iter()
        .map(|d| {
            let line_text = lines
                .get(d.line as usize)
                .copied()
                .unwrap_or("")
                .trim()
                .to_string();
            let fp = DiagFingerprint {
                code: d.code.clone(),
                message: d.message.clone(),
                line_text,
            };
            (d.clone(), fp)
        })
        .collect()
}

impl Module for LspDiagnosticsModule {
    fn name(&self) -> &str {
        "lsp_diagnostics"
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![self.query_tool.clone()]
    }

    fn after_tool_result<'a>(
        &'a self,
        tool_name: &'a str,
        input: &'a Value,
        output: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>> {
        let raw_fp = input
            .get("file_path")
            .and_then(Value::as_str)
            .map(str::to_string);
        let is_read = tool_name == "Read";
        let is_edit = matches!(tool_name, "Edit" | "Write") && !output.is_error;

        Box::pin(async move {
            if !is_read && !is_edit {
                return Ok(());
            }
            let Some(raw_fp) = raw_fp else {
                return Ok(());
            };
            let p = {
                let path = PathBuf::from(&raw_fp);
                let joined = if path.is_absolute() {
                    path
                } else {
                    self.pool.workspace_root.join(path)
                };
                joined.canonicalize().unwrap_or(joined)
            };
            let Some(lang) = LspLang::from_path(&p) else {
                return Ok(());
            };
            let Ok(text) = tokio::fs::read_to_string(&p).await else {
                return Ok(());
            };
            let Some(conn) = self.pool.get_server(lang).await else {
                return Ok(());
            };

            let notified = conn.diag_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let uri = conn.sync_document(&p, lang, &text).await;

            if is_read {
                // Warm up LSP index in background on first Read (Opus tip #4!)
                return Ok(());
            }

            // On Edit/Write: wait up to tight `self.budget` (default 400ms) for fresh diagnostics
            let _ = tokio::time::timeout(self.budget, notified).await;
            let raw_diags = conn
                .diagnostics
                .lock()
                .ok()
                .and_then(|m| m.get(&uri).cloned())
                .unwrap_or_default();
            let current = fingerprint_diags(&raw_diags, &text);
            let cur_set: HashSet<DiagFingerprint> =
                current.iter().map(|(_, fp)| fp.clone()).collect();

            let newly_added: Vec<RawDiag> = {
                let mut base = self.baseline.lock().expect("lock");
                let prev = base.entry(uri.clone()).or_default();
                let diff: Vec<RawDiag> = current
                    .into_iter()
                    .filter(|(_, fp)| !prev.contains(fp))
                    .map(|(d, _)| d)
                    .collect();
                *prev = cur_set;
                diff
            };

            // Silent when clean (0 extra tokens!); emit concise errors only on regressions!
            if !newly_added.is_empty() {
                let summary: Vec<String> = newly_added
                    .iter()
                    .take(5)
                    .map(|d| format!("  L{}: [{}] {}", d.line + 1, d.code, d.message))
                    .collect();
                let note = format!(
                    "\n[lsp-tier1] {} new diagnostic error(s) in {}:\n{}",
                    newly_added.len(),
                    raw_fp,
                    summary.join("\n")
                );
                if let Some(ToolResultBlock::Text { text }) = output.blocks.first_mut() {
                    text.push_str(&note);
                } else {
                    output.blocks.push(ToolResultBlock::text(note));
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn lsp_query_tool_handles_definition_references_hover_and_diagnostics() {
        let dir = std::env::temp_dir().join(format!("carnot-lsp-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let src = dir.join("lib.rs");
        std::fs::write(&src, "pub fn answer() -> i32 {\n    42\n}\n").expect("write rs");
        let srv_py = dir.join("mock_lsp.py");
        std::fs::write(
            &srv_py,
            r#"import sys, json
def send(obj):
    data = json.dumps(obj).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data)
    sys.stdout.buffer.flush()
buf = sys.stdin.buffer
while True:
    cl = None
    while True:
        line = buf.readline()
        if not line:
            sys.exit(0)
        s = line.decode("ascii", "ignore").strip()
        if not s:
            break
        if s.lower().startswith("content-length:"):
            cl = int(s.split(":", 1)[1].strip())
    if cl is None:
        continue
    msg = json.loads(buf.read(cl).decode("utf-8"))
    m = msg.get("method")
    rid = msg.get("id")
    p = msg.get("params") or {}
    if m == "initialize":
        send({"jsonrpc":"2.0","id":rid,"result":{"capabilities":{"definitionProvider":True,"referencesProvider":True,"hoverProvider":True}}})
    elif m == "textDocument/didOpen":
        uri = p["textDocument"]["uri"]
        send({"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{"uri":uri,"diagnostics":[{"severity":1,"code":"E0308","message":"mismatched types","range":{"start":{"line":1,"character":4},"end":{"line":1,"character":6}}}]}})
    elif m == "textDocument/definition":
        uri = p["textDocument"]["uri"]
        send({"jsonrpc":"2.0","id":rid,"result":[{"uri":uri,"range":{"start":{"line":0,"character":7},"end":{"line":0,"character":13}}}]})
    elif m == "textDocument/references":
        uri = p["textDocument"]["uri"]
        send({"jsonrpc":"2.0","id":rid,"result":[{"uri":uri,"range":{"start":{"line":0,"character":7},"end":{"line":0,"character":13}}}]})
    elif m == "textDocument/hover":
        send({"jsonrpc":"2.0","id":rid,"result":{"contents":{"kind":"markdown","value":"```rust\npub fn answer() -> i32\n```"}}})
"#,
        )
        .expect("write py");

        let mut custom = HashMap::new();
        custom.insert(
            LspLang::Rust,
            (
                "python3".to_string(),
                vec![srv_py.to_string_lossy().to_string()],
            ),
        );
        let module = LspDiagnosticsModule::with_custom_commands(dir.clone(), 400, custom);
        let tool = module.tools().into_iter().next().expect("LspQuery");
        assert!(tool.spec().defer_loading);

        let def = tool
            .execute(
                json!({"action": "definition", "file_path": "lib.rs", "line": 1, "column": 8}),
                CancellationToken::new(),
            )
            .await
            .expect("def");
        assert!(!def.is_error, "got: {}", def.text_only());
        assert!(def.text_only().contains("lib.rs:1:8"));
        assert!(def.text_only().contains("pub fn answer()"));

        let hover = tool
            .execute(
                json!({"action": "hover", "file_path": "lib.rs", "line": 1, "column": 8}),
                CancellationToken::new(),
            )
            .await
            .expect("hover");
        assert!(hover.text_only().contains("pub fn answer() -> i32"));

        // Diagnostics pushed during didOpen should be queryable cleanly!
        tokio::time::sleep(Duration::from_millis(50)).await;
        let diags = tool
            .execute(
                json!({"action": "diagnostics", "file_path": "lib.rs"}),
                CancellationToken::new(),
            )
            .await
            .expect("diags");
        assert!(diags.text_only().contains("L2: [E0308] mismatched types"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
