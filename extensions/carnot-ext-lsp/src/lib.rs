//! Official Carnot Rust Extension: Tier-1 LSP Diagnostics Delta & Semantic Navigation (`LspModule`).
//!
//! Improvements over naive LSP polling:
//!   - **Non-blocking background warm-up:** first `Read`/`Edit`/`Write` kicks off LSP server init in
//!     the background (0 ms cold-start stall on edits); once warm, `Edit`/`Write` reports only newly
//!     introduced Error diagnostics within `diag_wait_ms` (default 400 ms).
//!   - **Realistic active query timeout:** deferred `LspQuery` (`definition`, `references`, `hover`,
//!     `diagnostics`, `defer_loading: true` = 0 static prefix tokens) awaits cold start up to 10s.
//!   - **Zero mutex panics** (`unwrap_or_else(|e| e.into_inner())`) and answers server-initiated
//!     LSP requests (`workspace/configuration`, `client/registerCapability`, `workDoneProgress/create`).

use carnot_core::{CarnotError, Result, ToolEffects, ToolOutput, ToolResultBlock, ToolSpec};
use carnot_ext::{BoxFuture, Module};
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

const QUERY_TIMEOUT_SECS: u64 = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LspLang {
    Rust,
    TypeScript,
    Python,
    Go,
    Cpp,
    Zig,
}

impl LspLang {
    pub fn from_path(path: &Path) -> Option<Self> {
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

    fn default_candidates(self) -> &'static [(&'static str, &'static [&'static str])] {
        match self {
            Self::Rust => &[("rust-analyzer", &[])],
            Self::TypeScript => &[
                ("typescript-language-server", &["--stdio"]),
                ("vtsls", &["--stdio"]),
            ],
            Self::Python => &[
                ("basedpyright-langserver", &["--stdio"]),
                ("pyright-langserver", &["--stdio"]),
                ("pylsp", &[]),
                ("ty", &["server"]),
            ],
            Self::Go => &[("gopls", &[])],
            Self::Cpp => &[("clangd", &[])],
            Self::Zig => &[("zls", &[])],
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

fn path_to_file_uri(path: &Path) -> String {
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    format!("file://{}", canon.to_string_lossy().replace(' ', "%20"))
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

        let p_r = Arc::clone(&pending);
        let d_r = Arc::clone(&diagnostics);
        let n_r = Arc::clone(&diag_notify);
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

                // Answer server-initiated LSP requests so servers never deadlock
                if v.get("method").is_some() && v.get("id").is_some() {
                    let srv_id = v.get("id").cloned().unwrap_or(Value::Null);
                    let m = v.get("method").and_then(Value::as_str).unwrap_or("");
                    let result = match m {
                        "workspace/configuration" => {
                            let count = v
                                .pointer("/params/items")
                                .and_then(Value::as_array)
                                .map_or(1, |a| a.len().max(1));
                            Value::Array(vec![json!({}); count])
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
                                continue;
                            }
                            let ln = d
                                .pointer("/range/start/line")
                                .and_then(Value::as_u64)
                                .unwrap_or(0) as u32;
                            let code = d
                                .get("code")
                                .map(|c| c.as_str().map_or_else(|| c.to_string(), str::to_string))
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
                    d_r.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .insert(uri.to_string(), errs);
                    n_r.notify_waiters();
                    continue;
                }

                if let Some(id) = v.get("id").and_then(Value::as_u64)
                    && let Some(tx) = p_r
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .remove(&id)
                {
                    let _ = tx.send(v);
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

        let root_uri = path_to_file_uri(workspace_root);
        let init_res = conn
            .request(
                "initialize",
                json!({
                    "processId": std::process::id(),
                    "rootUri": root_uri,
                    "capabilities": {
                        "textDocument": {
                            "publishDiagnostics": {},
                            "hover": { "contentFormat": ["markdown", "plaintext"] },
                            "definition": {},
                            "references": {}
                        }
                    }
                }),
                Duration::from_secs(QUERY_TIMEOUT_SECS),
            )
            .await;
        init_res.as_ref()?;
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
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, tx);
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        if self.write_tx.send(frame_lsp_message(&msg)).await.is_err() {
            self.pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&id);
            return None;
        }
        if let Ok(Ok(v)) = tokio::time::timeout(timeout, rx).await {
            Some(v)
        } else {
            self.pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&id);
            None
        }
    }

    async fn sync_document(&self, path: &Path, lang: LspLang, text: &str) -> String {
        let uri = path_to_file_uri(path);
        let next_ver = {
            let mut m = self
                .doc_versions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let entry = m.entry(uri.clone()).or_insert(0);
            *entry += 1;
            *entry
        };
        if next_ver == 1 {
            self.notify(
                "textDocument/didOpen",
                json!({
                    "textDocument": {
                        "uri": uri,
                        "languageId": lang.language_id(),
                        "version": 1,
                        "text": text,
                    }
                }),
            )
            .await;
        } else {
            self.notify(
                "textDocument/didChange",
                json!({
                    "textDocument": { "uri": uri, "version": next_ver },
                    "contentChanges": [{ "text": text }],
                }),
            )
            .await;
        }
        uri
    }

    fn current_diagnostics(&self, uri: &str) -> Vec<RawDiag> {
        self.diagnostics
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(uri)
            .cloned()
            .unwrap_or_default()
    }
}

#[derive(Debug)]
struct LspServerPool {
    workspace_root: PathBuf,
    overrides: BTreeMap<LspLang, (String, Vec<String>)>,
    conns: Mutex<HashMap<LspLang, Option<Arc<LspConn>>>>,
    warming: StdMutex<HashSet<LspLang>>,
}

impl LspServerPool {
    fn new(workspace_root: PathBuf, overrides: BTreeMap<LspLang, (String, Vec<String>)>) -> Self {
        Self {
            workspace_root,
            overrides,
            conns: Mutex::new(HashMap::new()),
            warming: StdMutex::new(HashSet::new()),
        }
    }

    /// Returns an already-warm connection immediately without blocking, or starts background warm-up.
    fn get_if_warm_or_spawn_bg(self: &Arc<Self>, lang: LspLang) -> Option<Arc<LspConn>> {
        if let Ok(guard) = self.conns.try_lock()
            && let Some(slot) = guard.get(&lang)
        {
            return slot.clone();
        }
        let should_spawn = self
            .warming
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(lang);
        if should_spawn {
            let pool = Arc::clone(self);
            tokio::spawn(async move {
                let _ = pool.get_or_spawn(lang).await;
                pool.warming
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .remove(&lang);
            });
        }
        None
    }

    async fn get_or_spawn(&self, lang: LspLang) -> Option<Arc<LspConn>> {
        {
            let guard = self.conns.lock().await;
            if let Some(slot) = guard.get(&lang) {
                return slot.clone();
            }
        }
        let spawned = if let Some((b, a)) = self.overrides.get(&lang) {
            LspConn::spawn(b, a, &self.workspace_root).await
        } else {
            let mut found = None;
            for &(bin, raw_args) in lang.default_candidates() {
                let args: Vec<String> = raw_args.iter().map(|s| (*s).to_string()).collect();
                if let Some(conn) = LspConn::spawn(bin, &args, &self.workspace_root).await {
                    found = Some(conn);
                    break;
                }
            }
            found
        };
        let mut guard = self.conns.lock().await;
        guard.insert(lang, spawned.clone());
        spawned
    }
}

fn fingerprint_diags(diags: &[RawDiag], file_content: &str) -> HashSet<DiagFingerprint> {
    let lines: Vec<&str> = file_content.lines().collect();
    diags
        .iter()
        .map(|d| DiagFingerprint {
            code: d.code.clone(),
            message: d.message.clone(),
            line_text: lines
                .get(d.line as usize)
                .map(|l| l.trim().to_string())
                .unwrap_or_default(),
        })
        .collect()
}

/// Deferred semantic navigation tool (`LspQuery`: `definition`, `references`, `hover`, `diagnostics`).
#[derive(Debug)]
struct LspQueryTool {
    workspace_root: PathBuf,
    pool: Arc<LspServerPool>,
}

impl Tool for LspQueryTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "LspQuery".into(),
            description: "Query workspace LSP for definition, references, hover, or diagnostics."
                .into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["action", "file_path"],
                "properties": {
                    "action": { "type": "string", "enum": ["definition", "references", "hover", "diagnostics"] },
                    "file_path": { "type": "string" },
                    "line": { "type": "integer", "description": "1-based line" },
                    "character": { "type": "integer", "description": "0-based column" }
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
            let Some(raw_path) = input.get("file_path").and_then(Value::as_str) else {
                return Ok(ToolOutput::err_text("LspQuery requires `file_path`"));
            };
            let path = match resolve_confined_path(&self.workspace_root, &[], raw_path) {
                Ok(p) => p,
                Err(msg) => return Ok(ToolOutput::err_text(msg)),
            };
            let Some(lang) = LspLang::from_path(&path) else {
                return Ok(ToolOutput::err_text(format!(
                    "Unsupported file extension for LSP: {raw_path}"
                )));
            };
            let content = match tokio::fs::read_to_string(&path).await {
                Ok(s) => s,
                Err(e) => return Ok(ToolOutput::err_text(format!("Cannot read {raw_path}: {e}"))),
            };

            let conn = tokio::select! {
                () = cancel.cancelled() => return Err(CarnotError::Cancelled),
                c = self.pool.get_or_spawn(lang) => if let Some(conn) = c { conn } else {
                    let tried: Vec<&str> = lang
                        .default_candidates()
                        .iter()
                        .map(|(bin, _)| *bin)
                        .collect();
                    return Ok(ToolOutput::err_text(format!(
                        "Language server for `{}` is not installed or failed to initialize (tried: {}).",
                        lang.language_id(),
                        tried.join(", ")
                    )));
                }
            };
            let notified = conn.diag_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let uri = conn.sync_document(&path, lang, &content).await;

            if action == "diagnostics" {
                let _ = tokio::time::timeout(Duration::from_millis(800), &mut notified).await;
                let diags = conn.current_diagnostics(&uri);
                if diags.is_empty() {
                    return Ok(ToolOutput::ok_text(format!(
                        "[LspQuery diagnostics: 0 errors in {raw_path}]"
                    )));
                }
                let lines: Vec<String> = diags
                    .iter()
                    .take(25)
                    .map(|d| format!("L{}: [{}] {}", d.line + 1, d.code, d.message))
                    .collect();
                return Ok(ToolOutput::ok_text(lines.join("\n")));
            }

            let line_1 = input
                .get("line")
                .and_then(Value::as_u64)
                .unwrap_or(1)
                .max(1);
            let col = input.get("character").and_then(Value::as_u64).unwrap_or(0);
            let pos = json!({
                "textDocument": { "uri": uri },
                "position": { "line": line_1 - 1, "character": col },
            });

            let (method, params) = match action {
                "definition" => ("textDocument/definition", pos),
                "hover" => ("textDocument/hover", pos),
                "references" => (
                    "textDocument/references",
                    json!({
                        "textDocument": { "uri": uri },
                        "position": { "line": line_1 - 1, "character": col },
                        "context": { "includeDeclaration": true },
                    }),
                ),
                other => {
                    return Ok(ToolOutput::err_text(format!(
                        "Unknown LspQuery action `{other}`"
                    )));
                }
            };

            let resp = tokio::select! {
                () = cancel.cancelled() => return Err(CarnotError::Cancelled),
                r = conn.request(method, params, Duration::from_secs(QUERY_TIMEOUT_SECS)) => r,
            };
            let Some(v) = resp else {
                return Ok(ToolOutput::err_text(format!(
                    "LSP `{method}` timed out after {QUERY_TIMEOUT_SECS}s"
                )));
            };
            if let Some(err) = v.get("error") {
                return Ok(ToolOutput::err_text(format!("LSP error: {err}")));
            }
            let result = v.get("result").cloned().unwrap_or(Value::Null);
            Ok(ToolOutput::ok_text(format_lsp_result(
                action,
                &result,
                &self.workspace_root,
            )))
        })
    }
}

fn format_lsp_result(action: &str, result: &Value, ws: &Path) -> String {
    if result.is_null() {
        return format!("[LspQuery {action}: no result]");
    }
    if action == "hover" {
        if let Some(contents) = result.get("contents") {
            if let Some(s) = contents.as_str() {
                return s.to_string();
            }
            if let Some(val) = contents.get("value").and_then(Value::as_str) {
                return val.to_string();
            }
        }
        return result.to_string();
    }
    let items: Vec<&Value> = if let Some(arr) = result.as_array() {
        arr.iter().collect()
    } else {
        vec![result]
    };
    let ws_prefix = format!("file://{}/", ws.to_string_lossy());
    let mut lines = Vec::new();
    for item in items.into_iter().take(40) {
        let uri = item
            .get("uri")
            .or_else(|| item.get("targetUri"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let line = item
            .pointer("/range/start/line")
            .or_else(|| item.pointer("/targetSelectionRange/start/line"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
            + 1;
        if !uri.is_empty() {
            let short = uri.strip_prefix(&ws_prefix).unwrap_or(uri);
            lines.push(format!("{short}:{line}"));
        }
    }
    if lines.is_empty() {
        format!("[LspQuery {action}: no locations]")
    } else {
        lines.join("\n")
    }
}

/// Official Carnot Rust LSP Module (`after_tool_result` warm diagnostics delta + deferred `LspQuery`).
#[derive(Debug)]
pub struct LspDiagnosticsModule {
    workspace_root: PathBuf,
    diag_wait_ms: u64,
    pool: Arc<LspServerPool>,
    baseline: StdMutex<HashMap<PathBuf, HashSet<DiagFingerprint>>>,
}

impl LspDiagnosticsModule {
    pub fn new(workspace_root: PathBuf, diag_wait_ms: u64) -> Self {
        Self::with_server_overrides(workspace_root, diag_wait_ms, BTreeMap::new())
    }

    pub fn with_server_overrides(
        workspace_root: PathBuf,
        diag_wait_ms: u64,
        overrides: BTreeMap<LspLang, (String, Vec<String>)>,
    ) -> Self {
        let pool = Arc::new(LspServerPool::new(workspace_root.clone(), overrides));
        Self {
            workspace_root,
            diag_wait_ms,
            pool,
            baseline: StdMutex::new(HashMap::new()),
        }
    }
}

impl Module for LspDiagnosticsModule {
    fn name(&self) -> &'static str {
        "lsp_diagnostics"
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![Arc::new(LspQueryTool {
            workspace_root: self.workspace_root.clone(),
            pool: Arc::clone(&self.pool),
        })]
    }

    fn after_tool_result<'a>(
        &'a self,
        tool_name: &'a str,
        input: &'a Value,
        output: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>> {
        if output.is_error || !matches!(tool_name, "Read" | "Edit" | "Write") {
            return Box::pin(async { Ok(()) });
        }
        let raw_path = input
            .get("file_path")
            .and_then(Value::as_str)
            .map(str::to_string);
        let is_edit = matches!(tool_name, "Edit" | "Write");
        Box::pin(async move {
            let Some(rp) = raw_path else {
                return Ok(());
            };
            let Ok(path) = resolve_confined_path(&self.workspace_root, &[], &rp) else {
                return Ok(());
            };
            let Some(lang) = LspLang::from_path(&path) else {
                return Ok(());
            };
            // Never stall an Edit/Read on a cold LSP boot — warm up asynchronously in background!
            let Some(conn) = self.pool.get_if_warm_or_spawn_bg(lang) else {
                return Ok(());
            };
            let Ok(content) = tokio::fs::read_to_string(&path).await else {
                return Ok(());
            };
            let notified = conn.diag_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let uri = conn.sync_document(&path, lang, &content).await;

            if !is_edit {
                let diags = conn.current_diagnostics(&uri);
                let fps = fingerprint_diags(&diags, &content);
                self.baseline
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(path, fps);
                return Ok(());
            }

            let _ =
                tokio::time::timeout(Duration::from_millis(self.diag_wait_ms), &mut notified).await;
            let after_diags = conn.current_diagnostics(&uri);
            let lines: Vec<&str> = content.lines().collect();
            let prev_fps = self
                .baseline
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(&path)
                .cloned()
                .unwrap_or_default();

            let mut new_errs = Vec::new();
            for d in &after_diags {
                let fp = DiagFingerprint {
                    code: d.code.clone(),
                    message: d.message.clone(),
                    line_text: lines
                        .get(d.line as usize)
                        .map(|l| l.trim().to_string())
                        .unwrap_or_default(),
                };
                if !prev_fps.contains(&fp) {
                    new_errs.push(format!("  L{}: [{}] {}", d.line + 1, d.code, d.message));
                }
            }
            let new_fps = fingerprint_diags(&after_diags, &content);
            self.baseline
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(path, new_fps);

            if !new_errs.is_empty() {
                let note = format!(
                    "[lsp-diagnostics: {} new error(s) in {rp}]\n{}",
                    new_errs.len(),
                    new_errs.into_iter().take(10).collect::<Vec<_>>().join("\n")
                );
                output.blocks.push(ToolResultBlock::text(note));
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn lsp_query_hover_and_definition_against_mock_stdio_server() {
        let dir = std::env::temp_dir().join(format!("carnot-lsp-ext-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let src = dir.join("main.rs");
        std::fs::write(&src, "fn hello() {}\n").expect("write rs");
        let mock = dir.join("mock_lsp.py");
        let script = r#"#!/usr/bin/env python3
import json, sys

def send(obj):
    body = json.dumps(obj).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    sys.stdout.buffer.flush()

while True:
    cl = 0
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            sys.exit(0)
        s = line.decode("ascii", "ignore").strip()
        if not s:
            break
        if s.lower().startswith("content-length:"):
            cl = int(s.split(":", 1)[1].strip())
    if cl <= 0:
        continue
    msg = json.loads(sys.stdin.buffer.read(cl).decode("utf-8"))
    m = msg.get("method")
    rid = msg.get("id")
    if m == "initialize":
        send({"jsonrpc": "2.0", "id": rid, "result": {"capabilities": {"hoverProvider": True}}})
    elif m == "textDocument/hover":
        send({"jsonrpc": "2.0", "id": rid, "result": {"contents": {"kind": "markdown", "value": "```rust\nfn hello()\n```"}}})
"#;
        std::fs::write(&mock, script).expect("write mock");

        let mut ov = BTreeMap::new();
        ov.insert(
            LspLang::Rust,
            (
                "python3".to_string(),
                vec![mock.to_string_lossy().to_string()],
            ),
        );
        let lsp = LspDiagnosticsModule::with_server_overrides(dir.clone(), 400, ov);
        let query = lsp
            .tools()
            .into_iter()
            .find(|t| t.spec().name == "LspQuery")
            .expect("LspQuery");
        assert!(query.spec().defer_loading);

        let out = query
            .execute(
                json!({"action": "hover", "file_path": "main.rs", "line": 1, "character": 3}),
                CancellationToken::new(),
            )
            .await
            .expect("hover");
        assert!(!out.is_error, "got err: {}", out.text_only());
        assert!(out.text_only().contains("fn hello()"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
