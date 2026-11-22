//! Optional Tier-1 LSP Diagnostics Delta Module (`LspDiagnosticsModule`).
//!
//! Complements Carnot's <0.5 ms Tier-0 tree-sitter syntax check with full type/borrow/symbol
//! diagnostics from `rust-analyzer`, `typescript-language-server`, `pyright-langserver`, or `gopls`:
//!   - Lazy-spawns per language on first `Read`/`Edit`/`Write` (opt-in only for trusted workspaces!)
//!   - Exact byte-counted `Content-Length:` JSON-RPC 2.0 framing
//!   - Answers server-initiated `workspace/configuration`, `client/registerCapability`,
//!     `window/workDoneProgress/create` so servers never deadlock
//!   - Deduplicates diagnostics by `(code, message, line_text)` (immune to shifted line numbers!)
//!   - Strict latency budget (default 400 ms): emits 0 extra tokens on clean edits or timeouts!

use crate::{BoxFuture, Module};
use carnot_core::{Result, ToolOutput, ToolResultBlock};
use carnot_sandbox::apply_safe_env;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{Mutex, Notify, mpsc, oneshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LspLang {
    Rust,
    TypeScript,
    Python,
    Go,
}

impl LspLang {
    fn from_path(path: &Path) -> Option<Self> {
        match path.extension().and_then(|e| e.to_str())? {
            "rs" => Some(Self::Rust),
            "ts" | "tsx" | "js" | "jsx" => Some(Self::TypeScript),
            "py" => Some(Self::Python),
            "go" => Some(Self::Go),
            _ => None,
        }
    }

    fn language_id(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::TypeScript => "typescript",
            Self::Python => "python",
            Self::Go => "go",
        }
    }

    fn server_command(self) -> (&'static str, &'static [&'static str]) {
        match self {
            Self::Rust => ("rust-analyzer", &[]),
            Self::TypeScript => ("typescript-language-server", &["--stdio"]),
            Self::Python => ("pyright-langserver", &["--stdio"]),
            Self::Go => ("gopls", &[]),
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
    async fn spawn(lang: LspLang, workspace_root: &Path) -> Option<Arc<Self>> {
        let (bin, args) = lang.server_command();
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
                            "publishDiagnostics": { "relatedInformation": false }
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

/// Optional Tier-1 LSP Diagnostics module. Emits ONLY newly introduced errors after `Edit`/`Write`.
#[derive(Debug)]
pub struct LspDiagnosticsModule {
    workspace_root: PathBuf,
    budget: Duration,
    servers: Mutex<HashMap<LspLang, Option<Arc<LspConn>>>>,
    baseline: StdMutex<HashMap<String, HashSet<DiagFingerprint>>>,
}

impl LspDiagnosticsModule {
    pub fn new(workspace_root: PathBuf, budget_ms: u64) -> Self {
        Self {
            workspace_root,
            budget: Duration::from_millis(budget_ms.clamp(50, 5_000)),
            servers: Mutex::new(HashMap::new()),
            baseline: StdMutex::new(HashMap::new()),
        }
    }

    async fn get_server(&self, lang: LspLang) -> Option<Arc<LspConn>> {
        let mut map = self.servers.lock().await;
        if let Some(cached) = map.get(&lang) {
            return cached.clone();
        }
        let spawned = LspConn::spawn(lang, &self.workspace_root).await;
        map.insert(lang, spawned.clone());
        spawned
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
                    self.workspace_root.join(path)
                };
                joined.canonicalize().unwrap_or(joined)
            };
            let Some(lang) = LspLang::from_path(&p) else {
                return Ok(());
            };
            let Ok(text) = tokio::fs::read_to_string(&p).await else {
                return Ok(());
            };
            let Some(conn) = self.get_server(lang).await else {
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
