//! Lock-free multiplexed MCP Stdio Actor transport (`StdioActor`).

use super::{MAX_WIRE_MESSAGE_BYTES, ProgressCallback, TransportError, parse_rpc_error};
use crate::config::StdioServerConfig;
use carnot_core::{CarnotError, Result};
use carnot_sandbox::apply_safe_env;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// 1. Lock-Free Multiplexed Stdio Actor
// ---------------------------------------------------------------------------

type PendingMap =
    Arc<Mutex<HashMap<u64, oneshot::Sender<std::result::Result<Value, TransportError>>>>>;
type ProgressMap = Arc<Mutex<HashMap<String, ProgressCallback>>>;

pub(super) struct StdioActor {
    server_name: String,
    pub(super) write_tx: mpsc::Sender<String>,
    pending: PendingMap,
    progress: ProgressMap,
    next_id: AtomicU64,
    pub(super) closed: Arc<AtomicBool>,
    _child_guard: Arc<tokio::sync::Mutex<tokio::process::Child>>,
}

impl std::fmt::Debug for StdioActor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StdioActor")
            .field("server_name", &self.server_name)
            .finish()
    }
}

impl StdioActor {
    pub(super) async fn spawn(
        server_name: String,
        cache_key: String,
        cfg: &StdioServerConfig,
        workspace_root: PathBuf,
        list_changed: Arc<AtomicBool>,
    ) -> Result<Self> {
        let mut cmd = Command::new(&cfg.command);
        cmd.args(&cfg.args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        if let Some(ref cwd) = cfg.cwd {
            cmd.current_dir(cwd);
        } else {
            cmd.current_dir(&workspace_root);
        }
        apply_safe_env(&mut cmd, &cfg.env);

        let mut child = cmd.spawn().map_err(|e| CarnotError::Tool {
            tool: format!("mcp:{server_name}"),
            message: format!("failed to spawn `{}`: {e}", cfg.command),
        })?;

        let mut stdin = child.stdin.take().ok_or_else(|| CarnotError::Tool {
            tool: format!("mcp:{server_name}"),
            message: "missing stdin".into(),
        })?;
        let stdout = child.stdout.take().ok_or_else(|| CarnotError::Tool {
            tool: format!("mcp:{server_name}"),
            message: "missing stdout".into(),
        })?;
        let stderr = child.stderr.take();

        let (write_tx, mut write_rx) = mpsc::channel::<String>(256);
        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let progress: ProgressMap = Arc::new(Mutex::new(HashMap::new()));

        let srv_w = server_name.clone();
        tokio::spawn(async move {
            while let Some(line) = write_rx.recv().await {
                if stdin.write_all(line.as_bytes()).await.is_err() {
                    break;
                }
                if stdin.flush().await.is_err() {
                    break;
                }
            }
            tracing::debug!(server = %srv_w, "MCP stdio writer exited");
        });

        let srv_e = server_name.clone();
        if let Some(err_pipe) = stderr {
            tokio::spawn(async move {
                let mut reader = BufReader::new(err_pipe);
                let mut line = String::new();
                while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                    tracing::debug!(server = %srv_e, stderr = %line.trim(), "MCP server stderr");
                    line.clear();
                }
            });
        }

        let closed = Arc::new(AtomicBool::new(false));
        let closed_r = closed.clone();
        let pending_r = pending.clone();
        let progress_r = progress.clone();
        let write_tx_r = write_tx.clone();
        let srv_r = server_name.clone();
        let ws_uri = format!("file://{}", workspace_root.display());
        tokio::spawn(async move {
            use tokio::io::AsyncReadExt as _;
            let mut reader = BufReader::new(stdout);
            let mut buf = Vec::with_capacity(4096);
            loop {
                buf.clear();
                // Bound line read before allocating >16 MiB to prevent OOM.
                let n = (&mut reader)
                    .take((MAX_WIRE_MESSAGE_BYTES + 1) as u64)
                    .read_until(b'\n', &mut buf)
                    .await
                    .unwrap_or_default();
                if n == 0 || buf.len() > MAX_WIRE_MESSAGE_BYTES {
                    break;
                }
                let Ok(v) = serde_json::from_slice::<Value>(&buf) else {
                    continue;
                };
                dispatch_incoming_message(
                    &srv_r,
                    &cache_key,
                    &ws_uri,
                    v,
                    &pending_r,
                    &progress_r,
                    &list_changed,
                    Some(&write_tx_r),
                )
                .await;
            }
            closed_r.store(true, Ordering::Release);
            if let Ok(mut map) = pending_r.lock() {
                for (_, tx) in map.drain() {
                    let _ = tx.send(Err(TransportError::Other(CarnotError::Tool {
                        tool: format!("mcp:{srv_r}"),
                        message: "MCP server closed stdout".into(),
                    })));
                }
            }
        });

        Ok(Self {
            server_name,
            write_tx,
            pending,
            progress,
            next_id: AtomicU64::new(1),
            closed,
            _child_guard: Arc::new(tokio::sync::Mutex::new(child)),
        })
    }

    pub(super) async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let mut line = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }))?;
        line.push('\n');
        self.write_tx
            .send(line)
            .await
            .map_err(|_| CarnotError::Tool {
                tool: format!("mcp:{}", self.server_name),
                message: "MCP stdio transport closed".into(),
            })
    }

    pub(super) async fn request(
        &self,
        method: &str,
        mut params: Value,
        timeout: Duration,
        cancel: CancellationToken,
        on_progress: Option<ProgressCallback>,
    ) -> std::result::Result<Value, TransportError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let prog_token = on_progress.as_ref().map(|_| format!("pt-{id}"));
        if let (Some(tok), Some(cb)) = (&prog_token, on_progress) {
            merge_progress_token(&mut params, tok);
            if let Ok(mut pm) = self.progress.lock() {
                pm.insert(tok.clone(), cb);
            }
        }

        let (tx, rx) = oneshot::channel();
        if let Ok(mut map) = self.pending.lock() {
            map.insert(id, tx);
        }

        let mut line = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))
        .map_err(|e| TransportError::Other(e.into()))?;
        line.push('\n');
        if self.write_tx.send(line).await.is_err() {
            if let Ok(mut map) = self.pending.lock() {
                map.remove(&id);
            }
            if let Some(tok) = &prog_token
                && let Ok(mut pm) = self.progress.lock()
            {
                pm.remove(tok);
            }
            return Err(TransportError::Other(CarnotError::Tool {
                tool: format!("mcp:{}", self.server_name),
                message: "MCP stdio writer closed".into(),
            }));
        }

        let res = tokio::select! {
            _ = cancel.cancelled() => {
                if !matches!(method, "initialize" | "server/discover") {
                    let _ = self.notify("notifications/cancelled", json!({
                        "requestId": id,
                        "reason": "cancelled by Carnot agent",
                    })).await;
                }
                Err(TransportError::Other(CarnotError::Cancelled))
            }
            _ = tokio::time::sleep(timeout) => {
                if !matches!(method, "initialize" | "server/discover") {
                    let _ = self.notify("notifications/cancelled", json!({
                        "requestId": id,
                        "reason": "request timed out",
                    })).await;
                }
                Err(TransportError::Other(CarnotError::Tool {
                    tool: format!("mcp:{}", self.server_name),
                    message: format!("MCP `{method}` timed out after {}ms", timeout.as_millis()),
                }))
            }
            reply = rx => {
                reply.unwrap_or_else(|_| Err(TransportError::Other(CarnotError::Tool {
                    tool: format!("mcp:{}", self.server_name),
                    message: "MCP response channel dropped".into(),
                })))
            }
        };

        if let Ok(mut map) = self.pending.lock() {
            map.remove(&id);
        }
        if let Some(tok) = &prog_token
            && let Ok(mut pm) = self.progress.lock()
        {
            pm.remove(tok);
        }
        res
    }
}

pub(super) fn merge_progress_token(params: &mut Value, tok: &str) {
    let Some(obj) = params.as_object_mut() else {
        return;
    };
    let meta = obj
        .entry("_meta".to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if let Some(m) = meta.as_object_mut() {
        m.insert("progressToken".into(), Value::String(tok.to_string()));
    }
}

#[allow(clippy::too_many_arguments)]
async fn dispatch_incoming_message(
    server_name: &str,
    cache_key: &str,
    ws_uri: &str,
    v: Value,
    pending: &PendingMap,
    progress: &ProgressMap,
    list_changed: &AtomicBool,
    reply_tx: Option<&mpsc::Sender<String>>,
) {
    // 1. Legacy server-initiated request (`method` + `id`)
    if v.get("method").is_some() && v.get("id").is_some() {
        let srv_id = v.get("id").cloned().unwrap_or(Value::Null);
        let srv_method = v.get("method").and_then(Value::as_str).unwrap_or("");
        let reply = match srv_method {
            "ping" => json!({"jsonrpc": "2.0", "id": srv_id, "result": {}}),
            "roots/list" => json!({
                "jsonrpc": "2.0",
                "id": srv_id,
                "result": { "roots": [{ "uri": ws_uri, "name": "workspace" }] }
            }),
            _ => json!({
                "jsonrpc": "2.0",
                "id": srv_id,
                "error": {"code": -32601, "message": format!("client does not support {srv_method}")}
            }),
        };
        if let Some(tx) = reply_tx
            && let Ok(mut s) = serde_json::to_string(&reply)
        {
            s.push('\n');
            if tx.try_send(s.clone()).is_err() {
                let tx_bg = tx.clone();
                tokio::spawn(async move {
                    let _ = tx_bg.send(s).await;
                });
            }
        }
        return;
    }

    // 2. Server notification (`method` without `id`)
    if let Some(method) = v.get("method").and_then(Value::as_str) {
        match method {
            "notifications/tools/list_changed" => {
                list_changed.store(true, Ordering::Release);
                crate::cache::invalidate_cached_schema(cache_key);
                tracing::info!(server = %server_name, "MCP tools/list_changed received; invalidated schema cache for next session");
            }
            "notifications/progress" => {
                if let Some(p) = v.get("params")
                    && let Some(tok) = p.get("progressToken").and_then(Value::as_str)
                {
                    let prog = p.get("progress").and_then(Value::as_f64).unwrap_or(0.0);
                    let total = p.get("total").and_then(Value::as_f64);
                    let msg = p.get("message").and_then(Value::as_str).map(str::to_string);
                    let cb = progress.lock().ok().and_then(|pm| pm.get(tok).cloned());
                    if let Some(cb) = cb {
                        cb(prog, total, msg);
                    }
                }
            }
            _ => {}
        }
        return;
    }

    // 3. Response to one of our pending requests
    if let Some(id) = v.get("id").and_then(Value::as_u64) {
        let waiter = pending.lock().ok().and_then(|mut map| map.remove(&id));
        if let Some(tx) = waiter {
            if let Some(err) = v.get("error") {
                let _ = tx.send(Err(TransportError::Rpc(Box::new(parse_rpc_error(err)))));
            } else {
                let _ = tx.send(Ok(v.get("result").cloned().unwrap_or(Value::Null)));
            }
        }
    }
}
