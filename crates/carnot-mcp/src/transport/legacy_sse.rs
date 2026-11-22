//! Multiplexed Legacy HTTP+SSE MCP (`2024-11-05`) transport actor.

use super::{
    ProgressCallback, RpcErrorPayload, TransportError, build_mcp_http_client, find_sse_delim,
};
use crate::config::HttpServerConfig;
use carnot_core::{CarnotError, Result};
use futures::StreamExt;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

type PendingSseMap =
    Arc<Mutex<HashMap<u64, oneshot::Sender<std::result::Result<Value, TransportError>>>>>;

/// Multiplexed Legacy HTTP+SSE MCP (`2024-11-05`) transport actor:
/// opens a persistent `GET` SSE stream, captures the server-supplied `endpoint` POST URL,
/// dispatches JSON-RPC POST requests, and routes streamed `message` responses by request `id`.
#[derive(Debug)]
pub(super) struct LegacySseActor {
    server_name: String,
    post_endpoint: String,
    headers: BTreeMap<String, String>,
    client: reqwest::Client,
    next_id: AtomicU64,
    pending: PendingSseMap,
    pub(super) closed: Arc<AtomicBool>,
    cancel: CancellationToken,
}

impl LegacySseActor {
    pub(super) async fn connect(
        server_name: String,
        cfg: &HttpServerConfig,
        enforce_ssrf: bool,
        list_changed: Arc<AtomicBool>,
    ) -> Result<Self> {
        let timeout_secs = (cfg.timeout_ms / 1000).clamp(5, 600);
        let stream_client = build_mcp_http_client(&server_name, enforce_ssrf, None)?;
        let post_client = build_mcp_http_client(&server_name, enforce_ssrf, Some(timeout_secs))?;

        // Retry transient startup connection errors (e.g. sidecar MCP server still binding its port).
        let mut resp = None;
        let mut last_err: Option<CarnotError> = None;
        for attempt in 0..4u32 {
            let mut req = stream_client
                .get(&cfg.url)
                .header("Accept", "text/event-stream");
            for (k, v) in &cfg.headers {
                if !k.to_ascii_lowercase().starts_with("mcp-") {
                    req = req.header(k, v);
                }
            }
            match tokio::time::timeout(Duration::from_secs(15), req.send()).await {
                Ok(Ok(r)) if r.status().is_success() => {
                    resp = Some(r);
                    break;
                }
                Ok(Ok(r)) => {
                    let st = r.status();
                    last_err = Some(CarnotError::Tool {
                        tool: format!("mcp:{server_name}"),
                        message: format!("SSE connect HTTP {st} from {}", cfg.url),
                    });
                    if !st.is_server_error() && st.as_u16() != 429 {
                        break;
                    }
                }
                Ok(Err(e)) => {
                    last_err = Some(CarnotError::Tool {
                        tool: format!("mcp:{server_name}"),
                        message: format!("SSE connect failed to {}: {e}", cfg.url),
                    });
                }
                Err(_) => {
                    last_err = Some(CarnotError::Tool {
                        tool: format!("mcp:{server_name}"),
                        message: format!("SSE connect timeout to {}", cfg.url),
                    });
                }
            }
            if attempt < 3 {
                tokio::time::sleep(Duration::from_millis(250u64 << attempt)).await;
            }
        }
        let resp = match resp {
            Some(r) => r,
            None => {
                return Err(last_err.unwrap_or_else(|| CarnotError::Tool {
                    tool: format!("mcp:{server_name}"),
                    message: format!("SSE connect failed to {}", cfg.url),
                }));
            }
        };

        let base_url = url::Url::parse(&cfg.url).map_err(|e| CarnotError::Tool {
            tool: format!("mcp:{server_name}"),
            message: format!("invalid SSE URL {}: {e}", cfg.url),
        })?;

        let pending: PendingSseMap = Arc::new(Mutex::new(HashMap::new()));
        let closed = Arc::new(AtomicBool::new(false));
        let cancel = CancellationToken::new();
        let (ep_tx, ep_rx) = oneshot::channel::<std::result::Result<String, String>>();

        let pending_bg = Arc::clone(&pending);
        let closed_bg = Arc::clone(&closed);
        let cancel_bg = cancel.clone();
        let base_bg = base_url.clone();
        tokio::spawn(async move {
            let mut ep_tx = Some(ep_tx);
            let mut stream = resp.bytes_stream();
            let mut buf: Vec<u8> = Vec::with_capacity(4096);
            loop {
                let next_chunk = tokio::select! {
                    _ = cancel_bg.cancelled() => None,
                    c = stream.next() => c,
                };
                let Some(Ok(chunk)) = next_chunk else {
                    break;
                };
                buf.extend_from_slice(&chunk);
                while let Some((pos, delim_len)) = find_sse_delim(&buf) {
                    let frame_bytes = buf[..pos].to_vec();
                    buf.drain(..pos + delim_len);
                    let frame = String::from_utf8_lossy(&frame_bytes);
                    let mut event_name = String::new();
                    let mut data_lines: Vec<&str> = Vec::new();
                    for line in frame.lines() {
                        let line = line.trim_end_matches('\r');
                        if let Some(ev) = line.strip_prefix("event:") {
                            event_name = ev.trim().to_string();
                        } else if let Some(d) = line.strip_prefix("data:") {
                            data_lines.push(d.strip_prefix(' ').unwrap_or(d));
                        }
                    }
                    let data = data_lines.join("\n").trim().to_string();
                    if data.is_empty() {
                        continue;
                    }
                    if event_name == "endpoint" {
                        if let Some(tx) = ep_tx.take() {
                            let resolved = base_bg
                                .join(&data)
                                .map(|u| u.to_string())
                                .map_err(|e| format!("invalid SSE endpoint `{data}`: {e}"));
                            let _ = tx.send(resolved);
                        }
                        continue;
                    }
                    if let Ok(msg) = serde_json::from_str::<Value>(&data) {
                        if msg.get("method").and_then(Value::as_str)
                            == Some("notifications/tools/list_changed")
                        {
                            list_changed.store(true, Ordering::Release);
                            continue;
                        }
                        if let Some(id) = msg.get("id").and_then(Value::as_u64)
                            && let Some(tx) = pending_bg.lock().ok().and_then(|mut m| m.remove(&id))
                        {
                            if let Some(err) = msg.get("error") {
                                let code =
                                    err.get("code").and_then(Value::as_i64).unwrap_or(-32000);
                                let message = err
                                    .get("message")
                                    .and_then(Value::as_str)
                                    .unwrap_or("MCP SSE RPC error")
                                    .to_string();
                                let _ =
                                    tx.send(Err(TransportError::Rpc(Box::new(RpcErrorPayload {
                                        code,
                                        message,
                                        data: err.get("data").cloned(),
                                    }))));
                            } else {
                                let _ =
                                    tx.send(Ok(msg.get("result").cloned().unwrap_or(Value::Null)));
                            }
                        }
                    }
                }
            }
            closed_bg.store(true, Ordering::Release);
            if let Some(tx) = ep_tx.take() {
                let _ = tx.send(Err("SSE stream closed before endpoint event".into()));
            }
            if let Ok(mut map) = pending_bg.lock() {
                for (_, tx) in map.drain() {
                    let _ = tx.send(Err(TransportError::Other(CarnotError::Tool {
                        tool: "mcp:sse".into(),
                        message: "SSE stream closed".into(),
                    })));
                }
            }
        });

        let post_endpoint = tokio::time::timeout(Duration::from_secs(15), ep_rx)
            .await
            .map_err(|_| CarnotError::Tool {
                tool: format!("mcp:{server_name}"),
                message: "timed out waiting for SSE `endpoint` event".into(),
            })?
            .map_err(|_| CarnotError::Tool {
                tool: format!("mcp:{server_name}"),
                message: "SSE endpoint channel closed".into(),
            })?
            .map_err(|msg| CarnotError::Tool {
                tool: format!("mcp:{server_name}"),
                message: msg,
            })?;

        Ok(Self {
            server_name,
            post_endpoint,
            headers: cfg.headers.clone(),
            client: post_client,
            next_id: AtomicU64::new(1),
            pending,
            closed,
            cancel,
        })
    }

    pub(super) async fn request(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
        cancel: CancellationToken,
        _on_progress: Option<ProgressCallback>,
    ) -> std::result::Result<Value, TransportError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().expect("pending lock").insert(id, tx);

        let payload = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let mut req = self
            .client
            .post(&self.post_endpoint)
            .header("Content-Type", "application/json");
        for (k, v) in &self.headers {
            if !k.to_ascii_lowercase().starts_with("mcp-") {
                req = req.header(k, v);
            }
        }

        let resp = match req.json(&payload).send().await {
            Ok(r) => r,
            Err(e) => {
                if let Ok(mut m) = self.pending.lock() {
                    m.remove(&id);
                }
                return Err(TransportError::Other(CarnotError::Tool {
                    tool: format!("mcp:{}", self.server_name),
                    message: format!("SSE POST failed: {e}"),
                }));
            }
        };

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            if let Ok(mut m) = self.pending.lock() {
                m.remove(&id);
            }
            return Err(TransportError::HttpStatus {
                status,
                body,
                rpc_error: None,
            });
        }

        // Some servers reply inline with 200 JSON instead of 202 + SSE event; handle both cleanly!
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ct.contains("application/json")
            && let Ok(text) = resp.text().await
            && !text.trim().is_empty()
            && let Ok(val) = serde_json::from_str::<Value>(&text)
            && (val.get("result").is_some() || val.get("error").is_some())
        {
            if let Ok(mut m) = self.pending.lock() {
                m.remove(&id);
            }
            if let Some(err) = val.get("error") {
                return Err(TransportError::Rpc(Box::new(RpcErrorPayload {
                    code: err.get("code").and_then(Value::as_i64).unwrap_or(-32000),
                    message: err
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("MCP SSE error")
                        .to_string(),
                    data: err.get("data").cloned(),
                })));
            }
            return Ok(val.get("result").cloned().unwrap_or(Value::Null));
        }

        let waited = tokio::select! {
            _ = cancel.cancelled() => {
                if let Ok(mut m) = self.pending.lock() {
                    m.remove(&id);
                }
                return Err(TransportError::Other(CarnotError::Cancelled));
            }
            res = tokio::time::timeout(timeout, rx) => res,
        };
        match waited {
            Ok(Ok(inner)) => inner,
            Ok(Err(_)) => Err(TransportError::Other(CarnotError::Tool {
                tool: format!("mcp:{}", self.server_name),
                message: "SSE response channel dropped".into(),
            })),
            Err(_) => {
                if let Ok(mut m) = self.pending.lock() {
                    m.remove(&id);
                }
                Err(TransportError::Other(CarnotError::Tool {
                    tool: format!("mcp:{}", self.server_name),
                    message: format!("SSE request `{method}` timed out"),
                }))
            }
        }
    }

    pub(super) async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let payload = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        let mut req = self
            .client
            .post(&self.post_endpoint)
            .header("Content-Type", "application/json");
        for (k, v) in &self.headers {
            if !k.to_ascii_lowercase().starts_with("mcp-") {
                req = req.header(k, v);
            }
        }
        let _ = req.json(&payload).send().await;
        Ok(())
    }

    pub(super) fn shutdown(&self) {
        self.cancel.cancel();
    }
}
