//! MCP `2026-07-28` + Legacy `2025-11-25` Streamable HTTP / SSE transport.

use super::stdio::merge_progress_token;
use super::{
    MAX_WIRE_MESSAGE_BYTES, ProgressCallback, TransportError, build_mcp_http_client,
    encode_mcp_header_value, find_sse_delim, parse_rpc_error,
};
use crate::cache::{MCP_LEGACY_PROTOCOL_VERSION, ProtocolEra};
use crate::config::HttpServerConfig;
use carnot_core::{CarnotError, Result};
use futures::StreamExt;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// 2. MCP 2026-07-28 + Legacy 2025-11-25 Streamable HTTP / SSE Transport (HTTP/2 enabled!)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(super) struct StreamableHttpTransport {
    server_name: String,
    cache_key: String,
    endpoint: String,
    headers: BTreeMap<String, String>,
    client: reqwest::Client,
    session_id: Mutex<Option<String>>,
    next_id: AtomicU64,
    list_changed: Arc<AtomicBool>,
}

impl StreamableHttpTransport {
    pub(super) fn new(
        server_name: String,
        cache_key: String,
        cfg: &HttpServerConfig,
        enforce_ssrf: bool,
        list_changed: Arc<AtomicBool>,
    ) -> Result<Self> {
        let timeout_secs = (cfg.timeout_ms / 1000).clamp(5, 600);
        let client = build_mcp_http_client(&server_name, enforce_ssrf, Some(timeout_secs))?;
        Ok(Self {
            server_name,
            cache_key,
            endpoint: cfg.url.clone(),
            headers: cfg.headers.clone(),
            client,
            session_id: Mutex::new(None),
            next_id: AtomicU64::new(1),
            list_changed,
        })
    }

    pub(super) fn clear_session(&self) {
        if let Ok(mut g) = self.session_id.lock() {
            *g = None;
        }
    }

    /// Builds an HTTP POST with protocol headers set **after** user headers (
    /// protocol headers always win so `-32020 HeaderMismatch` can never be triggered by config!).
    fn build_post(
        &self,
        method: &str,
        params: &Value,
        extra_param_headers: &BTreeMap<String, String>,
        era: ProtocolEra,
        protocol_version: &str,
    ) -> reqwest::RequestBuilder {
        let mut req = self.client.post(&self.endpoint);
        // 1. User headers first (skipping any `Mcp-*` defensively)
        for (k, v) in &self.headers {
            if !k.to_ascii_lowercase().starts_with("mcp-") {
                req = req.header(k, v);
            }
        }
        // 2. Mandatory protocol headers always set LAST!
        req = req
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
            .header("MCP-Protocol-Version", protocol_version);

        match era {
            ProtocolEra::Modern20260728 => {
                // SEP-2243 mandatory request metadata headers!
                req = req.header("Mcp-Method", method);
                if let Some(mcp_name) = extract_mcp_name(method, params) {
                    req = req.header("Mcp-Name", encode_mcp_header_value(mcp_name));
                }
                for (hdr_name, hdr_val) in extra_param_headers {
                    req = req.header(hdr_name, hdr_val);
                }
            }
            ProtocolEra::Legacy20251125 => {
                if method != "initialize"
                    && let Ok(guard) = self.session_id.lock()
                    && let Some(ref sid) = *guard
                {
                    req = req.header("Mcp-Session-Id", sid);
                }
            }
        }
        req
    }

    pub(super) async fn notify(
        &self,
        method: &str,
        params: Value,
        era: ProtocolEra,
        protocol_version: &str,
    ) -> Result<()> {
        let body = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        let resp = self
            .build_post(method, &params, &BTreeMap::new(), era, protocol_version)
            .json(&body)
            .send()
            .await
            .map_err(|e| CarnotError::Tool {
                tool: format!("mcp:{}", self.server_name),
                message: format!("HTTP notify `{method}` failed: {e}"),
            })?;
        if resp.status().is_success() || resp.status().as_u16() == 202 {
            Ok(())
        } else {
            Err(CarnotError::Tool {
                tool: format!("mcp:{}", self.server_name),
                message: format!("HTTP notify `{method}` status {}", resp.status()),
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn request_once(
        &self,
        method: &str,
        mut params: Value,
        extra_headers: BTreeMap<String, String>,
        era: ProtocolEra,
        protocol_version: &str,
        timeout: Duration,
        cancel: CancellationToken,
        on_progress: Option<ProgressCallback>,
    ) -> std::result::Result<Value, TransportError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let prog_token = on_progress.as_ref().map(|_| format!("pt-{id}"));
        if let Some(ref tok) = prog_token {
            merge_progress_token(&mut params, tok);
        }

        let body = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let send_fut = self
            .build_post(method, &params, &extra_headers, era, protocol_version)
            .json(&body)
            .send();

        let resp = tokio::select! {
            _ = cancel.cancelled() => return Err(TransportError::Other(CarnotError::Cancelled)),
            r = tokio::time::timeout(timeout, send_fut) => {
                r.map_err(|_| TransportError::Other(CarnotError::Tool {
                    tool: format!("mcp:{}", self.server_name),
                    message: format!("HTTP MCP `{method}` timed out"),
                }))?
                .map_err(|e| TransportError::Other(CarnotError::Tool {
                    tool: format!("mcp:{}", self.server_name),
                    message: format!("HTTP MCP `{method}` network error: {e}"),
                }))?
            }
        };

        // Capture `Mcp-Session-Id` ONLY in legacy era (`2025-11-25`)
        if era == ProtocolEra::Legacy20251125
            && let Some(sid) = resp
                .headers()
                .get("mcp-session-id")
                .and_then(|v| v.to_str().ok())
            && let Ok(mut guard) = self.session_id.lock()
        {
            *guard = Some(sid.to_string());
        }

        let status = resp.status();
        if !status.is_success() {
            let txt = resp.text().await.unwrap_or_default();
            let rpc_err = serde_json::from_str::<Value>(&txt)
                .ok()
                .and_then(|v| v.get("error").map(|e| Box::new(parse_rpc_error(e))));
            return Err(TransportError::HttpStatus {
                status: status.as_u16(),
                body: txt,
                rpc_error: rpc_err,
            });
        }

        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();

        if content_type.contains("text/event-stream") {
            self.read_sse_response(resp, id, on_progress, cancel).await
        } else {
            // Bounded JSON body read (16 MiB cap) with cancellation support.
            let mut stream = resp.bytes_stream();
            let mut body_bytes = Vec::with_capacity(4096);
            loop {
                let next_chunk = tokio::select! {
                    _ = cancel.cancelled() => return Err(TransportError::Other(CarnotError::Cancelled)),
                    c = stream.next() => c,
                };
                let Some(chunk_res) = next_chunk else { break };
                let chunk = chunk_res.map_err(|e| {
                    TransportError::Other(CarnotError::Tool {
                        tool: format!("mcp:{}", self.server_name),
                        message: format!("HTTP MCP `{method}` read error: {e}"),
                    })
                })?;
                if body_bytes.len().saturating_add(chunk.len()) > MAX_WIRE_MESSAGE_BYTES {
                    return Err(TransportError::Other(CarnotError::Tool {
                        tool: format!("mcp:{}", self.server_name),
                        message: format!("HTTP MCP `{method}` response exceeded 16 MiB limit"),
                    }));
                }
                body_bytes.extend_from_slice(&chunk);
            }
            let v: Value = serde_json::from_slice(&body_bytes).map_err(|e| {
                TransportError::Other(CarnotError::Tool {
                    tool: format!("mcp:{}", self.server_name),
                    message: format!("invalid JSON from HTTP MCP `{method}`: {e}"),
                })
            })?;
            if let Some(err) = v.get("error") {
                Err(TransportError::Rpc(Box::new(parse_rpc_error(err))))
            } else {
                Ok(v.get("result").cloned().unwrap_or(Value::Null))
            }
        }
    }

    async fn read_sse_response(
        &self,
        resp: reqwest::Response,
        target_id: u64,
        on_progress: Option<ProgressCallback>,
        cancel: CancellationToken,
    ) -> std::result::Result<Value, TransportError> {
        let mut stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::with_capacity(4096);

        loop {
            let chunk = tokio::select! {
                _ = cancel.cancelled() => return Err(TransportError::Other(CarnotError::Cancelled)),
                c = stream.next() => c,
            };
            let Some(chunk_res) = chunk else {
                break;
            };
            let bytes = chunk_res.map_err(|e| {
                TransportError::Other(CarnotError::Tool {
                    tool: format!("mcp:{}", self.server_name),
                    message: format!("SSE stream read error: {e}"),
                })
            })?;
            if buf.len().saturating_add(bytes.len()) > MAX_WIRE_MESSAGE_BYTES {
                return Err(TransportError::Other(CarnotError::Tool {
                    tool: format!("mcp:{}", self.server_name),
                    message: "HTTP SSE stream buffer exceeded 16 MiB limit".into(),
                }));
            }
            buf.extend_from_slice(&bytes);

            while let Some((end, delim_len)) = find_sse_delim(&buf) {
                let frame = String::from_utf8_lossy(&buf[..end]).to_string();
                buf.drain(..end + delim_len);
                let mut data_lines = Vec::new();
                for line in frame.lines() {
                    if let Some(rest) = line.strip_prefix("data:") {
                        data_lines.push(rest.trim_start());
                    }
                }
                if data_lines.is_empty() {
                    continue;
                }
                let payload = data_lines.join("\n");
                let Ok(v) = serde_json::from_str::<Value>(&payload) else {
                    continue;
                };
                if let Some(m) = v.get("method").and_then(Value::as_str) {
                    if m == "notifications/tools/list_changed" {
                        self.list_changed.store(true, Ordering::Release);
                        crate::cache::invalidate_cached_schema(&self.cache_key);
                    } else if m == "notifications/progress"
                        && let Some(ref cb) = on_progress
                        && let Some(p) = v.get("params")
                    {
                        cb(
                            p.get("progress").and_then(Value::as_f64).unwrap_or(0.0),
                            p.get("total").and_then(Value::as_f64),
                            p.get("message").and_then(Value::as_str).map(str::to_string),
                        );
                    }
                    continue;
                }
                if v.get("id").and_then(Value::as_u64) == Some(target_id) {
                    if let Some(err) = v.get("error") {
                        return Err(TransportError::Rpc(Box::new(parse_rpc_error(err))));
                    }
                    return Ok(v.get("result").cloned().unwrap_or(Value::Null));
                }
            }
        }

        Err(TransportError::Other(CarnotError::Tool {
            tool: format!("mcp:{}", self.server_name),
            message: "HTTP SSE stream ended before receiving JSON-RPC response".into(),
        }))
    }

    pub(super) async fn delete_session(&self) {
        let sid = self.session_id.lock().ok().and_then(|mut g| g.take());
        let Some(sid) = sid else {
            return;
        };
        let mut req = self.client.delete(&self.endpoint);
        for (k, v) in &self.headers {
            if !k.to_ascii_lowercase().starts_with("mcp-") {
                req = req.header(k, v);
            }
        }
        req = req
            .header("MCP-Protocol-Version", MCP_LEGACY_PROTOCOL_VERSION)
            .header("Mcp-Session-Id", sid);
        let _ = tokio::time::timeout(Duration::from_secs(3), req.send()).await;
    }
}

fn extract_mcp_name<'a>(method: &str, params: &'a Value) -> Option<&'a str> {
    match method {
        "tools/call" | "prompts/get" => params.get("name").and_then(Value::as_str),
        "resources/read" => params.get("uri").and_then(Value::as_str),
        _ => None,
    }
}
