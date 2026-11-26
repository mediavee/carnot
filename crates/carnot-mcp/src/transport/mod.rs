//! Ultra-high-performance multiplexed MCP `2026-07-28` (stateless) + `2025-11-25` (legacy) transports:
//!   1. `StdioActor`: lock-free writer mpsc + dedicated reader + stderr drainer + `HashMap<u64, oneshot>`
//!      — supports N concurrent in-flight `tools/call` requests with ZERO head-of-line lock!
//!   2. `StreamableHttpTransport`: HTTP/2-capable MCP `2026-07-28` Streamable HTTP with required
//!      `MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name`, `Mcp-Param-*` (`SEP-2243` `=?base64?...?=`),
//!      optimistic zero-probe modern execution, automatic fallback to legacy `2025-11-25` (`initialize`
//!      + `Mcp-Session-Id` + legacy-only `404` re-init), and `401`/`403` cache purge.
//!
//! Full MCP `2026-07-28` features:
//!   - Stateless per-request `_meta` (`io.modelcontextprotocol/protocolVersion`, `clientCapabilities`,
//!     `clientInfo`, merged cleanly with `progressToken` — `SEP-2575`)
//!   - Dual-era negotiation via `server/discover` (stdio) or direct modern request (HTTP), honoring
//!     modern error codes `-32020..-32099` (`HeaderMismatch`, `MissingRequiredClientCapability`,
//!     `UnsupportedProtocolVersion` with `data.supported` retry!)
//!   - Persisted `ProtocolEra` for **0 negotiation round-trips** on warm starts!
//!   - Multi Round-Trip Requests (MRTR `SEP-2322`: `resultType: "input_required"` with bounded 4-round
//!     `roots/list` fulfillment & 64 KiB `requestState` cap; absent `resultType` treated as `"complete"`)
//!   - Official Tasks Extension (`io.modelcontextprotocol/tasks`, `SEP-2663`): transparent internal
//!     polling via `tasks/get` + `tasks/cancel` on `CancellationToken`, exposing ZERO extra tools!
//!   - `CacheableResult` (`SEP-2549`): captures `ttlMs` + `cacheScope` from `tools/list`.

use crate::cache::{MCP_LEGACY_PROTOCOL_VERSION, MCP_PROTOCOL_VERSION, ProtocolEra};
use crate::config::{HttpServerConfig, StdioServerConfig};
use base64::Engine;
use carnot_core::{CarnotError, Result};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

mod http;
mod legacy_sse;
mod stdio;

use http::StreamableHttpTransport;
use legacy_sse::LegacySseActor;
use stdio::StdioActor;

pub type ProgressCallback = Arc<dyn Fn(f64, Option<f64>, Option<String>) + Send + Sync>;

/// MCP `2026-07-28` specification-reserved JSON-RPC error codes (`-32020..-32099`).
pub const ERR_HEADER_MISMATCH: i64 = -32020;
pub const ERR_MISSING_CLIENT_CAPABILITY: i64 = -32021;
pub const ERR_UNSUPPORTED_PROTOCOL_VERSION: i64 = -32022;

const MAX_MRTR_ROUNDS: usize = 4;
const MAX_REQUEST_STATE_BYTES: usize = 64 * 1024;
const MAX_WIRE_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
const SUPPORTED_MODERN_VERSIONS: &[&str] = &[MCP_PROTOCOL_VERSION];
const SUPPORTED_LEGACY_VERSIONS: &[&str] =
    &[MCP_LEGACY_PROTOCOL_VERSION, "2025-06-18", "2025-03-26"];

/// Structured JSON-RPC error returned by an MCP server (preserves code + `data.supported`).
#[derive(Debug, Clone)]
pub struct RpcErrorPayload {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

impl RpcErrorPayload {
    pub fn is_modern_spec_error(&self) -> bool {
        (-32099..=-32020).contains(&self.code)
    }

    pub fn supported_versions(&self) -> Vec<String> {
        self.data
            .as_ref()
            .and_then(|d| d.get("supported").or_else(|| d.get("supportedVersions")))
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[derive(Debug)]
enum TransportError {
    Rpc(Box<RpcErrorPayload>),
    HttpStatus {
        status: u16,
        body: String,
        rpc_error: Option<Box<RpcErrorPayload>>,
    },
    Other(CarnotError),
}

impl TransportError {
    fn into_carnot(self, server_name: &str) -> CarnotError {
        match self {
            Self::Rpc(e) => CarnotError::Tool {
                tool: format!("mcp:{server_name}"),
                message: format!("MCP JSON-RPC error {}: {}", e.code, e.message),
            },
            Self::HttpStatus { status, body, .. } => CarnotError::Tool {
                tool: format!("mcp:{server_name}"),
                message: format!("HTTP {status}: {body}"),
            },
            Self::Other(e) => e,
        }
    }
}

/// Paginated `tools/list` response carrying `SEP-2549` `CacheableResult` metadata.
#[derive(Debug, Clone)]
pub struct ListToolsResponse {
    pub tools: Vec<Value>,
    pub ttl_ms: Option<u64>,
    pub cache_scope: String,
}

#[derive(Debug, Clone)]
struct ProtocolState {
    era: ProtocolEra,
    version: String,
    legacy_initialized: bool,
    instructions: Option<String>,
}

/// Encodes an MCP `2026-07-28` HTTP header value (`Mcp-Name` or `Mcp-Param-*`) per `SEP-2243`.
/// Plain visible ASCII (0x21..=0x7E, interior space/tab, no leading/trailing whitespace) is kept
/// verbatim; any other value uses the exact `=?base64?{b64}?=` sentinel format.
pub fn encode_mcp_header_value(raw: &str) -> String {
    let is_plain_safe = !raw.is_empty()
        && !raw.starts_with([' ', '\t'])
        && !raw.ends_with([' ', '\t'])
        && !raw.starts_with("=?base64?")
        && raw
            .bytes()
            .all(|b| b == b' ' || b == b'\t' || (0x21..=0x7E).contains(&b));
    if is_plain_safe {
        raw.to_string()
    } else {
        let encoded = base64::engine::general_purpose::STANDARD.encode(raw.as_bytes());
        format!("=?base64?{encoded}?=")
    }
}

/// Formats a primitive tool argument (`string`/`integer`/`boolean`) for `Mcp-Param-*` mirroring.
pub fn format_param_header_value(val: &Value) -> Option<String> {
    let raw = match val {
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else {
                n.as_u64()?.to_string()
            }
        }
        _ => return None,
    };
    Some(encode_mcp_header_value(&raw))
}

/// Picks only from client-implemented modern versions (never an unknown future spec).
fn pick_best_modern_version(supported: &[String]) -> Option<String> {
    SUPPORTED_MODERN_VERSIONS
        .iter()
        .find(|&&v| supported.iter().any(|s| s == v))
        .map(|s| (*s).to_string())
}

fn supports_any_legacy_version(supported: &[String]) -> bool {
    SUPPORTED_LEGACY_VERSIONS
        .iter()
        .any(|&v| supported.iter().any(|s| s == v))
}

fn inject_modern_meta(params: &mut Value, protocol_version: &str) {
    let Some(obj) = params.as_object_mut() else {
        return;
    };
    let meta = obj
        .entry("_meta".to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if !meta.is_object() {
        *meta = Value::Object(serde_json::Map::new());
    }
    if let Some(m) = meta.as_object_mut() {
        // Preserve any existing `progressToken`!
        m.insert(
            "io.modelcontextprotocol/protocolVersion".into(),
            Value::String(protocol_version.to_string()),
        );
        // Declare `roots` and the official `io.modelcontextprotocol/tasks` extension.
        m.insert(
            "io.modelcontextprotocol/clientCapabilities".into(),
            json!({
                "roots": {},
                "extensions": {
                    "io.modelcontextprotocol/tasks": {}
                }
            }),
        );
        m.insert(
            "io.modelcontextprotocol/clientInfo".into(),
            json!({ "name": "carnot", "version": env!("CARGO_PKG_VERSION") }),
        );
    }
}

/// Live multiplexed dual-era MCP client (`2026-07-28` stateless primary + `2025-11-25` legacy fallback).
#[derive(Debug)]
pub struct McpClient {
    server_name: String,
    cache_key: String,
    workspace_uri: String,
    timeout: Duration,
    inner: McpTransportInner,
    proto: Mutex<ProtocolState>,
    legacy_init_lock: tokio::sync::Mutex<()>,
    started_from_hint: bool,
    first_request_done: AtomicBool,
    list_changed: Arc<AtomicBool>,
}

#[derive(Debug)]
enum McpTransportInner {
    Stdio(StdioActor),
    Http(StreamableHttpTransport),
    LegacySse(LegacySseActor),
}

impl McpClient {
    /// Connects to a stdio MCP server with dual-era negotiation (`server/discover` probe on cold
    /// start, 0 probe RTTs when `hint_era` is loaded from disk cache!).
    pub async fn connect_stdio(
        server_name: String,
        cache_key: String,
        cfg: &StdioServerConfig,
        workspace_root: PathBuf,
        hint_era: Option<(ProtocolEra, String)>,
    ) -> Result<Self> {
        let list_changed = Arc::new(AtomicBool::new(false));
        let ws_uri = format!("file://{}", workspace_root.display());
        let actor = StdioActor::spawn(
            server_name.clone(),
            cache_key.clone(),
            cfg,
            &workspace_root,
            Arc::clone(&list_changed),
        )?;
        let timeout = Duration::from_millis(cfg.timeout_ms.clamp(1_000, 600_000));
        let has_hint = hint_era.as_ref().map(|(e, _)| *e);
        let (init_era, init_ver) = hint_era.unwrap_or((
            ProtocolEra::Modern20260728,
            MCP_PROTOCOL_VERSION.to_string(),
        ));
        let client = Self {
            server_name,
            cache_key,
            workspace_uri: ws_uri,
            timeout,
            inner: McpTransportInner::Stdio(actor),
            proto: Mutex::new(ProtocolState {
                era: init_era,
                version: init_ver,
                legacy_initialized: false,
                instructions: None,
            }),
            legacy_init_lock: tokio::sync::Mutex::new(()),
            started_from_hint: has_hint.is_some(),
            first_request_done: AtomicBool::new(false),
            list_changed,
        };

        if let Some(era) = has_hint {
            // Warm hint: 0 probe RTTs for modern servers; direct legacy handshake for legacy servers!
            if era == ProtocolEra::Legacy20251125 {
                client.ensure_legacy_handshake().await?;
            }
        } else {
            // Cold stdio dual-era negotiation via `server/discover` (using full startup timeout!)
            client.probe_stdio_era().await?;
        }

        Ok(client)
    }

    /// Connects to a Streamable HTTP MCP server with zero upfront probe RTTs:
    /// first real request goes directly as `2026-07-28`, falling back to `2025-11-25` only if needed!
    pub async fn connect_http(
        server_name: String,
        cache_key: String,
        cfg: &HttpServerConfig,
        workspace_root: PathBuf,
        enforce_ssrf: bool,
        hint_era: Option<(ProtocolEra, String)>,
    ) -> Result<Self> {
        let list_changed = Arc::new(AtomicBool::new(false));
        let ws_uri = format!("file://{}", workspace_root.display());
        let timeout = Duration::from_millis(cfg.timeout_ms.clamp(1_000, 600_000));

        if cfg.is_legacy_sse() {
            let sse = LegacySseActor::connect(
                server_name.clone(),
                cfg,
                enforce_ssrf,
                Arc::clone(&list_changed),
            )
            .await?;
            let client = Self {
                server_name,
                cache_key,
                workspace_uri: ws_uri,
                timeout,
                inner: McpTransportInner::LegacySse(sse),
                proto: Mutex::new(ProtocolState {
                    era: ProtocolEra::Legacy20251125,
                    version: "2024-11-05".to_string(),
                    legacy_initialized: false,
                    instructions: None,
                }),
                legacy_init_lock: tokio::sync::Mutex::new(()),
                started_from_hint: true,
                first_request_done: AtomicBool::new(false),
                list_changed,
            };
            client.ensure_legacy_handshake().await?;
            return Ok(client);
        }

        let http = StreamableHttpTransport::new(
            server_name.clone(),
            cache_key.clone(),
            cfg,
            enforce_ssrf,
            Arc::clone(&list_changed),
        )?;
        let has_hint = hint_era.as_ref().map(|(e, _)| *e);
        let (init_era, init_ver) = hint_era.unwrap_or((
            ProtocolEra::Modern20260728,
            MCP_PROTOCOL_VERSION.to_string(),
        ));
        let client = Self {
            server_name,
            cache_key,
            workspace_uri: ws_uri,
            timeout,
            inner: McpTransportInner::Http(http),
            proto: Mutex::new(ProtocolState {
                era: init_era,
                version: init_ver,
                legacy_initialized: false,
                instructions: None,
            }),
            legacy_init_lock: tokio::sync::Mutex::new(()),
            started_from_hint: has_hint.is_some(),
            first_request_done: AtomicBool::new(false),
            list_changed,
        };
        if init_era == ProtocolEra::Legacy20251125 {
            client.ensure_legacy_handshake().await?;
        }
        Ok(client)
    }

    pub fn protocol_era(&self) -> ProtocolEra {
        self.proto
            .lock()
            .map_or(ProtocolEra::Modern20260728, |p| p.era)
    }

    pub fn protocol_version(&self) -> String {
        self.proto
            .lock()
            .map_or_else(|_| MCP_PROTOCOL_VERSION.to_string(), |p| p.version.clone())
    }

    /// Returns the first non-empty line of server instructions/description returned during handshake.
    pub fn server_instructions(&self) -> Option<String> {
        self.proto.lock().ok().and_then(|p| p.instructions.clone())
    }

    fn capture_server_instructions(&self, res: &Value) {
        let raw = res.get("instructions").and_then(Value::as_str).or_else(|| {
            res.get("serverInfo")
                .and_then(|si| si.get("description"))
                .and_then(Value::as_str)
        });
        if let Some(first_line) = raw
            .and_then(|s| s.lines().map(str::trim).find(|l| !l.is_empty()))
            .filter(|l| !l.is_empty())
            && let Ok(mut p) = self.proto.lock()
        {
            let mut clipped = first_line.to_string();
            if clipped.len() > 160 {
                let mut end = 160;
                while end > 0 && !clipped.is_char_boundary(end) {
                    end -= 1;
                }
                clipped.truncate(end);
                clipped.push('…');
            }
            p.instructions = Some(clipped);
        }
    }

    fn set_protocol(&self, era: ProtocolEra, version: String, legacy_initialized: bool) {
        if let Ok(mut p) = self.proto.lock() {
            p.era = era;
            p.version = version;
            p.legacy_initialized = legacy_initialized;
        }
    }

    /// Stdio cold dual-era probe (`server/discover`):
    ///   - `Ok(DiscoverResult)` or modern error code (`-32020..-32099`) => `Modern20260728`
    ///   - Any other error => fallback to `Legacy20251125` (`initialize` + `notifications/initialized`)
    ///   - If fallback `initialize` itself returns a modern error code => flip back to `Modern20260728`!
    async fn probe_stdio_era(&self) -> Result<()> {
        let mut params = json!({});
        inject_modern_meta(&mut params, MCP_PROTOCOL_VERSION);
        match self
            .send_wire(
                "server/discover",
                params,
                BTreeMap::new(),
                ProtocolEra::Modern20260728,
                MCP_PROTOCOL_VERSION,
                CancellationToken::new(),
                None,
            )
            .await
        {
            Ok(res) => {
                self.capture_server_instructions(&res);
                let supported: Vec<String> = res
                    .get("supportedVersions")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                let ver = pick_best_modern_version(&supported)
                    .unwrap_or_else(|| MCP_PROTOCOL_VERSION.to_string());
                self.set_protocol(ProtocolEra::Modern20260728, ver, false);
                Ok(())
            }
            Err(TransportError::Rpc(ref e)) if e.is_modern_spec_error() => {
                let ver = pick_best_modern_version(&e.supported_versions())
                    .unwrap_or_else(|| MCP_PROTOCOL_VERSION.to_string());
                self.set_protocol(ProtocolEra::Modern20260728, ver, false);
                Ok(())
            }
            Err(_) => {
                // Any non-modern error indicates a legacy `2025-11-25` server.
                self.ensure_legacy_handshake().await
            }
        }
    }

    async fn ensure_legacy_handshake(&self) -> Result<()> {
        // Serialize concurrent legacy handshakes so parallel calls emit only one initialize.
        let _init_guard = self.legacy_init_lock.lock().await;
        let already = self
            .proto
            .lock()
            .is_ok_and(|p| p.era == ProtocolEra::Legacy20251125 && p.legacy_initialized);
        if already {
            return Ok(());
        }
        // Only advertise `roots` on Stdio where server-initiated bidirectional replies work.
        let caps = match self.inner {
            McpTransportInner::Stdio(_) => json!({ "roots": { "listChanged": false } }),
            McpTransportInner::Http(_) | McpTransportInner::LegacySse(_) => json!({}),
        };
        let init_version = match self.inner {
            McpTransportInner::LegacySse(_) => "2024-11-05",
            McpTransportInner::Stdio(_) | McpTransportInner::Http(_) => MCP_LEGACY_PROTOCOL_VERSION,
        };
        let params = json!({
            "protocolVersion": init_version,
            "capabilities": caps,
            "clientInfo": { "name": "carnot", "version": env!("CARGO_PKG_VERSION") },
        });
        match self
            .send_wire(
                "initialize",
                params,
                BTreeMap::new(),
                ProtocolEra::Legacy20251125,
                init_version,
                CancellationToken::new(),
                None,
            )
            .await
        {
            Ok(res) => {
                self.capture_server_instructions(&res);
                self.set_protocol(ProtocolEra::Legacy20251125, init_version.to_string(), true);
                let _ = self.notify("notifications/initialized", json!({})).await;
                Ok(())
            }
            Err(TransportError::Rpc(ref e)) if e.is_modern_spec_error() => {
                let ver = pick_best_modern_version(&e.supported_versions())
                    .unwrap_or_else(|| MCP_PROTOCOL_VERSION.to_string());
                self.set_protocol(ProtocolEra::Modern20260728, ver, false);
                Ok(())
            }
            Err(TransportError::HttpStatus {
                rpc_error: Some(ref e),
                ..
            }) if e.is_modern_spec_error() => {
                let ver = pick_best_modern_version(&e.supported_versions())
                    .unwrap_or_else(|| MCP_PROTOCOL_VERSION.to_string());
                self.set_protocol(ProtocolEra::Modern20260728, ver, false);
                Ok(())
            }
            Err(e) => Err(e.into_carnot(&self.server_name)),
        }
    }

    pub fn take_list_changed(&self) -> bool {
        self.list_changed.swap(false, Ordering::AcqRel)
    }

    /// Sends a JSON-RPC request with full `2026-07-28` dual-era fallback, `SEP-2243` `Mcp-Param-*`
    /// headers, MRTR (`resultType: "input_required"`), and Tasks (`io.modelcontextprotocol/tasks`).
    pub async fn request(
        &self,
        method: &str,
        params: Value,
        cancel: CancellationToken,
        on_progress: Option<ProgressCallback>,
    ) -> Result<Value> {
        self.request_with_headers(method, params, BTreeMap::new(), cancel, on_progress)
            .await
    }

    pub async fn request_with_headers(
        &self,
        method: &str,
        base_params: Value,
        extra_headers: BTreeMap<String, String>,
        cancel: CancellationToken,
        on_progress: Option<ProgressCallback>,
    ) -> Result<Value> {
        let mut current_params = base_params.clone();
        for _round in 0..=MAX_MRTR_ROUNDS {
            let result = self
                .request_with_negotiation(
                    method,
                    current_params.clone(),
                    extra_headers.clone(),
                    cancel.clone(),
                    on_progress.clone(),
                )
                .await?;

            // MCP 2026-07-28 SEP-2322: check `resultType` (missing => `"complete"` for legacy!)
            let result_type = result
                .get("resultType")
                .and_then(Value::as_str)
                .unwrap_or("complete");

            if result_type == "input_required" {
                let req_state = result.get("requestState").and_then(Value::as_str);
                if let Some(rs) = req_state
                    && rs.len() > MAX_REQUEST_STATE_BYTES
                {
                    return Err(CarnotError::Tool {
                        tool: format!("mcp:{}", self.server_name),
                        message: "MCP MRTR requestState exceeded 64 KiB cap".into(),
                    });
                }
                let input_responses = self.fulfill_mrtr_requests(&result);
                let mut retry_params = base_params.clone();
                if let Some(obj) = retry_params.as_object_mut() {
                    obj.insert("inputResponses".into(), Value::Object(input_responses));
                    if let Some(rs) = req_state {
                        obj.insert("requestState".into(), Value::String(rs.to_string()));
                    }
                }
                current_params = retry_params;
                continue;
            }

            // MCP 2026-07-28 SEP-2663 (`io.modelcontextprotocol/tasks`): if `tools/call` returned
            // an unsolicited async task handle, transparently poll `tasks/get` honoring `pollIntervalMs`/`ttlMs`!
            if method == "tools/call"
                && let Some(task_info) = extract_active_task_info(&result)
            {
                return self.poll_task_until_done(task_info, cancel).await;
            }

            return Ok(result);
        }

        Err(CarnotError::Tool {
            tool: format!("mcp:{}", self.server_name),
            message: format!("MCP `{method}` exceeded MRTR max rounds ({MAX_MRTR_ROUNDS})"),
        })
    }

    fn fulfill_mrtr_requests(&self, input_required: &Value) -> serde_json::Map<String, Value> {
        let mut responses = serde_json::Map::new();
        let Some(reqs) = input_required
            .get("inputRequests")
            .and_then(Value::as_object)
        else {
            return responses;
        };
        for (req_key, req_obj) in reqs {
            let sub_method = req_obj.get("method").and_then(Value::as_str).unwrap_or("");
            let resp = match sub_method {
                "roots/list" => json!({
                    "roots": [{ "uri": self.workspace_uri, "name": "workspace" }]
                }),
                "elicitation/create" => json!({
                    "action": "decline"
                }),
                _ => json!({
                    "error": {
                        "code": ERR_MISSING_CLIENT_CAPABILITY,
                        "message": format!("Carnot client does not support MRTR `{sub_method}`")
                    }
                }),
            };
            responses.insert(req_key.clone(), resp);
        }
        responses
    }

    async fn poll_task_until_done(
        &self,
        initial: TaskHandleInfo,
        cancel: CancellationToken,
    ) -> Result<Value> {
        let task_id = initial.task_id;
        let max_wait = initial
            .ttl_ms
            .map_or(
                self.timeout.max(Duration::from_secs(300)),
                Duration::from_millis,
            )
            .min(Duration::from_secs(600));
        let deadline = tokio::time::Instant::now() + max_wait;
        let mut interval_ms = initial.poll_interval_ms.unwrap_or(50).clamp(50, 5_000);
        let mut input_rounds = 0usize;

        loop {
            if tokio::time::Instant::now() >= deadline {
                let _ = self
                    .request_with_negotiation(
                        "tasks/cancel",
                        json!({ "taskId": task_id }),
                        BTreeMap::new(),
                        CancellationToken::new(),
                        None,
                    )
                    .await;
                return Err(CarnotError::Tool {
                    tool: format!("mcp:{}", self.server_name),
                    message: format!(
                        "MCP task `{task_id}` exceeded TTL/timeout ({}s)",
                        max_wait.as_secs()
                    ),
                });
            }
            tokio::select! {
                () = cancel.cancelled() => {
                    let _ = self.request_with_negotiation(
                        "tasks/cancel",
                        json!({ "taskId": task_id }),
                        BTreeMap::new(),
                        CancellationToken::new(),
                        None,
                    ).await;
                    return Err(CarnotError::Cancelled);
                }
                () = tokio::time::sleep(Duration::from_millis(interval_ms)) => {}
            }
            let res = self
                .request_with_negotiation(
                    "tasks/get",
                    json!({ "taskId": task_id }),
                    BTreeMap::new(),
                    cancel.clone(),
                    None,
                )
                .await?;
            let task_obj = res.get("task").unwrap_or(&res);
            if let Some(next_iv) = task_obj
                .get("pollIntervalMs")
                .or_else(|| task_obj.get("pollInterval"))
                .and_then(Value::as_u64)
            {
                interval_ms = next_iv.clamp(50, 5_000);
            } else {
                interval_ms = (interval_ms * 2).min(1_000);
            }
            let status = task_obj
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("completed");
            match status {
                "working" | "queued" | "pending" => {}
                "input_required" => {
                    // Handle task input_required via `tasks/update`.
                    input_rounds += 1;
                    if input_rounds > MAX_MRTR_ROUNDS {
                        return Err(CarnotError::Tool {
                            tool: format!("mcp:{}", self.server_name),
                            message: format!("MCP task `{task_id}` exceeded input_required rounds"),
                        });
                    }
                    let responses = self.fulfill_mrtr_requests(task_obj);
                    if responses.is_empty() {
                        return Ok(json!({
                            "content": [{ "type": "text", "text": format!("MCP task `{task_id}` requires unsupported user input: {}", task_obj) }],
                            "isError": true
                        }));
                    }
                    let mut upd = json!({
                        "taskId": task_id,
                        "inputResponses": Value::Object(responses),
                    });
                    if let Some(rs) = task_obj.get("requestState").and_then(Value::as_str)
                        && let Some(m) = upd.as_object_mut()
                    {
                        m.insert("requestState".into(), Value::String(rs.to_string()));
                    }
                    let _ = self
                        .request_with_negotiation(
                            "tasks/update",
                            upd,
                            BTreeMap::new(),
                            cancel.clone(),
                            None,
                        )
                        .await?;
                }
                "completed" => {
                    if let Some(inner) = res.get("result") {
                        return Ok(inner.clone());
                    }
                    return Ok(res);
                }
                "failed" => {
                    let msg = res
                        .get("error")
                        .or_else(|| task_obj.get("statusMessage"))
                        .map_or_else(|| "MCP async task failed".into(), Value::to_string);
                    return Ok(json!({
                        "content": [{ "type": "text", "text": msg }],
                        "isError": true
                    }));
                }
                "cancelled" => return Err(CarnotError::Cancelled),
                other => {
                    return Err(CarnotError::Tool {
                        tool: format!("mcp:{}", self.server_name),
                        message: format!("MCP task `{task_id}` returned unknown status `{other}`"),
                    });
                }
            }
        }
    }

    async fn request_with_negotiation(
        &self,
        method: &str,
        params: Value,
        extra_headers: BTreeMap<String, String>,
        cancel: CancellationToken,
        on_progress: Option<ProgressCallback>,
    ) -> Result<Value> {
        let (era, ver) = {
            let p = self
                .proto
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (p.era, p.version.clone())
        };
        let mut wire_params = params.clone();
        if era == ProtocolEra::Modern20260728 {
            inject_modern_meta(&mut wire_params, &ver);
        }

        let first = self
            .send_wire(
                method,
                wire_params,
                extra_headers.clone(),
                era,
                &ver,
                cancel.clone(),
                on_progress.clone(),
            )
            .await;
        let was_first_req = !self.first_request_done.swap(true, Ordering::AcqRel);

        match first {
            Ok(v) => Ok(v),
            // Case 0: `-32020 HeaderMismatch` -> invalidate stale cached tool schemas.
            Err(
                TransportError::Rpc(ref e)
                | TransportError::HttpStatus {
                    rpc_error: Some(ref e),
                    ..
                },
            ) if e.code == ERR_HEADER_MISMATCH => {
                crate::cache::invalidate_cached_schema(&self.cache_key);
                Err(TransportError::Rpc(e.clone()).into_carnot(&self.server_name))
            }
            // Case 1: `-32022 UnsupportedProtocolVersion` -> retry supported modern version or fallback to legacy.
            Err(TransportError::Rpc(ref e)) if e.code == ERR_UNSUPPORTED_PROTOCOL_VERSION => {
                let supported = e.supported_versions();
                if let Some(next_ver) = pick_best_modern_version(&supported) {
                    self.set_protocol(ProtocolEra::Modern20260728, next_ver.clone(), false);
                    let mut p2 = params;
                    inject_modern_meta(&mut p2, &next_ver);
                    return self
                        .send_wire(
                            method,
                            p2,
                            extra_headers,
                            ProtocolEra::Modern20260728,
                            &next_ver,
                            cancel,
                            on_progress,
                        )
                        .await
                        .map_err(|err| err.into_carnot(&self.server_name));
                } else if supports_any_legacy_version(&supported) {
                    self.ensure_legacy_handshake().await?;
                    return self
                        .send_wire(
                            method,
                            params,
                            BTreeMap::new(),
                            ProtocolEra::Legacy20251125,
                            MCP_LEGACY_PROTOCOL_VERSION,
                            cancel,
                            on_progress,
                        )
                        .await
                        .map_err(|err| err.into_carnot(&self.server_name));
                }
                Err(TransportError::Rpc(e.clone()).into_carnot(&self.server_name))
            }
            Err(TransportError::HttpStatus {
                status: 400,
                rpc_error: Some(ref e),
                ..
            }) if e.code == ERR_UNSUPPORTED_PROTOCOL_VERSION => {
                let supported = e.supported_versions();
                if let Some(next_ver) = pick_best_modern_version(&supported) {
                    self.set_protocol(ProtocolEra::Modern20260728, next_ver.clone(), false);
                    let mut p2 = params;
                    inject_modern_meta(&mut p2, &next_ver);
                    return self
                        .send_wire(
                            method,
                            p2,
                            extra_headers,
                            ProtocolEra::Modern20260728,
                            &next_ver,
                            cancel,
                            on_progress,
                        )
                        .await
                        .map_err(|err| err.into_carnot(&self.server_name));
                } else if supports_any_legacy_version(&supported) {
                    self.ensure_legacy_handshake().await?;
                    return self
                        .send_wire(
                            method,
                            params,
                            BTreeMap::new(),
                            ProtocolEra::Legacy20251125,
                            MCP_LEGACY_PROTOCOL_VERSION,
                            cancel,
                            on_progress,
                        )
                        .await
                        .map_err(|err| err.into_carnot(&self.server_name));
                }
                Err(TransportError::Rpc(e.clone()).into_carnot(&self.server_name))
            }
            // Case 2: HTTP 401/403 -> purge private cache immediately
            Err(TransportError::HttpStatus { status, body, .. })
                if status == 401 || status == 403 =>
            {
                crate::cache::invalidate_cached_schema(&self.cache_key);
                Err(CarnotError::Tool {
                    tool: format!("mcp:{}", self.server_name),
                    message: format!("HTTP {status} auth error (cache invalidated): {body}"),
                })
            }
            // Case 3: Optimistic modern HTTP hit a 4xx whose error body is not a
            // modern spec error (-32020..-32099) and NOT 404+-32601 -> fallback to legacy 2025-11-25!
            // Official TS/Python legacy SDKs return HTTP 400 with JSON-RPC `-32000` / `-32600`!
            Err(TransportError::HttpStatus {
                status,
                ref rpc_error,
                ..
            }) if era == ProtocolEra::Modern20260728
                && (400..500).contains(&status)
                && !rpc_error.as_ref().is_some_and(|e| {
                    e.is_modern_spec_error() || (status == 404 && e.code == -32601)
                }) =>
            {
                self.ensure_legacy_handshake().await?;
                self.send_wire(
                    method,
                    params,
                    BTreeMap::new(),
                    ProtocolEra::Legacy20251125,
                    MCP_LEGACY_PROTOCOL_VERSION,
                    cancel,
                    on_progress,
                )
                .await
                .map_err(|err| err.into_carnot(&self.server_name))
            }
            // Case 4: Legacy HTTP 404 session expired -> re-init ONCE & replay (STRICTLY legacy only!)
            Err(TransportError::HttpStatus { status: 404, .. })
                if era == ProtocolEra::Legacy20251125 && method != "initialize" =>
            {
                if let McpTransportInner::Http(h) = &self.inner {
                    h.clear_session();
                }
                self.set_protocol(
                    ProtocolEra::Legacy20251125,
                    MCP_LEGACY_PROTOCOL_VERSION.to_string(),
                    false,
                );
                self.ensure_legacy_handshake().await?;
                self.send_wire(
                    method,
                    params,
                    BTreeMap::new(),
                    ProtocolEra::Legacy20251125,
                    MCP_LEGACY_PROTOCOL_VERSION,
                    cancel,
                    on_progress,
                )
                .await
                .map_err(|err| err.into_carnot(&self.server_name))
            }
            // Case 5: Warm-hinted modern stdio fallback only on the very first
            // request, ONLY for -32601/-32600, and NEVER replaying non-idempotent `tools/call`!
            Err(TransportError::Rpc(ref e))
                if matches!(self.inner, McpTransportInner::Stdio(_))
                    && era == ProtocolEra::Modern20260728
                    && self.started_from_hint
                    && was_first_req
                    && matches!(e.code, -32601 | -32600)
                    && !matches!(method, "initialize" | "tools/call") =>
            {
                if self.ensure_legacy_handshake().await.is_ok()
                    && self.protocol_era() == ProtocolEra::Legacy20251125
                {
                    return self
                        .send_wire(
                            method,
                            params,
                            BTreeMap::new(),
                            ProtocolEra::Legacy20251125,
                            MCP_LEGACY_PROTOCOL_VERSION,
                            cancel,
                            on_progress,
                        )
                        .await
                        .map_err(|err| err.into_carnot(&self.server_name));
                }
                Err(TransportError::Rpc(e.clone()).into_carnot(&self.server_name))
            }
            Err(e) => Err(e.into_carnot(&self.server_name)),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "internal transport dispatcher forwards per-call wire parameters across stdio/http/sse"
    )]
    async fn send_wire(
        &self,
        method: &str,
        params: Value,
        extra_headers: BTreeMap<String, String>,
        era: ProtocolEra,
        protocol_version: &str,
        cancel: CancellationToken,
        on_progress: Option<ProgressCallback>,
    ) -> std::result::Result<Value, TransportError> {
        match &self.inner {
            McpTransportInner::Stdio(a) => {
                a.request(method, params, self.timeout, cancel, on_progress)
                    .await
            }
            McpTransportInner::Http(h) => {
                h.request_once(
                    method,
                    params,
                    extra_headers,
                    era,
                    protocol_version,
                    self.timeout,
                    cancel,
                    on_progress,
                )
                .await
            }
            McpTransportInner::LegacySse(s) => {
                s.request(method, params, self.timeout, cancel, on_progress)
                    .await
            }
        }
    }

    pub async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let (era, ver) = {
            let p = self
                .proto
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (p.era, p.version.clone())
        };
        match &self.inner {
            McpTransportInner::Stdio(a) => a.notify(method, params).await,
            McpTransportInner::Http(h) => h.notify(method, params, era, &ver).await,
            McpTransportInner::LegacySse(s) => s.notify(method, params).await,
        }
    }

    /// Paginates `tools/list` using `nextCursor` and captures `SEP-2549` `ttlMs` + `cacheScope`.
    pub async fn list_all_tools_with_meta(&self) -> Result<ListToolsResponse> {
        let mut all = Vec::new();
        let mut cursor: Option<String> = None;
        let mut min_ttl_ms: Option<u64> = None;
        let mut cache_scope = "public".to_string();

        for _page in 0..64 {
            let params = match &cursor {
                Some(c) => json!({ "cursor": c }),
                None => json!({}),
            };
            let res = self
                .request("tools/list", params, CancellationToken::new(), None)
                .await?;
            if let Some(ttl) = res.get("ttlMs").and_then(Value::as_u64) {
                min_ttl_ms = Some(min_ttl_ms.map_or(ttl, |cur| cur.min(ttl)));
            }
            if let Some(scope) = res.get("cacheScope").and_then(Value::as_str)
                && scope == "private"
            {
                cache_scope = "private".into();
            }
            if let Some(arr) = res.get("tools").and_then(Value::as_array) {
                all.extend(arr.iter().cloned());
            }
            cursor = res
                .get("nextCursor")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string);
            if cursor.is_none() {
                break;
            }
        }
        Ok(ListToolsResponse {
            tools: all,
            ttl_ms: min_ttl_ms,
            cache_scope,
        })
    }

    pub async fn list_all_tools(&self) -> Result<Vec<Value>> {
        Ok(self.list_all_tools_with_meta().await?.tools)
    }

    pub async fn list_resources(&self) -> Result<Vec<Value>> {
        let res = self
            .request("resources/list", json!({}), CancellationToken::new(), None)
            .await?;
        Ok(res
            .get("resources")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default())
    }

    pub async fn read_resource(&self, uri: &str, cancel: CancellationToken) -> Result<Value> {
        self.request("resources/read", json!({ "uri": uri }), cancel, None)
            .await
    }

    pub async fn list_prompts(&self) -> Result<Vec<Value>> {
        let res = self
            .request("prompts/list", json!({}), CancellationToken::new(), None)
            .await?;
        Ok(res
            .get("prompts")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default())
    }

    pub async fn get_prompt(
        &self,
        name: &str,
        arguments: Value,
        cancel: CancellationToken,
    ) -> Result<Value> {
        self.request(
            "prompts/get",
            json!({ "name": name, "arguments": arguments }),
            cancel,
            None,
        )
        .await
    }

    pub async fn shutdown(&self) {
        match &self.inner {
            McpTransportInner::Http(h) if self.protocol_era() == ProtocolEra::Legacy20251125 => {
                h.delete_session().await;
            }
            McpTransportInner::LegacySse(s) => {
                s.shutdown();
            }
            _ => {}
        }
    }

    pub fn is_closed(&self) -> bool {
        match &self.inner {
            McpTransportInner::Stdio(a) => {
                a.write_tx.is_closed() || a.closed.load(Ordering::Acquire)
            }
            McpTransportInner::Http(_) => false,
            McpTransportInner::LegacySse(s) => s.closed.load(Ordering::Acquire),
        }
    }

    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    pub fn cache_key(&self) -> &str {
        &self.cache_key
    }
}

struct TaskHandleInfo {
    task_id: String,
    poll_interval_ms: Option<u64>,
    ttl_ms: Option<u64>,
}

fn extract_active_task_info(res: &Value) -> Option<TaskHandleInfo> {
    let task_obj = res.get("task").unwrap_or(res);
    let task_id = task_obj.get("taskId").and_then(Value::as_str)?;
    let status = task_obj
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("working");
    if matches!(status, "working" | "queued" | "pending" | "input_required") {
        Some(TaskHandleInfo {
            task_id: task_id.to_string(),
            poll_interval_ms: task_obj
                .get("pollIntervalMs")
                .or_else(|| task_obj.get("pollInterval"))
                .and_then(Value::as_u64),
            ttl_ms: task_obj
                .get("ttlMs")
                .or_else(|| task_obj.get("ttl"))
                .and_then(Value::as_u64),
        })
    } else {
        None
    }
}

fn parse_rpc_error(err_val: &Value) -> RpcErrorPayload {
    RpcErrorPayload {
        code: err_val
            .get("code")
            .and_then(Value::as_i64)
            .unwrap_or(-32000),
        message: err_val
            .get("message")
            .and_then(Value::as_str)
            .map_or_else(|| err_val.to_string(), str::to_string),
        data: err_val.get("data").cloned(),
    }
}

fn find_sse_delim(buf: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i + 1 < buf.len() {
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some((i, 2));
        }
        if i + 3 < buf.len()
            && buf[i] == b'\r'
            && buf[i + 1] == b'\n'
            && buf[i + 2] == b'\r'
            && buf[i + 3] == b'\n'
        {
            return Some((i, 4));
        }
        i += 1;
    }
    None
}

fn build_mcp_http_client(
    server_name: &str,
    enforce_ssrf: bool,
    timeout_secs: Option<u64>,
) -> Result<reqwest::Client> {
    if enforce_ssrf {
        return crate::ssrf::build_ssrf_safe_client_with_timeout(timeout_secs)
            .map_err(CarnotError::Sandbox);
    }
    let make = || {
        let mut b = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(30));
        if let Some(secs) = timeout_secs {
            b = b.timeout(Duration::from_secs(secs));
        }
        b
    };
    if let Ok(client) = make().build() {
        return Ok(client);
    }
    let certs: Vec<reqwest::Certificate> = webpki_root_certs::TLS_SERVER_ROOT_CERTS
        .iter()
        .filter_map(|der| reqwest::Certificate::from_der(der.as_ref()).ok())
        .collect();
    make()
        .tls_certs_only(certs)
        .build()
        .map_err(|e| CarnotError::Tool {
            tool: format!("mcp:{server_name}"),
            message: format!("HTTP client init failed: {e}"),
        })
}
