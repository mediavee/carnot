//! Dedicated, ultra-high-performance MCP `2026-07-28` (stateless) + `2025-11-25` (legacy) crate.
//!
//! Architectural highlights:
//!   - **MCP `2026-07-28` stateless primary** (`SEP-2567`, `SEP-2575`, `SEP-2243`, `SEP-2549`,
//!     `SEP-2322`, `SEP-2663`, `SEP-2106`) with seamless backward compatibility to `2025-11-25`
//!   - Lock-free multiplexed Stdio actor + HTTP/2 Streamable HTTP (`MCP-Protocol-Version`,
//!     `Mcp-Method`, `Mcp-Name`, `Mcp-Param-*` with `=?base64?...?=` sentinel encoding)
//!   - Zero-cold-start lazy-spawn + user-level BLAKE3 disk schema cache persisting `ProtocolEra`,
//!     `ttlMs`, `cacheScope` & effective expanded auth hash (0 handshake RTTs on warm starts!)
//!   - MRTR (`resultType: "input_required"`) + official Tasks extension (`io.modelcontextprotocol/tasks`)
//!   - Canonicalized & minified JSON Schemas (`defer_loading: true` => 0 tokens in static cache prefix!)
//!   - Anti-DoS bounded `$ref` validation & MCPoison RCE/SSRF guards on `.mcp.json`

pub mod cache;
pub mod config;
pub mod ssrf;
pub mod transport;

pub use cache::{
    CachedServerSchema, CachedToolEntry, MCP_LEGACY_PROTOCOL_VERSION, MCP_PROTOCOL_VERSION,
    ProtocolEra, extract_and_validate_tool_schema, minify_and_canonicalize_schema,
};
pub use config::{
    HttpServerConfig, McpConfigFile, McpServerConfig, StdioServerConfig, load_mcp_configs,
};
pub use transport::{ListToolsResponse, McpClient, ProgressCallback, encode_mcp_header_value};

use carnot_cas::{CasStore, shape_text_output};
use carnot_core::{CarnotError, Result, ToolEffects, ToolOutput, ToolResultBlock, ToolSpec};
use carnot_tools::{BoxFuture, Tool};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Sanitizes & clamps MCP wire tool names to <=64 `[a-zA-Z0-9_-]` chars with BLAKE3 suffix on overflow.
pub fn sanitize_wire_name(server: &str, tool: &str) -> String {
    let raw = format!("mcp__{server}__{tool}");
    let clean: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if clean.len() <= 64 {
        clean
    } else {
        let hash = blake3::hash(raw.as_bytes()).to_hex();
        format!("{}_{}", &clean[..55], &hash[..8])
    }
}

/// Reconnectable lazy supervisor holding one MCP server's config, cached era hint, and live client.
#[derive(Debug)]
struct LazyServerHandle {
    name: String,
    cache_key: String,
    cfg: McpServerConfig,
    workspace_root: PathBuf,
    enforce_ssrf: bool,
    hint_era: Mutex<Option<(ProtocolEra, String)>>,
    conn: Mutex<Option<Arc<McpClient>>>,
}

impl LazyServerHandle {
    async fn get_or_connect(&self) -> Result<Arc<McpClient>> {
        let mut guard = self.conn.lock().await;
        if let Some(ref existing) = *guard
            && !existing.is_closed()
        {
            return Ok(Arc::clone(existing));
        }
        let hint = self.hint_era.lock().await.clone();
        let client = match &self.cfg {
            McpServerConfig::Stdio(s) => {
                McpClient::connect_stdio(
                    self.name.clone(),
                    self.cache_key.clone(),
                    s,
                    self.workspace_root.clone(),
                    hint,
                )
                .await?
            }
            McpServerConfig::Http(h) => {
                McpClient::connect_http(
                    self.name.clone(),
                    self.cache_key.clone(),
                    h,
                    self.workspace_root.clone(),
                    self.enforce_ssrf,
                    hint,
                )
                .await?
            }
        };
        *self.hint_era.lock().await = Some((client.protocol_era(), client.protocol_version()));
        let arc = Arc::new(client);
        *guard = Some(Arc::clone(&arc));
        Ok(arc)
    }

    async fn shutdown(&self) {
        let client = self.conn.lock().await.take();
        if let Some(c) = client {
            c.shutdown().await;
        }
    }
}

/// Summary of one registered MCP server (used by `carnot mcp list` and telemetry).
#[derive(Debug, Clone, serde::Serialize)]
pub struct McpServerSummary {
    pub name: String,
    pub transport: String,
    pub protocol_version: String,
    pub era: ProtocolEra,
    pub tools_count: usize,
    pub loaded_from_cache: bool,
    pub lazy: bool,
    pub trusted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Manages all configured MCP servers with zero-cold-start schema caching & lazy-spawn.
#[derive(Debug)]
pub struct McpManager {
    servers: BTreeMap<String, Arc<LazyServerHandle>>,
    summaries: Vec<McpServerSummary>,
    tools: Vec<Arc<dyn Tool>>,
    refresh_handles: std::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

impl McpManager {
    pub fn empty() -> Self {
        Self {
            servers: BTreeMap::new(),
            summaries: Vec::new(),
            tools: Vec::new(),
            refresh_handles: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Initializes an `McpManager` from validated server configs IN PARALLEL (`join_all`).
    ///   - Warm lazy cache registers tools in <1 ms with ZERO spawned processes & persisted `ProtocolEra`!
    ///   - When a cached schema is stale per `SEP-2549` `ttlMs`, serves the frozen snapshot for this
    ///     session (zero prompt-cache bust!) and spawns a background refresh for the *next* session!
    ///   - Per-server SSRF: enforced strictly on workspace-origin HTTP servers (user localhost/Tailscale allowed).
    pub async fn from_configs(
        configs: BTreeMap<String, McpServerConfig>,
        workspace_root: PathBuf,
        cas: Arc<CasStore>,
        cas_enabled: bool,
        spill_threshold: usize,
        preview_max: usize,
    ) -> Result<Self> {
        let init_futs =
            configs
                .into_iter()
                .filter(|(_, cfg)| cfg.is_enabled())
                .map(|(name, cfg)| {
                    let ws = workspace_root.clone();
                    async move {
                        let cache_key = cache::server_cache_key(&name, &cfg);
                        let (transport_kind, enforce_ssrf) = match &cfg {
                            McpServerConfig::Stdio(_) => ("stdio".to_string(), false),
                            McpServerConfig::Http(h) => {
                                ("streamable_http".to_string(), h.from_workspace)
                            }
                        };
                        let lazy = cfg.is_lazy();
                        let trusted = cfg.is_trusted();

                        let cached_opt = if lazy {
                            cache::load_cached_schema(&cache_key)
                        } else {
                            None
                        };
                        let initial_hint = cached_opt
                            .as_ref()
                            .map(|(c, _)| (c.era, c.protocol_version.clone()));

                        let handle = Arc::new(LazyServerHandle {
                            name: name.clone(),
                            cache_key: cache_key.clone(),
                            cfg: cfg.clone(),
                            workspace_root: ws,
                            enforce_ssrf,
                            hint_era: Mutex::new(initial_hint),
                            conn: Mutex::new(None),
                        });

                        let mut refresh_handle: Option<tokio::task::JoinHandle<()>> = None;
                        let load_res: Result<(Vec<CachedToolEntry>, bool, ProtocolEra, String)> =
                            if let Some((cached, is_stale)) = cached_opt {
                                if is_stale {
                                    // Stale-while-revalidate for NEXT session only (never mutates current prompt prefix!)
                                    // Clear era hint so a server upgraded from legacy -> 2026-07-28 re-negotiates (Opus #24!)
                                    let h_bg = Arc::clone(&handle);
                                    let nm_bg = name.clone();
                                    let ck_bg = cache_key.clone();
                                    refresh_handle = Some(tokio::spawn(async move {
                                        *h_bg.hint_era.lock().await = None;
                                        if let Ok(client) = h_bg.get_or_connect().await
                                            && let Ok(resp) =
                                                client.list_all_tools_with_meta().await
                                        {
                                            let parsed = parse_and_minify_tools(&nm_bg, resp.tools);
                                            cache::save_cached_schema(
                                                &ck_bg,
                                                CachedServerSchema {
                                                    format_version: 3,
                                                    protocol_version: client.protocol_version(),
                                                    era: client.protocol_era(),
                                                    server_name: nm_bg,
                                                    cache_scope: resp.cache_scope,
                                                    ttl_ms: resp.ttl_ms,
                                                    cached_at_ms: 0,
                                                    tools: parsed,
                                                },
                                            );
                                        }
                                    }));
                                }
                                Ok((cached.tools, true, cached.era, cached.protocol_version))
                            } else {
                                match handle.get_or_connect().await {
                                    Ok(client) => match client.list_all_tools_with_meta().await {
                                        Ok(resp) => {
                                            let era = client.protocol_era();
                                            let ver = client.protocol_version();
                                            let parsed = parse_and_minify_tools(&name, resp.tools);
                                            cache::save_cached_schema(
                                                &cache_key,
                                                CachedServerSchema {
                                                    format_version: 3,
                                                    protocol_version: ver.clone(),
                                                    era,
                                                    server_name: name.clone(),
                                                    cache_scope: resp.cache_scope,
                                                    ttl_ms: resp.ttl_ms,
                                                    cached_at_ms: 0,
                                                    tools: parsed.clone(),
                                                },
                                            );
                                            Ok((parsed, false, era, ver))
                                        }
                                        Err(e) => Err(e),
                                    },
                                    Err(e) => Err(e),
                                }
                            };

                        (
                            name,
                            transport_kind,
                            lazy,
                            trusted,
                            handle,
                            load_res,
                            refresh_handle,
                        )
                    }
                });

        let results = futures::future::join_all(init_futs).await;
        let mut servers = BTreeMap::new();
        let mut summaries = Vec::new();
        let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
        let mut refresh_handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();

        for (name, transport, lazy, trusted, handle, load_res, refresh_handle) in results {
            if let Some(jh) = refresh_handle {
                refresh_handles.push(jh);
            }
            match load_res {
                Ok((entries, from_cache, era, protocol_version)) => {
                    summaries.push(McpServerSummary {
                        name: name.clone(),
                        transport,
                        protocol_version,
                        era,
                        tools_count: entries.len(),
                        loaded_from_cache: from_cache,
                        lazy,
                        trusted,
                        error: None,
                    });
                    for entry in entries {
                        tools.push(Arc::new(McpToolProxy {
                            server: Arc::clone(&handle),
                            entry,
                            trusted,
                            cas: Arc::clone(&cas),
                            cas_enabled,
                            spill_threshold,
                            preview_max,
                        }));
                    }
                    servers.insert(name, handle);
                }
                Err(e) => {
                    tracing::warn!(server = %name, error = %e, "MCP server failed to initialize; skipping");
                    eprintln!("[carnot:mcp] warning: server `{name}` failed to initialize: {e}");
                    summaries.push(McpServerSummary {
                        name,
                        transport,
                        protocol_version: MCP_PROTOCOL_VERSION.into(),
                        era: ProtocolEra::Modern20260728,
                        tools_count: 0,
                        loaded_from_cache: false,
                        lazy,
                        trusted,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        // Deterministic tool order guarantees byte-identical prefix hashing across restarts
        tools.sort_by_key(|t| t.spec().name);

        Ok(Self {
            servers,
            summaries,
            tools,
            refresh_handles: std::sync::Mutex::new(refresh_handles),
        })
    }

    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    pub fn summaries(&self) -> &[McpServerSummary] {
        &self.summaries
    }

    pub async fn connect_server(&self, name: &str) -> Result<Arc<McpClient>> {
        let Some(handle) = self.servers.get(name) else {
            return Err(CarnotError::Tool {
                tool: "mcp".into(),
                message: format!("unknown MCP server `{name}`"),
            });
        };
        handle.get_or_connect().await
    }

    pub async fn shutdown_all(&self) {
        if let Ok(mut handles) = self.refresh_handles.lock() {
            for jh in handles.drain(..) {
                jh.abort();
            }
        }
        for h in self.servers.values() {
            h.shutdown().await;
        }
    }
}

impl Drop for McpManager {
    fn drop(&mut self) {
        if let Ok(mut handles) = self.refresh_handles.lock() {
            for jh in handles.drain(..) {
                jh.abort();
            }
        }
    }
}

fn parse_and_minify_tools(server_name: &str, raw_tools: Vec<Value>) -> Vec<CachedToolEntry> {
    let mut entries = Vec::with_capacity(raw_tools.len());
    for t in raw_tools {
        let remote_name = t
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if remote_name.is_empty() {
            continue;
        }
        let description = cache::truncate_description(
            t.get("description").and_then(Value::as_str).unwrap_or(""),
            280,
        );
        let raw_schema = t
            .get("inputSchema")
            .cloned()
            .unwrap_or_else(|| json!({ "type": "object" }));
        // Validate SEP-2106 ($ref/size bounds) and SEP-2243 (x-mcp-header) — exclude invalid tools!
        let (input_schema, header_params) = match extract_and_validate_tool_schema(&raw_schema) {
            Ok(pair) => pair,
            Err(reason) => {
                tracing::warn!(
                    server = %server_name,
                    tool = %remote_name,
                    reason = %reason,
                    "excluding MCP tool with invalid schema/annotation"
                );
                continue;
            }
        };
        let output_schema = t.get("outputSchema").map(minify_and_canonicalize_schema);
        let read_only_hint = t
            .pointer("/annotations/readOnlyHint")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        entries.push(CachedToolEntry {
            wire_name: sanitize_wire_name(server_name, &remote_name),
            remote_name,
            description,
            input_schema,
            output_schema,
            read_only_hint,
            header_params,
        });
    }
    entries.sort_by(|a, b| a.wire_name.cmp(&b.wire_name));
    entries
}

#[derive(Debug)]
struct McpToolProxy {
    server: Arc<LazyServerHandle>,
    entry: CachedToolEntry,
    trusted: bool,
    cas: Arc<CasStore>,
    cas_enabled: bool,
    spill_threshold: usize,
    preview_max: usize,
}

impl Tool for McpToolProxy {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.entry.wire_name.clone(),
            description: self.entry.description.clone(),
            input_schema: self.entry.input_schema.clone(),
            defer_loading: true,
        }
    }

    fn effects(&self, _input: &Value) -> ToolEffects {
        if self.trusted && self.entry.read_only_hint {
            ToolEffects::global_read()
        } else {
            ToolEffects::serial()
        }
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let client = self.server.get_or_connect().await?;
            // SEP-2243: mirror annotated `x-mcp-header` parameters (including nested paths!) into `Mcp-Param-*` headers
            let mut extra_headers = BTreeMap::new();
            for hp in &self.entry.header_params {
                let mut cur = &input;
                for seg in &hp.json_path {
                    match cur.get(seg) {
                        Some(next) => cur = next,
                        None => {
                            cur = &Value::Null;
                            break;
                        }
                    }
                }
                if let Some(encoded) = transport::format_param_header_value(cur) {
                    extra_headers.insert(hp.header_name.clone(), encoded);
                }
            }

            let res = client
                .request_with_headers(
                    "tools/call",
                    json!({ "name": self.entry.remote_name, "arguments": input }),
                    extra_headers,
                    cancel,
                    None,
                )
                .await?;

            let is_error = res.get("isError").and_then(Value::as_bool).unwrap_or(false);
            let mut text_parts: Vec<String> = Vec::new();
            let mut images: Vec<ToolResultBlock> = Vec::new();

            if let Some(arr) = res.get("content").and_then(Value::as_array) {
                for c in arr {
                    match c.get("type").and_then(Value::as_str) {
                        Some("image") => {
                            if let Some(data) = c.get("data").and_then(Value::as_str) {
                                let mime = c
                                    .get("mimeType")
                                    .and_then(Value::as_str)
                                    .unwrap_or("image/png");
                                if data.len() <= 7 * 1024 * 1024 {
                                    images.push(ToolResultBlock::Image {
                                        media_type: mime.to_string(),
                                        data_base64: data.to_string(),
                                    });
                                } else {
                                    text_parts.push(format!(
                                        "[MCP image ({mime}) omitted: exceeds 5 MiB size limit]"
                                    ));
                                }
                            }
                        }
                        Some("resource") => {
                            if let Some(r) = c.get("resource") {
                                let uri =
                                    r.get("uri").and_then(Value::as_str).unwrap_or("resource");
                                if let Some(t) = r.get("text").and_then(Value::as_str) {
                                    text_parts.push(format!("[resource: {uri}]\n{t}"));
                                } else {
                                    text_parts.push(format!("[binary resource: {uri}]"));
                                }
                            }
                        }
                        Some("resource_link") => {
                            let uri = c.get("uri").and_then(Value::as_str).unwrap_or("");
                            let name = c.get("name").and_then(Value::as_str).unwrap_or("link");
                            text_parts.push(format!("[resource_link: {name} -> {uri}]"));
                        }
                        Some("audio") => {
                            let mime = c.get("mimeType").and_then(Value::as_str).unwrap_or("audio");
                            text_parts.push(format!("[audio block ({mime}) omitted]"));
                        }
                        _ => {
                            if let Some(t) = c.get("text").and_then(Value::as_str) {
                                text_parts.push(t.to_string());
                            }
                        }
                    }
                }
            }

            // SEP-2106: `structuredContent` can be ANY JSON value (object, array, primitive)
            if text_parts.is_empty() && images.is_empty() {
                if let Some(sc) = res.get("structuredContent") {
                    text_parts.push(serde_json::to_string(sc).unwrap_or_else(|_| sc.to_string()));
                } else {
                    text_parts.push(res.to_string());
                }
            }

            let mut blocks = Vec::new();
            let mut handle_rel: Option<String> = None;
            if !text_parts.is_empty() {
                let joined = text_parts.join("\n");
                let shaped = shape_text_output(
                    &joined,
                    self.cas_enabled.then_some(self.cas.as_ref()),
                    self.spill_threshold,
                    self.preview_max,
                )?;
                handle_rel = shaped.spilled.map(|h| h.rel_path);
                blocks.push(ToolResultBlock::text(shaped.text));
            }
            blocks.extend(images);

            Ok(ToolOutput {
                blocks,
                is_error,
                details: json!({
                    "mcp_server": self.server.name,
                    "mcp_tool": self.entry.wire_name,
                    "protocol_version": client.protocol_version(),
                    "handle": handle_rel,
                }),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpListener;

    /// Proves MCP `2026-07-28` stateless stdio + MRTR `input_required` + Tasks `tasks/get` +
    /// 12-way lock-free multiplexing + persisted `ProtocolEra` warm zero-cold-start cache!
    #[tokio::test]
    async fn mcp_2026_07_28_stateless_stdio_mrtr_tasks_and_mux() {
        let dir = std::env::temp_dir().join(format!("carnot-mcp26-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let srv_py = dir.join("mock_mcp_2026.py");
        std::fs::write(
            &srv_py,
            r#"import sys, json, threading, time
lock = threading.Lock()
polls = {}
def reply(obj):
    with lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()
def handle(req):
    rid = req.get("id")
    m = req.get("method")
    p = req.get("params") or {}
    meta = p.get("_meta") or {}
    if m == "initialize":
        # Modern 2026-07-28 server rejects legacy initialize!
        reply({"jsonrpc":"2.0","id":rid,"error":{"code":-32022,"message":"stateless 2026-07-28 only","data":{"supported":["2026-07-28"]}}})
        return
    if meta.get("io.modelcontextprotocol/protocolVersion") != "2026-07-28":
        reply({"jsonrpc":"2.0","id":rid,"error":{"code":-32022,"message":"missing protocolVersion","data":{"supported":["2026-07-28"]}}})
        return
    if m == "server/discover":
        reply({"jsonrpc":"2.0","id":rid,"result":{"resultType":"complete","supportedVersions":["2026-07-28"],"capabilities":{"tools":{}},"ttlMs":60000,"cacheScope":"public"}})
    elif m == "tools/list":
        reply({"jsonrpc":"2.0","id":rid,"result":{"resultType":"complete","ttlMs":60000,"cacheScope":"public","tools":[
            {"name":"slow_echo","description":"Sleeps 100ms","inputSchema":{"type":"object","properties":{"x":{"type":"integer"}}},"annotations":{"readOnlyHint":True}},
            {"name":"mrtr_root","description":"Uses MRTR roots/list","inputSchema":{"type":"object","properties":{}}},
            {"name":"async_job","description":"Returns unsolicited task","inputSchema":{"type":"object","properties":{}}}
        ]}})
    elif m == "tools/call":
        name = p.get("name")
        if name == "slow_echo":
            x = (p.get("arguments") or {}).get("x", 0)
            time.sleep(0.10)
            reply({"jsonrpc":"2.0","id":rid,"result":{"resultType":"complete","content":[{"type":"text","text":f"echo:{x}"}]}})
        elif name == "mrtr_root":
            in_resp = p.get("inputResponses")
            if not in_resp:
                reply({"jsonrpc":"2.0","id":rid,"result":{
                    "resultType":"input_required",
                    "requestState":"state-xyz",
                    "inputRequests":{"ws":{"method":"roots/list","params":{}}}
                }})
            else:
                roots = in_resp.get("ws",{}).get("roots",[])
                uri = roots[0]["uri"] if roots else "none"
                reply({"jsonrpc":"2.0","id":rid,"result":{"resultType":"complete","content":[{"type":"text","text":f"root:{uri};state:{p.get('requestState')}"}]}})
        elif name == "async_job":
            reply({"jsonrpc":"2.0","id":rid,"result":{"resultType":"complete","task":{"taskId":"task-77","status":"working"}}})
    elif m == "tasks/get":
        tid = p.get("taskId")
        c = polls.get(tid, 0) + 1
        polls[tid] = c
        if c < 2:
            reply({"jsonrpc":"2.0","id":rid,"result":{"resultType":"complete","taskId":tid,"status":"working"}})
        else:
            reply({"jsonrpc":"2.0","id":rid,"result":{"resultType":"complete","taskId":tid,"status":"completed","result":{"resultType":"complete","content":[{"type":"text","text":"task-77-done"}]}}})
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    if "id" not in req: continue
    threading.Thread(target=handle, args=(req,), daemon=True).start()
"#,
        )
        .expect("write py");

        let cas = Arc::new(CasStore::open(&dir).expect("cas"));
        let srv_cfg = McpServerConfig::Stdio(StdioServerConfig {
            command: "python3".into(),
            args: vec![srv_py.to_string_lossy().to_string()],
            env: BTreeMap::new(),
            cwd: Some(dir.clone()),
            timeout_ms: 10_000,
            enabled: true,
            trusted: true,
            lazy: true,
        });
        let mut cfgs = BTreeMap::new();
        cfgs.insert("m26".to_string(), srv_cfg.clone());

        let mgr1 =
            McpManager::from_configs(cfgs.clone(), dir.clone(), cas.clone(), true, 4096, 1400)
                .await
                .expect("mgr1");
        assert_eq!(mgr1.summaries()[0].era, ProtocolEra::Modern20260728);
        assert_eq!(mgr1.summaries()[0].protocol_version, "2026-07-28");
        assert_eq!(mgr1.tools().len(), 3);

        // 1. Test 12-way concurrent multiplexing
        let slow_echo = mgr1
            .tools()
            .into_iter()
            .find(|t| t.spec().name == "mcp__m26__slow_echo")
            .expect("slow_echo");
        assert!(slow_echo.effects(&json!({})).global_read);
        let t0 = Instant::now();
        let mut handles = Vec::new();
        for i in 0..12 {
            let t = slow_echo.clone();
            handles.push(tokio::spawn(async move {
                t.execute(json!({"x": i}), CancellationToken::new()).await
            }));
        }
        for (i, h) in handles.into_iter().enumerate() {
            assert_eq!(
                h.await.expect("join").expect("exec").text_only(),
                format!("echo:{i}")
            );
        }
        assert!(
            t0.elapsed().as_millis() < 700,
            "multiplexed slow_echo should finish fast"
        );

        // 2. Test MRTR `input_required` -> `roots/list` round-trip!
        let mrtr_tool = mgr1
            .tools()
            .into_iter()
            .find(|t| t.spec().name == "mcp__m26__mrtr_root")
            .expect("mrtr");
        let mrtr_out = mrtr_tool
            .execute(json!({}), CancellationToken::new())
            .await
            .expect("mrtr exec");
        assert!(mrtr_out.text_only().contains("state:state-xyz"));
        assert!(mrtr_out.text_only().contains("root:file://"));

        // 3. Test unsolicited `io.modelcontextprotocol/tasks` polling!
        let task_tool = mgr1
            .tools()
            .into_iter()
            .find(|t| t.spec().name == "mcp__m26__async_job")
            .expect("task");
        let task_out = task_tool
            .execute(json!({}), CancellationToken::new())
            .await
            .expect("task exec");
        assert_eq!(task_out.text_only(), "task-77-done");

        // 4. Warm second manager loads from cache with persisted era & 0 subprocess spawns!
        let t_warm = Instant::now();
        let mgr2 = McpManager::from_configs(cfgs, dir.clone(), cas, true, 4096, 1400)
            .await
            .expect("mgr2");
        assert!(mgr2.summaries()[0].loaded_from_cache);
        assert_eq!(mgr2.summaries()[0].era, ProtocolEra::Modern20260728);
        assert!(t_warm.elapsed().as_millis() < 50);

        cache::invalidate_cached_schema(&cache::server_cache_key("m26", &srv_cfg));
        mgr1.shutdown_all().await;
        mgr2.shutdown_all().await;
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Proves MCP `2026-07-28` Streamable HTTP: 0 `initialize` calls, required `MCP-Protocol-Version`,
    /// `Mcp-Method`, `Mcp-Name`, `Mcp-Param-*` (`=?base64?...?=` for non-ASCII), and `ttlMs`!
    #[tokio::test]
    async fn streamable_http_2026_07_28_stateless_headers_and_zero_handshake() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let req_count = Arc::new(AtomicUsize::new(0));
        let rc = Arc::clone(&req_count);

        let srv_task = tokio::spawn(async move {
            for _ in 0..4 {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let rc = Arc::clone(&rc);
                tokio::spawn(async move {
                    let (r, mut w) = stream.into_split();
                    let mut br = BufReader::new(r);
                    let mut line = String::new();
                    let _ = br.read_line(&mut line).await;
                    let mut headers = HashMap::new();
                    let mut cl = 0usize;
                    loop {
                        let mut h = String::new();
                        let _ = br.read_line(&mut h).await;
                        if h.trim().is_empty() {
                            break;
                        }
                        if let Some((k, v)) = h.split_once(':') {
                            let lk = k.trim().to_ascii_lowercase();
                            if lk == "content-length" {
                                cl = v.trim().parse().unwrap_or(0);
                            }
                            headers.insert(lk, v.trim().to_string());
                        }
                    }
                    let mut body = vec![0u8; cl];
                    if cl > 0 {
                        use tokio::io::AsyncReadExt;
                        let _ = br.read_exact(&mut body).await;
                    }
                    rc.fetch_add(1, Ordering::Relaxed);
                    let req: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
                    let id = req.get("id").and_then(Value::as_u64).unwrap_or(1);

                    // Strict 2026-07-28 header validation!
                    assert_eq!(
                        headers.get("mcp-protocol-version").map(String::as_str),
                        Some("2026-07-28")
                    );
                    assert_eq!(headers.get("mcp-method").map(String::as_str), Some(method));

                    if method == "tools/list" {
                        let payload = json!({
                            "jsonrpc":"2.0",
                            "id":id,
                            "result":{
                                "resultType":"complete",
                                "ttlMs": 30000,
                                "cacheScope": "public",
                                "tools":[{
                                    "name":"greet",
                                    "description":"Greets a user",
                                    "inputSchema":{
                                        "type":"object",
                                        "properties":{
                                            "tenant":{"type":"string","x-mcp-header":"Tenant"}
                                        }
                                    }
                                }]
                            }
                        })
                        .to_string();
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                            payload.len()
                        );
                        let _ = w.write_all(resp.as_bytes()).await;
                    } else if method == "tools/call" {
                        assert_eq!(headers.get("mcp-name").map(String::as_str), Some("greet"));
                        let tenant_hdr =
                            headers.get("mcp-param-tenant").cloned().unwrap_or_default();
                        let payload = json!({
                            "jsonrpc":"2.0",
                            "id":id,
                            "result":{
                                "resultType":"complete",
                                "content":[{"type":"text","text":format!("hdr:{tenant_hdr}")}]
                            }
                        })
                        .to_string();
                        let sse = format!("data: {payload}\n\n");
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse}",
                            sse.len()
                        );
                        let _ = w.write_all(resp.as_bytes()).await;
                    }
                });
            }
        });

        let dir = std::env::temp_dir().join(format!("carnot-http26-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let cas = Arc::new(CasStore::open(&dir).expect("cas"));
        let srv_cfg = McpServerConfig::Http(HttpServerConfig {
            url: format!("http://{addr}/mcp"),
            transport: None,
            headers: BTreeMap::new(),
            timeout_ms: 5_000,
            enabled: true,
            trusted: false,
            lazy: false,
            from_workspace: false,
        });
        let mut cfgs = BTreeMap::new();
        cfgs.insert("h26".into(), srv_cfg.clone());

        let mgr = McpManager::from_configs(cfgs, dir.clone(), cas, true, 4096, 1400)
            .await
            .expect("mgr");
        let tool = mgr.tools()[0].clone();
        // Non-ASCII tenant triggers exact SEP-2243 `=?base64?...?=` sentinel encoding!
        let out = tool
            .execute(json!({"tenant": "café"}), CancellationToken::new())
            .await
            .expect("call");
        assert!(
            out.text_only().starts_with("hdr:=?base64?") && out.text_only().ends_with("?="),
            "expected SEP-2243 base64 sentinel header, got: {}",
            out.text_only()
        );
        // EXACTLY 2 HTTP round-trips total (tools/list + tools/call) — ZERO initialize handshake!
        assert_eq!(req_count.load(Ordering::Relaxed), 2);

        cache::invalidate_cached_schema(&cache::server_cache_key("h26", &srv_cfg));
        mgr.shutdown_all().await;
        srv_task.abort();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Proves seamless automatic fallback to Legacy `2025-11-25` Streamable HTTP (`Mcp-Session-Id`).
    #[tokio::test]
    async fn streamable_http_falls_back_cleanly_to_legacy_2025_11_25() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");

        let srv_task = tokio::spawn(async move {
            for _ in 0..6 {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    let (r, mut w) = stream.into_split();
                    let mut br = BufReader::new(r);
                    let mut line = String::new();
                    let _ = br.read_line(&mut line).await;
                    let mut cl = 0usize;
                    let mut has_session = false;
                    loop {
                        let mut h = String::new();
                        let _ = br.read_line(&mut h).await;
                        if h.trim().is_empty() {
                            break;
                        }
                        if let Some((k, v)) = h.split_once(':') {
                            if k.eq_ignore_ascii_case("content-length") {
                                cl = v.trim().parse().unwrap_or(0);
                            }
                            if k.eq_ignore_ascii_case("mcp-session-id") {
                                has_session = true;
                            }
                        }
                    }
                    let mut body = vec![0u8; cl];
                    if cl > 0 {
                        use tokio::io::AsyncReadExt;
                        let _ = br.read_exact(&mut body).await;
                    }
                    let req: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
                    let id = req.get("id").and_then(Value::as_u64).unwrap_or(1);

                    if method == "initialize" {
                        let payload = json!({"jsonrpc":"2.0","id":id,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"legacy","version":"1.0"}}}).to_string();
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: sess-99\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                            payload.len()
                        );
                        let _ = w.write_all(resp.as_bytes()).await;
                    } else if method == "notifications/initialized" {
                        let _ = w.write_all(b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await;
                    } else if !has_session {
                        // Official TypeScript/Python legacy MCP SDKs return HTTP 400 WITH a JSON-RPC `-32000` body (Opus P0 #1!)
                        let err_body = json!({"jsonrpc":"2.0","id":id,"error":{"code":-32000,"message":"Bad Request: No valid session ID provided"}}).to_string();
                        let resp = format!(
                            "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{err_body}",
                            err_body.len()
                        );
                        let _ = w.write_all(resp.as_bytes()).await;
                    } else if method == "tools/list" {
                        let payload = json!({"jsonrpc":"2.0","id":id,"result":{"tools":[{"name":"legacy_ping","description":"ok","inputSchema":{"type":"object"}}]}}).to_string();
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                            payload.len()
                        );
                        let _ = w.write_all(resp.as_bytes()).await;
                    }
                });
            }
        });

        let cfg = HttpServerConfig {
            url: format!("http://{addr}/mcp"),
            transport: None,
            headers: BTreeMap::new(),
            timeout_ms: 5_000,
            enabled: true,
            trusted: false,
            lazy: false,
            from_workspace: false,
        };
        let client = McpClient::connect_http(
            "legacy".into(),
            "test-legacy".into(),
            &cfg,
            std::env::temp_dir(),
            false,
            None,
        )
        .await
        .expect("connect");
        let tools = client.list_all_tools().await.expect("list");
        assert_eq!(client.protocol_era(), ProtocolEra::Legacy20251125);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "legacy_ping");
        client.shutdown().await;
        srv_task.abort();
    }

    #[tokio::test]
    async fn legacy_sse_2024_11_05_connect_list_and_call() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;
        use tokio::sync::broadcast;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let (sse_tx, _) = broadcast::channel::<String>(32);
        let sse_tx_srv = sse_tx.clone();

        let srv_task = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let tx = sse_tx_srv.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 16384];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    if req.starts_with("GET /sse") {
                        let mut rx = tx.subscribe();
                        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\r\nevent: endpoint\ndata: /messages?sessionId=s1\n\n";
                        if stream.write_all(head.as_bytes()).await.is_err() {
                            return;
                        }
                        while let Ok(frame) = rx.recv().await {
                            if stream.write_all(frame.as_bytes()).await.is_err() {
                                break;
                            }
                        }
                    } else if req.starts_with("POST /messages") {
                        let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
                        let val: Value = serde_json::from_str(body).unwrap_or(Value::Null);
                        let id = val.get("id").and_then(Value::as_u64);
                        let method = val.get("method").and_then(Value::as_str).unwrap_or("");
                        let ack = "HTTP/1.1 202 Accepted\r\nContent-Length: 8\r\nConnection: close\r\n\r\nAccepted";
                        let _ = stream.write_all(ack.as_bytes()).await;
                        if let Some(id) = id {
                            let res_payload = match method {
                                "initialize" => {
                                    assert_eq!(val["params"]["protocolVersion"], "2024-11-05");
                                    json!({"jsonrpc":"2.0","id":id,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"mock-sse","version":"1.0"}}})
                                }
                                "tools/list" => {
                                    json!({"jsonrpc":"2.0","id":id,"result":{"tools":[{"name":"sse_echo","description":"echoes text","inputSchema":{"type":"object","properties":{"msg":{"type":"string"}}}}]}})
                                }
                                "tools/call" => {
                                    let msg =
                                        val["params"]["arguments"]["msg"].as_str().unwrap_or("");
                                    json!({"jsonrpc":"2.0","id":id,"result":{"content":[{"type":"text","text":format!("echo:{msg}")}]}})
                                }
                                _ => json!({"jsonrpc":"2.0","id":id,"result":{}}),
                            };
                            let _ = tx.send(format!("event: message\ndata: {res_payload}\n\n"));
                        }
                    }
                });
            }
        });

        let cfg = HttpServerConfig {
            url: format!("http://{addr}/sse"),
            transport: Some("sse".into()),
            headers: BTreeMap::new(),
            timeout_ms: 5_000,
            enabled: true,
            trusted: false,
            lazy: false,
            from_workspace: false,
        };
        let client = McpClient::connect_http(
            "mock_sse".into(),
            "test-legacy-sse".into(),
            &cfg,
            std::env::temp_dir(),
            false,
            None,
        )
        .await
        .expect("connect legacy sse");
        assert_eq!(client.protocol_version(), "2024-11-05");
        let tools = client.list_all_tools().await.expect("list");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "sse_echo");
        let call = client
            .request(
                "tools/call",
                json!({"name": "sse_echo", "arguments": {"msg": "hi"}}),
                CancellationToken::new(),
                None,
            )
            .await
            .expect("call");
        assert_eq!(call["content"][0]["text"], "echo:hi");
        client.shutdown().await;
        srv_task.abort();
    }
}
