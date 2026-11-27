//! Agent Client Protocol (ACP) JSON-RPC 2.0 transport over stdio — Zed / T3 Code compliant:
//!   - Full ACP v1/v2 surface: `initialize`, `authenticate`, `session/new` (per-session `cwd` +
//!     `configOptions` for model & `thought_level` + `available_commands_update`),
//!     `session/set_config_option`, `session/set_model`, `session/prompt` (multimodal text + images + `/compact`),
//!     `session/cancel`, `session/steer`, `session/list` (real workspace `cwd` from JSONL headers),
//!     `session/load`/`session/resume`, `session/close`, `session/events`
//!   - True multi-session routing keyed by `sessionId` (concurrent sessions never overwrite each other)
//!   - Rich ACP `sessionUpdate` payloads (`agent_message_chunk`, `agent_thought_chunk`, `tool_call` with
//!     `kind`/`title`/`rawInput`/`locations`, `tool_call_update` with `content`/`rawOutput`, `usage_update`,
//!     `available_commands_update`)
//!   - Per-session bidirectional `AcpPermissionBroker` (`session/request_permission`)

mod brokers;
mod mapping;
#[cfg(test)]
mod tests;

use self::brokers::{AcpPermissionBroker, AcpQuestionBroker, write_frame};
use self::mapping::{
    acp_update_frame, available_commands_frame, build_config_options, extract_acp_prompt_blocks,
    list_recent_sessions, map_acp_stop_reason, parse_acp_effort, prompt_blocks_text,
    resolve_confined_session_path,
};
use carnot_agent::Agent;
use carnot_ai::Provider;
use carnot_core::{AgentEvent, CacheTtl, CarnotError, ContentBlock, ReasoningEffort, Result, Role};
use carnot_ext::BoxFuture;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinHandle;

type PendingReplies = Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>;

/// Advertised model descriptor exposed via ACP `configOptions` (`category: "model"`) and `models`.
#[derive(Debug, Clone)]
pub struct AcpModelInfo {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
}

/// Resolved provider + limits returned by [`AcpHost::resolve_model`].
#[derive(Debug, Clone)]
pub struct ResolvedAcpModel {
    pub provider: Arc<dyn Provider>,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u32>,
}

/// Optional host hooks enabling dynamic per-session `cwd` workspaces and live multi-provider
/// model switching inside `serve_acp_stdio_with_host`.
pub trait AcpHost: Send + Sync + std::fmt::Debug {
    /// Models advertised to the ACP client's model picker.
    fn available_models(&self) -> Vec<AcpModelInfo>;

    /// Resolves any requested model ID (advertised or custom) to its provider & window limits.
    fn resolve_model(&self, model_id: &str) -> Result<ResolvedAcpModel>;

    /// Creates a fresh `Agent` rooted at `cwd` for `session/new` (with optional ACP `mcpServers`).
    fn create_agent_for_cwd(
        &self,
        cwd: PathBuf,
        model: Option<String>,
        effort: Option<ReasoningEffort>,
        mcp_servers: Vec<Value>,
    ) -> BoxFuture<'_, Result<Arc<Agent>>>;
}

/// Resolves the target session `Agent` by explicit `params.sessionId` or current default session.
fn resolve_session_agent(
    sessions: &HashMap<String, Arc<Agent>>,
    default_sid: &str,
    params: &Value,
) -> std::result::Result<(String, Arc<Agent>), (i32, String)> {
    if let Some(req_sid) = params
        .get("sessionId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        sessions
            .get(req_sid)
            .map(|a| (req_sid.to_string(), Arc::clone(a)))
            .ok_or_else(|| (-32004, format!("unknown ACP sessionId `{req_sid}`")))
    } else {
        sessions
            .get(default_sid)
            .map(|a| (default_sid.to_string(), Arc::clone(a)))
            .ok_or_else(|| (-32004, "no active ACP session".into()))
    }
}

/// Starts a stdio ACP server bound to a single pre-built `Agent` (backwards-compatible entry point).
pub async fn serve_acp_stdio(agent: Arc<Agent>) -> Result<()> {
    serve_acp_stdio_inner(agent, None).await
}

/// Starts a full-featured stdio ACP server with dynamic per-session `cwd` creation, true multi-session
/// routing, live model picker advertising, and live multi-provider model switching (`T3 Code` / `Zed` compliant).
pub async fn serve_acp_stdio_with_host(
    initial_agent: Arc<Agent>,
    host: Arc<dyn AcpHost>,
) -> Result<()> {
    serve_acp_stdio_inner(initial_agent, Some(host)).await
}

async fn serve_acp_stdio_inner(
    initial_agent: Arc<Agent>,
    host: Option<Arc<dyn AcpHost>>,
) -> Result<()> {
    let stdin = BufReader::new(tokio::io::stdin());
    let stdout = Arc::new(Mutex::new(tokio::io::stdout()));
    let pending_replies: PendingReplies = Arc::new(Mutex::new(HashMap::new()));
    let next_client_req_id = Arc::new(AtomicU64::new(100_000));

    let initial_sid = initial_agent.session_id().await;
    let default_session_dir = initial_agent.config.session_dir.clone();
    let default_workspace_root = initial_agent.config.workspace_root.clone();
    let mut sessions: HashMap<String, Arc<Agent>> = HashMap::new();
    sessions.insert(initial_sid.clone(), initial_agent);
    let mut default_sid = initial_sid;
    let mut client_wants_perms = false;

    let mut lines = stdin.lines();
    let mut prompt_tasks: Vec<JoinHandle<()>> = Vec::new();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };

        // Route incoming JSON-RPC responses (e.g. client reply to `session/request_permission`)
        if msg.get("method").is_none() && msg.get("id").is_some() {
            if let Some(id_num) = msg.get("id").and_then(Value::as_u64)
                && let Some(tx) = pending_replies.lock().await.remove(&id_num)
            {
                let _ = tx.send(msg);
            }
            continue;
        }

        let has_id = msg.get("id").is_some();
        let id = msg.get("id").cloned().unwrap_or(Value::Null);
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        if method == "session/prompt" {
            prompt_tasks.retain(|h| !h.is_finished());
            let (sid, ag) = match resolve_session_agent(&sessions, &default_sid, &params) {
                Ok(pair) => pair,
                Err((code, message)) => {
                    if has_id {
                        let err_frame = json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {"code": code, "message": message}
                        });
                        write_frame(&stdout, &err_frame).await?;
                    }
                    continue;
                }
            };
            if ag.is_busy() {
                if has_id {
                    let busy_frame = json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {"code": -32001, "message": "agent is busy running a prompt; call session/steer or session/cancel first"}
                    });
                    write_frame(&stdout, &busy_frame).await?;
                }
                continue;
            }
            let prompt_blocks = extract_acp_prompt_blocks(&params);
            let prompt_text_only = prompt_blocks_text(&prompt_blocks);
            let out = Arc::clone(&stdout);
            prompt_tasks.push(tokio::spawn(async move {
                // Handle `/compact [focus]` slash command natively inside ACP!
                let trimmed = prompt_text_only.trim();
                let has_images = prompt_blocks
                    .iter()
                    .any(|b| matches!(b, ContentBlock::Image { .. }));
                if !has_images && (trimmed == "/compact" || trimmed.starts_with("/compact ")) {
                    let focus = trimmed
                        .strip_prefix("/compact")
                        .map(str::trim)
                        .filter(|s| !s.is_empty());
                    let frame = match ag.compact_now_with_focus(focus).await {
                        Ok((before, after, semantic)) => {
                            let mode_lbl = if semantic {
                                "cache-warm semantic"
                            } else {
                                "deterministic"
                            };
                            let msg_text = format!(
                                "Compacted session ({mode_lbl}): {before} -> {after} tokens."
                            );
                            let chunk = json!({
                                "jsonrpc": "2.0",
                                "method": "session/update",
                                "params": {
                                    "sessionId": sid,
                                    "update": {
                                        "sessionUpdate": "agent_message_chunk",
                                        "content": { "type": "text", "text": msg_text },
                                    }
                                }
                            });
                            let _ = write_frame(&out, &chunk).await;
                            let size = ag.current_context_window().await;
                            let met = ag.metrics().await;
                            let ttl = ag.current_cache_ttl().await;
                            let usage_frame = json!({
                                "jsonrpc": "2.0",
                                "method": "session/update",
                                "params": {
                                    "sessionId": sid,
                                    "update": {
                                        "sessionUpdate": "usage_update",
                                        "used": after,
                                        "size": size,
                                        "cost": { "amount": met.total_usage.cost_usd, "currency": "USD" },
                                        "_meta": {
                                            "carnot": {
                                                "cache_ttl": ttl.as_str(),
                                                "cache_hit_rate": met.cache_hit_rate(),
                                                "prefix_retention_rate": met.prefix_retention_rate(),
                                                "cache_breaks": met.cache_breaks,
                                            }
                                        }
                                    }
                                }
                            });
                            let _ = write_frame(&out, &usage_frame).await;
                            json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "result": { "stopReason": "end_turn" }
                            })
                        }
                        Err(e) => json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {"code": -32000, "message": e.to_string()}
                        }),
                    };
                    let _ = write_frame(&out, &frame).await;
                    return;
                }

                let mut ev_stream = ag.subscribe();
                let out_writer = Arc::clone(&out);
                let ag_obs = Arc::clone(&ag);
                let sid_obs = sid.clone();
                let pump = tokio::spawn(async move {
                    loop {
                        let env = ev_stream.next().await;
                        let is_done = matches!(
                            env.event,
                            AgentEvent::RunSettled { .. } | AgentEvent::Error { .. }
                        );
                        match &env.event {
                            AgentEvent::UsageRecorded { .. } => {
                                let used = ag_obs.current_context_tokens().await;
                                let size = ag_obs.current_context_window().await;
                                let met = ag_obs.metrics().await;
                                let ttl = ag_obs.current_cache_ttl().await;
                                let uframe = json!({
                                    "jsonrpc": "2.0",
                                    "method": "session/update",
                                    "params": {
                                        "sessionId": sid_obs,
                                        "seq": env.seq,
                                        "update": {
                                            "sessionUpdate": "usage_update",
                                            "used": used,
                                            "size": size,
                                            "cost": {
                                                "amount": met.total_usage.cost_usd,
                                                "currency": "USD",
                                            },
                                            "_meta": {
                                                "carnot": {
                                                    "cache_ttl": ttl.as_str(),
                                                    "cache_hit_rate": met.cache_hit_rate(),
                                                    "prefix_retention_rate": met.prefix_retention_rate(),
                                                    "cache_breaks": met.cache_breaks,
                                                }
                                            }
                                        }
                                    }
                                });
                                let _ = write_frame(&out_writer, &uframe).await;
                            }
                            AgentEvent::CompactionApplied {
                                mode,
                                tokens_before,
                                tokens_after,
                                semantic_llm,
                            } => {
                                let kind = if *semantic_llm {
                                    "semantic"
                                } else {
                                    "deterministic"
                                };
                                let thought = json!({
                                    "jsonrpc": "2.0",
                                    "method": "session/update",
                                    "params": {
                                        "sessionId": sid_obs,
                                        "seq": env.seq,
                                        "update": {
                                            "sessionUpdate": "agent_thought_chunk",
                                            "content": {
                                                "type": "text",
                                                "text": format!("[carnot:compact:{mode}/{kind}] {tokens_before} -> {tokens_after} tokens"),
                                            }
                                        }
                                    }
                                });
                                let _ = write_frame(&out_writer, &thought).await;
                                let size = ag_obs.current_context_window().await;
                                let met = ag_obs.metrics().await;
                                let ttl = ag_obs.current_cache_ttl().await;
                                let uframe = json!({
                                    "jsonrpc": "2.0",
                                    "method": "session/update",
                                    "params": {
                                        "sessionId": sid_obs,
                                        "seq": env.seq,
                                        "update": {
                                            "sessionUpdate": "usage_update",
                                            "used": *tokens_after,
                                            "size": size,
                                            "cost": {
                                                "amount": met.total_usage.cost_usd,
                                                "currency": "USD",
                                            },
                                            "_meta": {
                                                "carnot": {
                                                    "cache_ttl": ttl.as_str(),
                                                    "cache_hit_rate": met.cache_hit_rate(),
                                                    "prefix_retention_rate": met.prefix_retention_rate(),
                                                    "cache_breaks": met.cache_breaks,
                                                }
                                            }
                                        }
                                    }
                                });
                                let _ = write_frame(&out_writer, &uframe).await;
                            }
                            _ => {
                                if let Some(notif) = acp_update_frame(&sid_obs, &env) {
                                    let _ = write_frame(&out_writer, &notif).await;
                                }
                            }
                        }
                        if is_done {
                            break;
                        }
                    }
                });
                let outcome = ag.prompt_blocks(prompt_blocks).await;
                let _ = pump.await;
                let frame = match outcome {
                    Ok(o) => {
                        let stop = map_acp_stop_reason(o.halted_reason.as_deref());
                        json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "stopReason": stop,
                                "finalText": o.final_text,
                                "haltedReason": o.halted_reason,
                                "metrics": o.metrics,
                                "latestSeq": ag.latest_event_seq(),
                            }
                        })
                    }
                    Err(CarnotError::Cancelled) => json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": { "stopReason": "cancelled" }
                    }),
                    Err(e) => json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {"code": -32000, "message": e.to_string()}
                    }),
                };
                let _ = write_frame(&out, &frame).await;
            }));
            continue;
        }

        let mut post_send_commands_sid: Option<String> = None;

        let result: std::result::Result<Value, (i32, String)> = match method {
            "initialize" => {
                let client_caps = params.get("clientCapabilities");
                client_wants_perms = client_caps
                    .and_then(|c| {
                        c.get("permissionRequests")
                            .or_else(|| c.get("requestPermission"))
                    })
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                for (sid, ag) in &sessions {
                    ag.set_question_broker(Arc::new(AcpQuestionBroker {
                        session_id: sid.clone(),
                        stdout: Arc::clone(&stdout),
                        next_req_id: Arc::clone(&next_client_req_id),
                        pending: Arc::clone(&pending_replies),
                    }))
                    .await;
                    if client_wants_perms {
                        ag.set_permission_broker(Arc::new(AcpPermissionBroker::new(
                            sid.clone(),
                            Arc::clone(&stdout),
                            Arc::clone(&next_client_req_id),
                            Arc::clone(&pending_replies),
                        )))
                        .await;
                    }
                }
                Ok(json!({
                    "protocolVersion": 1,
                    "agentInfo": {
                        "name": "carnot",
                        "title": "Carnot",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "agentCapabilities": {
                        "loadSession": true,
                        "permissionRequests": true,
                        "askUser": true,
                        "mcpCapabilities": { "http": true, "sse": true },
                        "promptCapabilities": { "embeddedContext": true, "image": true },
                        "sessionCapabilities": { "list": {}, "resume": {} }
                    },
                    "authMethods": []
                }))
            }
            "authenticate" => Ok(json!({})),
            "session/new" => {
                let req_cwd = params
                    .get("cwd")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(PathBuf::from)
                    .filter(|p| p.is_dir());

                let req_mcp_servers: Vec<Value> = params
                    .get("mcpServers")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();

                let active_ag = if let Some(ref h) = host {
                    let prev_ag = sessions.get(&default_sid).cloned();
                    let target_cwd = req_cwd.unwrap_or_else(|| {
                        prev_ag.as_ref().map_or_else(
                            || default_workspace_root.clone(),
                            |a| a.config.workspace_root.clone(),
                        )
                    });
                    let (cur_model, cur_effort) = match prev_ag {
                        Some(ref a) => (
                            Some(a.current_model().await),
                            Some(a.current_effort().await),
                        ),
                        None => (None, None),
                    };
                    match h
                        .create_agent_for_cwd(target_cwd, cur_model, cur_effort, req_mcp_servers)
                        .await
                    {
                        Ok(new_agent) => {
                            let new_sid = new_agent.session_id().await;
                            new_agent
                                .set_question_broker(Arc::new(AcpQuestionBroker {
                                    session_id: new_sid.clone(),
                                    stdout: Arc::clone(&stdout),
                                    next_req_id: Arc::clone(&next_client_req_id),
                                    pending: Arc::clone(&pending_replies),
                                }))
                                .await;
                            if client_wants_perms {
                                new_agent
                                    .set_permission_broker(Arc::new(AcpPermissionBroker::new(
                                        new_sid.clone(),
                                        Arc::clone(&stdout),
                                        Arc::clone(&next_client_req_id),
                                        Arc::clone(&pending_replies),
                                    )))
                                    .await;
                            }
                            sessions.insert(new_sid.clone(), Arc::clone(&new_agent));
                            default_sid = new_sid;
                            Ok(new_agent)
                        }
                        Err(e) => Err((-32000, format!("failed to start session: {e}"))),
                    }
                } else {
                    sessions
                        .get(&default_sid)
                        .cloned()
                        .ok_or_else(|| (-32004, "no active ACP session".into()))
                };

                match active_ag {
                    Ok(ag) => {
                        let sid = ag.session_id().await;
                        post_send_commands_sid = Some(sid.clone());
                        let cur_model = ag.current_model().await;
                        let cur_effort = ag.current_effort().await;
                        let avail = host
                            .as_ref()
                            .map(|h| h.available_models())
                            .unwrap_or_default();
                        let cur_ttl = ag.current_cache_ttl().await;
                        let (config_options, models_state) =
                            build_config_options(&cur_model, cur_effort, cur_ttl, &avail);
                        let prefix_hash = ag.prefix_hash().await;
                        Ok(json!({
                            "sessionId": sid,
                            "configOptions": config_options,
                            "models": models_state,
                            "_meta": {
                                "carnot": {
                                    "model": cur_model,
                                    "prefixHash": prefix_hash,
                                    "latestSeq": ag.latest_event_seq(),
                                    "modules": ag.config.modules,
                                }
                            }
                        }))
                    }
                    Err(e) => Err(e),
                }
            }
            "session/set_config_option" => {
                match resolve_session_agent(&sessions, &default_sid, &params) {
                    Err(e) => Err(e),
                    Ok((_, ag)) => {
                        let config_id = params
                            .get("configId")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .trim();
                        let value = params
                            .get("value")
                            .or_else(|| params.get("optionId"))
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .trim();

                        let apply_res: std::result::Result<(), (i32, String)> = match config_id {
                            "model" => {
                                if value.is_empty() {
                                    Err((-32602, "model value must not be empty".into()))
                                } else if let Some(ref h) = host {
                                    match h.resolve_model(value) {
                                        Ok(resolved) => ag
                                            .set_model(
                                                value.to_string(),
                                                resolved.provider,
                                                resolved.context_window,
                                                resolved.max_output_tokens,
                                            )
                                            .await
                                            .map_err(|e| (-32001, e.to_string())),
                                        Err(e) => Err((-32602, e.to_string())),
                                    }
                                } else {
                                    Err((-32601, "dynamic model switching requires AcpHost".into()))
                                }
                            }
                            "thought_level" => match parse_acp_effort(value) {
                                Some(eff) => {
                                    ag.set_effort(eff).await;
                                    Ok(())
                                }
                                None => {
                                    Err((-32602, format!("unsupported thought_level `{value}`")))
                                }
                            },
                            "cache_ttl" | "cache_retention" => match CacheTtl::parse(value) {
                                Some(ttl) => {
                                    ag.set_cache_ttl(ttl).await;
                                    Ok(())
                                }
                                None => Err((
                                    -32602,
                                    format!(
                                        "unsupported cache_ttl `{value}` (expected short|hybrid|long)"
                                    ),
                                )),
                            },
                            other => Err((-32602, format!("unknown configId `{other}`"))),
                        };

                        match apply_res {
                            Ok(()) => {
                                let cur_model = ag.current_model().await;
                                let cur_effort = ag.current_effort().await;
                                let cur_ttl = ag.current_cache_ttl().await;
                                let avail = host
                                    .as_ref()
                                    .map(|h| h.available_models())
                                    .unwrap_or_default();
                                let (config_options, _) =
                                    build_config_options(&cur_model, cur_effort, cur_ttl, &avail);
                                Ok(json!({ "configOptions": config_options }))
                            }
                            Err(e) => Err(e),
                        }
                    }
                }
            }
            "session/set_model" => match resolve_session_agent(&sessions, &default_sid, &params) {
                Err(e) => Err(e),
                Ok((_, ag)) => {
                    let model_id = params
                        .get("modelId")
                        .or_else(|| params.get("model"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim();
                    if model_id.is_empty() {
                        Err((-32602, "modelId must not be empty".into()))
                    } else if let Some(ref h) = host {
                        match h.resolve_model(model_id) {
                            Ok(resolved) => match ag
                                .set_model(
                                    model_id.to_string(),
                                    resolved.provider,
                                    resolved.context_window,
                                    resolved.max_output_tokens,
                                )
                                .await
                            {
                                Ok(()) => {
                                    let cur_effort = ag.current_effort().await;
                                    let cur_ttl = ag.current_cache_ttl().await;
                                    let avail = h.available_models();
                                    let (config_options, models_state) =
                                        build_config_options(model_id, cur_effort, cur_ttl, &avail);
                                    Ok(json!({
                                        "configOptions": config_options,
                                        "models": models_state,
                                    }))
                                }
                                Err(e) => Err((-32001, e.to_string())),
                            },
                            Err(e) => Err((-32602, e.to_string())),
                        }
                    } else {
                        Err((-32601, "dynamic model switching requires AcpHost".into()))
                    }
                }
            },
            "session/list" => {
                let sessions_list =
                    list_recent_sessions(&default_session_dir, &default_workspace_root, 30);
                Ok(json!({ "sessions": sessions_list }))
            }
            "session/load" | "session/resume" => {
                let raw_sid = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let target_ag = sessions.get(&default_sid).cloned();
                match target_ag {
                    None => Err((-32004, "no active ACP session".into())),
                    Some(ag) if ag.is_busy() => Err((
                        -32001,
                        "agent is busy running a prompt; call session/cancel first".into(),
                    )),
                    Some(ag) => {
                        match resolve_confined_session_path(&ag.config.session_dir, raw_sid) {
                            Err(e) => Err((-32602, e.to_string())),
                            Ok(path) => match ag.load_session(&path).await {
                                Ok(sid) => {
                                    ag.set_question_broker(Arc::new(AcpQuestionBroker {
                                        session_id: sid.clone(),
                                        stdout: Arc::clone(&stdout),
                                        next_req_id: Arc::clone(&next_client_req_id),
                                        pending: Arc::clone(&pending_replies),
                                    }))
                                    .await;
                                    if client_wants_perms {
                                        ag.set_permission_broker(Arc::new(
                                            AcpPermissionBroker::new(
                                                sid.clone(),
                                                Arc::clone(&stdout),
                                                Arc::clone(&next_client_req_id),
                                                Arc::clone(&pending_replies),
                                            ),
                                        ))
                                        .await;
                                    }
                                    sessions.insert(sid.clone(), Arc::clone(&ag));
                                    default_sid = sid.clone();
                                    post_send_commands_sid = Some(sid.clone());
                                    let history = ag.active_messages().await;
                                    for msg in history {
                                        let txt = msg.text_concat();
                                        let trimmed = txt.trim();
                                        if trimmed.is_empty() || trimmed.starts_with("[carnot:") {
                                            continue;
                                        }
                                        let update_kind = match msg.role {
                                            Role::User => "user_message_chunk",
                                            _ => "agent_message_chunk",
                                        };
                                        let notif = json!({
                                            "jsonrpc": "2.0",
                                            "method": "session/update",
                                            "params": {
                                                "sessionId": sid,
                                                "update": {
                                                    "sessionUpdate": update_kind,
                                                    "content": { "type": "text", "text": txt },
                                                }
                                            }
                                        });
                                        let _ = write_frame(&stdout, &notif).await;
                                    }
                                    let cur_model = ag.current_model().await;
                                    let cur_effort = ag.current_effort().await;
                                    let avail = host
                                        .as_ref()
                                        .map(|h| h.available_models())
                                        .unwrap_or_default();
                                    let cur_ttl = ag.current_cache_ttl().await;
                                    let (config_options, models_state) = build_config_options(
                                        &cur_model, cur_effort, cur_ttl, &avail,
                                    );
                                    Ok(json!({
                                        "sessionId": sid,
                                        "loaded": true,
                                        "configOptions": config_options,
                                        "models": models_state,
                                    }))
                                }
                                Err(e) => Err((-32000, e.to_string())),
                            },
                        }
                    }
                }
            }
            "session/close" => match resolve_session_agent(&sessions, &default_sid, &params) {
                Ok((sid, ag)) => {
                    ag.abort().await;
                    if sessions.len() > 1 {
                        sessions.remove(&sid);
                        if default_sid == sid
                            && let Some(next_key) = sessions.keys().next().cloned()
                        {
                            default_sid = next_key;
                        }
                    }
                    Ok(json!({}))
                }
                Err(e) => Err(e),
            },
            "session/steer" => match resolve_session_agent(&sessions, &default_sid, &params) {
                Ok((_, ag)) => {
                    let blocks = extract_acp_prompt_blocks(&params);
                    ag.steer(prompt_blocks_text(&blocks)).await;
                    Ok(json!({"queued": true}))
                }
                Err(e) => Err(e),
            },
            "session/events" => match resolve_session_agent(&sessions, &default_sid, &params) {
                Ok((_, ag)) => {
                    let after = params.get("afterSeq").and_then(Value::as_u64).unwrap_or(0);
                    let events = ag.event_log().replay_after(after);
                    Ok(json!({"events": events, "latestSeq": ag.latest_event_seq()}))
                }
                Err(e) => Err(e),
            },
            "session/cancel" => match resolve_session_agent(&sessions, &default_sid, &params) {
                Ok((_, ag)) => {
                    ag.abort().await;
                    Ok(json!({"cancelled": true}))
                }
                Err(e) => Err(e),
            },
            _ => Err((-32601, format!("unsupported ACP method: {method}"))),
        };

        if has_id {
            let frame = match result {
                Ok(res) => json!({"jsonrpc": "2.0", "id": id, "result": res}),
                Err((code, message)) => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": code, "message": message}
                }),
            };
            write_frame(&stdout, &frame).await?;
        }

        if let Some(sid) = post_send_commands_sid {
            let _ = write_frame(&stdout, &available_commands_frame(&sid)).await;
        }
    }

    for task in prompt_tasks {
        let _ = task.await;
    }
    Ok(())
}
