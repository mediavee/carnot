//! Agent Client Protocol (ACP) JSON-RPC 2.0 transport over stdio — Zed / T3 Code compliant:
//!   - Full ACP v1/v2 surface: `initialize`, `authenticate`, `session/new` (per-session `cwd` +
//!     `configOptions` for model & `thought_level` + `available_commands_update`),
//!     `session/set_config_option`, `session/set_model`, `session/prompt` (incl. `/compact`),
//!     `session/cancel`, `session/steer`, `session/list`, `session/load`/`session/resume`,
//!     `session/close`, `session/events`
//!   - Emits canonical ACP `sessionUpdate` payloads (`agent_message_chunk`, `agent_thought_chunk`,
//!     `tool_call`, `tool_call_update`, `usage_update`, `available_commands_update`)
//!   - Bidirectional `AcpPermissionBroker` (`session/request_permission` client callback routed via
//!     pending JSON-RPC response map — fixes Opus #3!)
//!   - Concurrent `session/prompt` so `session/cancel`, `session/steer`, and permission replies
//!     execute immediately mid-run (N4 + Opus #3)
//!   - Real `session/load` backed by `carnot-session` JSONL replay with non-blocking busy check and
//!     strict `session_dir` confinement (Opus #7)

use carnot_agent::Agent;
use carnot_ai::Provider;
use carnot_core::{AgentEvent, CarnotError, EventEnvelope, ReasoningEffort, Result, Role};
use carnot_ext::{BoxFuture, PermissionBroker, ToolGateDecision};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
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

    /// Creates a fresh `Agent` rooted at `cwd` for `session/new`.
    fn create_agent_for_cwd(
        &self,
        cwd: PathBuf,
        model: Option<String>,
        effort: Option<ReasoningEffort>,
    ) -> BoxFuture<'_, Result<Arc<Agent>>>;
}

async fn write_frame(stdout: &Arc<Mutex<tokio::io::Stdout>>, frame: &Value) -> std::io::Result<()> {
    let mut s = serde_json::to_string(frame).unwrap_or_else(|_| "{}".into());
    s.push('\n');
    let mut w = stdout.lock().await;
    w.write_all(s.as_bytes()).await?;
    w.flush().await
}

fn effort_to_acp_str(e: ReasoningEffort) -> &'static str {
    match e {
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
}

fn parse_acp_effort(s: &str) -> Option<ReasoningEffort> {
    match s.trim().to_ascii_lowercase().as_str() {
        "low" => Some(ReasoningEffort::Low),
        "medium" | "med" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" | "extra_high" | "extra-high" => Some(ReasoningEffort::Xhigh),
        "max" | "ultrathink" => Some(ReasoningEffort::Max),
        _ => None,
    }
}

fn build_config_options(
    current_model: &str,
    current_effort: ReasoningEffort,
    available_models: &[AcpModelInfo],
) -> (Value, Value) {
    let mut model_choices: Vec<Value> = available_models
        .iter()
        .map(|m| {
            let mut obj = json!({
                "value": m.id,
                "name": m.name,
            });
            if let Some(ref desc) = m.description
                && !desc.is_empty()
                && let Some(map) = obj.as_object_mut()
            {
                map.insert("description".into(), json!(desc));
            }
            obj
        })
        .collect();
    if !available_models.iter().any(|m| m.id == current_model) {
        model_choices.insert(
            0,
            json!({
                "value": current_model,
                "name": current_model,
            }),
        );
    }

    let available_models_v1: Vec<Value> = model_choices
        .iter()
        .map(|c| {
            let mut obj = json!({
                "modelId": c["value"],
                "name": c["name"],
            });
            if let Some(desc) = c.get("description")
                && let Some(map) = obj.as_object_mut()
            {
                map.insert("description".into(), desc.clone());
            }
            obj
        })
        .collect();

    let config_options = json!([
        {
            "id": "model",
            "name": "Model",
            "description": "Active LLM model routed through Carnot's provider catalog.",
            "category": "model",
            "type": "select",
            "currentValue": current_model,
            "options": model_choices,
        },
        {
            "id": "thought_level",
            "name": "Reasoning effort",
            "description": "Thinking / reasoning effort level.",
            "category": "thought_level",
            "type": "select",
            "currentValue": effort_to_acp_str(current_effort),
            "options": [
                { "value": "low", "name": "Low" },
                { "value": "medium", "name": "Medium" },
                { "value": "high", "name": "High" },
                { "value": "xhigh", "name": "Extra High" },
                { "value": "max", "name": "Max" }
            ],
        }
    ]);

    let models_state = json!({
        "currentModelId": current_model,
        "availableModels": available_models_v1,
    });

    (config_options, models_state)
}

fn available_commands_frame(session_id: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": {
                "sessionUpdate": "available_commands_update",
                "availableCommands": [
                    {
                        "name": "compact",
                        "description": "Run Carnot cache-warm semantic compaction (preserves verbatim goal & file ledger).",
                        "input": { "hint": "[optional focus instructions]" }
                    }
                ]
            }
        }
    })
}

/// Bidirectional ACP `session/request_permission` broker (activated when the ACP client declares
/// `clientCapabilities.permissionRequests = true` during `initialize`).
#[derive(Debug)]
struct AcpPermissionBroker {
    session_id: Arc<Mutex<String>>,
    stdout: Arc<Mutex<tokio::io::Stdout>>,
    next_req_id: Arc<AtomicU64>,
    pending: PendingReplies,
}

impl PermissionBroker for AcpPermissionBroker {
    fn request_permission(
        &self,
        call_id: &str,
        tool_name: &str,
        input: &Value,
    ) -> BoxFuture<'_, Result<ToolGateDecision>> {
        let call_id = call_id.to_string();
        let tool_name = tool_name.to_string();
        let input = input.clone();
        Box::pin(async move {
            // Pure local reads do not need interactive user prompts
            if matches!(
                tool_name.as_str(),
                "Read" | "Grep" | "ToolSearch" | "LspQuery" | "WebSearch"
            ) {
                return Ok(ToolGateDecision::Allow);
            }
            let req_id = self.next_req_id.fetch_add(1, Ordering::Relaxed);
            let sid = self.session_id.lock().await.clone();
            let (tx, rx) = oneshot::channel();
            self.pending.lock().await.insert(req_id, tx);

            let frame = json!({
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "session/request_permission",
                "params": {
                    "sessionId": sid,
                    "toolCall": {
                        "toolCallId": call_id,
                        "title": tool_name,
                        "rawInput": input,
                    },
                    "options": [
                        { "optionId": "allow_once", "name": "Allow once", "kind": "allow_once" },
                        { "optionId": "reject_once", "name": "Reject", "kind": "reject_once" }
                    ]
                }
            });
            if let Err(e) = write_frame(&self.stdout, &frame).await {
                self.pending.lock().await.remove(&req_id);
                return Ok(ToolGateDecision::Deny {
                    reason: format!("ACP permission request write failed: {e}"),
                });
            }

            let resp = match tokio::time::timeout(Duration::from_secs(120), rx).await {
                Ok(Ok(v)) => v,
                Ok(Err(_)) => {
                    return Ok(ToolGateDecision::Deny {
                        reason: "ACP client disconnected before answering permission request"
                            .into(),
                    });
                }
                Err(_) => {
                    self.pending.lock().await.remove(&req_id);
                    return Ok(ToolGateDecision::Deny {
                        reason: "ACP permission request timed out (120s)".into(),
                    });
                }
            };

            if resp.get("error").is_some() {
                return Ok(ToolGateDecision::Deny {
                    reason: "ACP client rejected permission request".into(),
                });
            }
            let outcome = resp.get("result").and_then(|r| r.get("outcome"));
            let selected = outcome
                .and_then(|o| o.get("outcome"))
                .and_then(Value::as_str)
                == Some("selected");
            let opt_id = outcome
                .and_then(|o| o.get("optionId"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if selected && opt_id.starts_with("allow") {
                Ok(ToolGateDecision::Allow)
            } else {
                Ok(ToolGateDecision::Deny {
                    reason: format!("denied by ACP client (option={opt_id})"),
                })
            }
        })
    }
}

/// Starts a stdio ACP server bound to a single pre-built `Agent` (backwards-compatible entry point).
pub async fn serve_acp_stdio(agent: Arc<Agent>) -> Result<()> {
    serve_acp_stdio_inner(agent, None).await
}

/// Starts a full-featured stdio ACP server with dynamic per-session `cwd` creation, live model
/// picker advertising, and live multi-provider model switching (`T3 Code` / `Zed` compliant).
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
    let mut agent = initial_agent;
    let active_sid = Arc::new(Mutex::new(agent.session_id().await));
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

        // Route incoming JSON-RPC responses (e.g. client reply to `session/request_permission`) — Opus #3!
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
            if agent.is_busy() {
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
            let prompt_text = extract_acp_prompt(&params);
            let ag = agent.clone();
            let out = stdout.clone();
            prompt_tasks.push(tokio::spawn(async move {
                let sid = ag.session_id().await;

                // Handle `/compact [focus]` slash command natively inside ACP!
                let trimmed = prompt_text.trim();
                if trimmed == "/compact" || trimmed.starts_with("/compact ") {
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
                let out_writer = out.clone();
                let ag_obs = ag.clone();
                let sid_obs = sid.clone();
                let pump = tokio::spawn(async move {
                    loop {
                        let env = ev_stream.next().await;
                        let is_done = matches!(
                            env.event,
                            AgentEvent::RunSettled { .. } | AgentEvent::Error { .. }
                        );
                        if matches!(env.event, AgentEvent::UsageRecorded { .. }) {
                            let used = ag_obs.current_context_tokens().await;
                            let size = ag_obs.current_context_window().await;
                            let met = ag_obs.metrics().await;
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
                                        }
                                    }
                                }
                            });
                            let _ = write_frame(&out_writer, &uframe).await;
                        } else if let Some(notif) = acp_update_frame(&sid_obs, &env) {
                            let _ = write_frame(&out_writer, &notif).await;
                        }
                        if is_done {
                            break;
                        }
                    }
                });
                let outcome = ag.prompt(prompt_text).await;
                let _ = pump.await;
                let frame = match outcome {
                    Ok(o) => {
                        let stop = if let Some(ref hr) = o.halted_reason {
                            if hr.contains("MaxTokens") {
                                "max_tokens"
                            } else if hr.contains("Refusal") {
                                "refusal"
                            } else {
                                "end_turn"
                            }
                        } else {
                            "end_turn"
                        };
                        json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "stopReason": stop,
                                "finalText": o.final_text,
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
                if client_wants_perms {
                    agent
                        .set_permission_broker(Arc::new(AcpPermissionBroker {
                            session_id: Arc::clone(&active_sid),
                            stdout: Arc::clone(&stdout),
                            next_req_id: Arc::clone(&next_client_req_id),
                            pending: Arc::clone(&pending_replies),
                        }))
                        .await;
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

                if let Some(ref h) = host {
                    let target_cwd = req_cwd.unwrap_or_else(|| agent.config.workspace_root.clone());
                    let cur_model = agent.current_model().await;
                    let cur_effort = agent.current_effort().await;
                    match h
                        .create_agent_for_cwd(target_cwd, Some(cur_model), Some(cur_effort))
                        .await
                    {
                        Ok(new_agent) => {
                            if client_wants_perms {
                                new_agent
                                    .set_permission_broker(Arc::new(AcpPermissionBroker {
                                        session_id: Arc::clone(&active_sid),
                                        stdout: Arc::clone(&stdout),
                                        next_req_id: Arc::clone(&next_client_req_id),
                                        pending: Arc::clone(&pending_replies),
                                    }))
                                    .await;
                            }
                            agent = new_agent;
                        }
                        Err(e) => {
                            let err = (-32000, format!("failed to start session: {e}"));
                            if has_id {
                                let frame = json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "error": {"code": err.0, "message": err.1}
                                });
                                write_frame(&stdout, &frame).await?;
                            }
                            continue;
                        }
                    }
                }

                let sid = agent.session_id().await;
                *active_sid.lock().await = sid.clone();
                post_send_commands_sid = Some(sid.clone());
                let cur_model = agent.current_model().await;
                let cur_effort = agent.current_effort().await;
                let avail = host
                    .as_ref()
                    .map(|h| h.available_models())
                    .unwrap_or_default();
                let (config_options, models_state) =
                    build_config_options(&cur_model, cur_effort, &avail);
                let prefix_hash = agent.prefix_hash().await;
                Ok(json!({
                    "sessionId": sid,
                    "configOptions": config_options,
                    "models": models_state,
                    "_meta": {
                        "carnot": {
                            "model": cur_model,
                            "prefixHash": prefix_hash,
                            "latestSeq": agent.latest_event_seq(),
                            "modules": agent.config.modules,
                        }
                    }
                }))
            }
            "session/set_config_option" => {
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
                                Ok(resolved) => agent
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
                            agent.set_effort(eff).await;
                            Ok(())
                        }
                        None => Err((-32602, format!("unsupported thought_level `{value}`"))),
                    },
                    other => Err((-32602, format!("unknown configId `{other}`"))),
                };

                match apply_res {
                    Ok(()) => {
                        let cur_model = agent.current_model().await;
                        let cur_effort = agent.current_effort().await;
                        let avail = host
                            .as_ref()
                            .map(|h| h.available_models())
                            .unwrap_or_default();
                        let (config_options, _) =
                            build_config_options(&cur_model, cur_effort, &avail);
                        Ok(json!({ "configOptions": config_options }))
                    }
                    Err(e) => Err(e),
                }
            }
            "session/set_model" => {
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
                        Ok(resolved) => {
                            match agent
                                .set_model(
                                    model_id.to_string(),
                                    resolved.provider,
                                    resolved.context_window,
                                    resolved.max_output_tokens,
                                )
                                .await
                            {
                                Ok(()) => {
                                    let cur_effort = agent.current_effort().await;
                                    let avail = h.available_models();
                                    let (config_options, models_state) =
                                        build_config_options(model_id, cur_effort, &avail);
                                    Ok(json!({
                                        "configOptions": config_options,
                                        "models": models_state,
                                    }))
                                }
                                Err(e) => Err((-32001, e.to_string())),
                            }
                        }
                        Err(e) => Err((-32602, e.to_string())),
                    }
                } else {
                    Err((-32601, "dynamic model switching requires AcpHost".into()))
                }
            }
            "session/list" => {
                let sessions = list_recent_sessions(&agent.config.session_dir, 30);
                Ok(json!({ "sessions": sessions }))
            }
            "session/load" | "session/resume" => {
                if agent.is_busy() {
                    Err((
                        -32001,
                        "agent is busy running a prompt; call session/cancel first".into(),
                    ))
                } else {
                    let raw_sid = params
                        .get("sessionId")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    match resolve_confined_session_path(&agent.config.session_dir, raw_sid) {
                        Err(e) => Err((-32602, e.to_string())),
                        Ok(path) => match agent.load_session(&path).await {
                            Ok(sid) => {
                                *active_sid.lock().await = sid.clone();
                                post_send_commands_sid = Some(sid.clone());
                                let history = agent.active_messages().await;
                                for msg in history {
                                    let txt = msg.text_concat();
                                    if txt.is_empty() {
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
                                let cur_model = agent.current_model().await;
                                let cur_effort = agent.current_effort().await;
                                let avail = host
                                    .as_ref()
                                    .map(|h| h.available_models())
                                    .unwrap_or_default();
                                let (config_options, models_state) =
                                    build_config_options(&cur_model, cur_effort, &avail);
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
            "session/close" => {
                agent.abort().await;
                Ok(json!({}))
            }
            "session/steer" => {
                let text = extract_acp_prompt(&params);
                agent.steer(text).await;
                Ok(json!({"queued": true}))
            }
            "session/events" => {
                let after = params.get("afterSeq").and_then(Value::as_u64).unwrap_or(0);
                let events = agent.event_log().replay_after(after);
                Ok(json!({"events": events, "latestSeq": agent.latest_event_seq()}))
            }
            "session/cancel" => {
                agent.abort().await;
                Ok(json!({"cancelled": true}))
            }
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

fn list_recent_sessions(session_dir: &Path, limit: usize) -> Vec<Value> {
    let Ok(rd) = std::fs::read_dir(session_dir) else {
        return Vec::new();
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .map(|p| {
            let mtime = std::fs::metadata(&p)
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::UNIX_EPOCH);
            (mtime, p)
        })
        .collect();
    files.sort_by_key(|a| std::cmp::Reverse(a.0));
    let mut out = Vec::new();
    for (_, path) in files.into_iter().take(limit) {
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        out.push(json!({
            "sessionId": stem,
            "cwd": session_dir.display().to_string(),
            "title": format!("Carnot session {stem}"),
        }));
    }
    out
}

/// Strictly confines `session/load` to a single filename inside `session_dir` (blocks `..`, `/`
/// traversal, absolute paths, and symlink escapes — Opus #7!).
fn resolve_confined_session_path(session_dir: &Path, raw: &str) -> Result<PathBuf> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.contains('\0') {
        return Err(CarnotError::Session("sessionId must be non-empty".into()));
    }
    let p = Path::new(trimmed);
    let mut comps = p.components();
    let Some(Component::Normal(file_os)) = comps.next() else {
        return Err(CarnotError::Session(format!(
            "invalid sessionId `{trimmed}`: path traversal is forbidden"
        )));
    };
    if comps.next().is_some() {
        return Err(CarnotError::Session(format!(
            "invalid sessionId `{trimmed}`: subdirectories are forbidden"
        )));
    }
    let name = file_os.to_string_lossy();
    let filename = if name.ends_with(".jsonl") {
        name.to_string()
    } else {
        format!("{name}.jsonl")
    };
    let candidate = session_dir.join(&filename);
    let canon_dir =
        std::fs::canonicalize(session_dir).unwrap_or_else(|_| session_dir.to_path_buf());
    let canon_file = std::fs::canonicalize(&candidate).map_err(|_| {
        CarnotError::Session(format!("session not found in session_dir: {filename}"))
    })?;
    if !canon_file.starts_with(&canon_dir) {
        return Err(CarnotError::Session(
            "session symlink escapes session_dir".into(),
        ));
    }
    Ok(canon_file)
}

fn extract_acp_prompt(params: &Value) -> String {
    if let Some(s) = params.get("prompt").and_then(Value::as_str) {
        return s.to_string();
    }
    if let Some(arr) = params.get("prompt").and_then(Value::as_array) {
        let mut parts = Vec::new();
        for b in arr {
            if let Some(txt) = b.get("text").and_then(Value::as_str) {
                parts.push(txt.to_string());
            } else if let Some(res) = b.get("resource") {
                let uri = res.get("uri").and_then(Value::as_str).unwrap_or("resource");
                if let Some(txt) = res.get("text").and_then(Value::as_str) {
                    parts.push(format!("[embedded:{uri}]\n{txt}"));
                }
            }
        }
        return parts.join("\n");
    }
    params
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

/// Maps `AgentEvent` to standard ACP `sessionUpdate` union (Zed / T3 Code) + `_meta.carnot`.
/// Returns `None` for internal lifecycle/telemetry events to avoid empty-chunk UI spam.
fn acp_update_frame(session_id: &str, env: &EventEnvelope) -> Option<Value> {
    let update = match &env.event {
        AgentEvent::TextDelta { delta } if !delta.is_empty() => json!({
            "sessionUpdate": "agent_message_chunk",
            "content": { "type": "text", "text": delta },
        }),
        AgentEvent::ThinkingDelta { delta } if !delta.is_empty() => json!({
            "sessionUpdate": "agent_thought_chunk",
            "content": { "type": "text", "text": delta },
        }),
        AgentEvent::ToolStart {
            call_id,
            tool,
            eager,
        } => json!({
            "sessionUpdate": "tool_call",
            "toolCallId": call_id,
            "title": tool,
            "status": "in_progress",
            "_meta": { "carnot": { "eager": eager } },
        }),
        AgentEvent::ToolUpdate { call_id, message } => json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": call_id,
            "status": "in_progress",
            "content": [{ "type": "content", "content": { "type": "text", "text": message } }],
        }),
        AgentEvent::ToolEnd {
            call_id,
            tool,
            is_error,
            duration_ms,
        } => json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": call_id,
            "title": tool,
            "status": if *is_error { "failed" } else { "completed" },
            "_meta": { "carnot": { "durationMs": duration_ms } },
        }),
        AgentEvent::DaemonAdvisory { signal, message } => json!({
            "sessionUpdate": "agent_thought_chunk",
            "content": { "type": "text", "text": format!("[carnot:{signal}] {message}") },
        }),
        AgentEvent::RewindApplied {
            shed_tokens,
            expected_net_usd,
            ..
        } => json!({
            "sessionUpdate": "agent_thought_chunk",
            "content": { "type": "text", "text": format!("[carnot:rewind] shed ~{shed_tokens} tokens (net save ~${expected_net_usd:.4})") },
        }),
        _ => return None,
    };

    Some(json!({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "seq": env.seq,
            "update": update,
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_confined_session_path_blocks_traversal() {
        let dir = std::env::temp_dir().join(format!("carnot-acp-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let valid = dir.join("sess-123.jsonl");
        std::fs::write(&valid, "{}\n").expect("write");

        assert!(resolve_confined_session_path(&dir, "sess-123").is_ok());
        assert!(resolve_confined_session_path(&dir, "../etc/passwd").is_err());
        assert!(resolve_confined_session_path(&dir, "/etc/passwd").is_err());
        assert!(resolve_confined_session_path(&dir, "sub/sess-123").is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_options_advertises_model_and_thought_level_in_t3_format() {
        let models = vec![
            AcpModelInfo {
                id: "claude-sonnet-5-5".into(),
                name: "Claude Sonnet 5.5".into(),
                description: Some("Anthropic".into()),
            },
            AcpModelInfo {
                id: "gpt-6.1-sol".into(),
                name: "GPT 6.1 Sol".into(),
                description: Some("OpenAI Responses".into()),
            },
        ];
        let (cfg_opts, models_state) =
            build_config_options("claude-sonnet-5-5", ReasoningEffort::High, &models);
        let arr = cfg_opts.as_array().expect("array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["category"], "model");
        assert_eq!(arr[0]["currentValue"], "claude-sonnet-5-5");
        assert_eq!(arr[1]["category"], "thought_level");
        assert_eq!(arr[1]["currentValue"], "high");
        assert_eq!(models_state["currentModelId"], "claude-sonnet-5-5");
        assert_eq!(
            models_state["availableModels"].as_array().expect("m").len(),
            2
        );
    }
}
