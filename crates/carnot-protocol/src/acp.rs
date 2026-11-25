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

use carnot_agent::Agent;
use carnot_ai::Provider;
use carnot_core::{
    AgentEvent, CacheTtl, CarnotError, ContentBlock, EventEnvelope, ReasoningEffort, Result, Role,
    UserQuestion, UserQuestionAnswer,
};
use carnot_ext::{BoxFuture, PermissionBroker, ToolGateDecision, UserQuestionBroker};
use carnot_session::read_session_header;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
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

    /// Creates a fresh `Agent` rooted at `cwd` for `session/new` (with optional ACP `mcpServers`).
    fn create_agent_for_cwd(
        &self,
        cwd: PathBuf,
        model: Option<String>,
        effort: Option<ReasoningEffort>,
        mcp_servers: Vec<Value>,
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
    current_cache_ttl: CacheTtl,
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
        },
        {
            "id": "cache_ttl",
            "name": "Cache retention",
            "description": "Provider-agnostic prompt-cache retention tier (short, hybrid, or long).",
            "category": "cache_ttl",
            "type": "select",
            "currentValue": current_cache_ttl.as_str(),
            "options": [
                { "value": "hybrid", "name": "Hybrid (static long · tail short)" },
                { "value": "short", "name": "Short (5m / default)" },
                { "value": "long", "name": "Long (1h / 24h extended)" }
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

/// Per-session bidirectional ACP `session/request_permission` broker with `allow_always` session cache.
#[derive(Debug)]
struct AcpPermissionBroker {
    session_id: String,
    stdout: Arc<Mutex<tokio::io::Stdout>>,
    next_req_id: Arc<AtomicU64>,
    pending: PendingReplies,
    allowed_always: Mutex<HashSet<String>>,
}

impl AcpPermissionBroker {
    fn new(
        session_id: String,
        stdout: Arc<Mutex<tokio::io::Stdout>>,
        next_req_id: Arc<AtomicU64>,
        pending: PendingReplies,
    ) -> Self {
        Self {
            session_id,
            stdout,
            next_req_id,
            pending,
            allowed_always: Mutex::new(HashSet::new()),
        }
    }
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
        let sid = self.session_id.clone();
        Box::pin(async move {
            let effective_tool = if tool_name == "CallDeferredTool" {
                input
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(&tool_name)
                    .to_string()
            } else {
                tool_name
            };
            // Pure local reads & interactive AskUser do not need redundant permission prompts
            if matches!(
                effective_tool.as_str(),
                "Read" | "Grep" | "ToolSearch" | "AskUser" | "MemorySearch" | "LspQuery"
            ) || self.allowed_always.lock().await.contains(&effective_tool)
            {
                return Ok(ToolGateDecision::Allow);
            }
            let req_id = self.next_req_id.fetch_add(1, Ordering::Relaxed);
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
                        "title": format_tool_title(&effective_tool, &input),
                        "kind": acp_tool_kind(&effective_tool),
                        "rawInput": input,
                    },
                    "options": [
                        { "optionId": "allow_once", "name": "Allow once", "kind": "allow_once" },
                        { "optionId": "allow_always", "name": format!("Always allow {effective_tool} in session"), "kind": "allow_always" },
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
            if selected && opt_id == "allow_always" {
                self.allowed_always.lock().await.insert(effective_tool);
                Ok(ToolGateDecision::Allow)
            } else if selected && opt_id.starts_with("allow") {
                Ok(ToolGateDecision::Allow)
            } else {
                Ok(ToolGateDecision::Deny {
                    reason: format!("denied by ACP client (option={opt_id})"),
                })
            }
        })
    }
}

/// Per-session bidirectional ACP `AskUser` broker: tries `carnot/ask_user` first and gracefully
/// falls back to standard `session/request_permission` option buttons on generic ACP clients!
#[derive(Debug)]
struct AcpQuestionBroker {
    session_id: String,
    stdout: Arc<Mutex<tokio::io::Stdout>>,
    next_req_id: Arc<AtomicU64>,
    pending: PendingReplies,
}

impl UserQuestionBroker for AcpQuestionBroker {
    fn ask<'a>(
        &'a self,
        request_id: &'a str,
        questions: &'a [UserQuestion],
    ) -> BoxFuture<'a, Result<Vec<UserQuestionAnswer>>> {
        let sid = self.session_id.clone();
        let req_tag = request_id.to_string();
        let qs = questions.to_vec();
        Box::pin(async move {
            let rpc_id = self.next_req_id.fetch_add(1, Ordering::Relaxed);
            let (tx, rx) = oneshot::channel();
            self.pending.lock().await.insert(rpc_id, tx);

            let frame = json!({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "method": "carnot/ask_user",
                "params": {
                    "sessionId": sid,
                    "requestId": req_tag,
                    "questions": qs,
                }
            });
            if let Err(e) = write_frame(&self.stdout, &frame).await {
                self.pending.lock().await.remove(&rpc_id);
                return Err(CarnotError::Tool {
                    tool: "AskUser".into(),
                    message: format!("ACP ask_user write failed: {e}"),
                });
            }

            let resp = match tokio::time::timeout(Duration::from_secs(300), rx).await {
                Ok(Ok(v)) => v,
                Ok(Err(_)) => {
                    return Err(CarnotError::Tool {
                        tool: "AskUser".into(),
                        message: "ACP client disconnected before answering question".into(),
                    });
                }
                Err(_) => {
                    self.pending.lock().await.remove(&rpc_id);
                    return Err(CarnotError::Tool {
                        tool: "AskUser".into(),
                        message: "ACP ask_user timed out (300s)".into(),
                    });
                }
            };

            // If client returned `-32601 Method not found`, fall back to standard `session/request_permission`
            // so even generic ACP clients (e.g. Zed) can answer multiple-choice questions natively!
            if let Some(err) = resp.get("error") {
                let code = err.get("code").and_then(Value::as_i64).unwrap_or(0);
                if code == -32601
                    && let Some(first) = qs.first()
                    && !first.options.is_empty()
                {
                    let fb_id = self.next_req_id.fetch_add(1, Ordering::Relaxed);
                    let (ftx, frx) = oneshot::channel();
                    self.pending.lock().await.insert(fb_id, ftx);
                    let opts: Vec<Value> = first
                        .options
                        .iter()
                        .enumerate()
                        .map(|(i, o)| {
                            json!({
                                "optionId": format!("opt_{i}"),
                                "name": o.label,
                                "kind": "allow_once",
                            })
                        })
                        .collect();
                    let fb_frame = json!({
                        "jsonrpc": "2.0",
                        "id": fb_id,
                        "method": "session/request_permission",
                        "params": {
                            "sessionId": sid,
                            "toolCall": {
                                "toolCallId": req_tag,
                                "title": first.question,
                                "kind": "other",
                                "rawInput": { "questions": qs },
                            },
                            "options": opts,
                        }
                    });
                    if write_frame(&self.stdout, &fb_frame).await.is_ok()
                        && let Ok(Ok(fb_resp)) =
                            tokio::time::timeout(Duration::from_secs(300), frx).await
                    {
                        let chosen_idx = fb_resp
                            .pointer("/result/outcome/optionId")
                            .and_then(Value::as_str)
                            .and_then(|s| s.strip_prefix("opt_"))
                            .and_then(|n| n.parse::<usize>().ok());
                        if let Some(idx) = chosen_idx
                            && let Some(opt) = first.options.get(idx)
                        {
                            return Ok(vec![UserQuestionAnswer {
                                id: first.id.clone(),
                                selected: vec![opt.label.clone()],
                                text: None,
                            }]);
                        }
                    }
                }
                let msg = err
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("ACP client rejected ask_user");
                return Err(CarnotError::Tool {
                    tool: "AskUser".into(),
                    message: msg.to_string(),
                });
            }

            let answers_val = resp
                .pointer("/result/answers")
                .cloned()
                .unwrap_or_else(|| json!([]));
            let answers: Vec<UserQuestionAnswer> =
                serde_json::from_value(answers_val).unwrap_or_default();
            Ok(answers)
        })
    }
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
            let out = stdout.clone();
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
                        prev_ag
                            .as_ref()
                            .map(|a| a.config.workspace_root.clone())
                            .unwrap_or_else(|| default_workspace_root.clone())
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

fn map_acp_stop_reason(halted_reason: Option<&str>) -> &'static str {
    let Some(hr) = halted_reason else {
        return "end_turn";
    };
    if hr.contains("MaxTokens") {
        "max_tokens"
    } else if hr.contains("Refusal") {
        "refusal"
    } else if hr.contains("max_turns")
        || hr.contains("cost budget")
        || hr.contains("idle circuit breaker")
    {
        "max_turn_requests"
    } else {
        "end_turn"
    }
}

fn list_recent_sessions(session_dir: &Path, fallback_cwd: &Path, limit: usize) -> Vec<Value> {
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
        let header = read_session_header(&path);
        let cwd = header
            .as_ref()
            .and_then(|h| h.workspace_root.as_ref())
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| fallback_cwd.display().to_string());
        let short_stem: String = stem.chars().take(8).collect();
        let title = match header {
            Some(ref h) => format!("Carnot ({}) {short_stem}", h.model),
            None => format!("Carnot session {stem}"),
        };
        out.push(json!({
            "sessionId": stem,
            "cwd": cwd,
            "title": title,
        }));
    }
    out
}

/// Strictly confines `session/load` to a single filename inside `session_dir` (blocks `..`, `/`
/// traversal, absolute paths, and symlink escapes).
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

/// Extracts multimodal ACP prompt blocks (`text`, `resource`, `image`) into `Vec<ContentBlock>`.
fn extract_acp_prompt_blocks(params: &Value) -> Vec<ContentBlock> {
    if let Some(s) = params.get("prompt").and_then(Value::as_str) {
        return vec![ContentBlock::Text {
            text: s.to_string(),
        }];
    }
    if let Some(arr) = params.get("prompt").and_then(Value::as_array) {
        let mut blocks = Vec::new();
        for b in arr {
            let kind = b.get("type").and_then(Value::as_str).unwrap_or("");
            if kind == "image" {
                let data = b.get("data").and_then(Value::as_str);
                let mime = b
                    .get("mimeType")
                    .or_else(|| b.get("mediaType"))
                    .and_then(Value::as_str)
                    .unwrap_or("image/png");
                if let Some(d) = data
                    && !d.is_empty()
                {
                    blocks.push(ContentBlock::Image {
                        media_type: mime.to_string(),
                        data_base64: d.to_string(),
                    });
                    continue;
                }
                if let Some(uri) = b.get("uri").and_then(Value::as_str)
                    && let Some(rest) = uri.strip_prefix("data:")
                    && let Some((meta, b64)) = rest.split_once(";base64,")
                    && !b64.is_empty()
                {
                    blocks.push(ContentBlock::Image {
                        media_type: if meta.is_empty() {
                            mime.to_string()
                        } else {
                            meta.to_string()
                        },
                        data_base64: b64.to_string(),
                    });
                    continue;
                }
            }
            if let Some(txt) = b.get("text").and_then(Value::as_str) {
                blocks.push(ContentBlock::Text {
                    text: txt.to_string(),
                });
            } else if let Some(res) = b.get("resource") {
                let uri = res.get("uri").and_then(Value::as_str).unwrap_or("resource");
                if let Some(txt) = res.get("text").and_then(Value::as_str) {
                    blocks.push(ContentBlock::Text {
                        text: format!("[embedded:{uri}]\n{txt}"),
                    });
                }
            }
        }
        return blocks;
    }
    if let Some(msg) = params.get("message").and_then(Value::as_str) {
        return vec![ContentBlock::Text {
            text: msg.to_string(),
        }];
    }
    Vec::new()
}

fn prompt_blocks_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn acp_tool_kind(tool: &str) -> &'static str {
    match tool {
        "Read" => "read",
        "Edit" | "Write" => "edit",
        "Grep" | "ToolSearch" => "search",
        "Bash" => "execute",
        _ => "other",
    }
}

fn format_tool_title(tool: &str, input: &Value) -> String {
    let worker_prefix = input
        .get("_carnot_worker")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(|w| format!("[worker:{w}] "))
        .unwrap_or_default();
    let base = match tool {
        "Read" | "Edit" | "Write" => {
            if let Some(fp) = input.get("file_path").and_then(Value::as_str) {
                format!("{tool} {fp}")
            } else {
                tool.to_string()
            }
        }
        "Bash" => {
            if let Some(cmd) = input.get("command").and_then(Value::as_str) {
                let first = cmd.lines().next().unwrap_or(cmd).trim();
                let clipped: String = first.chars().take(72).collect();
                format!("Bash: {clipped}")
            } else if let Some(job) = input.get("job").and_then(Value::as_str) {
                format!("Bash ({job})")
            } else {
                tool.to_string()
            }
        }
        "Grep" => {
            if let Some(pat) = input.get("pattern").and_then(Value::as_str) {
                let clipped: String = pat.chars().take(48).collect();
                format!("Grep `{clipped}`")
            } else {
                tool.to_string()
            }
        }
        _ => tool.to_string(),
    };
    format!("{worker_prefix}{base}")
}

fn extract_tool_locations(input: &Value) -> Option<Vec<Value>> {
    let path = input
        .get("file_path")
        .or_else(|| input.get("path"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?;
    let mut loc = json!({ "path": path });
    if let Some(line) = input
        .get("start_line")
        .or_else(|| input.get("offset"))
        .and_then(Value::as_u64)
        && let Some(obj) = loc.as_object_mut()
    {
        obj.insert("line".into(), json!(line));
    }
    Some(vec![loc])
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
            input,
        } => {
            let mut u = json!({
                "sessionUpdate": "tool_call",
                "toolCallId": call_id,
                "title": format_tool_title(tool, input),
                "kind": acp_tool_kind(tool),
                "status": "in_progress",
                "rawInput": input,
                "_meta": { "carnot": { "tool": tool, "eager": eager } },
            });
            if let Some(locs) = extract_tool_locations(input)
                && let Some(obj) = u.as_object_mut()
            {
                obj.insert("locations".into(), json!(locs));
            }
            u
        }
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
            output_preview,
        } => {
            let mut u = json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": call_id,
                "title": tool,
                "status": if *is_error { "failed" } else { "completed" },
                "_meta": { "carnot": { "durationMs": duration_ms } },
            });
            if !output_preview.is_empty()
                && let Some(obj) = u.as_object_mut()
            {
                obj.insert(
                    "content".into(),
                    json!([{ "type": "content", "content": { "type": "text", "text": output_preview } }]),
                );
                obj.insert("rawOutput".into(), json!(output_preview));
            }
            u
        }
        AgentEvent::ModuleAdvisory { signal, message } => json!({
            "sessionUpdate": "agent_thought_chunk",
            "content": { "type": "text", "text": format!("[carnot:{signal}] {message}") },
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
    use carnot_ai::faux::{FauxProvider, ScriptedTurn};
    use carnot_cas::CasStore;
    use carnot_core::CarnotConfig;
    use carnot_tools::ToolRegistry;

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
        let (cfg_opts, models_state) = build_config_options(
            "claude-sonnet-5-5",
            ReasoningEffort::High,
            CacheTtl::Hybrid,
            &models,
        );
        let arr = cfg_opts.as_array().expect("array");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0]["category"], "model");
        assert_eq!(arr[2]["id"], "cache_ttl");
        assert_eq!(arr[2]["currentValue"], "hybrid");
        assert_eq!(arr[0]["currentValue"], "claude-sonnet-5-5");
        assert_eq!(arr[1]["category"], "thought_level");
        assert_eq!(arr[1]["currentValue"], "high");
        assert_eq!(models_state["currentModelId"], "claude-sonnet-5-5");
        assert_eq!(
            models_state["availableModels"].as_array().expect("m").len(),
            2
        );
    }

    #[test]
    fn extracts_multimodal_images_text_and_embedded_resources() {
        let params = json!({
            "prompt": [
                { "type": "text", "text": "Inspect this screenshot" },
                { "type": "image", "mimeType": "image/png", "data": "iVBORw0KGgo=" },
                { "type": "resource", "resource": { "uri": "file:///a.rs", "text": "fn main() {}" } }
            ]
        });
        let blocks = extract_acp_prompt_blocks(&params);
        assert_eq!(blocks.len(), 3);
        assert!(matches!(blocks[0], ContentBlock::Text { .. }));
        assert!(matches!(
            &blocks[1],
            ContentBlock::Image { media_type, data_base64 }
                if media_type == "image/png" && data_base64 == "iVBORw0KGgo="
        ));
        assert!(matches!(blocks[2], ContentBlock::Text { .. }));
    }

    #[test]
    fn emits_rich_acp_tool_call_with_kind_raw_input_locations_and_output() {
        let start_env = EventEnvelope {
            seq: 10,
            event: AgentEvent::ToolStart {
                call_id: "call_1".into(),
                tool: "Edit".into(),
                eager: false,
                input: json!({ "file_path": "src/lib.rs", "start_line": 42, "end_line": 45 }),
            },
        };
        let frame = acp_update_frame("sess-1", &start_env).expect("tool_call frame");
        let upd = &frame["params"]["update"];
        assert_eq!(upd["sessionUpdate"], "tool_call");
        assert_eq!(upd["kind"], "edit");
        assert_eq!(upd["title"], "Edit src/lib.rs");
        assert_eq!(upd["rawInput"]["file_path"], "src/lib.rs");
        assert_eq!(upd["locations"][0]["path"], "src/lib.rs");
        assert_eq!(upd["locations"][0]["line"], 42);

        let end_env = EventEnvelope {
            seq: 11,
            event: AgentEvent::ToolEnd {
                call_id: "call_1".into(),
                tool: "Edit".into(),
                is_error: false,
                duration_ms: 14,
                output_preview: "Edited src/lib.rs (1 replacement(s)).".into(),
            },
        };
        let end_frame = acp_update_frame("sess-1", &end_env).expect("tool_call_update");
        let eupd = &end_frame["params"]["update"];
        assert_eq!(eupd["sessionUpdate"], "tool_call_update");
        assert_eq!(eupd["status"], "completed");
        assert_eq!(eupd["rawOutput"], "Edited src/lib.rs (1 replacement(s)).");
        assert_eq!(
            eupd["content"][0]["content"]["text"],
            "Edited src/lib.rs (1 replacement(s))."
        );
    }

    #[tokio::test]
    async fn multi_session_routing_and_session_list_real_cwd() {
        let base = std::env::temp_dir().join(format!("carnot-acp-multi-{}", uuid::Uuid::new_v4()));
        let ws_a = base.join("ws_a");
        let ws_b = base.join("ws_b");
        let sdir = base.join("sessions");
        std::fs::create_dir_all(&ws_a).expect("ws_a");
        std::fs::create_dir_all(&ws_b).expect("ws_b");

        let mk_agent = |ws: PathBuf| {
            let mut cfg = CarnotConfig::new(ws.clone());
            cfg.session_dir = sdir.clone();
            cfg.modules.sandbox = false;
            let cas = Arc::new(CasStore::open(&ws).expect("cas"));
            let tools = ToolRegistry::with_canonical_lid_tools(&cfg, cas);
            let faux = Arc::new(FauxProvider::new(vec![ScriptedTurn::text("ok")]));
            Arc::new(Agent::new(cfg, faux, tools, Vec::new(), true).expect("agent"))
        };

        let ag_a = mk_agent(ws_a.clone());
        let ag_b = mk_agent(ws_b.clone());
        let sid_a = ag_a.session_id().await;
        let sid_b = ag_b.session_id().await;

        let mut map = HashMap::new();
        map.insert(sid_a.clone(), ag_a);
        map.insert(sid_b.clone(), ag_b);

        let (resolved_b, _) =
            resolve_session_agent(&map, &sid_a, &json!({"sessionId": sid_b})).expect("route b");
        assert_eq!(resolved_b, sid_b);
        assert!(resolve_session_agent(&map, &sid_a, &json!({"sessionId": "missing"})).is_err());

        // Verify `list_recent_sessions` reports the real persisted workspace_root (`ws_a` / `ws_b`),
        // NEVER `~/.carnot/sessions`!
        let listed = list_recent_sessions(&sdir, &ws_a, 10);
        assert_eq!(listed.len(), 2);
        let cwds: Vec<&str> = listed
            .iter()
            .filter_map(|v| v.get("cwd").and_then(Value::as_str))
            .collect();
        assert!(cwds.contains(&ws_a.display().to_string().as_str()));
        assert!(cwds.contains(&ws_b.display().to_string().as_str()));

        let _ = std::fs::remove_dir_all(&base);
    }
}
