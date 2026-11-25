//! Universal bidirectional Host JSON-RPC 2.0 transport over stdio.
//!
//! Features:
//! - Concurrent `prompt` / `continue` / `compact` execution so stdin stays responsive to
//!   `steer`, `follow_up`, `abort`, `get_state`, `ping`, and bidirectional host callbacks
//! - Optional `--host-extension` handshake (`host/register`) allowing any host to provide
//!   external tools, frozen static system prompt sections, and lifecycle hooks over the same stdio
//! - Live lossless `EventLog` streaming (`method: "event"`)
//! - Hot `set_model`, cache-safe `set_effort(scoped)`, `get_session_stats`, `get_entries`,
//!   `get_available_models`, `get_commands`, `append_entry`, `undo`, and interactive approvals

use crate::host_ext::{HostRegistration, HostRpcBridge};
use crate::interaction::ChannelInteractionBroker;
use carnot_agent::Agent;
use carnot_ai::Provider;
use carnot_core::{
    CacheTtl, CarnotError, ContentBlock, ReasoningEffort, Result, UserQuestionAnswer,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// Model entry exposed via `get_available_models`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcModelInfo {
    pub provider: String,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub effort_preserves_cache: bool,
}

/// Resolved target returned by `RpcHost::resolve_model`.
#[derive(Debug)]
pub struct ResolvedRpcModel {
    pub provider_id: String,
    pub model_id: String,
    pub provider: Arc<dyn Provider>,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u32>,
}

/// Slash command or skill metadata exposed via `get_commands`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcCommandInfo {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Optional host environment hooks supplied by the CLI / embedder for `set_model`,
/// `get_available_models`, and `get_commands`.
pub trait RpcHost: Send + Sync + std::fmt::Debug {
    fn list_models(&self) -> Vec<RpcModelInfo> {
        Vec::new()
    }

    fn resolve_model(
        &self,
        _model_spec: &str,
        _provider_override: Option<&str>,
    ) -> Result<ResolvedRpcModel> {
        Err(CarnotError::Session(
            "live model switching is not configured on this RPC host".into(),
        ))
    }

    fn list_commands(&self) -> Vec<RpcCommandInfo> {
        Vec::new()
    }
}

/// Handshake result returned by `read_host_registration` before building the `Agent`.
#[derive(Debug)]
pub struct PendingHostSession {
    pub request_id: Value,
    pub registration: HostRegistration,
    pub lines: Lines<BufReader<tokio::io::Stdin>>,
    pub stdout: Arc<Mutex<tokio::io::Stdout>>,
}

/// Waits for the initial `host/register` JSON-RPC request on stdin when `--host-extension` is set,
/// so all host tools and static prompt sections are frozen into the `Agent` before Turn 1.
pub async fn read_host_registration() -> Result<PendingHostSession> {
    let stdin = BufReader::new(tokio::io::stdin());
    let stdout = Arc::new(Mutex::new(tokio::io::stdout()));
    let mut lines = stdin.lines();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = serde_json::from_str(&line).map_err(|e| CarnotError::Extension {
            extension: "host-rpc".into(),
            message: format!("invalid JSON during host/register handshake: {e}"),
        })?;
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        if method != "host/register" {
            let err = json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32002,
                    "message": format!("expected `host/register` handshake first, got `{method}`")
                }
            });
            write_frame(&stdout, &err).await?;
            continue;
        }
        let params = req.get("params").cloned().unwrap_or_else(|| json!({}));
        let registration: HostRegistration =
            serde_json::from_value(params).map_err(|e| CarnotError::Extension {
                extension: "host-rpc".into(),
                message: format!("invalid `host/register` payload: {e}"),
            })?;
        return Ok(PendingHostSession {
            request_id: id,
            registration,
            lines,
            stdout,
        });
    }

    Err(CarnotError::Extension {
        extension: "host-rpc".into(),
        message: "stdin closed before `host/register` handshake".into(),
    })
}

async fn write_frame(stdout: &Arc<Mutex<tokio::io::Stdout>>, frame: &Value) -> std::io::Result<()> {
    let mut s = serde_json::to_string(frame).unwrap_or_else(|_| "{}".into());
    s.push('\n');
    let mut w = stdout.lock().await;
    w.write_all(s.as_bytes()).await?;
    w.flush().await
}

fn parse_effort(s: &str) -> Option<ReasoningEffort> {
    match s.trim().to_ascii_lowercase().as_str() {
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::Xhigh),
        "max" => Some(ReasoningEffort::Max),
        _ => None,
    }
}

fn effort_label(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
}

fn build_prompt_blocks(params: &Value) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();
    let text = params
        .get("message")
        .or_else(|| params.get("prompt"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if !text.is_empty() {
        blocks.push(ContentBlock::Text {
            text: text.to_string(),
        });
    }
    if let Some(images) = params.get("images").and_then(Value::as_array) {
        for img in images {
            let media_type = img
                .get("media_type")
                .or_else(|| img.get("mimeType"))
                .and_then(Value::as_str)
                .unwrap_or("image/png");
            if let Some(data) = img
                .get("data_base64")
                .or_else(|| img.get("data"))
                .and_then(Value::as_str)
                && !data.is_empty()
            {
                blocks.push(ContentBlock::Image {
                    media_type: media_type.to_string(),
                    data_base64: data.to_string(),
                });
            }
        }
    }
    if blocks.is_empty() {
        blocks.push(ContentBlock::Text {
            text: String::new(),
        });
    }
    blocks
}

async fn state_payload(agent: &Agent) -> Value {
    let sid = agent.session_id().await;
    let sfile = agent.session_file().await.map(|p| p.display().to_string());
    let provider = agent.current_provider_id().await;
    let model = agent.current_model().await;
    let effort = effort_label(agent.current_effort().await);
    let effort_preserves_cache = agent.effort_preserves_cache().await;
    let ph = agent.prefix_hash().await;
    let met = agent.metrics().await;
    let ttl = agent.current_cache_ttl().await;
    let pending_queue = agent.pending_queue_len().await;
    let (active_tools, deferred_tools) = agent.tool_names();
    json!({
        "session_id": sid,
        "session_file": sfile,
        "provider": provider,
        "model": model,
        "effort": effort,
        "effort_preserves_cache": effort_preserves_cache,
        "cache_ttl": ttl.as_str(),
        "prefix_retention_rate": met.prefix_retention_rate(),
        "prefix_hash": ph,
        "latest_seq": agent.latest_event_seq(),
        "busy": agent.is_busy(),
        "pending_queue": pending_queue,
        "metrics": met,
        "modules": agent.config.modules,
        "tools": {
            "active": active_tools,
            "deferred": deferred_tools,
        },
    })
}

/// Serves standard Host JSON-RPC 2.0 over stdio (without `--host-extension`).
pub async fn serve_rpc_stdio(agent: Arc<Agent>) -> Result<()> {
    let stdin = BufReader::new(tokio::io::stdin());
    let stdout = Arc::new(Mutex::new(tokio::io::stdout()));
    serve_rpc_loop(agent, None, None, stdin.lines(), stdout, None).await
}

/// Serves Host JSON-RPC 2.0 over stdio with an optional `RpcHost` (for `set_model`,
/// `get_available_models`, `get_commands`).
pub async fn serve_rpc_stdio_with_host(
    agent: Arc<Agent>,
    host: Option<Arc<dyn RpcHost>>,
) -> Result<()> {
    let stdin = BufReader::new(tokio::io::stdin());
    let stdout = Arc::new(Mutex::new(tokio::io::stdout()));
    serve_rpc_loop(agent, host, None, stdin.lines(), stdout, None).await
}

/// Completes the `--host-extension` handshake and runs the multiplexed bidirectional RPC loop.
pub async fn serve_rpc_stdio_with_extension(
    agent: Arc<Agent>,
    host: Option<Arc<dyn RpcHost>>,
    bridge: Arc<HostRpcBridge>,
    pending: PendingHostSession,
) -> Result<()> {
    serve_rpc_loop(
        agent,
        host,
        Some(bridge),
        pending.lines,
        pending.stdout,
        Some(pending.request_id),
    )
    .await
}

async fn serve_rpc_loop(
    agent: Arc<Agent>,
    host: Option<Arc<dyn RpcHost>>,
    bridge: Option<Arc<HostRpcBridge>>,
    mut lines: Lines<BufReader<tokio::io::Stdin>>,
    stdout: Arc<Mutex<tokio::io::Stdout>>,
    register_reply_id: Option<Value>,
) -> Result<()> {
    let broker = ChannelInteractionBroker::new(Arc::clone(&agent));
    agent
        .set_question_broker(Arc::clone(&broker) as Arc<dyn carnot_ext::UserQuestionBroker>)
        .await;

    // Reply to `host/register` now that the Agent is fully constructed with frozen prefix & tools.
    if let Some(reg_id) = register_reply_id {
        let ready = json!({
            "jsonrpc": "2.0",
            "id": reg_id,
            "result": state_payload(&agent).await,
        });
        write_frame(&stdout, &ready).await?;
    }

    // Stream live events losslessly from the ring-buffered EventLog (`method: "event"`).
    let mut ev_stream = agent.subscribe();
    let out_ev = Arc::clone(&stdout);
    let ev_task = tokio::spawn(async move {
        loop {
            let env = ev_stream.next().await;
            let notif = json!({"jsonrpc": "2.0", "method": "event", "params": env});
            if write_frame(&out_ev, &notif).await.is_err() {
                break;
            }
        }
    });

    // Track spawned prompt / continue / compact runs so stdin NEVER blocks during host callbacks!
    // `run_active` serializes prompt/continue admission with follow-up draining so no follow-up
    // queued near run settlement can ever be stranded.
    let mut bg_tasks: Vec<JoinHandle<()>> = Vec::new();
    let run_active = Arc::new(Mutex::new(false));
    let shutting_down = Arc::new(AtomicBool::new(false));
    let mut terminated_by_signal = false;
    #[cfg(unix)]
    let mut sigterm =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok();

    loop {
        let next_line = {
            #[cfg(unix)]
            {
                tokio::select! {
                    line = lines.next_line() => line?,
                    () = async {
                        if let Some(ref mut s) = sigterm {
                            s.recv().await;
                        } else {
                            std::future::pending::<()>().await;
                        }
                    } => {
                        terminated_by_signal = true;
                        None
                    }
                    _ = tokio::signal::ctrl_c() => {
                        terminated_by_signal = true;
                        None
                    }
                }
            }
            #[cfg(not(unix))]
            {
                tokio::select! {
                    line = lines.next_line() => line?,
                    _ = tokio::signal::ctrl_c() => {
                        terminated_by_signal = true;
                        None
                    }
                }
            }
        };
        let Some(line) = next_line else {
            break;
        };
        if line.trim().is_empty() {
            continue;
        }
        let req = match serde_json::from_str::<Value>(&line) {
            Ok(v) => v,
            Err(e) => {
                let err_reply = json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": {"code": -32700, "message": format!("parse error: {e}")}
                });
                write_frame(&stdout, &err_reply).await?;
                continue;
            }
        };

        // Route JSON-RPC responses (from host answering engine -> host callbacks) immediately.
        if req.get("method").is_none() {
            if let Some(ref br) = bridge {
                let _ = br.deliver_response(&req).await;
            }
            continue;
        }

        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(Value::Null);

        if method == "prompt" {
            bg_tasks.retain(|h| !h.is_finished());
            let mut active = run_active.lock().await;
            if *active || agent.is_busy() {
                let behavior = params
                    .get("streaming_behavior")
                    .or_else(|| params.get("streamingBehavior"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let text = params
                    .get("message")
                    .or_else(|| params.get("prompt"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                match behavior {
                    "steer" => {
                        agent.steer(text).await;
                        drop(active);
                        let ok = json!({"jsonrpc": "2.0", "id": id, "result": {"queued": "steer"}});
                        write_frame(&stdout, &ok).await?;
                        continue;
                    }
                    "follow_up" | "followUp" => {
                        agent.follow_up(text).await;
                        drop(active);
                        let ok =
                            json!({"jsonrpc": "2.0", "id": id, "result": {"queued": "follow_up"}});
                        write_frame(&stdout, &ok).await?;
                        continue;
                    }
                    _ => {
                        drop(active);
                        let busy = json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {"code": -32001, "message": "agent is busy running a prompt; use steer, follow_up, or abort"}
                        });
                        write_frame(&stdout, &busy).await?;
                        continue;
                    }
                }
            }
            *active = true;
            drop(active);

            let ack_only = params.get("ack").and_then(Value::as_bool).unwrap_or(false);
            let blocks = build_prompt_blocks(&params);
            let ag = Arc::clone(&agent);
            let out = Arc::clone(&stdout);
            let active_lock = Arc::clone(&run_active);
            let stop_flag = Arc::clone(&shutting_down);
            if ack_only {
                let ack = json!({"jsonrpc": "2.0", "id": id, "result": {"accepted": true}});
                write_frame(&stdout, &ack).await?;
            }
            bg_tasks.push(tokio::spawn(async move {
                let first_res = ag.prompt_blocks(blocks).await;
                if !ack_only {
                    let reply = match first_res {
                        Ok(o) => json!({"jsonrpc": "2.0", "id": id, "result": {
                            "final_text": o.final_text,
                            "metrics": o.metrics,
                            "halted_reason": o.halted_reason,
                            "latest_seq": ag.latest_event_seq(),
                        }}),
                        Err(e) => json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {"code": -32000, "message": e.to_string()}
                        }),
                    };
                    let _ = write_frame(&out, &reply).await;
                }
                // Atomically drain any follow-up/steer that arrived right as the run settled
                // unless the RPC transport is shutting down on EOF/SIGTERM.
                loop {
                    let next = {
                        let mut guard = active_lock.lock().await;
                        if stop_flag.load(Ordering::Acquire) {
                            *guard = false;
                            None
                        } else if let Some(msg) = ag.take_pending_wakeup().await {
                            Some(msg)
                        } else {
                            *guard = false;
                            None
                        }
                    };
                    let Some(followup) = next else {
                        break;
                    };
                    let _ = ag.prompt(followup).await;
                }
            }));
            continue;
        }

        if method == "continue" {
            bg_tasks.retain(|h| !h.is_finished());
            let mut active = run_active.lock().await;
            if *active || agent.is_busy() {
                drop(active);
                let busy = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32001, "message": "agent is already busy"}
                });
                write_frame(&stdout, &busy).await?;
                continue;
            }
            *active = true;
            drop(active);

            let ag = Arc::clone(&agent);
            let out = Arc::clone(&stdout);
            let active_lock = Arc::clone(&run_active);
            let stop_flag = Arc::clone(&shutting_down);
            bg_tasks.push(tokio::spawn(async move {
                let reply = match ag.continue_run().await {
                    Ok(Some(o)) => json!({"jsonrpc": "2.0", "id": id, "result": {
                        "continued": true,
                        "final_text": o.final_text,
                        "metrics": o.metrics,
                        "halted_reason": o.halted_reason,
                        "latest_seq": ag.latest_event_seq(),
                    }}),
                    Ok(None) => json!({"jsonrpc": "2.0", "id": id, "result": {
                        "continued": false,
                    }}),
                    Err(e) => json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {"code": -32000, "message": e.to_string()}
                    }),
                };
                let _ = write_frame(&out, &reply).await;
                loop {
                    let next = {
                        let mut guard = active_lock.lock().await;
                        if stop_flag.load(Ordering::Acquire) {
                            *guard = false;
                            None
                        } else if let Some(msg) = ag.take_pending_wakeup().await {
                            Some(msg)
                        } else {
                            *guard = false;
                            None
                        }
                    };
                    let Some(followup) = next else {
                        break;
                    };
                    let _ = ag.prompt(followup).await;
                }
            }));
            continue;
        }

        if method == "compact" {
            bg_tasks.retain(|h| !h.is_finished());
            let focus = params
                .get("focus")
                .or_else(|| params.get("custom_instructions"))
                .or_else(|| params.get("customInstructions"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string);
            let ag = Arc::clone(&agent);
            let out = Arc::clone(&stdout);
            // Spawn so `before_compact` host callbacks never deadlock on stdin!
            bg_tasks.push(tokio::spawn(async move {
                let reply = match ag.compact_now_with_focus(focus.as_deref()).await {
                    Ok((before, after, semantic)) => json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "tokens_before": before,
                            "tokens_after": after,
                            "semantic_llm": semantic,
                        }
                    }),
                    Err(e) => json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {"code": -32000, "message": e.to_string()}
                    }),
                };
                let _ = write_frame(&out, &reply).await;
            }));
            continue;
        }

        let reply = match method {
            "steer" => {
                let text = params
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                agent.steer(text).await;
                json!({"jsonrpc": "2.0", "id": id, "result": {"queued": true}})
            }
            "follow_up" => {
                let text = params
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                agent.follow_up(text).await;
                json!({"jsonrpc": "2.0", "id": id, "result": {"queued": true}})
            }
            "undo" => match agent.undo_last_edit() {
                Ok(path) => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "reverted": path.map(|p| p.display().to_string()) }
                }),
                Err(e) => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32000, "message": e.to_string()}
                }),
            },
            "replay_events" => {
                let after_seq = params.get("after_seq").and_then(Value::as_u64).unwrap_or(0);
                let events = agent.event_log().replay_after(after_seq);
                json!({"jsonrpc": "2.0", "id": id, "result": {"events": events, "latest_seq": agent.latest_event_seq()}})
            }
            "get_state" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": state_payload(&agent).await,
            }),
            "get_session_stats" => {
                let met = agent.metrics().await;
                let (ctx_tokens, ctx_breakdown) = agent.current_context_with_breakdown().await;
                let ctx_window = agent.current_context_window().await;
                let active_msgs = agent.active_messages().await;
                let msg_count = active_msgs.len();
                let tool_calls: usize = active_msgs
                    .iter()
                    .map(|m| {
                        m.content
                            .iter()
                            .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
                            .count()
                    })
                    .sum();
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "context_tokens": ctx_tokens,
                        "context_window": ctx_window,
                        "context_breakdown": ctx_breakdown,
                        "cost_usd": met.total_usage.cost_usd,
                        "tokens": {
                            "input": met.total_usage.input_tokens,
                            "output": met.total_usage.output_tokens,
                            "cache_read": met.total_usage.cache_read_tokens,
                            "cache_write_5m": met.total_usage.cache_write_5m_tokens,
                            "cache_write_1h": met.total_usage.cache_write_1h_tokens,
                            "reasoning": met.total_usage.reasoning_tokens,
                        },
                        "message_count": msg_count,
                        "messages": msg_count,
                        "tool_calls": tool_calls,
                        "turns": met.turns,
                        "requests": met.requests,
                        "cache_ttl": agent.current_cache_ttl().await.as_str(),
                        "cache_hit_rate": met.cache_hit_rate(),
                        "prefix_retention_rate": met.prefix_retention_rate(),
                        "cache_breaks": met.cache_breaks,
                    }
                })
            }
            "get_entries" => {
                let entries = agent.session_entries().await;
                let active: Vec<carnot_core::Message> = agent
                    .active_messages()
                    .await
                    .into_iter()
                    .map(|m| (*m).clone())
                    .collect();
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "entries": entries,
                        "active_messages": active,
                    }
                })
            }
            "append_entry" => {
                let kind = params
                    .get("kind")
                    .or_else(|| params.get("custom_type"))
                    .and_then(Value::as_str)
                    .unwrap_or("custom");
                let data = params.get("data").cloned().unwrap_or(Value::Null);
                match agent.append_custom_entry(kind, data).await {
                    Ok(entry_id) => json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": { "id": entry_id }
                    }),
                    Err(e) => json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {"code": -32000, "message": e.to_string()}
                    }),
                }
            }
            "set_model" => {
                let model_spec = params
                    .get("model")
                    .or_else(|| params.get("model_id"))
                    .or_else(|| params.get("modelId"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let provider_override = params
                    .get("provider")
                    .and_then(Value::as_str)
                    .filter(|s| !s.trim().is_empty());
                if model_spec.trim().is_empty() {
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {"code": -32602, "message": "missing `model` parameter"}
                    })
                } else if let Some(ref h) = host {
                    match h.resolve_model(model_spec, provider_override) {
                        Ok(resolved) => match agent
                            .set_model(
                                resolved.model_id.clone(),
                                resolved.provider,
                                resolved.context_window,
                                resolved.max_output_tokens,
                            )
                            .await
                        {
                            Ok(()) => json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "result": {
                                    "provider": resolved.provider_id,
                                    "model": resolved.model_id,
                                    "context_window": agent.current_context_window().await,
                                    "effort_preserves_cache": agent.effort_preserves_cache().await,
                                }
                            }),
                            Err(e) => json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "error": {"code": -32000, "message": e.to_string()}
                            }),
                        },
                        Err(e) => json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {"code": -32000, "message": e.to_string()}
                        }),
                    }
                } else {
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {"code": -32000, "message": "set_model requires an RpcHost"}
                    })
                }
            }
            "set_effort" => {
                let level = params
                    .get("effort")
                    .or_else(|| params.get("level"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let scoped = params
                    .get("scoped")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if let Some(eff) = parse_effort(level) {
                    let applied = if scoped {
                        agent.set_scoped_effort(eff).await
                    } else {
                        agent.set_effort(eff).await;
                        true
                    };
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "applied": applied,
                            "effort": effort_label(agent.current_effort().await),
                            "effort_preserves_cache": agent.effort_preserves_cache().await,
                        }
                    })
                } else {
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {"code": -32602, "message": format!("invalid effort `{level}` (expected low|medium|high|xhigh|max)")}
                    })
                }
            }
            "set_cache_ttl" => {
                let raw = params
                    .get("ttl")
                    .or_else(|| params.get("cache_ttl"))
                    .or_else(|| params.get("retention"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if let Some(ttl) = CacheTtl::parse(raw) {
                    agent.set_cache_ttl(ttl).await;
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "cache_ttl": agent.current_cache_ttl().await.as_str(),
                        }
                    })
                } else {
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {"code": -32602, "message": format!("invalid cache_ttl `{raw}` (expected short|hybrid|long)")}
                    })
                }
            }
            "get_available_models" => {
                let models = host.as_ref().map(|h| h.list_models()).unwrap_or_default();
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "models": models }
                })
            }
            "get_commands" => {
                let commands = host.as_ref().map(|h| h.list_commands()).unwrap_or_default();
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "commands": commands }
                })
            }
            "get_metrics" => {
                let met = agent.metrics().await;
                json!({"jsonrpc": "2.0", "id": id, "result": met})
            }
            "set_require_approval" => {
                let require = params
                    .get("enabled")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                if require {
                    agent
                        .set_permission_broker(
                            Arc::clone(&broker) as Arc<dyn carnot_ext::PermissionBroker>
                        )
                        .await;
                } else {
                    agent.clear_permission_broker().await;
                }
                json!({"jsonrpc": "2.0", "id": id, "result": {"require_approval": require}})
            }
            "list_pending" => {
                let pending = broker.list_pending().await;
                json!({"jsonrpc": "2.0", "id": id, "result": pending})
            }
            "respond_permission" => {
                let req_id = params
                    .get("request_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let decision = params
                    .get("decision")
                    .and_then(Value::as_str)
                    .unwrap_or("deny");
                let reason = params
                    .get("reason")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let ok = broker.respond_permission(req_id, decision, reason).await;
                json!({"jsonrpc": "2.0", "id": id, "result": {"resolved": ok}})
            }
            "respond_question" => {
                let req_id = params
                    .get("request_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let answers: Vec<UserQuestionAnswer> = params
                    .get("answers")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok())
                    .unwrap_or_default();
                let ok = broker.respond_question(req_id, answers).await;
                json!({"jsonrpc": "2.0", "id": id, "result": {"resolved": ok}})
            }
            "abort" => {
                agent.abort().await;
                json!({"jsonrpc": "2.0", "id": id, "result": {"aborted": true}})
            }
            "ping" => json!({"jsonrpc": "2.0", "id": id, "result": {"pong": true}}),
            other => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32601, "message": format!("method not found: {other}")}
            }),
        };

        write_frame(&stdout, &reply).await?;
    }

    shutting_down.store(true, Ordering::Release);
    if let Some(ref br) = bridge {
        br.fail_all(if terminated_by_signal {
            "carnot received termination signal"
        } else {
            "host stdin closed"
        })
        .await;
        agent.abort().await;
    } else if terminated_by_signal {
        agent.abort().await;
    }
    for task in bg_tasks {
        #[cfg(unix)]
        {
            tokio::pin!(task);
            tokio::select! {
                _ = &mut task => {}
                () = async {
                    if let Some(ref mut s) = sigterm {
                        s.recv().await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                } => {
                    agent.abort().await;
                    let _ = task.await;
                }
                _ = tokio::signal::ctrl_c() => {
                    agent.abort().await;
                    let _ = task.await;
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = task.await;
        }
    }
    ev_task.abort();
    Ok(())
}
