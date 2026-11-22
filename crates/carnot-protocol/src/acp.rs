//! Agent Client Protocol (ACP) JSON-RPC 2.0 transport over stdio — Zed / T3 Code compliant:
//!   - Emits canonical ACP `sessionUpdate` payloads (`agent_message_chunk`, `agent_thought_chunk`,
//!     `tool_call`, `tool_call_update`) without empty-chunk spam
//!   - Bidirectional `AcpPermissionBroker` (`session/request_permission` client callback routed via
//!     pending JSON-RPC response map — fixes Opus #3!)
//!   - Concurrent `session/prompt` so `session/cancel`, `session/steer`, and permission replies
//!     execute immediately mid-run (N4 + Opus #3)
//!   - Real `session/load` backed by `carnot-session` JSONL replay with non-blocking busy check and
//!     strict `session_dir` confinement (Opus #7)

use carnot_agent::Agent;
use carnot_core::{AgentEvent, CarnotError, EventEnvelope, Result, Role};
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

async fn write_frame(stdout: &Arc<Mutex<tokio::io::Stdout>>, frame: &Value) -> std::io::Result<()> {
    let mut s = serde_json::to_string(frame).unwrap_or_else(|_| "{}".into());
    s.push('\n');
    let mut w = stdout.lock().await;
    w.write_all(s.as_bytes()).await?;
    w.flush().await
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
            if matches!(tool_name.as_str(), "Read" | "ToolSearch") {
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

pub async fn serve_acp_stdio(agent: Arc<Agent>) -> Result<()> {
    let stdin = BufReader::new(tokio::io::stdin());
    let stdout = Arc::new(Mutex::new(tokio::io::stdout()));
    let pending_replies: PendingReplies = Arc::new(Mutex::new(HashMap::new()));
    let next_client_req_id = Arc::new(AtomicU64::new(100_000));
    let active_sid = Arc::new(Mutex::new(agent.session_id().await));

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

        let id = msg.get("id").cloned().unwrap_or(Value::Null);
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        if method == "session/prompt" {
            let prompt_text = extract_acp_prompt(&params);
            let ag = agent.clone();
            let out = stdout.clone();
            prompt_tasks.push(tokio::spawn(async move {
                let mut ev_stream = ag.subscribe();
                let out_writer = out.clone();
                let sid = ag.session_id().await;
                let pump = tokio::spawn(async move {
                    loop {
                        let env = ev_stream.next().await;
                        let is_done = matches!(
                            env.event,
                            AgentEvent::RunSettled { .. } | AgentEvent::Error { .. }
                        );
                        if let Some(notif) = acp_update_frame(&sid, &env) {
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

        let result: std::result::Result<Value, (i32, String)> = match method {
            "initialize" => {
                // Attach interactive ACP PermissionBroker when client advertises support!
                let client_caps = params.get("clientCapabilities");
                let wants_perms = client_caps
                    .and_then(|c| {
                        c.get("permissionRequests")
                            .or_else(|| c.get("requestPermission"))
                    })
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if wants_perms {
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
                    "agentInfo": { "name": "carnot", "version": env!("CARGO_PKG_VERSION") },
                    "agentCapabilities": {
                        "loadSession": true,
                        "permissionRequests": true,
                        "promptCapabilities": { "embeddedContext": true, "image": true }
                    }
                }))
            }
            "session/new" => {
                let sid = agent.session_id().await;
                *active_sid.lock().await = sid.clone();
                let prefix_hash = agent.prefix_hash().await;
                Ok(json!({
                    "sessionId": sid,
                    "_meta": {
                        "carnot": {
                            "model": agent.config.model,
                            "prefixHash": prefix_hash,
                            "latestSeq": agent.latest_event_seq(),
                            "modules": agent.config.modules,
                        }
                    }
                }))
            }
            "session/load" => {
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
                                Ok(json!({ "sessionId": sid, "loaded": true }))
                            }
                            Err(e) => Err((-32000, e.to_string())),
                        },
                    }
                }
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

    for task in prompt_tasks {
        let _ = task.await;
    }
    Ok(())
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
        return arr
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
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
}
