//! Per-session bidirectional ACP permission and user-question brokers over stdio JSON-RPC.

use super::PendingReplies;
use super::mapping::{acp_tool_kind, format_tool_title};
use carnot_core::{CarnotError, Result, UserQuestion, UserQuestionAnswer};
use carnot_ext::{BoxFuture, PermissionBroker, ToolGateDecision, UserQuestionBroker};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, oneshot};

pub(super) async fn write_frame(
    stdout: &Arc<Mutex<tokio::io::Stdout>>,
    frame: &Value,
) -> std::io::Result<()> {
    let mut s = serde_json::to_string(frame).unwrap_or_else(|_| "{}".into());
    s.push('\n');
    let mut w = stdout.lock().await;
    w.write_all(s.as_bytes()).await?;
    w.flush().await
}

/// Per-session bidirectional ACP `session/request_permission` broker with `allow_always` session cache.
#[derive(Debug)]
pub(super) struct AcpPermissionBroker {
    session_id: String,
    stdout: Arc<Mutex<tokio::io::Stdout>>,
    next_req_id: Arc<AtomicU64>,
    pending: PendingReplies,
    allowed_always: Mutex<HashSet<String>>,
}

impl AcpPermissionBroker {
    pub(super) fn new(
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
pub(super) struct AcpQuestionBroker {
    pub(super) session_id: String,
    pub(super) stdout: Arc<Mutex<tokio::io::Stdout>>,
    pub(super) next_req_id: Arc<AtomicU64>,
    pub(super) pending: PendingReplies,
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
