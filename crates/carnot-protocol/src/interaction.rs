//! Shared async channel broker for interactive approvals (`PermissionBroker`) and clarifying
//! questions (`UserQuestionBroker`) across Host RPC, HTTP/SSE, and embedded Rust hosts.

use carnot_agent::Agent;
use carnot_core::{AgentEvent, CarnotError, Result, UserQuestion, UserQuestionAnswer};
use carnot_ext::{BoxFuture, PermissionBroker, ToolGateDecision, UserQuestionBroker};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, oneshot};

const INTERACTION_TIMEOUT_SECS: u64 = 300;

#[derive(Debug)]
struct PendingPermission {
    call_id: String,
    tool: String,
    input: Value,
    tx: oneshot::Sender<ToolGateDecision>,
}

#[derive(Debug)]
struct PendingQuestion {
    questions: Vec<UserQuestion>,
    tx: oneshot::Sender<Vec<UserQuestionAnswer>>,
}

/// Unified broker emitting `PermissionRequested` / `UserQuestionRequested` onto `Agent`'s
/// `EventLog` and awaiting out-of-band host replies via `respond_permission` / `respond_question`.
#[derive(Debug)]
pub struct ChannelInteractionBroker {
    agent: Arc<Agent>,
    next_id: AtomicU64,
    allowed_always: Mutex<HashSet<String>>,
    pending_permissions: Mutex<HashMap<String, PendingPermission>>,
    pending_questions: Mutex<HashMap<String, PendingQuestion>>,
}

impl ChannelInteractionBroker {
    pub fn new(agent: Arc<Agent>) -> Arc<Self> {
        Arc::new(Self {
            agent,
            next_id: AtomicU64::new(1),
            allowed_always: Mutex::new(HashSet::new()),
            pending_permissions: Mutex::new(HashMap::new()),
            pending_questions: Mutex::new(HashMap::new()),
        })
    }

    /// Snapshot of currently pending permission & question requests (for `list_pending` / `GET /v1/pending`).
    pub async fn list_pending(&self) -> Value {
        let perms: Vec<Value> = self
            .pending_permissions
            .lock()
            .await
            .iter()
            .map(|(id, p)| {
                json!({
                    "request_id": id,
                    "call_id": p.call_id,
                    "tool": p.tool,
                    "input": p.input,
                })
            })
            .collect();
        let questions: Vec<Value> = self
            .pending_questions
            .lock()
            .await
            .iter()
            .map(|(id, q)| {
                json!({
                    "request_id": id,
                    "questions": q.questions,
                })
            })
            .collect();
        json!({
            "permissions": perms,
            "questions": questions,
        })
    }

    /// Resolves a pending permission request (`decision`: `"allow_once" | "allow_always" | "deny"`).
    pub async fn respond_permission(
        &self,
        request_id: &str,
        decision: &str,
        reason: Option<String>,
    ) -> bool {
        let Some(pending) = self.pending_permissions.lock().await.remove(request_id) else {
            return false;
        };
        let (allowed, gate) = match decision.trim().to_ascii_lowercase().as_str() {
            "allow" | "allow_once" | "accept" => (true, ToolGateDecision::Allow),
            "allow_always" | "accept_for_session" => {
                self.allowed_always.lock().await.insert(pending.tool);
                (true, ToolGateDecision::Allow)
            }
            _ => (
                false,
                ToolGateDecision::Deny {
                    reason: reason.unwrap_or_else(|| "denied by user".into()),
                },
            ),
        };
        self.agent.emit_event(AgentEvent::PermissionResolved {
            request_id: request_id.to_string(),
            allowed,
        });
        let _ = pending.tx.send(gate);
        true
    }

    /// Resolves a pending `AskUser` question request.
    pub async fn respond_question(
        &self,
        request_id: &str,
        answers: Vec<UserQuestionAnswer>,
    ) -> bool {
        let Some(pending) = self.pending_questions.lock().await.remove(request_id) else {
            return false;
        };
        self.agent.emit_event(AgentEvent::UserQuestionAnswered {
            request_id: request_id.to_string(),
        });
        let _ = pending.tx.send(answers);
        true
    }
}

impl PermissionBroker for ChannelInteractionBroker {
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
            let effective_tool = if tool_name == "CallDeferredTool" {
                input
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(&tool_name)
                    .to_string()
            } else {
                tool_name
            };
            if matches!(
                effective_tool.as_str(),
                "Read" | "Grep" | "ToolSearch" | "AskUser" | "MemorySearch" | "LspQuery"
            ) || self.allowed_always.lock().await.contains(&effective_tool)
            {
                return Ok(ToolGateDecision::Allow);
            }

            let req_id = format!("perm-{}", self.next_id.fetch_add(1, Ordering::Relaxed));
            let (tx, rx) = oneshot::channel();
            self.pending_permissions.lock().await.insert(
                req_id.clone(),
                PendingPermission {
                    call_id: call_id.clone(),
                    tool: effective_tool.clone(),
                    input: input.clone(),
                    tx,
                },
            );
            self.agent.emit_event(AgentEvent::PermissionRequested {
                request_id: req_id.clone(),
                call_id,
                tool: effective_tool,
                input,
            });

            match tokio::time::timeout(Duration::from_secs(INTERACTION_TIMEOUT_SECS), rx).await {
                Ok(Ok(dec)) => Ok(dec),
                Ok(Err(_)) => Ok(ToolGateDecision::Deny {
                    reason: "permission channel closed".into(),
                }),
                Err(_) => {
                    self.pending_permissions.lock().await.remove(&req_id);
                    Ok(ToolGateDecision::Deny {
                        reason: format!("permission request `{req_id}` timed out"),
                    })
                }
            }
        })
    }
}

impl UserQuestionBroker for ChannelInteractionBroker {
    fn ask<'a>(
        &'a self,
        request_id: &'a str,
        questions: &'a [UserQuestion],
    ) -> BoxFuture<'a, Result<Vec<UserQuestionAnswer>>> {
        let req_id = request_id.to_string();
        let qs = questions.to_vec();
        Box::pin(async move {
            let (tx, rx) = oneshot::channel();
            self.pending_questions.lock().await.insert(
                req_id.clone(),
                PendingQuestion {
                    questions: qs.clone(),
                    tx,
                },
            );
            self.agent.emit_event(AgentEvent::UserQuestionRequested {
                request_id: req_id.clone(),
                questions: qs,
            });

            match tokio::time::timeout(Duration::from_secs(INTERACTION_TIMEOUT_SECS), rx).await {
                Ok(Ok(answers)) => Ok(answers),
                Ok(Err(_)) => Err(CarnotError::Tool {
                    tool: "AskUser".into(),
                    message: "question channel closed before user answered".into(),
                }),
                Err(_) => {
                    self.pending_questions.lock().await.remove(&req_id);
                    Err(CarnotError::Tool {
                        tool: "AskUser".into(),
                        message: format!("question `{req_id}` timed out waiting for user answer"),
                    })
                }
            }
        })
    }
}
