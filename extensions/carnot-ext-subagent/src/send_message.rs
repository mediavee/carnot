//! Hub-and-spoke `SendMessage` tool for coordinator <-> worker communication.

use crate::FleetRef;
use crate::fleet::{WorkerState, spawn_worker_turn};
use carnot_agent::Agent;
use carnot_core::{Result, ToolEffects, ToolOutput, ToolSpec};
use carnot_tools::{BoxFuture, Tool};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, Weak};
use tokio_util::sync::CancellationToken;

/// Hub-and-spoke messaging tool registered identically on coordinator & workers (preserving
/// byte-identical static prefix for cache-warm `fork: true`).
///   - Coordinator -> worker `to`: wakes idle worker with `prompt()`, or steers/queues busy worker.
///   - Worker -> `"coordinator"`: pushes into `Fleet::coordinator_inbox` (single-channel delivery
///     as untrusted `Message::system_note`, never `[user-steer]`).
#[derive(Debug)]
pub(crate) struct SendMessageTool {
    /// `None` on the top-level coordinator; `Some(worker_name)` inside a child worker.
    pub(crate) sender_name: Option<String>,
    pub(crate) fleet: FleetRef,
    pub(crate) parent_agent: Arc<OnceLock<Weak<Agent>>>,
}

impl Tool for SendMessageTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "SendMessage".into(),
            description: "Message between coordinator and a named worker (wakes idle workers or queues/steers running ones; workers message `coordinator`).".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["to", "message"],
                "properties": {
                    "to": { "type": "string", "description": "Worker name or `coordinator`" },
                    "message": { "type": "string" },
                    "interrupt": { "type": "boolean", "description": "Steer mid-turn instead of queuing" }
                }
            }),
            defer_loading: false,
        }
    }

    fn effects(&self, input: &Value) -> ToolEffects {
        // Serialize concurrent messages targeting the SAME worker while allowing parallel sends
        // to distinct workers.
        let target = input
            .get("to")
            .and_then(Value::as_str)
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".into());
        ToolEffects::write_path(PathBuf::from(format!("worker:{target}")))
    }

    fn execute(
        &self,
        input: Value,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let Some(fleet) = self.fleet.upgrade() else {
                return Ok(ToolOutput::err_text(
                    "Fleet coordinator is no longer active.",
                ));
            };

            let Some(to) = input
                .get("to")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
            else {
                return Ok(ToolOutput::err_text("SendMessage requires non-empty `to`"));
            };
            let Some(message) = input
                .get("message")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
            else {
                return Ok(ToolOutput::err_text(
                    "SendMessage requires non-empty `message`",
                ));
            };

            // Worker -> Coordinator (single delivery channel: coordinator_inbox -> untrusted system_note)
            if let Some(ref worker_name) = self.sender_name {
                if !to.eq_ignore_ascii_case("coordinator") {
                    return Ok(ToolOutput::err_text(
                        "Workers may only message `coordinator` (hub-and-spoke topology).",
                    ));
                }
                fleet.push_coordinator_message(worker_name.clone(), message);
                return Ok(ToolOutput::ok_text(format!(
                    "Delivered message to coordinator from `{worker_name}`."
                )));
            }

            // Coordinator -> Named Worker
            let Some(worker) = fleet.get(to) else {
                return Ok(ToolOutput::err_text(format!(
                    "Unknown worker `{to}` — call Subagent{{\"list\":true}} to inspect active workers."
                )));
            };
            let interrupt = input
                .get("interrupt")
                .and_then(Value::as_bool)
                .unwrap_or(false);

            match worker.state() {
                WorkerState::Running => {
                    if interrupt {
                        worker.agent.steer(message).await;
                    } else {
                        worker.agent.follow_up(message).await;
                    }
                    // Dekker recheck: if the worker finished transitioning to `Idle` while we
                    // queued the message, atomically claim the wake so the message is never stranded.
                    if matches!(worker.state(), WorkerState::Idle | WorkerState::Failed(_))
                        && let Some(msg) = worker.agent.take_pending_wakeup().await
                    {
                        match fleet.try_claim_wake(&worker) {
                            Err(cap) => {
                                worker.agent.follow_up(msg).await;
                                return Ok(ToolOutput::err_text(format!(
                                    "Worker `{to}` just went idle, but max concurrent workers ({cap}) is reached; wait for or close a worker before waking `{to}`."
                                )));
                            }
                            Ok(true) => {
                                let parent_weak = self.parent_agent.get().cloned();
                                spawn_worker_turn(
                                    Arc::downgrade(&fleet),
                                    Arc::clone(&fleet.budget),
                                    Arc::clone(&worker),
                                    msg,
                                    parent_weak,
                                );
                                return Ok(ToolOutput::ok_text(format!(
                                    "Woke worker `{to}` with warm context (now running)."
                                )));
                            }
                            Ok(false) => {
                                worker.agent.follow_up(msg).await;
                            }
                        }
                    }
                    if interrupt {
                        Ok(ToolOutput::ok_text(format!(
                            "Steered running worker `{to}` (will apply at next tool boundary)."
                        )))
                    } else {
                        Ok(ToolOutput::ok_text(format!(
                            "Queued follow-up message for running worker `{to}`."
                        )))
                    }
                }
                WorkerState::Idle | WorkerState::Failed(_) => {
                    match fleet.try_claim_wake(&worker) {
                        Err(cap) => {
                            return Ok(ToolOutput::err_text(format!(
                                "max concurrent workers ({cap}) reached; wait for or close an existing worker before waking `{to}`"
                            )));
                        }
                        Ok(false) => {
                            if worker.state().is_running() {
                                if interrupt {
                                    worker.agent.steer(message).await;
                                } else {
                                    worker.agent.follow_up(message).await;
                                }
                                return Ok(ToolOutput::ok_text(format!(
                                    "Queued message for running worker `{to}`."
                                )));
                            }
                            return Ok(ToolOutput::err_text(format!(
                                "Worker `{to}` is closed; spawn a new worker instead."
                            )));
                        }
                        Ok(true) => {}
                    }
                    let parent_weak = self.parent_agent.get().cloned();
                    spawn_worker_turn(
                        Arc::downgrade(&fleet),
                        Arc::clone(&fleet.budget),
                        Arc::clone(&worker),
                        message,
                        parent_weak,
                    );
                    Ok(ToolOutput::ok_text(format!(
                        "Woke worker `{to}` with warm context (now running)."
                    )))
                }
                WorkerState::Closed => Ok(ToolOutput::err_text(format!(
                    "Worker `{to}` is closed; spawn a new worker instead."
                ))),
            }
        })
    }
}
