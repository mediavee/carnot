//! Pi/Lythe-compatible Host JSON-RPC 2.0 transport over stdio with concurrent `steer`/`abort`,
//! busy protection, `follow_up`/`compact`/`undo`, and lossless `EventLog` streaming.

use crate::interaction::ChannelInteractionBroker;
use carnot_agent::Agent;
use carnot_core::{Result, UserQuestionAnswer};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

async fn write_frame(stdout: &Arc<Mutex<tokio::io::Stdout>>, frame: &Value) -> std::io::Result<()> {
    let mut s = serde_json::to_string(frame).unwrap_or_else(|_| "{}".into());
    s.push('\n');
    let mut w = stdout.lock().await;
    w.write_all(s.as_bytes()).await?;
    w.flush().await
}

pub async fn serve_rpc_stdio(agent: Arc<Agent>) -> Result<()> {
    let broker = ChannelInteractionBroker::new(Arc::clone(&agent));
    agent.set_question_broker(broker.clone()).await;

    let stdin = BufReader::new(tokio::io::stdin());
    let stdout = Arc::new(Mutex::new(tokio::io::stdout()));
    let mut lines = stdin.lines();

    // Stream live events losslessly from the ring-buffered EventLog (`method: "event"`)
    let mut ev_stream = agent.subscribe();
    let out_ev = stdout.clone();
    let ev_task = tokio::spawn(async move {
        loop {
            let env = ev_stream.next().await;
            let notif = json!({"jsonrpc": "2.0", "method": "event", "params": env});
            let _ = write_frame(&out_ev, &notif).await;
        }
    });

    // Track spawned prompt runs so stdin stays responsive to `steer`, `follow_up`, `abort`, `get_state`, `ping`
    let mut prompt_tasks: Vec<JoinHandle<()>> = Vec::new();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(req) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(Value::Null);

        if method == "prompt" {
            prompt_tasks.retain(|h| !h.is_finished());
            if agent.is_busy() {
                let busy = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32001, "message": "agent is busy running a prompt; use steer, follow_up, or abort"}
                });
                write_frame(&stdout, &busy).await?;
                continue;
            }
            let text = params
                .get("message")
                .or_else(|| params.get("prompt"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let ag = agent.clone();
            let out = stdout.clone();
            prompt_tasks.push(tokio::spawn(async move {
                let reply = match ag.prompt(text).await {
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
            "compact" => {
                let focus = params
                    .get("focus")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty());
                match agent.compact_now_with_focus(focus).await {
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
                }
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
            "get_state" => {
                let sid = agent.session_id().await;
                let model = agent.current_model().await;
                let ph = agent.prefix_hash().await;
                let met = agent.metrics().await;
                json!({"jsonrpc": "2.0", "id": id, "result": {
                    "session_id": sid,
                    "model": model,
                    "prefix_hash": ph,
                    "latest_seq": agent.latest_event_seq(),
                    "busy": agent.is_busy(),
                    "metrics": met,
                    "modules": agent.config.modules,
                }})
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
                    agent.set_permission_broker(broker.clone()).await;
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

    for task in prompt_tasks {
        let _ = task.await;
    }
    ev_task.abort();
    Ok(())
}
