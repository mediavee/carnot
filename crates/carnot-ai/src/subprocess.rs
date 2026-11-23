//! Persistent long-lived external provider plugin over stdin/stdout NDJSON (`api = "subprocess-jsonl"`).
//!
//! Enables ANY external SDK, custom enterprise gateway, or language runtime (Python, Node, Go, Rust)
//! to plug in as a first-class Carnot provider via `providers.json` without recompiling Carnot!
//!
//! Protocol (`carnot-provider-v1`):
//! - Child process stays alive across turns (preserves HTTP connection pools, OAuth tokens, warm state).
//! - Startup handshake: host sends `{"type":"hello","protocol":"carnot-provider-v1"}`;
//!   child replies `{"type":"hello_ok","protocol":"carnot-provider-v1","native_tool_search":false}`.
//! - Per request: host sends `{"type":"stream","request_id":"r1","request":{...}}`;
//!   child streams NDJSON events (`text_delta`, `thinking_delta`, `tool_call_ready`, `done`, `error`).
//! - Cancellation: host sends `{"type":"cancel","request_id":"r1"}` and kills the child if it
//!   fails to settle within `cancel_kill_timeout_ms`.
//! - Security contract: external provider processes NEVER execute local tools!

use crate::catalog::{ResolvedProviderSpec, SubprocessProviderConfig};
use crate::registry::ApiAdapter;
use crate::{BoxFuture, Provider, ProviderRequest, StreamEvent};
use carnot_core::{CarnotError, Message, Result, StopReason, Usage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

const API_ID: &str = "subprocess-jsonl";
const PROTOCOL_VERSION: &str = "carnot-provider-v1";

static REQ_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct SubprocessJsonlAdapter;

impl ApiAdapter for SubprocessJsonlAdapter {
    fn api_id(&self) -> &'static str {
        API_ID
    }

    fn build(&self, spec: &ResolvedProviderSpec) -> Result<Arc<dyn Provider>> {
        let cfg = spec
            .subprocess
            .clone()
            .ok_or_else(|| CarnotError::Provider {
                provider: spec.provider_id.clone(),
                message: "subprocess-jsonl adapter requires a `subprocess` block in providers.json"
                    .into(),
                retryable: false,
                retry_after_ms: None,
            })?;
        Ok(Arc::new(SubprocessProvider::new(
            spec.provider_id.clone(),
            cfg,
            spec.native_tool_search,
        )))
    }
}

#[derive(Debug)]
struct LiveChild {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[derive(Debug)]
pub struct SubprocessProvider {
    provider_id: String,
    config: SubprocessProviderConfig,
    native_tool_search: AtomicBool,
    live: Arc<Mutex<Option<LiveChild>>>,
    stderr_tail: Arc<Mutex<Vec<String>>>,
}

impl SubprocessProvider {
    pub fn new(
        provider_id: String,
        config: SubprocessProviderConfig,
        native_tool_search: bool,
    ) -> Self {
        Self {
            provider_id,
            config,
            native_tool_search: AtomicBool::new(native_tool_search),
            live: Arc::new(Mutex::new(None)),
            stderr_tail: Arc::new(Mutex::new(Vec::new())),
        }
    }

    async fn spawn_and_handshake(&self) -> Result<LiveChild> {
        let mut cmd = Command::new(&self.config.command);
        cmd.args(&self.config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for (k, v) in &self.config.env {
            cmd.env(k, v);
        }

        let mut child = cmd.spawn().map_err(|e| CarnotError::Provider {
            provider: self.provider_id.clone(),
            message: format!(
                "failed to spawn subprocess provider `{}`: {e}",
                self.config.command
            ),
            retryable: false,
            retry_after_ms: None,
        })?;

        let mut stdin = child.stdin.take().ok_or_else(|| CarnotError::Provider {
            provider: self.provider_id.clone(),
            message: "subprocess stdin pipe unavailable".into(),
            retryable: false,
            retry_after_ms: None,
        })?;
        let stdout_raw = child.stdout.take().ok_or_else(|| CarnotError::Provider {
            provider: self.provider_id.clone(),
            message: "subprocess stdout pipe unavailable".into(),
            retryable: false,
            retry_after_ms: None,
        })?;
        if let Some(stderr_raw) = child.stderr.take() {
            let tail = Arc::clone(&self.stderr_tail);
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr_raw).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    let mut g = tail.lock().await;
                    if g.len() >= 32 {
                        g.remove(0);
                    }
                    g.push(line);
                }
            });
        }

        let hello = json!({
            "type": "hello",
            "protocol": PROTOCOL_VERSION,
        });
        let mut line = serde_json::to_string(&hello)?;
        line.push('\n');
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await?;

        let mut stdout = BufReader::new(stdout_raw);
        let mut resp_line = String::new();
        let n = tokio::time::timeout(Duration::from_secs(10), stdout.read_line(&mut resp_line))
            .await
            .map_err(|_| CarnotError::Provider {
                provider: self.provider_id.clone(),
                message: "subprocess provider handshake timed out (10s)".into(),
                retryable: false,
                retry_after_ms: None,
            })??;
        if n == 0 {
            let stderr = self.stderr_tail.lock().await.join(" | ");
            return Err(CarnotError::Provider {
                provider: self.provider_id.clone(),
                message: format!("subprocess provider exited before handshake (stderr: {stderr})"),
                retryable: false,
                retry_after_ms: None,
            });
        }
        let v: Value =
            serde_json::from_str(resp_line.trim()).map_err(|e| CarnotError::Provider {
                provider: self.provider_id.clone(),
                message: format!("invalid handshake JSON from subprocess provider: {e}"),
                retryable: false,
                retry_after_ms: None,
            })?;
        if v.get("type").and_then(Value::as_str) != Some("hello_ok")
            || v.get("protocol").and_then(Value::as_str) != Some(PROTOCOL_VERSION)
        {
            return Err(CarnotError::Provider {
                provider: self.provider_id.clone(),
                message: format!("incompatible subprocess provider handshake: {v}"),
                retryable: false,
                retry_after_ms: None,
            });
        }
        if let Some(native_ts) = v.get("native_tool_search").and_then(Value::as_bool) {
            self.native_tool_search.store(native_ts, Ordering::Relaxed);
        }

        Ok(LiveChild {
            child,
            stdin,
            stdout,
        })
    }
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    static_system_prompt: &'a str,
    tools: &'a [carnot_core::ToolSpec],
    prefix_hash: &'a str,
    messages: Vec<&'a Message>,
    checkpoint_index: Option<usize>,
    cache_ttl: carnot_core::CacheTtl,
    effort: carnot_core::ReasoningEffort,
    max_output_tokens: u32,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireStreamEvent {
    TextDelta {
        delta: String,
    },
    ThinkingDelta {
        delta: String,
    },
    ToolCallReady {
        id: String,
        name: String,
        input: Value,
        #[serde(default)]
        invalid_json: Option<String>,
    },
    Done {
        message: Message,
        #[serde(default)]
        usage: Usage,
        #[serde(default)]
        stop_reason: StopReason,
    },
    Error {
        message: String,
        #[serde(default)]
        retryable: bool,
    },
}

impl Provider for SubprocessProvider {
    fn name(&self) -> &str {
        &self.provider_id
    }

    fn supports_native_tool_search(&self) -> bool {
        self.native_tool_search.load(Ordering::Relaxed)
    }

    fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<mpsc::Receiver<StreamEvent>>> {
        Box::pin(async move {
            let live_arc = Arc::clone(&self.live);
            let mut guard = live_arc.lock().await;

            // Ensure child is alive (respawn transparently if it died between turns).
            let need_spawn = match guard.as_mut() {
                Some(lc) => lc.child.try_wait().ok().flatten().is_some(),
                None => true,
            };
            if need_spawn {
                let lc = self.spawn_and_handshake().await?;
                *guard = Some(lc);
            }

            let req_id = format!("req-{}", REQ_SEQ.fetch_add(1, Ordering::Relaxed));
            let wire_req = WireRequest {
                model: &request.model,
                static_system_prompt: &request.static_system_prompt,
                tools: &request.tools,
                prefix_hash: &request.prefix_hash,
                messages: request.messages.iter().map(|m| m.as_ref()).collect(),
                checkpoint_index: request.checkpoint_index,
                cache_ttl: request.cache_ttl,
                effort: request.effort,
                max_output_tokens: request.max_output_tokens,
            };
            let payload = json!({
                "type": "stream",
                "request_id": req_id,
                "request": wire_req,
            });
            let mut line = serde_json::to_string(&payload)?;
            line.push('\n');

            // Take ownership of the live child while streaming, then return it to the slot on completion.
            let Some(mut lc) = guard.take() else {
                return Err(CarnotError::Provider {
                    provider: self.provider_id.clone(),
                    message: "subprocess provider slot unexpectedly empty".into(),
                    retryable: true,
                    retry_after_ms: None,
                });
            };
            drop(guard);

            if let Err(e) = lc.stdin.write_all(line.as_bytes()).await {
                let _ = lc.child.kill().await;
                return Err(CarnotError::Provider {
                    provider: self.provider_id.clone(),
                    message: format!("failed writing request to subprocess provider: {e}"),
                    retryable: true,
                    retry_after_ms: None,
                });
            }
            if let Err(e) = lc.stdin.flush().await {
                let _ = lc.child.kill().await;
                return Err(CarnotError::Provider {
                    provider: self.provider_id.clone(),
                    message: format!("failed flushing subprocess stdin: {e}"),
                    retryable: true,
                    retry_after_ms: None,
                });
            }

            let (tx, rx) = mpsc::channel(64);
            let cancel_kill_ms = self.config.cancel_kill_timeout_ms;
            let stderr_tail = Arc::clone(&self.stderr_tail);

            tokio::spawn(async move {
                let mut line_buf = String::new();

                loop {
                    line_buf.clear();
                    let read_res = tokio::select! {
                        _ = cancel.cancelled() => {
                            let cancel_msg = format!(
                                "{}\n",
                                json!({"type": "cancel", "request_id": req_id})
                            );
                            let _ = lc.stdin.write_all(cancel_msg.as_bytes()).await;
                            let _ = lc.stdin.flush().await;
                            // Wait briefly for child to acknowledge or terminate, then force-kill.
                            let _ = tokio::time::timeout(
                                Duration::from_millis(cancel_kill_ms),
                                lc.stdout.read_line(&mut line_buf),
                            ).await;
                            let _ = lc.child.kill().await;
                            let _ = tx.send(StreamEvent::Error {
                                message: "cancelled".into(),
                                retryable: false,
                            }).await;
                            return;
                        }
                        res = tokio::time::timeout(Duration::from_secs(120), lc.stdout.read_line(&mut line_buf)) => res,
                    };

                    let Ok(Ok(n)) = read_res else {
                        let _ = lc.child.kill().await;
                        let _ = tx
                            .send(StreamEvent::Error {
                                message: "subprocess provider read timeout/error".into(),
                                retryable: true,
                            })
                            .await;
                        return;
                    };
                    if n == 0 {
                        let stderr = stderr_tail.lock().await.join(" | ");
                        let _ = tx
                            .send(StreamEvent::Error {
                                message: format!(
                                    "subprocess provider EOF before done (stderr: {stderr})"
                                ),
                                retryable: true,
                            })
                            .await;
                        return;
                    }

                    let trimmed = line_buf.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let Ok(ev) = serde_json::from_str::<WireStreamEvent>(trimmed) else {
                        continue;
                    };
                    match ev {
                        WireStreamEvent::TextDelta { delta } => {
                            let _ = tx.send(StreamEvent::TextDelta(delta)).await;
                        }
                        WireStreamEvent::ThinkingDelta { delta } => {
                            let _ = tx.send(StreamEvent::ThinkingDelta(delta)).await;
                        }
                        WireStreamEvent::ToolCallReady {
                            id,
                            name,
                            input,
                            invalid_json,
                        } => {
                            let _ = tx
                                .send(StreamEvent::ToolCallReady {
                                    id,
                                    name,
                                    input,
                                    invalid_json,
                                })
                                .await;
                        }
                        WireStreamEvent::Done {
                            message,
                            usage,
                            stop_reason,
                        } => {
                            let _ = tx
                                .send(StreamEvent::Done {
                                    message,
                                    usage,
                                    stop_reason,
                                })
                                .await;
                            break;
                        }
                        WireStreamEvent::Error { message, retryable } => {
                            let _ = tx.send(StreamEvent::Error { message, retryable }).await;
                            break;
                        }
                    }
                }

                let mut g = live_arc.lock().await;
                *g = Some(lc);
            });

            Ok(rx)
        })
    }
}
