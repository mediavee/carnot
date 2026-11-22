//! Out-of-process JSON-RPC 2.0 Extension Host (write Carnot modules in Python, TS, Go, or Rust)
//! with Opus I3 negotiated hooks, per-call timeouts, and fail-closed/fail-open policies.
//!
//! Protocol over newline-delimited JSON on stdio:
//!   Host -> Ext: `{"jsonrpc":"2.0","id":1,"method":"ext/init","params":{"protocolVersion":"carnot-ext/1"}}`
//!   Ext -> Host: `{"jsonrpc":"2.0","id":1,"result":{"tools":[...],"hooks":["before_tool"],"timeout_ms":5000,"failure_policy":"fail_closed"}}`
//!   Host -> Ext: `ext/session_start`, `ext/before_tool`, `ext/after_tool`, `ext/turn_boundary`, `ext/tool_call`

use crate::{BoundaryAction, BoxFuture, Module, ToolGateDecision, TurnBoundarySnapshot};
use carnot_core::{CarnotConfig, CarnotError, Result, ToolOutput, ToolSpec};
use carnot_sandbox::apply_safe_env;
use carnot_tools::Tool;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

const DEFAULT_EXT_TIMEOUT_MS: u64 = 5_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailurePolicy {
    FailClosed,
    FailOpen,
}

#[derive(Debug)]
struct ExtProcess {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    timeout: Duration,
}

impl ExtProcess {
    async fn call(&mut self, ext_name: &str, method: &str, params: Value) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        let req = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let mut line = serde_json::to_string(&req)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;

        let read_fut = async {
            let mut buf = String::new();
            loop {
                buf.clear();
                let n = self.stdout.read_line(&mut buf).await?;
                if n == 0 {
                    return Err(CarnotError::Extension {
                        extension: ext_name.into(),
                        message: "extension process closed stdout".into(),
                    });
                }
                let Ok(v) = serde_json::from_str::<Value>(&buf) else {
                    continue;
                };
                if v.get("id").and_then(Value::as_u64) == Some(id) {
                    if let Some(err) = v.get("error") {
                        return Err(CarnotError::Extension {
                            extension: ext_name.into(),
                            message: err.to_string(),
                        });
                    }
                    return Ok(v.get("result").cloned().unwrap_or(Value::Null));
                }
            }
        };

        tokio::time::timeout(self.timeout, read_fut)
            .await
            .map_err(|_| CarnotError::Extension {
                extension: ext_name.into(),
                message: format!(
                    "extension `{ext_name}` timed out on `{method}` after {}ms",
                    self.timeout.as_millis()
                ),
            })?
    }
}

/// Out-of-process JSON-RPC 2.0 extension implementing the exact same `Module` trait as Rust modules.
#[derive(Debug)]
pub struct SubprocessExtension {
    name: String,
    proc: Arc<Mutex<ExtProcess>>,
    declared_tools: Vec<ToolSpec>,
    subscribed_hooks: HashSet<String>,
    gate_policy: FailurePolicy,
}

impl SubprocessExtension {
    pub async fn spawn(name: String, program: &str, args: &[String]) -> Result<Self> {
        let mut cmd = Command::new(program);
        cmd.args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true);
        apply_safe_env(&mut cmd, &BTreeMap::new());
        let mut child = cmd.spawn()?;

        let stdin = child.stdin.take().ok_or_else(|| CarnotError::Extension {
            extension: name.clone(),
            message: "missing stdin".into(),
        })?;
        let stdout = child.stdout.take().ok_or_else(|| CarnotError::Extension {
            extension: name.clone(),
            message: "missing stdout".into(),
        })?;

        let mut ep = ExtProcess {
            _child: child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 0,
            timeout: Duration::from_millis(DEFAULT_EXT_TIMEOUT_MS),
        };

        let init = ep
            .call(
                &name,
                "ext/init",
                json!({"protocolVersion": "carnot-ext/1"}),
            )
            .await?;

        if let Some(ms) = init.get("timeout_ms").and_then(Value::as_u64) {
            ep.timeout = Duration::from_millis(ms.clamp(100, 120_000));
        }

        let declared_tools: Vec<ToolSpec> = init
            .get("tools")
            .cloned()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default();

        // Opus I3: negotiate subscribed hooks so unused hooks incur ZERO IPC!
        let subscribed_hooks: HashSet<String> = match init.get("hooks").and_then(Value::as_array) {
            Some(arr) => arr
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect(),
            None => [
                "session_start",
                "before_tool",
                "after_tool",
                "turn_boundary",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
        };

        let gate_policy = match init.get("failure_policy").and_then(Value::as_str) {
            Some("fail_open") => FailurePolicy::FailOpen,
            _ => FailurePolicy::FailClosed,
        };

        Ok(Self {
            name,
            proc: Arc::new(Mutex::new(ep)),
            declared_tools,
            subscribed_hooks,
            gate_policy,
        })
    }

    pub fn has_hook(&self, hook: &str) -> bool {
        self.subscribed_hooks.contains(hook)
    }
}

#[derive(Debug)]
struct SubprocessTool {
    ext_name: String,
    spec: ToolSpec,
    proc: Arc<Mutex<ExtProcess>>,
}

impl Tool for SubprocessTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> carnot_tools::BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let res = tokio::select! {
                _ = cancel.cancelled() => return Err(CarnotError::Cancelled),
                r = async {
                    self.proc
                        .lock()
                        .await
                        .call(
                            &self.ext_name,
                            "ext/tool_call",
                            json!({"name": self.spec.name, "input": input}),
                        )
                        .await
                } => r?,
            };
            let text = res
                .get("text")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| res.to_string());
            let is_error = res
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            Ok(ToolOutput {
                blocks: vec![carnot_core::ToolResultBlock::text(text)],
                is_error,
                details: res.get("details").cloned().unwrap_or(Value::Null),
            })
        })
    }
}

impl Module for SubprocessExtension {
    fn name(&self) -> &str {
        &self.name
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.declared_tools
            .iter()
            .map(|spec| {
                Arc::new(SubprocessTool {
                    ext_name: self.name.clone(),
                    spec: spec.clone(),
                    proc: self.proc.clone(),
                }) as Arc<dyn Tool>
            })
            .collect()
    }

    fn on_session_start(&self, config: &CarnotConfig) -> BoxFuture<'_, Result<Option<String>>> {
        if !self.has_hook("session_start") {
            return Box::pin(async { Ok(None) });
        }
        let cwd = config.workspace_root.display().to_string();
        Box::pin(async move {
            match self
                .proc
                .lock()
                .await
                .call(
                    &self.name,
                    "ext/session_start",
                    json!({"workspace_root": cwd}),
                )
                .await
            {
                Ok(res) => Ok(res
                    .get("context")
                    .and_then(Value::as_str)
                    .map(str::to_string)),
                Err(e) => {
                    tracing::warn!(ext = %self.name, error = %e, "ext/session_start failed (fail-open)");
                    Ok(None)
                }
            }
        })
    }

    fn before_tool_call(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> BoxFuture<'_, Result<ToolGateDecision>> {
        if !self.has_hook("before_tool") {
            return Box::pin(async { Ok(ToolGateDecision::Allow) });
        }
        let tn = tool_name.to_string();
        let inp = input.clone();
        let policy = self.gate_policy;
        Box::pin(async move {
            match self
                .proc
                .lock()
                .await
                .call(
                    &self.name,
                    "ext/before_tool",
                    json!({"tool": tn, "input": inp}),
                )
                .await
            {
                Ok(res) => match res.get("decision").and_then(Value::as_str) {
                    Some("allow") => Ok(ToolGateDecision::Allow),
                    Some("deny") => Ok(ToolGateDecision::Deny {
                        reason: res
                            .get("reason")
                            .and_then(Value::as_str)
                            .unwrap_or("denied by extension")
                            .to_string(),
                    }),
                    Some("rewrite") => Ok(ToolGateDecision::RewriteInput(
                        res.get("input").cloned().unwrap_or(inp),
                    )),
                    other => match policy {
                        FailurePolicy::FailClosed => Ok(ToolGateDecision::Deny {
                            reason: format!(
                                "extension `{}` returned unknown decision `{other:?}` (fail-closed)",
                                self.name
                            ),
                        }),
                        FailurePolicy::FailOpen => Ok(ToolGateDecision::Allow),
                    },
                },
                Err(e) => match policy {
                    FailurePolicy::FailClosed => Ok(ToolGateDecision::Deny {
                        reason: format!("extension `{}` gate error (fail-closed): {e}", self.name),
                    }),
                    FailurePolicy::FailOpen => {
                        tracing::warn!(ext = %self.name, error = %e, "ext/before_tool failed (fail-open)");
                        Ok(ToolGateDecision::Allow)
                    }
                },
            }
        })
    }

    fn after_tool_result<'a>(
        &'a self,
        tool_name: &'a str,
        input: &'a Value,
        output: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>> {
        if !self.has_hook("after_tool") {
            return Box::pin(async { Ok(()) });
        }
        let tn = tool_name.to_string();
        let inp = input.clone();
        let out_text = output.text_only();
        let is_error = output.is_error;
        Box::pin(async move {
            if let Err(e) = self
                .proc
                .lock()
                .await
                .call(
                    &self.name,
                    "ext/after_tool",
                    json!({"tool": tn, "input": inp, "is_error": is_error, "text": out_text}),
                )
                .await
            {
                tracing::warn!(ext = %self.name, error = %e, "ext/after_tool failed (fail-open)");
            }
            Ok(())
        })
    }

    fn on_turn_boundary<'a>(
        &'a self,
        snapshot: &'a TurnBoundarySnapshot,
    ) -> BoxFuture<'a, Result<BoundaryAction>> {
        if !self.has_hook("turn_boundary") {
            return Box::pin(async { Ok(BoundaryAction::Continue) });
        }
        let snap = snapshot.clone();
        Box::pin(async move {
            match self
                .proc
                .lock()
                .await
                .call(
                    &self.name,
                    "ext/turn_boundary",
                    serde_json::to_value(&snap)?,
                )
                .await
            {
                Ok(res) => match res.get("action").and_then(Value::as_str) {
                    Some("advisory") => Ok(BoundaryAction::AppendAdvisory {
                        signal: res
                            .get("signal")
                            .and_then(Value::as_str)
                            .unwrap_or("ext")
                            .to_string(),
                        message: res
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                    }),
                    Some("halt") => Ok(BoundaryAction::Halt {
                        reason: res
                            .get("reason")
                            .and_then(Value::as_str)
                            .unwrap_or("halted by extension")
                            .to_string(),
                    }),
                    _ => Ok(BoundaryAction::Continue),
                },
                Err(e) => {
                    tracing::warn!(ext = %self.name, error = %e, "ext/turn_boundary failed (fail-open)");
                    Ok(BoundaryAction::Continue)
                }
            }
        })
    }
}
