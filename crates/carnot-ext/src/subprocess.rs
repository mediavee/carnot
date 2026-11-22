//! Out-of-process JSON-RPC 2.0 Extension Host (write Carnot modules in Python, TS, Go, or Rust)
//! with Opus I3 negotiated hooks, lock-free multiplexed stdio I/O, bounded auto-restart on crash,
//! per-call timeouts, `before_compact` hook support, and fail-closed/fail-open policies.
//!
//! Protocol over newline-delimited JSON on stdio:
//!   Host -> Ext: `{"jsonrpc":"2.0","id":1,"method":"ext/init","params":{"protocolVersion":"carnot-ext/1"}}`
//!   Ext -> Host: `{"jsonrpc":"2.0","id":1,"result":{"tools":[...],"hooks":["before_tool","before_compact"],"timeout_ms":5000,"failure_policy":"fail_closed"}}`
//!   Host -> Ext: `ext/session_start`, `ext/before_tool`, `ext/after_tool`, `ext/turn_boundary`,
//!                `ext/before_compact`, `ext/tool_call`

use crate::{
    BoundaryAction, BoxFuture, CompactionHookAction, CompactionHookContext, Module,
    ToolGateDecision, TurnBoundarySnapshot,
};
use carnot_core::{CarnotConfig, CarnotError, Result, ToolOutput, ToolSpec};
use carnot_sandbox::apply_safe_env;
use carnot_tools::Tool;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

const DEFAULT_EXT_TIMEOUT_MS: u64 = 5_000;
const MAX_AUTO_RESTARTS: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailurePolicy {
    FailClosed,
    FailOpen,
}

/// Lock-free multiplexed JSON-RPC 2.0 stdio connection to one extension child process.
struct ExtConn {
    ext_name: String,
    write_tx: mpsc::Sender<Vec<u8>>,
    pending: Arc<StdMutex<HashMap<u64, oneshot::Sender<Result<Value>>>>>,
    closed: Arc<AtomicBool>,
    next_id: AtomicU64,
    timeout: Duration,
    _child: Mutex<Child>,
}

impl std::fmt::Debug for ExtConn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtConn")
            .field("ext_name", &self.ext_name)
            .field("closed", &self.closed.load(Ordering::Relaxed))
            .finish()
    }
}

impl ExtConn {
    async fn spawn_and_init(
        ext_name: &str,
        program: &str,
        args: &[String],
    ) -> Result<(Arc<Self>, Value)> {
        let mut cmd = Command::new(program);
        cmd.args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true);
        apply_safe_env(&mut cmd, &BTreeMap::new());
        let mut child = cmd.spawn()?;

        let mut stdin = child.stdin.take().ok_or_else(|| CarnotError::Extension {
            extension: ext_name.into(),
            message: "missing stdin".into(),
        })?;
        let stdout = child.stdout.take().ok_or_else(|| CarnotError::Extension {
            extension: ext_name.into(),
            message: "missing stdout".into(),
        })?;

        let (write_tx, mut write_rx) = mpsc::channel::<Vec<u8>>(256);
        let pending: Arc<StdMutex<HashMap<u64, oneshot::Sender<Result<Value>>>>> =
            Arc::new(StdMutex::new(HashMap::new()));
        let closed = Arc::new(AtomicBool::new(false));

        let closed_w = Arc::clone(&closed);
        tokio::spawn(async move {
            while let Some(bytes) = write_rx.recv().await {
                if stdin.write_all(&bytes).await.is_err() || stdin.flush().await.is_err() {
                    break;
                }
            }
            closed_w.store(true, Ordering::Release);
        });

        let pending_r = Arc::clone(&pending);
        let closed_r = Arc::clone(&closed);
        let name_r = ext_name.to_string();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            let mut buf = String::new();
            loop {
                buf.clear();
                match reader.read_line(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let Ok(v) = serde_json::from_str::<Value>(&buf) else {
                            continue;
                        };
                        if let Some(id) = v.get("id").and_then(Value::as_u64)
                            && let Some(tx) = pending_r.lock().ok().and_then(|mut m| m.remove(&id))
                        {
                            let res = if let Some(err) = v.get("error") {
                                Err(CarnotError::Extension {
                                    extension: name_r.clone(),
                                    message: err.to_string(),
                                })
                            } else {
                                Ok(v.get("result").cloned().unwrap_or(Value::Null))
                            };
                            let _ = tx.send(res);
                        }
                    }
                }
            }
            closed_r.store(true, Ordering::Release);
            if let Ok(mut m) = pending_r.lock() {
                for (_, tx) in m.drain() {
                    let _ = tx.send(Err(CarnotError::Extension {
                        extension: name_r.clone(),
                        message: "extension process closed stdout".into(),
                    }));
                }
            }
        });

        let mut conn = Self {
            ext_name: ext_name.to_string(),
            write_tx,
            pending,
            closed,
            next_id: AtomicU64::new(1),
            timeout: Duration::from_millis(DEFAULT_EXT_TIMEOUT_MS),
            _child: Mutex::new(child),
        };

        let init = conn
            .call_raw("ext/init", json!({"protocolVersion": "carnot-ext/1"}), None)
            .await?;

        if let Some(ms) = init.get("timeout_ms").and_then(Value::as_u64) {
            conn.timeout = Duration::from_millis(ms.clamp(100, 120_000));
        }
        Ok((Arc::new(conn), init))
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    async fn call_raw(
        &self,
        method: &str,
        params: Value,
        cancel: Option<&CancellationToken>,
    ) -> Result<Value> {
        if self.is_closed() {
            return Err(CarnotError::Extension {
                extension: self.ext_name.clone(),
                message: "extension process is closed".into(),
            });
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        if let Ok(mut m) = self.pending.lock() {
            m.insert(id, tx);
        }
        let req = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let mut line = serde_json::to_vec(&req)?;
        line.push(b'\n');

        if self.write_tx.send(line).await.is_err() {
            self.closed.store(true, Ordering::Release);
            if let Ok(mut m) = self.pending.lock() {
                m.remove(&id);
            }
            return Err(CarnotError::Extension {
                extension: self.ext_name.clone(),
                message: "extension process closed stdin".into(),
            });
        }

        let timed = async {
            match cancel {
                Some(tok) => {
                    tokio::select! {
                        _ = tok.cancelled() => Err(CarnotError::Cancelled),
                        r = tokio::time::timeout(self.timeout, rx) => match r {
                            Ok(Ok(val)) => val,
                            Ok(Err(_)) => Err(CarnotError::Extension {
                                extension: self.ext_name.clone(),
                                message: "extension process closed stdout".into(),
                            }),
                            Err(_) => Err(CarnotError::Extension {
                                extension: self.ext_name.clone(),
                                message: format!(
                                    "extension `{}` timed out on `{method}` after {}ms",
                                    self.ext_name,
                                    self.timeout.as_millis()
                                ),
                            }),
                        }
                    }
                }
                None => match tokio::time::timeout(self.timeout, rx).await {
                    Ok(Ok(val)) => val,
                    Ok(Err(_)) => Err(CarnotError::Extension {
                        extension: self.ext_name.clone(),
                        message: "extension process closed stdout".into(),
                    }),
                    Err(_) => Err(CarnotError::Extension {
                        extension: self.ext_name.clone(),
                        message: format!(
                            "extension `{}` timed out on `{method}` after {}ms",
                            self.ext_name,
                            self.timeout.as_millis()
                        ),
                    }),
                },
            }
        };

        let out = timed.await;
        if out.is_err()
            && let Ok(mut m) = self.pending.lock()
        {
            m.remove(&id);
        }
        out
    }
}

/// Auto-restarting supervisor around a multiplexed `ExtConn` (bounded to `MAX_AUTO_RESTARTS`).
#[derive(Debug)]
struct ExtSupervisor {
    name: String,
    program: String,
    args: Vec<String>,
    conn: Mutex<Option<Arc<ExtConn>>>,
    restarts: AtomicU32,
}

impl ExtSupervisor {
    async fn get_conn(&self) -> Result<Arc<ExtConn>> {
        let mut guard = self.conn.lock().await;
        if let Some(ref c) = *guard
            && !c.is_closed()
        {
            return Ok(Arc::clone(c));
        }
        let prev = self.restarts.load(Ordering::Relaxed);
        if prev >= MAX_AUTO_RESTARTS {
            return Err(CarnotError::Extension {
                extension: self.name.clone(),
                message: format!(
                    "extension `{}` crashed and exceeded max {} auto-restarts",
                    self.name, MAX_AUTO_RESTARTS
                ),
            });
        }
        self.restarts.fetch_add(1, Ordering::Relaxed);
        tracing::warn!(ext = %self.name, attempt = prev + 1, "restarting crashed subprocess extension");
        let (new_conn, _) = ExtConn::spawn_and_init(&self.name, &self.program, &self.args).await?;
        *guard = Some(Arc::clone(&new_conn));
        Ok(new_conn)
    }

    async fn call(
        &self,
        method: &str,
        params: Value,
        cancel: Option<&CancellationToken>,
    ) -> Result<Value> {
        let conn = self.get_conn().await?;
        match conn.call_raw(method, params.clone(), cancel).await {
            Ok(v) => Ok(v),
            Err(CarnotError::Cancelled) => Err(CarnotError::Cancelled),
            Err(first_err) if conn.is_closed() => {
                // Process crashed mid-call: transparently respawn once and retry!
                let Ok(fresh) = self.get_conn().await else {
                    return Err(first_err);
                };
                fresh.call_raw(method, params, cancel).await
            }
            Err(e) => Err(e),
        }
    }
}

/// Out-of-process JSON-RPC 2.0 extension implementing the exact same `Module` trait as Rust modules.
#[derive(Debug)]
pub struct SubprocessExtension {
    name: String,
    supervisor: Arc<ExtSupervisor>,
    declared_tools: Vec<ToolSpec>,
    subscribed_hooks: HashSet<String>,
    gate_policy: FailurePolicy,
}

impl SubprocessExtension {
    pub async fn spawn(name: String, program: &str, args: &[String]) -> Result<Self> {
        let (conn, init) = ExtConn::spawn_and_init(&name, program, args).await?;

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

        let supervisor = Arc::new(ExtSupervisor {
            name: name.clone(),
            program: program.to_string(),
            args: args.to_vec(),
            conn: Mutex::new(Some(conn)),
            restarts: AtomicU32::new(0),
        });

        Ok(Self {
            name,
            supervisor,
            declared_tools,
            subscribed_hooks,
            gate_policy,
        })
    }

    pub fn has_hook(&self, hook: &str) -> bool {
        self.subscribed_hooks.contains(hook)
    }

    /// Number of automatic crash-recovery restarts performed by this extension supervisor.
    pub fn restart_count(&self) -> u32 {
        self.supervisor.restarts.load(Ordering::Relaxed)
    }
}

#[derive(Debug)]
struct SubprocessTool {
    spec: ToolSpec,
    supervisor: Arc<ExtSupervisor>,
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
            let res = self
                .supervisor
                .call(
                    "ext/tool_call",
                    json!({"name": self.spec.name, "input": input}),
                    Some(&cancel),
                )
                .await?;
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
                    spec: spec.clone(),
                    supervisor: self.supervisor.clone(),
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
                .supervisor
                .call("ext/session_start", json!({"workspace_root": cwd}), None)
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
                .supervisor
                .call("ext/before_tool", json!({"tool": tn, "input": inp}), None)
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
                .supervisor
                .call(
                    "ext/after_tool",
                    json!({"tool": tn, "input": inp, "is_error": is_error, "text": out_text}),
                    None,
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
                .supervisor
                .call("ext/turn_boundary", serde_json::to_value(&snap)?, None)
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

    fn before_compact<'a>(
        &'a self,
        ctx: &'a CompactionHookContext,
    ) -> BoxFuture<'a, Result<CompactionHookAction>> {
        if !self.has_hook("before_compact") {
            return Box::pin(async { Ok(CompactionHookAction::Continue) });
        }
        let payload = ctx.clone();
        Box::pin(async move {
            match self
                .supervisor
                .call("ext/before_compact", serde_json::to_value(&payload)?, None)
                .await
            {
                Ok(res) => match res.get("action").and_then(Value::as_str) {
                    Some("add_focus") => {
                        let focus = res
                            .get("focus")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        if focus.is_empty() {
                            Ok(CompactionHookAction::Continue)
                        } else {
                            Ok(CompactionHookAction::AddFocus(focus))
                        }
                    }
                    Some("cancel") => Ok(CompactionHookAction::Cancel {
                        reason: res
                            .get("reason")
                            .and_then(Value::as_str)
                            .unwrap_or("cancelled by extension")
                            .to_string(),
                    }),
                    _ => Ok(CompactionHookAction::Continue),
                },
                Err(e) => {
                    tracing::warn!(ext = %self.name, error = %e, "ext/before_compact failed (fail-open)");
                    Ok(CompactionHookAction::Continue)
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn subprocess_extension_multiplexes_recovers_from_crash_and_runs_before_compact() {
        let dir = std::env::temp_dir().join(format!("carnot-ext-mux-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let marker = dir.join("crashed_once");
        let script = dir.join("mock_ext.py");
        std::fs::write(
            &script,
            format!(
                r#"import sys, json, threading, time, os
lock = threading.Lock()
MARKER = {marker:?}
def send(obj):
    with lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()
def handle(req):
    rid = req.get("id")
    m = req.get("method")
    p = req.get("params") or {{}}
    if m == "ext/init":
        send({{"jsonrpc":"2.0","id":rid,"result":{{
            "hooks":["before_compact"],
            "timeout_ms":5000,
            "tools":[{{
                "name":"ext_sleep",
                "description":"Sleeps 80ms",
                "input_schema":{{"type":"object","properties":{{"n":{{"type":"integer"}},"crash_once":{{"type":"boolean"}}}}}},
                "defer_loading":True
            }}]
        }}}})
    elif m == "ext/tool_call":
        inp = p.get("input") or {{}}
        if inp.get("crash_once") and not os.path.exists(MARKER):
            open(MARKER, "w").write("1")
            os._exit(1)
        time.sleep(0.08)
        send({{"jsonrpc":"2.0","id":rid,"result":{{"text":f"ok:{{inp.get('n',0)}}","is_error":False}}}})
    elif m == "ext/before_compact":
        send({{"jsonrpc":"2.0","id":rid,"result":{{"action":"add_focus","focus":"Preserve DB migration IDs"}}}})
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    threading.Thread(target=handle, args=(req,), daemon=True).start()
"#,
                marker = marker.to_string_lossy()
            ),
        )
        .expect("write py");

        let ext = SubprocessExtension::spawn(
            "mock".into(),
            "python3",
            &[script.to_string_lossy().to_string()],
        )
        .await
        .expect("spawn ext");

        // 1. Verify `before_compact` hook
        let act = ext
            .before_compact(&CompactionHookContext {
                mode: "manual".into(),
                tokens_before: 50_000,
                message_count: 14,
                custom_focus: None,
            })
            .await
            .expect("before_compact");
        assert_eq!(
            act,
            CompactionHookAction::AddFocus("Preserve DB migration IDs".into())
        );

        // 2. Verify 8-way lock-free multiplexing finishes well under serial 8*80ms = 640ms
        let tool = ext.tools().into_iter().next().expect("tool");
        let t0 = Instant::now();
        let mut futs = Vec::new();
        for i in 0..8 {
            let t = tool.clone();
            futs.push(tokio::spawn(async move {
                t.execute(json!({"n": i}), CancellationToken::new()).await
            }));
        }
        for (i, h) in futs.into_iter().enumerate() {
            assert_eq!(
                h.await.expect("join").expect("exec").text_only(),
                format!("ok:{i}")
            );
        }
        assert!(
            t0.elapsed().as_millis() < 450,
            "expected multiplexed 8x80ms < 450ms, got {}ms",
            t0.elapsed().as_millis()
        );

        // 3. Verify transparent crash recovery + auto-restart!
        let recovered = tool
            .execute(
                json!({"n": 99, "crash_once": true}),
                CancellationToken::new(),
            )
            .await
            .expect("recovered after crash");
        assert_eq!(recovered.text_only(), "ok:99");
        assert_eq!(ext.restart_count(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
