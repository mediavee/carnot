//! General-purpose bidirectional Host-RPC Extension bridge (`carnot --mode rpc --host-extension`).
//!
//! Enables any parent host process (chat bot, IDE, orchestrator, SDK) to register custom tools,
//! static system prompt sections, and lifecycle hooks (`gate`, `tool_result`, `before_prompt`,
//! `turn_end`, `before_compact`, `select_effort`) over the SAME stdio JSON-RPC 2.0 stream as the
//! RPC control plane — zero extra ports, zero auth tokens, and 100% frozen before Turn 1 for
//! BLAKE3 prompt-cache invariance.

use carnot_core::{
    CarnotError, ReasoningEffort, Result, ToolOutput, ToolResultBlock, ToolSpec, canonicalize_json,
};
use carnot_ext::{
    BoundaryAction, BoxFuture, CompactionHookAction, CompactionHookContext, Module,
    PromptInjection, ToolGateDecision, TurnBoundarySnapshot,
};
use carnot_tools::{BoxFuture as ToolBoxFuture, Tool};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, oneshot};
use tokio_util::sync::CancellationToken;

fn default_obj_schema() -> Value {
    json!({"type": "object", "properties": {}})
}

fn default_host_timeout_ms() -> u64 {
    300_000
}

/// Host-declared external tool registered at session startup via `host/register`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostToolDecl {
    pub name: String,
    pub description: String,
    #[serde(default = "default_obj_schema")]
    pub input_schema: Value,
    #[serde(default)]
    pub defer_loading: bool,
}

/// Handshake payload sent by a host in `host/register` when `--host-extension` is enabled.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HostRegistration {
    #[serde(default)]
    pub tools: Vec<HostToolDecl>,
    #[serde(default)]
    pub system_prompt_append: Option<String>,
    /// Enabled host lifecycle hooks: `"gate"`, `"tool_result"`, `"before_prompt"`, `"turn_end"`,
    /// `"before_compact"`, `"select_effort"`.
    #[serde(default)]
    pub hooks: Vec<String>,
    /// Per-callback timeout in milliseconds (default 300s to allow interactive approval gates).
    #[serde(default = "default_host_timeout_ms")]
    pub timeout_ms: u64,
}

/// Shared bidirectional JSON-RPC 2.0 writer + pending-response router for engine -> host requests.
pub struct HostRpcBridge {
    stdout: Arc<Mutex<tokio::io::Stdout>>,
    next_id: AtomicU64,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Result<Value>>>>>,
    closed: AtomicBool,
    timeout: Duration,
}

impl std::fmt::Debug for HostRpcBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostRpcBridge")
            .field("timeout_ms", &self.timeout.as_millis())
            .field("closed", &self.closed.load(Ordering::Acquire))
            .finish()
    }
}

/// RAII drop guard ensuring an abandoned/cancelled/timed-out host RPC call removes its entry from
/// `pending` and emits a best-effort `host/cancel` notification so the parent host can cancel work.
struct PendingCallGuard {
    id: String,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Result<Value>>>>>,
    stdout: Arc<Mutex<tokio::io::Stdout>>,
    armed: bool,
}

impl PendingCallGuard {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PendingCallGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let id = self.id.clone();
        let pending = Arc::clone(&self.pending);
        let stdout = Arc::clone(&self.stdout);
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let was_pending = pending.lock().await.remove(&id).is_some();
                if was_pending {
                    let frame = json!({
                        "jsonrpc": "2.0",
                        "method": "host/cancel",
                        "params": { "id": id },
                    });
                    let mut s = serde_json::to_string(&frame).unwrap_or_else(|_| "{}".into());
                    s.push('\n');
                    let mut w = stdout.lock().await;
                    let _ = w.write_all(s.as_bytes()).await;
                    let _ = w.flush().await;
                }
            });
        }
    }
}

impl HostRpcBridge {
    pub fn new(stdout: Arc<Mutex<tokio::io::Stdout>>, timeout: Duration) -> Self {
        Self {
            stdout,
            next_id: AtomicU64::new(1),
            pending: Arc::new(Mutex::new(HashMap::new())),
            closed: AtomicBool::new(false),
            timeout,
        }
    }

    /// Sends a JSON-RPC 2.0 request from engine to host and awaits its correlated response.
    pub async fn call(&self, method: &str, params: Value) -> Result<Value> {
        self.call_cancellable(method, params, CancellationToken::new())
            .await
    }

    /// Sends a JSON-RPC 2.0 request to the host, honouring `cancel` and emitting `host/cancel`
    /// if cancelled, dropped, or timed out (zero pending-map leaks).
    pub async fn call_cancellable(
        &self,
        method: &str,
        params: Value,
        cancel: CancellationToken,
    ) -> Result<Value> {
        if self.closed.load(Ordering::Acquire) {
            return Err(CarnotError::Extension {
                extension: "host-rpc".into(),
                message: format!("host bridge closed before {method}"),
            });
        }
        if cancel.is_cancelled() {
            return Err(CarnotError::Cancelled);
        }

        let id = format!("hreq-{}", self.next_id.fetch_add(1, Ordering::Relaxed));
        let (tx, rx) = oneshot::channel();
        {
            let mut map = self.pending.lock().await;
            if self.closed.load(Ordering::Acquire) {
                return Err(CarnotError::Extension {
                    extension: "host-rpc".into(),
                    message: format!("host bridge closed before {method}"),
                });
            }
            map.insert(id.clone(), tx);
        }

        let mut guard = PendingCallGuard {
            id: id.clone(),
            pending: Arc::clone(&self.pending),
            stdout: Arc::clone(&self.stdout),
            armed: true,
        };

        let frame = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let write_res = {
            let mut s = serde_json::to_string(&frame).unwrap_or_else(|_| "{}".into());
            s.push('\n');
            let mut w = self.stdout.lock().await;
            match w.write_all(s.as_bytes()).await {
                Ok(()) => w.flush().await,
                Err(e) => Err(e),
            }
        };
        if let Err(e) = write_res {
            guard.disarm();
            self.pending.lock().await.remove(&id);
            return Err(CarnotError::Extension {
                extension: "host-rpc".into(),
                message: format!("failed writing {method} to host stdout: {e}"),
            });
        }

        let outcome = tokio::select! {
            _ = cancel.cancelled() => Err(CarnotError::Cancelled),
            res = tokio::time::timeout(self.timeout, rx) => match res {
                Ok(Ok(inner)) => {
                    guard.disarm();
                    inner
                }
                Ok(Err(_)) => {
                    guard.disarm();
                    Err(CarnotError::Extension {
                        extension: "host-rpc".into(),
                        message: format!("host channel closed while awaiting {method}"),
                    })
                }
                Err(_) => Err(CarnotError::Extension {
                    extension: "host-rpc".into(),
                    message: format!("host callback `{method}` timed out"),
                }),
            }
        };
        outcome
    }

    /// Routes an incoming JSON-RPC response frame (no `"method"` field) to its waiting caller.
    /// Returns `true` if `id` matched a pending engine -> host callback.
    pub async fn deliver_response(&self, frame: &Value) -> bool {
        let Some(id) = frame.get("id").and_then(Value::as_str) else {
            return false;
        };
        let Some(tx) = self.pending.lock().await.remove(id) else {
            return false;
        };
        let res = if let Some(err) = frame.get("error") {
            let msg = err
                .get("message")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .unwrap_or_else(|| err.to_string());
            Err(CarnotError::Extension {
                extension: "host-rpc".into(),
                message: msg,
            })
        } else {
            Ok(frame.get("result").cloned().unwrap_or(Value::Null))
        };
        let _ = tx.send(res);
        true
    }

    /// Marks the bridge closed and fails all pending engine -> host calls cleanly on EOF/shutdown
    /// so no existing or future callback can hang.
    pub async fn fail_all(&self, reason: &str) {
        self.closed.store(true, Ordering::Release);
        let mut map = self.pending.lock().await;
        for (_, tx) in map.drain() {
            let _ = tx.send(Err(CarnotError::Extension {
                extension: "host-rpc".into(),
                message: reason.to_string(),
            }));
        }
    }
}

/// External tool executed in the parent host via `host/tool_call`.
#[derive(Debug)]
pub struct HostRpcTool {
    spec: ToolSpec,
    bridge: Arc<HostRpcBridge>,
}

impl HostRpcTool {
    pub fn new(decl: HostToolDecl, bridge: Arc<HostRpcBridge>) -> Self {
        Self {
            spec: ToolSpec {
                name: decl.name,
                description: decl.description,
                input_schema: canonicalize_json(&decl.input_schema),
                defer_loading: decl.defer_loading,
            },
            bridge,
        }
    }

    fn run_with_id(
        &self,
        call_id: &str,
        input: Value,
        cancel: CancellationToken,
    ) -> ToolBoxFuture<'_, Result<ToolOutput>> {
        let cid = call_id.to_string();
        let name = self.spec.name.clone();
        let bridge = Arc::clone(&self.bridge);
        Box::pin(async move {
            let res = bridge
                .call_cancellable(
                    "host/tool_call",
                    json!({
                        "call_id": cid,
                        "name": name,
                        "input": input,
                    }),
                    cancel,
                )
                .await;
            match res {
                Ok(val) => Ok(parse_host_tool_output(&val)),
                Err(CarnotError::Cancelled) => Ok(ToolOutput::err_text(format!(
                    "host tool `{name}` cancelled"
                ))),
                Err(e) => Ok(ToolOutput::err_text(format!(
                    "host tool `{name}` failed: {e}"
                ))),
            }
        })
    }
}

impl Tool for HostRpcTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> ToolBoxFuture<'_, Result<ToolOutput>> {
        self.run_with_id("", input, cancel)
    }

    fn execute_with_id<'a>(
        &'a self,
        call_id: &'a str,
        input: Value,
        cancel: CancellationToken,
    ) -> ToolBoxFuture<'a, Result<ToolOutput>> {
        self.run_with_id(call_id, input, cancel)
    }
}

fn parse_host_tool_output(val: &Value) -> ToolOutput {
    let is_error = val
        .get("is_error")
        .or_else(|| val.get("isError"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let details = val.get("details").cloned().unwrap_or(Value::Null);

    let mut content: Vec<ToolResultBlock> = Vec::new();
    if let Some(arr) = val.get("content").and_then(Value::as_array) {
        for item in arr {
            let btype = item.get("type").and_then(Value::as_str).unwrap_or("text");
            match btype {
                "image" => {
                    let media_type = item
                        .get("media_type")
                        .or_else(|| item.get("mimeType"))
                        .and_then(Value::as_str)
                        .unwrap_or("image/png")
                        .to_string();
                    if let Some(data) = item
                        .get("data_base64")
                        .or_else(|| item.get("data"))
                        .and_then(Value::as_str)
                        && !data.is_empty()
                    {
                        content.push(ToolResultBlock::Image {
                            media_type,
                            data_base64: data.to_string(),
                        });
                    }
                }
                _ => {
                    if let Some(text) = item.get("text").and_then(Value::as_str) {
                        content.push(ToolResultBlock::Text {
                            text: text.to_string(),
                        });
                    }
                }
            }
        }
    } else if let Some(text) = val
        .get("content")
        .and_then(Value::as_str)
        .or_else(|| val.get("text").and_then(Value::as_str))
    {
        content.push(ToolResultBlock::Text {
            text: text.to_string(),
        });
    }

    if content.is_empty() {
        content.push(ToolResultBlock::Text {
            text: String::new(),
        });
    }

    ToolOutput {
        blocks: content,
        is_error,
        details,
    }
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

/// Carnot `Module` backed by bidirectional JSON-RPC callbacks into the parent host process.
#[derive(Debug)]
pub struct HostRpcModule {
    bridge: Arc<HostRpcBridge>,
    tools: Vec<Arc<dyn Tool>>,
    hooks: HashSet<String>,
}

impl HostRpcModule {
    pub fn from_registration(reg: HostRegistration, bridge: Arc<HostRpcBridge>) -> Self {
        let tools: Vec<Arc<dyn Tool>> = reg
            .tools
            .into_iter()
            .map(|decl| Arc::new(HostRpcTool::new(decl, Arc::clone(&bridge))) as Arc<dyn Tool>)
            .collect();
        let hooks: HashSet<String> = reg
            .hooks
            .into_iter()
            .map(|h| h.trim().to_ascii_lowercase())
            .collect();
        Self {
            bridge,
            tools,
            hooks,
        }
    }

    fn has_hook(&self, name: &str) -> bool {
        self.hooks.contains(name)
    }

    fn run_gate(
        &self,
        call_id: &str,
        tool_name: &str,
        input: &Value,
    ) -> BoxFuture<'_, Result<ToolGateDecision>> {
        let cid = call_id.to_string();
        let name = tool_name.to_string();
        let args = input.clone();
        Box::pin(async move {
            if !self.has_hook("gate") {
                return Ok(ToolGateDecision::Allow);
            }
            let val = match self
                .bridge
                .call(
                    "host/gate",
                    json!({ "call_id": cid, "tool": name, "input": args }),
                )
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    // Fail closed: an unreachable/timed-out host gate MUST NOT grant execution.
                    return Ok(ToolGateDecision::Deny {
                        reason: format!("host gate unavailable: {e}"),
                    });
                }
            };
            let decision = val
                .get("decision")
                .and_then(Value::as_str)
                .unwrap_or("deny")
                .to_ascii_lowercase();
            match decision.as_str() {
                "allow" => {
                    if let Some(updated) = val.get("updated_input")
                        && !updated.is_null()
                    {
                        Ok(ToolGateDecision::RewriteInput(updated.clone()))
                    } else {
                        Ok(ToolGateDecision::Allow)
                    }
                }
                _ => {
                    let reason = val
                        .get("reason")
                        .and_then(Value::as_str)
                        .unwrap_or("blocked by host gate")
                        .to_string();
                    Ok(ToolGateDecision::Deny { reason })
                }
            }
        })
    }
}

impl Module for HostRpcModule {
    fn name(&self) -> &str {
        "host_rpc"
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    fn before_prompt<'a>(
        &'a self,
        prompt_text: &'a str,
    ) -> BoxFuture<'a, Result<Vec<PromptInjection>>> {
        Box::pin(async move {
            if !self.has_hook("before_prompt") {
                return Ok(Vec::new());
            }
            let val = match self
                .bridge
                .call("host/before_prompt", json!({ "prompt": prompt_text }))
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("host/before_prompt failed: {e}");
                    return Ok(Vec::new());
                }
            };
            let mut out = Vec::new();
            if let Some(arr) = val.get("injections").and_then(Value::as_array) {
                for item in arr {
                    if let Some(text) = item.get("text").and_then(Value::as_str)
                        && !text.trim().is_empty()
                    {
                        out.push(PromptInjection {
                            text: text.to_string(),
                            ephemeral: item
                                .get("ephemeral")
                                .and_then(Value::as_bool)
                                .unwrap_or(false),
                        });
                    }
                }
            }
            Ok(out)
        })
    }

    fn select_effort(&self) -> BoxFuture<'_, Result<Option<ReasoningEffort>>> {
        Box::pin(async move {
            if !self.has_hook("select_effort") {
                return Ok(None);
            }
            let val = match self.bridge.call("host/select_effort", json!({})).await {
                Ok(v) => v,
                Err(_) => return Ok(None),
            };
            Ok(val
                .get("effort")
                .and_then(Value::as_str)
                .and_then(parse_effort))
        })
    }

    fn before_tool_call(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> BoxFuture<'_, Result<ToolGateDecision>> {
        self.run_gate("", tool_name, input)
    }

    fn before_tool_call_with_id<'a>(
        &'a self,
        call_id: &'a str,
        tool_name: &'a str,
        input: &'a Value,
    ) -> BoxFuture<'a, Result<ToolGateDecision>> {
        self.run_gate(call_id, tool_name, input)
    }

    fn after_tool_result<'a>(
        &'a self,
        tool_name: &'a str,
        input: &'a Value,
        output: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if !self.has_hook("tool_result") {
                return Ok(());
            }
            let payload = json!({
                "tool": tool_name,
                "input": input,
                "output": {
                    "content": &output.blocks,
                    "is_error": output.is_error,
                    "details": &output.details,
                },
            });
            let Ok(val) = self.bridge.call("host/tool_result", payload).await else {
                return Ok(());
            };
            if val.get("replace").and_then(Value::as_bool).unwrap_or(false) {
                *output = parse_host_tool_output(&val);
            } else if let Some(extra) = val.get("append_text").and_then(Value::as_str)
                && !extra.is_empty()
            {
                output.blocks.push(ToolResultBlock::Text {
                    text: extra.to_string(),
                });
            }
            Ok(())
        })
    }

    fn on_turn_boundary<'a>(
        &'a self,
        snapshot: &'a TurnBoundarySnapshot,
    ) -> BoxFuture<'a, Result<BoundaryAction>> {
        Box::pin(async move {
            if !self.has_hook("turn_end") {
                return Ok(BoundaryAction::Continue);
            }
            let Ok(val) = self.bridge.call("host/turn_end", json!(snapshot)).await else {
                return Ok(BoundaryAction::Continue);
            };
            let action = val
                .get("action")
                .and_then(Value::as_str)
                .unwrap_or("continue")
                .to_ascii_lowercase();
            match action.as_str() {
                "append_advisory" => {
                    let signal = val
                        .get("signal")
                        .and_then(Value::as_str)
                        .unwrap_or("host_note")
                        .to_string();
                    let message = val
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    if message.is_empty() {
                        Ok(BoundaryAction::Continue)
                    } else {
                        Ok(BoundaryAction::AppendAdvisory { signal, message })
                    }
                }
                "continue_run" => {
                    let signal = val
                        .get("signal")
                        .and_then(Value::as_str)
                        .unwrap_or("host_continue")
                        .to_string();
                    let message = val
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    Ok(BoundaryAction::ContinueRun { signal, message })
                }
                "halt" => {
                    let reason = val
                        .get("reason")
                        .and_then(Value::as_str)
                        .unwrap_or("halted by host")
                        .to_string();
                    Ok(BoundaryAction::Halt { reason })
                }
                _ => Ok(BoundaryAction::Continue),
            }
        })
    }

    fn before_compact<'a>(
        &'a self,
        ctx: &'a CompactionHookContext,
    ) -> BoxFuture<'a, Result<CompactionHookAction>> {
        Box::pin(async move {
            if !self.has_hook("before_compact") {
                return Ok(CompactionHookAction::Continue);
            }
            let Ok(val) = self.bridge.call("host/before_compact", json!(ctx)).await else {
                return Ok(CompactionHookAction::Continue);
            };
            let action = val
                .get("action")
                .and_then(Value::as_str)
                .unwrap_or("continue")
                .to_ascii_lowercase();
            match action.as_str() {
                "add_focus" => {
                    let focus = val
                        .get("focus")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    if focus.trim().is_empty() {
                        Ok(CompactionHookAction::Continue)
                    } else {
                        Ok(CompactionHookAction::AddFocus(focus))
                    }
                }
                "cancel" => {
                    let reason = val
                        .get("reason")
                        .and_then(Value::as_str)
                        .unwrap_or("cancelled by host")
                        .to_string();
                    Ok(CompactionHookAction::Cancel { reason })
                }
                _ => Ok(CompactionHookAction::Continue),
            }
        })
    }
}
