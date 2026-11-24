//! Out-of-process drop-in file extension host (`carnot-ext/1` JSON-RPC 2.0 over multiplexed stdio).
//!
//! Lets users extend the compiled `carnot` binary without recompiling Rust — drop an executable
//! script (Python, Bun/TS, Bash) or native binary into `~/.carnot/extensions/` or pass `--ext <file>`.
//!
//! Security & performance invariants:
//!   - **Zero overhead when unused:** 0 processes spawned if no drop-in extensions are configured.
//!   - **Anti-RCE confinement:** workspace-local `.carnot/extensions/` is NEVER executed unless
//!     explicitly opted in via `--trust-workspace-ext`.
//!   - **Negotiated hooks:** extensions declare subscribed hooks in `ext/init`; unsubscribed hooks
//!     incur ZERO IPC calls.
//!   - **Prompt-cache safe:** extension tools default to `defer_loading: true` (0 static tokens).
//!   - **Resilient:** multiplexed stdio I/O, per-call timeouts, bounded auto-restart (3x), and
//!     configurable `fail_closed` / `fail_open` error policy.

use crate::{
    BoundaryAction, BoxFuture, CompactionHookAction, CompactionHookContext, Module,
    ToolGateDecision, TurnBoundarySnapshot,
};
use carnot_core::{CarnotConfig, CarnotError, Result, ToolEffects, ToolOutput, ToolSpec};
use carnot_sandbox::apply_safe_env;
use carnot_tools::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
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

/// Launch specification for one discovered drop-in file extension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropinExtensionSpec {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub source: String,
}

#[derive(Debug, Deserialize)]
struct ExtensionsManifest {
    #[serde(default)]
    extensions: BTreeMap<String, ManifestEntry>,
}

#[derive(Debug, Deserialize)]
struct ManifestEntry {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        (meta.permissions().mode() & 0o111) != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn spec_from_file(path: &Path, source: &str) -> Option<DropinExtensionSpec> {
    if !path.is_file() {
        return None;
    }
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("extension")
        .to_string();
    let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if is_executable_file(&abs) {
        return Some(DropinExtensionSpec {
            name,
            command: abs.to_string_lossy().to_string(),
            args: Vec::new(),
            env: BTreeMap::new(),
            source: source.to_string(),
        });
    }
    // Convenience: allow non-chmod+x `.py` / `.sh` scripts when explicitly passed via `--ext`
    if source == "cli" {
        match abs.extension().and_then(|e| e.to_str()) {
            Some("py") => {
                return Some(DropinExtensionSpec {
                    name,
                    command: "python3".into(),
                    args: vec![abs.to_string_lossy().to_string()],
                    env: BTreeMap::new(),
                    source: source.to_string(),
                });
            }
            Some("sh") => {
                return Some(DropinExtensionSpec {
                    name,
                    command: "/bin/sh".into(),
                    args: vec![abs.to_string_lossy().to_string()],
                    env: BTreeMap::new(),
                    source: source.to_string(),
                });
            }
            _ => {}
        }
    }
    None
}

fn scan_extensions_dir(dir: &Path, source: &str, out: &mut BTreeMap<String, DropinExtensionSpec>) {
    if !dir.is_dir() {
        return;
    }
    if std::fs::symlink_metadata(dir)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
    {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if fname.is_empty() || fname.starts_with('.') {
            continue;
        }
        if is_executable_file(&p)
            && let Some(spec) = spec_from_file(&p, source)
        {
            out.insert(spec.name.clone(), spec);
        }
    }
}

fn load_manifest(path: &Path, source: &str, out: &mut BTreeMap<String, DropinExtensionSpec>) {
    if !path.is_file() {
        return;
    }
    let Ok(raw) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(manifest) = serde_json::from_str::<ExtensionsManifest>(&raw) else {
        tracing::warn!(path = %path.display(), "ignoring malformed extensions.json manifest");
        return;
    };
    for (name, entry) in manifest.extensions {
        if entry.command.trim().is_empty() {
            continue;
        }
        out.insert(
            name.clone(),
            DropinExtensionSpec {
                name,
                command: entry.command,
                args: entry.args,
                env: entry.env,
                source: source.to_string(),
            },
        );
    }
}

/// Discovers drop-in extensions in deterministic precedence order (zero process spawns):
///   1. User-global `~/.carnot/extensions/` + `~/.carnot/extensions.json`
///   2. Workspace-local `.carnot/extensions/` + `.carnot/extensions.json` (ONLY when `trust_workspace == true`)
///   3. Explicit `--ext <path>` files (highest precedence)
pub fn discover_dropin_specs(
    workspace_root: &Path,
    explicit_paths: &[PathBuf],
    trust_workspace: bool,
) -> Vec<DropinExtensionSpec> {
    let mut found: BTreeMap<String, DropinExtensionSpec> = BTreeMap::new();

    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        let user_root = home.join(".carnot");
        load_manifest(
            &user_root.join("extensions.json"),
            "user-manifest",
            &mut found,
        );
        scan_extensions_dir(&user_root.join("extensions"), "user-dir", &mut found);
    }

    if trust_workspace {
        let ws_carnot = workspace_root.join(".carnot");
        load_manifest(
            &ws_carnot.join("extensions.json"),
            "workspace-manifest",
            &mut found,
        );
        scan_extensions_dir(&ws_carnot.join("extensions"), "workspace-dir", &mut found);
    }

    for p in explicit_paths {
        if let Some(spec) = spec_from_file(p, "cli") {
            found.insert(spec.name.clone(), spec);
        } else {
            tracing::warn!(path = %p.display(), "skipping non-executable or missing --ext file");
        }
    }

    found.into_values().collect()
}

/// Loads and initializes all discovered drop-in extensions as `Arc<dyn Module>`.
pub async fn load_dropin_extensions(
    workspace_root: &Path,
    explicit_paths: &[PathBuf],
    trust_workspace: bool,
) -> Result<Vec<Arc<dyn Module>>> {
    let specs = discover_dropin_specs(workspace_root, explicit_paths, trust_workspace);
    let mut modules: Vec<Arc<dyn Module>> = Vec::with_capacity(specs.len());
    for spec in specs {
        let ext = DropinExtension::spawn_from_spec(spec).await?;
        modules.push(Arc::new(ext));
    }
    Ok(modules)
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
    async fn spawn_and_init(spec: &DropinExtensionSpec) -> Result<(Arc<Self>, Value)> {
        let mut cmd = Command::new(&spec.command);
        cmd.args(&spec.args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true);
        apply_safe_env(&mut cmd, &spec.env);
        let mut child = cmd.spawn().map_err(|e| CarnotError::Extension {
            extension: spec.name.clone(),
            message: format!("failed to spawn `{}`: {e}", spec.command),
        })?;

        let mut stdin = child.stdin.take().ok_or_else(|| CarnotError::Extension {
            extension: spec.name.clone(),
            message: "missing stdin".into(),
        })?;
        let stdout = child.stdout.take().ok_or_else(|| CarnotError::Extension {
            extension: spec.name.clone(),
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
        let name_r = spec.name.clone();
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
                            && let Some(tx) = pending_r
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .remove(&id)
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
            let mut map = pending_r.lock().unwrap_or_else(|e| e.into_inner());
            for (_, tx) in map.drain() {
                let _ = tx.send(Err(CarnotError::Extension {
                    extension: name_r.clone(),
                    message: "extension process closed stdout".into(),
                }));
            }
        });

        let mut conn = Self {
            ext_name: spec.name.clone(),
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
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, tx);

        let req = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let mut line = serde_json::to_vec(&req)?;
        line.push(b'\n');

        if self.write_tx.send(line).await.is_err() {
            self.closed.store(true, Ordering::Release);
            self.pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&id);
            return Err(CarnotError::Extension {
                extension: self.ext_name.clone(),
                message: "extension process closed stdin".into(),
            });
        }

        let timed = async {
            match cancel {
                Some(tok) => tokio::select! {
                    _ = tok.cancelled() => Err(CarnotError::Cancelled),
                    r = tokio::time::timeout(self.timeout, rx) => match r {
                        Ok(Ok(val)) => val,
                        Ok(Err(_)) => Err(CarnotError::Extension {
                            extension: self.ext_name.clone(),
                            message: "extension disconnected before replying".into(),
                        }),
                        Err(_) => Err(CarnotError::Extension {
                            extension: self.ext_name.clone(),
                            message: format!("hook `{method}` timed out after {}ms", self.timeout.as_millis()),
                        }),
                    }
                },
                None => match tokio::time::timeout(self.timeout, rx).await {
                    Ok(Ok(val)) => val,
                    Ok(Err(_)) => Err(CarnotError::Extension {
                        extension: self.ext_name.clone(),
                        message: "extension disconnected before replying".into(),
                    }),
                    Err(_) => Err(CarnotError::Extension {
                        extension: self.ext_name.clone(),
                        message: format!(
                            "hook `{method}` timed out after {}ms",
                            self.timeout.as_millis()
                        ),
                    }),
                },
            }
        }
        .await;

        if timed.is_err() {
            self.pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&id);
        }
        timed
    }
}

#[derive(Debug)]
struct ExtSupervisor {
    spec: DropinExtensionSpec,
    conn: Mutex<Arc<ExtConn>>,
    restarts: AtomicU32,
}

impl ExtSupervisor {
    fn new(spec: DropinExtensionSpec, initial: Arc<ExtConn>) -> Self {
        Self {
            spec,
            conn: Mutex::new(initial),
            restarts: AtomicU32::new(0),
        }
    }

    async fn call(
        &self,
        method: &str,
        params: Value,
        cancel: Option<&CancellationToken>,
    ) -> Result<Value> {
        let current = self.conn.lock().await.clone();
        if !current.is_closed() {
            match current.call_raw(method, params.clone(), cancel).await {
                Ok(v) => return Ok(v),
                Err(CarnotError::Cancelled) => return Err(CarnotError::Cancelled),
                Err(first_err) => {
                    if !current.is_closed() {
                        return Err(first_err);
                    }
                }
            }
        }
        // Bounded auto-restart on unexpected crash
        let mut guard = self.conn.lock().await;
        if guard.is_closed() {
            let count = self.restarts.fetch_add(1, Ordering::Relaxed);
            if count >= MAX_AUTO_RESTARTS {
                return Err(CarnotError::Extension {
                    extension: self.spec.name.clone(),
                    message: format!("extension exceeded max auto-restarts ({MAX_AUTO_RESTARTS})"),
                });
            }
            tracing::warn!(extension = %self.spec.name, restart = count + 1, "restarting crashed drop-in extension");
            let (fresh, _) = ExtConn::spawn_and_init(&self.spec).await?;
            *guard = fresh;
        }
        let fresh = Arc::clone(&guard);
        drop(guard);
        fresh.call_raw(method, params, cancel).await
    }
}

#[derive(Debug)]
struct DropinTool {
    spec: ToolSpec,
    supervisor: Arc<ExtSupervisor>,
}

impl Tool for DropinTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn effects(&self, _input: &Value) -> ToolEffects {
        ToolEffects::serial()
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let res = self
                .supervisor
                .call(
                    "ext/tool_call",
                    json!({ "name": self.spec.name, "input": input }),
                    Some(&cancel),
                )
                .await;
            match res {
                Ok(val) => {
                    let text = val
                        .get("text")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| val.to_string());
                    let is_error = val
                        .get("is_error")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    if is_error {
                        Ok(ToolOutput::err_text(text))
                    } else {
                        Ok(ToolOutput::ok_text(text))
                    }
                }
                Err(CarnotError::Cancelled) => Err(CarnotError::Cancelled),
                Err(e) => Ok(ToolOutput::err_text(format!("extension tool error: {e}"))),
            }
        })
    }
}

/// Live drop-in file extension implementing Carnot's `Module` trait with negotiated hooks.
#[derive(Debug)]
pub struct DropinExtension {
    name: String,
    supervisor: Arc<ExtSupervisor>,
    hooks: HashSet<String>,
    failure_policy: FailurePolicy,
    tools: Vec<Arc<dyn Tool>>,
}

impl DropinExtension {
    pub async fn spawn_from_spec(spec: DropinExtensionSpec) -> Result<Self> {
        let (conn, init) = ExtConn::spawn_and_init(&spec).await?;
        let supervisor = Arc::new(ExtSupervisor::new(spec.clone(), conn));

        let failure_policy = match init.get("failure_policy").and_then(Value::as_str) {
            Some("fail_open") => FailurePolicy::FailOpen,
            _ => FailurePolicy::FailClosed,
        };

        let hooks: HashSet<String> = if let Some(arr) = init.get("hooks").and_then(Value::as_array)
        {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        } else {
            [
                "session_start",
                "before_tool",
                "after_tool",
                "turn_boundary",
                "before_compact",
            ]
            .into_iter()
            .map(str::to_string)
            .collect()
        };

        let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
        if let Some(arr) = init.get("tools").and_then(Value::as_array) {
            for t in arr {
                let Some(tname) = t.get("name").and_then(Value::as_str) else {
                    continue;
                };
                if tname.trim().is_empty() {
                    continue;
                }
                let desc = t
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let schema = t
                    .get("input_schema")
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
                // Default to `defer_loading: true` so drop-in tools cost 0 static prefix tokens!
                let defer_loading = t
                    .get("defer_loading")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                tools.push(Arc::new(DropinTool {
                    spec: ToolSpec {
                        name: tname.to_string(),
                        description: desc,
                        input_schema: schema,
                        defer_loading,
                    },
                    supervisor: Arc::clone(&supervisor),
                }));
            }
        }

        Ok(Self {
            name: spec.name,
            supervisor,
            hooks,
            failure_policy,
            tools,
        })
    }

    pub fn subscribed_hooks(&self) -> &HashSet<String> {
        &self.hooks
    }
}

impl Module for DropinExtension {
    fn name(&self) -> &str {
        &self.name
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    fn on_session_start(&self, config: &CarnotConfig) -> BoxFuture<'_, Result<Option<String>>> {
        if !self.hooks.contains("session_start") {
            return Box::pin(async { Ok(None) });
        }
        let ws = config.workspace_root.display().to_string();
        let model = config.model.clone();
        Box::pin(async move {
            match self
                .supervisor
                .call(
                    "ext/session_start",
                    json!({ "workspace_root": ws, "model": model }),
                    None,
                )
                .await
            {
                Ok(v) => Ok(v
                    .get("context")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .filter(|s| !s.trim().is_empty())),
                Err(e) => {
                    tracing::warn!(extension = %self.name, error = %e, "session_start hook failed");
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
        if !self.hooks.contains("before_tool") {
            return Box::pin(async { Ok(ToolGateDecision::Allow) });
        }
        let tool = tool_name.to_string();
        let inp = input.clone();
        Box::pin(async move {
            match self
                .supervisor
                .call(
                    "ext/before_tool",
                    json!({ "tool": tool, "input": inp }),
                    None,
                )
                .await
            {
                Ok(v) => match v.get("decision").and_then(Value::as_str) {
                    Some("deny") => Ok(ToolGateDecision::Deny {
                        reason: v
                            .get("reason")
                            .and_then(Value::as_str)
                            .unwrap_or("denied by extension")
                            .to_string(),
                    }),
                    Some("rewrite") => {
                        if let Some(new_in) = v.get("input").cloned() {
                            Ok(ToolGateDecision::RewriteInput(new_in))
                        } else {
                            Ok(ToolGateDecision::Allow)
                        }
                    }
                    _ => Ok(ToolGateDecision::Allow),
                },
                Err(e) => match self.failure_policy {
                    FailurePolicy::FailClosed => Ok(ToolGateDecision::Deny {
                        reason: format!("extension `{}` fail-closed: {e}", self.name),
                    }),
                    FailurePolicy::FailOpen => {
                        tracing::warn!(extension = %self.name, error = %e, "before_tool failed (fail-open)");
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
        if !self.hooks.contains("after_tool") {
            return Box::pin(async { Ok(()) });
        }
        let tool = tool_name.to_string();
        let inp = input.clone();
        let text = output.text_only();
        let is_err = output.is_error;
        Box::pin(async move {
            if let Ok(v) = self
                .supervisor
                .call(
                    "ext/after_tool",
                    json!({ "tool": tool, "input": inp, "is_error": is_err, "text": text }),
                    None,
                )
                .await
                && let Some(extra) = v.get("append_text").and_then(Value::as_str)
                && !extra.is_empty()
            {
                output
                    .blocks
                    .push(carnot_core::ToolResultBlock::text(extra));
            }
            Ok(())
        })
    }

    fn on_turn_boundary<'a>(
        &'a self,
        snapshot: &'a TurnBoundarySnapshot,
    ) -> BoxFuture<'a, Result<BoundaryAction>> {
        if !self.hooks.contains("turn_boundary") {
            return Box::pin(async { Ok(BoundaryAction::Continue) });
        }
        let payload = serde_json::to_value(snapshot).unwrap_or(Value::Null);
        Box::pin(async move {
            match self
                .supervisor
                .call("ext/turn_boundary", payload, None)
                .await
            {
                Ok(v) => match v.get("action").and_then(Value::as_str) {
                    Some("advisory") => Ok(BoundaryAction::AppendAdvisory {
                        signal: v
                            .get("signal")
                            .and_then(Value::as_str)
                            .unwrap_or("ext")
                            .to_string(),
                        message: v
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                    }),
                    Some("halt") => Ok(BoundaryAction::Halt {
                        reason: v
                            .get("reason")
                            .and_then(Value::as_str)
                            .unwrap_or("halted by extension")
                            .to_string(),
                    }),
                    _ => Ok(BoundaryAction::Continue),
                },
                Err(e) => {
                    tracing::warn!(extension = %self.name, error = %e, "turn_boundary failed");
                    Ok(BoundaryAction::Continue)
                }
            }
        })
    }

    fn before_compact<'a>(
        &'a self,
        ctx: &'a CompactionHookContext,
    ) -> BoxFuture<'a, Result<CompactionHookAction>> {
        if !self.hooks.contains("before_compact") {
            return Box::pin(async { Ok(CompactionHookAction::Continue) });
        }
        let payload = serde_json::to_value(ctx).unwrap_or(Value::Null);
        Box::pin(async move {
            match self
                .supervisor
                .call("ext/before_compact", payload, None)
                .await
            {
                Ok(v) => match v.get("action").and_then(Value::as_str) {
                    Some("add_focus") => Ok(CompactionHookAction::AddFocus(
                        v.get("focus")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                    )),
                    Some("cancel") => Ok(CompactionHookAction::Cancel {
                        reason: v
                            .get("reason")
                            .and_then(Value::as_str)
                            .unwrap_or("cancelled by extension")
                            .to_string(),
                    }),
                    _ => Ok(CompactionHookAction::Continue),
                },
                Err(e) => {
                    tracing::warn!(extension = %self.name, error = %e, "before_compact failed (fail-open)");
                    Ok(CompactionHookAction::Continue)
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn dropin_python_file_extension_negotiates_hooks_gates_tools_and_adds_compact_focus() {
        let dir = std::env::temp_dir().join(format!("carnot-dropin-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let ext_file = dir.join("guard_ext.py");
        let script = r#"#!/usr/bin/env python3
import json, sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    rid, m, p = req.get("id"), req.get("method"), req.get("params") or {}
    if m == "ext/init":
        res = {
            "protocolVersion": "carnot-ext/1",
            "hooks": ["before_tool", "before_compact"],
            "tools": [{"name": "EchoExt", "description": "Echoes msg", "input_schema": {"type": "object", "properties": {"msg": {"type": "string"}}}}],
            "failure_policy": "fail_closed",
        }
    elif m == "ext/before_tool":
        if p.get("tool") == "Bash" and "forbidden" in str(p.get("input", {})):
            res = {"decision": "deny", "reason": "blocked by dropin guard"}
        else:
            res = {"decision": "allow"}
    elif m == "ext/before_compact":
        res = {"action": "add_focus", "focus": "Preserve SQL migration constraints"}
    elif m == "ext/tool_call":
        res = {"text": f"echo:{p.get('input', {}).get('msg', '')}", "is_error": False}
    else:
        res = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": res}) + "\n")
    sys.stdout.flush()
"#;
        std::fs::write(&ext_file, script).expect("write ext");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&ext_file, std::fs::Permissions::from_mode(0o755))
                .expect("chmod");
        }

        // Workspace-local `.carnot/extensions` must NOT load when trust_workspace == false
        let ws_ext_dir = dir.join(".carnot/extensions");
        std::fs::create_dir_all(&ws_ext_dir).expect("ws ext dir");
        std::fs::copy(&ext_file, ws_ext_dir.join("untrusted.py")).expect("copy");
        let untrusted = discover_dropin_specs(&dir, &[], false);
        assert!(
            !untrusted.iter().any(|s| s.name == "untrusted"),
            "workspace extension must not load without trust_workspace"
        );
        let trusted = discover_dropin_specs(&dir, &[], true);
        assert!(trusted.iter().any(|s| s.name == "untrusted"));

        // Load explicit drop-in file extension
        let loaded = load_dropin_extensions(&dir, &[ext_file], false)
            .await
            .expect("load dropin");
        let ext = loaded
            .iter()
            .find(|m| m.name() == "guard_ext")
            .expect("guard_ext loaded");

        // 1. Custom tool defaults to defer_loading: true (0 static prefix tokens!)
        let tools = ext.tools();
        assert_eq!(tools.len(), 1);
        assert!(tools[0].spec().defer_loading);
        let tool_out = tools[0]
            .execute(json!({"msg": "hello"}), CancellationToken::new())
            .await
            .expect("tool exec");
        assert_eq!(tool_out.text_only(), "echo:hello");

        // 2. before_tool hook denies forbidden Bash
        let deny = ext
            .before_tool_call("Bash", &json!({"command": "echo forbidden"}))
            .await
            .expect("gate");
        assert!(
            matches!(deny, ToolGateDecision::Deny { ref reason } if reason.contains("dropin guard"))
        );

        // 3. before_compact hook adds focus
        let compact = ext
            .before_compact(&CompactionHookContext {
                mode: "manual".into(),
                tokens_before: 1000,
                message_count: 10,
                custom_focus: None,
            })
            .await
            .expect("before_compact");
        assert_eq!(
            compact,
            CompactionHookAction::AddFocus("Preserve SQL migration constraints".into())
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
