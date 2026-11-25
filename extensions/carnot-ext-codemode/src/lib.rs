//! Isolated `QuickJS` `CodeMode` extension for Carnot (`carnot-ext-codemode`).
//!
//! Lets the model execute a short async JavaScript snippet in a sandboxed `QuickJS` runtime
//! (no filesystem, network, env, or process APIs) to:
//! - batch parallel tool and MCP calls (`await Promise.all([...])`),
//! - filter, join, or aggregate large JSON/text payloads before they enter LLM context,
//! - discover deferred MCP/extension tool schemas on demand (`searchTools`, `describeTool`),
//! - carry small structured state across calls via transactional `store`/`load`.
//!
//! Every nested `tools.<name>(args)` call routes through `Agent::invoke_tool_gated`, preserving
//! all `PolicyGuard`, `Hooks`, `PermissionBroker`, `ReadTracker`, CAS Undo, and secret-redaction
//! invariants with zero bypass paths.

mod engine;

use carnot_agent::Agent;
use carnot_cas::{CasStore, redact_secrets_cow};
use carnot_core::{CarnotError, Result, ToolEffects, ToolOutput, ToolSpec};
use carnot_ext::Module;
use carnot_tools::{BoxFuture, Tool};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

const MAX_TIMEOUT_MS: u64 = 300_000;
const MIN_TIMEOUT_MS: u64 = 100;

/// Configuration and hard resource limits for the `QuickJS` `CodeMode` sandbox.
#[derive(Debug, Clone)]
pub struct CodeModeConfig {
    /// Default script timeout in milliseconds (can be overridden per call up to 300s).
    pub timeout_ms: u64,
    /// Hard `QuickJS` heap memory cap in bytes (default: 64 MiB).
    pub memory_limit_bytes: usize,
    /// Hard `QuickJS` call stack cap in bytes (default: 2 MiB).
    pub max_stack_size_bytes: usize,
    /// Maximum nested tool calls permitted within a single `CodeMode` invocation (default: 64).
    pub max_tool_calls: usize,
    /// Inline output byte threshold before spilling full output to CAS (default: 16 KiB).
    pub max_output_bytes: usize,
    /// Maximum total bytes captured by `console.*` per invocation (default: 64 KiB).
    pub max_log_bytes: usize,
    /// Maximum total bytes stored across session `store(k, v)` entries (default: 512 KiB).
    pub max_store_bytes: usize,
    /// Whether `CodeMode` is deferred behind `ToolSearch` (`false` by default so the agent can
    /// choose it spontaneously whenever batching/filtering is beneficial).
    pub defer_loading: bool,
}

impl Default for CodeModeConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 60_000,
            memory_limit_bytes: 64 * 1024 * 1024,
            max_stack_size_bytes: 2 * 1024 * 1024,
            max_tool_calls: 64,
            max_output_bytes: 16_384,
            max_log_bytes: 65_536,
            max_store_bytes: 512 * 1024,
            defer_loading: false,
        }
    }
}

/// Host bridge invoked by `QuickJS` scripts to list and execute tools through the harness.
pub trait CodeModeHost: Send + Sync + std::fmt::Debug {
    /// Returns all registered tools callable by this session.
    fn list_tools(&self) -> Vec<ToolSpec>;

    /// Executes one tool call through all security gates and hooks.
    fn call_tool(
        &self,
        call_id: String,
        name: String,
        args: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>>;
}

/// Weak-referenced host binding to a parent [`Agent`] (acyclic — never leaks `Agent`).
#[derive(Debug, Default)]
struct AgentHost {
    parent: RwLock<Weak<Agent>>,
}

impl AgentHost {
    fn attach(&self, agent: &Arc<Agent>) {
        if let Ok(mut guard) = self.parent.write() {
            *guard = Arc::downgrade(agent);
        }
    }

    fn upgrade(&self) -> Option<Arc<Agent>> {
        self.parent.read().ok().and_then(|g| g.upgrade())
    }
}

impl CodeModeHost for AgentHost {
    fn list_tools(&self) -> Vec<ToolSpec> {
        self.upgrade().map(|a| a.tool_specs()).unwrap_or_default()
    }

    fn call_tool(
        &self,
        call_id: String,
        name: String,
        args: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        let agent = self.upgrade();
        Box::pin(async move {
            let Some(agent) = agent else {
                return Err(CarnotError::Tool {
                    tool: "CodeMode".into(),
                    message: "CodeMode host agent is no longer attached".into(),
                });
            };
            agent.invoke_tool_gated(&call_id, &name, args, cancel).await
        })
    }
}

/// Canonical `CodeMode` tool executing isolated async JavaScript over session tools.
#[derive(Debug)]
pub struct CodeModeTool {
    host: Arc<dyn CodeModeHost>,
    cas: Arc<CasStore>,
    cas_handles: bool,
    redact_secrets: bool,
    config: CodeModeConfig,
    session_store: Arc<Mutex<BTreeMap<String, Value>>>,
}

impl CodeModeTool {
    /// Creates a `CodeModeTool` backed by a custom [`CodeModeHost`] (useful for testing or custom runtimes).
    pub fn with_host(
        host: Arc<dyn CodeModeHost>,
        cas: Arc<CasStore>,
        cas_handles: bool,
        redact_secrets: bool,
        config: CodeModeConfig,
    ) -> Self {
        Self {
            host,
            cas,
            cas_handles,
            redact_secrets,
            config,
            session_store: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    fn format_output(&self, out: &engine::RunOutput) -> String {
        let status = if out.ok { "ok" } else { "failed" };
        let mut body = String::new();
        let _ = writeln!(
            body,
            "[CodeMode {status} | {} tool call(s) | {}ms]",
            out.tool_calls, out.duration_ms
        );

        let mut items: Vec<&str> = out.text_items.iter().map(String::as_str).collect();
        if let Some(ref ret) = out.return_value {
            items.push(ret.as_str());
        }
        if items.len() == 1 {
            body.push_str(items[0]);
            if !body.ends_with('\n') {
                body.push('\n');
            }
        } else if items.len() > 1 {
            for (idx, item) in items.iter().enumerate() {
                let _ = writeln!(body, "==> output {}/{} <==", idx + 1, items.len());
                body.push_str(item);
                if !body.ends_with('\n') {
                    body.push('\n');
                }
            }
        }

        if !out.console_lines.is_empty() {
            body.push_str("<console_output>\n");
            for line in &out.console_lines {
                body.push_str(line);
                body.push('\n');
            }
            if out.logs_truncated {
                body.push_str("[console output truncated]\n");
            }
            body.push_str("</console_output>\n");
        } else if out.logs_truncated {
            body.push_str("[output truncated]\n");
        }

        if let Some(ref err) = out.error {
            let _ = writeln!(body, "Script error: {err}");
        }

        let trimmed = body.trim_end_matches('\n');
        let safe_text = if self.redact_secrets {
            redact_secrets_cow(trimmed)
        } else {
            std::borrow::Cow::Borrowed(trimmed)
        };

        if safe_text.len() <= self.config.max_output_bytes {
            return safe_text.into_owned();
        }

        // Spill oversized output losslessly into Carnot's BLAKE3 CAS and return head+tail preview.
        let head_budget = self.config.max_output_bytes / 2;
        let tail_budget = self.config.max_output_bytes.saturating_sub(head_budget);
        let head_end = floor_char_boundary(&safe_text, head_budget);
        let tail_start =
            ceil_char_boundary(&safe_text, safe_text.len().saturating_sub(tail_budget));
        let spill_note = if self.cas_handles
            && let Ok(handle) = self.cas.put_bytes(safe_text.as_bytes())
        {
            format!(
                "\n[... CodeMode output truncated ({} bytes total); full output saved at {} (inspect with Read/Grep) ...]\n",
                safe_text.len(),
                handle.path.display()
            )
        } else {
            format!(
                "\n[... CodeMode output truncated ({} bytes total) ...]\n",
                safe_text.len()
            )
        };
        format!(
            "{}{}{}",
            &safe_text[..head_end],
            spill_note,
            &safe_text[tail_start..]
        )
    }
}

fn floor_char_boundary(s: &str, max_bytes: usize) -> usize {
    if max_bytes >= s.len() {
        return s.len();
    }
    let mut i = max_bytes;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char_boundary(s: &str, min_bytes: usize) -> usize {
    if min_bytes >= s.len() {
        return s.len();
    }
    let mut i = min_bytes;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

impl Tool for CodeModeTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "CodeMode".into(),
            description: "Execute an async JavaScript snippet in an isolated QuickJS sandbox with access to all session tools via `await tools.<name>(args)` (supports `Promise.all`, `return`, `text()`, `console.log`, `exit()`, `store`/`load`, `searchTools(q)`, `describeTool(name)`). Use to batch parallel tool/MCP calls or filter/aggregate large payloads in one turn instead of pulling raw data into context. For single Read/Edit/Grep/Bash calls, call those tools directly. MCP/extension JSON outputs auto-parse; core text tools return strings; failed calls reject.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Async JavaScript body (top-level await and return supported; no markdown fences)."
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Optional timeout in milliseconds (default 60000, max 300000)."
                    }
                },
                "required": ["code"]
            }),
            defer_loading: self.config.defer_loading,
        }
    }

    fn effects(&self, _input: &Value) -> ToolEffects {
        ToolEffects::serial()
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        self.execute_with_id("cm", input, cancel)
    }

    fn execute_with_id<'a>(
        &'a self,
        call_id: &'a str,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<ToolOutput>> {
        let call_id = call_id.to_string();
        Box::pin(async move {
            let Some(code) = input.get("code").and_then(Value::as_str) else {
                return Ok(ToolOutput::err_text("CodeMode requires string `code`"));
            };
            if code.trim().is_empty() {
                return Ok(ToolOutput::err_text("CodeMode `code` must not be empty"));
            }

            let timeout_ms = input
                .get("timeout_ms")
                .and_then(Value::as_u64)
                .filter(|&v| v > 0)
                .map_or(self.config.timeout_ms, |v| {
                    v.clamp(MIN_TIMEOUT_MS, MAX_TIMEOUT_MS)
                });

            let initial_store = self
                .session_store
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();

            let run_out = engine::execute_script(
                code.to_string(),
                call_id,
                Arc::clone(&self.host),
                initial_store,
                self.config.clone(),
                Duration::from_millis(timeout_ms),
                cancel,
            )
            .await;

            // Commit store updates transactionally only on successful script completion.
            if run_out.ok
                && let Some(ref next_store) = run_out.next_store
            {
                let mut g = self
                    .session_store
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                g.clone_from(next_store);
            }

            let rendered = self.format_output(&run_out);
            let details = json!({
                "ok": run_out.ok,
                "tool_calls": run_out.tool_calls,
                "tools_called": run_out.tool_names_called,
                "duration_ms": run_out.duration_ms,
            });
            if run_out.ok {
                Ok(ToolOutput::ok_text(rendered).with_details(details))
            } else {
                Ok(ToolOutput::err_text(rendered).with_details(details))
            }
        })
    }
}

/// Official `CodeMode` module registering the sandboxed `QuickJS` orchestrator tool.
#[derive(Debug)]
pub struct CodeModeModule {
    host: Arc<AgentHost>,
    tool: Arc<CodeModeTool>,
}

impl CodeModeModule {
    pub fn new(
        cas: Arc<CasStore>,
        cas_handles: bool,
        redact_secrets: bool,
        config: CodeModeConfig,
    ) -> Self {
        let host = Arc::new(AgentHost::default());
        let tool = Arc::new(CodeModeTool::with_host(
            Arc::clone(&host) as Arc<dyn CodeModeHost>,
            cas,
            cas_handles,
            redact_secrets,
            config,
        ));
        Self { host, tool }
    }

    /// Attaches the owning [`Agent`] via a weak pointer (call right after `Arc::new(agent)`).
    pub fn attach_parent(&self, agent: &Arc<Agent>) {
        self.host.attach(agent);
    }
}

impl Module for CodeModeModule {
    fn name(&self) -> &'static str {
        "codemode"
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![Arc::clone(&self.tool) as Arc<dyn Tool>]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[derive(Debug)]
    struct MockHost {
        specs: Vec<ToolSpec>,
    }

    impl MockHost {
        fn new() -> Self {
            Self {
                specs: vec![
                    ToolSpec {
                        name: "Read".into(),
                        description: "Read a workspace file".into(),
                        input_schema: json!({"type": "object", "properties": {"file_path": {"type": "string"}}}),
                        defer_loading: false,
                    },
                    ToolSpec {
                        name: "mcp__dev-radius__search".into(),
                        description: "Search issues in Dev Radius tracker".into(),
                        input_schema: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
                        defer_loading: true,
                    },
                    ToolSpec {
                        name: "mcp__slow__fetch".into(),
                        description: "Slow async fetch for parallelism test".into(),
                        input_schema: json!({"type": "object", "properties": {"id": {"type": "integer"}}}),
                        defer_loading: true,
                    },
                    ToolSpec {
                        name: "mcp__fail__boom".into(),
                        description: "Always fails".into(),
                        input_schema: json!({"type": "object"}),
                        defer_loading: true,
                    },
                ],
            }
        }
    }

    impl CodeModeHost for MockHost {
        fn list_tools(&self) -> Vec<ToolSpec> {
            self.specs.clone()
        }

        fn call_tool(
            &self,
            _call_id: String,
            name: String,
            args: Value,
            _cancel: CancellationToken,
        ) -> BoxFuture<'_, Result<ToolOutput>> {
            Box::pin(async move {
                match name.as_str() {
                    "Read" => {
                        let p = args.get("file_path").and_then(Value::as_str).unwrap_or("?");
                        Ok(ToolOutput::ok_text(format!("1\t{{\"file\":\"{p}\"}}")))
                    }
                    "mcp__dev-radius__search" => {
                        let q = args.get("q").and_then(Value::as_str).unwrap_or("");
                        Ok(ToolOutput::ok_text(
                            json!([
                                {"id": 1, "title": format!("{q} bug"), "priority": 1},
                                {"id": 2, "title": "low prio", "priority": 4}
                            ])
                            .to_string(),
                        ))
                    }
                    "mcp__slow__fetch" => {
                        tokio::time::sleep(Duration::from_millis(80)).await;
                        let id = args.get("id").and_then(Value::as_i64).unwrap_or(0);
                        Ok(ToolOutput::ok_text(
                            json!({"id": id, "ok": true}).to_string(),
                        ))
                    }
                    "mcp__fail__boom" => Ok(ToolOutput::err_text("permission denied by policy")),
                    other => Ok(ToolOutput::err_text(format!("unknown mock tool {other}"))),
                }
            })
        }
    }

    fn make_tool(config: CodeModeConfig) -> (CodeModeTool, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("carnot-cm-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let cas = Arc::new(CasStore::open(&dir).expect("cas"));
        let tool = CodeModeTool::with_host(Arc::new(MockHost::new()), cas, true, true, config);
        (tool, dir)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn parallel_promise_all_sanitized_mcp_names_and_discovery_work() {
        let (tool, dir) = make_tool(CodeModeConfig::default());

        let t0 = Instant::now();
        let out = tool
            .execute(
                json!({
                    "code": r#"
                        const found = searchTools("radius");
                        const desc = describeTool("mcp__dev_radius__search");
                        const issues = await tools.mcp__dev_radius__search({ q: "cache" });
                        const urgent = issues.filter(i => i.priority === 1);
                        const fetched = await Promise.all([
                            tools.mcp__slow__fetch({ id: 10 }),
                            tools.mcp__slow__fetch({ id: 20 }),
                            tools.mcp__slow__fetch({ id: 30 }),
                        ]);
                        const rawRead = await tools.Read({ file_path: "a.json" });
                        console.log("urgent:", urgent.length);
                        return {
                            found_name: found[0].name,
                            has_schema: Boolean(desc && desc.input_schema),
                            urgent_title: urgent[0].title,
                            fetched_ids: fetched.map(x => x.id),
                            read_is_string: typeof rawRead === "string",
                        };
                    "#
                }),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        let elapsed = t0.elapsed();

        assert!(!out.is_error, "expected ok, got: {}", out.text_only());
        // 3 x 80ms in parallel must complete well under 3 x 80ms = 240ms.
        assert!(
            elapsed < Duration::from_millis(210),
            "Promise.all should execute tools concurrently, elapsed={elapsed:?}"
        );
        let txt = out.text_only();
        assert!(
            txt.contains("\"found_name\":\"mcp__dev-radius__search\""),
            "got:\n{txt}"
        );
        assert!(txt.contains("\"has_schema\":true"), "got:\n{txt}");
        assert!(
            txt.contains("\"urgent_title\":\"cache bug\""),
            "got:\n{txt}"
        );
        assert!(txt.contains("\"fetched_ids\":[10,20,30]"), "got:\n{txt}");
        assert!(txt.contains("\"read_is_string\":true"), "got:\n{txt}");
        assert!(
            txt.contains("<console_output>\nurgent: 1\n</console_output>"),
            "got:\n{txt}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transactional_store_exit_and_error_rollback() {
        let (tool, dir) = make_tool(CodeModeConfig::default());

        // 1. Successful script with exit() commits store.
        let r1 = tool
            .execute(
                json!({
                    "code": "store('cursor', { page: 2 }); exit('early-ok'); store('cursor', { page: 99 });"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("r1");
        assert!(!r1.is_error, "exit should succeed: {}", r1.text_only());
        assert!(
            r1.text_only().contains("early-ok"),
            "got: {}",
            r1.text_only()
        );

        // 2. Failing script rolls back store mutations!
        let r2 = tool
            .execute(
                json!({
                    "code": "store('cursor', { page: 999 }); await tools.mcp__fail__boom({});"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("r2");
        assert!(r2.is_error, "boom must fail");
        assert!(
            r2.text_only().contains("permission denied by policy"),
            "got: {}",
            r2.text_only()
        );

        // 3. Third script still sees committed page=2!
        let r3 = tool
            .execute(
                json!({
                    "code": "return { cursor: load('cursor'), keys: storedKeys() };"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("r3");
        assert!(!r3.is_error, "r3 failed: {}", r3.text_only());
        assert!(
            r3.text_only().contains("\"page\":2"),
            "rollback must preserve page=2, got: {}",
            r3.text_only()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn timeout_kills_infinite_loop_and_caps_tool_calls_and_spills_redacted_cas() {
        let cfg = CodeModeConfig {
            max_tool_calls: 2,
            max_output_bytes: 512,
            ..Default::default()
        };
        let (tool, dir) = make_tool(cfg);

        // 1. Infinite loop killed quickly by QuickJS interrupt handler.
        let t0 = Instant::now();
        let inf = tool
            .execute(
                json!({"code": "while (true) {}", "timeout_ms": 150}),
                CancellationToken::new(),
            )
            .await
            .expect("inf");
        assert!(inf.is_error, "infinite loop must fail");
        assert!(
            inf.text_only().contains("timed out"),
            "expected timeout message, got: {}",
            inf.text_only()
        );
        assert!(
            t0.elapsed() < Duration::from_secs(2),
            "interrupt handler must kill tight loop promptly"
        );

        // 2. Self-recursion blocked.
        let rec = tool
            .execute(
                json!({"code": "await tools.CodeMode({ code: 'return 1' });"}),
                CancellationToken::new(),
            )
            .await
            .expect("rec");
        assert!(rec.is_error, "recursion must be blocked");
        assert!(
            rec.text_only().contains("recursively"),
            "got: {}",
            rec.text_only()
        );

        // 3. Tool call limit enforced.
        let cap = tool
            .execute(
                json!({
                    "code": "await tools.Read({file_path:'1'}); await tools.Read({file_path:'2'}); await tools.Read({file_path:'3'});"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("cap");
        assert!(cap.is_error, "exceeding max_tool_calls must fail");
        assert!(
            cap.text_only().contains("max_tool_calls"),
            "got: {}",
            cap.text_only()
        );

        // 4. Large output with secret spills to CAS with secret redacted!
        let big = tool
            .execute(
                json!({
                    "code": "```js\nreturn 'sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456 ' + 'x'.repeat(2000);\n```"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("big");
        assert!(!big.is_error, "big failed: {}", big.text_only());
        let txt = big.text_only();
        assert!(
            txt.contains("[REDACTED:ANTHROPIC_KEY]"),
            "secret must be redacted, got: {txt}"
        );
        assert!(
            !txt.contains("sk-ant-api03-"),
            "raw secret must never appear"
        );
        assert!(
            txt.contains("full output saved at"),
            "oversized output must spill to CAS, got: {txt}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
