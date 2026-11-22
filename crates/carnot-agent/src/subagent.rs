//! Disposable-Context Synchronous & Non-Blocking Background `SubagentModule`
//! (lives in `carnot-agent` to keep the 12-crate DAG 100% acyclic!).
//!
//! Massive AUC, latency & cost multiplier:
//!   - Offloads noisy multi-file research or isolated subtasks to a fresh child `Agent` whose
//!     20k-80k token exploration trace is discarded on exit, returning ONLY a concise bounded report
//!     + git status/diff patch + exact cost to the parent.
//!   - **Non-blocking background mode (`background: true`)**: spawns child agents in parallel and
//!     returns a `sub-N` handle in <1 ms so the parent can keep working, poll live progress
//!     (`turns`, `tools`, `cost_so_far`, `last_activity`), wait (`wait: true`), list (`list: true`),
//!     or cancel (`kill: true`) with zero lost wakeups (`watch::channel`) and zero double-counted costs!
//!   - **Opus architectural safeguards preserved:**
//!     - Byte-identical static prefix with parent: child keeps the EXACT same tool schemas so it
//!       hits the parent's warm prompt cache on Turn 1; recursion is blocked in `before_tool_call`.
//!     - Without `isolated_worktree: true` (in sync mode), declares `ToolEffects::serial()`.
//!     - With `isolated_worktree: true`, creates a safe `IsolatedForkWorktree` (`git worktree add --detach`
//!       + dedicated `CARGO_TARGET_DIR`) and ALWAYS cleans it up even on cancellation/kill/Drop!
//!     - Strict cost budget (`subagent_max_cost_usd`, including cache writes) & linked cancellation.

use crate::Agent;
use carnot_ai::Provider;
use carnot_cas::CasStore;
use carnot_core::{AgentEvent, CarnotConfig, Result, ToolEffects, ToolOutput, ToolSpec};
use carnot_ext::{
    BoxFuture as ExtFuture, Module, ToolGateDecision, warm_fork::IsolatedForkWorktree,
};
use carnot_tools::{BoxFuture, Tool, ToolRegistry};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

const MAX_SUBAGENT_SUMMARY_CHARS: usize = 6_000;
const MAX_RETAINED_SUBAGENTS: usize = 32;

#[derive(Debug, Clone)]
struct SubagentFinished {
    report: String,
    is_error: bool,
    model: String,
    turns: u64,
    cost_usd: f64,
    isolated: bool,
    diff_handle: Option<String>,
}

#[derive(Debug)]
struct BackgroundSubagent {
    id: String,
    task_preview: String,
    model: String,
    isolated: bool,
    started: Instant,
    turns_completed: Arc<AtomicU64>,
    tools_called: Arc<AtomicU64>,
    cost_micros: Arc<AtomicU64>,
    last_activity: Arc<Mutex<String>>,
    cancel: CancellationToken,
    done_rx: watch::Receiver<Option<SubagentFinished>>,
}

impl BackgroundSubagent {
    fn progress_line(&self) -> String {
        let elapsed = self.started.elapsed().as_secs_f64();
        let turns = self.turns_completed.load(Ordering::Relaxed);
        let tools = self.tools_called.load(Ordering::Relaxed);
        let cost = self.cost_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let last = self
            .last_activity
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "starting".into());
        let state = if self.done_rx.borrow().is_some() {
            "finished (ready to collect)"
        } else {
            "running"
        };
        let iso = if self.isolated { " | isolated" } else { "" };
        format!(
            "{} [{} | {:.1}s | model={}{} | turns={} tools={} cost=${:.4} | last={}] — {}",
            self.id, state, elapsed, self.model, iso, turns, tools, cost, last, self.task_preview
        )
    }
}

/// Manages non-blocking background subagents with deterministic cleanup on `Drop`.
#[derive(Debug, Default)]
pub struct SubagentRegistry {
    jobs: Mutex<HashMap<String, Arc<BackgroundSubagent>>>,
    counter: AtomicU64,
}

impl SubagentRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    fn next_id(&self) -> String {
        let n = self.counter.fetch_add(1, Ordering::Relaxed) + 1;
        format!("sub-{n}")
    }

    fn insert(&self, job: Arc<BackgroundSubagent>) {
        let mut map = self.jobs.lock().expect("lock");
        if map.len() >= MAX_RETAINED_SUBAGENTS {
            let finished: Vec<String> = map
                .iter()
                .filter(|(_, j)| j.done_rx.borrow().is_some())
                .map(|(k, _)| k.clone())
                .collect();
            for k in finished {
                map.remove(&k);
            }
        }
        map.insert(job.id.clone(), job);
    }

    fn get(&self, id: &str) -> Option<Arc<BackgroundSubagent>> {
        self.jobs.lock().expect("lock").get(id).cloned()
    }

    fn remove(&self, id: &str) -> Option<Arc<BackgroundSubagent>> {
        self.jobs.lock().expect("lock").remove(id)
    }

    fn list_summaries(&self) -> Vec<String> {
        let map = self.jobs.lock().expect("lock");
        let mut items: Vec<&Arc<BackgroundSubagent>> = map.values().collect();
        items.sort_by(|a, b| a.id.cmp(&b.id));
        items.into_iter().map(|j| j.progress_line()).collect()
    }

    pub fn kill_all(&self) {
        if let Ok(map) = self.jobs.lock() {
            for j in map.values() {
                j.cancel.cancel();
            }
        }
    }
}

impl Drop for SubagentRegistry {
    fn drop(&mut self) {
        self.kill_all();
    }
}

#[derive(Debug)]
struct SubagentTool {
    parent_config: CarnotConfig,
    provider: Arc<dyn Provider>,
    extra_modules: Vec<Arc<dyn Module>>,
    registry: Arc<SubagentRegistry>,
}

impl SubagentTool {
    fn prepare_child_config(&self, input: &Value) -> (CarnotConfig, bool) {
        let isolated = input
            .get("isolated_worktree")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let max_turns = input
            .get("max_turns")
            .and_then(Value::as_u64)
            .unwrap_or(25)
            .clamp(1, 80);

        let mut child_cfg = self.parent_config.clone();
        child_cfg.subagent_depth = self.parent_config.subagent_depth.saturating_add(1);
        child_cfg.max_turns = max_turns;
        if let Some(m) = input.get("model").and_then(Value::as_str)
            && !m.trim().is_empty()
        {
            let overridden = m.trim().to_string();
            if let Ok(cat) = carnot_ai::catalog::ProviderCatalog::embedded()
                && let Some(entry) = cat.find_model(&overridden)
            {
                if let Some(cw) = entry.context_window {
                    child_cfg.context_window = cw;
                }
                if let Some(mo) = entry.max_output_tokens {
                    child_cfg.max_output_tokens = mo;
                }
            }
            child_cfg.model = overridden;
        }
        (child_cfg, isolated)
    }

    fn build_child_agent(&self, child_cfg: &CarnotConfig) -> Result<Arc<Agent>> {
        let cas = Arc::new(CasStore::open(&child_cfg.workspace_root)?);
        let tools = ToolRegistry::with_canonical_lid_tools(child_cfg, cas);

        // Preserve byte-identical tool list with parent (including Subagent schema!) so child hits
        // parent's prompt cache on Turn 1; recursion is blocked in `before_tool_call` (Opus #3!).
        // Put `extra_modules` (which includes PolicyGuardModule) AFTER SubagentModule so policy
        // gates ALWAYS inspect effective inputs last (Opus #2!).
        let mut child_modules: Vec<Arc<dyn Module>> = vec![Arc::new(SubagentModule::new(
            child_cfg.clone(),
            self.provider.clone(),
            self.extra_modules.clone(),
        ))];
        child_modules.extend(self.extra_modules.iter().cloned());

        Ok(Arc::new(Agent::new(
            child_cfg.clone(),
            self.provider.clone(),
            tools,
            child_modules,
            false,
        )?))
    }
}

fn finish_subagent_report(
    child_cfg: &CarnotConfig,
    parent_ws: &std::path::Path,
    isolated: bool,
    fork_guard: Option<IsolatedForkWorktree>,
    run_res: Result<crate::RunOutcome>,
) -> SubagentFinished {
    let ws_status = collect_workspace_status(&child_cfg.workspace_root);
    let diff_handle = if isolated {
        export_isolated_diff_to_parent_cas(&child_cfg.workspace_root, parent_ws)
    } else {
        None
    };
    if let Some(fk) = fork_guard {
        fk.cleanup();
    }

    match run_res {
        Ok(outcome) => {
            let mut summary = outcome.final_text;
            if summary.chars().count() > MAX_SUBAGENT_SUMMARY_CHARS {
                summary = summary.chars().take(MAX_SUBAGENT_SUMMARY_CHARS).collect();
                summary.push_str("\n[... subagent summary truncated ...]");
            }
            let mut report = format!(
                "[subagent-report model={} turns={} cost=${:.4}{}]\n{}",
                child_cfg.model,
                outcome.metrics.turns,
                outcome.metrics.total_usage.cost_usd,
                if isolated {
                    " isolated_worktree=true"
                } else {
                    ""
                },
                summary
            );
            if let Some(st) = ws_status {
                report.push_str(&format!("\n[subagent-workspace-changes]\n{st}"));
            }
            if let Some(ref dh) = diff_handle {
                report.push_str(&format!(
                    "\n[subagent-diff-patch] Saved isolated worktree diff to `{dh}` (apply via `git apply {dh}`)"
                ));
            }
            SubagentFinished {
                report,
                is_error: false,
                model: child_cfg.model.clone(),
                turns: outcome.metrics.turns,
                cost_usd: outcome.metrics.total_usage.cost_usd,
                isolated,
                diff_handle,
            }
        }
        Err(e) => SubagentFinished {
            report: format!("Subagent failed: {e}"),
            is_error: true,
            model: child_cfg.model.clone(),
            turns: 0,
            cost_usd: 0.0,
            isolated,
            diff_handle,
        },
    }
}

fn finished_to_tool_output(fin: SubagentFinished, job_id: Option<&str>) -> ToolOutput {
    let details = json!({
        "job_id": job_id,
        "model": fin.model,
        "turns": fin.turns,
        "cost_usd": fin.cost_usd,
        "isolated_worktree": fin.isolated,
        "diff_handle": fin.diff_handle,
        "finished": true,
    });
    if fin.is_error {
        ToolOutput::err_text(fin.report).with_details(details)
    } else {
        ToolOutput::ok_text(fin.report).with_details(details)
    }
}

impl Tool for SubagentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "Subagent".into(),
            description: "Spawn an isolated child Carnot agent with a disposable context window (sync or non-blocking `background: true`). Poll/collect/wait/kill background agents via `job`, or inspect active ones via `list: true`.".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "task": { "type": "string", "description": "Clear, self-contained task description and expected output" },
                    "model": { "type": "string", "description": "Optional cheaper/faster model override (e.g. claude-haiku-5-5)" },
                    "max_turns": { "type": "integer", "description": "Max turns for child agent (default 25, max 80)" },
                    "isolated_worktree": { "type": "boolean", "description": "Run inside an isolated detached git worktree + dedicated CARGO_TARGET_DIR" },
                    "background": { "type": "boolean", "description": "Spawn non-blocking in the background and return a `sub-N` handle immediately" },
                    "job": { "type": "string", "description": "Background subagent job ID to poll, wait for, or kill (e.g. `sub-1`)" },
                    "wait": { "type": "boolean", "description": "With `job`: block until the background subagent finishes (default false = instant progress poll)" },
                    "timeout": { "type": "integer", "description": "With `job` + `wait`: max seconds to wait (default 120, max 600)" },
                    "kill": { "type": "boolean", "description": "With `job`: stop and cancel the running background subagent" },
                    "list": { "type": "boolean", "description": "List all active and uncollected background subagents" }
                }
            }),
            defer_loading: false,
        }
    }

    fn effects(&self, input: &Value) -> ToolEffects {
        let is_management = input.get("job").is_some()
            || input.get("list").and_then(Value::as_bool).unwrap_or(false)
            || input
                .get("background")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        let isolated = input
            .get("isolated_worktree")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if is_management || isolated {
            // Concurrent launches/polls/isolated worktrees never block each other, and NEVER run
            // speculatively mid-stream (paid LLM side effects — Opus #8!).
            ToolEffects::global_read_no_eager()
        } else {
            // Shared workspace synchronous runs MUST serialize with parent edits (Opus point #3!)
            ToolEffects::serial()
        }
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            // Mode A: List active/uncollected background subagents
            if input.get("list").and_then(Value::as_bool).unwrap_or(false) {
                let rows = self.registry.list_summaries();
                if rows.is_empty() {
                    return Ok(ToolOutput::ok_text("No active background subagents."));
                }
                return Ok(ToolOutput::ok_text(rows.join("\n")).with_details(json!({
                    "count": rows.len(),
                })));
            }

            // Mode B: Poll / Wait / Kill an existing background subagent
            if let Some(job_id) = input.get("job").and_then(Value::as_str) {
                let Some(job) = self.registry.get(job_id) else {
                    return Ok(ToolOutput::err_text(format!(
                        "Unknown subagent job id: {job_id}"
                    )));
                };
                if input.get("kill").and_then(Value::as_bool).unwrap_or(false) {
                    job.cancel.cancel();
                    let accrued_cost = job.cost_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;
                    self.registry.remove(job_id);
                    return Ok(ToolOutput::ok_text(format!(
                        "Killed background subagent {job_id} ({}) [cost_so_far=${accrued_cost:.4}]",
                        job.task_preview
                    ))
                    .with_details(json!({
                        "job_id": job_id,
                        "killed": true,
                        "cost_usd": accrued_cost,
                    })));
                }

                let wait = input.get("wait").and_then(Value::as_bool).unwrap_or(false);
                let mut rx = job.done_rx.clone();
                if wait && rx.borrow().is_none() {
                    let wait_secs = input
                        .get("timeout")
                        .and_then(Value::as_u64)
                        .unwrap_or(120)
                        .clamp(1, 600);
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            return Ok(ToolOutput::err_text(format!(
                                "Cancelled while waiting for subagent {job_id}"
                            )));
                        }
                        _ = tokio::time::timeout(
                            Duration::from_secs(wait_secs),
                            rx.wait_for(|v| v.is_some()),
                        ) => {}
                    }
                }

                if let Some(fin) = rx.borrow().clone() {
                    self.registry.remove(job_id);
                    return Ok(finished_to_tool_output(fin, Some(job_id)));
                }

                let cost_so_far = job.cost_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;
                return Ok(ToolOutput::ok_text(format!(
                    "[{} — still running; call Subagent{{\"job\":\"{job_id}\",\"wait\":true}} to wait or {{\"job\":\"{job_id}\",\"kill\":true}} to cancel]",
                    job.progress_line()
                ))
                .with_details(json!({
                    "job_id": job_id,
                    "finished": false,
                    "turns": job.turns_completed.load(Ordering::Relaxed),
                    "tools_called": job.tools_called.load(Ordering::Relaxed),
                    "cost_so_far_usd": cost_so_far,
                })));
            }

            // Mode C: Spawn a new child subagent (synchronous OR non-blocking background)
            if self.parent_config.subagent_depth >= self.parent_config.subagent_max_depth {
                return Ok(ToolOutput::err_text(format!(
                    "Subagent recursion depth limit ({}) reached; complete the task directly.",
                    self.parent_config.subagent_max_depth
                )));
            }
            let Some(task) = input
                .get("task")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
            else {
                return Ok(ToolOutput::err_text(
                    "Subagent requires `task` (or `job` / `list: true`)",
                ));
            };
            let background = input
                .get("background")
                .and_then(Value::as_bool)
                .unwrap_or(false);

            let (mut child_cfg, isolated) = self.prepare_child_config(&input);

            let mut fork_guard: Option<IsolatedForkWorktree> = None;
            if isolated {
                let fork_id = uuid::Uuid::new_v4().simple().to_string();
                let fk =
                    IsolatedForkWorktree::create(&self.parent_config.workspace_root, &fork_id)?;
                child_cfg.workspace_root = fk.worktree_path.clone();
                child_cfg.extra_env.insert(
                    "CARGO_TARGET_DIR".into(),
                    fk.cargo_target_dir.to_string_lossy().to_string(),
                );
                fork_guard = Some(fk);
            }

            let child_agent = self.build_child_agent(&child_cfg)?;
            let parent_ws = self.parent_config.workspace_root.clone();

            if background {
                let job_id = self.registry.next_id();
                let task_preview: String = task.chars().take(80).collect();
                let turns_completed = Arc::new(AtomicU64::new(0));
                let tools_called = Arc::new(AtomicU64::new(0));
                let cost_micros = Arc::new(AtomicU64::new(0));
                let last_activity = Arc::new(Mutex::new("starting".to_string()));
                let job_cancel = CancellationToken::new();
                let (done_tx, done_rx) = watch::channel::<Option<SubagentFinished>>(None);

                let bg_entry = Arc::new(BackgroundSubagent {
                    id: job_id.clone(),
                    task_preview: task_preview.clone(),
                    model: child_cfg.model.clone(),
                    isolated,
                    started: Instant::now(),
                    turns_completed: Arc::clone(&turns_completed),
                    tools_called: Arc::clone(&tools_called),
                    cost_micros: Arc::clone(&cost_micros),
                    last_activity: Arc::clone(&last_activity),
                    cancel: job_cancel.clone(),
                    done_rx,
                });
                self.registry.insert(bg_entry);

                // Live event observer updating progress counters while the child runs!
                let mut ev_stream = child_agent.subscribe();
                let tc = Arc::clone(&turns_completed);
                let tools_c = Arc::clone(&tools_called);
                let cm = Arc::clone(&cost_micros);
                let la = Arc::clone(&last_activity);
                let obs_task = tokio::spawn(async move {
                    loop {
                        let env = ev_stream.next().await;
                        match env.event {
                            AgentEvent::TurnStart { turn, .. } => {
                                tc.store(turn, Ordering::Relaxed);
                                if let Ok(mut g) = la.lock() {
                                    *g = format!("turn:{turn}");
                                }
                            }
                            AgentEvent::ToolStart { tool, .. } => {
                                tools_c.fetch_add(1, Ordering::Relaxed);
                                if let Ok(mut g) = la.lock() {
                                    *g = format!("tool:{tool}");
                                }
                            }
                            AgentEvent::UsageRecorded { usage, .. } => {
                                let delta = (usage.cost_usd.max(0.0) * 1_000_000.0) as u64;
                                cm.fetch_add(delta, Ordering::Relaxed);
                            }
                            AgentEvent::RunSettled { .. } | AgentEvent::Error { .. } => {
                                break;
                            }
                            _ => {}
                        }
                    }
                });

                // Spawn background execution + cancellation watcher + deterministic worktree cleanup
                let ca_abort = Arc::clone(&child_agent);
                let jc = job_cancel.clone();
                tokio::spawn(async move {
                    let cancel_watcher = tokio::spawn(async move {
                        jc.cancelled().await;
                        ca_abort.abort().await;
                    });
                    let run_res = child_agent.prompt(task).await;
                    cancel_watcher.abort();
                    obs_task.abort();
                    let finished = finish_subagent_report(
                        &child_cfg, &parent_ws, isolated, fork_guard, run_res,
                    );
                    let _ = done_tx.send(Some(finished));
                });

                return Ok(ToolOutput::ok_text(format!(
                    "[subagent spawned -> {job_id} running in background (model={}, isolated_worktree={isolated}) — continue working and call Subagent{{\"job\":\"{job_id}\"}} to poll, {{\"job\":\"{job_id}\",\"wait\":true}} to collect when done, or {{\"job\":\"{job_id}\",\"kill\":true}} to stop]",
                    self.registry
                        .get(&job_id)
                        .map(|j| j.model.clone())
                        .unwrap_or_default(),
                ))
                .with_details(json!({
                    "job_id": job_id,
                    "background": true,
                    "isolated_worktree": isolated,
                    "finished": false,
                })));
            }

            // Synchronous mode: propagate parent cancellation directly into child!
            let child_cancel = cancel.child_token();
            let ca_abort = child_agent.clone();
            let cancel_watcher = tokio::spawn(async move {
                child_cancel.cancelled().await;
                ca_abort.abort().await;
            });

            let run_res = child_agent.prompt(task).await;
            cancel_watcher.abort();

            let fin = finish_subagent_report(&child_cfg, &parent_ws, isolated, fork_guard, run_res);
            Ok(finished_to_tool_output(fin, None))
        })
    }
}

fn export_isolated_diff_to_parent_cas(
    child_ws: &std::path::Path,
    parent_ws: &std::path::Path,
) -> Option<String> {
    // Stage intent-to-add inside this disposable worktree so newly created untracked files are
    // included in `git diff --binary` before the worktree is removed!
    let _ = carnot_sandbox::host_safe_git_command(child_ws)
        .args(["add", "-N", "."])
        .output();
    let out = carnot_sandbox::host_safe_git_command(child_ws)
        .args(["diff", "--binary"])
        .output()
        .ok()?;
    if !out.status.success() || out.stdout.is_empty() {
        return None;
    }
    let cas = CasStore::open(parent_ws).ok()?;
    let diff_text = String::from_utf8_lossy(&out.stdout);
    let handle = cas.put_text(&diff_text).ok()?;
    Some(handle.rel_path)
}

fn collect_workspace_status(ws: &std::path::Path) -> Option<String> {
    let out = carnot_sandbox::host_safe_git_command(ws)
        .args(["status", "--short"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

/// First-class module registering the `Subagent` tool (sync + non-blocking background) and enforcing
/// recursion-depth gating.
#[derive(Debug)]
pub struct SubagentModule {
    depth: u8,
    max_depth: u8,
    registry: Arc<SubagentRegistry>,
    tool: Arc<dyn Tool>,
}

impl SubagentModule {
    pub fn new(
        config: CarnotConfig,
        provider: Arc<dyn Provider>,
        extra_modules: Vec<Arc<dyn Module>>,
    ) -> Self {
        let depth = config.subagent_depth;
        let max_depth = config.subagent_max_depth;
        let registry = Arc::new(SubagentRegistry::new());
        let tool = Arc::new(SubagentTool {
            parent_config: config,
            provider,
            extra_modules,
            registry: Arc::clone(&registry),
        });
        Self {
            depth,
            max_depth,
            registry,
            tool,
        }
    }

    /// Cancels all running background subagents managed by this module.
    pub fn kill_all_background(&self) {
        self.registry.kill_all();
    }
}

impl Module for SubagentModule {
    fn name(&self) -> &str {
        "subagent"
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![self.tool.clone()]
    }

    fn before_tool_call(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> ExtFuture<'_, Result<ToolGateDecision>> {
        // Block spawning NEW child subagents at max depth, while allowing job polling/listing!
        let is_spawn = input.get("job").is_none()
            && !input.get("list").and_then(Value::as_bool).unwrap_or(false);
        let blocked = tool_name == "Subagent" && is_spawn && self.depth >= self.max_depth;
        let max_d = self.max_depth;
        Box::pin(async move {
            if blocked {
                Ok(ToolGateDecision::Deny {
                    reason: format!("subagent max recursion depth ({max_d}) reached"),
                })
            } else {
                Ok(ToolGateDecision::Allow)
            }
        })
    }
}
