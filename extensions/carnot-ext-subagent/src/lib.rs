//! Unified, Long-Run-Ready Multi-Agent Orchestration & Subagent Extension
//! (`extensions/carnot-ext-subagent`, opt-in via `--features ext-subagent` / `--subagent`).
//!
//! Architecture invariants:
//!   - **Two tools, identical on coordinator & workers** (`Subagent` + `SendMessage`) so
//!     `prefix_hash` stays byte-identical when parent and worker share the same tool set
//!     (`fork: true` is cache-warm in that configuration, and always shares full conversation
//!     context). Role boundaries are enforced in `before_tool_call` (workers are blocked from all
//!     `Subagent` actions and restricted to hub-and-spoke `SendMessage`).
//!   - **Multi-turn persistent workers (`keep_alive: true`)**: live `Arc<Agent>` + persistent
//!     isolated git worktree stay warm between turns; `SendMessage` wakes idle workers via
//!     `prompt_with_cancel()` or steers/queues running workers via `steer()`/`follow_up()`.
//!   - **Incremental atomic worktree landing (`land: true`)**: holds `worker.run_lock`, diffs
//!     strictly against `base_sha` (capturing parent's dirty/untracked files at fork time), checks
//!     `git apply --check --binary` from `repo_root`, and advances `base_sha` after every land.
//!   - **Zero-CPU reactive coordination for multi-hour runs**: `wait` supports `any`/`all` over
//!     multiple workers via `tokio::sync::Notify`, and `on_turn_boundary` wakes reactively on
//!     worker completion, worker messages, user `steer`, or 240s cache-safe heartbeats (each
//!     wakeup resets `idle_turn_streak` while workers are live; `max_turns` and `max_cost_usd`
//!     bound total runtime) — delivering worker updates strictly as untrusted `Message::system_note`.
//!   - **Acyclic ownership & unified budget**: coordinator owns `Arc<Fleet>`, workers hold
//!     `Weak<Fleet>` so `Fleet::drop()` runs deterministically on session exit. `FleetBudget`
//!     reads the parent `Agent`'s unified `total_cost_micros` counter (parent + compaction + all
//!     sync and background workers).

pub mod budget;
pub mod fleet;

use self::budget::FleetBudget;
use self::fleet::{
    DelegatingPermissionBroker, Fleet, MAX_SUBAGENT_SUMMARY_CHARS, WorkerHandle, WorkerReport,
    WorkerState, report_to_details, spawn_worker_turn,
};
use carnot_agent::Agent;
use carnot_ai::{Provider, registry::ProviderRegistry};
use carnot_cas::CasStore;
use carnot_core::{
    AgentEvent, CacheTtl, CarnotConfig, CarnotError, ReasoningEffort, Result, ToolEffects,
    ToolOutput, ToolSpec,
};
use carnot_ext::{
    BoundaryAction, BoxFuture as ExtFuture, Module, ToolGateDecision, TurnBoundarySnapshot,
    warm_fork::{IsolatedForkWorktree, LandOutcome},
};
use carnot_tools::{BoxFuture, Tool, ToolRegistry};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

/// 4-minute heartbeat stays strictly under typical 5-minute provider prompt-cache TTLs.
const HEARTBEAT_SECS: u64 = 240;

/// Acyclic fleet reference: coordinator owns `Strong(Arc<Fleet>)`, workers hold `Weak(Weak<Fleet>)`
/// so dropping the coordinator `Agent` deterministically drops `Fleet` and cleans up all worktrees.
#[derive(Debug, Clone)]
enum FleetRef {
    Strong(Arc<Fleet>),
    Weak(Weak<Fleet>),
}

impl FleetRef {
    fn upgrade(&self) -> Option<Arc<Fleet>> {
        match self {
            Self::Strong(f) => Some(Arc::clone(f)),
            Self::Weak(w) => w.upgrade(),
        }
    }
}

fn non_empty_str<'a>(input: &'a Value, key: &str) -> Option<&'a str> {
    input
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn non_empty_jobs(input: &Value) -> Vec<String> {
    input
        .get("jobs")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_effort(s: &str) -> Option<ReasoningEffort> {
    match s.trim().to_ascii_lowercase().as_str() {
        "low" => Some(ReasoningEffort::Low),
        "medium" | "med" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::Xhigh),
        "max" => Some(ReasoningEffort::Max),
        _ => None,
    }
}

fn sanitize_worker_name(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("coordinator") {
        return None;
    }
    let clean: String = trimmed
        .chars()
        .take(32)
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    if clean.is_empty() || clean == "coordinator" {
        None
    } else {
        Some(clean)
    }
}

/// Lightweight schema-mirroring placeholder registered on `fork: true` workers for any
/// coordinator-only tool (Web, Memory, LSP, MCP, custom extra tools) so the worker's `tools`
/// wire prefix stays 100% byte-identical with the parent's warm KV-cache without duplicating
/// stateful servers or exposing unauthorized side effects.
#[derive(Debug)]
struct MirroredCoordinatorTool {
    spec: ToolSpec,
}

impl Tool for MirroredCoordinatorTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn effects(&self, _input: &Value) -> ToolEffects {
        ToolEffects::global_read_no_eager()
    }

    fn execute(
        &self,
        _input: Value,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        let name = self.spec.name.clone();
        Box::pin(async move {
            Ok(ToolOutput::err_text(format!(
                "Tool `{name}` is only available on the coordinator; use `SendMessage` to ask `coordinator` if you need it."
            )))
        })
    }
}

#[derive(Debug)]
struct SubagentTool {
    parent_config: CarnotConfig,
    provider: Arc<dyn Provider>,
    provider_configs: Vec<PathBuf>,
    parent_agent: Arc<OnceLock<Weak<Agent>>>,
    extra_modules: Vec<Arc<dyn Module>>,
    fleet: FleetRef,
}

struct ChildRuntimePlan {
    name: String,
    config: CarnotConfig,
    provider: Arc<dyn Provider>,
    provider_id: String,
    isolated: bool,
    fork: bool,
    keep_alive: bool,
}

impl SubagentTool {
    async fn prepare_child_plan(
        &self,
        fleet: &Fleet,
        input: &Value,
        background: bool,
    ) -> Result<ChildRuntimePlan> {
        let isolated = input
            .get("isolated_worktree")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let fork = input.get("fork").and_then(Value::as_bool).unwrap_or(false);
        let explicit_name = input
            .get("name")
            .and_then(Value::as_str)
            .and_then(sanitize_worker_name);
        let keep_alive = input
            .get("keep_alive")
            .and_then(Value::as_bool)
            .unwrap_or(background && explicit_name.is_some());
        let name = explicit_name.unwrap_or_else(|| fleet.next_auto_name());
        let max_turns = input
            .get("max_turns")
            .and_then(Value::as_u64)
            .filter(|&v| v > 0)
            .unwrap_or(40)
            .clamp(1, 200);

        let mut child_cfg = self.parent_config.clone();
        child_cfg.subagent_depth = self.parent_config.subagent_depth.saturating_add(1);
        child_cfg.max_turns = max_turns;

        // Inherit live parent model/effort/cache_ttl/provider when attached (reflects live switches)
        let mut live_provider = Arc::clone(&self.provider);
        let mut parent_cache_ttl = self.parent_config.cache_ttl;
        if let Some(parent) = self.parent_agent.get().and_then(Weak::upgrade) {
            child_cfg.model = parent.current_model().await;
            child_cfg.effort = parent.current_effort().await;
            child_cfg.context_window = parent.current_context_window().await;
            parent_cache_ttl = parent.current_cache_ttl().await;
            live_provider = parent.current_provider().await;
        }
        child_cfg.cache_ttl = subagent_cache_ttl(parent_cache_ttl, fork, keep_alive);
        let live_provider_id = live_provider.name().to_string();

        if let Some(eff_str) = input.get("effort").and_then(Value::as_str)
            && let Some(eff) = parse_effort(eff_str)
        {
            child_cfg.effort = eff;
        }

        let req_provider = input
            .get("provider")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let req_model = input
            .get("model")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty());

        let (resolved_provider, resolved_provider_id) = if req_provider.is_none()
            && req_model.is_none()
        {
            (live_provider, live_provider_id)
        } else if let Ok(reg) = ProviderRegistry::load_default(&self.provider_configs) {
            let target_prov_name = if let Some(p) = req_provider {
                reg.catalog()
                    .find_provider(p)
                    .map(|(canon, _)| canon.to_string())
                    .unwrap_or_else(|| p.to_string())
            } else if let Some(m) = req_model
                && let Some(routed) = reg.catalog().provider_for_model(m)
            {
                routed.to_string()
            } else {
                live_provider_id.clone()
            };

            let has_model_routes = reg
                .catalog()
                .find_provider(&target_prov_name)
                .map(|(_, e)| !e.model_routes.is_empty())
                .unwrap_or(false);
            let same_model = req_model.is_none() || req_model == Some(child_cfg.model.as_str());

            if target_prov_name.eq_ignore_ascii_case(&live_provider_id)
                && (same_model || !has_model_routes)
            {
                if let Some(m) = req_model {
                    child_cfg.model = m.to_string();
                    if let Some(cw) = reg.context_window_for_model(m) {
                        child_cfg.context_window = cw;
                    }
                    if let Some(mo) = reg.max_output_tokens_for_model(m) {
                        child_cfg.max_output_tokens = mo;
                    }
                }
                let safe_max_out =
                    ((child_cfg.context_window / 2).max(4096)).min(u32::MAX as u64) as u32;
                if child_cfg.max_output_tokens > safe_max_out {
                    child_cfg.max_output_tokens = safe_max_out;
                }
                (live_provider, live_provider_id)
            } else {
                let target = reg.resolve_and_build(req_provider, req_model, &live_provider_id)?;
                child_cfg.model = target.model_id;
                if let Some(cw) = target.context_window {
                    child_cfg.context_window = cw;
                }
                if let Some(mo) = target.max_output_tokens {
                    child_cfg.max_output_tokens = mo;
                }
                let safe_max_out =
                    ((child_cfg.context_window / 2).max(4096)).min(u32::MAX as u64) as u32;
                if child_cfg.max_output_tokens > safe_max_out {
                    child_cfg.max_output_tokens = safe_max_out;
                }
                (target.provider, target.provider_id)
            }
        } else {
            if let Some(m) = req_model {
                child_cfg.model = m.to_string();
            }
            (live_provider, live_provider_id)
        };

        Ok(ChildRuntimePlan {
            name,
            config: child_cfg,
            provider: resolved_provider,
            provider_id: resolved_provider_id,
            isolated,
            fork,
            keep_alive,
        })
    }

    async fn build_child_agent(
        &self,
        fleet: &Arc<Fleet>,
        plan: &ChildRuntimePlan,
    ) -> Result<Arc<Agent>> {
        let cas = Arc::new(CasStore::open(&plan.config.workspace_root)?);
        let mut tools = ToolRegistry::with_canonical_lid_tools(&plan.config, cas);

        // Preserve byte-identical tool schemas with parent (`Subagent` + `SendMessage` + any
        // mirrored coordinator tools on `fork: true`) so forked workers hit the parent's warm
        // prompt cache on Turn 1 even when Web/Memory/LSP/MCP extensions are active!
        // Put `extra_modules` (Hooks, Claude plugins, custom modules, PolicyGuard) AFTER
        // SubagentModule so policy gates always inspect effective inputs last.
        let child_sub_mod = SubagentModule::for_worker(
            plan.name.clone(),
            plan.config.clone(),
            Arc::clone(&plan.provider),
            self.provider_configs.clone(),
            self.extra_modules.clone(),
            Arc::downgrade(fleet),
            Arc::clone(&fleet.budget),
            Arc::clone(&self.parent_agent),
        );
        let mut child_modules: Vec<Arc<dyn Module>> = vec![Arc::new(child_sub_mod)];
        child_modules.extend(self.extra_modules.iter().cloned());

        let child = if plan.fork
            && let Some(parent) = self.parent_agent.get().and_then(Weak::upgrade)
        {
            let snap = parent.fork_snapshot_with_prefix().await;
            let mut fork_cfg = plan.config.clone();
            fork_cfg.native_web = snap.native_web;

            // Mirror any coordinator-only tool specs not already provided by child's canonical
            // tools or modules (`AskUser`, `ToolSearch`, `CallDeferredTool` are registered
            // automatically by `Agent::new_forked_with_cache_key` with matching deferred set).
            let mut known: std::collections::HashSet<String> =
                tools.specs().into_iter().map(|s| s.name).collect();
            for m in &child_modules {
                for t in m.tools() {
                    known.insert(t.spec().name);
                }
            }
            if fork_cfg.enable_ask_user {
                known.insert("AskUser".to_string());
            }
            known.insert("ToolSearch".to_string());
            known.insert("CallDeferredTool".to_string());

            for spec in snap.tool_specs {
                if !known.contains(&spec.name) {
                    known.insert(spec.name.clone());
                    tools.register(Arc::new(MirroredCoordinatorTool { spec }));
                }
            }

            Arc::new(Agent::new_forked_with_cache_key(
                fork_cfg,
                Arc::clone(&plan.provider),
                tools,
                child_modules,
                false,
                &snap.messages,
                snap.checkpoint_index,
                Some(snap.cache_key_id),
            )?)
        } else {
            Arc::new(Agent::new(
                plan.config.clone(),
                Arc::clone(&plan.provider),
                tools,
                child_modules,
                false,
            )?)
        };

        // Delegate parent's PermissionBroker into the child (preserving original tool_name and
        // tagging `_carnot_worker` on input) so subagents never bypass user approvals.
        if let Some(parent) = self.parent_agent.get().and_then(Weak::upgrade)
            && let Some(parent_broker) = parent.permission_broker().await
        {
            child
                .set_permission_broker(Arc::new(DelegatingPermissionBroker::new(
                    plan.name.clone(),
                    parent_broker,
                )))
                .await;
        }

        Ok(child)
    }

    async fn handle_land(&self, fleet: &Fleet, job_id: &str) -> Result<ToolOutput> {
        let Some(worker) = fleet.get(job_id) else {
            return Ok(ToolOutput::err_text(format!(
                "Unknown subagent/worker `{job_id}`"
            )));
        };
        if worker.state().is_running() {
            return Ok(ToolOutput::err_text(format!(
                "Worker `{job_id}` is still running — wait for it to settle before landing its worktree."
            )));
        }
        // Hold `run_lock` across diff + apply + baseline commit so a concurrent turn cannot
        // write into the worktree mid-land.
        let Ok(_turn_guard) = worker.run_lock.try_lock() else {
            return Ok(ToolOutput::err_text(format!(
                "Worker `{job_id}` is busy executing a turn — wait for it to settle before landing its worktree."
            )));
        };
        let w_clone = Arc::clone(&worker);
        let land_res = tokio::task::spawn_blocking(move || w_clone.land_worktree())
            .await
            .map_err(|e| CarnotError::Tool {
                tool: "Subagent".into(),
                message: format!("worktree land task failed: {e}"),
            })??;

        match land_res {
            LandOutcome::NoChanges => Ok(ToolOutput::ok_text(format!(
                "Worker `{job_id}` has no changes to land."
            ))
            .with_details(json!({ "job_id": job_id, "landed": false, "reason": "no_changes" }))),
            LandOutcome::Applied { stat } => Ok(ToolOutput::ok_text(format!(
                "Landed worker `{job_id}` changes cleanly into parent workspace:\n{stat}"
            ))
            .with_details(json!({ "job_id": job_id, "landed": true, "stat": stat }))),
            LandOutcome::Conflict { stderr, diff_bytes } => {
                let ws_root = self.parent_config.workspace_root.clone();
                let cas_path = tokio::task::spawn_blocking(move || {
                    CasStore::open(&ws_root).ok().and_then(|cas| {
                        cas.put_text(&String::from_utf8_lossy(&diff_bytes))
                            .ok()
                            .map(|h| h.rel_path)
                    })
                })
                .await
                .ok()
                .flatten();
                let mut msg = format!(
                    "Conflict checking worker `{job_id}` patch (parent workspace was NOT modified):\n{stderr}"
                );
                if let Some(ref cp) = cas_path {
                    msg.push_str(&format!(
                        "\nPatch saved to `{cp}` (you can resolve via `git apply --3way {cp}` or steer `{job_id}` via `SendMessage`)."
                    ));
                }
                Ok(ToolOutput::err_text(msg).with_details(json!({
                    "job_id": job_id,
                    "landed": false,
                    "conflict": true,
                    "diff_handle": cas_path,
                })))
            }
        }
    }
}

impl Tool for SubagentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "Subagent".into(),
            description: "Spawn and orchestrate isolated child Carnot agents (sync or non-blocking `background: true`, fresh or `fork: true`, cross-provider `provider`/`model`/`effort`, persistent multi-turn `keep_alive: true`, isolated git worktrees with atomic `land: true`). Use `list: true` for fleet status, `job`/`jobs` + `wait: true` (`wait_mode: any|all`) to wait, `kill: true` to close, and `SendMessage` to talk to named workers.".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "task": { "type": "string", "description": "Clear task description and expected output" },
                    "name": { "type": "string", "description": "Optional friendly worker name (e.g. `backend`, `reviewer`); defaults to `sub-N`. With `background: true`, implies `keep_alive: true`." },
                    "provider": { "type": "string", "description": "Optional target provider ID or alias from catalog (auto-routed from `model` when omitted)" },
                    "model": { "type": "string", "description": "Optional model override (same provider or cross-provider, from `[available-subagent-models]`)" },
                    "effort": { "type": "string", "enum": ["low", "medium", "high", "xhigh", "max"], "description": "Optional reasoning effort override" },
                    "fork": { "type": "boolean", "description": "Inherit a sanitized snapshot of parent conversation history (default false; true = cache-warm fork on same model or full-context cross-provider review)" },
                    "keep_alive": { "type": "boolean", "description": "Keep worker & its worktree alive after finishing its turn so you can follow up with `SendMessage` and/or `land: true`" },
                    "max_turns": { "type": "integer", "description": "Max turns per worker run (default 40, max 200)" },
                    "isolated_worktree": { "type": "boolean", "description": "Run inside an isolated detached git worktree + baseline commit + dedicated CARGO_TARGET_DIR" },
                    "background": { "type": "boolean", "description": "Spawn non-blocking in the background and return a handle immediately" },
                    "job": { "type": "string", "description": "Worker name or `sub-N` ID to poll, wait for, `land`, or `kill`" },
                    "jobs": { "type": "array", "items": { "type": "string" }, "description": "Multiple worker names/IDs to wait on together" },
                    "wait": { "type": "boolean", "description": "Block with 0 CPU until target worker(s) finish or send a message (default false = instant poll)" },
                    "wait_mode": { "type": "string", "enum": ["any", "all"], "description": "With `wait: true` and multiple/all workers: wake on first completion (`any`, default) or wait for `all`" },
                    "timeout": { "type": "integer", "description": "With `wait: true`: max seconds to wait (default 180, max 1800)" },
                    "land": { "type": "boolean", "description": "With `job`: atomically check & apply the worker's isolated worktree diff onto the parent workspace" },
                    "kill": { "type": "boolean", "description": "With `job`: stop worker and clean up its worktree" },
                    "list": { "type": "boolean", "description": "Fleet status overview (all workers, states, models, turns, individual & total fleet cost)" }
                }
            }),
            defer_loading: false,
        }
    }

    fn effects(&self, input: &Value) -> ToolEffects {
        let is_management = non_empty_str(input, "job").is_some()
            || !non_empty_jobs(input).is_empty()
            || input.get("list").and_then(Value::as_bool).unwrap_or(false)
            || input.get("wait").and_then(Value::as_bool).unwrap_or(false)
            || input
                .get("background")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        let is_land = input.get("land").and_then(Value::as_bool).unwrap_or(false);
        let isolated = input
            .get("isolated_worktree")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if is_land {
            // Landing writes into the parent workspace -> serialize with parent edits
            ToolEffects::serial()
        } else if is_management || isolated {
            ToolEffects::global_read_no_eager()
        } else {
            ToolEffects::serial()
        }
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let Some(fleet) = self.fleet.upgrade() else {
                return Ok(ToolOutput::err_text(
                    "Fleet coordinator is no longer active.",
                ));
            };

            // Mode A: Fleet status overview (`list: true`)
            if input.get("list").and_then(Value::as_bool).unwrap_or(false) {
                let (rows, total_cost, running) = fleet.list_overview();
                if rows.is_empty() {
                    return Ok(ToolOutput::ok_text("No active background subagents."));
                }
                let limit_str = fleet
                    .budget
                    .limit_usd()
                    .map(|l| format!(" / ${l:.2} cap"))
                    .unwrap_or_default();
                let header = format!(
                    "[fleet: {} workers ({} running) · total_cost=${:.4}{}]",
                    rows.len(),
                    running,
                    total_cost,
                    limit_str
                );
                return Ok(
                    ToolOutput::ok_text(format!("{header}\n{}", rows.join("\n"))).with_details(
                        json!({
                            "count": rows.len(),
                            "running": running,
                            "fleet_cost_usd": total_cost,
                        }),
                    ),
                );
            }

            // Mode B: Atomic worktree landing (`job` + `land: true`)
            if input.get("land").and_then(Value::as_bool).unwrap_or(false) {
                let Some(job_id) = non_empty_str(&input, "job") else {
                    return Ok(ToolOutput::err_text("`land: true` requires `job`"));
                };
                return self.handle_land(&fleet, job_id).await;
            }

            // Mode C: Multi-worker or fleet-wide wait (`jobs: [...]` OR `wait: true` without `job`)
            let explicit_jobs = non_empty_jobs(&input);
            let wait_flag = input.get("wait").and_then(Value::as_bool).unwrap_or(false);
            if !explicit_jobs.is_empty() || (wait_flag && non_empty_str(&input, "job").is_none()) {
                let wait_all = input
                    .get("wait_mode")
                    .and_then(Value::as_str)
                    .is_some_and(|m| m.eq_ignore_ascii_case("all"));
                let wait_secs = input
                    .get("timeout")
                    .and_then(Value::as_u64)
                    .filter(|&v| v > 0)
                    .unwrap_or(180)
                    .clamp(1, 1800);
                let _ = fleet
                    .wait_for(
                        &explicit_jobs,
                        wait_all,
                        Duration::from_secs(wait_secs),
                        &cancel,
                    )
                    .await;
                let (inbox, reports) = fleet.drain_coordinator_updates();
                let (rows, total_cost, running) = fleet.list_overview();
                let mut parts = Vec::new();
                for msg in &inbox {
                    parts.push(format!(
                        "[worker-update untrusted]\n[message from {}]: {}",
                        msg.from, msg.message
                    ));
                }
                for rep in &reports {
                    parts.push(rep.report.clone());
                }
                if !rows.is_empty() {
                    parts.push(format!(
                        "[fleet: {} workers ({} running) · total_cost=${:.4}]\n{}",
                        rows.len(),
                        running,
                        total_cost,
                        rows.join("\n")
                    ));
                } else if parts.is_empty() {
                    parts.push("No active workers.".into());
                }
                return Ok(ToolOutput::ok_text(parts.join("\n\n")).with_details(json!({
                    "reports_collected": reports.len(),
                    "messages_collected": inbox.len(),
                    "running": running,
                    "fleet_cost_usd": total_cost,
                })));
            }

            // Mode D: Poll / Wait / Kill a single worker (`job`)
            if let Some(job_id) = non_empty_str(&input, "job") {
                if input.get("kill").and_then(Value::as_bool).unwrap_or(false) {
                    let Some((worker, accrued_cost)) = fleet.close_worker(job_id) else {
                        return Ok(ToolOutput::err_text(format!(
                            "Unknown subagent job id: {job_id}"
                        )));
                    };
                    return Ok(ToolOutput::ok_text(format!(
                        "Closed worker `{job_id}` ({}) [cost_so_far=${accrued_cost:.4}]",
                        worker.task_preview
                    ))
                    .with_details(json!({
                        "job_id": job_id,
                        "killed": true,
                        "cost_usd": accrued_cost,
                    })));
                }

                let Some(worker) = fleet.get(job_id) else {
                    return Ok(ToolOutput::err_text(format!(
                        "Unknown subagent job id: {job_id}"
                    )));
                };

                if wait_flag && worker.state().is_running() {
                    let wait_secs = input
                        .get("timeout")
                        .and_then(Value::as_u64)
                        .filter(|&v| v > 0)
                        .unwrap_or(180)
                        .clamp(1, 1800);
                    let targets = vec![job_id.to_string()];
                    let _ = fleet
                        .wait_for(&targets, true, Duration::from_secs(wait_secs), &cancel)
                        .await;
                }

                if !worker.state().is_running() {
                    let rep_opt = worker.last_report.lock().ok().and_then(|g| g.clone());
                    worker.report_unread.store(false, Ordering::Release);
                    if !worker.keep_alive {
                        fleet.remove(job_id);
                    }
                    if let Some(rep) = rep_opt {
                        let details = report_to_details(&rep);
                        return Ok(if rep.is_error {
                            ToolOutput::err_text(rep.report).with_details(details)
                        } else {
                            ToolOutput::ok_text(rep.report).with_details(details)
                        });
                    }
                }

                let cost_so_far = worker.cost_usd();
                return Ok(ToolOutput::ok_text(format!(
                    "[{} — still running; call Subagent{{\"job\":\"{job_id}\",\"wait\":true}} to wait, SendMessage{{\"to\":\"{job_id}\",\"message\":\"...\"}} to steer, or Subagent{{\"job\":\"{job_id}\",\"kill\":true}} to stop]",
                    worker.progress_line()
                ))
                .with_details(json!({
                    "job_id": job_id,
                    "finished": false,
                    "turns": worker.turns_completed.load(Ordering::Relaxed),
                    "tools_called": worker.tools_called.load(Ordering::Relaxed),
                    "cost_so_far_usd": cost_so_far,
                })));
            }

            // Mode E: Spawn a new child subagent / worker (sync OR non-blocking background)
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
                    "Subagent requires `task` (or `job` / `jobs` / `list: true`)",
                ));
            };
            let background = input
                .get("background")
                .and_then(Value::as_bool)
                .unwrap_or(false);

            let mut plan = match self.prepare_child_plan(&fleet, &input, background).await {
                Ok(p) => p,
                Err(e) => {
                    return Ok(ToolOutput::err_text(format!(
                        "Failed to resolve subagent provider/model: {e}"
                    )));
                }
            };

            let mut fork_guard: Option<IsolatedForkWorktree> = None;
            if plan.isolated {
                let ws_root = self.parent_config.workspace_root.clone();
                let fork_id = uuid::Uuid::new_v4().simple().to_string();
                let fk = tokio::task::spawn_blocking(move || {
                    IsolatedForkWorktree::create(&ws_root, &fork_id)
                })
                .await
                .map_err(|e| CarnotError::Session(format!("worktree create task failed: {e}")))??;
                plan.config.workspace_root = fk.worker_workspace.clone();
                plan.config.extra_env.insert(
                    "CARGO_TARGET_DIR".into(),
                    fk.cargo_target_dir.to_string_lossy().to_string(),
                );
                fork_guard = Some(fk);
            }

            let child_agent = self.build_child_agent(&fleet, &plan).await?;
            let parent_ws = self.parent_config.workspace_root.clone();
            let parent_weak = self.parent_agent.get().cloned();

            if background || plan.keep_alive {
                let job_id = plan.name.clone();
                let task_preview: String = task.chars().take(80).collect();
                let (state_tx, state_rx) = watch::channel(WorkerState::Running);
                let worker = Arc::new(WorkerHandle {
                    name: job_id.clone(),
                    task_preview,
                    provider_id: plan.provider_id.clone(),
                    model: plan.config.model.clone(),
                    forked: plan.fork,
                    keep_alive: plan.keep_alive,
                    isolated: plan.isolated,
                    started: Instant::now(),
                    agent: child_agent,
                    parent_agent: parent_weak.clone(),
                    parent_workspace: parent_ws,
                    worktree: Mutex::new(fork_guard),
                    turns_completed: Arc::new(AtomicU64::new(0)),
                    tools_called: Arc::new(AtomicU64::new(0)),
                    cost_micros: Arc::new(AtomicU64::new(0)),
                    last_activity: Arc::new(Mutex::new("starting".to_string())),
                    cancel: CancellationToken::new(),
                    state_lock: Mutex::new(()),
                    state_tx,
                    state_rx,
                    last_report: Mutex::new(None),
                    report_unread: AtomicBool::new(false),
                    run_lock: tokio::sync::Mutex::new(()),
                });

                if let Err(e) = fleet.insert(Arc::clone(&worker)) {
                    worker.drop_worktree();
                    return Ok(ToolOutput::err_text(e.to_string()));
                }

                spawn_worker_turn(
                    Arc::downgrade(&fleet),
                    Arc::clone(&fleet.budget),
                    Arc::clone(&worker),
                    task,
                    parent_weak,
                );

                if background {
                    return Ok(ToolOutput::ok_text(format!(
                        "[subagent spawned -> {job_id} running in background (provider={}, model={}, fork={}, keep_alive={}, isolated_worktree={}) — continue working, call Subagent{{\"job\":\"{job_id}\",\"wait\":true}} to wait, SendMessage{{\"to\":\"{job_id}\",\"message\":\"...\"}} to message, or Subagent{{\"job\":\"{job_id}\",\"kill\":true}} to close]",
                        plan.provider_id,
                        plan.config.model,
                        plan.fork,
                        plan.keep_alive,
                        plan.isolated,
                    ))
                    .with_details(json!({
                        "job_id": job_id,
                        "provider": plan.provider_id,
                        "model": plan.config.model,
                        "fork": plan.fork,
                        "keep_alive": plan.keep_alive,
                        "background": true,
                        "isolated_worktree": plan.isolated,
                        "finished": false,
                    })));
                }

                // Synchronous + `keep_alive: true`: wait for first turn to complete, keeping worker alive!
                let targets = vec![job_id.clone()];
                let _ = fleet
                    .wait_for(&targets, true, Duration::from_secs(1800), &cancel)
                    .await;
                worker.report_unread.store(false, Ordering::Release);
                if let Some(rep) = worker.last_report.lock().ok().and_then(|g| g.clone()) {
                    let details = report_to_details(&rep);
                    return Ok(if rep.is_error {
                        ToolOutput::err_text(rep.report).with_details(details)
                    } else {
                        ToolOutput::ok_text(rep.report).with_details(details)
                    });
                }
                return Ok(ToolOutput::err_text(format!(
                    "Cancelled while waiting for worker `{job_id}`"
                )));
            }

            // Pure synchronous one-shot subagent with live cost accounting (even on error/cancel/halt!)
            let mut ev_stream = child_agent.subscribe_from(child_agent.latest_event_seq());
            let sync_cost_micros = Arc::new(AtomicU64::new(0));
            let sync_cost_obs = Arc::clone(&sync_cost_micros);
            let budget_obs = Arc::clone(&fleet.budget);
            let fleet_obs = Arc::downgrade(&fleet);
            let parent_obs = parent_weak.clone();
            let child_for_budget = Arc::clone(&child_agent);
            let obs_task = tokio::spawn(async move {
                loop {
                    let env = ev_stream.next().await;
                    match env.event {
                        AgentEvent::UsageRecorded { usage, .. } => {
                            let delta = (usage.cost_usd.max(0.0) * 1_000_000.0).round() as u64;
                            sync_cost_obs.fetch_add(delta, Ordering::Relaxed);
                            budget_obs
                                .record_worker_cost(usage.cost_usd, parent_obs.as_ref())
                                .await;
                            if budget_obs.exceeded().is_some() {
                                child_for_budget.abort().await;
                                if let Some(f) = fleet_obs.upgrade() {
                                    f.kill_all();
                                }
                            }
                        }
                        AgentEvent::RunSettled { .. } | AgentEvent::Error { .. } => {
                            break;
                        }
                        _ => {}
                    }
                }
            });

            let child_cancel = cancel.child_token();
            let cancel_for_watcher = child_cancel.clone();
            let ca_abort = Arc::clone(&child_agent);
            let cancel_watcher = tokio::spawn(async move {
                cancel_for_watcher.cancelled().await;
                ca_abort.abort().await;
            });

            let run_res = child_agent.prompt_with_cancel(task, child_cancel).await;
            cancel_watcher.abort();
            let _ = tokio::time::timeout(Duration::from_millis(150), obs_task).await;

            let ws_for_status = plan.config.workspace_root.clone();
            let (diff_handle, ws_status) = tokio::task::spawn_blocking(move || {
                let dh = fork_guard.as_ref().and_then(|fk| {
                    fk.export_worker_diff().ok().flatten().and_then(|bytes| {
                        CasStore::open(&parent_ws).ok().and_then(|cas| {
                            cas.put_text(&String::from_utf8_lossy(&bytes))
                                .ok()
                                .map(|h| h.rel_path)
                        })
                    })
                });
                let st = if let Some(ref fk) = fork_guard {
                    fk.changed_files_summary()
                } else {
                    collect_workspace_status(&ws_for_status)
                };
                if let Some(fk) = fork_guard {
                    fk.cleanup();
                }
                (dh, st)
            })
            .await
            .unwrap_or((None, None));

            let recorded_cost = sync_cost_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;
            match run_res {
                Ok(outcome) => {
                    let cost = outcome.metrics.total_usage.cost_usd.max(recorded_cost);
                    let mut summary = outcome.final_text;
                    if summary.chars().count() > MAX_SUBAGENT_SUMMARY_CHARS {
                        summary = summary.chars().take(MAX_SUBAGENT_SUMMARY_CHARS).collect();
                        summary.push_str("\n[... subagent summary truncated ...]");
                    }
                    let halted_tag = outcome
                        .halted_reason
                        .as_deref()
                        .map(|h| format!(" halted=\"{h}\""))
                        .unwrap_or_default();
                    let mut report = format!(
                        "[subagent-report worker={} provider={} model={} turns={} cost=${:.4}{}{}{}]\n{}",
                        plan.name,
                        plan.provider_id,
                        plan.config.model,
                        outcome.metrics.turns,
                        cost,
                        if plan.fork { " fork=true" } else { "" },
                        if plan.isolated {
                            " isolated_worktree=true"
                        } else {
                            ""
                        },
                        halted_tag,
                        summary
                    );
                    if let Some(ref hr) = outcome.halted_reason {
                        report.push_str(&format!("\n[subagent-halted] {hr}"));
                    }
                    if let Some(st) = ws_status {
                        report.push_str(&format!("\n[subagent-workspace-changes]\n{st}"));
                    }
                    if let Some(ref dh) = diff_handle {
                        report.push_str(&format!(
                            "\n[subagent-diff-patch] Saved isolated worktree diff to `{dh}` (apply via `git apply {dh}`)"
                        ));
                    }
                    let rep = WorkerReport {
                        worker_name: plan.name,
                        report: report.clone(),
                        is_error: false,
                        halted_reason: outcome.halted_reason,
                        provider: plan.provider_id,
                        model: plan.config.model,
                        forked: plan.fork,
                        keep_alive: false,
                        turns: outcome.metrics.turns,
                        cost_usd: cost,
                        isolated: plan.isolated,
                        diff_handle,
                    };
                    Ok(ToolOutput::ok_text(report).with_details(report_to_details(&rep)))
                }
                Err(e) => Ok(
                    ToolOutput::err_text(format!("Subagent failed: {e}")).with_details(json!({
                        "provider": plan.provider_id,
                        "model": plan.config.model,
                        "fork": plan.fork,
                        "isolated_worktree": plan.isolated,
                        "cost_usd": recorded_cost,
                        "finished": true,
                    })),
                ),
            }
        })
    }
}

/// Hub-and-spoke messaging tool registered identically on coordinator & workers (preserving
/// byte-identical static prefix for cache-warm `fork: true`).
///   - Coordinator -> worker `to`: wakes idle worker with `prompt()`, or steers/queues busy worker.
///   - Worker -> `"coordinator"`: pushes into `Fleet::coordinator_inbox` (single-channel delivery
///     as untrusted `Message::system_note`, never `[user-steer]`).
#[derive(Debug)]
struct SendMessageTool {
    /// `None` on the top-level coordinator; `Some(worker_name)` inside a child worker.
    sender_name: Option<String>,
    fleet: FleetRef,
    parent_agent: Arc<OnceLock<Weak<Agent>>>,
}

impl Tool for SendMessageTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "SendMessage".into(),
            description: "Send a message between the coordinator and a named worker. Coordinator targets a worker name (wakes idle `keep_alive` workers with warm cache, or queues/steers running ones); workers target `coordinator` to report progress, blockers, or questions.".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["to", "message"],
                "properties": {
                    "to": { "type": "string", "description": "Target worker name (from coordinator) or `coordinator` (from a worker)" },
                    "message": { "type": "string", "description": "Message content" },
                    "interrupt": { "type": "boolean", "description": "When targeting a currently running worker: inject mid-turn via `steer` instead of queuing post-turn `follow_up` (default false)" }
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

/// Unified orchestration & subagent module (`Subagent` + `SendMessage`, shared `Fleet`,
/// `FleetBudget`, role-gated `before_tool_call`, and reactive `on_turn_boundary` auto-wait).
#[derive(Debug)]
pub struct SubagentModule {
    depth: u8,
    max_depth: u8,
    worker_name: Option<String>,
    provider_name: String,
    provider_configs: Vec<PathBuf>,
    parent_agent: Arc<OnceLock<Weak<Agent>>>,
    fleet: FleetRef,
    budget: Arc<FleetBudget>,
    subagent_tool: Arc<SubagentTool>,
    send_msg_tool: Arc<SendMessageTool>,
}

impl SubagentModule {
    pub fn new(
        config: CarnotConfig,
        provider: Arc<dyn Provider>,
        extra_modules: Vec<Arc<dyn Module>>,
    ) -> Self {
        Self::with_configs(config, provider, Vec::new(), extra_modules)
    }

    pub fn with_provider_configs(self, provider_configs: Vec<PathBuf>) -> Self {
        Self::with_configs(
            self.subagent_tool.parent_config.clone(),
            Arc::clone(&self.subagent_tool.provider),
            provider_configs,
            self.subagent_tool.extra_modules.clone(),
        )
    }

    fn with_configs(
        config: CarnotConfig,
        provider: Arc<dyn Provider>,
        provider_configs: Vec<PathBuf>,
        extra_modules: Vec<Arc<dyn Module>>,
    ) -> Self {
        let depth = config.subagent_depth;
        let max_depth = config.subagent_max_depth;
        let provider_name = provider.name().to_string();
        let fleet = Arc::new(Fleet::new(config.max_cost_usd));
        let budget = Arc::clone(&fleet.budget);
        let fleet_ref = FleetRef::Strong(fleet);
        let parent_agent = Arc::new(OnceLock::new());
        let subagent_tool = Arc::new(SubagentTool {
            parent_config: config,
            provider,
            provider_configs: provider_configs.clone(),
            parent_agent: Arc::clone(&parent_agent),
            extra_modules,
            fleet: fleet_ref.clone(),
        });
        let send_msg_tool = Arc::new(SendMessageTool {
            sender_name: None,
            fleet: fleet_ref.clone(),
            parent_agent: Arc::clone(&parent_agent),
        });
        Self {
            depth,
            max_depth,
            worker_name: None,
            provider_name,
            provider_configs,
            parent_agent,
            fleet: fleet_ref,
            budget,
            subagent_tool,
            send_msg_tool,
        }
    }

    /// Creates a worker-role `SubagentModule` holding `Weak<Fleet>` (acyclic ownership!) while
    /// registering the exact same 2 tools (preserving byte-identical `prefix_hash`).
    #[allow(clippy::too_many_arguments)]
    fn for_worker(
        worker_name: String,
        config: CarnotConfig,
        provider: Arc<dyn Provider>,
        provider_configs: Vec<PathBuf>,
        extra_modules: Vec<Arc<dyn Module>>,
        fleet_weak: Weak<Fleet>,
        budget: Arc<FleetBudget>,
        parent_agent: Arc<OnceLock<Weak<Agent>>>,
    ) -> Self {
        let depth = config.subagent_depth;
        let max_depth = config.subagent_max_depth;
        let provider_name = provider.name().to_string();
        let fleet_ref = FleetRef::Weak(fleet_weak);
        let subagent_tool = Arc::new(SubagentTool {
            parent_config: config,
            provider,
            provider_configs: provider_configs.clone(),
            parent_agent: Arc::clone(&parent_agent),
            extra_modules,
            fleet: fleet_ref.clone(),
        });
        let send_msg_tool = Arc::new(SendMessageTool {
            sender_name: Some(worker_name.clone()),
            fleet: fleet_ref.clone(),
            parent_agent: Arc::clone(&parent_agent),
        });
        Self {
            depth,
            max_depth,
            worker_name: Some(worker_name),
            provider_name,
            provider_configs,
            parent_agent,
            fleet: fleet_ref,
            budget,
            subagent_tool,
            send_msg_tool,
        }
    }

    /// Attaches a `Weak<Agent>` handle to the parent coordinator `Agent` AND links `FleetBudget`
    /// to the parent's unified `total_cost_micros` counter so forked subagents, delegated
    /// permissions, combined budget guardrails, and reactive `on_turn_boundary` wakeups work seamlessly.
    pub fn attach_parent(&self, agent: &Arc<Agent>) {
        let _ = self.parent_agent.set(Arc::downgrade(agent));
        self.budget
            .attach_parent_counter(agent.total_cost_micros_handle());
    }

    /// Returns the shared `Fleet` handle when alive (useful for TUI/host status indicators or `/stop-all`).
    pub fn fleet(&self) -> Option<Arc<Fleet>> {
        self.fleet.upgrade()
    }

    /// Cancels all running background workers and cleans up their worktrees.
    pub fn kill_all_background(&self) {
        if let Some(f) = self.fleet.upgrade() {
            f.kill_all();
        }
    }
}

impl Module for SubagentModule {
    fn name(&self) -> &str {
        "subagent"
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![self.subagent_tool.clone(), self.send_msg_tool.clone()]
    }

    fn on_session_start(&self, cfg: &CarnotConfig) -> ExtFuture<'_, Result<Option<String>>> {
        let depth = self.depth;
        let is_faux = self.provider_name == "faux";
        let configs = self.provider_configs.clone();
        let ws_root = cfg.workspace_root.clone();
        Box::pin(async move {
            if depth == 0 && !is_faux {
                // Opportunistically prune any orphaned fork worktrees from a prior SIGKILL'd run
                // on a blocking thread (strictly PID-safe: never touches live Carnot instances).
                tokio::task::spawn_blocking(move || {
                    let _ = IsolatedForkWorktree::prune_stale_forks(&ws_root);
                });
            }
            if depth > 0 || is_faux {
                return Ok(None);
            }
            let Ok(reg) = ProviderRegistry::load_default(&configs) else {
                return Ok(None);
            };
            let models = reg.available_models(true);
            if models.is_empty() {
                return Ok(None);
            }
            let mut by_provider: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for m in models {
                let list = by_provider.entry(m.provider).or_default();
                if list.len() < 8 {
                    list.push(m.id);
                }
            }
            let mut lines = vec![
                "[available-subagent-models]".to_string(),
                "`Subagent` supports `name`, `keep_alive`, `fork`, `isolated_worktree`, `land`, `provider`, `model`, `effort`; use `SendMessage` to follow up with named workers. Configured providers & models:".to_string(),
            ];
            for (prov, ids) in by_provider {
                lines.push(format!("- {prov}: {}", ids.join(", ")));
            }
            Ok(Some(lines.join("\n")))
        })
    }

    fn before_tool_call(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> ExtFuture<'_, Result<ToolGateDecision>> {
        // 1. Shared fleet USD budget circuit-breaker (parent + all workers combined)
        if let Some((spent, limit)) = self.budget.exceeded() {
            return Box::pin(async move {
                Ok(ToolGateDecision::Deny {
                    reason: format!(
                        "fleet cost budget reached (${spent:.4} >= ${limit:.2}); stop and summarize."
                    ),
                })
            });
        }

        // 2. Role boundary: workers may NEVER call any `Subagent` action (spawn, list, wait, land, kill)
        if tool_name == "Subagent" && self.worker_name.is_some() {
            return Box::pin(async {
                Ok(ToolGateDecision::Deny {
                    reason:
                        "workers cannot call `Subagent` (use `SendMessage` to coordinate with `coordinator`)"
                            .into(),
                })
            });
        }

        // 3. Recursion depth guard for any non-worker depth-limited caller
        let is_spawn = non_empty_str(input, "job").is_none()
            && non_empty_jobs(input).is_empty()
            && !input.get("list").and_then(Value::as_bool).unwrap_or(false)
            && !input.get("wait").and_then(Value::as_bool).unwrap_or(false);
        if tool_name == "Subagent" && is_spawn && self.depth >= self.max_depth {
            let max_d = self.max_depth;
            return Box::pin(async move {
                Ok(ToolGateDecision::Deny {
                    reason: format!("subagent max recursion depth ({max_d}) reached"),
                })
            });
        }

        // 4. Hub-and-spoke enforcement on `SendMessage` (workers may only message `coordinator`)
        if tool_name == "SendMessage" && self.worker_name.is_some() {
            let to_ok = input
                .get("to")
                .and_then(Value::as_str)
                .is_some_and(|s| s.trim().eq_ignore_ascii_case("coordinator"));
            if !to_ok {
                return Box::pin(async move {
                    Ok(ToolGateDecision::Deny {
                        reason: "workers may only send messages to `coordinator` (hub-and-spoke)"
                            .into(),
                    })
                });
            }
        }

        Box::pin(async { Ok(ToolGateDecision::Allow) })
    }

    fn on_turn_boundary(
        &self,
        snapshot: &TurnBoundarySnapshot,
    ) -> ExtFuture<'_, Result<BoundaryAction>> {
        let is_worker = self.depth > 0 || self.worker_name.is_some();
        let had_tools = snapshot.had_tool_call_in_turn;
        let fleet_ref = self.fleet.clone();
        let budget = Arc::clone(&self.budget);
        let parent_ref = Arc::clone(&self.parent_agent);

        Box::pin(async move {
            // 1. Enforce unified parent + worker fleet budget at every turn boundary
            if let Some((spent, limit)) = budget.exceeded() {
                if !is_worker && let Some(fleet) = fleet_ref.upgrade() {
                    fleet.kill_all();
                }
                return Ok(BoundaryAction::Halt {
                    reason: format!("fleet cost budget reached (${spent:.4} >= ${limit:.2})"),
                });
            }

            if is_worker {
                return Ok(BoundaryAction::Continue);
            }

            let Some(fleet) = fleet_ref.upgrade() else {
                return Ok(BoundaryAction::Continue);
            };
            let Some(parent) = parent_ref.get().and_then(Weak::upgrade) else {
                return Ok(BoundaryAction::Continue);
            };

            // 2. On active tool turns, deliver ONLY incoming worker->coordinator inbox messages
            //    immediately as untrusted system notes (never evicting workers or stealing reports
            //    that the coordinator may explicitly collect with `Subagent{job, wait:true}`).
            if had_tools {
                let inbox = fleet.drain_coordinator_inbox();
                if !inbox.is_empty() {
                    let blocks: Vec<String> = inbox
                        .into_iter()
                        .map(|msg| {
                            format!(
                                "[worker-update untrusted]\n[message from worker `{}`]\n{}",
                                msg.from, msg.message
                            )
                        })
                        .collect();
                    parent.queue_system_note(blocks.join("\n\n")).await;
                }
                return Ok(BoundaryAction::Continue);
            }

            // 3. Coordinator stopped calling tools while workers may be live:
            //    If a human steer is already pending, yield immediately without waiting.
            if parent.has_pending_steers().await {
                return Ok(BoundaryAction::Continue);
            }

            if !fleet.has_unread_updates() {
                if !fleet.has_running_workers() {
                    return Ok(BoundaryAction::Continue);
                }
                // Wait reactively (0% CPU) until any worker finishes/fails/messages, user steers,
                // 240s cache-safe heartbeat fires, or user cancels!
                let cancel = parent.current_cancel_token().await;
                let steer_notify = parent.steer_notify();
                let fleet_notified = fleet.notify.notified();
                let steer_notified = steer_notify.notified();
                tokio::pin!(fleet_notified);
                tokio::pin!(steer_notified);
                // Register both waiters in Tokio's notify queues BEFORE checking conditions!
                fleet_notified.as_mut().enable();
                steer_notified.as_mut().enable();
                if !fleet.has_unread_updates()
                    && fleet.has_running_workers()
                    && !parent.has_pending_steers().await
                {
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            return Ok(BoundaryAction::Continue);
                        }
                        _ = &mut steer_notified => {
                            return Ok(BoundaryAction::Continue);
                        }
                        _ = &mut fleet_notified => {}
                        _ = tokio::time::sleep(Duration::from_secs(HEARTBEAT_SECS)) => {}
                    }
                    if cancel.is_cancelled() || parent.has_pending_steers().await {
                        return Ok(BoundaryAction::Continue);
                    }
                }
            }

            // Re-check budget right after reactive wait in case a background worker hit the cap
            if let Some((spent, limit)) = budget.exceeded() {
                fleet.kill_all();
                return Ok(BoundaryAction::Halt {
                    reason: format!("fleet cost budget reached (${spent:.4} >= ${limit:.2})"),
                });
            }

            let (inbox, reports) = fleet.drain_coordinator_updates();
            let mut update_blocks = Vec::new();
            for msg in inbox {
                update_blocks.push(format!(
                    "[worker-update untrusted]\n[message from worker `{}`]\n{}",
                    msg.from, msg.message
                ));
            }
            for rep in reports {
                update_blocks.push(format!("[worker-update untrusted]\n{}", rep.report));
            }
            if update_blocks.is_empty() && fleet.has_running_workers() {
                let (rows, total_cost, running) = fleet.list_overview();
                update_blocks.push(format!(
                    "[fleet-heartbeat: {running} worker(s) still running · fleet_cost=${total_cost:.4}]\n{}\nContinue coordinating or wait for completion.",
                    rows.join("\n")
                ));
            }
            if !update_blocks.is_empty() {
                parent.queue_system_note(update_blocks.join("\n\n")).await;
            }
            Ok(BoundaryAction::Continue)
        })
    }
}

/// Selects the optimal prompt-cache retention policy for a spawned subagent:
/// - Persistent workers (`keep_alive`) inherit the parent policy since they may idle between turns.
/// - Disposable fresh subagents (`!fork && !keep_alive`) use `CacheTtl::Short` (avoids paying
///   long-retention write surcharges on workers destroyed as soon as their autonomous task ends).
/// - Disposable forked subagents (`fork && !keep_alive`) downgrade `Long -> Hybrid` so they still
///   hit the parent's warm static/Turn-1 prefix while writing their own deltas at short retention.
fn subagent_cache_ttl(parent_ttl: CacheTtl, fork: bool, keep_alive: bool) -> CacheTtl {
    if keep_alive {
        parent_ttl
    } else if !fork {
        CacheTtl::Short
    } else {
        match parent_ttl {
            CacheTtl::Long => CacheTtl::Hybrid,
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subagent_cache_ttl_policy_matches_worker_lifecycle() {
        // Ephemeral fresh workers always use Short retention regardless of parent tier.
        assert_eq!(
            subagent_cache_ttl(CacheTtl::Long, false, false),
            CacheTtl::Short
        );
        assert_eq!(
            subagent_cache_ttl(CacheTtl::Hybrid, false, false),
            CacheTtl::Short
        );
        assert_eq!(
            subagent_cache_ttl(CacheTtl::Short, false, false),
            CacheTtl::Short
        );

        // Ephemeral forked workers downgrade Long -> Hybrid to share parent prefix cheaply.
        assert_eq!(
            subagent_cache_ttl(CacheTtl::Long, true, false),
            CacheTtl::Hybrid
        );
        assert_eq!(
            subagent_cache_ttl(CacheTtl::Hybrid, true, false),
            CacheTtl::Hybrid
        );
        assert_eq!(
            subagent_cache_ttl(CacheTtl::Short, true, false),
            CacheTtl::Short
        );

        // Persistent keep_alive workers preserve parent retention tier.
        assert_eq!(
            subagent_cache_ttl(CacheTtl::Long, false, true),
            CacheTtl::Long
        );
        assert_eq!(
            subagent_cache_ttl(CacheTtl::Hybrid, true, true),
            CacheTtl::Hybrid
        );
    }
}
