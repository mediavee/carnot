//! `Subagent` tool implementation: spawning, forking, waiting, worktree landing, and mirrored
//! coordinator tool schemas for cache-warm workers.

use crate::fleet::{
    DelegatingPermissionBroker, Fleet, MAX_SUBAGENT_SUMMARY_CHARS, WorkerHandle, WorkerReport,
    WorkerState, report_to_details, spawn_worker_turn,
};
use crate::{FleetRef, SubagentModule, collect_workspace_status, subagent_cache_ttl};
use carnot_agent::Agent;
use carnot_ai::{Provider, registry::ProviderRegistry};
use carnot_cas::CasStore;
use carnot_core::{
    AgentEvent, CarnotConfig, CarnotError, ReasoningEffort, Result, ToolEffects, ToolOutput,
    ToolSpec,
};
use carnot_ext::{
    Module,
    warm_fork::{IsolatedForkWorktree, LandOutcome},
};
use carnot_tools::{BoxFuture, Tool, ToolRegistry};
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

pub(crate) fn non_empty_str<'a>(input: &'a Value, key: &str) -> Option<&'a str> {
    input
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

pub(crate) fn non_empty_jobs(input: &Value) -> Vec<String> {
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

pub(crate) fn sanitize_worker_name(raw: &str) -> Option<String> {
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
pub(crate) struct SubagentTool {
    pub(crate) parent_config: CarnotConfig,
    pub(crate) provider: Arc<dyn Provider>,
    pub(crate) provider_configs: Vec<PathBuf>,
    pub(crate) parent_agent: Arc<OnceLock<Weak<Agent>>>,
    pub(crate) extra_modules: Vec<Arc<dyn Module>>,
    pub(crate) fleet: FleetRef,
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
                    .map_or_else(|| p.to_string(), |(canon, _)| canon.to_string())
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
                .is_some_and(|(_, e)| !e.model_routes.is_empty());
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
                    ((child_cfg.context_window / 2).max(4096)).min(u64::from(u32::MAX)) as u32;
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
                    ((child_cfg.context_window / 2).max(4096)).min(u64::from(u32::MAX)) as u32;
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
                    let _ = write!(
                        msg,
                        "\nPatch saved to `{cp}` (you can resolve via `git apply --3way {cp}` or steer `{job_id}` via `SendMessage`)."
                    );
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
                plan.config.workspace_root.clone_from(&fk.worker_workspace);
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
                        let _ = write!(report, "\n[subagent-halted] {hr}");
                    }
                    if let Some(st) = ws_status {
                        let _ = write!(report, "\n[subagent-workspace-changes]\n{st}");
                    }
                    if let Some(ref dh) = diff_handle {
                        let _ = write!(
                            report,
                            "\n[subagent-diff-patch] Saved isolated worktree diff to `{dh}` (apply via `git apply {dh}`)"
                        );
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
