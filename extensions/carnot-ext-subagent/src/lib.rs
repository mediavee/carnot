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
mod send_message;
mod subagent_tool;

use self::budget::FleetBudget;
use self::fleet::Fleet;
use self::send_message::SendMessageTool;
use self::subagent_tool::{SubagentTool, non_empty_jobs, non_empty_str};
use carnot_agent::Agent;
use carnot_ai::{Provider, registry::ProviderRegistry};
use carnot_core::{CacheTtl, CarnotConfig, Result};
use carnot_ext::{
    BoundaryAction, BoxFuture as ExtFuture, Module, ToolGateDecision, TurnBoundarySnapshot,
    warm_fork::IsolatedForkWorktree,
};
use carnot_tools::Tool;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

/// 4-minute heartbeat stays strictly under typical 5-minute provider prompt-cache TTLs.
const HEARTBEAT_SECS: u64 = 240;

/// Acyclic fleet reference: coordinator owns `Strong(Arc<Fleet>)`, workers hold `Weak(Weak<Fleet>)`
/// so dropping the coordinator `Agent` deterministically drops `Fleet` and cleans up all worktrees.
#[derive(Debug, Clone)]
pub(crate) enum FleetRef {
    Strong(Arc<Fleet>),
    Weak(Weak<Fleet>),
}

impl FleetRef {
    pub(crate) fn upgrade(&self) -> Option<Arc<Fleet>> {
        match self {
            Self::Strong(f) => Some(Arc::clone(f)),
            Self::Weak(w) => w.upgrade(),
        }
    }
}

pub(crate) fn collect_workspace_status(ws: &std::path::Path) -> Option<String> {
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

    #[must_use]
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
    #[expect(
        clippy::too_many_arguments,
        reason = "internal worker constructor forwards acyclic Weak<Fleet> and shared budget handles"
    )]
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
    fn name(&self) -> &'static str {
        "subagent"
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![
            Arc::clone(&self.subagent_tool) as Arc<dyn Tool>,
            Arc::clone(&self.send_msg_tool) as Arc<dyn Tool>,
        ]
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
                        () = cancel.cancelled() => {
                            return Ok(BoundaryAction::Continue);
                        }
                        () = &mut steer_notified => {
                            return Ok(BoundaryAction::Continue);
                        }
                        () = &mut fleet_notified => {}
                        () = tokio::time::sleep(Duration::from_secs(HEARTBEAT_SECS)) => {}
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
pub(crate) fn subagent_cache_ttl(parent_ttl: CacheTtl, fork: bool, keep_alive: bool) -> CacheTtl {
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
