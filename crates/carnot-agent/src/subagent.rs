//! Disposable-Context `SubagentModule` (lives in `carnot-agent` to keep DAG 100% acyclic!).
//!
//! Massive AUC & cost multiplier: offloads noisy multi-file research or isolated subtasks to a
//! fresh child `Agent` whose 20k-80k token exploration trace is discarded on exit, returning ONLY
//! a concise bounded summary + git status/diff stat + exact cost to the parent!
//!
//! Incorporates EVERY Opus architectural safeguard:
//!   - Byte-identical static prefix with parent: child keeps the EXACT same tool schemas so it
//!     hits the parent's warm prompt cache on Turn 1; recursion is blocked in `before_tool_call`
//!     when `depth >= max_depth`!
//!   - Without `isolated_worktree: true`, declares `ToolEffects::serial()` (cannot race parent edits).
//!   - With `isolated_worktree: true`, creates a safe `IsolatedForkWorktree` (`git worktree add --detach`
//!     + dedicated `CARGO_TARGET_DIR`) and ALWAYS cleans it up even on cancellation!
//!   - Strict cost budget (`subagent_max_cost_usd`, including cache writes) & linked `child_token()`.

use crate::Agent;
use carnot_ai::Provider;
use carnot_cas::CasStore;
use carnot_core::{CarnotConfig, Result, ToolEffects, ToolOutput, ToolSpec};
use carnot_ext::{
    BoxFuture as ExtFuture, Module, ToolGateDecision, warm_fork::IsolatedForkWorktree,
};
use carnot_tools::{BoxFuture, Tool, ToolRegistry};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

const MAX_SUBAGENT_SUMMARY_CHARS: usize = 6_000;

#[derive(Debug)]
struct SubagentTool {
    parent_config: CarnotConfig,
    provider: Arc<dyn Provider>,
    extra_modules: Vec<Arc<dyn Module>>,
}

impl Tool for SubagentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "Subagent".into(),
            description: "Spawn an isolated child Carnot agent with a disposable context window for focused research or parallelizable subtasks. Returns only a concise report + diff summary.".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["task"],
                "properties": {
                    "task": { "type": "string", "description": "Clear, self-contained task description and expected output" },
                    "model": { "type": "string", "description": "Optional cheaper/faster model override (e.g. claude-haiku-5-5)" },
                    "max_turns": { "type": "integer", "description": "Max turns for child agent (default 25, max 80)" },
                    "isolated_worktree": { "type": "boolean", "description": "Run inside an isolated detached git worktree + dedicated CARGO_TARGET_DIR" }
                }
            }),
            defer_loading: false,
        }
    }

    fn effects(&self, input: &Value) -> ToolEffects {
        let isolated = input
            .get("isolated_worktree")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if isolated {
            // Isolated worktree cannot touch parent workspace files, and MUST NOT launch
            // speculatively mid-stream (paid LLM side effects — Opus #8!).
            ToolEffects::global_read_no_eager()
        } else {
            // Shared workspace MUST serialize with parent edits (Opus point #3!)
            ToolEffects::serial()
        }
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            if self.parent_config.subagent_depth >= self.parent_config.subagent_max_depth {
                return Ok(ToolOutput::err_text(format!(
                    "Subagent recursion depth limit ({}) reached; complete the task directly.",
                    self.parent_config.subagent_max_depth
                )));
            }
            let Some(task) = input.get("task").and_then(Value::as_str) else {
                return Ok(ToolOutput::err_text("Subagent requires `task`"));
            };
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
                child_cfg.model = m.trim().to_string();
            }

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

            let cas = Arc::new(CasStore::open(&child_cfg.workspace_root)?);
            let tools = ToolRegistry::with_canonical_lid_tools(&child_cfg, cas);

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

            let child_agent = Arc::new(Agent::new(
                child_cfg.clone(),
                self.provider.clone(),
                tools,
                child_modules,
                false,
            )?);

            // Propagate parent cancellation into child agent via child_token!
            let child_cancel = cancel.child_token();
            let ca_abort = child_agent.clone();
            let cancel_watcher = tokio::spawn(async move {
                child_cancel.cancelled().await;
                ca_abort.abort().await;
            });

            let run_res = child_agent.prompt(task).await;
            cancel_watcher.abort();

            // Collect git status AND export binary patch to parent CAS before cleaning up isolated worktree!
            let ws_status = collect_workspace_status(&child_cfg.workspace_root);
            let diff_handle = if isolated {
                export_isolated_diff_to_parent_cas(
                    &child_cfg.workspace_root,
                    &self.parent_config.workspace_root,
                )
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
                    Ok(ToolOutput::ok_text(report).with_details(json!({
                        "model": child_cfg.model,
                        "turns": outcome.metrics.turns,
                        "cost_usd": outcome.metrics.total_usage.cost_usd,
                        "isolated_worktree": isolated,
                        "diff_handle": diff_handle,
                    })))
                }
                Err(e) => Ok(ToolOutput::err_text(format!("Subagent failed: {e}"))),
            }
        })
    }
}

fn export_isolated_diff_to_parent_cas(
    child_ws: &std::path::Path,
    parent_ws: &std::path::Path,
) -> Option<String> {
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

/// First-class module registering the `Subagent` tool and enforcing recursion-depth gating.
#[derive(Debug)]
pub struct SubagentModule {
    depth: u8,
    max_depth: u8,
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
        let tool = Arc::new(SubagentTool {
            parent_config: config,
            provider,
            extra_modules,
        });
        Self {
            depth,
            max_depth,
            tool,
        }
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
        _input: &Value,
    ) -> ExtFuture<'_, Result<ToolGateDecision>> {
        let blocked = tool_name == "Subagent" && self.depth >= self.max_depth;
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
