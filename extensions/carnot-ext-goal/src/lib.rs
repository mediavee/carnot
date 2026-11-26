//! Goal-driven autonomous verification loop for Carnot (`carnot-ext-goal`).
//!
//! Combines the best of deterministic verification and Worker ≠ Verifier separation with
//! zero bloat in `carnot-core` and zero bytes added to the static cached system prompt:
//!
//! 1. **Deterministic sandboxed checks** (`verify_commands`, e.g. `cargo test --workspace`) run
//!    inside Carnot's fail-closed OS sandbox whenever the worker believes it is done. On failure,
//!    shaped/CAS-spilled output is injected via `BoundaryAction::ContinueRun` with zero extra LLM
//!    verifier cost.
//! 2. **Independent virgin-context LLM verifier** (`criterion`) runs only after deterministic
//!    checks pass, in a clean single-turn context (never the worker's saturated history) using real
//!    `git status` / `git diff` evidence and strict JSON verdicts (`met` | `not_met` | `unreachable`),
//!    with optional cross-provider/cross-model routing (`verifier_model`).
//! 3. **Anti-stagnation & iteration caps** halt cleanly if the exact same diff + failure repeats
//!    or `max_iterations` is reached.

mod verifier;

pub use verifier::VerifierDecision;

use carnot_agent::Agent;
use carnot_ai::Provider;
use carnot_ai::registry::ProviderRegistry;
use carnot_cas::CasStore;
use carnot_core::{CarnotConfig, ReasoningEffort, Result, StopReason};
use carnot_ext::{BoundaryAction, BoxFuture, Module, PromptInjection, TurnBoundarySnapshot};
use carnot_sandbox::SandboxPolicy;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Configuration for a goal-driven verification loop.
#[derive(Debug, Clone)]
pub struct GoalConfig {
    /// Natural-language success condition evaluated by an independent virgin-context verifier.
    pub criterion: Option<String>,
    /// Deterministic shell checks executed inside Carnot's OS sandbox before LLM verification.
    pub verify_commands: Vec<String>,
    /// Maximum continuation iterations after the initial attempt (default: 8).
    pub max_iterations: u32,
    /// Halt as [`GoalStatus::Stagnated`] after this many consecutive identical diff+failure states (default: 3).
    pub stagnation_limit: u32,
    /// Optional model/provider override for the independent LLM verifier (e.g. `"gemini/gemini-2.5-flash"`
    /// or `"claude-haiku-4-5"`). Defaults to the session's active provider and model.
    pub verifier_model: Option<String>,
    /// Reasoning effort for the independent LLM verifier (default: `ReasoningEffort::Low`).
    pub verifier_effort: ReasoningEffort,
    /// Per-command timeout in milliseconds for deterministic checks (default: `120_000`).
    pub command_timeout_ms: u64,
    /// Maximum `git diff` bytes passed to the independent LLM verifier (default: 16 KiB).
    pub max_diff_bytes: usize,
}

impl Default for GoalConfig {
    fn default() -> Self {
        Self {
            criterion: None,
            verify_commands: Vec::new(),
            max_iterations: 8,
            stagnation_limit: 3,
            verifier_model: None,
            verifier_effort: ReasoningEffort::Low,
            command_timeout_ms: 120_000,
            max_diff_bytes: 16_384,
        }
    }
}

impl GoalConfig {
    /// Returns `true` when either a criterion or at least one deterministic check is configured.
    pub fn is_active(&self) -> bool {
        self.criterion
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty())
            || self.verify_commands.iter().any(|c| !c.trim().is_empty())
    }
}

/// Lifecycle state of an active or completed goal loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalStatus {
    Idle,
    Running,
    Met,
    Unreachable,
    Stagnated,
    MaxIterations,
    Cleared,
}

impl GoalStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Running => "running",
            Self::Met => "met",
            Self::Unreachable => "unreachable",
            Self::Stagnated => "stagnated",
            Self::MaxIterations => "max_iterations",
            Self::Cleared => "cleared",
        }
    }
}

/// Snapshot of the current goal loop state (for CLI/TUI `/goal` status displays).
#[derive(Debug, Clone, serde::Serialize)]
pub struct GoalState {
    pub status: GoalStatus,
    pub criterion: Option<String>,
    pub verify_commands: Vec<String>,
    pub iterations: u32,
    pub max_iterations: u32,
    pub last_feedback: Option<String>,
}

#[derive(Debug)]
struct InnerGoalState {
    config: GoalConfig,
    status: GoalStatus,
    revision: u64,
    injected_revision: u64,
    iterations: u32,
    last_failure_fingerprint: Option<[u8; 32]>,
    consecutive_same_failure: u32,
    last_feedback: Option<String>,
}

impl InnerGoalState {
    fn from_config(config: GoalConfig) -> Self {
        let active = config.is_active();
        Self {
            config,
            status: if active {
                GoalStatus::Running
            } else {
                GoalStatus::Idle
            },
            revision: u64::from(active),
            injected_revision: 0,
            iterations: 0,
            last_failure_fingerprint: None,
            consecutive_same_failure: 0,
            last_feedback: None,
        }
    }

    fn snapshot(&self) -> GoalState {
        GoalState {
            status: self.status,
            criterion: self.config.criterion.clone(),
            verify_commands: self.config.verify_commands.clone(),
            iterations: self.iterations,
            max_iterations: self.config.max_iterations,
            last_feedback: self.last_feedback.clone(),
        }
    }
}

/// Thread-safe runtime handle for inspecting, replacing, or clearing an active goal (e.g. TUI `/goal`).
#[derive(Debug, Clone)]
pub struct GoalHandle {
    state: Arc<Mutex<InnerGoalState>>,
}

impl GoalHandle {
    /// Creates a new shared goal handle initialized from `config` that can be passed to
    /// [`GoalModule::from_handle`] or `CarnotBuilder::goal_handle` for live `/goal` control.
    pub fn new(config: GoalConfig) -> Self {
        Self {
            state: Arc::new(Mutex::new(InnerGoalState::from_config(config))),
        }
    }

    /// Returns a snapshot of the current goal status and progress.
    pub fn status(&self) -> GoalState {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .snapshot()
    }

    /// Sets or replaces the active goal configuration and resets iteration/stagnation counters.
    pub fn set(&self, config: GoalConfig) {
        let mut g = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let active = config.is_active();
        g.config = config;
        g.status = if active {
            GoalStatus::Running
        } else {
            GoalStatus::Idle
        };
        g.revision = g.revision.saturating_add(1);
        g.iterations = 0;
        g.last_failure_fingerprint = None;
        g.consecutive_same_failure = 0;
        g.last_feedback = None;
    }

    /// Clears the active goal so the current session settles normally.
    pub fn clear(&self) {
        let mut g = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        g.config.criterion = None;
        g.config.verify_commands.clear();
        g.status = GoalStatus::Cleared;
        g.revision = g.revision.saturating_add(1);
        g.last_failure_fingerprint = None;
        g.consecutive_same_failure = 0;
    }
}

impl carnot_ext::GoalController for GoalHandle {
    fn status_json(&self) -> serde_json::Value {
        serde_json::to_value(self.status()).unwrap_or(serde_json::Value::Null)
    }

    fn set_goal(
        &self,
        criterion: Option<String>,
        verify_commands: Vec<String>,
        max_iterations: Option<u32>,
        verifier_model: Option<String>,
    ) -> serde_json::Value {
        let mut cfg = GoalConfig {
            criterion: criterion
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            verify_commands: verify_commands
                .into_iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            verifier_model: verifier_model
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            ..GoalConfig::default()
        };
        if let Some(m) = max_iterations {
            cfg.max_iterations = m.max(1);
        }
        self.set(cfg);
        self.status_json()
    }

    fn clear_goal(&self) -> serde_json::Value {
        self.clear();
        self.status_json()
    }
}

/// Official Carnot goal-driven verification module.
#[derive(Debug)]
pub struct GoalModule {
    workspace_root: PathBuf,
    sandbox: SandboxPolicy,
    extra_env: BTreeMap<String, String>,
    provider_config_paths: Vec<PathBuf>,
    cas: Arc<CasStore>,
    cas_handles: bool,
    cas_spill_threshold: usize,
    cas_preview_max: usize,
    redact_secrets: bool,
    state: Arc<Mutex<InnerGoalState>>,
    parent: RwLock<Weak<Agent>>,
    /// Optional explicit provider override (primarily for deterministic testing).
    verifier_provider_override: RwLock<Option<(Arc<dyn Provider>, String)>>,
}

impl GoalModule {
    pub fn new(carnot_config: &CarnotConfig, cas: Arc<CasStore>, goal_config: GoalConfig) -> Self {
        Self::from_handle(carnot_config, cas, GoalHandle::new(goal_config))
    }

    /// Creates a `GoalModule` backed by an externally-held [`GoalHandle`] so callers (CLI/TUI/SDK)
    /// can inspect status or dynamically set/clear goals at runtime.
    pub fn from_handle(
        carnot_config: &CarnotConfig,
        cas: Arc<CasStore>,
        handle: GoalHandle,
    ) -> Self {
        let sandbox = SandboxPolicy::for_workspace(
            carnot_config.workspace_root.clone(),
            carnot_config.modules.sandbox,
            carnot_config.modules.sandbox_network,
        )
        .with_extra_writable_roots(carnot_config.extra_writable_roots.clone())
        .with_scrubbed_env_vars(carnot_config.scrubbed_env_vars.clone());

        Self {
            workspace_root: carnot_config.workspace_root.clone(),
            sandbox,
            extra_env: carnot_config.extra_env.clone(),
            provider_config_paths: carnot_config.provider_config_paths.clone(),
            cas,
            cas_handles: carnot_config.modules.cas_handles,
            cas_spill_threshold: carnot_config.cas_spill_threshold_bytes,
            cas_preview_max: carnot_config.cas_preview_max_bytes,
            redact_secrets: carnot_config.redact_cas_secrets,
            state: handle.state,
            parent: RwLock::new(Weak::new()),
            verifier_provider_override: RwLock::new(None),
        }
    }

    /// Attaches the owning [`Agent`] via a weak reference (acyclic).
    pub fn attach_parent(&self, agent: &Arc<Agent>) {
        if let Ok(mut g) = self.parent.write() {
            *g = Arc::downgrade(agent);
        }
    }

    /// Overrides the verifier's `(Provider, model)` directly (useful for custom judges or unit tests).
    #[must_use]
    pub fn with_verifier_provider(
        self,
        provider: Arc<dyn Provider>,
        model: impl Into<String>,
    ) -> Self {
        if let Ok(mut g) = self.verifier_provider_override.write() {
            *g = Some((provider, model.into()));
        }
        self
    }

    /// Returns a cloneable [`GoalHandle`] for live status/set/clear control (e.g. TUI `/goal`).
    pub fn handle(&self) -> GoalHandle {
        GoalHandle {
            state: Arc::clone(&self.state),
        }
    }

    fn upgrade_parent(&self) -> Option<Arc<Agent>> {
        self.parent.read().ok().and_then(|g| g.upgrade())
    }

    async fn resolve_verifier(
        &self,
        model_override: Option<&str>,
    ) -> Result<(Arc<dyn Provider>, String)> {
        if let Ok(g) = self.verifier_provider_override.read()
            && let Some((ref p, ref m)) = *g
        {
            return Ok((Arc::clone(p), m.clone()));
        }
        let parent = self.upgrade_parent();
        let active_provider_id = match parent.as_ref() {
            Some(a) => a.current_provider_id().await,
            None => "anthropic".to_string(),
        };

        if let Some(raw_spec) = model_override.map(str::trim).filter(|s| !s.is_empty()) {
            let reg = ProviderRegistry::load_default(&self.provider_config_paths)?;
            let (prov_id, resolved_model) = reg
                .resolve_model_spec(raw_spec, None)
                .unwrap_or_else(|| (active_provider_id.clone(), raw_spec.to_string()));
            if let Some(ref a) = parent
                && prov_id == active_provider_id
            {
                return Ok((a.current_provider().await, resolved_model));
            }
            let prov = reg.build_provider(&prov_id, Some(&resolved_model))?;
            return Ok((prov, resolved_model));
        }

        if let Some(a) = parent {
            let p = a.current_provider().await;
            let m = a.current_model().await;
            return Ok((p, m));
        }

        Err(carnot_core::CarnotError::Extension {
            extension: "goal".to_string(),
            message: "GoalModule requires an attached Agent or explicit verifier provider"
                .to_string(),
        })
    }

    fn record_failure_and_decide(
        &self,
        expected_revision: u64,
        fingerprint: [u8; 32],
        signal: &str,
        feedback: &str,
    ) -> BoundaryAction {
        let mut g = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if g.revision != expected_revision || g.status != GoalStatus::Running {
            return BoundaryAction::Continue;
        }
        g.iterations = g.iterations.saturating_add(1);
        if g.last_failure_fingerprint == Some(fingerprint) {
            g.consecutive_same_failure = g.consecutive_same_failure.saturating_add(1);
        } else {
            g.last_failure_fingerprint = Some(fingerprint);
            g.consecutive_same_failure = 1;
        }
        g.last_feedback = Some(feedback.to_string());

        if g.config.stagnation_limit > 0 && g.consecutive_same_failure >= g.config.stagnation_limit
        {
            g.status = GoalStatus::Stagnated;
            return BoundaryAction::Halt {
                reason: format!(
                    "goal stagnated ({} consecutive identical diff/failure checks): {feedback}",
                    g.consecutive_same_failure
                ),
            };
        }
        if g.iterations > g.config.max_iterations {
            g.status = GoalStatus::MaxIterations;
            return BoundaryAction::Halt {
                reason: format!(
                    "goal reached max_iterations ({}) without meeting criterion: {feedback}",
                    g.config.max_iterations
                ),
            };
        }

        let iter_num = g.iterations;
        let max_iter = g.config.max_iterations;
        BoundaryAction::ContinueRun {
            signal: signal.to_string(),
            message: format!(
                "[carnot:goal iteration {iter_num}/{max_iter}]\n{feedback}\nFix the remaining gap(s) and verify before finishing."
            ),
        }
    }
}

impl Module for GoalModule {
    fn name(&self) -> &'static str {
        "goal"
    }

    fn before_prompt(&self, _user_prompt: &str) -> BoxFuture<'_, Result<Vec<PromptInjection>>> {
        let injection = {
            let mut g = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if g.status == GoalStatus::Running
                && g.config.is_active()
                && g.injected_revision != g.revision
            {
                g.injected_revision = g.revision;
                let mut note = String::from("[carnot:goal]\n");
                if let Some(ref crit) = g.config.criterion
                    && !crit.trim().is_empty()
                {
                    let _ = writeln!(note, "Success condition: {}", crit.trim());
                }
                if !g.config.verify_commands.is_empty() {
                    note.push_str("Required verification command(s) checked at turn end:\n");
                    for cmd in &g.config.verify_commands {
                        let _ = writeln!(note, "- `{cmd}`");
                    }
                }
                note.push_str(
                    "Do not stop until the success condition and all verification commands pass.",
                );
                Some(PromptInjection {
                    text: note,
                    ephemeral: false,
                })
            } else {
                None
            }
        };
        Box::pin(async move { Ok(injection.into_iter().collect()) })
    }

    fn on_turn_boundary<'a>(
        &'a self,
        snap: &'a TurnBoundarySnapshot,
    ) -> BoxFuture<'a, Result<BoundaryAction>> {
        Box::pin(async move {
            // Only evaluate when the worker actually stopped cleanly with no pending tool calls.
            if snap.stop_reason != StopReason::EndTurn || snap.had_tool_call_in_turn {
                return Ok(BoundaryAction::Continue);
            }

            let (cfg, start_revision) = {
                let g = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if g.status != GoalStatus::Running || !g.config.is_active() {
                    return Ok(BoundaryAction::Continue);
                }
                (g.config.clone(), g.revision)
            };

            let cancel = match self.upgrade_parent() {
                Some(a) => a.current_cancel_token().await,
                None => CancellationToken::new(),
            };
            if cancel.is_cancelled() {
                return Ok(BoundaryAction::Continue);
            }

            let evidence =
                verifier::collect_workspace_evidence(&self.workspace_root, cfg.max_diff_bytes)
                    .await;

            // Step 1: Deterministic sandboxed checks (zero LLM verifier cost, zero hallucination).
            let mut check_results = Vec::with_capacity(cfg.verify_commands.len());
            let cmd_timeout = Duration::from_millis(cfg.command_timeout_ms.max(1_000));
            for cmd in &cfg.verify_commands {
                let trimmed = cmd.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let res = verifier::run_verify_command(
                    trimmed,
                    &self.sandbox,
                    &self.extra_env,
                    &self.cas,
                    self.cas_handles,
                    self.cas_spill_threshold,
                    self.cas_preview_max,
                    self.redact_secrets,
                    cmd_timeout,
                    cancel.clone(),
                )
                .await?;
                let failed = !res.passed;
                check_results.push(res);
                if failed {
                    let Some(last) = check_results.last() else {
                        break;
                    };
                    let mut fp = blake3::Hasher::new();
                    fp.update(&evidence.state_hash);
                    fp.update(last.command.as_bytes());
                    fp.update(&last.exit_code.to_le_bytes());
                    fp.update(&last.output_hash);
                    let detail = if last.timed_out {
                        format!(
                            "Deterministic verification command timed out (`{}`):\n{}",
                            last.command, last.shaped_output
                        )
                    } else {
                        format!(
                            "Deterministic verification command failed (`{}`, exit {}):\n{}",
                            last.command, last.exit_code, last.shaped_output
                        )
                    };
                    return Ok(self.record_failure_and_decide(
                        start_revision,
                        *fp.finalize().as_bytes(),
                        "goal_check_failed",
                        &detail,
                    ));
                }
            }

            // Pure deterministic mode (no natural-language criterion): all checks passed -> MET!
            let Some(criterion) = cfg
                .criterion
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            else {
                let mut g = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if g.revision == start_revision && g.status == GoalStatus::Running {
                    g.status = GoalStatus::Met;
                    g.last_feedback =
                        Some("All deterministic verification commands passed.".into());
                }
                return Ok(BoundaryAction::Continue);
            };

            // Abort before launching an LLM verifier if the goal was cleared/replaced during checks.
            {
                let g = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if g.revision != start_revision || g.status != GoalStatus::Running {
                    return Ok(BoundaryAction::Continue);
                }
            }

            // Step 2: Independent virgin-context LLM verifier (Worker != Verifier).
            let (verifier_provider, verifier_model) =
                self.resolve_verifier(cfg.verifier_model.as_deref()).await?;
            let (decision, cost_usd) = verifier::run_llm_verifier(
                &verifier_provider,
                &verifier_model,
                cfg.verifier_effort,
                criterion,
                &check_results,
                &evidence,
                &snap.files_edited_in_segment,
                &snap.last_assistant_text,
                cancel,
            )
            .await?;

            if cost_usd > 0.0
                && let Some(parent) = self.upgrade_parent()
            {
                parent.record_external_cost(cost_usd).await;
            }

            match decision {
                VerifierDecision::Met { feedback } => {
                    let mut g = self
                        .state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if g.revision == start_revision && g.status == GoalStatus::Running {
                        g.status = GoalStatus::Met;
                        g.last_feedback = Some(feedback);
                    }
                    Ok(BoundaryAction::Continue)
                }
                VerifierDecision::Unreachable { feedback } => {
                    let mut g = self
                        .state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if g.revision != start_revision || g.status != GoalStatus::Running {
                        return Ok(BoundaryAction::Continue);
                    }
                    g.status = GoalStatus::Unreachable;
                    g.last_feedback = Some(feedback.clone());
                    Ok(BoundaryAction::Halt {
                        reason: format!("goal unreachable: {feedback}"),
                    })
                }
                VerifierDecision::NotMet { feedback } => {
                    let mut fp = blake3::Hasher::new();
                    fp.update(&evidence.state_hash);
                    fp.update(feedback.trim().as_bytes());
                    let msg = format!("Independent verifier judged goal NOT MET:\n{feedback}");
                    Ok(self.record_failure_and_decide(
                        start_revision,
                        *fp.finalize().as_bytes(),
                        "goal_not_met",
                        &msg,
                    ))
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use carnot_ai::faux::{FauxProvider, ScriptedTurn};

    fn make_snap(text: &str) -> TurnBoundarySnapshot {
        TurnBoundarySnapshot {
            turn_index: 1,
            stop_reason: StopReason::EndTurn,
            total_context_tokens: 1_000,
            had_edit_in_segment: true,
            had_tool_call_in_turn: false,
            files_read_in_segment: Vec::new(),
            files_edited_in_segment: vec!["app.txt".into()],
            handles_created_in_segment: Vec::new(),
            last_assistant_text: text.into(),
            failed_attempts_summary: Vec::new(),
        }
    }

    #[test]
    fn verifier_json_parser_handles_clean_fenced_and_fails_closed_on_garbage() {
        assert_eq!(
            verifier::parse_verifier_json(r#"{"status":"met","feedback":"all good"}"#),
            VerifierDecision::Met {
                feedback: "all good".into()
            }
        );
        assert_eq!(
            verifier::parse_verifier_json(
                "```json\n{\"status\": \"unreachable\", \"feedback\": \"missing key\"}\n```"
            ),
            VerifierDecision::Unreachable {
                feedback: "missing key".into()
            }
        );
        assert!(matches!(
            verifier::parse_verifier_json("not json at all"),
            VerifierDecision::NotMet { .. }
        ));
    }

    #[tokio::test]
    async fn deterministic_check_continues_until_fixed_and_detects_stagnation() {
        let dir = std::env::temp_dir().join(format!("carnot-goal-det-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let marker = dir.join("ready.txt");
        let mut cfg = CarnotConfig::new(dir.clone());
        cfg.modules.sandbox = false;
        let cas = Arc::new(CasStore::open(&dir).expect("cas"));

        let gcfg = GoalConfig {
            criterion: None,
            verify_commands: vec!["test -f ready.txt".into()],
            max_iterations: 5,
            stagnation_limit: 2,
            ..Default::default()
        };
        let gm = GoalModule::new(&cfg, cas, gcfg);
        let handle = gm.handle();

        // 1. Missing marker -> ContinueRun
        let a1 = gm.on_turn_boundary(&make_snap("done")).await.expect("b1");
        assert!(
            matches!(a1, BoundaryAction::ContinueRun { ref signal, .. } if signal == "goal_check_failed"),
            "expected ContinueRun, got {a1:?}"
        );
        assert_eq!(handle.status().iterations, 1);

        // 2. Same unchanged failure again -> hits stagnation_limit=2 -> Halt!
        let a2 = gm
            .on_turn_boundary(&make_snap("still done"))
            .await
            .expect("b2");
        assert!(
            matches!(a2, BoundaryAction::Halt { ref reason } if reason.contains("stagnated")),
            "expected stagnation halt, got {a2:?}"
        );
        assert_eq!(handle.status().status, GoalStatus::Stagnated);

        // 3. Reset goal via handle, create marker -> passes immediately -> Met!
        handle.set(GoalConfig {
            criterion: None,
            verify_commands: vec!["test -f ready.txt".into()],
            max_iterations: 5,
            stagnation_limit: 2,
            ..Default::default()
        });
        std::fs::write(&marker, "ok\n").expect("create marker");
        let a3 = gm.on_turn_boundary(&make_snap("fixed")).await.expect("b3");
        assert!(matches!(a3, BoundaryAction::Continue), "got {a3:?}");
        assert_eq!(handle.status().status, GoalStatus::Met);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn independent_llm_verifier_drives_not_met_then_met_and_unreachable() {
        let dir = std::env::temp_dir().join(format!("carnot-goal-llm-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let mut cfg = CarnotConfig::new(dir.clone());
        cfg.modules.sandbox = false;
        let cas = Arc::new(CasStore::open(&dir).expect("cas"));

        let faux = Arc::new(FauxProvider::new(vec![
            ScriptedTurn::text(r#"{"status":"not_met","feedback":"missing edge-case test"}"#),
            ScriptedTurn::text(r#"{"status":"met","feedback":"all criteria verified"}"#),
            ScriptedTurn::text(r#"{"status":"unreachable","feedback":"external API offline"}"#),
        ]));

        let gcfg = GoalConfig {
            criterion: Some("Feature implemented with edge-case test".into()),
            verify_commands: vec!["true".into()],
            max_iterations: 4,
            stagnation_limit: 3,
            ..Default::default()
        };
        let gm = GoalModule::new(&cfg, cas, gcfg)
            .with_verifier_provider(Arc::clone(&faux) as Arc<dyn Provider>, "verifier-v1");
        let handle = gm.handle();

        // Turn 1: verifier says not_met -> ContinueRun carrying verifier feedback
        let a1 = gm
            .on_turn_boundary(&make_snap("implemented"))
            .await
            .expect("v1");
        assert!(
            matches!(a1, BoundaryAction::ContinueRun { ref message, .. } if message.contains("missing edge-case test")),
            "expected not_met ContinueRun, got {a1:?}"
        );

        // Turn 2: verifier says met -> Continue + GoalStatus::Met
        let a2 = gm
            .on_turn_boundary(&make_snap("added test"))
            .await
            .expect("v2");
        assert!(matches!(a2, BoundaryAction::Continue), "got {a2:?}");
        assert_eq!(handle.status().status, GoalStatus::Met);

        // New goal -> verifier says unreachable -> Halt
        handle.set(GoalConfig {
            criterion: Some("Deploy to prod".into()),
            verify_commands: Vec::new(),
            max_iterations: 4,
            stagnation_limit: 3,
            ..Default::default()
        });
        let a3 = gm
            .on_turn_boundary(&make_snap("cannot deploy"))
            .await
            .expect("v3");
        assert!(
            matches!(a3, BoundaryAction::Halt { ref reason } if reason.contains("unreachable")),
            "expected unreachable halt, got {a3:?}"
        );
        assert_eq!(handle.status().status, GoalStatus::Unreachable);

        // Verify virgin context: every verifier request had exactly 1 user message and 0 tools!
        let reqs = faux.recorded_requests();
        assert_eq!(reqs.len(), 3);
        for req in &reqs {
            assert_eq!(
                req.messages.len(),
                1,
                "verifier must run in virgin 1-message context"
            );
            assert!(req.tools.is_empty(), "verifier must have zero tools");
            assert!(req.skip_tail_cache_write, "verifier must skip cache writes");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn clear_goal_during_inflight_verification_never_reverts_to_met() {
        let dir = std::env::temp_dir().join(format!("carnot-goal-race-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let mut cfg = CarnotConfig::new(dir.clone());
        cfg.modules.sandbox = false;
        let cas = Arc::new(CasStore::open(&dir).expect("cas"));

        let gcfg = GoalConfig {
            criterion: None,
            verify_commands: vec!["sleep 0.25".into()],
            max_iterations: 4,
            stagnation_limit: 3,
            ..Default::default()
        };
        let handle = GoalHandle::new(gcfg);
        let gm = Arc::new(GoalModule::from_handle(&cfg, cas, handle.clone()));

        let gm_task = Arc::clone(&gm);
        let verify_jh = tokio::spawn(async move {
            gm_task
                .on_turn_boundary(&make_snap("done"))
                .await
                .expect("boundary")
        });

        // Clear the goal while `sleep 0.25` is still running!
        tokio::time::sleep(Duration::from_millis(60)).await;
        handle.clear();
        assert_eq!(handle.status().status, GoalStatus::Cleared, "cleared");

        let act = verify_jh.await.expect("join");
        assert!(matches!(act, BoundaryAction::Continue), "got {act:?}");
        assert_eq!(
            handle.status().status,
            GoalStatus::Cleared,
            "stale verifier completion must NEVER overwrite Cleared with Met"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
