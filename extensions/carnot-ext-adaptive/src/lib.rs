//! Cache-safe adaptive reasoning-effort router + ephemeral thrash & edit-oscillation guard.
//!
//! Zero static system-prompt bytes, zero added tools, zero cache invalidation:
//! 1. **Adaptive Effort Routing (`Module::select_effort`)**: Dynamically scales reasoning effort
//!    within `[min_effort, max_effort]` based on live turn signals (read-only exploration streak ->
//!    `min_effort`, clean edits -> `Medium`, build/test failure or thrashing -> `max_effort`).
//!    `carnot-agent` ONLY applies `select_effort()` when `provider.effort_preserves_cache(model)`
//!    is true, guaranteeing zero prompt-cache churn on fixed-budget models.
//! 2. **Thrash & Edit-Oscillation Guard (`after_tool_result` + `on_turn_boundary`)**:
//!    - Detects repeated identical failing command/tool fingerprints (default 3x) or consecutive
//!      tool error streaks (default 4x).
//!    - Detects per-file `A -> B -> A` edit-revert oscillation by hashing post-edit workspace
//!      file bytes with BLAKE3.
//!    - Emits a concise one-turn **ephemeral** advisory (`BoundaryAction::AppendEphemeralAdvisory`)
//!      delivered after all provider cache breakpoints (`ephemeral_tail_notes`) and never persisted
//!      into session history — breaking stuck loops with zero permanent context bloat.

#![forbid(unsafe_code)]

use carnot_core::{ReasoningEffort, Result, ToolOutput};
use carnot_ext::{BoundaryAction, BoxFuture, Module, TurnBoundarySnapshot};
use carnot_tools::files::resolve_confined_path;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Mutex;

const MAX_TRACKED_FILES: usize = 64;
const MAX_HASH_HISTORY_PER_FILE: usize = 6;
const READONLY_STEPDOWN_STREAK: u32 = 2;

/// Configuration for [`AdaptiveModule`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Enable cache-safe per-turn `select_effort()` routing (default: `true`).
    #[serde(default = "default_true")]
    pub dynamic_effort: bool,
    /// Lower bound for adaptive effort de-escalation on clean read-only exploration turns (default: `Low`).
    #[serde(default = "default_min_effort")]
    pub min_effort: ReasoningEffort,
    /// Upper bound for adaptive effort escalation on failures / thrash / oscillation (default: `High`).
    #[serde(default = "default_max_effort")]
    pub max_effort: ReasoningEffort,
    /// Enable repeated-failure / error-streak / `A -> B -> A` edit-oscillation detection (default: `true`).
    #[serde(default = "default_true")]
    pub thrash_guard: bool,
    /// Number of consecutive identical failing tool/command fingerprints before emitting an advisory (default: `3`).
    #[serde(default = "default_repeat_threshold")]
    pub repeat_failure_threshold: u32,
    /// Number of consecutive failing tool calls (even distinct commands) before emitting an advisory (default: `4`).
    #[serde(default = "default_streak_threshold")]
    pub error_streak_threshold: u32,
    /// Detect `A -> B -> A` per-file content hash oscillation after `Edit`/`Write` (default: `true`).
    #[serde(default = "default_true")]
    pub detect_edit_oscillation: bool,
}

const fn default_true() -> bool {
    true
}

const fn default_min_effort() -> ReasoningEffort {
    ReasoningEffort::Low
}

const fn default_max_effort() -> ReasoningEffort {
    ReasoningEffort::High
}

const fn default_repeat_threshold() -> u32 {
    3
}

const fn default_streak_threshold() -> u32 {
    4
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            dynamic_effort: true,
            min_effort: ReasoningEffort::Low,
            max_effort: ReasoningEffort::High,
            thrash_guard: true,
            repeat_failure_threshold: 3,
            error_streak_threshold: 4,
            detect_edit_oscillation: true,
        }
    }
}

/// Runtime telemetry snapshot exported by [`AdaptiveModule::telemetry`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptiveTelemetry {
    pub current_effort: Option<ReasoningEffort>,
    pub effort_transitions: u64,
    pub repeat_failure_alerts: u64,
    pub error_streak_alerts: u64,
    pub edit_oscillation_alerts: u64,
}

#[derive(Debug, Default)]
struct TurnActivity {
    tool_calls: u32,
    errors: u32,
    mutations: u32,
    readonly_only: bool,
    oscillated_file: Option<String>,
    last_failed_summary: Option<String>,
}

impl TurnActivity {
    fn reset(&mut self) {
        self.tool_calls = 0;
        self.errors = 0;
        self.mutations = 0;
        self.readonly_only = true;
        self.oscillated_file = None;
        self.last_failed_summary = None;
    }
}

#[derive(Debug)]
struct AdaptiveState {
    current_effort: Option<ReasoningEffort>,
    consecutive_readonly_turns: u32,
    consecutive_error_streak: u32,
    last_failure_fp: Option<String>,
    repeat_failure_count: u32,
    advisory_cooldown: u32,
    file_hashes: HashMap<PathBuf, VecDeque<String>>,
    turn: TurnActivity,
    telemetry: AdaptiveTelemetry,
}

impl Default for AdaptiveState {
    fn default() -> Self {
        Self {
            current_effort: None,
            consecutive_readonly_turns: 0,
            consecutive_error_streak: 0,
            last_failure_fp: None,
            repeat_failure_count: 0,
            advisory_cooldown: 0,
            file_hashes: HashMap::new(),
            turn: TurnActivity {
                readonly_only: true,
                ..TurnActivity::default()
            },
            telemetry: AdaptiveTelemetry::default(),
        }
    }
}

/// First-class Carnot [`Module`] implementing cache-safe adaptive effort routing and
/// ephemeral thrash / edit-oscillation recovery.
#[derive(Debug)]
pub struct AdaptiveModule {
    workspace_root: PathBuf,
    extra_writable_roots: Vec<PathBuf>,
    config: AdaptiveConfig,
    state: Mutex<AdaptiveState>,
}

impl AdaptiveModule {
    pub fn new(
        workspace_root: PathBuf,
        extra_writable_roots: Vec<PathBuf>,
        config: AdaptiveConfig,
    ) -> Self {
        Self {
            workspace_root,
            extra_writable_roots,
            config,
            state: Mutex::new(AdaptiveState::default()),
        }
    }

    /// Returns a live telemetry snapshot (useful for observability, CLI stats, and tests).
    pub fn telemetry(&self) -> AdaptiveTelemetry {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .telemetry
            .clone()
    }

    fn clamp_effort(&self, desired: ReasoningEffort) -> ReasoningEffort {
        let min_r = effort_rank(self.config.min_effort);
        let max_r = effort_rank(self.config.max_effort).max(min_r);
        rank_to_effort(effort_rank(desired).clamp(min_r, max_r))
    }

    fn set_effort(&self, state: &mut AdaptiveState, desired: ReasoningEffort) {
        let clamped = self.clamp_effort(desired);
        if state.current_effort != Some(clamped) {
            state.current_effort = Some(clamped);
            state.telemetry.current_effort = Some(clamped);
            state.telemetry.effort_transitions += 1;
        }
    }

    fn record_file_mutation(&self, state: &mut AdaptiveState, raw_path: &str) -> Option<String> {
        let Ok(resolved) =
            resolve_confined_path(&self.workspace_root, &self.extra_writable_roots, raw_path)
        else {
            return None;
        };
        let Ok(bytes) = std::fs::read(&resolved) else {
            return None;
        };
        let hash = blake3::hash(&bytes).to_hex().to_string();

        if state.file_hashes.len() >= MAX_TRACKED_FILES
            && !state.file_hashes.contains_key(&resolved)
            && let Some(evict_key) = state.file_hashes.keys().next().cloned()
        {
            state.file_hashes.remove(&evict_key);
        }
        let history = state.file_hashes.entry(resolved.clone()).or_default();
        // No-op rewrite producing identical content doesn't count as A -> B -> A oscillation.
        if history.back().is_some_and(|last| last == &hash) {
            return None;
        }
        // Detect returning to a prior distinct state (e.g. A -> B -> A or A -> B -> C -> A).
        let oscillated = history.len() >= 2 && history.iter().any(|prev| prev == &hash);
        history.push_back(hash);
        while history.len() > MAX_HASH_HISTORY_PER_FILE {
            history.pop_front();
        }
        if oscillated {
            let ws_canon = std::fs::canonicalize(&self.workspace_root)
                .unwrap_or_else(|_| self.workspace_root.clone());
            let rel = resolved
                .strip_prefix(&ws_canon)
                .unwrap_or(&resolved)
                .to_string_lossy()
                .replace('\\', "/");
            Some(rel)
        } else {
            None
        }
    }
}

const fn effort_rank(e: ReasoningEffort) -> u8 {
    match e {
        ReasoningEffort::Low => 0,
        ReasoningEffort::Medium => 1,
        ReasoningEffort::High => 2,
        ReasoningEffort::Xhigh => 3,
        ReasoningEffort::Max => 4,
    }
}

const fn rank_to_effort(r: u8) -> ReasoningEffort {
    match r {
        0 => ReasoningEffort::Low,
        1 => ReasoningEffort::Medium,
        2 => ReasoningEffort::High,
        3 => ReasoningEffort::Xhigh,
        _ => ReasoningEffort::Max,
    }
}

fn is_readonly_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "Read" | "Grep" | "ToolSearch" | "WebSearch" | "WebFetch" | "Lsp"
    )
}

fn is_file_mutation_tool(tool_name: &str) -> bool {
    matches!(tool_name, "Edit" | "Write")
}

/// Computes a stable BLAKE3 fingerprint for a failing tool call so repeated identical failures
/// (even with slight timing/elapsed-seconds variance) collapse to the same key.
fn failure_fingerprint(tool_name: &str, input: &Value, output: &ToolOutput) -> (String, String) {
    let target = match tool_name {
        "Bash" => input
            .get("command")
            .and_then(Value::as_str)
            .or_else(|| input.get("job").and_then(Value::as_str))
            .unwrap_or("")
            .trim(),
        _ => input
            .get("file_path")
            .and_then(Value::as_str)
            .or_else(|| input.get("pattern").and_then(Value::as_str))
            .unwrap_or("")
            .trim(),
    };
    let text = output.text_only();
    let sig_line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("[exit=") && !l.starts_with("[diagnostic-context"))
        .unwrap_or("");
    let norm_line = normalize_volatile_tokens(sig_line);
    let mut hasher = blake3::Hasher::new();
    hasher.update(tool_name.as_bytes());
    hasher.update(b"\0");
    hasher.update(target.as_bytes());
    hasher.update(b"\0");
    hasher.update(norm_line.as_bytes());
    let fp = hasher.finalize().to_hex().to_string();

    let summary_target = truncate_chars(target, 72);
    let summary_err = truncate_chars(sig_line, 96);
    let label = if summary_target.is_empty() {
        format!("{tool_name}: {summary_err}")
    } else if summary_err.is_empty() {
        format!("{tool_name}({summary_target})")
    } else {
        format!("{tool_name}({summary_target}) -> {summary_err}")
    };
    (fp, label)
}

/// Replaces digits with `#` so timestamps/durations/PIDs don't defeat duplicate-error detection.
fn normalize_volatile_tokens(s: &str) -> String {
    let mut out = String::with_capacity(s.len().min(160));
    let mut prev_digit = false;
    for ch in s.chars().take(160) {
        if ch.is_ascii_digit() {
            if !prev_digit {
                out.push('#');
                prev_digit = true;
            }
        } else {
            out.push(ch);
            prev_digit = false;
        }
    }
    out
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max).collect();
        t.push('…');
        t
    }
}

impl Module for AdaptiveModule {
    fn name(&self) -> &'static str {
        "adaptive"
    }

    fn select_effort(&self) -> BoxFuture<'_, Result<Option<ReasoningEffort>>> {
        Box::pin(async move {
            if !self.config.dynamic_effort {
                return Ok(None);
            }
            let eff = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .current_effort;
            Ok(eff)
        })
    }

    fn after_tool_result<'a>(
        &'a self,
        tool_name: &'a str,
        input: &'a Value,
        output: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.turn.tool_calls += 1;

            if !is_readonly_tool(tool_name) {
                state.turn.readonly_only = false;
            }
            if is_file_mutation_tool(tool_name) {
                state.turn.mutations += 1;
            }

            if output.is_error {
                state.turn.errors += 1;
                state.consecutive_error_streak += 1;
                let (fp, label) = failure_fingerprint(tool_name, input, output);
                if state.last_failure_fp.as_deref() == Some(fp.as_str()) {
                    state.repeat_failure_count += 1;
                } else {
                    state.last_failure_fp = Some(fp);
                    state.repeat_failure_count = 1;
                }
                state.turn.last_failed_summary = Some(label);
            } else {
                // Successful tool call breaks the consecutive failure streak.
                state.consecutive_error_streak = 0;
                state.last_failure_fp = None;
                state.repeat_failure_count = 0;

                if self.config.thrash_guard
                    && self.config.detect_edit_oscillation
                    && is_file_mutation_tool(tool_name)
                    && let Some(raw_path) = input.get("file_path").and_then(Value::as_str)
                    && let Some(rel) = self.record_file_mutation(&mut state, raw_path)
                {
                    state.turn.oscillated_file = Some(rel);
                }
            }

            Ok(())
        })
    }

    fn on_turn_boundary<'a>(
        &'a self,
        snapshot: &'a TurnBoundarySnapshot,
    ) -> BoxFuture<'a, Result<BoundaryAction>> {
        Box::pin(async move {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);

            if state.advisory_cooldown > 0 {
                state.advisory_cooldown -= 1;
            }

            let had_tools = snapshot.had_tool_call_in_turn && state.turn.tool_calls > 0;
            let had_error = state.turn.errors > 0;
            let oscillated_file = state.turn.oscillated_file.take();
            let last_failed = state.turn.last_failed_summary.take();
            let readonly_turn = had_tools && !had_error && state.turn.readonly_only;
            let clean_progress_turn = had_tools && !had_error && oscillated_file.is_none();

            // 1. Update adaptive effort state machine (clamped to [min_effort, max_effort]).
            if self.config.dynamic_effort && had_tools {
                if had_error || oscillated_file.is_some() {
                    state.consecutive_readonly_turns = 0;
                    self.set_effort(&mut state, self.config.max_effort);
                } else if readonly_turn {
                    state.consecutive_readonly_turns += 1;
                    if state.consecutive_readonly_turns >= READONLY_STEPDOWN_STREAK {
                        self.set_effort(&mut state, self.config.min_effort);
                    }
                } else {
                    state.consecutive_readonly_turns = 0;
                    self.set_effort(&mut state, ReasoningEffort::Medium);
                }
            } else if !had_tools {
                state.consecutive_readonly_turns = 0;
            }

            // Reset cooldown immediately once a clean productive mutation or check succeeds.
            if clean_progress_turn && state.turn.mutations > 0 {
                state.advisory_cooldown = 0;
            }

            // 2. Evaluate thrash & oscillation guardrails -> 1-turn ephemeral tail advisory!
            let mut action = BoundaryAction::Continue;
            if self.config.thrash_guard && had_tools && state.advisory_cooldown == 0 {
                if let Some(rel_path) = oscillated_file {
                    state.telemetry.edit_oscillation_alerts += 1;
                    state.advisory_cooldown = 2;
                    action = BoundaryAction::AppendEphemeralAdvisory {
                        signal: "adaptive_edit_oscillation".into(),
                        message: format!(
                            "[carnot:adaptive] Edit oscillation detected on `{rel_path}` (content reverted to a prior state in an A→B→A cycle). Pause and re-check the root cause or test expectation before modifying this file again."
                        ),
                    };
                } else if self.config.repeat_failure_threshold > 0
                    && state.repeat_failure_count >= self.config.repeat_failure_threshold
                {
                    let count = state.repeat_failure_count;
                    let detail = last_failed.unwrap_or_else(|| "the same failing command".into());
                    state.telemetry.repeat_failure_alerts += 1;
                    state.advisory_cooldown = 2;
                    action = BoundaryAction::AppendEphemeralAdvisory {
                        signal: "adaptive_repeat_failure".into(),
                        message: format!(
                            "[carnot:adaptive] {count} consecutive identical failures ({detail}). Do not repeat the same attempt—inspect the underlying assumption, check logs/environment, or change strategy."
                        ),
                    };
                } else if self.config.error_streak_threshold > 0
                    && state.consecutive_error_streak >= self.config.error_streak_threshold
                {
                    let streak = state.consecutive_error_streak;
                    state.telemetry.error_streak_alerts += 1;
                    state.advisory_cooldown = 2;
                    action = BoundaryAction::AppendEphemeralAdvisory {
                        signal: "adaptive_error_streak".into(),
                        message: format!(
                            "[carnot:adaptive] {streak} consecutive tool errors without progress. Step back, verify your assumptions with a minimal read/check, and pivot approach."
                        ),
                    };
                }
            }

            state.turn.reset();
            Ok(action)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use carnot_core::StopReason;
    use serde_json::json;

    fn snap(turn: u64, had_tools: bool) -> TurnBoundarySnapshot {
        TurnBoundarySnapshot {
            turn_index: turn,
            stop_reason: if had_tools {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            },
            total_context_tokens: 1_000,
            had_edit_in_segment: false,
            had_tool_call_in_turn: had_tools,
            files_read_in_segment: Vec::new(),
            files_edited_in_segment: Vec::new(),
            handles_created_in_segment: Vec::new(),
            last_assistant_text: String::new(),
            failed_attempts_summary: Vec::new(),
        }
    }

    #[tokio::test]
    async fn adaptive_effort_deescalates_on_readonly_streak_and_escalates_on_error() {
        let dir = std::env::temp_dir().join(format!("carnot-adapt-eff-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let m = AdaptiveModule::new(dir.clone(), Vec::new(), AdaptiveConfig::default());

        // Turn 1 starts with None (respects user default effort).
        assert_eq!(m.select_effort().await.expect("select"), None);

        // Two consecutive clean read-only exploration turns -> step down to Low!
        for t in 1..=2 {
            let mut out = ToolOutput::ok_text("file contents");
            m.after_tool_result("Read", &json!({"file_path": "src/lib.rs"}), &mut out)
                .await
                .expect("after");
            let act = m.on_turn_boundary(&snap(t, true)).await.expect("boundary");
            assert!(matches!(act, BoundaryAction::Continue), "no alert on clean read");
        }
        assert_eq!(
            m.select_effort().await.expect("select"),
            Some(ReasoningEffort::Low)
        );

        // Clean mutation turn -> Medium.
        let mut out = ToolOutput::ok_text("edited");
        m.after_tool_result("Edit", &json!({"file_path": "src/lib.rs"}), &mut out)
            .await
            .expect("after");
        let _ = m.on_turn_boundary(&snap(3, true)).await.expect("boundary");
        assert_eq!(
            m.select_effort().await.expect("select"),
            Some(ReasoningEffort::Medium)
        );

        // Failing Bash command -> immediate escalation to High (max_effort).
        let mut fail = ToolOutput::err_text("error[E0308]: mismatched types\n[exit=101 in 0.4s]");
        m.after_tool_result("Bash", &json!({"command": "cargo test"}), &mut fail)
            .await
            .expect("after");
        let _ = m.on_turn_boundary(&snap(4, true)).await.expect("boundary");
        assert_eq!(
            m.select_effort().await.expect("select"),
            Some(ReasoningEffort::High)
        );
        assert_eq!(m.telemetry().effort_transitions, 3);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn repeat_failure_and_error_streak_emit_ephemeral_advisory_with_cooldown() {
        let dir = std::env::temp_dir().join(format!("carnot-adapt-thrash-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let m = AdaptiveModule::new(dir.clone(), Vec::new(), AdaptiveConfig::default());

        // 3 identical failing runs (with slightly varying timestamps/numbers) trigger repeat-failure alert.
        for t in 1..=3 {
            let mut fail = ToolOutput::err_text(format!(
                "FAILED tests/test_auth.py::test_login (0.{t}2s)\n[exit=1 in 0.{t}2s]"
            ));
            m.after_tool_result("Bash", &json!({"command": "pytest"}), &mut fail)
                .await
                .expect("after");
            let act = m.on_turn_boundary(&snap(t, true)).await.expect("boundary");
            if t < 3 {
                assert!(matches!(act, BoundaryAction::Continue), "no early alert");
            } else {
                match act {
                    BoundaryAction::AppendEphemeralAdvisory { signal, message } => {
                        assert_eq!(signal, "adaptive_repeat_failure");
                        assert!(
                            message.contains("3 consecutive identical failures"),
                            "got: {message}"
                        );
                    }
                    other => panic!("expected AppendEphemeralAdvisory, got: {other:?}"),
                }
            }
        }
        assert_eq!(m.telemetry().repeat_failure_alerts, 1);

        // Next turn is on cooldown -> does not spam.
        let mut fail4 = ToolOutput::err_text("FAILED tests/test_auth.py::test_login (0.99s)");
        m.after_tool_result("Bash", &json!({"command": "pytest"}), &mut fail4)
            .await
            .expect("after");
        let act4 = m.on_turn_boundary(&snap(4, true)).await.expect("boundary");
        assert!(
            matches!(act4, BoundaryAction::Continue),
            "cooldown must suppress consecutive advisory spam"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn detects_a_b_a_file_edit_oscillation_via_blake3() {
        let dir = std::env::temp_dir().join(format!("carnot-adapt-osc-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let file = dir.join("app.py");
        let m = AdaptiveModule::new(dir.clone(), Vec::new(), AdaptiveConfig::default());

        // State A
        std::fs::write(&file, "x = 1\n").expect("write A");
        let mut ok = ToolOutput::ok_text("ok");
        m.after_tool_result("Edit", &json!({"file_path": "app.py"}), &mut ok)
            .await
            .expect("after A");
        assert!(
            matches!(
                m.on_turn_boundary(&snap(1, true)).await.expect("b1"),
                BoundaryAction::Continue
            ),
            "initial state A has no oscillation"
        );

        // State B
        std::fs::write(&file, "x = 2\n").expect("write B");
        let mut ok2 = ToolOutput::ok_text("ok");
        m.after_tool_result("Edit", &json!({"file_path": "app.py"}), &mut ok2)
            .await
            .expect("after B");
        assert!(
            matches!(
                m.on_turn_boundary(&snap(2, true)).await.expect("b2"),
                BoundaryAction::Continue
            ),
            "state B has no oscillation"
        );

        // Back to State A -> A -> B -> A oscillation detected + effort escalated to High!
        std::fs::write(&file, "x = 1\n").expect("write A again");
        let mut ok3 = ToolOutput::ok_text("ok");
        m.after_tool_result("Edit", &json!({"file_path": "app.py"}), &mut ok3)
            .await
            .expect("after A2");
        let act3 = m.on_turn_boundary(&snap(3, true)).await.expect("b3");
        match act3 {
            BoundaryAction::AppendEphemeralAdvisory { signal, message } => {
                assert_eq!(signal, "adaptive_edit_oscillation");
                assert!(message.contains("app.py"), "got: {message}");
            }
            other => panic!("expected oscillation advisory, got: {other:?}"),
        }
        assert_eq!(
            m.select_effort().await.expect("select"),
            Some(ReasoningEffort::High)
        );
        assert_eq!(m.telemetry().edit_oscillation_alerts, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
