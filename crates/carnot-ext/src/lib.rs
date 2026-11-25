//! Modular plugin architecture (`Module` trait + `PermissionBroker`) and lean core modules:
//!   - `ExecMicroSkillsModule` (Turn-1 repo facts + `AGENTS.md`/`CLAUDE.md`/`SKILL.md`)
//!   - `prefix_guard` (continuous runtime & CI static prefix-invariance auditor)
//!   - `IsolatedForkWorktree` (safe `git worktree add --detach` + isolated `CARGO_TARGET_DIR`)
//!   - `PolicyGuardModule` (enterprise command/path guardrails + secret redaction + JSONL audit)

pub mod ask_user;
pub mod dropin;
pub mod microskills;
pub mod policy;
pub mod prefix_guard;
pub mod warm_fork;

use carnot_core::{
    CarnotConfig, ReasoningEffort, Result, StopReason, ToolOutput, UserQuestion, UserQuestionAnswer,
};
use carnot_tools::Tool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Decision returned by a module's `before_tool_call` hook (permission gate / rewrite).
#[derive(Debug, Clone)]
pub enum ToolGateDecision {
    Allow,
    RewriteInput(Value),
    Deny { reason: String },
}

/// Interactive permission broker trait.
/// Lives in `carnot-ext` so `carnot-agent` invokes it without depending on `carnot-protocol`,
/// while ACP implements it via `session/request_permission`.
pub trait PermissionBroker: Send + Sync + std::fmt::Debug {
    fn request_permission(
        &self,
        call_id: &str,
        tool_name: &str,
        input: &Value,
    ) -> BoxFuture<'_, Result<ToolGateDecision>>;
}

/// Interactive user question broker trait for the deferred `AskUser` tool.
/// Lives in `carnot-ext` so `carnot-agent` and `AskUserModule` invoke it cleanly across ACP,
/// Host RPC, HTTP/SSE, terminal REPL, and custom Rust UI embedders.
pub trait UserQuestionBroker: Send + Sync + std::fmt::Debug {
    fn ask<'a>(
        &'a self,
        request_id: &'a str,
        questions: &'a [UserQuestion],
    ) -> BoxFuture<'a, Result<Vec<UserQuestionAnswer>>>;
}

/// Action returned by a module at a turn boundary (`on_turn_boundary`).
#[derive(Debug, Clone)]
pub enum BoundaryAction {
    Continue,
    /// Append an advisory `system` note (append-only, persisted to session, cache-safe).
    AppendAdvisory {
        signal: String,
        message: String,
    },
    /// Attach a one-turn ephemeral advisory note AFTER all provider cache breakpoints
    /// (`ephemeral_tail_notes`, never persisted to session history — zero context bloat).
    AppendEphemeralAdvisory {
        signal: String,
        message: String,
    },
    /// Append a `system` note and request another turn even when the model produced no tool calls.
    ContinueRun {
        signal: String,
        message: String,
    },
    /// Stop autonomous loop cleanly.
    Halt {
        reason: String,
    },
}

/// Snapshot of turn-level facts provided to modules at `on_turn_boundary`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnBoundarySnapshot {
    pub turn_index: u64,
    #[serde(default)]
    pub stop_reason: StopReason,
    pub total_context_tokens: u64,
    pub had_edit_in_segment: bool,
    pub had_tool_call_in_turn: bool,
    pub files_read_in_segment: Vec<String>,
    pub files_edited_in_segment: Vec<String>,
    pub handles_created_in_segment: Vec<String>,
    pub last_assistant_text: String,
    pub failed_attempts_summary: Vec<String>,
}

/// Context passed to `Module::before_compact` before proactive or manual compaction runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionHookContext {
    pub mode: String,
    pub tokens_before: u64,
    pub message_count: usize,
    pub custom_focus: Option<String>,
}

/// Decision returned by `Module::before_compact`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionHookAction {
    Continue,
    /// Append extra focus instructions to the semantic compaction handoff prompt.
    AddFocus(String),
    /// Cancel a non-emergency compaction (ignored during `emergency` overflow recovery).
    Cancel {
        reason: String,
    },
}

/// Per-turn context note returned by `Module::before_prompt`.
/// Always injected append-only into conversation messages (NEVER into the static system prompt).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptInjection {
    pub text: String,
    /// When `true`, attached as an ephemeral system note for this turn rather than a durable note.
    #[serde(default)]
    pub ephemeral: bool,
}

/// First-class in-process Rust module interface.
///
/// Every built-in Carnot capability implements this exact trait, so external Rust embedders
/// have the same lifecycle hooks as core modules.
pub trait Module: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;

    /// Optional extra tools contributed by this module (registered at session start).
    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        Vec::new()
    }

    /// Optional Turn-1 dynamic context injected in the FIRST user message (NEVER in static system prompt!).
    fn on_session_start(&self, _config: &CarnotConfig) -> BoxFuture<'_, Result<Option<String>>> {
        Box::pin(async { Ok(None) })
    }

    /// Optional per-prompt append-only context / reminders injected before the model runs.
    fn before_prompt<'a>(
        &'a self,
        _prompt_text: &'a str,
    ) -> BoxFuture<'a, Result<Vec<PromptInjection>>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    /// Optional per-turn reasoning effort selector (applied ONLY when the active provider/model
    /// preserves its prompt cache across effort changes).
    fn select_effort(&self) -> BoxFuture<'_, Result<Option<ReasoningEffort>>> {
        Box::pin(async { Ok(None) })
    }

    /// Permission gate or input transformation before a tool runs.
    fn before_tool_call(
        &self,
        _tool_name: &str,
        _input: &Value,
    ) -> BoxFuture<'_, Result<ToolGateDecision>> {
        Box::pin(async { Ok(ToolGateDecision::Allow) })
    }

    /// Permission gate carrying the provider's `call_id` for correlation with live UI / host approvals.
    fn before_tool_call_with_id<'a>(
        &'a self,
        _call_id: &'a str,
        tool_name: &'a str,
        input: &'a Value,
    ) -> BoxFuture<'a, Result<ToolGateDecision>> {
        self.before_tool_call(tool_name, input)
    }

    /// Observation hook immediately after a tool completes.
    fn after_tool_result<'a>(
        &'a self,
        _tool_name: &'a str,
        _input: &'a Value,
        _output: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    /// Turn-boundary hook invoked after all tool calls of a turn finish.
    fn on_turn_boundary<'a>(
        &'a self,
        _snapshot: &'a TurnBoundarySnapshot,
    ) -> BoxFuture<'a, Result<BoundaryAction>> {
        Box::pin(async { Ok(BoundaryAction::Continue) })
    }

    /// Optional pre-compaction hook (add custom focus constraints or veto non-emergency compaction).
    fn before_compact<'a>(
        &'a self,
        _ctx: &'a CompactionHookContext,
    ) -> BoxFuture<'a, Result<CompactionHookAction>> {
        Box::pin(async { Ok(CompactionHookAction::Continue) })
    }
}

/// Lightweight runtime control bridge for goal-driven verification (`carnot-ext-goal`).
///
/// Exposed via `Agent::goal_controller()` so TUI (`/goal`), RPC, HTTP, and SDKs can inspect,
/// set, or clear an active goal live without coupling core crates to `carnot-ext-goal`.
pub trait GoalController: Send + Sync + std::fmt::Debug {
    /// Returns a JSON snapshot of the current goal state.
    fn status_json(&self) -> Value;

    /// Replaces the active goal configuration and returns the updated JSON state.
    fn set_goal(
        &self,
        criterion: Option<String>,
        verify_commands: Vec<String>,
        max_iterations: Option<u32>,
        verifier_model: Option<String>,
    ) -> Value;

    /// Clears any active goal and returns the updated JSON state.
    fn clear_goal(&self) -> Value;
}
