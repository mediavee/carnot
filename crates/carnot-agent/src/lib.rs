//! Core asynchronous agent loop with:
//!   - Effect-Based Parallel Scheduler (gate-before-eager reads + parallel disjoint `Edit`/`Write`
//!     + strict causal ordering on overlapping paths & `Bash` barriers)
//!   - Lossless monotone-`seq` `EventLog` (immune to slow-reader `Lagged` disconnects)
//!   - Mid-turn `steer()` injection preserving `tool_result`-first ordering
//!   - Safe `StopReason` (`MaxTokens`, `Refusal`, `ContextWindowExceeded`) & malformed tool-JSON guards
//!   - Module lifecycle orchestration and live session telemetry

pub mod compaction;
mod context;
pub mod event_log;
mod locks;
mod scheduler;
mod segment;
mod session_ops;
#[cfg(test)]
mod tests;
pub mod tool_search;
mod turn;
mod turn_tools;

use crate::compaction::hydrate_anchors_from_session;
use crate::context::{
    PrefixFingerprint, apply_excluded_tools, effective_static_system_prompt,
    last_date_tag_in_messages,
};
use crate::locks::ToolEffectLocks;
use crate::segment::SegmentTracker;

pub use event_log::{EventLog, EventStream};

use carnot_ai::Provider;
use carnot_core::{
    AgentEvent, CacheTtl, CarnotConfig, Message, NativeWebConfig, ReasoningEffort, Result,
    SessionMetrics, ToolResultBlock, ToolSpec,
};
use carnot_ext::{
    Module, PermissionBroker,
    ask_user::{AskUserModule, SharedQuestionBroker},
};
use carnot_session::SessionStore;
use carnot_tools::ToolRegistry;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Mutex, Notify};
use tokio_util::sync::CancellationToken;

pub(crate) fn add_cost_micros(counter: &AtomicU64, cost_usd: f64) {
    if cost_usd.is_finite() && cost_usd > 0.0 {
        let micros = (cost_usd * 1_000_000.0).round() as u64;
        if micros > 0 {
            counter.fetch_add(micros, Ordering::Relaxed);
        }
    }
}

/// Snapshot captured from a parent `Agent` to construct a prompt-cache-warm forked child.
#[derive(Debug, Clone)]
pub struct ForkSnapshot {
    pub messages: Vec<Arc<Message>>,
    pub checkpoint_index: Option<usize>,
    pub cache_key_id: String,
    pub tool_specs: Vec<ToolSpec>,
    pub native_web: NativeWebConfig,
}

#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub final_text: String,
    pub metrics: SessionMetrics,
    pub halted_reason: Option<String>,
}

#[derive(Debug, Clone)]
struct ActiveRuntime {
    provider: Arc<dyn Provider>,
    model: String,
    effort: ReasoningEffort,
    cache_ttl: CacheTtl,
    context_window: u64,
    max_output_tokens: u32,
}

/// The Carnot Agent runtime holding one session, provider, tool registry, module chain, and `EventLog`.
#[derive(Debug)]
pub struct Agent {
    pub config: CarnotConfig,
    runtime: Mutex<ActiveRuntime>,
    tools: Arc<ToolRegistry>,
    modules: Vec<Arc<dyn Module>>,
    session: Mutex<SessionStore>,
    metrics: Mutex<SessionMetrics>,
    total_cost_micros: Arc<AtomicU64>,
    events: Arc<EventLog>,
    run_lock: Mutex<()>,
    cancel_current: Mutex<CancellationToken>,
    turn1_injected: Mutex<bool>,
    raw_user_goal: Mutex<Option<String>>,
    user_instructions: Mutex<Vec<String>>,
    pending_steers: Mutex<Vec<String>>,
    pending_system_notes: Mutex<Vec<String>>,
    steer_notify: Arc<Notify>,
    pending_follow_ups: Mutex<VecDeque<String>>,
    permission_broker: Mutex<Option<Arc<dyn PermissionBroker>>>,
    question_broker: SharedQuestionBroker,
    goal_controller: std::sync::RwLock<Option<Arc<dyn carnot_ext::GoalController>>>,
    deferred_server_descriptions: std::sync::Mutex<std::collections::BTreeMap<String, String>>,
    /// Live runtime prefix-freeze fingerprint checked before every provider request (< 5 µs)
    /// to detect unexpected static-prefix or reused-history cache breaks immediately.
    last_prefix_state: Mutex<Option<PrefixFingerprint>>,
    /// Provider-billed context tokens from the previous turn on the same unbroken prefix,
    /// used to compute `SessionMetrics::prefix_retention_rate()`.
    last_prompt_context_tokens: Mutex<Option<u64>>,
    /// Effect-based async lock coordinator for nested tool calls (`invoke_tool_gated`).
    effect_locks: ToolEffectLocks,
    /// Active segment facts shared by both direct tool waves and nested `invoke_tool_gated` calls
    /// so compaction never loses track of files mutated or CAS handles created inside `CodeMode`.
    active_segment: Mutex<SegmentTracker>,
    /// Cached UTC `YYYY-MM-DD` of the last emitted `[date: ... UTC]` note (O(1) per turn).
    last_emitted_date: Mutex<Option<String>>,
}

impl Agent {
    pub fn new(
        config: CarnotConfig,
        provider: Arc<dyn Provider>,
        mut tools: ToolRegistry,
        mut modules: Vec<Arc<dyn Module>>,
        persist_session: bool,
    ) -> Result<Self> {
        let question_broker: SharedQuestionBroker = Arc::new(Mutex::new(None));
        if config.enable_ask_user {
            modules.push(Arc::new(AskUserModule::new(Arc::clone(&question_broker))));
        }
        for m in &modules {
            for t in m.tools() {
                tools.register(t);
            }
        }
        apply_excluded_tools(&mut tools, &config.excluded_tools);
        // Always register universal `ToolSearch` + `CallDeferredTool` when deferred tools exist so
        // live cross-provider model switching (Anthropic <-> OpenAI/Gemini/...) always works!
        // `effective_tools_for_provider()` hides them on Anthropic's native tool-search requests.
        let deferred = tools.deferred_tools();
        if !deferred.is_empty() {
            tools.register(Arc::new(tool_search::ToolSearchTool::new(&deferred)));
            tools.register(Arc::new(tool_search::CallDeferredTool::new(&deferred)));
        }

        let tool_specs = tool_search::effective_tools_for_provider(
            &tools.specs(),
            provider.supports_native_tool_search(),
        );
        let session_dir = persist_session.then_some(config.session_dir.as_path());
        let session = SessionStore::create_with_workspace(
            session_dir,
            Some(config.workspace_root.clone()),
            config.model.clone(),
            effective_static_system_prompt(&config),
            tool_specs,
        )?;
        Ok(Self::from_session(
            config,
            provider,
            tools,
            modules,
            session,
            false,
            question_broker,
        ))
    }

    /// Resumes an existing crash-safe JSONL v2 session from disk (`carnot --resume` & ACP `session/load`).
    pub fn resume(
        session_path: &Path,
        config: CarnotConfig,
        provider: Arc<dyn Provider>,
        mut tools: ToolRegistry,
        mut modules: Vec<Arc<dyn Module>>,
    ) -> Result<Self> {
        let question_broker: SharedQuestionBroker = Arc::new(Mutex::new(None));
        if config.enable_ask_user {
            modules.push(Arc::new(AskUserModule::new(Arc::clone(&question_broker))));
        }
        for m in &modules {
            for t in m.tools() {
                tools.register(t);
            }
        }
        apply_excluded_tools(&mut tools, &config.excluded_tools);
        let deferred = tools.deferred_tools();
        if !deferred.is_empty() {
            tools.register(Arc::new(tool_search::ToolSearchTool::new(&deferred)));
            tools.register(Arc::new(tool_search::CallDeferredTool::new(&deferred)));
        }
        let mut session = SessionStore::load(session_path)?;
        let _ = session.repair_dangling_tool_uses()?;
        let already_started = !session.active_messages().is_empty();
        Ok(Self::from_session(
            config,
            provider,
            tools,
            modules,
            session,
            already_started,
            question_broker,
        ))
    }

    /// Forks a saved JSONL v2 session from disk into a brand-new session (`carnot --resume <id> --fork`),
    /// leaving the original session file untouched while preserving the warm prompt-cache prefix.
    pub fn resume_forked(
        session_path: &Path,
        config: CarnotConfig,
        provider: Arc<dyn Provider>,
        mut tools: ToolRegistry,
        mut modules: Vec<Arc<dyn Module>>,
        persist_session: bool,
    ) -> Result<Self> {
        let question_broker: SharedQuestionBroker = Arc::new(Mutex::new(None));
        if config.enable_ask_user {
            modules.push(Arc::new(AskUserModule::new(Arc::clone(&question_broker))));
        }
        for m in &modules {
            for t in m.tools() {
                tools.register(t);
            }
        }
        apply_excluded_tools(&mut tools, &config.excluded_tools);
        let deferred = tools.deferred_tools();
        if !deferred.is_empty() {
            tools.register(Arc::new(tool_search::ToolSearchTool::new(&deferred)));
            tools.register(Arc::new(tool_search::CallDeferredTool::new(&deferred)));
        }
        let original = SessionStore::load_readonly(session_path)?;
        let session_dir = persist_session.then_some(config.session_dir.as_path());
        let mut forked = original.fork(session_dir)?;
        let _ = forked.repair_dangling_tool_uses()?;
        let already_started = !forked.active_messages().is_empty();
        Ok(Self::from_session(
            config,
            provider,
            tools,
            modules,
            forked,
            already_started,
            question_broker,
        ))
    }

    /// Creates a new `Agent` initialized from a parent conversation snapshot (zero-copy `Arc<Message>`
    /// fork for prompt-cache-warm `Subagent` execution or programmatic branching).
    pub fn new_forked(
        config: CarnotConfig,
        provider: Arc<dyn Provider>,
        tools: ToolRegistry,
        modules: Vec<Arc<dyn Module>>,
        persist_session: bool,
        parent_messages: &[Arc<Message>],
        checkpoint_index: Option<usize>,
    ) -> Result<Self> {
        Self::new_forked_with_cache_key(
            config,
            provider,
            tools,
            modules,
            persist_session,
            parent_messages,
            checkpoint_index,
            None,
        )
    }

    /// Creates a forked `Agent` that also inherits the parent's `cache_key_id` so `OpenAI`
    /// `prompt_cache_key` shards stay hot on the forked branch.
    #[expect(
        clippy::too_many_arguments,
        reason = "forked agent constructor preserves explicit arguments aligned with Agent::new"
    )]
    pub fn new_forked_with_cache_key(
        config: CarnotConfig,
        provider: Arc<dyn Provider>,
        mut tools: ToolRegistry,
        mut modules: Vec<Arc<dyn Module>>,
        persist_session: bool,
        parent_messages: &[Arc<Message>],
        checkpoint_index: Option<usize>,
        inherited_cache_key_id: Option<String>,
    ) -> Result<Self> {
        let question_broker: SharedQuestionBroker = Arc::new(Mutex::new(None));
        if config.enable_ask_user {
            modules.push(Arc::new(AskUserModule::new(Arc::clone(&question_broker))));
        }
        for m in &modules {
            for t in m.tools() {
                tools.register(t);
            }
        }
        apply_excluded_tools(&mut tools, &config.excluded_tools);
        let deferred = tools.deferred_tools();
        if !deferred.is_empty() {
            tools.register(Arc::new(tool_search::ToolSearchTool::new(&deferred)));
            tools.register(Arc::new(tool_search::CallDeferredTool::new(&deferred)));
        }
        let tool_specs = tool_search::effective_tools_for_provider(
            &tools.specs(),
            provider.supports_native_tool_search(),
        );
        let session_dir = persist_session.then_some(config.session_dir.as_path());
        let session = SessionStore::from_forked_snapshot_with_cache_key(
            session_dir,
            Some(config.workspace_root.clone()),
            config.model.clone(),
            effective_static_system_prompt(&config),
            tool_specs,
            parent_messages,
            checkpoint_index,
            inherited_cache_key_id,
        )?;
        let already_started = !session.active_messages().is_empty();
        Ok(Self::from_session(
            config,
            provider,
            tools,
            modules,
            session,
            already_started,
            question_broker,
        ))
    }

    fn from_session(
        config: CarnotConfig,
        provider: Arc<dyn Provider>,
        tools: ToolRegistry,
        modules: Vec<Arc<dyn Module>>,
        mut session: SessionStore,
        turn1_already_injected: bool,
        question_broker: SharedQuestionBroker,
    ) -> Self {
        let (hydrated_goal, hydrated_followups) =
            hydrate_anchors_from_session(session.active_messages());
        let events = Arc::new(EventLog::new());

        // Detect tool-set drift on `--resume` / `--resume --fork` (e.g. session originally ran with
        // `--web` and was resumed without it): refresh `prefix_hash` so `prompt_cache_key` stays
        // accurate and emit `PrefixAudit { stable: false }`.
        let live_wire_tools = tool_search::effective_tools_for_provider(
            &tools.specs(),
            provider.supports_native_tool_search(),
        );
        if let Ok(live_hash) = carnot_session::compute_static_prefix_hash(
            &session.static_system_prompt,
            &live_wire_tools,
        ) && live_hash != session.prefix_hash
        {
            let old_hash = std::mem::replace(&mut session.prefix_hash, live_hash.clone());
            events.push(AgentEvent::PrefixAudit {
                prefix_hash: live_hash,
                stable: false,
                note: Some(format!(
                    "session resumed with a different tool set than saved header ({old_hash}); provider prompt cache will cold-start"
                )),
            });
        }

        events.push(AgentEvent::SessionStarted {
            session_id: session.session_id.clone(),
            model: config.model.clone(),
            prefix_hash: session.prefix_hash.clone(),
        });

        let runtime = ActiveRuntime {
            provider,
            model: config.model.clone(),
            effort: config.effort,
            cache_ttl: config.cache_ttl,
            context_window: config.context_window,
            max_output_tokens: config.max_output_tokens,
        };

        let initial_date = last_date_tag_in_messages(session.active_messages());

        Self {
            config,
            runtime: Mutex::new(runtime),
            tools: Arc::new(tools),
            modules,
            session: Mutex::new(session),
            metrics: Mutex::new(SessionMetrics::default()),
            total_cost_micros: Arc::new(AtomicU64::new(0)),
            events,
            run_lock: Mutex::new(()),
            cancel_current: Mutex::new(CancellationToken::new()),
            turn1_injected: Mutex::new(turn1_already_injected),
            raw_user_goal: Mutex::new(hydrated_goal),
            user_instructions: Mutex::new(hydrated_followups),
            pending_steers: Mutex::new(Vec::new()),
            pending_system_notes: Mutex::new(Vec::new()),
            steer_notify: Arc::new(Notify::new()),
            pending_follow_ups: Mutex::new(VecDeque::new()),
            permission_broker: Mutex::new(None),
            question_broker,
            goal_controller: std::sync::RwLock::new(None),
            deferred_server_descriptions: std::sync::Mutex::new(std::collections::BTreeMap::new()),
            last_prefix_state: Mutex::new(None),
            last_prompt_context_tokens: Mutex::new(None),
            effect_locks: ToolEffectLocks::default(),
            active_segment: Mutex::new(SegmentTracker::default()),
            last_emitted_date: Mutex::new(initial_date),
        }
    }
}

/// Convenience helper returning a `ToolResultBlock` vector for external callers.
pub fn text_result(s: impl Into<String>) -> Vec<ToolResultBlock> {
    vec![ToolResultBlock::text(s)]
}
