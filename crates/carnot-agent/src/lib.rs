//! Core asynchronous agent loop with:
//!   - Opus I2 Effect-Based Parallel Scheduler (gate-before-eager reads + parallel disjoint `Edit`/`Write`
//!     + strict causal ordering on overlapping paths & `Bash` barriers)
//!   - Opus I4 Lossless monotone-`seq` `EventLog` (immune to slow-reader `Lagged` disconnects)
//!   - Mid-turn `steer()` injection preserving `tool_result`-first ordering
//!   - Safe `StopReason` (`MaxTokens`, `Refusal`, `ContextWindowExceeded`) & malformed tool-JSON guards
//!   - Module lifecycle orchestration and live thermodynamic telemetry

pub mod compaction;
pub mod event_log;
mod scheduler;
mod segment;
pub mod tool_search;

use crate::compaction::{
    CompactionMode, clip_anchor_text, effective_proactive_trigger, execute_compaction,
    hydrate_anchors_from_session,
};
use crate::scheduler::{ActiveEager, EagerLaunch, ScheduledCall, partition_conflict_free_waves};
use crate::segment::SegmentTracker;

pub use event_log::{EventLog, EventStream};

use carnot_ai::{Provider, ProviderRequest, StreamEvent};
use carnot_core::{
    AgentEvent, CacheTtl, CarnotConfig, CarnotError, ContentBlock, ContextBreakdown, Message,
    NativeWebConfig, ReasoningEffort, Result, Role, StopReason, ThermodynamicMetrics, ToolEffects,
    ToolOutput, ToolResultBlock, ToolSpec, Usage,
};
use carnot_ext::{
    BoundaryAction, Module, PermissionBroker, ToolGateDecision, TurnBoundarySnapshot,
    UserQuestionBroker,
    ask_user::{AskUserModule, SharedQuestionBroker},
    prefix_guard::static_system_prompt,
};
use carnot_session::SessionStore;
use carnot_tools::ToolRegistry;
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
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
struct PrefixFingerprint {
    provider: String,
    model: String,
    static_hash: String,
    msg_count: usize,
    msg_prefix_hash: String,
}

fn compute_wire_static_hash(
    static_system_prompt: &str,
    wire_tools: &[ToolSpec],
    native_web: &NativeWebConfig,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(static_system_prompt.as_bytes());
    h.update(&[0u8]);
    if let Ok(tb) = serde_json::to_vec(wire_tools) {
        h.update(&tb);
    }
    h.update(&[0u8]);
    if let Ok(nb) = serde_json::to_vec(native_web) {
        h.update(&nb);
    }
    h.finalize().to_hex().to_string()
}

fn compute_messages_prefix_hash(messages: &[Arc<Message>]) -> String {
    let mut h = blake3::Hasher::new();
    for m in messages {
        if let Ok(b) = serde_json::to_vec(m.as_ref()) {
            h.update(&(b.len() as u64).to_le_bytes());
            h.update(&b);
        }
    }
    h.finalize().to_hex().to_string()
}

/// Output of a completed `Agent::prompt()` run.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub final_text: String,
    pub metrics: ThermodynamicMetrics,
    pub halted_reason: Option<String>,
}

#[derive(Debug, Clone)]
struct ActiveRuntime {
    provider: Arc<dyn Provider>,
    model: String,
    effort: ReasoningEffort,
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
    metrics: Mutex<ThermodynamicMetrics>,
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
    deferred_server_descriptions: std::sync::Mutex<std::collections::BTreeMap<String, String>>,
    /// Live runtime prefix-freeze fingerprint checked before every provider request (< 5 µs)
    /// to detect unexpected static-prefix or reused-history cache breaks immediately.
    last_prefix_state: Mutex<Option<PrefixFingerprint>>,
}

fn effective_static_system_prompt(config: &CarnotConfig) -> String {
    let base: &str = config
        .static_system_prompt_override
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| static_system_prompt());
    match config
        .static_system_prompt_append
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(extra) => format!("{base}\n\n{extra}"),
        None => base.to_string(),
    }
}

fn apply_excluded_tools(tools: &mut ToolRegistry, excluded: &[String]) {
    for name in excluded {
        let trimmed = name.trim();
        if !trimmed.is_empty() {
            tools.remove(trimmed);
        }
    }
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
        Self::from_session(
            config,
            provider,
            tools,
            modules,
            session,
            false,
            question_broker,
        )
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
        Self::from_session(
            config,
            provider,
            tools,
            modules,
            session,
            already_started,
            question_broker,
        )
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
        Self::from_session(
            config,
            provider,
            tools,
            modules,
            forked,
            already_started,
            question_broker,
        )
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

    /// Creates a forked `Agent` that also inherits the parent's `cache_key_id` so OpenAI
    /// `prompt_cache_key` shards stay hot on the forked branch.
    #[allow(clippy::too_many_arguments)]
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
            static_system_prompt().to_string(),
            tool_specs,
            parent_messages,
            checkpoint_index,
            inherited_cache_key_id,
        )?;
        let already_started = !session.active_messages().is_empty();
        Self::from_session(
            config,
            provider,
            tools,
            modules,
            session,
            already_started,
            question_broker,
        )
    }

    fn from_session(
        config: CarnotConfig,
        provider: Arc<dyn Provider>,
        tools: ToolRegistry,
        modules: Vec<Arc<dyn Module>>,
        mut session: SessionStore,
        turn1_already_injected: bool,
        question_broker: SharedQuestionBroker,
    ) -> Result<Self> {
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
            let old_hash = session.prefix_hash.clone();
            session.prefix_hash = live_hash.clone();
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
            context_window: config.context_window,
            max_output_tokens: config.max_output_tokens,
        };

        Ok(Self {
            config,
            runtime: Mutex::new(runtime),
            tools: Arc::new(tools),
            modules,
            session: Mutex::new(session),
            metrics: Mutex::new(ThermodynamicMetrics::default()),
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
            deferred_server_descriptions: std::sync::Mutex::new(std::collections::BTreeMap::new()),
            last_prefix_state: Mutex::new(None),
        })
    }

    /// Attaches one-line MCP server descriptions displayed in the Turn-1 `[available-deferred-tools]`
    /// directory (zero impact on the static cached system prompt).
    pub fn set_deferred_server_descriptions(
        &self,
        descriptions: std::collections::BTreeMap<String, String>,
    ) {
        if let Ok(mut guard) = self.deferred_server_descriptions.lock() {
            *guard = descriptions;
        }
    }

    /// Loads an existing session file into this running `Agent` (used by ACP `session/load`).
    /// Fails immediately if a prompt is currently running so ACP stdin never deadlocks (Opus #7).
    pub async fn load_session(&self, session_path: &Path) -> Result<String> {
        let Ok(_guard) = self.run_lock.try_lock() else {
            return Err(CarnotError::Session(
                "agent is busy running a prompt; cancel first before loading a session".into(),
            ));
        };
        let loaded = SessionStore::load(session_path)?;
        let sid = loaded.session_id.clone();
        let has_msgs = !loaded.active_messages().is_empty();
        let (hydrated_goal, hydrated_followups) =
            hydrate_anchors_from_session(loaded.active_messages());
        *self.session.lock().await = loaded;
        *self.turn1_injected.lock().await = has_msgs;
        *self.raw_user_goal.lock().await = hydrated_goal;
        *self.user_instructions.lock().await = hydrated_followups;
        *self.last_prefix_state.lock().await = None;
        self.tools.reset_read_tracker();
        Ok(sid)
    }

    /// Returns a snapshot of active conversation messages (for ACP `session/load` history replay).
    pub async fn active_messages(&self) -> Vec<Arc<Message>> {
        self.session.lock().await.active_messages().to_vec()
    }

    /// Returns a sanitized zero-copy snapshot `(active_messages, checkpoint_index)` safe to seed
    /// a forked session or forked subagent — even while a parent tool call is in flight.
    pub async fn snapshot_for_fork(&self) -> (Vec<Arc<Message>>, Option<usize>) {
        let s = self.session.lock().await;
        let msgs = SessionStore::sanitize_fork_messages(s.active_messages());
        let cp = s.checkpoint_index().filter(|&idx| idx < msgs.len());
        (msgs, cp)
    }

    /// Returns a full prompt-cache-warm fork snapshot including `cache_key_id`, canonical tool
    /// specs, and `native_web` config so forked subagents share a 100% byte-identical wire prefix
    /// with the coordinator even when optional extensions or MCP servers are active.
    pub async fn fork_snapshot_with_prefix(&self) -> ForkSnapshot {
        let s = self.session.lock().await;
        let messages = SessionStore::sanitize_fork_messages(s.active_messages());
        let checkpoint_index = s.checkpoint_index().filter(|&idx| idx < messages.len());
        let cache_key_id = s.cache_key_id.clone();
        drop(s);
        ForkSnapshot {
            messages,
            checkpoint_index,
            cache_key_id,
            tool_specs: self.tools.specs(),
            native_web: self.config.native_web.clone(),
        }
    }

    /// Forks this live session into a brand-new session ID (persisting a new JSONL file if the
    /// current session is persisted) and switches this `Agent` onto the fork while leaving the
    /// original session file untouched on disk.
    pub async fn fork_session(&self) -> Result<(String, Option<std::path::PathBuf>)> {
        let Ok(_guard) = self.run_lock.try_lock() else {
            return Err(CarnotError::Session(
                "agent is busy running a prompt; cancel first before forking session".into(),
            ));
        };
        let (forked, new_id, new_path, new_prefix, cur_model, sanitized_shrank) = {
            let s = self.session.lock().await;
            let prev_len = s.active_messages().len();
            let dir = s.path.as_ref().and_then(|p| p.parent());
            let forked = s.fork(dir)?;
            let shrank = forked.active_messages().len() != prev_len;
            let new_id = forked.session_id.clone();
            let new_path = forked.path.clone();
            let new_prefix = forked.prefix_hash.clone();
            let cur_model = self.runtime.lock().await.model.clone();
            (forked, new_id, new_path, new_prefix, cur_model, shrank)
        };
        *self.session.lock().await = forked;
        if sanitized_shrank {
            *self.last_prefix_state.lock().await = None;
        }
        self.events.push(AgentEvent::SessionStarted {
            session_id: new_id.clone(),
            model: cur_model,
            prefix_hash: new_prefix,
        });
        Ok((new_id, new_path))
    }

    /// Returns the currently active `Provider` (reflects live `/model` switches).
    pub async fn current_provider(&self) -> Arc<dyn Provider> {
        Arc::clone(&self.runtime.lock().await.provider)
    }

    /// Attaches an interactive `PermissionBroker` (e.g. ACP `session/request_permission`, RPC/HTTP).
    pub async fn set_permission_broker(&self, broker: Arc<dyn PermissionBroker>) {
        *self.permission_broker.lock().await = Some(broker);
    }

    /// Detaches any active `PermissionBroker` (returns to auto-allow outside `PolicyGuard`/hooks).
    pub async fn clear_permission_broker(&self) {
        *self.permission_broker.lock().await = None;
    }

    /// Returns the currently attached `PermissionBroker`, if any (used by child workers so they
    /// delegate approvals to the parent broker instead of bypassing user consent!).
    pub async fn permission_broker(&self) -> Option<Arc<dyn PermissionBroker>> {
        self.permission_broker.lock().await.clone()
    }

    /// Returns a clone of the current run's `CancellationToken` so module hooks (such as reactive
    /// fleet waiting in `on_turn_boundary`) cancel immediately on `Esc` / `abort()`.
    pub async fn current_cancel_token(&self) -> CancellationToken {
        self.cancel_current.lock().await.clone()
    }

    /// Rolls up external cost (e.g. from background/orchestrated child workers) into this agent's
    /// cumulative session metrics and lock-free atomic counter so `max_cost_usd` and fleet budgets
    /// reflect true combined spend.
    pub async fn record_external_cost(&self, cost_usd: f64) {
        if cost_usd.is_finite() && cost_usd > 0.0 {
            self.metrics.lock().await.total_usage.cost_usd += cost_usd;
            add_cost_micros(&self.total_cost_micros, cost_usd);
        }
    }

    /// Returns a shared handle to this agent's monotonic micro-dollar total cost counter
    /// (includes own turns, compaction calls, and rolled-up external worker costs).
    pub fn total_cost_micros_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.total_cost_micros)
    }

    /// Returns a shared `Notify` fired whenever a human `steer()` is queued, allowing reactive
    /// turn-boundary waits (e.g. fleet orchestration) to wake immediately on user intervention.
    pub fn steer_notify(&self) -> Arc<Notify> {
        Arc::clone(&self.steer_notify)
    }

    /// True if at least one human `steer()` is waiting to be delivered.
    pub async fn has_pending_steers(&self) -> bool {
        !self.pending_steers.lock().await.is_empty()
    }

    /// Queues an untrusted/internal system note (such as a worker update or fleet heartbeat)
    /// delivered as `Message::system_note` without user-steer authority or compaction pinning.
    pub async fn queue_system_note(&self, note: impl Into<String>) {
        let msg = note.into();
        if !msg.trim().is_empty() {
            self.pending_system_notes.lock().await.push(msg);
        }
    }

    /// Pops one queued steer or follow-up message if present (used by worker lifecycle loops so
    /// messages queued right as a turn finishes are never stranded while transitioning to idle).
    pub async fn take_pending_wakeup(&self) -> Option<String> {
        {
            let mut steers = self.pending_steers.lock().await;
            if !steers.is_empty() {
                return Some(std::mem::take(&mut *steers).join("\n"));
            }
        }
        self.pending_follow_ups.lock().await.pop_front()
    }

    /// Returns `(eager_tool_names, deferred_tool_names)` registered on this agent.
    pub fn tool_names(&self) -> (Vec<String>, Vec<String>) {
        let mut eager = Vec::new();
        let mut deferred = Vec::new();
        for s in self.tools.specs() {
            if s.defer_loading {
                deferred.push(s.name);
            } else {
                eager.push(s.name);
            }
        }
        eager.sort();
        deferred.sort();
        (eager, deferred)
    }

    /// Attaches an interactive `UserQuestionBroker` for the deferred `AskUser` tool.
    pub async fn set_question_broker(&self, broker: Arc<dyn UserQuestionBroker>) {
        *self.question_broker.lock().await = Some(broker);
    }

    /// Pushes an external structured event (e.g. `PermissionRequested`, `UserQuestionRequested`)
    /// into this agent's lossless ring-buffered `EventLog`.
    pub fn emit_event(&self, event: AgentEvent) {
        self.events.push(event);
    }

    /// True if a `prompt()` run is currently active (used by HTTP server to return 409 Conflict — N5).
    pub fn is_busy(&self) -> bool {
        self.run_lock.try_lock().is_err()
    }

    pub fn subscribe(&self) -> EventStream {
        self.events.subscribe()
    }

    pub fn subscribe_from(&self, after_seq: u64) -> EventStream {
        self.events.subscribe_from(after_seq)
    }

    pub fn latest_event_seq(&self) -> u64 {
        self.events.latest_seq()
    }

    pub fn event_log(&self) -> Arc<EventLog> {
        Arc::clone(&self.events)
    }

    /// Queues a mid-turn steering instruction delivered after the next tool-result batch.
    pub async fn steer(&self, message: impl Into<String>) {
        let msg = message.into();
        if !msg.trim().is_empty() {
            self.pending_steers.lock().await.push(msg);
            self.steer_notify.notify_waiters();
        }
    }

    /// Queues a post-turn follow-up prompt delivered ONLY after the current turn settles
    /// (unlike `steer()`, which interrupts mid-turn right after the next tool batch).
    pub async fn follow_up(&self, message: impl Into<String>) {
        let msg = message.into();
        if !msg.trim().is_empty() {
            self.pending_follow_ups.lock().await.push_back(msg);
        }
    }

    /// Switches the active model & provider on this live session (used by ACP `session/set_config_option`).
    pub async fn set_model(
        &self,
        model: String,
        provider: Arc<dyn Provider>,
        context_window: Option<u64>,
        max_output_tokens: Option<u32>,
    ) -> Result<()> {
        let Ok(_guard) = self.run_lock.try_lock() else {
            return Err(CarnotError::Session(
                "cannot switch model while a prompt is running; cancel first".into(),
            ));
        };
        let mut rt = self.runtime.lock().await;
        rt.model = model;
        rt.provider = provider;
        if let Some(cw) = context_window {
            rt.context_window = cw;
        }
        if let Some(mo) = max_output_tokens {
            rt.max_output_tokens = mo;
        }
        drop(rt);
        *self.last_prefix_state.lock().await = None;
        Ok(())
    }

    /// Updates the reasoning effort for subsequent turns on this session.
    pub async fn set_effort(&self, effort: ReasoningEffort) {
        self.runtime.lock().await.effort = effort;
    }

    /// Updates reasoning effort ONLY when the active provider/model preserves its prompt cache
    /// across effort changes. Returns `true` if applied, `false` otherwise.
    pub async fn set_scoped_effort(&self, effort: ReasoningEffort) -> bool {
        let mut rt = self.runtime.lock().await;
        if !rt.provider.effort_preserves_cache(&rt.model) {
            return false;
        }
        rt.effort = effort;
        true
    }

    pub async fn effort_preserves_cache(&self) -> bool {
        let rt = self.runtime.lock().await;
        rt.provider.effort_preserves_cache(&rt.model)
    }

    pub async fn current_provider_id(&self) -> String {
        self.runtime.lock().await.provider.name().to_string()
    }

    pub async fn current_model(&self) -> String {
        self.runtime.lock().await.model.clone()
    }

    pub async fn current_effort(&self) -> ReasoningEffort {
        self.runtime.lock().await.effort
    }

    pub async fn current_context_window(&self) -> u64 {
        self.runtime.lock().await.context_window
    }

    pub async fn current_context_tokens(&self) -> u64 {
        self.session.lock().await.effective_context_tokens()
    }

    /// Returns `(context_tokens, context_breakdown)` atomically from a single session snapshot.
    /// `context_breakdown` is `None` before the first provider response or right after compaction /
    /// rewind / cold prune until the next provider usage measurement arrives.
    pub async fn current_context_with_breakdown(&self) -> (u64, Option<ContextBreakdown>) {
        let supports_native_ts = self
            .runtime
            .lock()
            .await
            .provider
            .supports_native_tool_search();
        let wire_tools =
            tool_search::effective_tools_for_provider(&self.tools.specs(), supports_native_ts);
        let session = self.session.lock().await;
        let total = session.effective_context_tokens();
        if !session.has_provider_usage_anchor() {
            return (total, None);
        }
        let proj = session.project_context();
        let raw =
            estimate_context_breakdown(&proj.static_system_prompt, &wire_tools, &proj.messages);
        (total, Some(raw.scaled_to(total)))
    }

    /// Estimated per-category context breakdown scaled to the current provider-grounded total,
    /// or `None` when no provider measurement anchors the active history (e.g. right after compaction).
    pub async fn context_breakdown(&self) -> Option<ContextBreakdown> {
        self.current_context_with_breakdown().await.1
    }

    pub async fn pending_queue_len(&self) -> usize {
        self.pending_steers.lock().await.len() + self.pending_follow_ups.lock().await.len()
    }

    pub async fn session_file(&self) -> Option<std::path::PathBuf> {
        self.session.lock().await.path.clone()
    }

    pub async fn session_entries(&self) -> Vec<carnot_session::SessionEntry> {
        self.session.lock().await.entries().to_vec()
    }

    pub async fn append_custom_entry(
        &self,
        kind: impl Into<String>,
        data: serde_json::Value,
    ) -> Result<String> {
        self.session.lock().await.append_custom(kind, data)
    }

    async fn effective_config_and_provider(&self) -> (CarnotConfig, Arc<dyn Provider>) {
        let rt = self.runtime.lock().await;
        let mut cfg = self.config.clone();
        cfg.model = rt.model.clone();
        cfg.effort = rt.effort;
        cfg.context_window = rt.context_window;
        cfg.max_output_tokens = rt.max_output_tokens;
        (cfg, Arc::clone(&rt.provider))
    }

    pub async fn session_id(&self) -> String {
        self.session.lock().await.session_id.clone()
    }

    pub async fn prefix_hash(&self) -> String {
        self.session.lock().await.prefix_hash.clone()
    }

    pub async fn metrics(&self) -> ThermodynamicMetrics {
        self.metrics.lock().await.clone()
    }

    pub async fn abort(&self) {
        self.cancel_current.lock().await.cancel();
        self.tools.job_registry.kill_all();
    }

    /// Atomically rolls back the most recent `Edit`/`Write` via its verified BLAKE3 CAS snapshot.
    pub fn undo_last_edit(&self) -> Result<Option<std::path::PathBuf>> {
        self.tools.undo_last_edit()
    }

    /// Manually compacts the active session using cache-warm semantic synthesis (with deterministic
    /// fallback), preserving the verbatim user objective + recent tail + verified workspace state.
    pub async fn compact_now(&self) -> Result<(u64, u64, bool)> {
        self.compact_now_with_focus(None).await
    }

    /// Manually compacts the active session with an optional caller focus instruction
    /// (e.g. `/compact preserve SQL migration schema and failing test output`).
    pub async fn compact_now_with_focus(&self, focus: Option<&str>) -> Result<(u64, u64, bool)> {
        let _run_guard = self.run_lock.lock().await;
        let (eff_cfg, eff_provider) = self.effective_config_and_provider().await;
        let seg = SegmentTracker::default();
        let res = execute_compaction(
            CompactionMode::Manual,
            &eff_cfg,
            &eff_provider,
            &self.tools,
            &self.modules,
            &self.session,
            &self.metrics,
            &self.total_cost_micros,
            &self.events,
            &self.raw_user_goal,
            &self.user_instructions,
            &seg,
            focus,
            CancellationToken::new(),
        )
        .await;
        if res.is_ok() {
            *self.last_prefix_state.lock().await = None;
        }
        res
    }

    async fn run_before_gates(
        &self,
        call_id: &str,
        tool_name: &str,
        raw_input: &serde_json::Value,
    ) -> Result<(serde_json::Value, Option<String>)> {
        let mut effective_input = raw_input.clone();
        for m in &self.modules {
            match m
                .before_tool_call_with_id(call_id, tool_name, &effective_input)
                .await?
            {
                ToolGateDecision::Allow => {}
                ToolGateDecision::RewriteInput(new_in) => effective_input = new_in,
                ToolGateDecision::Deny { reason } => {
                    return Ok((effective_input, Some(reason)));
                }
            }
        }
        let broker = self.permission_broker.lock().await.clone();
        if let Some(b) = broker {
            match b
                .request_permission(call_id, tool_name, &effective_input)
                .await?
            {
                ToolGateDecision::Allow => {}
                ToolGateDecision::RewriteInput(new_in) => effective_input = new_in,
                ToolGateDecision::Deny { reason } => {
                    return Ok((effective_input, Some(reason)));
                }
            }
        }
        Ok((effective_input, None))
    }

    fn discard_eager_launches(&self, eager_launches: HashMap<String, EagerLaunch>) {
        // Abort speculative tasks AND invalidate their ReadTracker entries so next Read never returns
        // a false "unchanged" stub for outputs the model never saw (fixes Opus N3!).
        for (_, launch) in eager_launches {
            if let Some(jh) = launch.handle {
                jh.abort();
            }
            for p in &launch.effects.reads {
                self.tools.invalidate_read_path(p);
            }
        }
    }

    /// Runs a user text prompt through the full multi-turn agent loop until completion or halt.
    /// Serialized per agent (N5) and guaranteed to always emit a terminal `RunSettled` or `Error` event (N8).
    pub async fn prompt(&self, user_prompt: impl Into<String>) -> Result<RunOutcome> {
        self.prompt_with_cancel(user_prompt, CancellationToken::new())
            .await
    }

    /// Runs a user text prompt bound to an external `CancellationToken` (e.g. a worker lifecycle
    /// or parent tool cancellation token) so cancellation fired right before the run starts is
    /// never lost.
    pub async fn prompt_with_cancel(
        &self,
        user_prompt: impl Into<String>,
        external_cancel: CancellationToken,
    ) -> Result<RunOutcome> {
        self.prompt_blocks_with_cancel(
            vec![ContentBlock::Text {
                text: user_prompt.into(),
            }],
            external_cancel,
        )
        .await
    }

    /// Runs a multimodal user prompt (text + images) through the full multi-turn agent loop.
    pub async fn prompt_blocks(&self, user_blocks: Vec<ContentBlock>) -> Result<RunOutcome> {
        self.prompt_blocks_with_cancel(user_blocks, CancellationToken::new())
            .await
    }

    /// Continues an interrupted session turn without appending a new user message.
    /// Returns `Ok(None)` if the session is empty or already settled on an assistant reply.
    async fn emit_terminal_error_events(&self, err: &CarnotError) {
        let metrics = self.metrics.lock().await.clone();
        if matches!(err, CarnotError::Cancelled) {
            self.events.push(AgentEvent::TurnEnd {
                turn: metrics.turns,
                stop_reason: StopReason::Aborted,
            });
        } else {
            self.events.push(AgentEvent::Error {
                message: err.to_string(),
            });
        }
        self.events.push(AgentEvent::RunSettled {
            final_text: String::new(),
            metrics,
        });
    }

    /// Resumes an interrupted session without injecting a new user message when the last persisted
    /// message is not a settled assistant reply. Returns `Ok(None)` if the session is already settled.
    pub async fn continue_run(&self) -> Result<Option<RunOutcome>> {
        let _run_guard = self.run_lock.lock().await;
        let can_continue = {
            let mut s = self.session.lock().await;
            let _ = s.repair_dangling_tool_uses()?;
            let msgs = s.active_messages();
            match msgs.last() {
                None => false,
                Some(last) => last.role != Role::Assistant || last.has_tool_use(),
            }
        };
        if !can_continue {
            return Ok(None);
        }
        match self
            .prompt_inner_opts(Vec::new(), CancellationToken::new(), true)
            .await
        {
            Ok(outcome) => Ok(Some(outcome)),
            Err(e) => {
                self.emit_terminal_error_events(&e).await;
                Err(e)
            }
        }
    }

    /// Runs a multimodal user prompt bound to an external `CancellationToken`.
    pub async fn prompt_blocks_with_cancel(
        &self,
        user_blocks: Vec<ContentBlock>,
        external_cancel: CancellationToken,
    ) -> Result<RunOutcome> {
        let _run_guard = self.run_lock.lock().await;
        match self
            .prompt_inner_opts(user_blocks, external_cancel, false)
            .await
        {
            Ok(outcome) => Ok(outcome),
            Err(e) => {
                self.emit_terminal_error_events(&e).await;
                Err(e)
            }
        }
    }

    async fn prompt_inner_opts(
        &self,
        user_blocks: Vec<ContentBlock>,
        external_cancel: CancellationToken,
        continue_existing: bool,
    ) -> Result<RunOutcome> {
        let cancel = external_cancel.child_token();
        *self.cancel_current.lock().await = cancel.clone();
        if cancel.is_cancelled() {
            return Err(CarnotError::Cancelled);
        }

        // Extract text portion for Tier-0 sacred objective anchor (preserving any image blocks intact!)
        let mut text_parts: Vec<String> = Vec::new();
        let mut extra_blocks: Vec<ContentBlock> = Vec::new();
        for b in user_blocks {
            match b {
                ContentBlock::Text { text } => {
                    if !text.is_empty() {
                        text_parts.push(text);
                    }
                }
                other => extra_blocks.push(other),
            }
        }
        let mut initial_text = text_parts.join("\n");
        let raw_prompt_for_hooks = initial_text.clone();

        // Capture full verbatim user objective BEFORE Turn-1 `[workspace-facts]`/`AGENTS.md` injection
        // (and track any subsequent follow-up prompts so compaction NEVER drops user requirements!)
        if !initial_text.trim().is_empty() {
            let mut goal = self.raw_user_goal.lock().await;
            if goal.is_none() {
                *goal = Some(clip_anchor_text(&initial_text, 24_000));
            } else {
                drop(goal);
                let mut followups = self.user_instructions.lock().await;
                followups.push(clip_anchor_text(&initial_text, 4_000));
                if followups.len() > 16 {
                    followups.remove(0);
                }
            }
        }

        // Inject execution-verified Turn-1 context (NEVER into static system prompt!)
        {
            let mut injected = self.turn1_injected.lock().await;
            if !*injected {
                *injected = true;
                let mut turn1_blocks = Vec::new();
                for m in &self.modules {
                    if let Ok(Some(ctx)) = m.on_session_start(&self.config).await {
                        turn1_blocks.push(ctx);
                    }
                }
                let supports_native_ts = self
                    .runtime
                    .lock()
                    .await
                    .provider
                    .supports_native_tool_search();
                if !tool_search::should_promote_deferred_directly(
                    &self.tools.specs(),
                    supports_native_ts,
                ) {
                    let deferred_descs = self
                        .deferred_server_descriptions
                        .lock()
                        .map(|g| g.clone())
                        .unwrap_or_default();
                    if let Some(dir) = tool_search::format_deferred_tools_directory(
                        &self.tools.deferred_tools(),
                        &deferred_descs,
                    ) {
                        turn1_blocks.push(dir);
                    }
                }
                if !turn1_blocks.is_empty() {
                    if initial_text.is_empty() {
                        initial_text = turn1_blocks.join("\n\n");
                    } else {
                        initial_text = format!("{}\n\n{initial_text}", turn1_blocks.join("\n\n"));
                    }
                }
            }
        }

        let mut initial_content: Vec<ContentBlock> = Vec::with_capacity(1 + extra_blocks.len());
        if !initial_text.is_empty() || extra_blocks.is_empty() {
            initial_content.push(ContentBlock::Text { text: initial_text });
        }
        initial_content.extend(extra_blocks);

        // Cold-cache opportunistic tool-result pruning (inspired by Hermes):
        // ONLY when the provider's rolling prompt cache is ALREADY expired (zero cache breaks,
        // zero LLM calls), replace oversized stale tool results outside the last 4 messages
        // with compact handle-preserving stubs before the cold cache rewrite!
        if self.config.modules.compaction {
            let ttl_ms = match self.config.cache_ttl {
                CacheTtl::Long1h => 3_610_000u64,
                CacheTtl::Short5m | CacheTtl::Hybrid => 310_000u64,
            };
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            let reclaimed = {
                let mut s = self.session.lock().await;
                let cache_expired = s
                    .last_activity_unix_ms()
                    .map(|last| now_ms.saturating_sub(last) >= ttl_ms)
                    .unwrap_or(false);
                if cache_expired && s.active_messages().len() >= 8 {
                    s.prune_cold_tool_results(4, 2_000, 24_000)?
                } else {
                    0
                }
            };
            if reclaimed > 0 {
                self.tools.reset_read_tracker();
                *self.last_prefix_state.lock().await = None;
                self.metrics.lock().await.heat_shed_tokens += (reclaimed as u64) / 4;
            }
        }

        // Collect per-prompt module injections (`before_prompt`: durable vs ephemeral)
        let mut ephemeral_notes: Vec<String> = Vec::new();
        let mut durable_notes: Vec<String> = Vec::new();
        if !continue_existing {
            for m in &self.modules {
                if let Ok(injs) = m.before_prompt(&raw_prompt_for_hooks).await {
                    for inj in injs {
                        let t = inj.text.trim();
                        if t.is_empty() {
                            continue;
                        }
                        if inj.ephemeral {
                            ephemeral_notes.push(t.to_string());
                        } else {
                            durable_notes.push(t.to_string());
                        }
                    }
                }
            }
        }

        // Record user turn and mark Checkpoint C at the user boundary
        if !continue_existing {
            let mut s = self.session.lock().await;
            if s.active_messages().is_empty() && !durable_notes.is_empty() {
                let prefix = durable_notes.join("\n\n");
                let mut combined = Vec::with_capacity(1 + initial_content.len());
                combined.push(ContentBlock::Text { text: prefix });
                combined.extend(initial_content);
                s.push_message(Message {
                    role: Role::User,
                    content: combined,
                })?;
            } else {
                for note in durable_notes {
                    s.push_message(Message::system_note(note))?;
                }
                s.push_message(Message {
                    role: Role::User,
                    content: initial_content,
                })?;
            }
            s.mark_checkpoint("user_turn")?;
        }

        let mut seg = SegmentTracker::default();
        let mut final_text = String::new();
        let mut halted_reason: Option<String> = None;
        let mut settled_normally = false;
        let mut consecutive_emergency_compactions: u32 = 0;
        let mut last_compaction_turn: Option<u64> = None;
        let mut consecutive_stream_retries: u32 = 0;
        let mut consecutive_max_tokens: u8 = 0;
        let mut consecutive_pause_turns: u8 = 0;
        let mut idle_turn_streak: u32 = 0;
        let mut empty_reply_retried = false;

        for _step in 0..self.config.max_turns {
            if cancel.is_cancelled() {
                return Err(CarnotError::Cancelled);
            }

            let (eff_cfg, active_provider) = self.effective_config_and_provider().await;

            // Proactive cache-warm thermodynamic compaction (provider-grounded token count + cooldown)
            if eff_cfg.modules.compaction {
                let (pre_est, msg_count) = {
                    let s = self.session.lock().await;
                    (s.effective_context_tokens(), s.active_messages().len())
                };
                let current_turns = self.metrics.lock().await.turns;
                let cooldown_ok = last_compaction_turn
                    .map(|t| current_turns.saturating_sub(t) >= eff_cfg.compaction_cooldown_turns)
                    .unwrap_or(true);
                let trigger = effective_proactive_trigger(&eff_cfg);
                if trigger > 0 && msg_count >= 10 && cooldown_ok && pre_est >= trigger {
                    last_compaction_turn = Some(current_turns);
                    self.compact_with_mode(CompactionMode::Soft, &seg, cancel.clone())
                        .await?;
                    seg.reset();
                }
            }

            // Optional per-turn effort selector (applied ONLY when effort preserves prompt cache)
            let mut turn_effort = eff_cfg.effort;
            if active_provider.effort_preserves_cache(&eff_cfg.model) {
                for m in &self.modules {
                    if let Ok(Some(selected)) = m.select_effort().await {
                        turn_effort = selected;
                    }
                }
            }

            let (req, turn_num, ctx_est) = {
                let s = self.session.lock().await;
                let ctx_est = s.effective_context_tokens();
                let proj = s.project_context();
                let mut m = self.metrics.lock().await;
                m.turns += 1;
                m.requests += 1;
                let turn_num = m.turns;
                let wire_tools = tool_search::effective_tools_for_provider(
                    &self.tools.specs(),
                    active_provider.supports_native_tool_search(),
                );
                let req = ProviderRequest {
                    model: eff_cfg.model.clone(),
                    static_system_prompt: proj.static_system_prompt,
                    tools: wire_tools,
                    prefix_hash: proj.prefix_hash,
                    session_id: proj.cache_key_id,
                    messages: proj.messages,
                    checkpoint_index: proj.checkpoint_index,
                    cache_ttl: eff_cfg.cache_ttl,
                    effort: turn_effort,
                    max_output_tokens: eff_cfg.max_output_tokens,
                    skip_tail_cache_write: false,
                    native_web: eff_cfg.native_web.clone(),
                    ephemeral_tail_notes: ephemeral_notes.clone(),
                };
                (req, turn_num, ctx_est)
            };

            // Live runtime prefix-freeze verification (< 5 µs BLAKE3) over PERSISTED messages.
            // Ephemeral notes travel separately in `req.ephemeral_tail_notes` AFTER all provider
            // cache breakpoints so they never bust prefix cache on Turn 1 or subsequent turns!
            {
                let prov_name = active_provider.name().to_string();
                let static_hash = compute_wire_static_hash(
                    &req.static_system_prompt,
                    &req.tools,
                    &req.native_web,
                );
                let persisted_slice = &req.messages;
                let mut prev_guard = self.last_prefix_state.lock().await;
                let mut detected_break: Option<(String, String, String)> = None;
                if let Some(ref prev) = *prev_guard
                    && prev.provider == prov_name
                    && prev.model == req.model
                {
                    if prev.static_hash != static_hash {
                        detected_break = Some((
                            "static prefix (system/tools/native_web) mutated mid-session".into(),
                            prev.static_hash.clone(),
                            static_hash.clone(),
                        ));
                    } else if persisted_slice.len() < prev.msg_count {
                        detected_break = Some((
                            format!(
                                "conversation history shrank unexpectedly ({} -> {} messages) without compaction",
                                prev.msg_count,
                                persisted_slice.len()
                            ),
                            prev.msg_prefix_hash.clone(),
                            compute_messages_prefix_hash(persisted_slice),
                        ));
                    } else {
                        let reused_hash =
                            compute_messages_prefix_hash(&persisted_slice[..prev.msg_count]);
                        if reused_hash != prev.msg_prefix_hash {
                            detected_break = Some((
                                format!(
                                    "reused conversation prefix (first {} messages) mutated mid-session",
                                    prev.msg_count
                                ),
                                prev.msg_prefix_hash.clone(),
                                reused_hash,
                            ));
                        }
                    }
                } else {
                    self.events.push(AgentEvent::PrefixAudit {
                        prefix_hash: static_hash.clone(),
                        stable: true,
                        note: None,
                    });
                }
                if let Some((reason, expected_hash, actual_hash)) = detected_break {
                    self.metrics.lock().await.cache_breaks += 1;
                    tracing::warn!(
                        target: "carnot::cache",
                        turn = turn_num,
                        reason = %reason,
                        expected = %expected_hash,
                        actual = %actual_hash,
                        "unexpected_cache_break_detected"
                    );
                    self.events.push(AgentEvent::CacheBreakDetected {
                        turn: turn_num,
                        reason,
                        expected_hash,
                        actual_hash,
                    });
                }
                *prev_guard = Some(PrefixFingerprint {
                    provider: prov_name,
                    model: req.model.clone(),
                    static_hash,
                    msg_count: persisted_slice.len(),
                    msg_prefix_hash: compute_messages_prefix_hash(persisted_slice),
                });
            }

            self.events.push(AgentEvent::TurnStart {
                turn: turn_num,
                context_tokens_est: ctx_est,
            });

            // Stream provider response, starting gate-verified eager read-only tools mid-stream!
            let mut rx = match active_provider.stream(req, cancel.clone()).await {
                Ok(r) => r,
                Err(CarnotError::ContextOverflow(_))
                    if self.config.modules.compaction && consecutive_emergency_compactions < 1 =>
                {
                    consecutive_emergency_compactions += 1;
                    last_compaction_turn = Some(turn_num);
                    self.compact_with_mode(CompactionMode::EmergencyOverflow, &seg, cancel.clone())
                        .await?;
                    seg.reset();
                    continue;
                }
                Err(e) => return Err(e),
            };

            let mut eager_launches: HashMap<String, EagerLaunch> = HashMap::new();
            let mut seen_stream_effects: Vec<ToolEffects> = Vec::new();
            let mut assistant_msg = Message::assistant_text("");
            let mut usage = Usage::default();
            let mut stop_reason = StopReason::EndTurn;
            let mut rate_limits = Vec::new();
            let mut stream_err: Option<(String, bool)> = None;
            let mut stream_overflow: Option<String> = None;

            while let Some(ev) = rx.recv().await {
                match ev {
                    StreamEvent::TextDelta(delta) => {
                        self.events.push(AgentEvent::TextDelta { delta });
                    }
                    StreamEvent::ThinkingDelta(delta) => {
                        self.events.push(AgentEvent::ThinkingDelta { delta });
                    }
                    StreamEvent::ToolCallReady {
                        id,
                        name,
                        input,
                        invalid_json,
                    } => {
                        let tool_opt = self.tools.get(&name);
                        let raw_eff = tool_opt
                            .as_ref()
                            .map(|t| t.effects(&input))
                            .unwrap_or_else(ToolEffects::serial);

                        let can_eager = self.config.modules.eager_read
                            && invalid_json.is_none()
                            && raw_eff.is_eager_safe()
                            && seen_stream_effects
                                .iter()
                                .all(|e| !e.conflicts_with(&raw_eff));

                        if can_eager {
                            if let Some(tool) = tool_opt {
                                // Run `before_tool_call` gate BEFORE launching eager execution!
                                let (eff_input, denied) =
                                    match self.run_before_gates(&id, &name, &input).await {
                                        Ok(pair) => pair,
                                        Err(e) => (input.clone(), Some(e.to_string())),
                                    };
                                let eff = tool.effects(&eff_input);
                                let still_safe = denied.is_none()
                                    && eff.is_eager_safe()
                                    && seen_stream_effects.iter().all(|e| !e.conflicts_with(&eff));
                                let handle = if still_safe {
                                    self.events.push(AgentEvent::ToolStart {
                                        call_id: id.clone(),
                                        tool: name.clone(),
                                        eager: true,
                                        input: eff_input.clone(),
                                    });
                                    self.metrics.lock().await.eager_tools_started += 1;
                                    let c = cancel.clone();
                                    let cid_exec = id.clone();
                                    let inp_exec = eff_input.clone();
                                    Some(tokio::spawn(async move {
                                        let t0 = Instant::now();
                                        let out = tool
                                            .execute_with_id(&cid_exec, inp_exec, c)
                                            .await
                                            .unwrap_or_else(|e| {
                                                ToolOutput::err_text(e.to_string())
                                            });
                                        (out, t0.elapsed().as_millis() as u64)
                                    }))
                                } else {
                                    None
                                };
                                seen_stream_effects.push(eff.clone());
                                eager_launches.insert(
                                    id,
                                    EagerLaunch {
                                        effects: eff,
                                        effective_input: eff_input,
                                        denied,
                                        handle,
                                    },
                                );
                            }
                        } else {
                            seen_stream_effects.push(raw_eff);
                        }
                    }
                    StreamEvent::Done {
                        message,
                        usage: u,
                        stop_reason: sr,
                        rate_limits: rl,
                    } => {
                        assistant_msg = message;
                        usage = u;
                        stop_reason = sr;
                        rate_limits = rl;
                    }
                    StreamEvent::Error { message, retryable } => {
                        stream_err = Some((message, retryable));
                    }
                    StreamEvent::ContextOverflow(message) => {
                        stream_overflow = Some(message);
                    }
                }
            }

            if let Some(ov_msg) = stream_overflow {
                self.discard_eager_launches(eager_launches);
                if self.config.modules.compaction && consecutive_emergency_compactions < 1 {
                    consecutive_emergency_compactions += 1;
                    last_compaction_turn = Some(turn_num);
                    self.compact_with_mode(CompactionMode::EmergencyOverflow, &seg, cancel.clone())
                        .await?;
                    seg.reset();
                    continue;
                }
                return Err(CarnotError::ContextOverflow(ov_msg));
            }

            if let Some((err, retryable)) = stream_err {
                self.discard_eager_launches(eager_launches);
                if retryable && consecutive_stream_retries < 5 {
                    consecutive_stream_retries += 1;
                    let delay = std::time::Duration::from_secs(
                        (1u64 << (consecutive_stream_retries - 1)).min(16),
                    );
                    tracing::warn!(
                        provider = %active_provider.name(),
                        attempt = consecutive_stream_retries,
                        delay_ms = delay.as_millis() as u64,
                        error = %err,
                        "retrying transient mid-stream provider error"
                    );
                    tokio::select! {
                        _ = cancel.cancelled() => return Err(CarnotError::Cancelled),
                        _ = tokio::time::sleep(delay) => {}
                    }
                    continue;
                }
                return Err(CarnotError::Provider {
                    provider: active_provider.name().into(),
                    message: err,
                    retryable,
                    retry_after_ms: None,
                });
            }
            consecutive_stream_retries = 0;

            // Context window overflow reported as stop_reason -> abort eager & compact at most once (N10)
            if stop_reason == StopReason::ContextWindowExceeded
                && self.config.modules.compaction
                && consecutive_emergency_compactions < 1
            {
                consecutive_emergency_compactions += 1;
                last_compaction_turn = Some(turn_num);
                self.discard_eager_launches(eager_launches);
                self.compact_with_mode(CompactionMode::EmergencyOverflow, &seg, cancel.clone())
                    .await?;
                seg.reset();
                continue;
            }
            consecutive_emergency_compactions = 0;

            // Record usage & live thermodynamic AUC + structured cache telemetry (Opus #1!)
            {
                let mut m = self.metrics.lock().await;
                m.context_auc_tokens += usage.total_context_tokens();
                if usage.cache_read_tokens > 0 {
                    m.prefix_cache_hits += 1;
                } else {
                    m.prefix_cache_misses += 1;
                }
                m.total_usage.add_assign(&usage);
                add_cost_micros(&self.total_cost_micros, usage.cost_usd);
                tracing::info!(
                    target: "carnot::cache",
                    turn = turn_num,
                    model = %eff_cfg.model,
                    cache_read = usage.cache_read_tokens,
                    cache_write_5m = usage.cache_write_5m_tokens,
                    cache_write_1h = usage.cache_write_1h_tokens,
                    uncached_input = usage.input_tokens,
                    output = usage.output_tokens,
                    cost_usd = usage.cost_usd,
                    cum_hit_rate = m.cache_hit_rate(),
                    "turn_usage"
                );
            }
            {
                let mut s = self.session.lock().await;
                s.record_usage(turn_num, usage.clone())?;
                s.push_message(assistant_msg.clone())?;
            }
            self.events.push(AgentEvent::UsageRecorded {
                turn: turn_num,
                usage,
                rate_limits,
            });

            let last_text = assistant_msg.text_concat();
            if !last_text.is_empty() {
                final_text = last_text.clone();
            }

            // Never execute partial/truncated tool calls on MaxTokens or Refusal — AND pair any
            // truncated `tool_use` blocks with `is_error` `tool_result`s so the session stays valid (N2)!
            if matches!(stop_reason, StopReason::MaxTokens | StopReason::Refusal) {
                self.discard_eager_launches(eager_launches);
                let mut recovery_blocks: Vec<ContentBlock> = assistant_msg
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolUse { id, name, .. } => Some(ContentBlock::ToolResult {
                            tool_use_id: id.clone(),
                            content: vec![ToolResultBlock::text(format!(
                                "Tool `{name}` was not executed because assistant response stopped with {stop_reason:?}."
                            ))],
                            is_error: true,
                        }),
                        _ => None,
                    })
                    .collect();

                // Enforce cost guardrails BEFORE MaxTokens continuation so caps can never be bypassed
                let spent_now = self.metrics.lock().await.total_usage.cost_usd;
                let cost_exceeded = self
                    .config
                    .max_cost_usd
                    .is_some_and(|limit| spent_now >= limit)
                    || (self.config.subagent_depth > 0
                        && spent_now >= self.config.subagent_max_cost_usd);

                if stop_reason == StopReason::MaxTokens
                    && !cost_exceeded
                    && !is_repetition_dominated(&last_text)
                    && consecutive_max_tokens < self.config.max_tokens_continuations
                {
                    consecutive_max_tokens += 1;
                    recovery_blocks.push(ContentBlock::Text {
                        text: "Your previous response hit max_tokens and was truncated. Continue from where you left off (split large writes/edits into smaller calls if needed).".into(),
                    });
                    self.session.lock().await.push_message(Message {
                        role: Role::User,
                        content: recovery_blocks,
                    })?;
                    self.events.push(AgentEvent::TurnEnd {
                        turn: turn_num,
                        stop_reason,
                    });
                    continue;
                }

                if !recovery_blocks.is_empty() {
                    self.session.lock().await.push_message(Message {
                        role: Role::User,
                        content: recovery_blocks,
                    })?;
                }
                self.events.push(AgentEvent::TurnEnd {
                    turn: turn_num,
                    stop_reason,
                });
                halted_reason = Some(format!("stopped: {stop_reason:?}"));
                break;
            }
            consecutive_max_tokens = 0;

            // Extract tool calls (preserving invalid_json flag!)
            let tool_calls: Vec<(String, String, serde_json::Value, Option<String>)> =
                assistant_msg
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolUse {
                            id,
                            name,
                            input,
                            invalid_json,
                            ..
                        } => Some((
                            id.clone(),
                            name.clone(),
                            input.clone(),
                            invalid_json.clone(),
                        )),
                        _ => None,
                    })
                    .collect();

            let had_tools = !tool_calls.is_empty();
            if had_tools {
                let mut outputs: Vec<Option<(serde_json::Value, ToolOutput, u64)>> =
                    vec![None; tool_calls.len()];
                let mut pending_calls: Vec<ScheduledCall> = Vec::new();
                let mut active_eager: Vec<ActiveEager> = Vec::new();

                // Phase 1: Gate non-eager calls, reject malformed JSON, classify into eager vs wave queue
                for (idx, (call_id, tool_name, raw_input, invalid_json)) in
                    tool_calls.iter().enumerate()
                {
                    if let Some(bad_json) = invalid_json {
                        if let Some(launch) = eager_launches.remove(call_id)
                            && let Some(jh) = launch.handle
                        {
                            jh.abort();
                        }
                        outputs[idx] = Some((
                            raw_input.clone(),
                            ToolOutput::err_text(format!(
                                "Malformed JSON arguments for `{tool_name}`: {bad_json}. Please retry with valid JSON."
                            )),
                            0,
                        ));
                        continue;
                    }

                    if let Some(launch) = eager_launches.remove(call_id) {
                        if let Some(reason) = launch.denied {
                            if let Some(jh) = launch.handle {
                                jh.abort();
                            }
                            outputs[idx] = Some((
                                launch.effective_input,
                                ToolOutput::err_text(format!("Denied: {reason}")),
                                0,
                            ));
                        } else if let Some(jh) = launch.handle {
                            active_eager.push(ActiveEager {
                                orig_idx: idx,
                                effects: launch.effects,
                                effective_input: launch.effective_input,
                                handle: jh,
                            });
                        } else {
                            pending_calls.push(ScheduledCall {
                                orig_idx: idx,
                                call_id: call_id.clone(),
                                tool_name: tool_name.clone(),
                                effective_input: launch.effective_input,
                                effects: launch.effects,
                            });
                        }
                        continue;
                    }

                    // Gate the real tool directly (for `CallDeferredTool`, gate ONLY the inner
                    // target tool so policies keyed on tool names aren't double-gated or blocked).
                    let (effective_input, denied) = if tool_name == "CallDeferredTool" {
                        let mut wrapper_input = raw_input.clone();
                        if let Some(inner_name) = wrapper_input
                            .get("name")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                        {
                            let inner_args = wrapper_input
                                .get("arguments")
                                .cloned()
                                .unwrap_or_else(|| serde_json::json!({}));
                            match self
                                .run_before_gates(call_id, &inner_name, &inner_args)
                                .await
                            {
                                Ok((rewritten, inner_denied)) => {
                                    if inner_denied.is_none()
                                        && let Some(obj) = wrapper_input.as_object_mut()
                                    {
                                        obj.insert("arguments".into(), rewritten);
                                    }
                                    (wrapper_input, inner_denied)
                                }
                                Err(e) => (wrapper_input, Some(format!("inner gate error: {e}"))),
                            }
                        } else {
                            (
                                wrapper_input,
                                Some("CallDeferredTool missing target tool `name`".into()),
                            )
                        }
                    } else {
                        match self.run_before_gates(call_id, tool_name, raw_input).await {
                            Ok(pair) => pair,
                            Err(e) => (raw_input.clone(), Some(format!("gate error: {e}"))),
                        }
                    };
                    if let Some(reason) = denied {
                        outputs[idx] = Some((
                            effective_input,
                            ToolOutput::err_text(format!("Denied: {reason}")),
                            0,
                        ));
                        continue;
                    }
                    let effects = self
                        .tools
                        .get(tool_name)
                        .map(|t| t.effects(&effective_input))
                        .unwrap_or_else(ToolEffects::serial);
                    pending_calls.push(ScheduledCall {
                        orig_idx: idx,
                        call_id: call_id.clone(),
                        tool_name: tool_name.clone(),
                        effective_input,
                        effects,
                    });
                }

                // Abort any stray eager handles not present in final assistant message & invalidate tracker (N3)
                self.discard_eager_launches(eager_launches);

                // Phase 2 (Opus I2): Partition pending calls into conflict-free parallel waves!
                let waves = partition_conflict_free_waves(pending_calls);

                // Execute each wave concurrently (awaiting any conflicting eager reads first!)
                for wave in waves {
                    // Drain any still-running eager tasks whose effects conflict with this wave
                    let mut remaining_eager = Vec::with_capacity(active_eager.len());
                    for ae in active_eager {
                        if wave.iter().any(|c| c.effects.conflicts_with(&ae.effects)) {
                            let (out, dur) = ae.handle.await.unwrap_or_else(|e| {
                                (ToolOutput::err_text(format!("eager task panicked: {e}")), 0)
                            });
                            outputs[ae.orig_idx] = Some((ae.effective_input, out, dur));
                        } else {
                            remaining_eager.push(ae);
                        }
                    }
                    active_eager = remaining_eager;

                    let write_count = wave.iter().filter(|c| !c.effects.writes.is_empty()).count();
                    if write_count >= 2 {
                        self.metrics.lock().await.parallel_edits_executed += write_count as u64;
                    }

                    let mut wave_handles = Vec::with_capacity(wave.len());
                    for call in wave {
                        self.events.push(AgentEvent::ToolStart {
                            call_id: call.call_id.clone(),
                            tool: call.tool_name.clone(),
                            eager: false,
                            input: call.effective_input.clone(),
                        });
                        let tool_opt = self.tools.get(&call.tool_name);
                        let c = cancel.clone();
                        let cid = call.call_id.clone();
                        let inp = call.effective_input.clone();
                        let tname = call.tool_name.clone();
                        let jh = tokio::spawn(async move {
                            let t0 = Instant::now();
                            let out = match tool_opt {
                                Some(t) => t
                                    .execute_with_id(&cid, inp, c)
                                    .await
                                    .unwrap_or_else(|e| ToolOutput::err_text(e.to_string())),
                                None => ToolOutput::err_text(format!("Unknown tool: {tname}")),
                            };
                            (out, t0.elapsed().as_millis() as u64)
                        });
                        wave_handles.push((call.orig_idx, call.effective_input, jh));
                    }

                    for (idx, eff_inp, jh) in wave_handles {
                        let (out, dur) = jh.await.unwrap_or_else(|e| {
                            (ToolOutput::err_text(format!("tool task panicked: {e}")), 0)
                        });
                        outputs[idx] = Some((eff_inp, out, dur));
                    }
                }

                // Collect any remaining non-conflicting eager tasks
                for ae in active_eager {
                    let (out, dur) = ae.handle.await.unwrap_or_else(|e| {
                        (ToolOutput::err_text(format!("eager task panicked: {e}")), 0)
                    });
                    outputs[ae.orig_idx] = Some((ae.effective_input, out, dur));
                }

                // Phase 3: Run `after_tool_result` hooks, emit `ToolEnd`, and build ordered `ToolResult`s
                let mut result_blocks = Vec::with_capacity(tool_calls.len());
                for (idx, (call_id, tool_name, _, _)) in tool_calls.into_iter().enumerate() {
                    let (effective_input, mut out, dur_ms) =
                        outputs[idx].take().unwrap_or_else(|| {
                            (
                                serde_json::Value::Null,
                                ToolOutput::err_text("missing tool output"),
                                0,
                            )
                        });

                    for m in &self.modules {
                        let _ = m
                            .after_tool_result(&tool_name, &effective_input, &mut out)
                            .await;
                    }

                    self.record_segment_facts(&tool_name, &effective_input, &out, &mut seg)
                        .await;

                    let output_preview = clip_utf8_preview(&out.text_only(), 8_192);
                    self.events.push(AgentEvent::ToolEnd {
                        call_id: call_id.clone(),
                        tool: tool_name,
                        is_error: out.is_error,
                        duration_ms: dur_ms,
                        output_preview,
                    });

                    result_blocks.push(ContentBlock::ToolResult {
                        tool_use_id: call_id,
                        content: out.blocks,
                        is_error: out.is_error,
                    });
                }

                // Append any queued mid-turn steering text AFTER `tool_result` blocks (cache & API safe)
                let steers: Vec<String> = {
                    let mut q = self.pending_steers.lock().await;
                    std::mem::take(&mut *q)
                };
                for st in steers {
                    {
                        let mut followups = self.user_instructions.lock().await;
                        followups.push(clip_anchor_text(&st, 4_000));
                        if followups.len() > 16 {
                            followups.remove(0);
                        }
                    }
                    result_blocks.push(ContentBlock::Text {
                        text: format!("[user-steer]\n{st}"),
                    });
                }

                self.session.lock().await.push_message(Message {
                    role: Role::User,
                    content: result_blocks,
                })?;

                // If aborted mid-tool, paired `tool_result`s are now safely persisted — exit cleanly
                // without firing `Stop` / turn-boundary hooks on an aborted turn!
                if cancel.is_cancelled() {
                    self.events.push(AgentEvent::TurnEnd {
                        turn: turn_num,
                        stop_reason: StopReason::Aborted,
                    });
                    return Err(CarnotError::Cancelled);
                }
            }

            self.events.push(AgentEvent::TurnEnd {
                turn: turn_num,
                stop_reason,
            });

            // Built-in lightweight idle-turn circuit breaker (prevents runaway autonomous/gate loops).
            // Never count provider server-side `PauseTurn` (e.g. native web_search/web_fetch loops) as idle!
            let mut should_halt = false;
            if stop_reason == StopReason::PauseTurn {
                consecutive_pause_turns = consecutive_pause_turns.saturating_add(1);
                if consecutive_pause_turns > 8 {
                    halted_reason = Some(format!(
                        "server pause_turn cap reached ({consecutive_pause_turns} consecutive continuations)"
                    ));
                    should_halt = true;
                }
            } else {
                consecutive_pause_turns = 0;
                if !had_tools {
                    idle_turn_streak += 1;
                    if self.config.idle_turn_circuit_breaker > 0
                        && idle_turn_streak >= self.config.idle_turn_circuit_breaker
                    {
                        halted_reason = Some(format!(
                            "idle circuit breaker triggered ({idle_turn_streak} consecutive turns without tool calls)"
                        ));
                        should_halt = true;
                    }
                } else {
                    idle_turn_streak = 0;
                }
            }

            // Evaluate turn-boundary module hooks
            let mut boundary_resume = false;
            if !should_halt && !self.modules.is_empty() {
                let total_ctx_est = self.session.lock().await.effective_context_tokens();
                let snap = TurnBoundarySnapshot {
                    turn_index: turn_num,
                    stop_reason,
                    total_context_tokens: total_ctx_est,
                    had_edit_in_segment: seg.had_edit,
                    had_tool_call_in_turn: had_tools,
                    files_read_in_segment: seg.files_read.clone(),
                    files_edited_in_segment: seg.files_edited.clone(),
                    handles_created_in_segment: seg.handles_created.clone(),
                    last_assistant_text: last_text,
                    failed_attempts_summary: seg.failed_attempts.clone(),
                };

                for m in &self.modules {
                    match m.on_turn_boundary(&snap).await? {
                        BoundaryAction::Continue => {}
                        BoundaryAction::AppendAdvisory { signal, message } => {
                            self.session
                                .lock()
                                .await
                                .push_message(Message::system_note(message.clone()))?;
                            self.events
                                .push(AgentEvent::ModuleAdvisory { signal, message });
                        }
                        BoundaryAction::ContinueRun { signal, message } => {
                            let note = if message.trim().is_empty() {
                                "Continue working on the remaining tasks.".to_string()
                            } else {
                                message
                            };
                            self.session
                                .lock()
                                .await
                                .push_message(Message::system_note(note.clone()))?;
                            self.events.push(AgentEvent::ModuleAdvisory {
                                signal,
                                message: note,
                            });
                            boundary_resume = true;
                        }
                        BoundaryAction::Halt { reason } => {
                            halted_reason = Some(reason);
                            should_halt = true;
                            break;
                        }
                    }
                }
            }

            // Enforce cost budget guardrails (includes cache writes — root CI cap + subagent cap!)
            let spent = self.metrics.lock().await.total_usage.cost_usd;
            if let Some(limit) = self.config.max_cost_usd
                && spent >= limit
            {
                halted_reason = Some(format!(
                    "session cost budget reached (${spent:.4} >= ${limit:.2})"
                ));
                break;
            }
            if self.config.subagent_depth > 0 && spent >= self.config.subagent_max_cost_usd {
                halted_reason = Some(format!(
                    "subagent cost budget reached (${spent:.4} >= ${:.2})",
                    self.config.subagent_max_cost_usd
                ));
                break;
            }

            if should_halt {
                break;
            }

            if had_tools {
                // Flush any untrusted/internal system notes queued by `on_turn_boundary`
                // (e.g. worker inbox messages) so the next LLM turn sees them immediately.
                let notes: Vec<String> = {
                    let mut q = self.pending_system_notes.lock().await;
                    std::mem::take(&mut *q)
                };
                if !notes.is_empty() {
                    self.session
                        .lock()
                        .await
                        .push_message(Message::system_note(notes.join("\n\n")))?;
                }
            } else {
                // Anthropic server-side `pause_turn` or module-requested continuation: continue loop!
                if stop_reason == StopReason::PauseTurn || boundary_resume {
                    idle_turn_streak = 0;
                    continue;
                }

                // Silent empty-reply guard (Prime/Hermes pattern): if the provider returned EndTurn
                // with zero tool calls and empty visible text, retry once before settling empty.
                if stop_reason == StopReason::EndTurn
                    && final_text.trim().is_empty()
                    && !empty_reply_retried
                {
                    empty_reply_retried = true;
                    self.session.lock().await.push_message(Message::user_text(
                        "Your previous response contained no visible text or tool calls. Please complete the task or provide your concise final answer.",
                    ))?;
                    continue;
                }

                // Deliver queued human steers (with user authority) and/or internal system notes
                // (e.g. untrusted worker updates — never tagged `[user-steer]` or pinned to `user_instructions`).
                let steers: Vec<String> = {
                    let mut q = self.pending_steers.lock().await;
                    std::mem::take(&mut *q)
                };
                let notes: Vec<String> = {
                    let mut q = self.pending_system_notes.lock().await;
                    std::mem::take(&mut *q)
                };
                if !steers.is_empty() || !notes.is_empty() {
                    idle_turn_streak = 0;
                    if !steers.is_empty() {
                        {
                            let mut followups = self.user_instructions.lock().await;
                            for st in &steers {
                                followups.push(clip_anchor_text(st, 4_000));
                                if followups.len() > 16 {
                                    followups.remove(0);
                                }
                            }
                        }
                        self.session
                            .lock()
                            .await
                            .push_message(Message::user_text(format!(
                                "[user-steer]\n{}",
                                steers.join("\n")
                            )))?;
                    }
                    if !notes.is_empty() {
                        self.session
                            .lock()
                            .await
                            .push_message(Message::system_note(notes.join("\n\n")))?;
                    }
                    continue;
                }

                // Check autonomous quality gates if configured
                if !self.config.autonomous_gates.is_empty()
                    && let Some(fail_msg) = self.run_autonomous_gates(&cancel).await?
                {
                    self.session
                        .lock()
                        .await
                        .push_message(Message::user_text(fail_msg))?;
                    continue;
                }

                // Deliver next queued post-turn follow-up prompt (FIFO) only after current turn settled
                let next_followup = self.pending_follow_ups.lock().await.pop_front();
                if let Some(followup) = next_followup {
                    {
                        let mut followups = self.user_instructions.lock().await;
                        followups.push(clip_anchor_text(&followup, 4_000));
                        if followups.len() > 16 {
                            followups.remove(0);
                        }
                    }
                    let mut s = self.session.lock().await;
                    s.push_message(Message::user_text(followup))?;
                    s.mark_checkpoint("user_followup")?;
                    continue;
                }

                settled_normally = true;
                break;
            }
        }

        if !settled_normally && halted_reason.is_none() {
            halted_reason = Some(format!("max_turns reached ({})", self.config.max_turns));
        }

        let metrics = self.metrics.lock().await.clone();
        self.events.push(AgentEvent::RunSettled {
            final_text: final_text.clone(),
            metrics: metrics.clone(),
        });
        Ok(RunOutcome {
            final_text,
            metrics,
            halted_reason,
        })
    }

    async fn record_segment_facts(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
        out: &ToolOutput,
        seg: &mut SegmentTracker,
    ) {
        let mut metrics = self.metrics.lock().await;
        seg.record_tool_outcome(tool_name, input, out, &mut metrics);
    }

    async fn run_autonomous_gates(&self, cancel: &CancellationToken) -> Result<Option<String>> {
        let Some(bash) = self.tools.get("Bash") else {
            return Ok(None);
        };
        for gate in &self.config.autonomous_gates {
            let out = bash
                .execute(serde_json::json!({"command": gate}), cancel.clone())
                .await?;
            if out.is_error {
                let msg = format!(
                    "[autonomous-gate failed: `{gate}`]\n{}\nFix the failure before concluding.",
                    out.text_only()
                );
                return Ok(Some(msg));
            }
        }
        Ok(None)
    }

    async fn compact_with_mode(
        &self,
        mode: CompactionMode,
        seg: &SegmentTracker,
        cancel: CancellationToken,
    ) -> Result<()> {
        let (eff_cfg, eff_provider) = self.effective_config_and_provider().await;
        let _ = execute_compaction(
            mode,
            &eff_cfg,
            &eff_provider,
            &self.tools,
            &self.modules,
            &self.session,
            &self.metrics,
            &self.total_cost_micros,
            &self.events,
            &self.raw_user_goal,
            &self.user_instructions,
            seg,
            None,
            cancel,
        )
        .await?;
        *self.last_prefix_state.lock().await = None;
        Ok(())
    }
}

/// Conservative repetition-loop detector (inspired by Hermes `repetition_guard.py`):
/// returns `true` if a long fragment (>=400 chars) is dominated by >=5 verbatim repeats of a single
/// normalized line or 60-char chunk covering >=50% of the output, preventing `MaxTokens` auto-continue
/// from stitching degenerate loops.
fn is_repetition_dominated(text: &str) -> bool {
    const MIN_LEN: usize = 400;
    const MIN_REPEATS: usize = 5;
    if text.len() < MIN_LEN {
        return false;
    }
    let mut line_counts: HashMap<&str, usize> = HashMap::new();
    for line in text.lines() {
        let t = line.trim();
        if t.len() >= 24 {
            let c = line_counts.entry(t).or_insert(0);
            *c += 1;
            if *c >= MIN_REPEATS && (*c * t.len()) * 2 >= text.len() {
                return true;
            }
        }
    }
    // Check 60-char non-overlapping stride windows for single-line repetition loops
    let chars: Vec<char> = text.chars().collect();
    if chars.len() >= MIN_LEN {
        let mut win_counts: HashMap<String, usize> = HashMap::new();
        for chunk in chars.chunks(60) {
            if chunk.len() < 60 {
                continue;
            }
            let key: String = chunk.iter().collect();
            let c = win_counts.entry(key).or_insert(0);
            *c += 1;
            if *c >= MIN_REPEATS && (*c * 60) * 2 >= chars.len() {
                return true;
            }
        }
    }
    false
}

/// Bounded UTF-8 preview helper for `AgentEvent::ToolEnd::output_preview` (ACP/RPC/SSE UIs).
fn clip_utf8_preview(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

/// Convenience helper returning a `ToolResultBlock` vector for external callers.
pub fn text_result(s: impl Into<String>) -> Vec<ToolResultBlock> {
    vec![ToolResultBlock::text(s)]
}

/// Estimates unscaled token counts per context category (~4 chars/token; images = 1,500 tokens).
/// Deferred tool schemas (`defer_loading == true`) cost 0 tokens until loaded.
fn estimate_context_breakdown(
    static_system_prompt: &str,
    wire_tools: &[ToolSpec],
    messages: &[Arc<Message>],
) -> ContextBreakdown {
    let mut out = ContextBreakdown {
        estimated: true,
        ..ContextBreakdown::default()
    };

    if !static_system_prompt.is_empty() {
        out.system = (static_system_prompt.len() as u64).div_ceil(4);
    }

    for tool in wire_tools {
        if tool.defer_loading {
            continue;
        }
        let schema_len = tool.input_schema.to_string().len();
        let chars = tool.name.len() + tool.description.len() + schema_len;
        out.tools = out.tools.saturating_add((chars as u64).div_ceil(4).max(1));
    }

    for msg in messages {
        for block in &msg.content {
            match block {
                ContentBlock::Text { text } => {
                    if msg.role == Role::User {
                        classify_user_text_into_breakdown(text, &mut out);
                    } else {
                        out.messages = out.messages.saturating_add((text.len() as u64) / 4 + 1);
                    }
                }
                ContentBlock::Thinking { thinking, .. } => {
                    out.messages = out.messages.saturating_add((thinking.len() as u64) / 4 + 1);
                }
                ContentBlock::Image { .. } => {
                    out.messages = out.messages.saturating_add(1_500);
                }
                ContentBlock::Opaque { raw, .. } => {
                    out.messages = out
                        .messages
                        .saturating_add((raw.to_string().len() as u64) / 4 + 1);
                }
                ContentBlock::ToolUse { name, input, .. } => {
                    let chars = name.len() + input.to_string().len();
                    out.tool_results = out.tool_results.saturating_add((chars as u64) / 4 + 4);
                }
                ContentBlock::ToolResult { content, .. } => {
                    for tb in content {
                        match tb {
                            ToolResultBlock::Text { text } => {
                                out.tool_results =
                                    out.tool_results.saturating_add((text.len() as u64) / 4 + 1);
                            }
                            ToolResultBlock::Image { .. } => {
                                out.tool_results = out.tool_results.saturating_add(1_500);
                            }
                        }
                    }
                }
            }
        }
    }

    out
}

fn classify_user_text_into_breakdown(text: &str, out: &mut ContextBreakdown) {
    if text.is_empty() {
        return;
    }
    for segment in text.split("\n\n") {
        let trimmed = segment.trim_start();
        let est = (segment.len() as u64) / 4 + 1;
        if trimmed.starts_with("[available-deferred-tools]") {
            out.tools = out.tools.saturating_add(est);
        } else if trimmed.starts_with("[workspace-facts]")
            || trimmed.starts_with("[global-instructions:")
            || trimmed.starts_with("[project-instructions:")
            || trimmed.starts_with("[carnot-skill:")
            || trimmed.starts_with("[agent-skills]")
            || trimmed.starts_with("[carnot-memory")
        {
            out.memory = out.memory.saturating_add(est);
        } else {
            out.messages = out.messages.saturating_add(est);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use carnot_ai::faux::{FauxProvider, ScriptedTurn};
    use carnot_cas::CasStore;
    use serde_json::json;

    #[tokio::test]
    async fn follow_up_waits_for_turn_settle_and_runs_in_fifo_order() {
        let dir = std::env::temp_dir().join(format!("carnot-followup-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("note.txt"), "hello").expect("write");
        let mut cfg = CarnotConfig::new(dir.clone());
        cfg.modules.sandbox = false;
        let cas = Arc::new(CasStore::open(&dir).expect("cas"));
        let tools = ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let faux = Arc::new(FauxProvider::new(vec![
            ScriptedTurn::tool_calls(vec![("c1", "Read", json!({"file_path": "note.txt"}))]),
            ScriptedTurn::text("first done"),
            ScriptedTurn::text("second done"),
        ]));
        let agent = Agent::new(cfg, faux, tools, Vec::new(), false).expect("agent");

        agent.follow_up("now do follow-up task").await;
        let out = agent.prompt("start main task").await.expect("prompt");
        assert_eq!(out.final_text, "second done");
        assert_eq!(out.metrics.turns, 3);

        // Verify follow-up was NOT injected into the tool_result message (index 2),
        // and instead arrived as its own User message AFTER "first done" (index 4)!
        let msgs = agent.active_messages().await;
        assert_eq!(msgs.len(), 6);
        assert!(msgs[2].has_tool_result());
        assert!(!msgs[2].text_concat().contains("now do follow-up task"));
        assert_eq!(msgs[3].text_concat(), "first done");
        assert_eq!(msgs[4].text_concat(), "now do follow-up task");
        assert_eq!(msgs[5].text_concat(), "second done");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn repetition_guard_catches_degenerate_loops_only() {
        let loop_text = "This is a stuck repeated line in output.\n".repeat(20);
        assert!(is_repetition_dominated(&loop_text));
        let normal_text = (0..30)
            .map(|i| format!("Unique diagnostic line number {i} with distinct payload {i:#x}\n"))
            .collect::<String>();
        assert!(!is_repetition_dominated(&normal_text));
    }

    #[tokio::test]
    async fn context_breakdown_tracks_categories_scales_to_total_and_clears_after_compaction() {
        let dir = std::env::temp_dir().join(format!("carnot-breakdown-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("note.txt"), "hello from tool result\n").expect("write");
        let mut cfg = CarnotConfig::new(dir.clone());
        cfg.modules.sandbox = false;
        let cas = Arc::new(CasStore::open(&dir).expect("cas"));
        let tools = ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let faux = Arc::new(FauxProvider::new(vec![
            ScriptedTurn::tool_calls(vec![("c1", "Read", json!({"file_path": "note.txt"}))]),
            ScriptedTurn::text("all done"),
            ScriptedTurn::text("compacted summary"),
            ScriptedTurn::text("after compaction"),
        ]));
        let agent = Agent::new(cfg, faux, tools, Vec::new(), false).expect("agent");

        // Before the first provider turn, no provider anchor exists -> breakdown is None.
        assert!(agent.context_breakdown().await.is_none());

        agent.prompt("inspect note.txt").await.expect("prompt");
        let (total, breakdown) = agent.current_context_with_breakdown().await;
        let b = breakdown.expect("breakdown after provider turn");
        assert!(total > 0);
        assert_eq!(
            b.total(),
            total,
            "breakdown must sum to exact context total"
        );
        assert!(b.estimated);
        assert!(b.system > 0);
        assert!(b.tools > 0);
        assert!(b.messages > 0);
        assert!(b.tool_results > 0);

        // Immediately after compaction, provider anchor is invalidated -> breakdown is None.
        agent.compact_now().await.expect("compact");
        assert!(agent.context_breakdown().await.is_none());

        // On the next provider turn, breakdown resumes and matches the new exact total.
        agent.prompt("continue").await.expect("post-compact prompt");
        let (total2, breakdown2) = agent.current_context_with_breakdown().await;
        let b2 = breakdown2.expect("breakdown restored after next turn");
        assert_eq!(b2.total(), total2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn estimate_context_breakdown_skips_deferred_tools_and_classifies_memory_markers() {
        let active_tool = ToolSpec {
            name: "Read".into(),
            description: "Read a file from disk.".into(),
            input_schema: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            defer_loading: false,
        };
        let deferred_tool = ToolSpec {
            name: "mcp__huge__tool".into(),
            description: "x".repeat(4000),
            input_schema: json!({"type": "object", "description": "y".repeat(4000)}),
            defer_loading: true,
        };
        let msg = Arc::new(Message::user_text(
            "[project-instructions:AGENTS.md]\nFollow repo rules.\n\nHello user prompt",
        ));
        let raw = estimate_context_breakdown(
            "Static system prompt text.",
            &[active_tool, deferred_tool],
            &[msg],
        );
        assert!(raw.system > 0);
        // Deferred tool with 8 KiB description must contribute 0 tokens!
        assert!(raw.tools < 100);
        assert!(raw.memory > 0);
        assert!(raw.messages > 0);
        assert_eq!(raw.tool_results, 0);
    }
}
