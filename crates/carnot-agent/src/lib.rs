//! Core asynchronous agent loop with:
//!   - Opus I2 Effect-Based Parallel Scheduler (gate-before-eager reads + parallel disjoint `Edit`/`Write`
//!     + strict causal ordering on overlapping paths & `Bash` barriers)
//!   - Opus I4 Lossless monotone-`seq` `EventLog` (immune to slow-reader `Lagged` disconnects)
//!   - Mid-turn `steer()` injection preserving `tool_result`-first ordering
//!   - Safe `StopReason` (`MaxTokens`, `Refusal`, `ContextWindowExceeded`) & malformed tool-JSON guards
//!   - Module lifecycle orchestration, automatic Rewind-Receipt, and live thermodynamic telemetry

pub mod compaction;
pub mod event_log;
mod scheduler;
mod segment;
pub mod subagent;
pub mod tool_search;

use crate::compaction::{
    CompactionMode, clip_anchor_text, effective_proactive_trigger, execute_compaction,
    hydrate_anchors_from_session,
};
use crate::scheduler::{ActiveEager, EagerLaunch, ScheduledCall, partition_conflict_free_waves};
use crate::segment::SegmentTracker;

pub use event_log::{EventLog, EventStream};
pub use subagent::SubagentModule;

use carnot_ai::{Provider, ProviderRequest, StreamEvent};
use carnot_core::{
    AgentEvent, CacheTtl, CarnotConfig, CarnotError, ContentBlock, Message, ReasoningEffort,
    Result, Role, StopReason, ThermodynamicMetrics, ToolEffects, ToolOutput, ToolResultBlock,
    Usage,
};
use carnot_ext::{
    BoundaryAction, Module, PermissionBroker, ToolGateDecision, TurnBoundarySnapshot,
    prefix_guard::static_system_prompt,
};
use carnot_session::SessionStore;
use carnot_tools::ToolRegistry;
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

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
    events: Arc<EventLog>,
    run_lock: Mutex<()>,
    cancel_current: Mutex<CancellationToken>,
    turn1_injected: Mutex<bool>,
    raw_user_goal: Mutex<Option<String>>,
    user_instructions: Mutex<Vec<String>>,
    pending_steers: Mutex<Vec<String>>,
    pending_follow_ups: Mutex<VecDeque<String>>,
    permission_broker: Mutex<Option<Arc<dyn PermissionBroker>>>,
    deferred_server_descriptions: std::sync::Mutex<std::collections::BTreeMap<String, String>>,
}

impl Agent {
    pub fn new(
        config: CarnotConfig,
        provider: Arc<dyn Provider>,
        mut tools: ToolRegistry,
        modules: Vec<Arc<dyn Module>>,
        persist_session: bool,
    ) -> Result<Self> {
        for m in &modules {
            for t in m.tools() {
                tools.register(t);
            }
        }
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
        let session = SessionStore::create(
            session_dir,
            config.model.clone(),
            static_system_prompt().to_string(),
            tool_specs,
        )?;
        Self::from_session(config, provider, tools, modules, session, false)
    }

    /// Resumes an existing crash-safe JSONL v2 session from disk (`carnot --resume` & ACP `session/load`).
    pub fn resume(
        session_path: &Path,
        config: CarnotConfig,
        provider: Arc<dyn Provider>,
        mut tools: ToolRegistry,
        modules: Vec<Arc<dyn Module>>,
    ) -> Result<Self> {
        for m in &modules {
            for t in m.tools() {
                tools.register(t);
            }
        }
        let deferred = tools.deferred_tools();
        if !deferred.is_empty() {
            tools.register(Arc::new(tool_search::ToolSearchTool::new(&deferred)));
            tools.register(Arc::new(tool_search::CallDeferredTool::new(&deferred)));
        }
        let session = SessionStore::load(session_path)?;
        let already_started = !session.active_messages().is_empty();
        Self::from_session(config, provider, tools, modules, session, already_started)
    }

    fn from_session(
        config: CarnotConfig,
        provider: Arc<dyn Provider>,
        tools: ToolRegistry,
        modules: Vec<Arc<dyn Module>>,
        session: SessionStore,
        turn1_already_injected: bool,
    ) -> Result<Self> {
        let (hydrated_goal, hydrated_followups) =
            hydrate_anchors_from_session(session.active_messages());
        let events = Arc::new(EventLog::new());
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
            events,
            run_lock: Mutex::new(()),
            cancel_current: Mutex::new(CancellationToken::new()),
            turn1_injected: Mutex::new(turn1_already_injected),
            raw_user_goal: Mutex::new(hydrated_goal),
            user_instructions: Mutex::new(hydrated_followups),
            pending_steers: Mutex::new(Vec::new()),
            pending_follow_ups: Mutex::new(VecDeque::new()),
            permission_broker: Mutex::new(None),
            deferred_server_descriptions: std::sync::Mutex::new(std::collections::BTreeMap::new()),
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
        self.tools.reset_read_tracker();
        Ok(sid)
    }

    /// Returns a snapshot of active conversation messages (for ACP `session/load` history replay).
    pub async fn active_messages(&self) -> Vec<Arc<Message>> {
        self.session.lock().await.active_messages().to_vec()
    }

    /// Attaches an interactive `PermissionBroker` (e.g. ACP `session/request_permission`).
    pub async fn set_permission_broker(&self, broker: Arc<dyn PermissionBroker>) {
        *self.permission_broker.lock().await = Some(broker);
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
        Ok(())
    }

    /// Updates the reasoning effort for subsequent turns on this session.
    pub async fn set_effort(&self, effort: ReasoningEffort) {
        self.runtime.lock().await.effort = effort;
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
        execute_compaction(
            CompactionMode::Manual,
            &eff_cfg,
            &eff_provider,
            &self.tools,
            &self.modules,
            &self.session,
            &self.metrics,
            &self.events,
            &self.raw_user_goal,
            &self.user_instructions,
            &seg,
            focus,
            CancellationToken::new(),
        )
        .await
    }

    async fn run_before_gates(
        &self,
        call_id: &str,
        tool_name: &str,
        raw_input: &serde_json::Value,
    ) -> Result<(serde_json::Value, Option<String>)> {
        let mut effective_input = raw_input.clone();
        for m in &self.modules {
            match m.before_tool_call(tool_name, &effective_input).await? {
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

    /// Runs a user prompt through the full multi-turn agent loop until completion or halt.
    /// Serialized per agent (N5) and guaranteed to always emit a terminal `RunSettled` or `Error` event (N8).
    pub async fn prompt(&self, user_prompt: impl Into<String>) -> Result<RunOutcome> {
        let _run_guard = self.run_lock.lock().await;
        match self.prompt_inner(user_prompt.into()).await {
            Ok(outcome) => Ok(outcome),
            Err(e) => {
                self.events.push(AgentEvent::Error {
                    message: e.to_string(),
                });
                Err(e)
            }
        }
    }

    async fn prompt_inner(&self, mut initial_text: String) -> Result<RunOutcome> {
        let cancel = CancellationToken::new();
        *self.cancel_current.lock().await = cancel.clone();

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
                if !turn1_blocks.is_empty() {
                    initial_text = format!("{}\n\n{initial_text}", turn1_blocks.join("\n\n"));
                }
            }
        }

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
                self.metrics.lock().await.heat_shed_tokens += (reclaimed as u64) / 4;
            }
        }

        // Record user turn and mark Checkpoint C at the user boundary
        {
            let mut s = self.session.lock().await;
            s.push_message(Message::user_text(initial_text))?;
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
                    session_id: s.session_id.clone(),
                    messages: proj.messages,
                    checkpoint_index: proj.checkpoint_index,
                    cache_ttl: eff_cfg.cache_ttl,
                    effort: eff_cfg.effort,
                    max_output_tokens: eff_cfg.max_output_tokens,
                    skip_tail_cache_write: false,
                };
                (req, turn_num, ctx_est)
            };

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
                                    });
                                    self.metrics.lock().await.eager_tools_started += 1;
                                    let c = cancel.clone();
                                    let inp_exec = eff_input.clone();
                                    Some(tokio::spawn(async move {
                                        let t0 = Instant::now();
                                        let out =
                                            tool.execute(inp_exec, c).await.unwrap_or_else(|e| {
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
                    } => {
                        assistant_msg = message;
                        usage = u;
                        stop_reason = sr;
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

                    let (mut effective_input, mut denied) =
                        match self.run_before_gates(call_id, tool_name, raw_input).await {
                            Ok(pair) => pair,
                            Err(e) => (raw_input.clone(), Some(format!("gate error: {e}"))),
                        };
                    // Never allow `CallDeferredTool` to bypass module/broker gates keyed on the real tool name!
                    if denied.is_none()
                        && tool_name == "CallDeferredTool"
                        && let Some(inner_name) =
                            effective_input.get("name").and_then(|v| v.as_str())
                    {
                        let inner_args = effective_input
                            .get("arguments")
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!({}));
                        match self
                            .run_before_gates(call_id, inner_name, &inner_args)
                            .await
                        {
                            Ok((rewritten, inner_denied)) => {
                                if let Some(reason) = inner_denied {
                                    denied = Some(reason);
                                } else if let Some(obj) = effective_input.as_object_mut() {
                                    obj.insert("arguments".into(), rewritten);
                                }
                            }
                            Err(e) => {
                                denied = Some(format!("inner gate error: {e}"));
                            }
                        }
                    }
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
                        });
                        let tool_opt = self.tools.get(&call.tool_name);
                        let c = cancel.clone();
                        let inp = call.effective_input.clone();
                        let tname = call.tool_name.clone();
                        let jh = tokio::spawn(async move {
                            let t0 = Instant::now();
                            let out = match tool_opt {
                                Some(t) => t
                                    .execute(inp, c)
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

                    self.events.push(AgentEvent::ToolEnd {
                        call_id: call_id.clone(),
                        tool: tool_name,
                        is_error: out.is_error,
                        duration_ms: dur_ms,
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
            }

            self.events.push(AgentEvent::TurnEnd {
                turn: turn_num,
                stop_reason,
            });

            // Evaluate turn-boundary modules (Maxwell's Daemon + Rewind-Receipt)
            let (tokens_since_cp, total_ctx_est) = {
                let s = self.session.lock().await;
                (
                    s.tokens_since_checkpoint(),
                    s.project_context().estimated_tokens(),
                )
            };

            let snap = TurnBoundarySnapshot {
                turn_index: turn_num,
                tokens_since_checkpoint: tokens_since_cp,
                total_context_tokens: total_ctx_est,
                had_edit_in_segment: seg.had_edit,
                had_tool_call_in_turn: had_tools,
                verified_green_command: seg.verified_green_cmd.clone(),
                verified_green_summary: seg.verified_green_summary.clone(),
                files_read_in_segment: seg.files_read.clone(),
                files_edited_in_segment: seg.files_edited.clone(),
                handles_created_in_segment: seg.handles_created.clone(),
                last_assistant_text: last_text,
                failed_attempts_summary: seg.failed_attempts.clone(),
            };

            let mut should_halt = false;
            for m in &self.modules {
                match m.on_turn_boundary(&snap).await? {
                    BoundaryAction::Continue => {}
                    BoundaryAction::AppendAdvisory { signal, message } => {
                        self.metrics.lock().await.daemon_advisories += 1;
                        self.session
                            .lock()
                            .await
                            .push_message(Message::system_note(message.clone()))?;
                        self.events
                            .push(AgentEvent::DaemonAdvisory { signal, message });
                    }
                    BoundaryAction::TriggerRewind {
                        shed_tokens,
                        expected_net_usd,
                        receipt_text,
                    } => {
                        let cp = self.session.lock().await.checkpoint_index();
                        if let Some(cp_idx) = cp {
                            let receipt_tokens = (receipt_text.len() as u64 / 4).max(120);
                            if self
                                .session
                                .lock()
                                .await
                                .apply_rewind(cp_idx, shed_tokens, expected_net_usd, receipt_text)
                                .is_ok()
                            {
                                // Critical Red-Team guardrail: reset read-tracker after rewind!
                                self.tools.reset_read_tracker();
                                seg.reset();
                                {
                                    let mut met = self.metrics.lock().await;
                                    met.rewinds_applied += 1;
                                    met.heat_shed_tokens += shed_tokens;
                                }
                                self.events.push(AgentEvent::RewindApplied {
                                    checkpoint_index: cp_idx,
                                    shed_tokens,
                                    receipt_tokens,
                                    expected_net_usd,
                                });
                            }
                        }
                    }
                    BoundaryAction::Halt { reason } => {
                        halted_reason = Some(reason);
                        should_halt = true;
                        break;
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

            if !had_tools {
                // Anthropic server-side `pause_turn`: continue the loop without requiring a new user turn (A5)
                if stop_reason == StopReason::PauseTurn {
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

                // If a mid-turn steer arrived before settling, deliver it and continue!
                let steers: Vec<String> = {
                    let mut q = self.pending_steers.lock().await;
                    std::mem::take(&mut *q)
                };
                if !steers.is_empty() {
                    self.session
                        .lock()
                        .await
                        .push_message(Message::user_text(steers.join("\n")))?;
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
        // Roll subagent spend into parent metrics so `--max-cost-usd` can never be bypassed!
        if tool_name == "Subagent"
            && let Some(child_cost) = out.details.get("cost_usd").and_then(|v| v.as_f64())
            && child_cost.is_finite()
            && child_cost > 0.0
        {
            metrics.total_usage.cost_usd += child_cost;
        }
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
            &self.events,
            &self.raw_user_goal,
            &self.user_instructions,
            seg,
            None,
            cancel,
        )
        .await?;
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

/// Convenience helper returning a `ToolResultBlock` vector for external callers.
pub fn text_result(s: impl Into<String>) -> Vec<ToolResultBlock> {
    vec![ToolResultBlock::text(s)]
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
}
