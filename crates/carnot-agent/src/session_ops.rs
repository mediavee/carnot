//! Session management, forking, steering/follow-up queues, model/effort configuration,
//! manual compaction, and gated tool invocation helpers on [`Agent`].

use crate::compaction::{CompactionMode, execute_compaction, hydrate_anchors_from_session};
use crate::context::{
    clip_utf8_preview, current_utc_date_ymd, estimate_context_breakdown, last_date_tag_in_messages,
};
use crate::segment::SegmentTracker;
use crate::{Agent, EventLog, EventStream, ForkSnapshot, add_cost_micros, tool_search};
use carnot_ai::Provider;
use carnot_core::{
    AgentEvent, CacheTtl, CarnotConfig, CarnotError, ContextBreakdown, Message, ReasoningEffort,
    Result, SessionMetrics, ToolOutput, ToolSpec,
};
use carnot_ext::{PermissionBroker, UserQuestionBroker};
use carnot_session::SessionStore;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Instant;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

impl Agent {
    pub(crate) fn make_progress_sink(&self, call_id: String) -> carnot_tools::ToolProgressSink {
        let events = Arc::clone(&self.events);
        Arc::new(move |message: String| {
            events.push(AgentEvent::ToolUpdate {
                call_id: call_id.clone(),
                message,
            });
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
    /// Fails immediately if a prompt is currently running so ACP stdin never deadlocks.
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
        let loaded_date = last_date_tag_in_messages(loaded.active_messages());
        *self.session.lock().await = loaded;
        *self.turn1_injected.lock().await = has_msgs;
        *self.raw_user_goal.lock().await = hydrated_goal;
        *self.user_instructions.lock().await = hydrated_followups;
        *self.last_prefix_state.lock().await = None;
        *self.last_prompt_context_tokens.lock().await = None;
        *self.active_segment.lock().await = SegmentTracker::default();
        *self.last_emitted_date.lock().await = loaded_date;
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
            *self.last_prompt_context_tokens.lock().await = None;
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

    /// Attaches a runtime `GoalController` (from `carnot-ext-goal`) for live TUI/RPC/HTTP control.
    pub fn set_goal_controller(&self, controller: Arc<dyn carnot_ext::GoalController>) {
        if let Ok(mut g) = self.goal_controller.write() {
            *g = Some(controller);
        }
    }

    /// Returns the attached `GoalController`, if goal mode is available on this agent.
    pub fn goal_controller(&self) -> Option<Arc<dyn carnot_ext::GoalController>> {
        self.goal_controller.read().ok().and_then(|g| g.clone())
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
        eager.sort_unstable();
        deferred.sort_unstable();
        (eager, deferred)
    }

    /// Returns all registered tool specifications in deterministic canonical order
    /// (including deferred MCP and extension tools).
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.tools.specs()
    }

    /// Executes one tool call through Carnot's full security, concurrency, and observability
    /// pipeline:
    /// - emits balanced `ToolStart` / `ToolEnd` events (so TUI/ACP correlate permission requests),
    /// - runs `before_tool_call` modules (`PolicyGuard`, `Hooks`) and `PermissionBroker`,
    /// - enforces effect-based locking (`serial` vs per-path `RwLock`s) so concurrent `Promise.all`
    ///   calls to `Edit`/`Write`/`Bash` never race or lose updates,
    /// - suppresses model-facing `ReadTracker` view/stub mutations so internal reads never poison
    ///   the LLM's view and always return full text to the script,
    /// - runs `after_tool_result` hooks (secret redaction, LSP diagnostics) and updates the active
    ///   `SegmentTracker` so compaction retains every edited file and CAS handle.
    pub async fn invoke_tool_gated(
        &self,
        call_id: &str,
        tool_name: &str,
        raw_input: serde_json::Value,
        cancel: CancellationToken,
    ) -> Result<ToolOutput> {
        let Some(tool) = self.tools.get(tool_name) else {
            return Ok(ToolOutput::err_text(format!("Unknown tool `{tool_name}`")));
        };
        let t0 = Instant::now();
        self.events.push(AgentEvent::ToolStart {
            call_id: call_id.to_string(),
            tool: tool_name.to_string(),
            eager: false,
            input: raw_input.clone(),
        });

        let (effective_input, denied) = self.gate_tool_call(call_id, tool_name, &raw_input).await;
        if let Some(reason) = denied {
            let out = ToolOutput::err_text(format!("Tool `{tool_name}` denied: {reason}"));
            self.events.push(AgentEvent::ToolEnd {
                call_id: call_id.to_string(),
                tool: tool_name.to_string(),
                is_error: true,
                duration_ms: u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
                output_preview: clip_utf8_preview(&out.text_only(), 8_192),
            });
            return Ok(out);
        }

        let effects = tool.effects(&effective_input);
        let effect_guard = tokio::select! {
            biased;
            () = cancel.cancelled() => {
                self.events.push(AgentEvent::ToolEnd {
                    call_id: call_id.to_string(),
                    tool: tool_name.to_string(),
                    is_error: true,
                    duration_ms: u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
                    output_preview: "Cancelled".to_string(),
                });
                return Err(CarnotError::Cancelled);
            }
            guard = self.effect_locks.acquire(&effects) => guard,
        };

        let progress = self.make_progress_sink(call_id.to_string());
        let mut out = match carnot_tools::tracker::with_suppressed_read_tracking(
            tool.execute_with_progress(call_id, effective_input.clone(), cancel, Some(progress)),
        )
        .await
        {
            Ok(out) => out,
            Err(CarnotError::Cancelled) => {
                self.events.push(AgentEvent::ToolEnd {
                    call_id: call_id.to_string(),
                    tool: tool_name.to_string(),
                    is_error: true,
                    duration_ms: u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
                    output_preview: "Cancelled".to_string(),
                });
                return Err(CarnotError::Cancelled);
            }
            Err(e) => ToolOutput::err_text(format!("Tool `{tool_name}` error: {e}")),
        };
        drop(effect_guard);

        for m in &self.modules {
            let _ = m
                .after_tool_result(tool_name, &effective_input, &mut out)
                .await;
        }
        {
            let mut seg = self.active_segment.lock().await;
            self.record_segment_facts(tool_name, &effective_input, &out, &mut seg)
                .await;
        }

        self.events.push(AgentEvent::ToolEnd {
            call_id: call_id.to_string(),
            tool: tool_name.to_string(),
            is_error: out.is_error,
            duration_ms: u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
            output_preview: clip_utf8_preview(&out.text_only(), 8_192),
        });
        Ok(out)
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

    /// True if a `prompt()` run is currently active (used by HTTP server to return 409 Conflict).
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
        let safe_max_out = ((rt.context_window / 2).max(4096)).min(u64::from(u32::MAX)) as u32;
        if rt.max_output_tokens > safe_max_out {
            rt.max_output_tokens = safe_max_out;
        }
        drop(rt);
        *self.last_prefix_state.lock().await = None;
        *self.last_prompt_context_tokens.lock().await = None;
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

    pub async fn current_cache_ttl(&self) -> CacheTtl {
        self.runtime.lock().await.cache_ttl
    }

    /// Updates the live prompt-cache retention policy for subsequent provider requests.
    pub async fn set_cache_ttl(&self, ttl: CacheTtl) {
        self.runtime.lock().await.cache_ttl = ttl;
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

    pub(crate) async fn effective_config_and_provider(&self) -> (CarnotConfig, Arc<dyn Provider>) {
        let rt = self.runtime.lock().await;
        let mut cfg = self.config.clone();
        cfg.model.clone_from(&rt.model);
        cfg.effort = rt.effort;
        cfg.cache_ttl = rt.cache_ttl;
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

    pub async fn metrics(&self) -> SessionMetrics {
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
        let seg_snapshot = {
            let g = self.active_segment.lock().await;
            g.clone()
        };
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
            &seg_snapshot,
            focus,
            CancellationToken::new(),
        )
        .await;
        if res.is_ok() {
            *self.last_prefix_state.lock().await = None;
            *self.last_prompt_context_tokens.lock().await = None;
            *self.active_segment.lock().await = SegmentTracker::default();
            *self.last_emitted_date.lock().await = Some(current_utc_date_ymd());
        }
        res
    }

}
