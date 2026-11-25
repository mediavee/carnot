//! Core turn execution loop, provider streaming, retry/backoff, effect-scheduled tool waves,
//! permission/hook gating, autonomous verification gates, and proactive compaction.

use crate::compaction::{
    CompactionMode, clip_anchor_text, effective_proactive_trigger, execute_compaction,
};
use crate::context::{
    PrefixFingerprint, compute_messages_prefix_hash, compute_wire_static_hash,
    current_utc_date_ymd, is_repetition_dominated,
};
use crate::scheduler::EagerLaunch;
use crate::{Agent, RunOutcome, add_cost_micros, tool_search};
use carnot_ai::{ProviderRequest, StreamEvent};
use carnot_core::{
    AgentEvent, CarnotError, ContentBlock, Message, Result, Role, StopReason, ToolEffects,
    ToolOutput, ToolResultBlock, Usage,
};
use carnot_ext::{BoundaryAction, TurnBoundarySnapshot};
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio_util::sync::CancellationToken;

impl Agent {
    /// Runs a user text prompt through the full multi-turn agent loop until completion or halt.
    /// Serialized per agent and guaranteed to always emit a terminal `RunSettled` or `Error` event.
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

        // Sparse UTC calendar date note (NEVER in the static system prompt, so `audit-prefix`
        // stays byte-identical): emitted once on the first user turn (placed AFTER Turn-1 context
        // blocks and directly before the user prompt) and again only if the UTC calendar date
        // changed across a resumed session.
        if !continue_existing {
            let today = current_utc_date_ymd();
            let mut last_date = self.last_emitted_date.lock().await;
            if last_date.as_deref() != Some(today.as_str()) {
                *last_date = Some(today.clone());
                let tag = format!("[date: {today} UTC]");
                if initial_text.is_empty() {
                    initial_text = tag;
                } else {
                    initial_text = format!("{tag}\n\n{initial_text}");
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

        // Cold-cache opportunistic tool-result pruning:
        // Only when the provider's rolling prompt cache is already expired (zero cache breaks,
        // zero LLM calls), replace oversized stale tool results outside the last 4 messages
        // with compact handle-preserving stubs before the cold cache rewrite.
        if self.config.modules.compaction {
            let live_model = self.current_model().await;
            let live_ttl = self.current_cache_ttl().await;
            let ttl_ms = self
                .current_provider()
                .await
                .rolling_cache_ttl_ms(&live_model, live_ttl);
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_millis() as u64);
            let reclaimed = {
                let mut s = self.session.lock().await;
                let cache_expired = s
                    .last_activity_unix_ms()
                    .is_some_and(|last| now_ms.saturating_sub(last) >= ttl_ms);
                if cache_expired && s.active_messages().len() >= 8 {
                    s.prune_cold_tool_results(4, 2_000, 24_000)?
                } else {
                    0
                }
            };
            if reclaimed > 0 {
                self.tools.reset_read_tracker();
                *self.last_prefix_state.lock().await = None;
                *self.last_prompt_context_tokens.lock().await = None;
                self.metrics.lock().await.compacted_tokens_saved += (reclaimed as u64) / 4;
            }
        }

        // Collect per-prompt module injections (`before_prompt`: durable vs ephemeral)
        let mut ephemeral_notes: Vec<String> = Vec::new();
        let mut next_turn_ephemeral_advisories: Vec<String> = Vec::new();
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

        self.active_segment.lock().await.reset();
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

            // Proactive cache-warm compaction (provider-grounded token count + cooldown)
            if eff_cfg.modules.compaction {
                let (pre_est, msg_count) = {
                    let s = self.session.lock().await;
                    (s.effective_context_tokens(), s.active_messages().len())
                };
                let current_turns = self.metrics.lock().await.turns;
                let cooldown_ok = last_compaction_turn.is_none_or(|t| {
                    current_turns.saturating_sub(t) >= eff_cfg.compaction_cooldown_turns
                });
                let trigger = effective_proactive_trigger(&eff_cfg);
                if trigger > 0 && msg_count >= 10 && cooldown_ok && pre_est >= trigger {
                    last_compaction_turn = Some(current_turns);
                    self.compact_with_mode(CompactionMode::Soft, cancel.clone())
                        .await?;
                    self.active_segment.lock().await.reset();
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
                let mut turn_ephemeral_notes = ephemeral_notes.clone();
                turn_ephemeral_notes.append(&mut next_turn_ephemeral_advisories);
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
                    ephemeral_tail_notes: turn_ephemeral_notes,
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
                    self.compact_with_mode(CompactionMode::EmergencyOverflow, cancel.clone())
                        .await?;
                    self.active_segment.lock().await.reset();
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
                            .map_or_else(ToolEffects::serial, |t| t.effects(&input));

                        // Never block SSE stream draining on interactive human permission prompts;
                        // when a PermissionBroker is attached, defer gating & execution to Stage 1.
                        let has_interactive_broker = self.permission_broker.lock().await.is_some();
                        let can_eager = self.config.modules.eager_read
                            && !has_interactive_broker
                            && invalid_json.is_none()
                            && raw_eff.is_eager_safe()
                            && seen_stream_effects
                                .iter()
                                .all(|e| !e.conflicts_with(&raw_eff));

                        if can_eager {
                            if let Some(tool) = tool_opt {
                                // Run unified gate (unwrapping CallDeferredTool identically to Stage 1)
                                // BEFORE launching eager execution!
                                let (eff_input, denied) =
                                    self.gate_tool_call(&id, &name, &input).await;
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
                                    let progress = self.make_progress_sink(id.clone());
                                    Some(tokio::spawn(async move {
                                        let t0 = Instant::now();
                                        let out = tool
                                            .execute_with_progress(
                                                &cid_exec,
                                                inp_exec,
                                                c,
                                                Some(progress),
                                            )
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
                    self.compact_with_mode(CompactionMode::EmergencyOverflow, cancel.clone())
                        .await?;
                    self.active_segment.lock().await.reset();
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
                        () = cancel.cancelled() => return Err(CarnotError::Cancelled),
                        () = tokio::time::sleep(delay) => {}
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

            // Context window overflow reported as stop_reason -> abort eager & compact at most once.
            if stop_reason == StopReason::ContextWindowExceeded
                && self.config.modules.compaction
                && consecutive_emergency_compactions < 1
            {
                consecutive_emergency_compactions += 1;
                last_compaction_turn = Some(turn_num);
                self.discard_eager_launches(eager_launches);
                self.compact_with_mode(CompactionMode::EmergencyOverflow, cancel.clone())
                    .await?;
                self.active_segment.lock().await.reset();
                continue;
            }
            consecutive_emergency_compactions = 0;

            // Record usage, context AUC, and structured cache telemetry.
            {
                let mut prev_ctx = self.last_prompt_context_tokens.lock().await;
                let mut m = self.metrics.lock().await;
                let turn_ctx_tokens = usage.total_context_tokens();
                if let Some(prev) = *prev_ctx {
                    m.record_prefix_retention(prev, usage.cache_read_tokens);
                }
                *prev_ctx = Some(turn_ctx_tokens);
                m.context_auc_tokens += turn_ctx_tokens;
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
                    prefix_retention = m.prefix_retention_rate(),
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
            // truncated `tool_use` blocks with `is_error` `tool_result`s so the session stays valid.
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
                self.execute_turn_tool_calls(turn_num, tool_calls, eager_launches, &cancel)
                    .await?;
            } else {
                self.discard_eager_launches(eager_launches);
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
                if had_tools {
                    idle_turn_streak = 0;
                } else {
                    idle_turn_streak += 1;
                    if self.config.idle_turn_circuit_breaker > 0
                        && idle_turn_streak >= self.config.idle_turn_circuit_breaker
                    {
                        halted_reason = Some(format!(
                            "idle circuit breaker triggered ({idle_turn_streak} consecutive turns without tool calls)"
                        ));
                        should_halt = true;
                    }
                }
            }

            // Evaluate turn-boundary module hooks
            let mut boundary_resume = false;
            if !should_halt && !self.modules.is_empty() {
                let total_ctx_est = self.session.lock().await.effective_context_tokens();
                let seg_snap = self.active_segment.lock().await.clone();
                let snap = TurnBoundarySnapshot {
                    turn_index: turn_num,
                    stop_reason,
                    total_context_tokens: total_ctx_est,
                    had_edit_in_segment: seg_snap.had_edit,
                    had_tool_call_in_turn: had_tools,
                    files_read_in_segment: seg_snap.files_read,
                    files_edited_in_segment: seg_snap.files_edited,
                    handles_created_in_segment: seg_snap.handles_created,
                    last_assistant_text: last_text,
                    failed_attempts_summary: seg_snap.failed_attempts,
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
                        BoundaryAction::AppendEphemeralAdvisory { signal, message } => {
                            let trimmed = message.trim();
                            if !trimmed.is_empty() {
                                next_turn_ephemeral_advisories.push(trimmed.to_string());
                            }
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

                // Silent empty-reply guard: if the provider returned EndTurn
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

    async fn compact_with_mode(
        &self,
        mode: CompactionMode,
        cancel: CancellationToken,
    ) -> Result<()> {
        let (eff_cfg, eff_provider) = self.effective_config_and_provider().await;
        let seg_snap = self.active_segment.lock().await.clone();
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
            &seg_snap,
            None,
            cancel,
        )
        .await?;
        *self.last_prefix_state.lock().await = None;
        *self.last_prompt_context_tokens.lock().await = None;
        *self.last_emitted_date.lock().await = Some(current_utc_date_ymd());
        Ok(())
    }
}

