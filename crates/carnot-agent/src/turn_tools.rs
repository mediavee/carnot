//! Tool gating, eager-launch cleanup, conflict-free parallel wave execution,
//! post-tool hooks, segment-fact tracking, and autonomous verification gates.

use crate::Agent;
use crate::compaction::clip_anchor_text;
use crate::context::clip_utf8_preview;
use crate::scheduler::{ActiveEager, EagerLaunch, ScheduledCall, partition_conflict_free_waves};
use crate::segment::SegmentTracker;
use carnot_core::{
    AgentEvent, CarnotError, ContentBlock, Message, Result, Role, StopReason, ToolEffects,
    ToolOutput,
};
use carnot_ext::ToolGateDecision;
use std::collections::HashMap;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

impl Agent {
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

    /// Unified tool gate used identically by BOTH eager mid-stream launches and Stage 1 execution.
    /// For `CallDeferredTool`, unwraps and gates ONLY the real inner target tool so policies,
    /// hooks, and host-RPC gates keyed on tool names can never be bypassed or double-gated.
    pub(crate) async fn gate_tool_call(
        &self,
        call_id: &str,
        tool_name: &str,
        raw_input: &serde_json::Value,
    ) -> (serde_json::Value, Option<String>) {
        if tool_name == "CallDeferredTool" {
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
        }
    }

    pub(crate) fn discard_eager_launches(&self, eager_launches: HashMap<String, EagerLaunch>) {
        // Abort speculative tasks AND invalidate their ReadTracker entries so next Read never returns
        // a false "unchanged" stub for outputs the model never saw.
        for (_, launch) in eager_launches {
            if let Some(jh) = launch.handle {
                jh.abort();
            }
            for p in &launch.effects.reads {
                self.tools.invalidate_read_path(p);
            }
        }
    }

    pub(crate) async fn execute_turn_tool_calls(
        &self,
        turn_num: u64,
        tool_calls: Vec<(String, String, serde_json::Value, Option<String>)>,
        mut eager_launches: HashMap<String, EagerLaunch>,
        cancel: &CancellationToken,
    ) -> Result<()> {
        let mut outputs: Vec<Option<(serde_json::Value, ToolOutput, u64)>> =
            vec![None; tool_calls.len()];
        let mut pending_calls: Vec<ScheduledCall> = Vec::new();
        let mut active_eager: Vec<ActiveEager> = Vec::new();

        // Stage 1: Gate non-eager calls, reject malformed JSON, classify into eager vs wave queue.
        for (idx, (call_id, tool_name, raw_input, invalid_json)) in tool_calls.iter().enumerate() {
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

            // Gate the real tool directly via unified `gate_tool_call`.
            let (effective_input, denied) =
                self.gate_tool_call(call_id, tool_name, raw_input).await;
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
                .map_or_else(ToolEffects::serial, |t| t.effects(&effective_input));
            pending_calls.push(ScheduledCall {
                orig_idx: idx,
                call_id: call_id.clone(),
                tool_name: tool_name.clone(),
                effective_input,
                effects,
            });
        }

        // Abort any stray eager handles not present in final assistant message & invalidate tracker.
        self.discard_eager_launches(eager_launches);

        // Stage 2: Partition pending calls into conflict-free parallel waves.
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
                let (progress, progress_closed) = self.make_progress_sink(call.call_id.clone());
                let jh = tokio::spawn(async move {
                    let t0 = Instant::now();
                    let out = match tool_opt {
                        Some(t) => t
                            .execute_with_progress(&cid, inp, c, Some(progress))
                            .await
                            .unwrap_or_else(|e| ToolOutput::err_text(e.to_string())),
                        None => ToolOutput::err_text(format!("Unknown tool: {tname}")),
                    };
                    (out, t0.elapsed().as_millis() as u64)
                });
                wave_handles.push((call.orig_idx, call.effective_input, jh, progress_closed));
            }

            for (idx, eff_inp, mut jh, progress_closed) in wave_handles {
                let (out, dur) = tokio::select! {
                    biased;
                    () = cancel.cancelled() => {
                        progress_closed.store(true, std::sync::atomic::Ordering::Release);
                        jh.abort();
                        (ToolOutput::err_text("Cancelled"), 0)
                    }
                    res = &mut jh => {
                        progress_closed.store(true, std::sync::atomic::Ordering::Release);
                        res.unwrap_or_else(|e| {
                            (ToolOutput::err_text(format!("tool task panicked: {e}")), 0)
                        })
                    }
                };
                outputs[idx] = Some((eff_inp, out, dur));
            }
        }

        // Collect any remaining non-conflicting eager tasks
        for mut ae in active_eager {
            let (out, dur) = tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    ae.handle.abort();
                    (ToolOutput::err_text("Cancelled"), 0)
                }
                res = &mut ae.handle => res.unwrap_or_else(|e| {
                    (ToolOutput::err_text(format!("eager task panicked: {e}")), 0)
                }),
            };
            outputs[ae.orig_idx] = Some((ae.effective_input, out, dur));
        }

        // Stage 3: Run `after_tool_result` hooks, emit `ToolEnd`, and build ordered `ToolResult`s.
        let mut result_blocks = Vec::with_capacity(tool_calls.len());
        for (idx, (call_id, tool_name, _, _)) in tool_calls.into_iter().enumerate() {
            let (effective_input, mut out, dur_ms) = outputs[idx].take().unwrap_or_else(|| {
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

            {
                let mut seg = self.active_segment.lock().await;
                self.record_segment_facts(&tool_name, &effective_input, &out, &mut seg)
                    .await;
            }

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
        Ok(())
    }

    pub(crate) async fn record_segment_facts(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
        out: &ToolOutput,
        seg: &mut SegmentTracker,
    ) {
        let mut metrics = self.metrics.lock().await;
        seg.record_tool_outcome(tool_name, input, out, &mut metrics);
    }

    pub(crate) async fn run_autonomous_gates(
        &self,
        cancel: &CancellationToken,
    ) -> Result<Option<String>> {
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
}
