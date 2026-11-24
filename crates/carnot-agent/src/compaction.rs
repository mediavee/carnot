//! Two-Tier Cache-Warm Thermodynamic Compaction Engine.
//!
//! Preserves maximum task fidelity while keeping long-horizon sessions (150-500+ turns) sharp and
//! economical:
//!   - **Tier 0 (Sacred Anchor):** The original user prompt (up to 24 KiB, never truncated to 600 chars)
//!     and all subsequent direct user instructions/steers are anchored verbatim at the top of the
//!     compacted state so no spec, path, schema, or tolerance can ever be lost.
//!   - **Tier 1 (Cache-Warm Semantic Handoff):** Reuses the exact byte-identical `static_system_prompt`,
//!     `tools`, and conversation `messages` to ask the model for a structured 6-section engineering
//!     handoff. Because the prefix is byte-identical to the prior turn, ~99% of compaction input hits
//!     the warm prompt cache (0.05x input rate on Anthropic 5.5).
//!   - **Ground-Truth Harness Footer:** Always appends verified `git status`/`git diff --stat`,
//!     edited files, lossless CAS `.carnot/h/<hash>.log` handles, and failed attempts so nothing relies
//!     solely on model memory.
//!   - **Bulletproof Fallback:** On hard `ContextOverflow` or if the semantic call fails/times out,
//!     gracefully falls back to deterministic compaction without ever losing user requirements or
//!     breaking `tool_use`/`tool_result` pairing.

use crate::segment::SegmentTracker;
use carnot_ai::{Provider, ProviderRequest, StreamEvent};
use carnot_core::{
    AgentEvent, CarnotConfig, ContentBlock, Message, ReasoningEffort, Result, Role, StopReason,
    ThermodynamicMetrics, Usage,
};
use carnot_ext::{CompactionHookAction, CompactionHookContext, Module};
use carnot_session::SessionStore;
use carnot_tools::ToolRegistry;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

const MAX_ANCHORED_GOAL_CHARS: usize = 24_000;
const MAX_FOLLOWUP_INSTRUCTION_CHARS: usize = 4_000;
const SEMANTIC_COMPACTION_TIMEOUT: Duration = Duration::from_secs(90);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionMode {
    /// Proactive thermodynamic compaction when context crosses `soft_compaction_tokens`.
    Soft,
    /// Explicit caller/user compaction (`/compact` or `Agent::compact_now()`).
    Manual,
    /// Reactive hard-overflow recovery (`ContextOverflow` / `ContextWindowExceeded`).
    EmergencyOverflow,
}

impl CompactionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Soft => "soft",
            Self::Manual => "manual",
            Self::EmergencyOverflow => "emergency",
        }
    }
}

/// Computes the effective proactive compaction trigger, clamped safely below combined
/// `context_window - max_output_tokens - compaction_reserve_tokens` (with an anti-thrash floor).
pub fn effective_proactive_trigger(config: &CarnotConfig) -> u64 {
    let output_headroom = u64::from(config.max_output_tokens).min(config.context_window / 2);
    let hard_trigger = config
        .context_window
        .saturating_sub(output_headroom)
        .saturating_sub(config.compaction_reserve_tokens);
    let floor = config
        .compaction_tail_budget_tokens
        .saturating_add(config.compaction_reserve_tokens)
        .min(hard_trigger);
    match config.soft_compaction_tokens {
        Some(soft) if soft > 0 => soft.min(hard_trigger).max(floor),
        _ => hard_trigger,
    }
}

/// Captures a bounded verbatim user instruction for the sacred compaction anchor.
pub fn clip_anchor_text(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }
    let head_len = (max_chars * 2) / 3;
    let tail_len = max_chars.saturating_sub(head_len + 64);
    let head: String = trimmed.chars().take(head_len).collect();
    let tail: String = trimmed
        .chars()
        .rev()
        .take(tail_len)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{head}\n[... middle clipped for anchor cap ...]\n{tail}")
}

fn extract_carnot_section<'a>(doc: &'a str, marker: &str) -> Option<&'a str> {
    let start = doc.find(marker)? + marker.len();
    let rest = doc[start..].strip_prefix('\n').unwrap_or(&doc[start..]);
    let end = rest.find("\n[carnot:").unwrap_or(rest.len());
    let body = rest[..end].trim();
    (!body.is_empty()).then_some(body)
}

/// Rehydrates Tier-0 sacred user anchors (`raw_user_goal` + chronological follow-ups) directly
/// from persisted `SessionStore::active_messages()`. Ensures `--resume`, ACP `session/load`,
/// and multi-compaction chains NEVER lose Phase-1 constraints or follow-up corrections.
pub fn hydrate_anchors_from_session(messages: &[Arc<Message>]) -> (Option<String>, Vec<String>) {
    let mut goal: Option<String> = None;
    let mut followups: Vec<String> = Vec::new();

    for msg in messages {
        let full_text = msg.text_concat();
        let trimmed = full_text.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Round-trip prior compaction anchors verbatim across multiple compactions & resumes
        if trimmed.starts_with("[carnot:compaction ") {
            if goal.is_none()
                && let Some(obj) = extract_carnot_section(trimmed, "[carnot:anchored-objective]")
            {
                goal = Some(clip_anchor_text(obj, MAX_ANCHORED_GOAL_CHARS));
            }
            if let Some(fsec) = extract_carnot_section(trimmed, "[carnot:user-followups]") {
                for chunk in fsec.split("### Follow-up ") {
                    let c = chunk.trim();
                    if c.is_empty() {
                        continue;
                    }
                    let body = c.split_once('\n').map(|(_, rest)| rest.trim()).unwrap_or(c);
                    if !body.is_empty() {
                        followups.push(clip_anchor_text(body, MAX_FOLLOWUP_INSTRUCTION_CHARS));
                    }
                }
            }
            continue;
        }

        if msg.role == Role::User
            && msg.has_text()
            && !trimmed.starts_with("[carnot:")
            && !trimmed.starts_with("[autonomous-gate ")
        {
            if goal.is_none() {
                goal = Some(clip_anchor_text(trimmed, MAX_ANCHORED_GOAL_CHARS));
            } else {
                let clipped = clip_anchor_text(trimmed, MAX_FOLLOWUP_INSTRUCTION_CHARS);
                if followups.last().map(String::as_str) != Some(clipped.as_str())
                    && goal.as_deref() != Some(clipped.as_str())
                {
                    followups.push(clipped);
                    if followups.len() > 16 {
                        followups.remove(0);
                    }
                }
            }
        }
    }

    (goal, followups)
}

/// Chooses the optimal compaction effort:
/// - On models whose effort parameter preserves the prompt cache (e.g. Claude 5.5 `output_config.effort`,
///   verified empirically at 99.1% cache hit), cap compaction at `Medium` (`Low` stays `Low`) so
///   high/max-effort sessions don't burn excessive thinking tokens just to write a handoff.
/// - On legacy `thinking.budget_tokens` models (or unknown providers), preserve exact `config.effort`
///   so prompt caches are never invalidated.
pub fn compaction_effort_for_provider(
    session_effort: ReasoningEffort,
    effort_preserves_cache: bool,
) -> ReasoningEffort {
    if effort_preserves_cache {
        match session_effort {
            ReasoningEffort::Low => ReasoningEffort::Low,
            _ => ReasoningEffort::Medium,
        }
    } else {
        session_effort
    }
}

/// Deterministically reconstructs `(modified_files, read_only_files)` from the persisted session
/// messages, any prior compaction footer, and the current turn's `SegmentTracker`.
/// Ensures file tracking survives `--resume` and multiple chained compactions without duplicating
/// modified files in the read-only list.
pub(crate) fn extract_file_ops_from_session(
    messages: &[Arc<Message>],
    seg: &SegmentTracker,
) -> (Vec<String>, Vec<String>) {
    let mut modified: BTreeSet<String> = BTreeSet::new();
    let mut read: BTreeSet<String> = BTreeSet::new();

    for f in &seg.files_edited {
        let t = f.trim();
        if !t.is_empty() {
            modified.insert(t.to_string());
        }
    }
    for f in &seg.files_read {
        let t = f.trim();
        if !t.is_empty() {
            read.insert(t.to_string());
        }
    }

    for msg in messages {
        let full_text = msg.text_concat();
        let trimmed = full_text.trim();
        // Carry forward file lists from any prior compaction summary
        if trimmed.starts_with("[carnot:compaction ")
            && let Some(ws_sec) =
                extract_carnot_section(trimmed, "[carnot:verified-workspace-state]")
        {
            for line in ws_sec.lines() {
                if let Some(rest) = line.strip_prefix("Files edited:") {
                    for p in rest.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                        modified.insert(p.to_string());
                    }
                } else if (line.starts_with("Read-only inspected files")
                    || line.starts_with("Previously inspected files"))
                    && let Some((_, rest)) = line.split_once(": ")
                {
                    for p in rest.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                        read.insert(p.to_string());
                    }
                }
            }
        }

        // Extract deterministic tool file targets directly from persisted assistant tool calls
        if msg.role == Role::Assistant {
            for block in &msg.content {
                if let ContentBlock::ToolUse { name, input, .. } = block
                    && let Some(path) = input
                        .get("file_path")
                        .or_else(|| input.get("path"))
                        .and_then(|v| v.as_str())
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                {
                    match name.as_str() {
                        "Edit" | "Write" | "edit" | "write" => {
                            modified.insert(path.to_string());
                        }
                        "Read" | "read" => {
                            read.insert(path.to_string());
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let read_only: Vec<String> = read
        .into_iter()
        .filter(|p| !modified.contains(p))
        .take(30)
        .collect();
    let modified_list: Vec<String> = modified.into_iter().take(40).collect();
    (modified_list, read_only)
}

/// Builds the concise, high-precision handoff prompt appended to the warm conversation prefix.
fn build_semantic_compaction_prompt(
    has_prior_compaction: bool,
    custom_focus: Option<&str>,
) -> String {
    let merge_note = if has_prior_compaction {
        "An earlier `[carnot:compaction]` handoff appears at the top of this conversation. MERGE and preserve every still-relevant fact, constraint, file path, and refuted hypothesis from it into your updated handoff.\n"
    } else {
        ""
    };
    let focus_note = match custom_focus.map(str::trim).filter(|s| !s.is_empty()) {
        Some(f) => format!("Additional caller focus to prioritize verbatim: {f}\n"),
        None => String::new(),
    };
    format!(
        "[carnot:compaction-request]\n\
Produce an ultra-dense, high-recall task handoff (~400-1,000 words, crisp bullet points, zero narrative fluff) so work can continue seamlessly after older turns are folded. Do NOT call any tools.\n\
Maximize exact recall: preserve every concrete number, constant, identifier, file path, schema/format rule, error message, decision, and user correction verbatim so no already-inspected file needs to be re-read.\n\
{merge_note}{focus_note}\
Use these exact 6 markdown headings:\n\
## 1. Exact Constraints & Edge Cases\n\
Every concrete requirement, output artifact/path, format or schema rule, numerical tolerance, user correction, and edge case.\n\
## 2. Verified State, Artifacts & Measurements\n\
Files or artifacts created/modified/deleted, checks/commands/tests executed, exact passing vs failing counts, and measured values.\n\
## 3. Key Discoveries, Decisions & References\n\
Exact references (file paths, line numbers, symbols, document sections, rows, formulas, or URLs), diagnosed root causes, accepted decisions, and invariants.\n\
## 4. Rejected Approaches & Dead Ends (DO NOT REPEAT)\n\
Approaches already tried or rejected and the exact reason each failed.\n\
## 5. Open Questions & Remaining Verification\n\
Unverified edge cases, ambiguities, or remaining requirements not yet confirmed.\n\
## 6. Immediate Next Step\n\
The exact concrete next action to execute immediately upon resuming."
    )
}

#[derive(Debug, Default)]
struct WorkspaceGitFacts {
    status_short: Option<String>,
    diff_stat: Option<String>,
}

async fn collect_git_facts(ws: std::path::PathBuf) -> WorkspaceGitFacts {
    tokio::task::spawn_blocking(move || {
        let status_short = carnot_sandbox::host_safe_git_command(&ws)
            .args(["status", "--short"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .take(40)
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .filter(|s| !s.trim().is_empty());

        let diff_stat = carnot_sandbox::host_safe_git_command(&ws)
            .args(["diff", "--stat"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .take(40)
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .filter(|s| !s.trim().is_empty());

        WorkspaceGitFacts {
            status_short,
            diff_stat,
        }
    })
    .await
    .unwrap_or_default()
}

/// Attempts a cache-warm LLM semantic handoff reusing the exact conversation prefix (with 1 retry
/// on transient stream errors, rejecting truncated `MaxTokens` or rogue tool-calling outputs).
/// Returns `Some((handoff_text, usage))` on success, or `None` to trigger deterministic fallback.
async fn try_semantic_llm_handoff(
    config: &CarnotConfig,
    provider: &Arc<dyn Provider>,
    tools: &ToolRegistry,
    session: &Mutex<SessionStore>,
    custom_focus: Option<&str>,
    cancel: CancellationToken,
) -> Option<(String, Usage)> {
    let req = {
        let s = session.lock().await;
        let proj = s.project_context();
        if proj.messages.len() < 4 {
            return None;
        }
        let has_prior = proj
            .messages
            .first()
            .is_some_and(|m| m.text_concat().contains("[carnot:compaction "));

        // Ensure trailing messages don't leave an unanswered `tool_use` before appending our User prompt
        let mut msgs = proj.messages;
        if msgs.last().is_some_and(|m| m.has_tool_use()) {
            return None;
        }
        msgs.push(Arc::new(Message::user_text(
            build_semantic_compaction_prompt(has_prior, custom_focus),
        )));

        let compaction_effort = compaction_effort_for_provider(
            config.effort,
            provider.effort_preserves_cache(&config.model),
        );

        // Reuse the warm conversation prefix verbatim (`system`, `tools`, `messages`, `cache_ttl`)
        // with `skip_tail_cache_write: true` so the existing prefix hits at 0.05x while the
        // discarded tail + compaction request pays 1.0x instead of 1.25x/2.0x cache-write.
        let wire_tools = crate::tool_search::effective_tools_for_provider(
            &tools.specs(),
            provider.supports_native_tool_search(),
        );
        ProviderRequest {
            model: config.model.clone(),
            static_system_prompt: proj.static_system_prompt,
            tools: wire_tools,
            prefix_hash: proj.prefix_hash,
            session_id: proj.cache_key_id,
            messages: msgs,
            checkpoint_index: proj.checkpoint_index,
            cache_ttl: config.cache_ttl,
            effort: compaction_effort,
            max_output_tokens: config.max_output_tokens.min(16_384),
            skip_tail_cache_write: true,
            native_web: config.native_web.clone(),
            ephemeral_tail_notes: Vec::new(),
        }
    };

    for attempt in 0..2u8 {
        if cancel.is_cancelled() {
            return None;
        }
        let rx_res = match tokio::time::timeout(
            SEMANTIC_COMPACTION_TIMEOUT,
            provider.stream(req.clone(), cancel.clone()),
        )
        .await
        {
            Ok(Ok(rx)) => rx,
            Ok(Err(_)) | Err(_) => {
                if attempt == 0 && !cancel.is_cancelled() {
                    tokio::time::sleep(Duration::from_millis(1_500)).await;
                    continue;
                }
                return None;
            }
        };
        let mut rx = rx_res;

        let mut final_text = String::new();
        let mut usage = Usage::default();
        let mut failed = false;
        let mut retryable_err = false;
        let mut rejected_incomplete = false;

        let drain = async {
            while let Some(ev) = rx.recv().await {
                match ev {
                    StreamEvent::Done {
                        message,
                        usage: u,
                        stop_reason,
                        ..
                    } => {
                        // Never persist a length-truncated or tool-calling summary.
                        if stop_reason == StopReason::MaxTokens || message.has_tool_use() {
                            rejected_incomplete = true;
                        } else {
                            final_text = message.text_concat();
                            usage = u;
                        }
                    }
                    StreamEvent::Error { retryable, .. } => {
                        failed = true;
                        retryable_err = retryable;
                    }
                    _ => {}
                }
            }
        };

        if tokio::time::timeout(SEMANTIC_COMPACTION_TIMEOUT, drain)
            .await
            .is_err()
        {
            if attempt == 0 && !cancel.is_cancelled() {
                tokio::time::sleep(Duration::from_millis(1_500)).await;
                continue;
            }
            return None;
        }
        if failed {
            if retryable_err && attempt == 0 && !cancel.is_cancelled() {
                tokio::time::sleep(Duration::from_millis(1_500)).await;
                continue;
            }
            return None;
        }
        if rejected_incomplete {
            return None;
        }

        let trimmed = final_text.trim();
        if trimmed.len() < 40 {
            return None;
        }
        return Some((trimmed.to_string(), usage));
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute_compaction(
    mode: CompactionMode,
    config: &CarnotConfig,
    provider: &Arc<dyn Provider>,
    tools: &Arc<ToolRegistry>,
    modules: &[Arc<dyn Module>],
    session: &Mutex<SessionStore>,
    metrics: &Mutex<ThermodynamicMetrics>,
    total_cost_micros: &std::sync::atomic::AtomicU64,
    events: &crate::EventLog,
    raw_user_goal: &Mutex<Option<String>>,
    user_instructions: &Mutex<Vec<String>>,
    seg: &SegmentTracker,
    custom_focus: Option<&str>,
    cancel: CancellationToken,
) -> Result<(u64, u64, bool)> {
    // Guard against re-compacting an already-compacted single-summary session.
    let (pre_tokens, pre_msg_count) = {
        let s = session.lock().await;
        let tok = s.effective_context_tokens();
        let cnt = s.active_messages().len();
        if cnt <= 1
            && s.active_messages().first().is_some_and(|m| {
                m.text_concat()
                    .trim_start()
                    .starts_with("[carnot:compaction ")
            })
        {
            return Ok((tok, tok, false));
        }
        (tok, cnt)
    };

    // Invoke `Module::before_compact` hooks (`EmergencyOverflow` cannot be vetoed)
    let mut focus_parts: Vec<String> = Vec::new();
    if let Some(f) = custom_focus.map(str::trim).filter(|s| !s.is_empty()) {
        focus_parts.push(f.to_string());
    }
    if !modules.is_empty() {
        let hook_ctx = CompactionHookContext {
            mode: mode.as_str().to_string(),
            tokens_before: pre_tokens,
            message_count: pre_msg_count,
            custom_focus: custom_focus.map(str::to_string),
        };
        for m in modules {
            match m.before_compact(&hook_ctx).await {
                Ok(CompactionHookAction::Continue) => {}
                Ok(CompactionHookAction::AddFocus(extra)) => {
                    let t = extra.trim();
                    if !t.is_empty() {
                        focus_parts.push(t.to_string());
                    }
                }
                Ok(CompactionHookAction::Cancel { reason }) => {
                    if mode != CompactionMode::EmergencyOverflow {
                        tracing::info!(module = %m.name(), %reason, "compaction cancelled by module hook");
                        return Ok((pre_tokens, pre_tokens, false));
                    }
                }
                Err(e) => {
                    tracing::warn!(module = %m.name(), error = %e, "before_compact hook failed (fail-open)");
                }
            }
        }
    }
    let effective_focus = if focus_parts.is_empty() {
        None
    } else {
        Some(focus_parts.join("\n"))
    };

    let git_facts = collect_git_facts(config.workspace_root.clone()).await;

    // Tier 1: Try cache-warm semantic LLM handoff for Soft & Manual modes
    // (Skip LLM call in EmergencyOverflow since the prompt already exceeds the physical window!)
    let semantic_handoff = if mode != CompactionMode::EmergencyOverflow {
        if let Some((text, usage)) = try_semantic_llm_handoff(
            config,
            provider,
            tools,
            session,
            effective_focus.as_deref(),
            cancel,
        )
        .await
        {
            // Account for compaction usage accurately in session & metrics
            let turn_num = {
                let mut m = metrics.lock().await;
                m.requests += 1;
                m.context_auc_tokens += usage.total_context_tokens();
                if usage.cache_read_tokens > 0 {
                    m.prefix_cache_hits += 1;
                } else {
                    m.prefix_cache_misses += 1;
                }
                m.total_usage.add_assign(&usage);
                crate::add_cost_micros(total_cost_micros, usage.cost_usd);
                m.turns
            };
            let _ = session.lock().await.record_usage(turn_num, usage.clone());
            events.push(AgentEvent::UsageRecorded {
                turn: turn_num,
                usage,
                rate_limits: Vec::new(),
            });
            Some(text)
        } else {
            None
        }
    } else {
        None
    };

    let used_semantic_llm = semantic_handoff.is_some();

    // Tier 0: Gather sacred verbatim user objective & follow-up instructions
    let captured_goal = raw_user_goal.lock().await.clone();
    let followups = user_instructions.lock().await.clone();

    let mut s = session.lock().await;
    let orig_goal = captured_goal.unwrap_or_else(|| {
        s.active_messages()
            .first()
            .map(|m| clip_anchor_text(&m.text_concat(), MAX_ANCHORED_GOAL_CHARS))
            .unwrap_or_default()
    });
    let last_assistant_note = s
        .active_messages()
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant && m.has_text())
        .map(|m| clip_anchor_text(&m.text_concat(), 1_600));

    // Assemble the final zero-loss compacted state message
    let mut summary = String::with_capacity(4096);
    summary.push_str(&format!(
        "[carnot:compaction mode={} semantic_llm={} session={}]\n",
        mode.as_str(),
        used_semantic_llm,
        s.session_id
    ));
    summary.push_str(
        "Older turns were compacted (full verbatim audit log remains on disk in session JSONL).\n",
    );

    if !orig_goal.is_empty() {
        summary.push_str("\n[carnot:anchored-objective]\n");
        summary.push_str(&clip_anchor_text(&orig_goal, MAX_ANCHORED_GOAL_CHARS));
        summary.push('\n');
    }
    if !followups.is_empty() {
        summary.push_str("\n[carnot:user-followups]\n");
        for (idx, ins) in followups.iter().enumerate() {
            summary.push_str(&format!(
                "### Follow-up {}\n{}\n",
                idx + 1,
                clip_anchor_text(ins, MAX_FOLLOWUP_INSTRUCTION_CHARS)
            ));
        }
    }

    if let Some(ref llm_handoff) = semantic_handoff {
        summary.push_str("\n[carnot:task-handoff]\n");
        summary.push_str(llm_handoff);
        summary.push('\n');
    } else if let Some(ref note) = last_assistant_note {
        summary.push_str("\n[carnot:latest-assistant-note]\n");
        summary.push_str(note);
        summary.push('\n');
    }

    // Ground-truth deterministic harness footer (restorable pointers — never hallucinated!)
    summary.push_str("\n[carnot:verified-workspace-state]\n");
    if let Some(ref sp) = s.path {
        summary.push_str(&format!("Full verbatim session log: {}\n", sp.display()));
    }
    if let Some(ref st) = git_facts.status_short {
        summary.push_str(&format!("git status --short:\n{st}\n"));
    }
    if let Some(ref ds) = git_facts.diff_stat {
        summary.push_str(&format!("git diff --stat:\n{ds}\n"));
    }
    let (modified_files, read_only_files) = extract_file_ops_from_session(s.active_messages(), seg);
    if !modified_files.is_empty() {
        summary.push_str(&format!("Files edited: {}\n", modified_files.join(", ")));
    }
    if !read_only_files.is_empty() {
        summary.push_str(&format!(
            "Read-only inspected files (all key discoveries are already captured in [carnot:task-handoff] above; only re-read a file if you need to edit it or inspect an uncaptured detail): {}\n",
            read_only_files.join(", ")
        ));
    }
    if !seg.handles_created.is_empty() {
        summary.push_str(&format!(
            "Lossless CAS output handles (inspect with Read/Grep): {}\n",
            seg.handles_created.join(", ")
        ));
    }
    if !seg.failed_attempts.is_empty() {
        summary.push_str("Observed failed attempts (do NOT repeat):\n");
        for f in seg.failed_attempts.iter().take(10) {
            summary.push_str(&format!("- {f}\n"));
        }
    }
    summary.push_str(
        "Continue immediately from [carnot:task-handoff] without redundantly re-reading already-summarized files.",
    );

    let last_is_fresh_user_prompt = s
        .active_messages()
        .last()
        .is_some_and(|m| m.role == Role::User && m.has_text() && !m.has_tool_result());

    let (min_tail, max_tail, budget) = match mode {
        CompactionMode::EmergencyOverflow => {
            (2, 4, config.compaction_tail_budget_tokens.min(8_000))
        }
        CompactionMode::Soft | CompactionMode::Manual => {
            if used_semantic_llm {
                // Semantic LLM just synthesized the full conversation including the latest tool
                // outputs. Keep small recent turns up to `budget`, and always keep an active
                // fresh user prompt (`min_tail = 1`), without dragging giant already-summarized
                // tool blobs past compaction.
                let min_t = usize::from(last_is_fresh_user_prompt);
                (min_t, 6, config.compaction_tail_budget_tokens)
            } else {
                (2, 6, config.compaction_tail_budget_tokens)
            }
        }
    };

    let (tokens_before, tokens_after) =
        s.apply_compaction_budgeted(summary, budget, min_tail, max_tail)?;
    drop(s);
    tools.reset_read_tracker();
    {
        let mut met = metrics.lock().await;
        met.compactions_applied += 1;
        met.heat_shed_tokens += tokens_before.saturating_sub(tokens_after);
    }

    events.push(AgentEvent::CompactionApplied {
        mode: mode.as_str().to_string(),
        tokens_before,
        tokens_after,
        semantic_llm: used_semantic_llm,
    });

    Ok((tokens_before, tokens_after, used_semantic_llm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_proactive_trigger_clamps_to_model_window() {
        let mut cfg = CarnotConfig::new(std::path::PathBuf::from("/tmp"));
        cfg.context_window = 1_000_000;
        cfg.max_output_tokens = 64_000;
        cfg.compaction_reserve_tokens = 16_384;
        cfg.soft_compaction_tokens = Some(220_000);
        assert_eq!(effective_proactive_trigger(&cfg), 220_000);

        cfg.context_window = 200_000;
        cfg.max_output_tokens = 32_768;
        // 200_000 - 32_768 - 16_384 = 150_848
        assert_eq!(effective_proactive_trigger(&cfg), 150_848);

        cfg.soft_compaction_tokens = None;
        assert_eq!(effective_proactive_trigger(&cfg), 150_848);
    }

    #[test]
    fn clip_anchor_preserves_full_normal_prompt_and_head_tail_of_giant_prompt() {
        let short = "Requirement A\nRequirement B\nExact path: /app/out.json";
        assert_eq!(clip_anchor_text(short, 24_000), short);

        let giant = format!(
            "{}MIDDLE_FILLER{}FINAL_TOLERANCE=1e-9",
            "A".repeat(200),
            "B".repeat(500)
        );
        let clipped = clip_anchor_text(&giant, 300);
        assert!(clipped.starts_with("AAAA"));
        assert!(clipped.ends_with("FINAL_TOLERANCE=1e-9"));
    }

    #[test]
    fn hydrate_anchors_survives_resume_and_double_compaction() {
        let msgs = vec![
            Arc::new(Message::user_text("Phase 1: engine_version=carnot-v4.2")),
            Arc::new(Message::assistant_text("Done Phase 1")),
            Arc::new(Message::user_text("Phase 2:\nLine A\nLine B")),
        ];
        let (goal, followups) = hydrate_anchors_from_session(&msgs);
        assert_eq!(goal.as_deref(), Some("Phase 1: engine_version=carnot-v4.2"));
        assert_eq!(followups, vec!["Phase 2:\nLine A\nLine B".to_string()]);

        // Simulate resuming AFTER a prior compaction and adding Phase 3
        let compacted = Message::system_note(
            "[carnot:compaction mode=soft semantic_llm=true session=s1]\n\
Older turns were compacted.\n\n\
[carnot:anchored-objective]\n\
Phase 1: engine_version=carnot-v4.2\n\n\
[carnot:user-followups]\n\
### Follow-up 1\n\
Phase 2:\nLine A\nLine B\n\n\
[carnot:task-handoff]\n\
## 1. Exact Constraints\n- ok\n",
        );
        let after_msgs = vec![
            Arc::new(compacted),
            Arc::new(Message::user_text("Phase 3: export CSV")),
        ];
        let (goal2, followups2) = hydrate_anchors_from_session(&after_msgs);
        assert_eq!(
            goal2.as_deref(),
            Some("Phase 1: engine_version=carnot-v4.2")
        );
        assert_eq!(
            followups2,
            vec![
                "Phase 2:\nLine A\nLine B".to_string(),
                "Phase 3: export CSV".to_string(),
            ]
        );
    }

    #[test]
    fn compaction_effort_caps_at_medium_only_when_cache_safe() {
        assert_eq!(
            compaction_effort_for_provider(ReasoningEffort::Max, true),
            ReasoningEffort::Medium
        );
        assert_eq!(
            compaction_effort_for_provider(ReasoningEffort::High, true),
            ReasoningEffort::Medium
        );
        assert_eq!(
            compaction_effort_for_provider(ReasoningEffort::Low, true),
            ReasoningEffort::Low
        );
        // Budget-thinking models preserve exact session effort so cache is never busted
        assert_eq!(
            compaction_effort_for_provider(ReasoningEffort::Max, false),
            ReasoningEffort::Max
        );
    }

    #[test]
    fn file_ops_ledger_survives_resume_and_separates_modified_from_read_only() {
        let prior = Message::system_note(
            "[carnot:compaction mode=soft semantic_llm=true session=s1]\n\n\
[carnot:verified-workspace-state]\n\
Files edited: src/a.rs\n\
Read-only inspected files (details above): docs/spec.md, src/b.rs\n",
        );
        let turn = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "Read".into(),
                    input: serde_json::json!({"file_path": "src/c.rs"}),
                    invalid_json: None,
                    provider_meta: None,
                    origin: None,
                },
                ContentBlock::ToolUse {
                    id: "t2".into(),
                    name: "Edit".into(),
                    input: serde_json::json!({"file_path": "src/b.rs"}),
                    invalid_json: None,
                    provider_meta: None,
                    origin: None,
                },
            ],
        };
        let msgs = vec![Arc::new(prior), Arc::new(turn)];
        let (modified, read_only) =
            extract_file_ops_from_session(&msgs, &SegmentTracker::default());
        assert_eq!(modified, vec!["src/a.rs", "src/b.rs"]);
        assert_eq!(read_only, vec!["docs/spec.md", "src/c.rs"]);
    }
}
