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
    AgentEvent, CarnotConfig, Message, ReasoningEffort, Result, Role, ThermodynamicMetrics, Usage,
};
use carnot_session::SessionStore;
use carnot_tools::ToolRegistry;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

const MAX_ANCHORED_GOAL_CHARS: usize = 24_000;
const MAX_FOLLOWUP_INSTRUCTION_CHARS: usize = 4_000;
const SEMANTIC_COMPACTION_MAX_OUTPUT_TOKENS: u32 = 6_144;
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

/// Computes the effective proactive compaction trigger (clamped safely below `context_window`).
pub fn effective_proactive_trigger(config: &CarnotConfig) -> u64 {
    let hard_trigger = config
        .context_window
        .saturating_sub(config.compaction_reserve_tokens);
    match config.soft_compaction_tokens {
        Some(soft) if soft > 0 => soft.min(hard_trigger),
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

/// Builds the concise, high-precision handoff prompt appended to the warm conversation prefix.
fn build_semantic_compaction_prompt(has_prior_compaction: bool) -> String {
    let merge_note = if has_prior_compaction {
        "An earlier `[carnot:compaction]` handoff appears at the top of this conversation. MERGE and preserve every still-relevant fact, constraint, file path, and refuted hypothesis from it into your updated handoff.\n"
    } else {
        ""
    };
    format!(
        "[carnot:compaction-request]\n\
Produce a dense, lossless engineering handoff so work can continue seamlessly after older turns are folded. Do NOT call any tools.\n\
{merge_note}\
Structure your response with these exact 6 markdown headings (omit nothing critical, stay crisp and factual):\n\
## 1. Exact Constraints & Edge Cases\n\
Every concrete requirement, output file path, schema/format rule, numerical tolerance, and edge case.\n\
## 2. Verified Workspace State & Measurements\n\
Files created/modified/deleted, commands/tests executed, exact passing vs failing counts, and measured numbers.\n\
## 3. Key Technical Discoveries & Invariants\n\
Exact `path:line` or symbol names, diagnosed root causes, data/API quirks, and invariants relied on.\n\
## 4. Dead Ends & Failed Approaches (DO NOT REPEAT)\n\
Approaches already tried that failed and the exact reason each failed.\n\
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

/// Attempts a single cache-warm LLM semantic handoff reusing the exact conversation prefix.
/// Returns `Some((handoff_text, usage))` on success, or `None` to trigger deterministic fallback.
async fn try_semantic_llm_handoff(
    config: &CarnotConfig,
    provider: &Arc<dyn Provider>,
    session: &Mutex<SessionStore>,
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
            .is_some_and(|m| m.text_concat().contains("[carnot:compaction]"));

        // Ensure trailing messages don't leave an unanswered `tool_use` before appending our User prompt
        let mut msgs = proj.messages;
        if msgs.last().is_some_and(|m| m.has_tool_use()) {
            return None;
        }
        msgs.push(Arc::new(Message::user_text(
            build_semantic_compaction_prompt(has_prior),
        )));

        ProviderRequest {
            model: config.model.clone(),
            static_system_prompt: proj.static_system_prompt,
            tools: proj.tools,
            prefix_hash: proj.prefix_hash,
            messages: msgs,
            checkpoint_index: proj.checkpoint_index,
            cache_ttl: config.cache_ttl,
            effort: ReasoningEffort::Medium,
            max_output_tokens: SEMANTIC_COMPACTION_MAX_OUTPUT_TOKENS.min(config.max_output_tokens),
        }
    };

    let rx_res = tokio::time::timeout(
        SEMANTIC_COMPACTION_TIMEOUT,
        provider.stream(req, cancel.clone()),
    )
    .await
    .ok()?
    .ok()?;
    let mut rx = rx_res;

    let mut final_text = String::new();
    let mut usage = Usage::default();
    let mut failed = false;

    let drain = async {
        while let Some(ev) = rx.recv().await {
            match ev {
                StreamEvent::Done {
                    message, usage: u, ..
                } => {
                    final_text = message.text_concat();
                    usage = u;
                }
                StreamEvent::Error { .. } => {
                    failed = true;
                }
                _ => {}
            }
        }
    };

    if tokio::time::timeout(SEMANTIC_COMPACTION_TIMEOUT, drain)
        .await
        .is_err()
        || failed
    {
        return None;
    }

    let trimmed = final_text.trim();
    if trimmed.len() < 40 {
        return None;
    }
    Some((trimmed.to_string(), usage))
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute_compaction(
    mode: CompactionMode,
    config: &CarnotConfig,
    provider: &Arc<dyn Provider>,
    tools: &Arc<ToolRegistry>,
    session: &Mutex<SessionStore>,
    metrics: &Mutex<ThermodynamicMetrics>,
    events: &crate::EventLog,
    raw_user_goal: &Mutex<Option<String>>,
    user_instructions: &Mutex<Vec<String>>,
    seg: &SegmentTracker,
    cancel: CancellationToken,
) -> Result<(u64, u64, bool)> {
    let git_facts = collect_git_facts(config.workspace_root.clone()).await;

    // Tier 1: Try cache-warm semantic LLM handoff for Soft & Manual modes
    // (Skip LLM call in EmergencyOverflow since the prompt already exceeds the physical window!)
    let semantic_handoff = if mode != CompactionMode::EmergencyOverflow {
        if let Some((text, usage)) =
            try_semantic_llm_handoff(config, provider, session, cancel).await
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
                m.turns
            };
            let _ = session.lock().await.record_usage(turn_num, usage.clone());
            events.push(AgentEvent::UsageRecorded {
                turn: turn_num,
                usage,
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
                "{}. {}\n",
                idx + 1,
                clip_anchor_text(ins, MAX_FOLLOWUP_INSTRUCTION_CHARS)
            ));
        }
    }

    if let Some(ref llm_handoff) = semantic_handoff {
        summary.push_str("\n[carnot:engineering-handoff]\n");
        summary.push_str(llm_handoff);
        summary.push('\n');
    } else if let Some(ref note) = last_assistant_note {
        summary.push_str("\n[carnot:latest-assistant-note]\n");
        summary.push_str(note);
        summary.push('\n');
    }

    // Ground-truth deterministic harness footer (never hallucinated!)
    summary.push_str("\n[carnot:verified-workspace-state]\n");
    if let Some(ref st) = git_facts.status_short {
        summary.push_str(&format!("git status --short:\n{st}\n"));
    }
    if let Some(ref ds) = git_facts.diff_stat {
        summary.push_str(&format!("git diff --stat:\n{ds}\n"));
    }
    if !seg.files_edited.is_empty() {
        summary.push_str(&format!("Files edited: {}\n", seg.files_edited.join(", ")));
    }
    if !seg.files_read.is_empty() {
        let recent_reads: Vec<&str> = seg
            .files_read
            .iter()
            .rev()
            .take(20)
            .map(String::as_str)
            .collect();
        summary.push_str(&format!(
            "Recently inspected files (re-read if needed; read-tracker reset): {}\n",
            recent_reads.join(", ")
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
        "Most recent turns follow verbatim below (thinking/images stripped; read-tracker reset).",
    );

    let (min_tail, max_tail, budget) = match mode {
        CompactionMode::EmergencyOverflow => {
            (4, 6, config.compaction_tail_budget_tokens.min(12_000))
        }
        CompactionMode::Soft | CompactionMode::Manual => {
            (6, 14, config.compaction_tail_budget_tokens)
        }
    };

    let (tokens_before, tokens_after) =
        s.apply_compaction_budgeted(summary, budget, min_tail, max_tail)?;
    drop(s);
    tools.reset_read_tracker();

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
        cfg.compaction_reserve_tokens = 16_384;
        cfg.soft_compaction_tokens = Some(220_000);
        assert_eq!(effective_proactive_trigger(&cfg), 220_000);

        cfg.context_window = 200_000;
        assert_eq!(effective_proactive_trigger(&cfg), 183_616);

        cfg.soft_compaction_tokens = None;
        assert_eq!(effective_proactive_trigger(&cfg), 183_616);
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
}
