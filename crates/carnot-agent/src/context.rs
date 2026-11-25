//! Context projection helpers, prefix fingerprinting, calibrated token breakdowns,
//! date tagging, and repetition-loop detection.

use carnot_core::{
    CarnotConfig, ContentBlock, ContextBreakdown, Message, NativeWebConfig, Role, ToolResultBlock,
    ToolSpec,
};
use carnot_ext::prefix_guard::static_system_prompt;
use carnot_tools::ToolRegistry;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Formats a Unix timestamp (seconds since 1970-01-01 UTC) as `YYYY-MM-DD` using Howard
/// Hinnant's civil-from-days algorithm (zero external date crates, exact for all Gregorian dates).
pub(crate) fn format_utc_date_ymd(unix_secs: u64) -> String {
    let z = (unix_secs / 86_400).cast_signed() + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe.cast_signed() + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = y + i64::from(m <= 2);
    format!("{year:04}-{m:02}-{d:02}")
}

/// Returns today's UTC calendar date as `YYYY-MM-DD`.
pub(crate) fn current_utc_date_ymd() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format_utc_date_ymd(secs)
}

/// Extracts the most recent `[date: YYYY-MM-DD UTC]` (or legacy `[date: YYYY-MM-DD]`) line
/// recorded in `Role::User` messages, ignoring assistant text so models cannot spoof the tag.
pub(crate) fn last_date_tag_in_messages(messages: &[Arc<Message>]) -> Option<String> {
    for msg in messages.iter().rev() {
        if msg.role != Role::User {
            continue;
        }
        for block in msg.content.iter().rev() {
            if let ContentBlock::Text { text } = block {
                for line in text.lines().rev() {
                    let trimmed = line.trim();
                    if let Some(rest) = trimmed.strip_prefix("[date: ")
                        && let Some(inner) = rest.strip_suffix(']')
                    {
                        let date = inner.strip_suffix(" UTC").unwrap_or(inner);
                        let b = date.as_bytes();
                        if b.len() == 10
                            && b[4] == b'-'
                            && b[7] == b'-'
                            && b[0..4].iter().all(u8::is_ascii_digit)
                            && b[5..7].iter().all(u8::is_ascii_digit)
                            && b[8..10].iter().all(u8::is_ascii_digit)
                        {
                            return Some(date.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}


#[derive(Debug, Clone)]
pub(crate) struct PrefixFingerprint {
    pub(crate) provider: String,
    pub(crate) model: String,
    pub(crate) static_hash: String,
    pub(crate) msg_count: usize,
    pub(crate) msg_prefix_hash: String,
}

pub(crate) fn compute_wire_static_hash(
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

pub(crate) fn compute_messages_prefix_hash(messages: &[Arc<Message>]) -> String {
    let mut h = blake3::Hasher::new();
    for m in messages {
        if let Ok(b) = serde_json::to_vec(m.as_ref()) {
            h.update(&(b.len() as u64).to_le_bytes());
            h.update(&b);
        }
    }
    h.finalize().to_hex().to_string()
}

pub(crate) fn effective_static_system_prompt(config: &CarnotConfig) -> String {
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

pub(crate) fn apply_excluded_tools(tools: &mut ToolRegistry, excluded: &[String]) {
    for name in excluded {
        let trimmed = name.trim();
        if !trimmed.is_empty() {
            tools.remove(trimmed);
        }
    }
}


/// Conservative repetition-loop detector:
/// returns `true` if a long fragment (>=400 chars) is dominated by >=5 verbatim repeats of a single
/// normalized line or 60-char chunk covering >=50% of the output, preventing `MaxTokens` auto-continue
/// from stitching degenerate loops.
pub(crate) fn is_repetition_dominated(text: &str) -> bool {
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
pub(crate) fn clip_utf8_preview(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}


/// Estimates unscaled token counts per context category using the content-aware estimator
/// (`carnot_core::estimate_text_tokens`; images = 1,500 tokens).
/// Deferred tool schemas (`defer_loading == true`) cost 0 tokens until loaded.
pub(crate) fn estimate_context_breakdown(
    static_system_prompt: &str,
    wire_tools: &[ToolSpec],
    messages: &[Arc<Message>],
) -> ContextBreakdown {
    let mut out = ContextBreakdown {
        estimated: true,
        ..ContextBreakdown::default()
    };

    if !static_system_prompt.is_empty() {
        out.system = carnot_core::estimate_text_tokens(static_system_prompt);
    }

    for tool in wire_tools {
        if tool.defer_loading {
            continue;
        }
        let tok = carnot_core::estimate_text_tokens(&tool.name)
            + carnot_core::estimate_text_tokens(&tool.description)
            + carnot_core::estimate_text_tokens(&tool.input_schema.to_string());
        out.tools = out.tools.saturating_add(tok.max(1));
    }

    for msg in messages {
        for block in &msg.content {
            match block {
                ContentBlock::Text { text } => {
                    if msg.role == Role::User {
                        classify_user_text_into_breakdown(text, &mut out);
                    } else {
                        out.messages = out
                            .messages
                            .saturating_add(carnot_core::estimate_text_tokens(text) + 1);
                    }
                }
                ContentBlock::Thinking { thinking, .. } => {
                    out.messages = out
                        .messages
                        .saturating_add(carnot_core::estimate_text_tokens(thinking) + 1);
                }
                ContentBlock::Image { .. } => {
                    out.messages = out.messages.saturating_add(1_500);
                }
                ContentBlock::Opaque { raw, .. } => {
                    out.messages = out
                        .messages
                        .saturating_add(carnot_core::estimate_text_tokens(&raw.to_string()) + 1);
                }
                ContentBlock::ToolUse { name, input, .. } => {
                    let tok = carnot_core::estimate_text_tokens(name)
                        + carnot_core::estimate_text_tokens(&input.to_string())
                        + 4;
                    out.tool_results = out.tool_results.saturating_add(tok);
                }
                ContentBlock::ToolResult { content, .. } => {
                    for tb in content {
                        match tb {
                            ToolResultBlock::Text { text } => {
                                out.tool_results = out
                                    .tool_results
                                    .saturating_add(carnot_core::estimate_text_tokens(text) + 1);
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
        let est = carnot_core::estimate_text_tokens(segment) + 1;
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

