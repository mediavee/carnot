//! Pure formatting, diff-preview, and git-branch display helpers for `carnot-tui`.

use crate::markdown::truncate_display;
use crate::theme::Palette;
use carnot::ReasoningEffort;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use serde_json::Value;
use similar::{ChangeTag, TextDiff};
use std::path::{Path, PathBuf};

pub(crate) fn shorten_path(p: &Path) -> String {
    let full = p.display().to_string();
    if let Some(home) = std::env::var_os("HOME") {
        let h = PathBuf::from(home).display().to_string();
        if let Some(rest) = full.strip_prefix(&h) {
            return format!("~{rest}");
        }
    }
    full
}

pub(crate) fn clip_one_line(s: &str, max_chars: usize) -> String {
    let first = s
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    let count = first.chars().count();
    if count <= max_chars {
        first.to_string()
    } else {
        let head: String = first.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{head}…")
    }
}

pub(crate) fn summarize_tool_call(tool: &str, input: &Value) -> String {
    let (effective_tool, effective_input) = if tool == "CallDeferredTool" {
        let inner_name = input
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("CallDeferredTool");
        let inner_args = input.get("arguments").unwrap_or(input);
        (inner_name, inner_args)
    } else {
        (tool, input)
    };

    let detail = match effective_tool {
        "Read" | "Edit" | "Write" => effective_input
            .get("file_path")
            .or_else(|| effective_input.get("path"))
            .and_then(Value::as_str)
            .map(|s| clip_one_line(s, 72)),
        "Bash" => effective_input
            .get("command")
            .and_then(Value::as_str)
            .map(|s| clip_one_line(s, 72)),
        "CodeMode" => effective_input
            .get("code")
            .and_then(Value::as_str)
            .map(|s| {
                let first = s
                    .lines()
                    .map(str::trim)
                    .find(|l| !l.is_empty() && !l.starts_with("```"))
                    .unwrap_or("js");
                clip_one_line(first, 64)
            }),
        "Grep" => effective_input
            .get("pattern")
            .and_then(Value::as_str)
            .map(|s| format!("/{}/", clip_one_line(s, 56))),
        "WebFetch" => effective_input
            .get("url")
            .and_then(Value::as_str)
            .map(|s| clip_one_line(s, 72)),
        "WebSearch" | "ToolSearch" | "MemorySearch" => effective_input
            .get("query")
            .and_then(Value::as_str)
            .map(|s| clip_one_line(s, 64)),
        "MemorySave" => effective_input
            .get("title")
            .or_else(|| effective_input.get("id"))
            .and_then(Value::as_str)
            .map(|s| clip_one_line(s, 64)),
        "LspQuery" => {
            let op = effective_input
                .get("action")
                .and_then(Value::as_str)
                .unwrap_or("query");
            let fp = effective_input
                .get("file_path")
                .and_then(Value::as_str)
                .unwrap_or("");
            Some(clip_one_line(&format!("{op} {fp}"), 72))
        }
        "Subagent" => {
            let mut tags = Vec::new();
            if let Some(n) = effective_input
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                tags.push(n.to_string());
            }
            if effective_input
                .get("land")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                tags.push("land".to_string());
            }
            if effective_input
                .get("fork")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                tags.push("fork".to_string());
            }
            if let Some(m) = effective_input
                .get("model")
                .or_else(|| effective_input.get("provider"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                tags.push(m.to_string());
            }
            let body = effective_input
                .get("task")
                .or_else(|| effective_input.get("job"))
                .and_then(Value::as_str)
                .map(|s| clip_one_line(s, 56))
                .unwrap_or_default();
            if tags.is_empty() {
                (!body.is_empty()).then_some(body)
            } else if body.is_empty() {
                Some(format!("[{}]", tags.join(" · ")))
            } else {
                Some(format!("[{}] {body}", tags.join(" · ")))
            }
        }
        "SendMessage" => {
            let to = effective_input
                .get("to")
                .and_then(Value::as_str)
                .unwrap_or("?");
            let msg = effective_input
                .get("message")
                .and_then(Value::as_str)
                .map(|s| clip_one_line(s, 52))
                .unwrap_or_default();
            Some(format!("→ {to}: {msg}"))
        }
        _ => None,
    };

    let worker_prefix = input
        .get("_carnot_worker")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(|w| format!("[worker:{w}] "))
        .unwrap_or_default();

    match detail {
        Some(d) if !d.is_empty() => format!("{worker_prefix}{effective_tool} {d}"),
        _ => format!("{worker_prefix}{effective_tool}"),
    }
}

pub(crate) fn fmt_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

pub(crate) fn parse_effort(s: &str) -> Option<ReasoningEffort> {
    match s.trim().to_ascii_lowercase().as_str() {
        "low" => Some(ReasoningEffort::Low),
        "medium" | "med" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::Xhigh),
        "max" => Some(ReasoningEffort::Max),
        _ => None,
    }
}

pub(crate) fn fmt_duration_ms(ms: u64) -> String {
    if ms >= 1000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        format!("{ms}ms")
    }
}

pub(crate) fn extract_edit_diff(tool: &str, input: &Value) -> Option<(String, String)> {
    if tool != "Edit" {
        return None;
    }
    if let (Some(old_s), Some(new_s)) = (
        input.get("old_string").and_then(Value::as_str),
        input.get("new_string").and_then(Value::as_str),
    ) && old_s != new_s
    {
        return Some((old_s.to_string(), new_s.to_string()));
    }
    if let Some(arr) = input.get("edits").and_then(Value::as_array) {
        let mut old_parts = Vec::new();
        let mut new_parts = Vec::new();
        for item in arr {
            let o = item.get("old_string").and_then(Value::as_str).unwrap_or("");
            let n = item.get("new_string").and_then(Value::as_str).unwrap_or("");
            if o != n {
                old_parts.push(o);
                new_parts.push(n);
            }
        }
        if !old_parts.is_empty() {
            return Some((old_parts.join("\n"), new_parts.join("\n")));
        }
    }
    None
}

pub(crate) fn count_diff_stats(old_s: &str, new_s: &str) -> (usize, usize) {
    let diff = TextDiff::from_lines(old_s, new_s);
    let mut adds = 0usize;
    let mut dels = 0usize;
    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Insert => adds += 1,
            ChangeTag::Delete => dels += 1,
            ChangeTag::Equal => {}
        }
    }
    (adds, dels)
}

pub(crate) fn render_diff_preview(
    old_s: &str,
    new_s: &str,
    max_lines: usize,
    content_w: usize,
    p: &Palette,
    out: &mut Vec<Line<'static>>,
) {
    let diff = TextDiff::from_lines(old_s, new_s);
    let mut changed: Vec<(char, String, Style)> = Vec::new();
    for change in diff.iter_all_changes() {
        let line_text = change.value().trim_end_matches(['\r', '\n']).to_string();
        match change.tag() {
            ChangeTag::Delete => changed.push(('-', line_text, p.danger())),
            ChangeTag::Insert => changed.push(('+', line_text, p.success())),
            ChangeTag::Equal => {}
        }
    }
    let total = changed.len();
    for (sign, text, style) in changed.into_iter().take(max_lines) {
        out.push(Line::from(vec![
            Span::styled("    │ ", p.border()),
            Span::styled(
                format!(
                    "{sign} {}",
                    truncate_display(&text, content_w.saturating_sub(10))
                ),
                style,
            ),
        ]));
    }
    if total > max_lines {
        out.push(Line::from(vec![
            Span::styled("    │ ", p.border()),
            Span::styled(
                format!("… ({} more diff lines — ^O to expand)", total - max_lines),
                p.muted(),
            ),
        ]));
    }
}

pub(crate) fn detect_git_branch(workspace: &Path) -> Option<String> {
    let mut dir = workspace.to_path_buf();
    loop {
        let git_path = dir.join(".git");
        if git_path.is_dir() {
            let head = std::fs::read_to_string(git_path.join("HEAD")).ok()?;
            return parse_git_head(&head);
        }
        if git_path.is_file() {
            let content = std::fs::read_to_string(&git_path).ok()?;
            let rel = content.trim().strip_prefix("gitdir:")?.trim();
            let resolved = if Path::new(rel).is_absolute() {
                PathBuf::from(rel)
            } else {
                dir.join(rel)
            };
            let head = std::fs::read_to_string(resolved.join("HEAD")).ok()?;
            return parse_git_head(&head);
        }
        if !dir.pop() {
            return None;
        }
    }
}

pub(crate) fn parse_git_head(head: &str) -> Option<String> {
    let trimmed = head.trim();
    if let Some(r) = trimmed.strip_prefix("ref: refs/heads/") {
        Some(r.to_string())
    } else if trimmed.chars().count() >= 7 {
        Some(trimmed.chars().take(7).collect())
    } else {
        None
    }
}
