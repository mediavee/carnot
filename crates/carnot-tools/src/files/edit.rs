//! Canonical `Edit` tool: atomic line-range & resilient 4-tier text patching with CAS Undo and syntax guard.

use super::path::{atomic_write_confined, effect_key_path, resolve_confined_path};
use crate::syntax::syntax_regression_warning;
use crate::{BoxFuture, Tool, UndoEntry, UndoKind};
use carnot_cas::CasStore;
use carnot_core::{CarnotError, Result, ToolEffects, ToolOutput, ToolSpec};
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
pub struct EditTool {
    workspace_root: PathBuf,
    extra_roots: Vec<PathBuf>,
    cas: Arc<CasStore>,
    undo_stack: Arc<Mutex<Vec<UndoEntry>>>,
    syntax_check: bool,
}

impl EditTool {
    pub fn new(
        workspace_root: PathBuf,
        extra_roots: Vec<PathBuf>,
        cas: Arc<CasStore>,
        undo_stack: Arc<Mutex<Vec<UndoEntry>>>,
        syntax_check: bool,
    ) -> Self {
        Self {
            workspace_root,
            extra_roots,
            cas,
            undo_stack,
            syntax_check,
        }
    }
}

#[derive(Debug)]
enum EditHunk {
    Text {
        old_string: String,
        new_string: String,
        replace_all: bool,
    },
    LineRange {
        start_line: usize,
        end_line: usize,
        new_string: String,
    },
}

fn parse_edit_hunk(item: &Value, label: &str) -> std::result::Result<EditHunk, String> {
    let has_start = item
        .get("start_line")
        .and_then(Value::as_u64)
        .is_some_and(|v| v >= 1);
    let has_end = item
        .get("end_line")
        .and_then(Value::as_u64)
        .is_some_and(|v| v >= 1);
    if has_start || has_end {
        if item
            .get("old_string")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
        {
            return Err(format!(
                "{label} cannot specify both `old_string` and `start_line`/`end_line`"
            ));
        }
        let start_line = item
            .get("start_line")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v >= 1)
            .ok_or_else(|| format!("{label} requires positive integer `start_line` (>= 1)"))?;
        let end_line = item
            .get("end_line")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v >= start_line)
            .ok_or_else(|| {
                format!("{label} requires integer `end_line` >= `start_line` ({start_line})")
            })?;
        let new_string = item
            .get("new_string")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{label} missing `new_string`"))?
            .to_string();
        return Ok(EditHunk::LineRange {
            start_line,
            end_line,
            new_string,
        });
    }

    let Some(os) = item.get("old_string").and_then(Value::as_str) else {
        return Err(format!(
            "{label} requires `old_string` (or `start_line`+`end_line`)"
        ));
    };
    if os.is_empty() {
        return Err(format!("{label}.old_string must not be empty"));
    }
    let Some(ns) = item.get("new_string").and_then(Value::as_str) else {
        return Err(format!("{label} missing `new_string`"));
    };
    let ra = item
        .get("replace_all")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(EditHunk::Text {
        old_string: os.to_string(),
        new_string: ns.to_string(),
        replace_all: ra,
    })
}

/// Applies one or more non-overlapping 1-based inclusive line-range replacements against the
/// ORIGINAL file line numbers (bottom-up so earlier hunks never shift later line numbers!).
fn apply_line_range_hunks(
    content: &str,
    ranges: &[(usize, usize, &str)],
    raw_path: &str,
) -> std::result::Result<String, String> {
    let uses_crlf = content.contains("\r\n");
    let eol = if uses_crlf { "\r\n" } else { "\n" };
    let has_trailing_nl = content.ends_with('\n');
    let body = content
        .strip_suffix("\r\n")
        .or_else(|| content.strip_suffix('\n'))
        .unwrap_or(content);
    let mut lines: Vec<String> = if body.is_empty() && !has_trailing_nl {
        Vec::new()
    } else {
        body.split('\n')
            .map(|l| l.trim_end_matches('\r').to_string())
            .collect()
    };
    let total = lines.len();

    let mut ordered: Vec<(usize, usize, usize, &str)> = Vec::with_capacity(ranges.len());
    for (idx, &(s, e, ns)) in ranges.iter().enumerate() {
        if s == 0 || e < s || e > total {
            return Err(format!(
                "line range L{s}-L{e} out of bounds for {raw_path} ({total} lines)"
            ));
        }
        ordered.push((s, e, idx + 1, ns));
    }
    ordered.sort_by_key(|&(s, e, _, _)| (s, e));
    for w in ordered.windows(2) {
        if w[1].0 <= w[0].1 {
            return Err(format!(
                "overlapping line ranges L{}-L{} and L{}-L{} in {raw_path}; merge them into one edit",
                w[0].0, w[0].1, w[1].0, w[1].1
            ));
        }
    }

    // Apply bottom-up (descending start_line) so all indices stay valid against original file!
    for &(s, e, _, ns) in ordered.iter().rev() {
        let repl: Vec<String> = if ns.is_empty() {
            Vec::new()
        } else {
            let trimmed = ns
                .strip_suffix("\r\n")
                .or_else(|| ns.strip_suffix('\n'))
                .unwrap_or(ns);
            trimmed
                .split('\n')
                .map(|l| l.trim_end_matches('\r').to_string())
                .collect()
        };
        lines.splice((s - 1)..e, repl);
    }

    let mut out = lines.join(eol);
    if has_trailing_nl && !lines.is_empty() {
        out.push_str(eol);
    }
    Ok(out)
}

#[derive(Debug)]
pub(super) enum HunkMatchMode {
    Exact,
    TrailingWsUnique,
    IndentAdjustedUnique(usize, usize),
}

impl Tool for EditTool {
    fn spec(&self) -> ToolSpec {
        crate::canonical_tool_specs()
            .into_iter()
            .find(|s| s.name == "Edit")
            .expect("Edit spec")
    }

    fn effects(&self, input: &Value) -> ToolEffects {
        match effect_key_path(&self.workspace_root, &self.extra_roots, input) {
            Some(p) => ToolEffects::write_path(p),
            None => ToolEffects::serial(),
        }
    }

    fn execute(
        &self,
        input: Value,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        let ws = self.workspace_root.clone();
        let extra = self.extra_roots.clone();
        let syntax_check = self.syntax_check;
        let cas = Arc::clone(&self.cas);
        let undo_stack = Arc::clone(&self.undo_stack);

        Box::pin(async move {
            let Some(raw_path) = input
                .get("file_path")
                .and_then(Value::as_str)
                .map(str::to_string)
            else {
                return Ok(ToolOutput::err_text("Edit requires `file_path`"));
            };

            let mut hunks = Vec::new();
            let parsed_edits_val: Option<Value> = match input.get("edits") {
                Some(Value::String(s)) => serde_json::from_str::<Value>(s).ok(),
                Some(Value::Null) | None => None,
                Some(Value::Array(arr)) if arr.is_empty() => {
                    // Strict-schema models often emit `"edits": []` alongside top-level `old_string` / `start_line`.
                    let has_top_level = input
                        .get("old_string")
                        .and_then(Value::as_str)
                        .is_some_and(|s| !s.is_empty())
                        || input
                            .get("start_line")
                            .and_then(Value::as_u64)
                            .is_some_and(|v| v >= 1);
                    if has_top_level {
                        None
                    } else {
                        Some(Value::Array(Vec::new()))
                    }
                }
                Some(v) => Some(v.clone()),
            };
            if let Some(val) = parsed_edits_val {
                let items: Vec<Value> = match val {
                    Value::Array(arr) => arr,
                    obj @ Value::Object(_) => vec![obj],
                    _ => {
                        return Ok(ToolOutput::err_text(
                            "Edit `edits` must be an array of edit objects",
                        ));
                    }
                };
                if items.is_empty() {
                    return Ok(ToolOutput::err_text("Edit `edits` array must not be empty"));
                }
                for (i, item) in items.iter().enumerate() {
                    match parse_edit_hunk(item, &format!("edits[{i}]")) {
                        Ok(h) => hunks.push(h),
                        Err(msg) => return Ok(ToolOutput::err_text(msg)),
                    }
                }
            } else {
                match parse_edit_hunk(&input, "Edit") {
                    Ok(h) => hunks.push(h),
                    Err(msg) => return Ok(ToolOutput::err_text(msg)),
                }
            }

            tokio::task::spawn_blocking(move || -> Result<ToolOutput> {
                let path = match resolve_confined_path(&ws, &extra, &raw_path) {
                    Ok(p) => p,
                    Err(msg) => return Ok(ToolOutput::err_text(msg)),
                };
                let before = match fs::read_to_string(&path) {
                    Ok(s) => s,
                    Err(e) => return Ok(ToolOutput::err_text(format!("Cannot read {}: {e}", path.display()))),
                };

                // Apply all hunks transactionally in memory BEFORE touching disk!
                let mut current = before.clone();
                let mut total_replacements = 0usize;
                let mut fuzzy_notes: Vec<String> = Vec::new();
                let mut fuzzy_count = 0usize;

                let has_line_range = hunks
                    .iter()
                    .any(|h| matches!(h, EditHunk::LineRange { .. }));
                let has_text = hunks.iter().any(|h| matches!(h, EditHunk::Text { .. }));
                if has_line_range && has_text {
                    return Ok(ToolOutput::err_text(
                        "Edit cannot mix `start_line`/`end_line` hunks with `old_string` hunks in the same `edits` array (line numbers would shift). Use either all line-range hunks or all `old_string` hunks.",
                    )
                    .with_details(json!({
                        "file_path": raw_path,
                        "edit_failed": true,
                    })));
                }
                if has_line_range {
                    let ranges: Vec<(usize, usize, &str)> = hunks
                        .iter()
                        .map(|h| match h {
                            EditHunk::LineRange {
                                start_line,
                                end_line,
                                new_string,
                            } => (*start_line, *end_line, new_string.as_str()),
                            EditHunk::Text { .. } => unreachable!(),
                        })
                        .collect();
                    match apply_line_range_hunks(&current, &ranges, &raw_path) {
                        Ok(next) => {
                            current = next;
                            total_replacements = ranges.len();
                        }
                        Err(diag_err) => {
                            return Ok(ToolOutput::err_text(diag_err).with_details(json!({
                                "file_path": raw_path,
                                "edit_failed": true,
                            })));
                        }
                    }
                } else {
                    for (h_idx, hunk) in hunks.iter().enumerate() {
                        match hunk {
                            EditHunk::LineRange { .. } => unreachable!(),
                            EditHunk::Text {
                                old_string,
                                new_string,
                                replace_all,
                            } => match apply_resilient_hunk(
                                &current,
                                old_string,
                                new_string,
                                *replace_all,
                                &raw_path,
                            ) {
                                Ok((next, count, mode)) => {
                                    current = next;
                                    total_replacements += count;
                                    match mode {
                                        HunkMatchMode::Exact => {}
                                        HunkMatchMode::TrailingWsUnique => {
                                            fuzzy_count += 1;
                                            fuzzy_notes.push(format!(
                                                "[hunk #{}: matched via trailing-whitespace-normalized unique match]",
                                                h_idx + 1
                                            ));
                                        }
                                        HunkMatchMode::IndentAdjustedUnique(s, e) => {
                                            fuzzy_count += 1;
                                            fuzzy_notes.push(format!(
                                                "[hunk #{}: matched via indentation-adjusted unique match at L{s}-L{e}]",
                                                h_idx + 1
                                            ));
                                        }
                                    }
                                }
                                Err(diag_err) => {
                                    let prefix = if hunks.len() > 1 {
                                        format!(
                                            "Atomic batch aborted at hunk #{}/{}: ",
                                            h_idx + 1,
                                            hunks.len()
                                        )
                                    } else {
                                        String::new()
                                    };
                                    return Ok(ToolOutput::err_text(format!("{prefix}{diag_err}"))
                                        .with_details(json!({
                                            "file_path": raw_path,
                                            "edit_failed": true,
                                        })));
                                }
                            },
                        }
                    }
                }

                // Store atomic CAS snapshot for instant undo (`undo_last_edit`) — fail-closed if snapshot fails.
                let undo_hash = match cas.put_bytes(before.as_bytes()) {
                    Ok(h) => h.hash,
                    Err(e) => {
                        return Ok(ToolOutput::err_text(format!(
                            "Edit aborted: failed to snapshot {} into CAS before writing: {e}",
                            raw_path
                        )));
                    }
                };
                if let Err(msg) = atomic_write_confined(&ws, &extra, &path, &current) {
                    return Ok(ToolOutput::err_text(msg));
                }
                let post_hash = blake3::hash(current.as_bytes()).to_hex().to_string();
                if let Ok(mut st) = undo_stack.lock() {
                    st.push(UndoEntry {
                        path: path.clone(),
                        kind: UndoKind::Restored {
                            prev_hash: undo_hash.clone(),
                            post_hash,
                        },
                    });
                }

                // Silent Tier-0 tree-sitter ERROR-node delta: 0 extra bytes on clean edit!
                let mut reply = format!("Edited {} ({total_replacements} replacement(s)).", raw_path);
                for note in &fuzzy_notes {
                    reply.push('\n');
                    reply.push_str(note);
                }
                if syntax_check
                    && let Some(warn) = syntax_regression_warning(&path, Some(&before), &current) {
                        reply.push('\n');
                        reply.push_str(&warn);
                    }

                Ok(ToolOutput::ok_text(reply).with_details(json!({
                    "file_path": raw_path,
                    "replacements": total_replacements,
                    "fuzzy_matches": fuzzy_count,
                    "undo_cas_hash": undo_hash,
                })))
            })
            .await
            .map_err(|e| CarnotError::Tool {
                tool: "Edit".into(),
                message: format!("blocking edit task failed: {e}"),
            })?
        })
    }
}

/// 4-tier resilient Edit matcher (exact -> line-trimmed -> whitespace-collapsed -> relative re-indent):
///   1. Exact match (with CRLF normalization)
///   2. Unique trailing-whitespace (`rstrip`) normalized match (NEVER applied if >1 match!)
///   3. Unique indentation-trimmed (`trim`) match with automatic relative re-indentation of `new_string`
///   4. On failure/ambiguity: returns closest 20-line numbered window + exact mismatch diagnosis!
pub(super) fn apply_resilient_hunk(
    content: &str,
    old_s: &str,
    new_s: &str,
    replace_all: bool,
    raw_path: &str,
) -> std::result::Result<(String, usize, HunkMatchMode), String> {
    let uses_crlf = content.contains("\r\n");
    let eol = if uses_crlf { "\r\n" } else { "\n" };

    // Tier 1: Exact match (with CRLF normalization for both old_s and new_s).
    let eff_old = if uses_crlf && !old_s.contains("\r\n") && !content.contains(old_s) {
        old_s.replace('\n', "\r\n")
    } else {
        old_s.to_string()
    };
    let eff_new = if uses_crlf && !new_s.contains("\r\n") && new_s.contains('\n') {
        new_s.replace('\n', "\r\n")
    } else {
        new_s.to_string()
    };
    let exact_count = content.matches(&eff_old).count();
    if exact_count > 1 && !replace_all {
        let lines = find_matching_line_numbers(content, &eff_old);
        return Err(format!(
            "old_string matched {exact_count} times in {raw_path} (at lines {lines:?}) — provide more surrounding context or set replace_all=true"
        ));
    }
    if exact_count > 0 {
        let after = if replace_all {
            content.replace(&eff_old, &eff_new)
        } else {
            content.replacen(&eff_old, &eff_new, 1)
        };
        return Ok((after, exact_count, HunkMatchMode::Exact));
    }

    // Tier 2 & 3 only apply when `replace_all == false` and match is STRICTLY UNIQUE!
    if !replace_all {
        // Strip a single trailing newline from `old_s`/`new_s` so trailing-LF needles match cleanly.
        let trimmed_old = old_s
            .strip_suffix("\r\n")
            .or_else(|| old_s.strip_suffix('\n'))
            .unwrap_or(old_s);
        let trimmed_new = if trimmed_old.len() < old_s.len() {
            new_s
                .strip_suffix("\r\n")
                .or_else(|| new_s.strip_suffix('\n'))
                .unwrap_or(new_s)
        } else {
            new_s
        };
        let has_trailing_nl = content.ends_with('\n');
        let content_body = content
            .strip_suffix("\r\n")
            .or_else(|| content.strip_suffix('\n'))
            .unwrap_or(content);
        let file_lines: Vec<&str> = if content_body.is_empty() {
            Vec::new()
        } else {
            content_body
                .split('\n')
                .map(|l| l.trim_end_matches('\r'))
                .collect()
        };
        let needle_lines: Vec<&str> = trimmed_old
            .split('\n')
            .map(|l| l.trim_end_matches('\r'))
            .collect();
        let n_len = needle_lines.len();

        if n_len > 0 && n_len <= file_lines.len() {
            // Tier 2: trailing-whitespace normalized (`trim_end`)
            let rstrip_hits: Vec<usize> = (0..=file_lines.len() - n_len)
                .filter(|&i| {
                    (0..n_len).all(|j| file_lines[i + j].trim_end() == needle_lines[j].trim_end())
                })
                .collect();
            if rstrip_hits.len() == 1 {
                let start = rstrip_hits[0];
                let repl_lines: Vec<String> = if new_s.is_empty() {
                    Vec::new()
                } else {
                    trimmed_new
                        .split('\n')
                        .map(|l| l.trim_end_matches('\r').to_string())
                        .collect()
                };
                let rebuilt = splice_lines(
                    &file_lines,
                    start,
                    start + n_len,
                    &repl_lines,
                    eol,
                    has_trailing_nl,
                );
                return Ok((rebuilt, 1, HunkMatchMode::TrailingWsUnique));
            }

            // Tier 3: indentation-trimmed (`trim`) unique match + relative re-indentation of `new_string`
            let non_trivial = n_len >= 2 || needle_lines[0].trim().len() >= 12;
            if non_trivial {
                let trim_hits: Vec<usize> = (0..=file_lines.len() - n_len)
                    .filter(|&i| {
                        (0..n_len).all(|j| file_lines[i + j].trim() == needle_lines[j].trim())
                    })
                    .collect();
                if trim_hits.len() == 1 {
                    let start = trim_hits[0];
                    let repl_lines = if new_s.is_empty() {
                        Vec::new()
                    } else {
                        reindent_replacement(
                            &file_lines[start..start + n_len],
                            &needle_lines,
                            trimmed_new,
                        )
                    };
                    let rebuilt = splice_lines(
                        &file_lines,
                        start,
                        start + n_len,
                        &repl_lines,
                        eol,
                        has_trailing_nl,
                    );
                    return Ok((
                        rebuilt,
                        1,
                        HunkMatchMode::IndentAdjustedUnique(start + 1, start + n_len),
                    ));
                } else if trim_hits.len() > 1 {
                    let hit_lines: Vec<usize> = trim_hits.iter().map(|i| i + 1).collect();
                    return Err(format!(
                        "old_string did not match verbatim in {raw_path}, and indentation-trimmed search found {} ambiguous candidates at lines {hit_lines:?} — add surrounding context.",
                        trim_hits.len()
                    ));
                }
            }
        }
    }

    // Tier 4: Rich anti-retry diagnostic with closest 20-line numbered window + root-cause hint!
    Err(build_edit_miss_diagnostic(content, old_s, raw_path))
}

fn splice_lines(
    file_lines: &[&str],
    start: usize,
    end: usize,
    repl: &[String],
    eol: &str,
    had_trailing_nl: bool,
) -> String {
    let mut out: Vec<&str> = Vec::with_capacity(file_lines.len() - (end - start) + repl.len());
    out.extend_from_slice(&file_lines[..start]);
    for r in repl {
        out.push(r.as_str());
    }
    out.extend_from_slice(&file_lines[end..]);
    let mut joined = out.join(eol);
    if had_trailing_nl && !joined.is_empty() {
        joined.push_str(eol);
    }
    joined
}

fn leading_ws(s: &str) -> &str {
    let idx = s.find(|c: char| !c.is_whitespace()).unwrap_or(s.len());
    &s[..idx]
}

fn reindent_replacement(matched_slice: &[&str], needle_lines: &[&str], new_s: &str) -> Vec<String> {
    // Anchor on the minimum-indented non-empty pair so outer closing braces / `else` lines
    // also get re-indented cleanly instead of failing `strip_prefix`.
    let (file_indent, needle_indent) = matched_slice
        .iter()
        .zip(needle_lines.iter())
        .filter(|(f, n)| !f.trim().is_empty() && !n.trim().is_empty())
        .min_by_key(|(_, n)| leading_ws(n).len())
        .map(|(f, n)| (leading_ws(f), leading_ws(n)))
        .unwrap_or(("", ""));

    new_s
        .split('\n')
        .map(|raw| {
            let line = raw.trim_end_matches('\r');
            if line.trim().is_empty() {
                return String::new();
            }
            if let Some(rest) = line.strip_prefix(needle_indent) {
                format!("{file_indent}{rest}")
            } else {
                line.to_string()
            }
        })
        .collect()
}

fn find_matching_line_numbers(content: &str, needle: &str) -> Vec<usize> {
    let mut lines = Vec::new();
    let mut offset = 0usize;
    while let Some(pos) = content[offset..].find(needle) {
        let abs = offset + pos;
        let line_no = content[..abs].bytes().filter(|&b| b == b'\n').count() + 1;
        lines.push(line_no);
        offset = abs + needle.len().max(1);
    }
    lines
}

fn build_edit_miss_diagnostic(content: &str, old_s: &str, raw_path: &str) -> String {
    let file_lines: Vec<&str> = content.lines().collect();
    let needle_lines: Vec<&str> = old_s
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();

    // Detect likely root cause
    let mut hints = Vec::new();
    if content.contains('\t') && !old_s.contains('\t') {
        hints.push("file uses tabs while old_string uses spaces");
    } else if !content.contains('\t') && old_s.contains('\t') {
        hints.push("file uses spaces while old_string uses tabs");
    }
    if content.contains("\r\n") && !old_s.contains("\r\n") {
        hints.push("file uses CRLF line endings");
    }

    if file_lines.is_empty() || needle_lines.is_empty() {
        return format!("old_string not found in {raw_path}");
    }

    // Score sliding windows to locate the closest region (token + line overlap)
    let first_anchor = needle_lines[0];
    let win_len = needle_lines.len().min(file_lines.len()).max(1);
    let mut best_idx = 0usize;
    let mut best_score = 0usize;
    for i in 0..file_lines.len() {
        let mut score = 0usize;
        for (j, nl) in needle_lines.iter().enumerate().take(8) {
            if let Some(fl) = file_lines.get(i + j) {
                let ft = fl.trim();
                if ft == *nl {
                    score += 10;
                } else if !nl.is_empty() && (ft.contains(nl) || nl.contains(ft) && ft.len() >= 4) {
                    score += 5;
                } else {
                    for tok in nl
                        .split(|c: char| !c.is_alphanumeric() && c != '_')
                        .filter(|t| t.len() >= 3)
                    {
                        if ft.contains(tok) {
                            score += 1;
                        }
                    }
                }
            }
        }
        if file_lines[i].trim().contains(first_anchor) {
            score += 6;
        }
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    let hint_str = if hints.is_empty() {
        String::new()
    } else {
        format!(" (hint: {})", hints.join("; "))
    };
    if best_score == 0 {
        return format!("old_string not found in {raw_path}{hint_str}.");
    }

    let start = best_idx.saturating_sub(3);
    let end = (best_idx + win_len + 4)
        .min(file_lines.len())
        .min(start + 20);
    let snippet: String = file_lines[start..end]
        .iter()
        .enumerate()
        .map(|(k, l)| format!("{}\t{l}", start + k + 1))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "old_string not found in {raw_path}{hint_str}.\nClosest matching region (L{}-L{}):\n{snippet}",
        start + 1,
        end
    )
}
