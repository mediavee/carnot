//! Canonical `Edit` tool: atomic line-range & resilient 4-tier text patching with CAS Undo and syntax guard.

use super::path::{
    atomic_write_confined, effect_key_path, reject_protected_git_write, resolve_confined_path,
    validate_regular_file_for_read,
};
use crate::syntax::{find_symbol_ranges, structural_outline, syntax_regression_warning};
use crate::tracker::{
    PathLockPool, ReadTracker, ViewedLine, hash_line, is_read_tracking_suppressed,
    split_logical_lines,
};
use crate::{BoxFuture, Tool, UndoEntry, UndoKind};
use carnot_cas::CasStore;
use carnot_core::{CarnotError, Result, ToolEffects, ToolOutput, ToolSpec};
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
pub struct EditTool {
    workspace_root: PathBuf,
    extra_roots: Vec<PathBuf>,
    cas: Arc<CasStore>,
    undo_stack: Arc<Mutex<Vec<UndoEntry>>>,
    tracker: Arc<Mutex<ReadTracker>>,
    path_locks: Arc<PathLockPool>,
    syntax_check: bool,
}

impl EditTool {
    pub fn new(
        workspace_root: PathBuf,
        extra_roots: Vec<PathBuf>,
        cas: Arc<CasStore>,
        undo_stack: Arc<Mutex<Vec<UndoEntry>>>,
        tracker: Arc<Mutex<ReadTracker>>,
        path_locks: Arc<PathLockPool>,
        syntax_check: bool,
    ) -> Self {
        Self {
            workspace_root,
            extra_roots,
            cas,
            undo_stack,
            tracker,
            path_locks,
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
    Symbol {
        symbol: String,
        new_string: String,
    },
}

fn parse_edit_hunk(item: &Value, label: &str) -> std::result::Result<EditHunk, String> {
    let sym = item
        .get("symbol")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let has_start = item
        .get("start_line")
        .and_then(Value::as_u64)
        .is_some_and(|v| v >= 1);
    let has_end = item
        .get("end_line")
        .and_then(Value::as_u64)
        .is_some_and(|v| v >= 1);
    let has_old = item
        .get("old_string")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty());

    if let Some(symbol) = sym {
        if has_start || has_end || has_old {
            return Err(format!(
                "{label} cannot combine `symbol` with `old_string` or `start_line`/`end_line`"
            ));
        }
        let new_string = item
            .get("new_string")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{label} missing `new_string`"))?
            .to_string();
        return Ok(EditHunk::Symbol {
            symbol: symbol.to_string(),
            new_string,
        });
    }

    if has_start || has_end {
        if has_old {
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
            "{label} requires `symbol`, `start_line`+`end_line`, or `old_string`"
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

/// Preserves original leading indentation for nested AST members (e.g. `impl` / `class` methods)
/// when the caller supplies an unindented replacement block, while leaving already-indented or
/// empty replacements untouched.
fn preserve_symbol_indentation(disk_lines: &[&str], start_line: usize, new_string: &str) -> String {
    if new_string.is_empty() || start_line == 0 || start_line > disk_lines.len() {
        return new_string.to_string();
    }
    let orig_line = disk_lines[start_line - 1];
    let orig_indent: String = orig_line
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect();
    if orig_indent.is_empty() {
        return new_string.to_string();
    }
    let first_nonempty = new_string.lines().find(|l| !l.trim().is_empty());
    if let Some(first) = first_nonempty
        && first.starts_with([' ', '\t'])
    {
        return new_string.to_string();
    }
    let trailing_nl = new_string.ends_with('\n');
    let body = new_string
        .strip_suffix("\r\n")
        .or_else(|| new_string.strip_suffix('\n'))
        .unwrap_or(new_string);
    let mut out = String::with_capacity(new_string.len() + orig_indent.len() * 4);
    for (idx, line) in body.split('\n').enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        let clean = line.trim_end_matches('\r');
        if !clean.trim().is_empty() {
            out.push_str(&orig_indent);
        }
        out.push_str(clean);
    }
    if trailing_nl {
        out.push('\n');
    }
    out
}

type OwnedLineRange = (usize, usize, String);

/// Resolves one or more AST `symbol` hunks into concrete 1-based `(start_line, end_line, replacement)`
/// ranges after verifying that `path` was read and has not changed externally on disk.
fn resolve_symbol_hunks(
    content: &str,
    symbols: &[(&str, &str)],
    path: &Path,
    raw_path: &str,
    tracker: &Mutex<ReadTracker>,
) -> std::result::Result<(Vec<OwnedLineRange>, Vec<String>), String> {
    // 1. Require disk-coherent Read (full file, symbol slice, or outline) before blind AST overwrite.
    let disk_digest = *blake3::hash(content.as_bytes()).as_bytes();
    let tracked_digest = tracker
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .disk_hash(path);
    match tracked_digest {
        None => {
            return Err(format!(
                "Cannot apply symbol Edit on {raw_path}: file has not been read in this session (or tracker was cleared after compaction). Call Read (with `outline: true`, `symbol`, or full file) first."
            ));
        }
        Some(seen) if seen != disk_digest => {
            return Err(format!(
                "Cannot apply symbol Edit on {raw_path}: file changed on disk since your last Read/Edit. Re-read {raw_path} first so external changes are not clobbered."
            ));
        }
        Some(_) => {}
    }

    let disk_lines = split_logical_lines(content);
    let mut resolved = Vec::with_capacity(symbols.len());
    let mut notes = Vec::with_capacity(symbols.len());

    for &(sym, new_str) in symbols {
        let matches = find_symbol_ranges(path, content, sym);
        match matches.as_slice() {
            [] => {
                if let Some(outline) = structural_outline(path, content) {
                    let preview: String = outline.lines().take(25).collect::<Vec<_>>().join("\n");
                    return Err(format!(
                        "Symbol `{sym}` not found in {raw_path}.\nAvailable symbols:\n{preview}"
                    ));
                }
                return Err(format!(
                    "Symbol `{sym}` cannot be resolved in {raw_path} (AST symbol edits support Rust, Python, TypeScript/JavaScript, and Go). Use `start_line`/`end_line` or `old_string` instead."
                ));
            }
            &[(s, e)] => {
                let adjusted = preserve_symbol_indentation(&disk_lines, s, new_str);
                resolved.push((s, e, adjusted));
                notes.push(format!("[symbol `{sym}` -> L{s}-L{e}]"));
            }
            many => {
                let spans: Vec<String> = many.iter().map(|(s, e)| format!("L{s}-L{e}")).collect();
                return Err(format!(
                    "Symbol `{sym}` is ambiguous in {raw_path} (matched {} locations: {}). Qualify with `Type::method` / `Class.method` or use `start_line`/`end_line`.",
                    many.len(),
                    spans.join(", ")
                ));
            }
        }
    }

    Ok((resolved, notes))
}

/// Formats a small numbered window of the current disk file around `target_s..=target_e` so
/// the model can immediately recover from a stale line-range edit without an extra Read turn.
fn format_current_window(disk_lines: &[&str], target_s: usize, target_e: usize) -> String {
    if disk_lines.is_empty() {
        return "[file is currently empty]".to_string();
    }
    let total = disk_lines.len();
    let win_s = target_s.saturating_sub(5).max(1).min(total);
    let win_e = target_e
        .saturating_add(5)
        .min(total)
        .min(win_s.saturating_add(29));
    let mut out = String::new();
    for (idx, line) in disk_lines[win_s - 1..win_e].iter().enumerate() {
        let _ = writeln!(out, "{}\t{line}", win_s + idx);
    }
    out.trim_end_matches('\n').to_string()
}

/// Implicitly anchors line-range edits against the model's last `Read` view stored in
/// [`ReadTracker`]:
///   1. Every requested line `s..=e` must have been read by the model in the current view.
///   2. If the disk content at `s..=e` matches the viewed line hashes, the hunk applies directly.
///   3. If the disk content shifted (e.g. an earlier `Edit` inserted/deleted lines), attempts a
///      unique relocation of the non-trivial viewed block (`>= 2` lines with `>= 4` non-ws chars,
///      or 1 line with `>= 12` non-ws chars) and emits a relocation note.
///   4. Otherwise rejects atomically with the current numbered window around `s..=e`.
type LineRangeRepl<'a> = (usize, usize, &'a str);

fn resolve_anchored_line_ranges<'a>(
    content: &str,
    ranges: &[LineRangeRepl<'a>],
    path: &Path,
    raw_path: &str,
    tracker: &Mutex<ReadTracker>,
) -> std::result::Result<(Vec<LineRangeRepl<'a>>, Vec<String>), String> {
    // 1. Validate requested ranges and reject overlaps before consulting disk/tracker.
    let mut req_sorted: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(s, e, _) in ranges {
        if s == 0 || e < s {
            return Err(format!("invalid line range L{s}-L{e} in {raw_path}"));
        }
        req_sorted.push((s, e));
    }
    req_sorted.sort_unstable();
    for w in req_sorted.windows(2) {
        if w[1].0 <= w[0].1 {
            return Err(format!(
                "overlapping line ranges L{}-L{} and L{}-L{} in {raw_path}; merge them into one edit",
                w[0].0, w[0].1, w[1].0, w[1].1
            ));
        }
    }

    // 2. Snapshot the model's viewed line hashes for every requested hunk under a brief lock.
    let viewed_per_hunk: Vec<Option<Vec<ViewedLine>>> = {
        let guard = tracker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ranges
            .iter()
            .map(|&(s, e, _)| guard.viewed_range(path, s, e))
            .collect()
    };

    for (&(s, e, _), viewed) in ranges.iter().zip(&viewed_per_hunk) {
        if viewed.is_none() {
            return Err(format!(
                "Cannot apply line-range Edit L{s}-L{e} on {raw_path}: lines were not previously read in this session (or tracker was cleared after compaction). Re-read the target range with Read or use `old_string`/`new_string`."
            ));
        }
    }

    // 3. Split current disk content identically to `apply_line_range_hunks`.
    let has_trailing_nl = content.ends_with('\n');
    let body = content
        .strip_suffix("\r\n")
        .or_else(|| content.strip_suffix('\n'))
        .unwrap_or(content);
    let disk_lines: Vec<&str> = if body.is_empty() && !has_trailing_nl {
        Vec::new()
    } else {
        body.split('\n').map(|l| l.trim_end_matches('\r')).collect()
    };
    let disk_hashes: Vec<u64> = disk_lines.iter().map(|l| hash_line(l)).collect();

    let mut resolved: Vec<(usize, usize, &'a str)> = Vec::with_capacity(ranges.len());
    let mut notes: Vec<String> = Vec::new();

    for (idx, (&(s, e, ns), viewed_opt)) in ranges.iter().zip(viewed_per_hunk).enumerate() {
        let Some(expected) = viewed_opt else {
            return Err(format!(
                "Cannot apply line-range Edit L{s}-L{e} on {raw_path}: lines were not previously read in this session."
            ));
        };
        let len = expected.len();

        // Direct hit at original coordinates.
        if e <= disk_hashes.len()
            && disk_hashes[s - 1..e]
                .iter()
                .zip(&expected)
                .all(|(&dh, vl)| dh == vl.hash)
        {
            resolved.push((s, e, ns));
            continue;
        }

        // Attempt unique relocation if the viewed block is non-trivial.
        let total_trimmed: usize = expected.iter().map(|v| v.trimmed_len as usize).sum();
        let nontrivial = (len >= 2 && total_trimmed >= 4) || (len == 1 && total_trimmed >= 12);

        let mut match_starts: Vec<usize> = Vec::new();
        if nontrivial && len <= disk_hashes.len() {
            for start_0 in 0..=disk_hashes.len() - len {
                if disk_hashes[start_0..start_0 + len]
                    .iter()
                    .zip(&expected)
                    .all(|(&dh, vl)| dh == vl.hash)
                {
                    match_starts.push(start_0);
                    if match_starts.len() >= 2 {
                        break;
                    }
                }
            }
        }

        if match_starts.len() == 1 {
            let new_s = match_starts[0] + 1;
            let new_e = new_s + len - 1;
            resolved.push((new_s, new_e, ns));
            notes.push(format!(
                "[hunk #{}: relocated L{s}-L{e} -> L{new_s}-L{new_e}]",
                idx + 1
            ));
        } else {
            let reason = if match_starts.len() >= 2 {
                format!(
                    "lines at L{s}-L{e} in {raw_path} shifted since your last Read and match multiple locations (ambiguous)"
                )
            } else if !nontrivial {
                format!(
                    "lines at L{s}-L{e} in {raw_path} changed or shifted since your last Read (block too short/trivial to relocate safely)"
                )
            } else {
                format!(
                    "lines at L{s}-L{e} in {raw_path} changed since your last Read (no matching block found)"
                )
            };
            let window = format_current_window(&disk_lines, s, e);
            return Err(format!(
                "Stale line-range Edit rejected: {reason}.\nCurrent lines around L{s}-L{e} ({} total lines):\n{window}\nRe-read the file or retry with updated line numbers / `old_string`.",
                disk_lines.len()
            ));
        }
    }

    // 4. Verify resolved ranges still do not overlap after any relocation.
    let mut res_sorted: Vec<(usize, usize)> = resolved.iter().map(|&(s, e, _)| (s, e)).collect();
    res_sorted.sort_unstable();
    for w in res_sorted.windows(2) {
        if w[1].0 <= w[0].1 {
            return Err(format!(
                "relocated line ranges L{}-L{} and L{}-L{} overlap in {raw_path}; re-read and merge them",
                w[0].0, w[0].1, w[1].0, w[1].1
            ));
        }
    }

    Ok((resolved, notes))
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
        let tracker = Arc::clone(&self.tracker);
        let path_locks = Arc::clone(&self.path_locks);

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
                    // Strict-schema models often emit `"edits": []` alongside top-level `old_string` / `start_line` / `symbol`.
                    let has_top_level = input
                        .get("old_string")
                        .and_then(Value::as_str)
                        .is_some_and(|s| !s.is_empty())
                        || input
                            .get("symbol")
                            .and_then(Value::as_str)
                            .is_some_and(|s| !s.trim().is_empty())
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

            let suppress_tracking = is_read_tracking_suppressed();
            let path_lock = effect_key_path(&ws, &extra, &input).map(|p| path_locks.lock_for(&p));
            let _path_guard = match path_lock {
                Some(ref m) => Some(m.lock().await),
                None => None,
            };

            tokio::task::spawn_blocking(move || -> Result<ToolOutput> {
                let path = match resolve_confined_path(&ws, &extra, &raw_path) {
                    Ok(p) => p,
                    Err(msg) => return Ok(ToolOutput::err_text(msg)),
                };
                if let Err(msg) = reject_protected_git_write(&ws, &path) {
                    return Ok(ToolOutput::err_text(msg));
                }
                if let Err(msg) = validate_regular_file_for_read(&path) {
                    return Ok(ToolOutput::err_text(msg));
                }
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
                let has_symbol = hunks.iter().any(|h| matches!(h, EditHunk::Symbol { .. }));
                let has_text = hunks.iter().any(|h| matches!(h, EditHunk::Text { .. }));
                if (has_line_range || has_symbol) && has_text {
                    return Ok(ToolOutput::err_text(
                        "Edit cannot mix `symbol` or `start_line`/`end_line` hunks with `old_string` hunks in the same `edits` array (line numbers would shift). Use either all structural/line-range hunks or all `old_string` hunks.",
                    )
                    .with_details(json!({
                        "file_path": raw_path,
                        "edit_failed": true,
                    })));
                }
                if has_line_range || has_symbol {
                    let mut owned_ranges: Vec<(usize, usize, String)> = Vec::new();

                    if has_line_range {
                        let ranges: Vec<(usize, usize, &str)> = hunks
                            .iter()
                            .filter_map(|h| match h {
                                EditHunk::LineRange {
                                    start_line,
                                    end_line,
                                    new_string,
                                } => Some((*start_line, *end_line, new_string.as_str())),
                                EditHunk::Symbol { .. } | EditHunk::Text { .. } => None,
                            })
                            .collect();
                        let (anchored_ranges, reloc_notes) = match resolve_anchored_line_ranges(
                            &current, &ranges, &path, &raw_path, &tracker,
                        ) {
                            Ok(ok) => ok,
                            Err(diag_err) => {
                                return Ok(ToolOutput::err_text(diag_err).with_details(json!({
                                    "file_path": raw_path,
                                    "edit_failed": true,
                                })));
                            }
                        };
                        fuzzy_notes.extend(reloc_notes);
                        for (s, e, ns) in anchored_ranges {
                            owned_ranges.push((s, e, ns.to_string()));
                        }
                    }

                    if has_symbol {
                        let symbols: Vec<(&str, &str)> = hunks
                            .iter()
                            .filter_map(|h| match h {
                                EditHunk::Symbol { symbol, new_string } => {
                                    Some((symbol.as_str(), new_string.as_str()))
                                }
                                EditHunk::LineRange { .. } | EditHunk::Text { .. } => None,
                            })
                            .collect();
                        let (sym_ranges, sym_notes) = match resolve_symbol_hunks(
                            &current, &symbols, &path, &raw_path, &tracker,
                        ) {
                            Ok(ok) => ok,
                            Err(diag_err) => {
                                return Ok(ToolOutput::err_text(diag_err).with_details(json!({
                                    "file_path": raw_path,
                                    "edit_failed": true,
                                })));
                            }
                        };
                        fuzzy_notes.extend(sym_notes);
                        owned_ranges.extend(sym_ranges);
                    }

                    let borrowed_ranges: Vec<(usize, usize, &str)> = owned_ranges
                        .iter()
                        .map(|(s, e, ns)| (*s, *e, ns.as_str()))
                        .collect();
                    match apply_line_range_hunks(&current, &borrowed_ranges, &raw_path) {
                        Ok(next) => {
                            current = next;
                            total_replacements = borrowed_ranges.len();
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
                            EditHunk::LineRange { .. } | EditHunk::Symbol { .. } => unreachable!(),
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
                            "Edit aborted: failed to snapshot {raw_path} into CAS before writing: {e}"
                        )));
                    }
                };
                if let Err(msg) = atomic_write_confined(&ws, &extra, &path, &current) {
                    return Ok(ToolOutput::err_text(msg));
                }
                // Rebase model's line view in-place so current file line numbers stay the single
                // unambiguous coordinate system (and report line shifts). When called inside
                // CodeMode (`suppress_tracking`), clear the model's line view instead.
                let shift_note = {
                    let mut guard = tracker
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if suppress_tracking {
                        guard.record_write(&path, current.as_bytes());
                        None
                    } else {
                        guard.rebase_after_edit(&path, &before, &current)
                    }
                };
                if let Some(note) = shift_note {
                    fuzzy_notes.push(note);
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
                let mut reply = format!("Edited {raw_path} ({total_replacements} replacement(s)).");
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
        .map_or(("", ""), |(f, n)| (leading_ws(f), leading_ws(n)));

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
