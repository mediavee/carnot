//! Disk-coherent read tracker using direct BLAKE3 hashing of file contents and per-line views,
//! plus task-scoped tracking suppression for nested orchestration tools (`CodeMode`) and
//! deadlock-free per-path async write locks.
//!
//! Tracks two distinct facts per file:
//! - `disk_hash`: the last full-file BLAKE3 digest known to the harness (updated by `Read`, `Edit`,
//!   and `Write`) — used by `Write` to prevent overwriting unread or externally modified files.
//! - `view`: the 1-based line hashes corresponding to the model's current view (populated by `Read`
//!   and rebased in-place after model-initiated `Edit`s so current file line numbers are always the
//!   single unambiguous coordinate system; never updated by external or `CodeMode` changes).

use similar::{ChangeTag, DiffOp, TextDiff};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

tokio::task_local! {
    static SUPPRESS_MODEL_READ_TRACKING: bool;
}

/// Runs `fut` with model-facing `ReadTracker` view updates and stub/diff short-circuits disabled.
///
/// Used by `Agent::invoke_tool_gated` (e.g. `CodeMode` scripts) so:
/// - `tools.Read(...)` inside JS always returns actual file content (never an `[unchanged]` stub),
/// - reads performed inside JS never trick the harness into thinking the LLM saw the file,
/// - edits/writes performed inside JS invalidate the model's line view for that path.
pub async fn with_suppressed_read_tracking<F: std::future::Future>(fut: F) -> F::Output {
    SUPPRESS_MODEL_READ_TRACKING.scope(true, fut).await
}

/// Returns `true` when executing inside [`with_suppressed_read_tracking`].
pub fn is_read_tracking_suppressed() -> bool {
    SUPPRESS_MODEL_READ_TRACKING
        .try_with(|&v| v)
        .unwrap_or(false)
}

/// Per-path async mutex pool shared by `EditTool` and `WriteTool` so concurrent mutations to the
/// same canonical file path (e.g. via `CodeMode` `Promise.all` or direct `ToolRegistry` calls)
/// serialize cleanly without losing updates or corrupting `undo_stack`.
#[derive(Debug, Default)]
pub struct PathLockPool {
    locks: Mutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>,
}

impl PathLockPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lock_for(&self, path: &Path) -> Arc<tokio::sync::Mutex<()>> {
        let key = tracker_key(path);
        let mut g = self
            .locks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Arc::clone(
            g.entry(key)
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        )
    }
}

/// Compact fingerprint of a single line as shown to the model by `Read` (or written by `Edit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewedLine {
    pub hash: u64,
    pub trimmed_len: u32,
}

/// Computes a deterministic 64-bit BLAKE3 prefix hash of one line (ignoring any trailing `\r`).
pub fn hash_line(line: &str) -> u64 {
    let clean = line.trim_end_matches('\r');
    let digest = blake3::hash(clean.as_bytes());
    let b = digest.as_bytes();
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// Builds a [`ViewedLine`] fingerprint for one raw line.
pub fn viewed_line(line: &str) -> ViewedLine {
    let clean = line.trim_end_matches('\r');
    ViewedLine {
        hash: hash_line(clean),
        trimmed_len: u32::try_from(clean.trim().chars().count()).unwrap_or(u32::MAX),
    }
}

/// Splits file text into logical lines identically to `EditTool::apply_line_range_hunks`.
pub(crate) fn split_logical_lines(content: &str) -> Vec<&str> {
    let has_trailing_nl = content.ends_with('\n');
    let body = content
        .strip_suffix("\r\n")
        .or_else(|| content.strip_suffix('\n'))
        .unwrap_or(content);
    if body.is_empty() && !has_trailing_nl {
        Vec::new()
    } else {
        body.split('\n').map(|l| l.trim_end_matches('\r')).collect()
    }
}

#[derive(Debug, Clone)]
struct LastSlice {
    range: (usize, usize),
    is_full_file: bool,
    raw_content: String,
    hash: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct TrackedFile {
    /// Full-file BLAKE3 digest last observed/written by the harness.
    pub disk_hash: [u8; 32],
    /// Full-file BLAKE3 digest corresponding to `view` (cleared whenever `Read` sees a new digest
    /// so `view` never mixes lines from two different file versions).
    pub view_disk_hash: Option<[u8; 32]>,
    /// 1-based line number -> line fingerprint matching the model's current view.
    pub view: BTreeMap<usize, ViewedLine>,
    last_slice: Option<LastSlice>,
}

#[derive(Debug, Default)]
pub struct ReadTracker {
    files: HashMap<PathBuf, TrackedFile>,
}

#[derive(Debug)]
pub enum ReadDelta {
    FullNeeded,
    UnchangedStub(String),
    UnifiedDiff(String),
}

pub(crate) fn tracker_key(path: &Path) -> PathBuf {
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    {
        PathBuf::from(path.to_string_lossy().to_ascii_lowercase())
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        path.to_path_buf()
    }
}

/// Formats a unified diff between `old_raw` and `new_raw` with **absolute 1-based file line numbers**
/// starting at `start_line`, plus an explicit line-shift note whenever line count changes.
fn format_absolute_unified_diff(
    old_raw: &str,
    new_raw: &str,
    start_line: usize,
    end_line: usize,
) -> String {
    let old_norm;
    let old_src = if !old_raw.is_empty() && !old_raw.ends_with('\n') {
        old_norm = format!("{old_raw}\n");
        old_norm.as_str()
    } else {
        old_raw
    };
    let new_norm;
    let new_src = if !new_raw.is_empty() && !new_raw.ends_with('\n') {
        new_norm = format!("{new_raw}\n");
        new_norm.as_str()
    } else {
        new_raw
    };

    let diff = TextDiff::from_lines(old_src, new_src);
    let groups = diff.grouped_ops(2);
    if groups.is_empty() {
        return String::new();
    }

    let base_offset = start_line.saturating_sub(1);
    let mut out = String::from("--- prev\n+++ curr\n");
    let mut cum_delta: isize = 0;
    let mut any_hunk_shifted = false;
    let mut last_shift_old_line: usize = start_line;

    for group in &groups {
        let (Some(first), Some(last)) = (group.first(), group.last()) else {
            continue;
        };
        let old_start = first.old_range().start;
        let old_end = last.old_range().end;
        let new_start = first.new_range().start;
        let new_end = last.new_range().end;
        let old_len = old_end.saturating_sub(old_start);
        let new_len = new_end.saturating_sub(new_start);

        let old_hdr = if old_len == 0 {
            base_offset + old_start
        } else {
            base_offset + old_start + 1
        };
        let new_hdr = if new_len == 0 {
            base_offset + new_start
        } else {
            base_offset + new_start + 1
        };
        let _ = writeln!(out, "@@ -{old_hdr},{old_len} +{new_hdr},{new_len} @@");

        let hunk_delta = new_len.cast_signed() - old_len.cast_signed();
        if hunk_delta != 0 {
            any_hunk_shifted = true;
            cum_delta += hunk_delta;
            let non_eq_old_end = group
                .iter()
                .rev()
                .find(|op| !matches!(op, DiffOp::Equal { .. }))
                .map_or(old_end, |op| op.old_range().end);
            last_shift_old_line = (base_offset + non_eq_old_end).max(start_line);
        }

        for op in group {
            for change in diff.iter_changes(op) {
                let sign = match change.tag() {
                    ChangeTag::Delete => '-',
                    ChangeTag::Insert => '+',
                    ChangeTag::Equal => ' ',
                };
                out.push(sign);
                out.push_str(change.value());
                if !change.value().ends_with('\n') {
                    out.push('\n');
                }
            }
        }
    }

    if cum_delta != 0 {
        let _ = writeln!(
            out,
            "[lines after L{last_shift_old_line} shifted by {cum_delta:+}]"
        );
    } else if any_hunk_shifted {
        let _ = writeln!(
            out,
            "[line numbers inside L{start_line}-L{end_line} shifted; see @@ headers]"
        );
    }

    out
}

impl ReadTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.files.clear();
    }

    /// Invalidates one path (used when an eager Read is aborted or discarded, or on Undo).
    pub fn invalidate(&mut self, path: &Path) {
        self.files.remove(&tracker_key(path));
    }

    /// Records only the full-file `disk_hash` without touching `view` or `last_slice`.
    /// Used by `Read` in `outline`/empty/image mode.
    pub fn record_disk_hash(&mut self, path: &Path, full_file_bytes: &[u8]) {
        let disk_hash = *blake3::hash(full_file_bytes).as_bytes();
        let entry = self
            .files
            .entry(tracker_key(path))
            .or_insert_with(|| TrackedFile {
                disk_hash,
                view_disk_hash: None,
                view: BTreeMap::new(),
                last_slice: None,
            });
        entry.disk_hash = disk_hash;
    }

    /// Returns `true` when the model already holds an up-to-date line view covering
    /// `start_line..=end_line` for the current on-disk content of `path`.
    pub fn has_fresh_view_range(
        &self,
        path: &Path,
        full_file_bytes: &[u8],
        start_line: usize,
        end_line: usize,
    ) -> bool {
        let disk_hash = *blake3::hash(full_file_bytes).as_bytes();
        let Some(entry) = self.files.get(&tracker_key(path)) else {
            return false;
        };
        if entry.view_disk_hash != Some(disk_hash) {
            return false;
        }
        (start_line..=end_line).all(|ln| entry.view.contains_key(&ln))
    }

    /// Seeds `view` with a compact diagnostic-prefetched line window (without mutating
    /// `last_slice`) so follow-up `Edit(start_line, end_line)` succeeds immediately while a
    /// subsequent full `Read` still returns full file content rather than a stub.
    pub fn record_diagnostic_slice(
        &mut self,
        path: &Path,
        full_file_bytes: &[u8],
        start_line: usize,
        slice_lines: &[&str],
    ) {
        let disk_hash = *blake3::hash(full_file_bytes).as_bytes();
        let entry = self
            .files
            .entry(tracker_key(path))
            .or_insert_with(|| TrackedFile {
                disk_hash,
                view_disk_hash: None,
                view: BTreeMap::new(),
                last_slice: None,
            });
        entry.disk_hash = disk_hash;
        if entry.view_disk_hash != Some(disk_hash) {
            entry.view.clear();
            entry.view_disk_hash = Some(disk_hash);
        }
        for (idx, line) in slice_lines.iter().enumerate() {
            entry.view.insert(start_line + idx, viewed_line(line));
        }
    }

    /// Rebases `view` and updates `disk_hash` after a model-initiated `Edit` transforms `before`
    /// into `after`. Returns an optional line-shift note (`[lines after Lx shifted by ±n]`) when
    /// the line count changed.
    ///
    /// Keeping `view` rebased to current file line numbers makes current numbering the single
    /// unambiguous convention for follow-up `Edit`s while preserving `last_slice` so a subsequent
    /// `Read` still produces a compact unified diff.
    pub fn rebase_after_edit(&mut self, path: &Path, before: &str, after: &str) -> Option<String> {
        let before_hash = *blake3::hash(before.as_bytes()).as_bytes();
        let after_hash = *blake3::hash(after.as_bytes()).as_bytes();
        let before_lines = split_logical_lines(before);
        let after_lines = split_logical_lines(after);
        let diff = TextDiff::from_slices(&before_lines, &after_lines);
        let ops = diff.ops();

        let entry = self
            .files
            .entry(tracker_key(path))
            .or_insert_with(|| TrackedFile {
                disk_hash: after_hash,
                view_disk_hash: None,
                view: BTreeMap::new(),
                last_slice: None,
            });
        entry.disk_hash = after_hash;

        if entry.view_disk_hash == Some(before_hash) && !entry.view.is_empty() {
            let mut new_view = BTreeMap::new();
            for op in ops {
                match *op {
                    DiffOp::Equal {
                        old_index,
                        new_index,
                        len,
                    } => {
                        for k in 0..len {
                            if entry.view.contains_key(&(old_index + k + 1)) {
                                new_view.insert(
                                    new_index + k + 1,
                                    viewed_line(after_lines[new_index + k]),
                                );
                            }
                        }
                    }
                    DiffOp::Replace {
                        new_index, new_len, ..
                    }
                    | DiffOp::Insert {
                        new_index, new_len, ..
                    } => {
                        for k in 0..new_len {
                            new_view
                                .insert(new_index + k + 1, viewed_line(after_lines[new_index + k]));
                        }
                    }
                    DiffOp::Delete { .. } => {}
                }
            }
            entry.view = new_view;
            entry.view_disk_hash = Some(after_hash);
        }

        let net_delta = after_lines.len().cast_signed() - before_lines.len().cast_signed();
        if net_delta != 0 {
            let anchor = ops
                .iter()
                .rev()
                .find(|op| !matches!(op, DiffOp::Equal { .. }))
                .map_or(1, |op| op.old_range().end.max(1));
            Some(format!("[lines after L{anchor} shifted by {net_delta:+}]"))
        } else {
            None
        }
    }

    /// Records a full-file `Write` (or suppressed `CodeMode` mutation): updates `disk_hash` and
    /// clears `view`/`last_slice` so stale pre-write line numbers can never match post-write text.
    pub fn record_write(&mut self, path: &Path, full_file_bytes: &[u8]) {
        let disk_hash = *blake3::hash(full_file_bytes).as_bytes();
        self.files.insert(
            tracker_key(path),
            TrackedFile {
                disk_hash,
                view_disk_hash: None,
                view: BTreeMap::new(),
                last_slice: None,
            },
        );
    }

    /// Records a contiguous `Read` slice (full, partial, or diff-updated) into `disk_hash`,
    /// `view`, and `last_slice`.
    pub fn record_slice(
        &mut self,
        path: &Path,
        full_file_bytes: &[u8],
        raw_slice: &str,
        range: (usize, usize),
        is_full_file: bool,
    ) {
        let file_hash = *blake3::hash(full_file_bytes).as_bytes();
        let slice_hash = *blake3::hash(raw_slice.as_bytes()).as_bytes();
        let entry = self
            .files
            .entry(tracker_key(path))
            .or_insert_with(|| TrackedFile {
                disk_hash: file_hash,
                view_disk_hash: Some(file_hash),
                view: BTreeMap::new(),
                last_slice: None,
            });
        entry.disk_hash = file_hash;
        if entry.view_disk_hash != Some(file_hash) {
            entry.view.clear();
            entry.view_disk_hash = Some(file_hash);
        }
        if range.0 >= 1 && range.1 >= range.0 {
            for (idx, line) in raw_slice.split('\n').enumerate() {
                let ln = range.0 + idx;
                if ln > range.1 {
                    break;
                }
                entry.view.insert(ln, viewed_line(line));
            }
        }
        entry.last_slice = Some(LastSlice {
            range,
            is_full_file,
            raw_content: raw_slice.to_string(),
            hash: slice_hash,
        });
    }

    /// Records one or more displayed spans from `Read(symbol=...)` into `disk_hash` and `view`.
    pub fn record_viewed_spans(
        &mut self,
        path: &Path,
        full_file_bytes: &[u8],
        spans: &[(usize, &[&str])],
    ) {
        let file_hash = *blake3::hash(full_file_bytes).as_bytes();
        let entry = self
            .files
            .entry(tracker_key(path))
            .or_insert_with(|| TrackedFile {
                disk_hash: file_hash,
                view_disk_hash: Some(file_hash),
                view: BTreeMap::new(),
                last_slice: None,
            });
        entry.disk_hash = file_hash;
        if entry.view_disk_hash != Some(file_hash) {
            entry.view.clear();
            entry.view_disk_hash = Some(file_hash);
        }
        for &(start_line, lines) in spans {
            for (idx, line) in lines.iter().enumerate() {
                entry.view.insert(start_line + idx, viewed_line(line));
            }
        }
    }

    /// Returns the last full-file BLAKE3 digest known to the harness for `path`, if any.
    pub fn disk_hash(&self, path: &Path) -> Option<[u8; 32]> {
        self.files.get(&tracker_key(path)).map(|f| f.disk_hash)
    }

    /// Returns the exact sequence of [`ViewedLine`] fingerprints for `start_line..=end_line`
    /// if and only if every line in that inclusive range was previously shown to the model.
    pub fn viewed_range(
        &self,
        path: &Path,
        start_line: usize,
        end_line: usize,
    ) -> Option<Vec<ViewedLine>> {
        if start_line == 0 || end_line < start_line {
            return None;
        }
        let entry = self.files.get(&tracker_key(path))?;
        let mut out = Vec::with_capacity(end_line - start_line + 1);
        for ln in start_line..=end_line {
            let vl = *entry.view.get(&ln)?;
            out.push(vl);
        }
        Some(out)
    }

    /// Checks whether a re-read of `path` over `range` can return a 1-line stub or compact diff.
    ///
    /// Only returns [`ReadDelta::UnchangedStub`] when BOTH the slice hash AND the full-file
    /// `disk_hash` are unchanged, preventing a stale-`Write` / unchanged-`Read` deadlock when
    /// only line endings, trailing newline, or lines outside a partial slice changed on disk.
    pub fn check_reread(
        &self,
        path: &Path,
        full_file_bytes: &[u8],
        current_raw: &str,
        range: (usize, usize),
        is_full_file: bool,
    ) -> ReadDelta {
        let Some(entry) = self.files.get(&tracker_key(path)) else {
            return ReadDelta::FullNeeded;
        };
        let Some(ref prev) = entry.last_slice else {
            return ReadDelta::FullNeeded;
        };
        let same_window = prev.range == range;
        let both_full = prev.is_full_file && is_full_file && prev.range.0 == range.0;
        if !same_window && !both_full {
            return ReadDelta::FullNeeded;
        }
        let cur_slice_hash = *blake3::hash(current_raw.as_bytes()).as_bytes();
        let cur_file_hash = *blake3::hash(full_file_bytes).as_bytes();
        if same_window && cur_slice_hash == prev.hash {
            if cur_file_hash == entry.disk_hash {
                return ReadDelta::UnchangedStub(format!(
                    "[unchanged: {} lines {}-{} identical to previous read]",
                    path.display(),
                    range.0,
                    range.1
                ));
            }
            // Full file bytes changed (e.g. CRLF/trailing newline or out-of-window edit):
            // force a fresh read so `disk_hash` syncs and `Write` never loops on stale hash.
            return ReadDelta::FullNeeded;
        }

        let unified =
            format_absolute_unified_diff(&prev.raw_content, current_raw, range.0, range.1);
        if !unified.is_empty() && unified.len() + 64 < current_raw.len() {
            ReadDelta::UnifiedDiff(format!(
                "[changed since previous read — unified diff for {} lines {}-{}]\n{unified}",
                path.display(),
                range.0,
                range.1
            ))
        } else {
            ReadDelta::FullNeeded
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unchanged_and_same_second_edit_detected_by_blake3() {
        let mut tracker = ReadTracker::new();
        let p = Path::new("/ws/a.rs");
        let v1 = "fn a() {\n    let x = 1;\n    let y = 2;\n    let z = 3;\n    let w = 4;\n    let q = 5;\n    let r = 6;\n}";
        let v2 = "fn a() {\n    let x = 9;\n    let y = 2;\n    let z = 3;\n    let w = 4;\n    let q = 5;\n    let r = 6;\n}";
        tracker.record_slice(p, v1.as_bytes(), v1, (1, 8), true);
        assert!(
            matches!(
                tracker.check_reread(p, v1.as_bytes(), v1, (1, 8), true),
                ReadDelta::UnchangedStub(_)
            ),
            "identical content should return UnchangedStub"
        );
        // Trailing-newline-only change MUST NOT return UnchangedStub (avoids Write/Read deadlock!).
        let v1_nl = format!("{v1}\n");
        assert!(
            matches!(
                tracker.check_reread(p, v1_nl.as_bytes(), v1, (1, 8), true),
                ReadDelta::FullNeeded
            ),
            "trailing-newline change must refresh disk_hash instead of stubbing"
        );
        // Even with identical byte length and same mtime, BLAKE3 catches the edit!
        assert!(
            matches!(
                tracker.check_reread(p, v2.as_bytes(), v2, (1, 8), true),
                ReadDelta::FullNeeded | ReadDelta::UnifiedDiff(_)
            ),
            "edited content must not return UnchangedStub"
        );
    }

    #[test]
    fn reread_diff_uses_absolute_line_numbers_and_reports_line_shifts() {
        let mut tracker = ReadTracker::new();
        let p = Path::new("/ws/big.rs");

        // 1. Offset slice starting at line 100: diff header must say @@ -100,... @@, NEVER @@ -1,... @@
        let mut old_lines: Vec<String> = (100..=140)
            .map(|i| format!("    let line_{i} = compute_value({i});"))
            .collect();
        let old_raw = old_lines.join("\n");
        tracker.record_slice(p, old_raw.as_bytes(), &old_raw, (100, 140), false);

        old_lines[2] = "    let line_102 = compute_value(999);".to_string();
        let new_raw = old_lines.join("\n");

        match tracker.check_reread(p, new_raw.as_bytes(), &new_raw, (100, 140), false) {
            ReadDelta::UnifiedDiff(diff) => {
                assert!(
                    diff.contains("@@ -100,5 +100,5 @@"),
                    "expected absolute @@ header starting at 100, got:\n{diff}"
                );
                assert!(
                    !diff.contains("@@ -1,"),
                    "must never emit slice-relative @@ -1 header, got:\n{diff}"
                );
            }
            other => panic!("expected UnifiedDiff, got: {other:?}"),
        }

        // 2. Full-file re-read after inserting 3 lines reports exact shift note
        let full_v1: Vec<String> = (1..=40)
            .map(|i| format!("pub fn item_{i}() -> usize {{ {i} }}"))
            .collect();
        let full_v1_raw = full_v1.join("\n");
        tracker.record_slice(p, full_v1_raw.as_bytes(), &full_v1_raw, (1, 40), true);

        let mut full_v2 = full_v1;
        full_v2.splice(
            9..10,
            vec![
                "pub fn item_10_a() -> usize { 101 }".to_string(),
                "pub fn item_10_b() -> usize { 102 }".to_string(),
                "pub fn item_10_c() -> usize { 103 }".to_string(),
                "pub fn item_10_d() -> usize { 104 }".to_string(),
            ],
        );
        let full_v2_raw = full_v2.join("\n");
        match tracker.check_reread(p, full_v2_raw.as_bytes(), &full_v2_raw, (1, 43), true) {
            ReadDelta::UnifiedDiff(diff) => {
                assert!(
                    diff.contains("@@ -8,5 +8,8 @@"),
                    "expected absolute @@ -8,5 +8,8 @@ header, got:\n{diff}"
                );
                assert!(
                    diff.contains("[lines after L10 shifted by +3]"),
                    "expected shift note, got:\n{diff}"
                );
            }
            other => panic!("expected UnifiedDiff for shifted full file, got: {other:?}"),
        }
    }
}
