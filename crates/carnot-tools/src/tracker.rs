//! Disk-coherent read tracker using direct BLAKE3 hashing of raw (unnumbered) line slices.
//!
//! Fixes Opus findings F3 (no coarse mtime shortcut!) and F4 (stores & diffs raw unnumbered lines
//! so line-number shifts never produce bogus full-file diffs).

use similar::TextDiff;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct TrackedFile {
    pub hash: [u8; 32],
    pub raw_content: String,
    pub range: (usize, usize),
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

fn tracker_key(path: &Path) -> PathBuf {
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    {
        PathBuf::from(path.to_string_lossy().to_ascii_lowercase())
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        path.to_path_buf()
    }
}

impl ReadTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.files.clear();
    }

    /// Invalidates one path (used when an eager Read is aborted/discarded — fixes Opus N3).
    pub fn invalidate(&mut self, path: &Path) {
        self.files.remove(&tracker_key(path));
    }

    pub fn record(&mut self, path: &Path, raw_content: &str, range: (usize, usize)) {
        let hash = *blake3::hash(raw_content.as_bytes()).as_bytes();
        self.files.insert(
            tracker_key(path),
            TrackedFile {
                hash,
                raw_content: raw_content.to_string(),
                range,
            },
        );
    }

    /// Checks whether a re-read of `path` over `range` can return a 1-line stub or compact diff.
    /// `current_raw` MUST be raw unnumbered text for the exact `(start, end)` range actually shown.
    pub fn check_reread(&self, path: &Path, current_raw: &str, range: (usize, usize)) -> ReadDelta {
        let Some(prev) = self.files.get(&tracker_key(path)) else {
            return ReadDelta::FullNeeded;
        };
        if prev.range != range {
            return ReadDelta::FullNeeded;
        }
        let cur_hash = *blake3::hash(current_raw.as_bytes()).as_bytes();
        if cur_hash == prev.hash {
            return ReadDelta::UnchangedStub(format!(
                "[unchanged: {} lines {}-{} identical to previous read]",
                path.display(),
                range.0,
                range.1
            ));
        }

        // Compute compact unified diff on raw unnumbered lines; use it only when strictly shorter.
        let diff = TextDiff::from_lines(&prev.raw_content, current_raw);
        let unified = diff
            .unified_diff()
            .context_radius(2)
            .header("prev", "curr")
            .to_string();
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
        let v1 = "fn a() {\n    let x = 1;\n    let y = 2;\n    let z = 3;\n    let w = 4;\n    let q = 5;\n    let r = 6;\n}\n";
        let v2 = "fn a() {\n    let x = 9;\n    let y = 2;\n    let z = 3;\n    let w = 4;\n    let q = 5;\n    let r = 6;\n}\n";
        tracker.record(p, v1, (1, 8));
        assert!(matches!(
            tracker.check_reread(p, v1, (1, 8)),
            ReadDelta::UnchangedStub(_)
        ));
        // Even with identical byte length and same mtime, BLAKE3 catches the edit!
        assert!(matches!(
            tracker.check_reread(p, v2, (1, 8)),
            ReadDelta::FullNeeded | ReadDelta::UnifiedDiff(_)
        ));
    }
}
