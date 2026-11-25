//! Zero-turn diagnostic file-slice prefetch for failing `Bash` commands.
//!
//! When a build/test command exits non-zero (`exit != 0`), scans its output for workspace-confined
//! compiler/test diagnostic locations (`rustc`/`cargo`, `pytest`/Python, `tsc`/`eslint`/`node`,
//! `go test`, `gcc`/`clang`), skips files already freshly viewed in [`ReadTracker`] or sensitive
//! paths, and attaches compact `±4`-line code windows directly to the `Bash` result while seeding
//! [`ReadTracker`] so the agent can call `Edit(start_line, end_line)` immediately on the next turn
//! without a wasted intermediate `Read` round-trip.

use crate::files::resolve_confined_path;
use crate::tracker::{ReadTracker, split_logical_lines};
use carnot_cas::redact_secrets_cow;
use regex::Regex;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

const WINDOW_RADIUS: usize = 4;
const MAX_SNIPPETS: usize = 3;
const MAX_TOTAL_BYTES: usize = 1_600;
const MAX_LINE_CHARS: usize = 160;
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

static RUST_ARROW_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:-->|panicked at)\s+([^\s:'`]+):(\d+)(?::\d+)?").expect("valid rust diag regex")
});
static PYTHON_FILE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"File "([^"]+)", line (\d+)"#).expect("valid python diag regex")
});
static TSC_PAREN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^\s*([^\s(:]+?\.[A-Za-z0-9_]+)\((\d+),\d+\):\s*error\b")
        .expect("valid tsc diag regex")
});
static GENERIC_DIAG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)^\s*([^\s:]+?\.(?:rs|py|ts|tsx|js|jsx|mjs|cjs|go|c|cc|cpp|cxx|h|hpp|java|rb|sh|toml|json|yaml|yml)):(\d+)(?::\d+)?:\s*(?:error|Error|FAIL|AssertionError|TypeError|SyntaxError|NameError|ValueError|expected|undefined|cannot|mismatched|unused)",
    )
    .expect("valid generic diag regex")
});

/// Result of diagnostic prefetching for a failing command.
#[derive(Debug, Clone)]
pub(crate) struct DiagnosticPrefetch {
    pub(crate) block: String,
    pub(crate) snippets_attached: usize,
}

/// Extracts candidate `(raw_path, 1-based line)` diagnostic locations in order of appearance.
pub(crate) fn extract_diagnostic_locations(output: &str) -> Vec<(String, usize)> {
    let mut found: Vec<(usize, String, usize)> = Vec::new();
    for re in [
        &*RUST_ARROW_RE,
        &*PYTHON_FILE_RE,
        &*TSC_PAREN_RE,
        &*GENERIC_DIAG_RE,
    ] {
        for caps in re.captures_iter(output) {
            let Some(m0) = caps.get(0) else { continue };
            let Some(p) = caps.get(1) else { continue };
            let Some(ln) = caps.get(2).and_then(|m| m.as_str().parse::<usize>().ok()) else {
                continue;
            };
            if ln == 0 {
                continue;
            }
            let raw_path = p.as_str().trim_matches(|c| c == '\'' || c == '"' || c == '`');
            if raw_path.is_empty() || raw_path.starts_with('<') {
                continue;
            }
            found.push((m0.start(), raw_path.to_string(), ln));
        }
    }
    found.sort_by_key(|(pos, _, _)| *pos);
    let mut deduped: Vec<(String, usize)> = Vec::new();
    for (_, path, line) in found {
        if !deduped.iter().any(|(p, l)| p == &path && *l == line) {
            deduped.push((path, line));
        }
    }
    deduped
}

fn is_sensitive_path(path: &Path) -> bool {
    for comp in path.components() {
        let s = comp.as_os_str().to_string_lossy().to_ascii_lowercase();
        if s == ".git" || s == ".ssh" || s == ".aws" || s == ".gnupg" || s == ".secrets" {
            return true;
        }
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if name == ".env"
        || name.starts_with(".env.")
        || name == "auth.json"
        || name == "credentials"
        || name == "credentials.toml"
        || name == ".netrc"
        || name == ".npmrc"
        || name == ".pypirc"
        || name.starts_with("id_rsa")
        || name.starts_with("id_ed25519")
    {
        return true;
    }
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("pem" | "key" | "p12" | "pfx" | "jks")
    )
}

/// Prefetches compact, workspace-confined, secret-redacted source windows for failing diagnostics
/// and seeds `read_tracker` so follow-up `Edit(start_line, end_line)` works immediately.
pub(crate) fn prefetch_diagnostic_context(
    workspace_root: &Path,
    exec_dir: &Path,
    extra_roots: &[PathBuf],
    raw_output: &str,
    read_tracker: Option<&Mutex<ReadTracker>>,
    redact_secrets: bool,
) -> Option<DiagnosticPrefetch> {
    let raw_locs = extract_diagnostic_locations(raw_output);
    if raw_locs.is_empty() {
        return None;
    }

    let ws_canon = fs::canonicalize(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf());

    // Resolve and merge overlapping windows per canonical file while preserving appearance order.
    let mut order: Vec<PathBuf> = Vec::new();
    let mut per_file: BTreeMap<PathBuf, Vec<(usize, usize)>> = BTreeMap::new();

    for (raw_path, line) in raw_locs {
        let resolved = if exec_dir != workspace_root && !Path::new(&raw_path).is_absolute() {
            let candidate = exec_dir.join(&raw_path);
            let via_exec = candidate
                .to_str()
                .and_then(|s| resolve_confined_path(workspace_root, extra_roots, s).ok())
                .filter(|p| p.is_file());
            match via_exec {
                Some(p) => p,
                None => match resolve_confined_path(workspace_root, extra_roots, &raw_path) {
                    Ok(p) => p,
                    Err(_) => continue,
                },
            }
        } else {
            match resolve_confined_path(workspace_root, extra_roots, &raw_path) {
                Ok(p) => p,
                Err(_) => continue,
            }
        };
        if is_sensitive_path(&resolved) {
            continue;
        }
        let Ok(meta) = fs::metadata(&resolved) else {
            continue;
        };
        if !meta.is_file() || meta.len() == 0 || meta.len() > MAX_FILE_BYTES {
            continue;
        }
        let start = line.saturating_sub(WINDOW_RADIUS).max(1);
        let end = line.saturating_add(WINDOW_RADIUS);
        let windows = per_file.entry(resolved.clone()).or_insert_with(|| {
            order.push(resolved.clone());
            Vec::new()
        });
        if let Some(existing) = windows
            .iter_mut()
            .find(|(s, e)| start <= e.saturating_add(2) && end.saturating_add(2) >= *s)
        {
            existing.0 = existing.0.min(start);
            existing.1 = existing.1.max(end);
        } else {
            windows.push((start, end));
        }
        let total_windows: usize = per_file.values().map(Vec::len).sum();
        if total_windows >= MAX_SNIPPETS {
            break;
        }
    }

    if order.is_empty() {
        return None;
    }

    let mut block = String::from("[diagnostic-context: auto-prefetched source around failing lines]\n");
    let mut attached = 0usize;

    'outer: for path in order {
        let Some(windows) = per_file.get(&path) else {
            continue;
        };
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        if bytes[..bytes.len().min(4096)].contains(&0) {
            continue;
        }
        let Ok(text) = std::str::from_utf8(&bytes) else {
            continue;
        };
        let all_lines = split_logical_lines(text);
        if all_lines.is_empty() {
            continue;
        }
        let rel = path
            .strip_prefix(&ws_canon)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");

        for &(raw_start, raw_end) in windows {
            if attached >= MAX_SNIPPETS {
                break 'outer;
            }
            let start = raw_start.min(all_lines.len()).max(1);
            let end = raw_end.min(all_lines.len());
            if start > end {
                continue;
            }

            // Skip if the model already holds an up-to-date view of these exact lines!
            if let Some(tracker_lock) = read_tracker
                && let Ok(tracker) = tracker_lock.lock()
                && tracker.has_fresh_view_range(&path, &bytes, start, end)
            {
                continue;
            }

            let slice = &all_lines[start - 1..end];
            let mut snippet = format!("--- {rel} (L{start}-L{end}) ---\n");
            for (idx, line) in slice.iter().enumerate() {
                let ln = start + idx;
                let safe_line = if redact_secrets {
                    redact_secrets_cow(line)
                } else {
                    std::borrow::Cow::Borrowed(*line)
                };
                let rendered: String = if safe_line.chars().count() > MAX_LINE_CHARS {
                    let s: String = safe_line.chars().take(MAX_LINE_CHARS).collect();
                    format!("{s}…")
                } else {
                    safe_line.into_owned()
                };
                let _ = writeln!(snippet, "{ln}\t{rendered}");
            }

            if block.len() + snippet.len() > MAX_TOTAL_BYTES && attached > 0 {
                break 'outer;
            }
            block.push_str(&snippet);
            attached += 1;

            if let Some(tracker_lock) = read_tracker
                && let Ok(mut tracker) = tracker_lock.lock()
            {
                tracker.record_diagnostic_slice(&path, &bytes, start, slice);
            }
        }
    }

    if attached == 0 {
        return None;
    }
    Some(DiagnosticPrefetch {
        block: block.trim_end().to_string(),
        snippets_attached: attached,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_rust_python_ts_and_go_diagnostic_locations() {
        let out = "\
error[E0308]: mismatched types
 --> src/auth.rs:42:17
File \"pkg/worker.py\", line 19, in run
src/ui.ts(105,4): error TS2322: Type 'string' is not assignable
    handler_test.go:77: expected 200 got 500
";
        let locs = extract_diagnostic_locations(out);
        assert_eq!(
            locs,
            vec![
                ("src/auth.rs".to_string(), 42),
                ("pkg/worker.py".to_string(), 19),
                ("src/ui.ts".to_string(), 105),
                ("handler_test.go".to_string(), 77),
            ]
        );
    }

    #[test]
    fn prefetches_bounded_redacted_slices_skips_sensitive_and_seeds_read_tracker() {
        let dir = std::env::temp_dir().join(format!("carnot-diag-pf-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(dir.join("src")).expect("mkdir");
        fs::write(
            dir.join("src/lib.rs"),
            "fn l1() {}\nfn l2() {}\nfn l3() {}\nfn l4() {}\nlet bad: u32 = \"oops\";\nfn l6() {}\nfn l7() {}\n",
        )
        .expect("write lib");
        fs::write(dir.join(".env"), "API_KEY=supersecret\n").expect("write env");

        let tracker = Mutex::new(ReadTracker::new());
        let stderr = "error[E0308]: mismatched types\n --> src/lib.rs:5:16\n --> .env:1:1\n --> /etc/passwd:1:1\n";

        let pf = prefetch_diagnostic_context(&dir, &dir, &[], stderr, Some(&tracker), true)
            .expect("prefetch");
        assert_eq!(pf.snippets_attached, 1, "only src/lib.rs must be prefetched");
        assert!(pf.block.contains("5\tlet bad: u32 = \"oops\";"), "got:\n{}", pf.block);
        assert!(!pf.block.contains("supersecret"), "must skip .env");
        assert!(!pf.block.contains("/etc/passwd"), "must block escape");

        // Second call with identical file must skip (already freshly viewed in ReadTracker!).
        let pf2 = prefetch_diagnostic_context(&dir, &dir, &[], stderr, Some(&tracker), true);
        assert!(pf2.is_none(), "already-viewed lines must not be duplicated");

        let _ = fs::remove_dir_all(&dir);
    }
}
