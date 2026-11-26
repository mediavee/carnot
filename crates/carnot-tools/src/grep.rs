//! Native workspace-confined `Grep` tool (regex search powered by `ignore` + `globset` —
//! the exact directory-walking and `.gitignore` engine of `ripgrep`).
//!
//! Runs concurrently with `Read` during streaming (`ToolEffects::global_read()`), honors
//! `.gitignore` + nested `.gitignore` + `.git/info/exclude` + `!` negations, skips binary files
//! and build artifacts, stops walking immediately once `limit` matches are found, and enforces
//! strict symlink confinement.

use crate::files::resolve_confined_path;
use crate::{BoxFuture, Tool};
use carnot_core::{Result, ToolEffects, ToolOutput, ToolSpec};
use globset::{GlobBuilder, GlobMatcher};
use ignore::WalkBuilder;
use regex::{Regex, RegexBuilder};
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

const DEFAULT_MATCH_LIMIT: usize = 60;
const MAX_MATCH_LIMIT: usize = 200;
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;
const MAX_FILES_VISITED: usize = 25_000;
const MAX_LINE_CHARS: usize = 240;

const ALWAYS_SKIP_DIRS: &[&str] = &[
    ".git",
    ".carnot",
    "target",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
    ".cache",
];

#[derive(Debug)]
pub struct GrepTool {
    workspace_root: PathBuf,
    extra_roots: Vec<PathBuf>,
}

impl GrepTool {
    pub fn new(workspace_root: PathBuf, extra_roots: Vec<PathBuf>) -> Self {
        Self {
            workspace_root,
            extra_roots,
        }
    }
}

impl Tool for GrepTool {
    fn spec(&self) -> ToolSpec {
        crate::canonical_tool_specs()
            .into_iter()
            .find(|s| s.name == "Grep")
            .expect("Grep spec")
    }

    fn effects(&self, _input: &Value) -> ToolEffects {
        ToolEffects::global_read()
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        let ws = self.workspace_root.clone();
        let extra = self.extra_roots.clone();

        Box::pin(async move {
            let Some(pattern) = input
                .get("pattern")
                .and_then(Value::as_str)
                .map(str::to_string)
            else {
                return Ok(ToolOutput::err_text("Grep requires `pattern`"));
            };
            if pattern.is_empty() {
                return Ok(ToolOutput::err_text("Grep `pattern` must not be empty"));
            }
            let raw_path = input
                .get("path")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or(".")
                .to_string();
            let ext_filter = input
                .get("ext")
                .and_then(Value::as_str)
                .map(|s| s.trim().trim_start_matches('.').to_ascii_lowercase())
                .filter(|s| !s.is_empty());
            let glob_raw = input
                .get("glob")
                .and_then(Value::as_str)
                .map(|s| s.trim().trim_start_matches("./").trim_start_matches('/'))
                .filter(|s| !s.is_empty())
                .map(str::to_string);
            let case_insensitive = input
                .get("case_insensitive")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let limit = input
                .get("limit")
                .and_then(Value::as_u64)
                .filter(|&v| v > 0)
                .map_or(DEFAULT_MATCH_LIMIT, |v| {
                    (v as usize).clamp(1, MAX_MATCH_LIMIT)
                });

            tokio::task::spawn_blocking(move || -> Result<ToolOutput> {
                let re = match RegexBuilder::new(&pattern)
                    .case_insensitive(case_insensitive)
                    .size_limit(2 * 1024 * 1024)
                    .build()
                {
                    Ok(r) => r,
                    Err(e) => {
                        return Ok(ToolOutput::err_text(format!(
                            "Invalid regex `{pattern}`: {e}"
                        )));
                    }
                };

                let glob_filter = match glob_raw {
                    Some(ref pat) => match GlobBuilder::new(pat).literal_separator(true).build() {
                        Ok(g) => Some((g.compile_matcher(), pat.contains('/'))),
                        Err(e) => {
                            return Ok(ToolOutput::err_text(format!("Invalid glob `{pat}`: {e}")));
                        }
                    },
                    None => None,
                };

                let target = match resolve_confined_path(&ws, &extra, &raw_path) {
                    Ok(p) => p,
                    Err(msg) => return Ok(ToolOutput::err_text(msg)),
                };
                let ws_canon = fs::canonicalize(&ws).unwrap_or_else(|_| ws.clone());
                let search_root_canon =
                    fs::canonicalize(&target).unwrap_or_else(|_| target.clone());

                let mut state = SearchState {
                    ws_canon: &ws_canon,
                    search_root_canon: &search_root_canon,
                    re: &re,
                    ext_filter: ext_filter.as_deref(),
                    glob_filter: glob_filter.as_ref().map(|(m, h)| (m, *h)),
                    limit,
                    visited_files: 0,
                    files_scanned: 0,
                    matches: Vec::new(),
                    truncated: false,
                };

                if target.is_file() {
                    state.search_file(&target, &target);
                } else if target.is_dir() {
                    let mut builder = WalkBuilder::new(&target);
                    builder
                        .hidden(false)
                        .ignore(false)
                        .parents(false)
                        .git_ignore(true)
                        .git_exclude(false)
                        .git_global(false)
                        .require_git(false)
                        .follow_links(false)
                        .sort_by_file_name(std::cmp::Ord::cmp)
                        .filter_entry(|entry| {
                            if entry.depth() == 0 {
                                return true;
                            }
                            let is_dir = entry.file_type().is_some_and(|ft| ft.is_dir());
                            if is_dir
                                && let Some(name) = entry.file_name().to_str()
                                && ALWAYS_SKIP_DIRS.contains(&name)
                            {
                                return false;
                            }
                            true
                        });

                    // Confine ignore rules strictly inside `ws_canon` (never load `~/.gitignore`
                    // or parent dirs above the workspace!). Still honor `.git/info/exclude` and
                    // intermediate `.gitignore` files between `ws_canon` and a subdir `target`.
                    let git_exclude = ws_canon.join(".git").join("info").join("exclude");
                    if git_exclude.is_file() {
                        let _ = builder.add_ignore(&git_exclude);
                    }
                    if let Ok(rel_sub) = search_root_canon.strip_prefix(&ws_canon)
                        && !rel_sub.as_os_str().is_empty()
                    {
                        let mut cur = ws_canon.clone();
                        let root_gi = cur.join(".gitignore");
                        if root_gi.is_file() {
                            let _ = builder.add_ignore(&root_gi);
                        }
                        if let Some(parent) = rel_sub.parent() {
                            for comp in parent.components() {
                                cur.push(comp);
                                let gi = cur.join(".gitignore");
                                if gi.is_file() {
                                    let _ = builder.add_ignore(&gi);
                                }
                            }
                        }
                    }

                    for entry in builder.build().flatten() {
                        if state.truncated
                            || state.visited_files >= MAX_FILES_VISITED
                            || cancel.is_cancelled()
                        {
                            break;
                        }
                        if entry.depth() == 0 {
                            continue;
                        }
                        let Some(ft) = entry.file_type() else {
                            continue;
                        };
                        let path = entry.path();
                        if ft.is_symlink() {
                            let Ok(resolved) =
                                resolve_confined_path(&ws, &extra, &path.to_string_lossy())
                            else {
                                continue;
                            };
                            state.consider_file(&resolved, path);
                        } else if ft.is_file() {
                            state.consider_file(path, path);
                        }
                    }
                } else {
                    return Ok(ToolOutput::err_text(format!("Path not found: {raw_path}")));
                }

                if cancel.is_cancelled() {
                    return Ok(ToolOutput::err_text("Grep cancelled"));
                }

                let scan_capped = state.visited_files >= MAX_FILES_VISITED && !state.truncated;
                if state.matches.is_empty() {
                    let mut msg = format!("[no matches for `{pattern}` in {raw_path}]");
                    if scan_capped {
                        let _ = write!(
                            msg,
                            "\n[warning: file scan capped at {MAX_FILES_VISITED} files — narrow `path` or `ext`]"
                        );
                    }
                    return Ok(ToolOutput::ok_text(msg).with_details(json!({
                        "matches": 0,
                        "files_scanned": state.files_scanned,
                        "scan_capped": scan_capped,
                    })));
                }

                let mut out = state.matches.join("\n");
                if state.truncated {
                    let _ = write!(
                        out,
                        "\n[truncated at {limit} matches — narrow `path`/`ext` or increase `limit`]"
                    );
                }
                if scan_capped {
                    let _ = write!(
                        out,
                        "\n[warning: file scan capped at {MAX_FILES_VISITED} files — narrow `path` or `ext`]"
                    );
                }
                Ok(ToolOutput::ok_text(out).with_details(json!({
                    "matches": state.matches.len(),
                    "truncated": state.truncated,
                    "files_scanned": state.files_scanned,
                    "scan_capped": scan_capped,
                })))
            })
            .await
            .map_err(|e| carnot_core::CarnotError::Tool {
                tool: "Grep".into(),
                message: format!("task join error: {e}"),
            })?
        })
    }
}

struct SearchState<'a> {
    ws_canon: &'a Path,
    search_root_canon: &'a Path,
    re: &'a Regex,
    ext_filter: Option<&'a str>,
    glob_filter: Option<(&'a GlobMatcher, bool)>,
    limit: usize,
    visited_files: usize,
    files_scanned: usize,
    matches: Vec<String>,
    truncated: bool,
}

impl SearchState<'_> {
    fn rel_path(&self, path: &Path) -> String {
        path.strip_prefix(self.ws_canon)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    }

    fn consider_file(&mut self, open_path: &Path, display_path: &Path) {
        self.visited_files += 1;
        if let Some(want_ext) = self.ext_filter {
            let got_ext = display_path
                .extension()
                .and_then(|s| s.to_str())
                .map(str::to_ascii_lowercase);
            if got_ext.as_deref() != Some(want_ext) {
                return;
            }
        }
        if let Some((gm, has_slash)) = self.glob_filter {
            let rel = self.rel_path(display_path);
            let rel_to_root = display_path
                .strip_prefix(self.search_root_canon)
                .ok()
                .map(|p| p.to_string_lossy().replace('\\', "/"));
            let name = display_path
                .file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default();
            let matched = if has_slash {
                gm.is_match(&rel) || rel_to_root.as_deref().is_some_and(|r| gm.is_match(r))
            } else {
                gm.is_match(name.as_ref())
                    || gm.is_match(&rel)
                    || rel_to_root.as_deref().is_some_and(|r| gm.is_match(r))
            };
            if !matched {
                return;
            }
        }
        self.search_file(open_path, display_path);
    }

    fn search_file(&mut self, open_path: &Path, display_path: &Path) {
        // Single open + handle-bound metadata & bounded read eliminates size/symlink TOCTOU.
        let Ok(file) = fs::File::open(open_path) else {
            return;
        };
        let Ok(meta) = file.metadata() else { return };
        if !meta.is_file() || meta.len() > MAX_FILE_BYTES {
            return;
        }
        let mut bytes = Vec::with_capacity((meta.len() as usize).min(64 * 1024));
        if file
            .take(MAX_FILE_BYTES + 1)
            .read_to_end(&mut bytes)
            .is_err()
        {
            return;
        }
        if (bytes.len() as u64) > MAX_FILE_BYTES {
            return;
        }
        if bytes[..bytes.len().min(4096)].contains(&0) {
            return;
        }
        let Ok(text) = std::str::from_utf8(&bytes) else {
            return;
        };
        self.files_scanned += 1;
        let rel = self.rel_path(display_path);

        for (idx, line) in text.lines().enumerate() {
            if self.re.is_match(line) {
                if self.matches.len() >= self.limit {
                    self.truncated = true;
                    break;
                }
                let clipped: String = if line.chars().count() > MAX_LINE_CHARS {
                    let s: String = line.chars().take(MAX_LINE_CHARS).collect();
                    format!("{s}...")
                } else {
                    line.to_string()
                };
                self.matches.push(format!("{rel}:{}: {clipped}", idx + 1));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn grep_finds_matches_and_skips_git_dir() {
        let dir = std::env::temp_dir().join(format!("carnot-grep-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(dir.join("src")).expect("mkdir");
        fs::create_dir_all(dir.join(".git")).expect("mkdir git");
        fs::write(dir.join("src/main.rs"), "fn alpha() {}\nfn beta() {}\n").expect("w");
        fs::write(dir.join(".git/HEAD"), "fn alpha() {}\n").expect("w2");

        let grep = GrepTool::new(dir.clone(), Vec::new());
        let out = grep
            .execute(
                json!({"pattern": "fn alpha", "ext": "rs"}),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        assert!(!out.is_error, "grep error: {}", out.text_only());
        let txt = out.text_only();
        assert!(txt.contains("src/main.rs:1: fn alpha() {}"), "got: {txt}");
        assert!(!txt.contains(".git/HEAD"), "got: {txt}");

        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_honors_gitignore_wildcards_dir_rules_negation_and_nested_files() {
        let dir = std::env::temp_dir().join(format!("carnot-grep-gi-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(dir.join("src")).expect("mkdir");
        fs::create_dir_all(dir.join("generated/sub")).expect("mkdir gen");
        fs::create_dir_all(dir.join("logs")).expect("mkdir logs");
        fs::write(
            dir.join(".gitignore"),
            "*.log\n!important.log\ngenerated/\n",
        )
        .expect("gi");
        fs::write(dir.join("logs/.gitignore"), "local_ignored.txt\n").expect("nested gi");
        fs::write(dir.join("src/app.rs"), "NEEDLE_TOKEN in src\n").expect("src");
        fs::write(dir.join("debug.log"), "NEEDLE_TOKEN in ignored log\n").expect("log");
        fs::write(dir.join("important.log"), "NEEDLE_TOKEN in kept log\n").expect("kept");
        fs::write(dir.join("generated/sub/a.rs"), "NEEDLE_TOKEN in gen\n").expect("gen");
        fs::write(
            dir.join("logs/local_ignored.txt"),
            "NEEDLE_TOKEN in nested\n",
        )
        .expect("nested");

        let grep = GrepTool::new(dir.clone(), Vec::new());
        let out = grep
            .execute(json!({"pattern": "NEEDLE_TOKEN"}), CancellationToken::new())
            .await
            .expect("exec");
        assert!(!out.is_error, "grep error: {}", out.text_only());
        let txt = out.text_only();
        assert!(txt.contains("src/app.rs:1:"), "got: {txt}");
        assert!(
            txt.contains("important.log:1:"),
            "negated rule must match: {txt}"
        );
        assert!(
            !txt.contains("debug.log"),
            "ignored *.log must be skipped: {txt}"
        );
        assert!(
            !txt.contains("generated/"),
            "ignored dir must be skipped: {txt}"
        );
        assert!(
            !txt.contains("local_ignored.txt"),
            "nested .gitignore must apply: {txt}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_honors_git_info_exclude() {
        let dir =
            std::env::temp_dir().join(format!("carnot-grep-exclude-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(dir.join(".git/info")).expect("mkdir git info");
        fs::write(dir.join(".git/info/exclude"), "secret_notes.txt\n").expect("exclude");
        fs::write(dir.join("secret_notes.txt"), "EXCLUDE_NEEDLE\n").expect("secret");
        fs::write(dir.join("public.txt"), "EXCLUDE_NEEDLE\n").expect("public");

        let grep = GrepTool::new(dir.clone(), Vec::new());
        let out = grep
            .execute(
                json!({"pattern": "EXCLUDE_NEEDLE"}),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        assert!(!out.is_error, "grep error: {}", out.text_only());
        let txt = out.text_only();
        assert!(txt.contains("public.txt:1:"), "got: {txt}");
        assert!(
            !txt.contains("secret_notes.txt"),
            ".git/info/exclude must be honored: {txt}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_streams_and_stops_early_when_limit_reached() {
        let dir = std::env::temp_dir().join(format!("carnot-grep-early-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("mkdir");
        for i in 0..30 {
            fs::write(dir.join(format!("f_{i:02}.txt")), "HIT_LINE\n").expect("w");
        }
        let grep = GrepTool::new(dir.clone(), Vec::new());
        let out = grep
            .execute(
                json!({"pattern": "HIT_LINE", "limit": 3}),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        assert_eq!(out.details["matches"], 3, "exact limit");
        assert_eq!(out.details["truncated"], true, "truncated flag");
        assert_eq!(
            out.details["files_scanned"], 4,
            "early stop right after confirming (limit + 1)-th match"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_supports_glob_paths_braces_and_empty_strict_default() {
        let dir = std::env::temp_dir().join(format!("carnot-grep-glob-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(dir.join("crates/a/tests")).expect("mkdir");
        fs::create_dir_all(dir.join("crates/a/src")).expect("mkdir");
        fs::write(dir.join("crates/a/tests/unit.rs"), "needle_hit\n").expect("w");
        fs::write(dir.join("crates/a/src/lib.rs"), "needle_hit\n").expect("w");
        fs::write(dir.join("crates/a/Cargo.toml"), "needle_hit\n").expect("w");
        fs::write(dir.join("README.md"), "needle_hit\n").expect("w");

        let grep = GrepTool::new(dir.clone(), Vec::new());

        // Path glob: **/tests/*.rs
        let out_tests = grep
            .execute(
                json!({"pattern": "needle_hit", "glob": "**/tests/*.rs"}),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        let txt_tests = out_tests.text_only();
        assert!(
            txt_tests.contains("crates/a/tests/unit.rs"),
            "got: {txt_tests}"
        );
        assert!(!txt_tests.contains("src/lib.rs"), "got: {txt_tests}");
        assert!(!txt_tests.contains("Cargo.toml"), "got: {txt_tests}");

        // Brace glob: *.{rs,toml}
        let out_braces = grep
            .execute(
                json!({"pattern": "needle_hit", "glob": "*.{rs,toml}"}),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        let txt_braces = out_braces.text_only();
        assert!(
            txt_braces.contains("unit.rs") && txt_braces.contains("Cargo.toml"),
            "got: {txt_braces}"
        );
        assert!(!txt_braces.contains("README.md"), "got: {txt_braces}");

        // Empty strict-schema default: glob: "" -> ignored cleanly
        let out_empty = grep
            .execute(
                json!({"pattern": "needle_hit", "glob": ""}),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        assert!(
            out_empty.text_only().contains("README.md"),
            "got: {}",
            out_empty.text_only()
        );

        // Wildcard inside brace alternation: {*.rs,*.md}
        let out_wild_braces = grep
            .execute(
                json!({"pattern": "needle_hit", "glob": "{*.rs,*.md}"}),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        let txt_wb = out_wild_braces.text_only();
        assert!(
            txt_wb.contains("unit.rs") && txt_wb.contains("README.md"),
            "got: {txt_wb}"
        );
        assert!(!txt_wb.contains("Cargo.toml"), "got: {txt_wb}");

        // Negated character class `[!u]*.rs` matches `lib.rs` but excludes `unit.rs`
        let out_neg = grep
            .execute(
                json!({"pattern": "needle_hit", "glob": "[!u]*.rs"}),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        let txt_neg = out_neg.text_only();
        assert!(txt_neg.contains("src/lib.rs"), "got: {txt_neg}");
        assert!(!txt_neg.contains("unit.rs"), "got: {txt_neg}");

        let _ = fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_ignores_parent_gitignore_above_workspace_and_exact_limit_not_truncated() {
        let outer =
            std::env::temp_dir().join(format!("carnot-grep-outer-{}", uuid::Uuid::new_v4()));
        let ws = outer.join("workspace");
        fs::create_dir_all(ws.join("src")).expect("mkdir");
        // Hostile parent .gitignore above workspace root MUST NOT hide workspace files!
        fs::write(outer.join(".gitignore"), "*\n").expect("write outer gitignore");
        fs::write(ws.join("src/a.rs"), "needle_one\nneedle_two\n").expect("write a.rs");

        let grep = GrepTool::new(ws.clone(), Vec::new());
        let out = grep
            .execute(
                json!({"pattern": "needle_", "limit": 2}),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        assert!(
            out.text_only().contains("needle_one") && out.text_only().contains("needle_two"),
            "parent .gitignore above workspace must not mask workspace files: {}",
            out.text_only()
        );
        assert_eq!(
            out.details["truncated"], false,
            "exactly `limit` matches must not report truncated"
        );
        assert!(
            !out.text_only().contains("truncated"),
            "unexpected truncated footer: {}",
            out.text_only()
        );

        let _ = fs::remove_dir_all(&outer);
    }
}
