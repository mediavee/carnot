//! Native workspace-confined `Grep` tool (regex search, `.gitignore`-aware streaming walk,
//! zero shell spawn, eager-parallelizable).
//!
//! Runs concurrently with `Read` during streaming (`ToolEffects::global_read()`), honors
//! `.gitignore` + `.git/info/exclude` (including nested `.gitignore` and `!` negations),
//! skips binary files and build artifacts, stops walking immediately once `limit` matches are
//! found, and enforces strict symlink confinement.

use crate::files::resolve_confined_path;
use crate::{BoxFuture, Tool};
use carnot_core::{Result, ToolEffects, ToolOutput, ToolSpec};
use regex::{Regex, RegexBuilder};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::fs;
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
struct GitignoreRule {
    base_rel: String,
    negated: bool,
    dir_only: bool,
    anchored: bool,
    regex: Regex,
}

impl GitignoreRule {
    fn parse(base_rel: &str, raw_line: &str) -> Option<Self> {
        let mut line = raw_line.trim_end_matches(['\r', '\n']);
        // Strip trailing unescaped spaces
        while line.ends_with(' ') && !line.ends_with("\\ ") {
            line = &line[..line.len() - 1];
        }
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let negated = if let Some(rest) = line.strip_prefix('!') {
            line = rest;
            true
        } else {
            false
        };
        if let Some(rest) = line
            .strip_prefix("\\#")
            .or_else(|| line.strip_prefix("\\!"))
        {
            line = rest;
        }
        let dir_only = if let Some(rest) = line.strip_suffix('/') {
            line = rest;
            true
        } else {
            false
        };
        if line.is_empty() {
            return None;
        }
        let has_leading_slash = line.starts_with('/');
        let pat = line.strip_prefix('/').unwrap_or(line);
        let anchored = has_leading_slash || pat.contains('/');
        let regex = compile_gitignore_glob(pat)?;
        Some(Self {
            base_rel: base_rel.to_string(),
            negated,
            dir_only,
            anchored,
            regex,
        })
    }

    fn matches(&self, rel_path: &str, file_name: &str, is_dir: bool) -> bool {
        if self.dir_only && !is_dir {
            return false;
        }
        let sub_rel = if self.base_rel.is_empty() {
            rel_path
        } else if let Some(rest) = rel_path
            .strip_prefix(&self.base_rel)
            .and_then(|r| r.strip_prefix('/'))
        {
            rest
        } else {
            return false;
        };
        if self.anchored {
            self.regex.is_match(sub_rel)
        } else {
            self.regex.is_match(file_name) || self.regex.is_match(sub_rel)
        }
    }
}

fn glob_to_regex_body(pattern: &str, re: &mut String, depth: usize) {
    let chars: Vec<char> = pattern.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        match chars[i] {
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    // `**/` matches zero or more directories
                    if i + 2 < chars.len() && chars[i + 2] == '/' {
                        re.push_str("(?:.*/)?");
                        i += 3;
                    } else {
                        re.push_str(".*");
                        i += 2;
                    }
                } else {
                    re.push_str("[^/]*");
                    i += 1;
                }
            }
            '?' => {
                re.push_str("[^/]");
                i += 1;
            }
            '\\' if i + 1 < chars.len() => {
                re.push_str(&regex::escape(&chars[i + 1].to_string()));
                i += 2;
            }
            '[' => {
                if let Some(end_rel) = chars[i + 1..].iter().position(|&c| c == ']') {
                    let inner: Vec<char> = chars[i + 1..i + 1 + end_rel].to_vec();
                    let (negated, body) = match inner.first() {
                        Some('!' | '^') => (true, &inner[1..]),
                        _ => (false, &inner[..]),
                    };
                    if body.is_empty() {
                        re.push_str("\\[");
                        i += 1;
                    } else if negated {
                        re.push_str("[^/");
                        for &ch in body {
                            if ch == '/' {
                                continue;
                            }
                            if matches!(ch, '\\' | '[' | ']') {
                                re.push('\\');
                            }
                            re.push(ch);
                        }
                        re.push(']');
                        i += end_rel + 2;
                    } else {
                        // Exclude `/` from positive character classes
                        re.push('[');
                        let mut pushed = false;
                        for &ch in body {
                            if ch == '/' {
                                continue;
                            }
                            if matches!(ch, '\\' | '[' | ']') {
                                re.push('\\');
                            }
                            re.push(ch);
                            pushed = true;
                        }
                        if !pushed {
                            re.pop();
                            re.push_str("\\[");
                            i += 1;
                        } else {
                            re.push(']');
                            i += end_rel + 2;
                        }
                    }
                } else {
                    re.push_str("\\[");
                    i += 1;
                }
            }
            '{' if depth < 8 => {
                // Find matching `}` honoring nested `{...}` and split top-level comma alternatives.
                let mut brace_depth = 1usize;
                let mut j = i + 1;
                let mut alts: Vec<String> = Vec::new();
                let mut cur = String::new();
                while j < chars.len() && brace_depth > 0 {
                    match chars[j] {
                        '{' => {
                            brace_depth += 1;
                            cur.push('{');
                        }
                        '}' => {
                            brace_depth -= 1;
                            if brace_depth == 0 {
                                alts.push(std::mem::take(&mut cur));
                            } else {
                                cur.push('}');
                            }
                        }
                        ',' if brace_depth == 1 => {
                            alts.push(std::mem::take(&mut cur));
                        }
                        ch => cur.push(ch),
                    }
                    j += 1;
                }
                if brace_depth == 0 && !alts.is_empty() {
                    re.push_str("(?:");
                    for (idx, alt) in alts.iter().enumerate() {
                        if idx > 0 {
                            re.push('|');
                        }
                        glob_to_regex_body(alt, re, depth + 1);
                    }
                    re.push(')');
                    i = j;
                } else {
                    re.push_str("\\{");
                    i += 1;
                }
            }
            c => {
                re.push_str(&regex::escape(&c.to_string()));
                i += 1;
            }
        }
    }
}

fn compile_gitignore_glob(pattern: &str) -> Option<Regex> {
    let mut re = String::from("^");
    glob_to_regex_body(pattern, &mut re, 0);
    re.push('$');
    RegexBuilder::new(&re).size_limit(256 * 1024).build().ok()
}

fn load_gitignore_file(path: &Path, base_rel: &str, rules: &mut Vec<GitignoreRule>) {
    let Ok(content) = fs::read_to_string(path) else {
        return;
    };
    for line in content.lines() {
        if let Some(rule) = GitignoreRule::parse(base_rel, line) {
            rules.push(rule);
        }
    }
}

fn is_ignored_by_rules(rules: &[GitignoreRule], rel_path: &str, name: &str, is_dir: bool) -> bool {
    let mut ignored = false;
    for rule in rules {
        if rule.matches(rel_path, name, is_dir) {
            ignored = !rule.negated;
        }
    }
    ignored
}

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
                .map(|v| (v as usize).clamp(1, MAX_MATCH_LIMIT))
                .unwrap_or(DEFAULT_MATCH_LIMIT);

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
                    Some(ref pat) => {
                        let Some(gre) = compile_gitignore_glob(pat) else {
                            return Ok(ToolOutput::err_text(format!("Invalid glob `{pat}`")));
                        };
                        Some((gre, pat.contains('/')))
                    }
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
                    ws: &ws,
                    ws_canon: &ws_canon,
                    search_root_canon: &search_root_canon,
                    extra: &extra,
                    re: &re,
                    ext_filter: ext_filter.as_deref(),
                    glob_filter: glob_filter.as_ref().map(|(r, h)| (r, *h)),
                    limit,
                    cancel: &cancel,
                    visited_files: 0,
                    files_scanned: 0,
                    visited_dirs: HashSet::new(),
                    matches: Vec::new(),
                    truncated: false,
                };

                if target.is_file() {
                    state.search_file(&target);
                } else if target.is_dir() {
                    let mut rules = Vec::new();
                    // Load repo-level exclude + root `.gitignore` + any intermediate `.gitignore` down to `target`
                    load_gitignore_file(&ws_canon.join(".git/info/exclude"), "", &mut rules);
                    load_gitignore_file(&ws_canon.join(".gitignore"), "", &mut rules);
                    if let Ok(rel_target) = target.strip_prefix(&ws_canon) {
                        let mut cur = ws_canon.clone();
                        for comp in rel_target.components() {
                            cur.push(comp);
                            if cur != target {
                                let base = cur
                                    .strip_prefix(&ws_canon)
                                    .unwrap_or(&cur)
                                    .to_string_lossy()
                                    .replace('\\', "/");
                                load_gitignore_file(&cur.join(".gitignore"), &base, &mut rules);
                            }
                        }
                    }
                    state.walk_and_search(&target, &mut rules);
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
                        msg.push_str(&format!(
                            "\n[warning: file scan capped at {MAX_FILES_VISITED} files — narrow `path` or `ext`]"
                        ));
                    }
                    return Ok(ToolOutput::ok_text(msg).with_details(json!({
                        "matches": 0,
                        "files_scanned": state.files_scanned,
                        "scan_capped": scan_capped,
                    })));
                }

                let mut out = state.matches.join("\n");
                if state.truncated {
                    out.push_str(&format!(
                        "\n[truncated at {limit} matches — narrow `path`/`ext` or increase `limit`]"
                    ));
                }
                if scan_capped {
                    out.push_str(&format!(
                        "\n[warning: file scan capped at {MAX_FILES_VISITED} files — narrow `path` or `ext`]"
                    ));
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
    ws: &'a Path,
    ws_canon: &'a Path,
    search_root_canon: &'a Path,
    extra: &'a [PathBuf],
    re: &'a Regex,
    ext_filter: Option<&'a str>,
    glob_filter: Option<(&'a Regex, bool)>,
    limit: usize,
    cancel: &'a CancellationToken,
    visited_files: usize,
    files_scanned: usize,
    visited_dirs: HashSet<PathBuf>,
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

    fn walk_and_search(&mut self, dir: &Path, rules: &mut Vec<GitignoreRule>) {
        if self.truncated || self.visited_files >= MAX_FILES_VISITED || self.cancel.is_cancelled() {
            return;
        }
        let canon_dir = fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
        if !self.visited_dirs.insert(canon_dir.clone()) {
            return;
        }

        let prev_rule_len = rules.len();
        if canon_dir != self.ws_canon {
            let base = self.rel_path(&canon_dir);
            load_gitignore_file(&canon_dir.join(".gitignore"), &base, rules);
        }

        if let Ok(rd) = fs::read_dir(dir) {
            let mut entries: Vec<_> = rd.flatten().collect();
            entries.sort_by_key(|e| e.file_name());

            // Deterministic ordering: search files in this directory first, then recurse into subdirs.
            let mut subdirs: Vec<PathBuf> = Vec::new();
            for entry in entries {
                if self.truncated
                    || self.visited_files >= MAX_FILES_VISITED
                    || self.cancel.is_cancelled()
                {
                    break;
                }
                let path = entry.path();
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                let Ok(ft) = entry.file_type() else { continue };
                if ft.is_symlink()
                    && resolve_confined_path(self.ws, self.extra, &path.to_string_lossy()).is_err()
                {
                    continue;
                }
                let (is_dir, is_file) = if ft.is_symlink() {
                    match fs::metadata(&path) {
                        Ok(m) => (m.is_dir(), m.is_file()),
                        Err(_) => continue,
                    }
                } else {
                    (ft.is_dir(), ft.is_file())
                };

                let rel = self.rel_path(&path);
                if is_dir {
                    if ALWAYS_SKIP_DIRS.contains(&name_str.as_ref())
                        || is_ignored_by_rules(rules, &rel, &name_str, true)
                    {
                        continue;
                    }
                    subdirs.push(path);
                } else if is_file {
                    if is_ignored_by_rules(rules, &rel, &name_str, false) {
                        continue;
                    }
                    self.visited_files += 1;
                    if let Some(want_ext) = self.ext_filter {
                        let got_ext = path
                            .extension()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_ascii_lowercase());
                        if got_ext.as_deref() != Some(want_ext) {
                            continue;
                        }
                    }
                    if let Some((gre, has_slash)) = self.glob_filter {
                        let rel_to_root = path
                            .strip_prefix(self.search_root_canon)
                            .ok()
                            .map(|p| p.to_string_lossy().replace('\\', "/"));
                        let matched = if has_slash {
                            gre.is_match(&rel)
                                || rel_to_root.as_deref().is_some_and(|r| gre.is_match(r))
                        } else {
                            gre.is_match(&name_str)
                                || gre.is_match(&rel)
                                || rel_to_root.as_deref().is_some_and(|r| gre.is_match(r))
                        };
                        if !matched {
                            continue;
                        }
                    }
                    self.search_file(&path);
                }
            }

            for sub in subdirs {
                if self.truncated
                    || self.visited_files >= MAX_FILES_VISITED
                    || self.cancel.is_cancelled()
                {
                    break;
                }
                self.walk_and_search(&sub, rules);
            }
        }

        rules.truncate(prev_rule_len);
    }

    fn search_file(&mut self, file: &Path) {
        let Ok(meta) = fs::metadata(file) else { return };
        if meta.len() > MAX_FILE_BYTES {
            return;
        }
        let Ok(bytes) = fs::read(file) else { return };
        if bytes[..bytes.len().min(4096)].contains(&0) {
            return;
        }
        let Ok(text) = std::str::from_utf8(&bytes) else {
            return;
        };
        self.files_scanned += 1;
        let rel = self.rel_path(file);

        for (idx, line) in text.lines().enumerate() {
            if self.re.is_match(line) {
                let clipped: String = if line.chars().count() > MAX_LINE_CHARS {
                    let s: String = line.chars().take(MAX_LINE_CHARS).collect();
                    format!("{s}...")
                } else {
                    line.to_string()
                };
                self.matches.push(format!("{rel}:{}: {clipped}", idx + 1));
                if self.matches.len() >= self.limit {
                    self.truncated = true;
                    break;
                }
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
        assert!(!out.is_error);
        let txt = out.text_only();
        assert!(txt.contains("src/main.rs:1: fn alpha() {}"));
        assert!(!txt.contains(".git/HEAD"));

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
        assert!(!out.is_error);
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
        assert_eq!(out.details["matches"], 3);
        assert_eq!(out.details["truncated"], true);
        assert_eq!(out.details["files_scanned"], 3);

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
        assert!(txt_braces.contains("unit.rs") && txt_braces.contains("Cargo.toml"));
        assert!(!txt_braces.contains("README.md"));

        // Empty strict-schema default: glob: "" -> ignored cleanly
        let out_empty = grep
            .execute(
                json!({"pattern": "needle_hit", "glob": ""}),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        assert!(out_empty.text_only().contains("README.md"));

        // Wildcard inside brace alternation: {*.rs,*.md}
        let out_wild_braces = grep
            .execute(
                json!({"pattern": "needle_hit", "glob": "{*.rs,*.md}"}),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        let txt_wb = out_wild_braces.text_only();
        assert!(txt_wb.contains("unit.rs") && txt_wb.contains("README.md"));
        assert!(!txt_wb.contains("Cargo.toml"));

        // Negated character class `[!u]*.rs` matches `lib.rs` but excludes `unit.rs`
        let out_neg = grep
            .execute(
                json!({"pattern": "needle_hit", "glob": "[!u]*.rs"}),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        let txt_neg = out_neg.text_only();
        assert!(txt_neg.contains("lib.rs"), "got: {txt_neg}");
        assert!(!txt_neg.contains("unit.rs"), "got: {txt_neg}");

        // Relative glob scoped to `path` subdirectory: path="crates/a", glob="src/*.rs"
        let out_sub = grep
            .execute(
                json!({"pattern": "needle_hit", "path": "crates/a", "glob": "src/*.rs"}),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        let txt_sub = out_sub.text_only();
        assert!(txt_sub.contains("crates/a/src/lib.rs"), "got: {txt_sub}");
        assert!(!txt_sub.contains("unit.rs"), "got: {txt_sub}");

        let _ = fs::remove_dir_all(&dir);
    }
}
