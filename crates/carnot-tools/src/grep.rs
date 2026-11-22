//! Native workspace-confined `Grep` tool (regex search, zero shell spawn, eager-parallelizable).
//!
//! Inspired by `lythe` and Claude Code: eliminates slow shell+sandbox spawns for codebase search,
//! runs concurrently with `Read` during streaming (`ToolEffects::global_read()`), auto-skips
//! build artifacts / `.git` / `.carnot` / binary files, and enforces strict symlink confinement.

use crate::files::resolve_confined_path;
use crate::{BoxFuture, Tool};
use carnot_core::{Result, ToolEffects, ToolOutput, ToolSpec};
use regex::RegexBuilder;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

const DEFAULT_MATCH_LIMIT: usize = 60;
const MAX_MATCH_LIMIT: usize = 200;
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;
const MAX_FILES_VISITED: usize = 5_000;
const MAX_LINE_CHARS: usize = 240;

const SKIP_DIRS: &[&str] = &[
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
                .unwrap_or(".")
                .to_string();
            let ext_filter = input
                .get("ext")
                .and_then(Value::as_str)
                .map(|s| s.trim_start_matches('.').to_ascii_lowercase());
            let case_insensitive = input
                .get("case_insensitive")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let limit = input
                .get("limit")
                .and_then(Value::as_u64)
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

                let target = match resolve_confined_path(&ws, &extra, &raw_path) {
                    Ok(p) => p,
                    Err(msg) => return Ok(ToolOutput::err_text(msg)),
                };
                let ws_canon = fs::canonicalize(&ws).unwrap_or_else(|_| ws.clone());

                let mut files: Vec<PathBuf> = Vec::new();
                let mut visited = 0usize;
                let mut visited_dirs: HashSet<PathBuf> = HashSet::new();
                if target.is_file() {
                    files.push(target);
                } else if target.is_dir() {
                    collect_files(
                        &target,
                        &ws,
                        &extra,
                        ext_filter.as_deref(),
                        &cancel,
                        &mut visited,
                        &mut visited_dirs,
                        &mut files,
                    );
                } else {
                    return Ok(ToolOutput::err_text(format!("Path not found: {raw_path}")));
                }
                if cancel.is_cancelled() {
                    return Ok(ToolOutput::err_text("Grep cancelled"));
                }
                files.sort();

                let mut matches: Vec<String> = Vec::new();
                let mut truncated = false;
                for file in &files {
                    if cancel.is_cancelled() {
                        return Ok(ToolOutput::err_text("Grep cancelled"));
                    }
                    let Ok(meta) = fs::metadata(file) else {
                        continue;
                    };
                    if meta.len() > MAX_FILE_BYTES {
                        continue;
                    }
                    let Ok(bytes) = fs::read(file) else { continue };
                    // Skip binary files (NUL byte in first 4 KiB)
                    if bytes[..bytes.len().min(4096)].contains(&0) {
                        continue;
                    }
                    let Ok(text) = std::str::from_utf8(&bytes) else {
                        continue;
                    };
                    let rel = file
                        .strip_prefix(&ws_canon)
                        .unwrap_or(file)
                        .to_string_lossy()
                        .replace('\\', "/");

                    for (idx, line) in text.lines().enumerate() {
                        if re.is_match(line) {
                            let clipped: String = if line.chars().count() > MAX_LINE_CHARS {
                                let s: String = line.chars().take(MAX_LINE_CHARS).collect();
                                format!("{s}...")
                            } else {
                                line.to_string()
                            };
                            matches.push(format!("{rel}:{}: {clipped}", idx + 1));
                            if matches.len() >= limit {
                                truncated = true;
                                break;
                            }
                        }
                    }
                    if truncated {
                        break;
                    }
                }

                let scan_capped = visited >= MAX_FILES_VISITED;
                if matches.is_empty() {
                    let mut msg = format!("[no matches for `{pattern}` in {raw_path}]");
                    if scan_capped {
                        msg.push_str(&format!(
                            "\n[warning: file scan capped at {MAX_FILES_VISITED} files — narrow `path` or `ext`]"
                        ));
                    }
                    return Ok(ToolOutput::ok_text(msg).with_details(json!({
                        "matches": 0,
                        "files_scanned": files.len(),
                        "scan_capped": scan_capped,
                    })));
                }

                let mut out = matches.join("\n");
                if truncated {
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
                    "matches": matches.len(),
                    "truncated": truncated,
                    "files_scanned": files.len(),
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

#[allow(clippy::too_many_arguments)]
fn collect_files(
    dir: &Path,
    ws: &Path,
    extra: &[PathBuf],
    ext_filter: Option<&str>,
    cancel: &CancellationToken,
    visited: &mut usize,
    visited_dirs: &mut HashSet<PathBuf>,
    out: &mut Vec<PathBuf>,
) {
    if *visited >= MAX_FILES_VISITED || cancel.is_cancelled() {
        return;
    }
    let canon_dir = fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    if !visited_dirs.insert(canon_dir) {
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        if *visited >= MAX_FILES_VISITED {
            break;
        }
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let Ok(ft) = entry.file_type() else { continue };
        // Never follow symlinks escaping confinement
        if ft.is_symlink() && resolve_confined_path(ws, extra, &path.to_string_lossy()).is_err() {
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
        if is_dir {
            if SKIP_DIRS.contains(&name_str.as_ref()) {
                continue;
            }
            collect_files(
                &path,
                ws,
                extra,
                ext_filter,
                cancel,
                visited,
                visited_dirs,
                out,
            );
        } else if is_file {
            *visited += 1;
            if let Some(want_ext) = ext_filter {
                let got_ext = path
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_ascii_lowercase());
                if got_ext.as_deref() != Some(want_ext) {
                    continue;
                }
            }
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn grep_finds_matches_and_skips_git_dir() {
        let dir = std::env::temp_dir().join(format!("carnot-grep-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(dir.join(".git")).expect("mkdir");
        fs::create_dir_all(dir.join("src")).expect("mkdir");
        fs::write(dir.join(".git/secret.txt"), "needle_token\n").expect("write");
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn needle_token() -> i32 { 42 }\n",
        )
        .expect("write");

        let tool = GrepTool::new(dir.clone(), Vec::new());
        let out = tool
            .execute(
                json!({"pattern": "needle_token", "ext": "rs"}),
                CancellationToken::new(),
            )
            .await
            .expect("exec");
        assert!(!out.is_error);
        let txt = out.text_only();
        assert!(txt.contains("src/lib.rs:1:"));
        assert!(!txt.contains(".git/secret.txt"));

        let _ = fs::remove_dir_all(&dir);
    }
}
