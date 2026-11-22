//! Execution-verified Turn-1 repo micro-skills + `AGENTS.md`/`CLAUDE.md`/`.carnot/skills/*.md` loader.
//!
//! Hardened against Opus N14:
//!   - Runs off the async runtime in `tokio::task::spawn_blocking`
//!   - Checks toolchain availability on `PATH` (ZERO untrusted repo binary execution!)
//!   - Uses `host_safe_git_command` (`-c core.fsmonitor=false -c core.hooksPath=/dev/null`)
//!   - Loads project instructions (`AGENTS.md`, `CLAUDE.md`, `.carnot/skills/*.md`) into Turn-1
//!     user context ONLY (capped at 8 KiB) — NEVER inside the static cached system prompt!

use crate::{BoxFuture, Module};
use carnot_core::{CarnotConfig, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const MAX_INSTRUCTIONS_BYTES: usize = 8 * 1024;
const OVERVIEW_MAX_FILES: usize = 60;
const OVERVIEW_MAX_VISITED: usize = 4_000;
const OVERVIEW_SKIP_DIRS: &[&str] = &[
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

#[derive(Debug, Default)]
pub struct ExecMicroSkillsModule;

impl ExecMicroSkillsModule {
    pub fn new() -> Self {
        Self
    }
}

fn has_binary_on_path(bin: &str) -> bool {
    let Some(path_var) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path_var).any(|dir| dir.join(bin).is_file())
}

fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn read_confined_markdown(root: &Path, path: &Path) -> Option<String> {
    // Reject symlinks escaping workspace root (fixes Opus #11: `.carnot/skills/a.md -> ~/.aws/credentials`)
    let meta = fs::symlink_metadata(path).ok()?;
    if meta.file_type().is_symlink() {
        return None;
    }
    let resolved =
        carnot_tools::files::resolve_confined_path(root, &[], &path.to_string_lossy()).ok()?;
    if !resolved.is_file() {
        return None;
    }
    fs::read_to_string(resolved).ok()
}

fn load_repo_instructions(root: &Path) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    let mut budget = MAX_INSTRUCTIONS_BYTES;

    for fname in ["AGENTS.md", "CLAUDE.md"] {
        if budget == 0 {
            break;
        }
        let p = root.join(fname);
        if let Some(txt) = read_confined_markdown(root, &p) {
            let trimmed = txt.trim();
            if !trimmed.is_empty() {
                let slice = truncate_utf8(trimmed, budget);
                let suffix = if slice.len() < trimmed.len() {
                    "\n[... truncated at 8 KiB budget ...]"
                } else {
                    ""
                };
                parts.push(format!("[project-instructions:{fname}]\n{slice}{suffix}"));
                budget = budget.saturating_sub(slice.len());
            }
        }
    }

    let skills_dir = root.join(".carnot").join("skills");
    if budget > 0
        && skills_dir.is_dir()
        && !fs::symlink_metadata(&skills_dir)
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(true)
        && let Ok(rd) = fs::read_dir(&skills_dir)
    {
        let mut skill_files: Vec<_> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("md"))
            .collect();
        skill_files.sort();
        for sf in skill_files {
            if budget == 0 {
                break;
            }
            if let Some(txt) = read_confined_markdown(root, &sf) {
                let trimmed = txt.trim();
                if !trimmed.is_empty() {
                    let slice = truncate_utf8(trimmed, budget);
                    let name = sf
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("skill.md");
                    parts.push(format!("[carnot-skill:{name}]\n{slice}"));
                    budget = budget.saturating_sub(slice.len());
                }
            }
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

/// Parses optional YAML frontmatter (`name`, `description`) from an Agent Skills `SKILL.md`.
fn parse_skill_frontmatter(raw: &str) -> (Option<String>, Option<String>) {
    let mut name: Option<String> = None;
    let mut desc: Option<String> = None;
    let normalized = raw.replace("\r\n", "\n");
    let trimmed = normalized.trim_start();
    if let Some(rest) = trimmed.strip_prefix("---")
        && let Some(end) = rest.find("\n---")
    {
        for line in rest[..end].lines() {
            let l = line.trim();
            if let Some(v) = l.strip_prefix("name:") {
                let s = v.trim().trim_matches('"').trim_matches('\'').trim();
                if !s.is_empty() {
                    name = Some(s.to_string());
                }
            } else if let Some(v) = l.strip_prefix("description:") {
                let s = v.trim().trim_matches('"').trim_matches('\'').trim();
                if !s.is_empty() {
                    desc = Some(s.to_string());
                }
            }
        }
    }
    if desc.is_none() {
        for line in trimmed.lines() {
            let l = line.trim();
            if !l.is_empty() && !l.starts_with("---") && !l.starts_with('#') {
                desc = Some(truncate_utf8(l, 160).to_string());
                break;
            }
        }
    }
    (name, desc)
}

/// Discovers open Agent Skills (`<dir>/<skill>/SKILL.md`) in standard workspace paths
/// Discovers Agent Skills (`<skill>/SKILL.md`) under workspace directories
/// (`.agents/skills`, `.claude/skills`) and any explicit `--skills-dir` directories.
/// Returns `(name, display_path, description)` tuples in deterministic sorted order.
pub fn discover_agent_skills(root: &Path, extra_dirs: &[PathBuf]) -> Vec<(String, String, String)> {
    let mut search_roots: Vec<(PathBuf, bool)> = vec![
        (root.join(".agents").join("skills"), false),
        (root.join(".claude").join("skills"), false),
    ];
    for d in extra_dirs {
        search_roots.push((d.clone(), true));
    }

    let mut seen_paths: BTreeSet<PathBuf> = BTreeSet::new();
    let mut found: Vec<(String, String, String)> = Vec::new();

    for (dir, is_explicit) in search_roots {
        if !dir.is_dir() {
            continue;
        }
        if !is_explicit
            && fs::symlink_metadata(&dir)
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(true)
        {
            continue;
        }
        let Ok(rd) = fs::read_dir(&dir) else {
            continue;
        };
        let mut subdirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        subdirs.sort();
        for sub in subdirs {
            let skill_md = sub.join("SKILL.md");
            if !skill_md.is_file() {
                continue;
            }
            let canon = fs::canonicalize(&skill_md).unwrap_or_else(|_| skill_md.clone());
            if !seen_paths.insert(canon.clone()) {
                continue;
            }
            let content = if is_explicit {
                fs::read_to_string(&skill_md).ok()
            } else {
                read_confined_markdown(root, &skill_md)
            };
            let Some(raw) = content else {
                continue;
            };
            let (fm_name, fm_desc) = parse_skill_frontmatter(&raw);
            let fallback_name = sub
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("skill")
                .to_string();
            let name = fm_name.unwrap_or(fallback_name);
            let desc = fm_desc.unwrap_or_else(|| "Workspace skill instructions".to_string());
            let disp_path = skill_md
                .strip_prefix(root)
                .unwrap_or(&skill_md)
                .display()
                .to_string();
            found.push((name, disp_path, truncate_utf8(&desc, 180).to_string()));
            if found.len() >= 32 {
                break;
            }
        }
    }
    found
}

/// Emits a compact progressive-disclosure index into Turn-1 user context ONLY (zero static prefix drift).
fn load_agent_skills(root: &Path, extra_dirs: &[PathBuf]) -> Option<String> {
    let found = discover_agent_skills(root, extra_dirs);
    if found.is_empty() {
        None
    } else {
        let entries: Vec<String> = found
            .into_iter()
            .map(|(name, disp_path, desc)| format!("- {name} ({disp_path}): {desc}"))
            .collect();
        Some(format!(
            "[agent-skills]\nRead a skill's SKILL.md with `Read` when relevant:\n{}",
            entries.join("\n")
        ))
    }
}

/// Pure-Rust bounded project file overview (inspired by `lythe`: 0 subprocesses, ~0.3ms,
/// injected ONLY into Turn-1 user context so the static cached prefix stays 100% invariant!).
fn build_project_overview(root: &Path) -> (Option<String>, bool) {
    let mut files: Vec<String> = Vec::new();
    let mut visited = 0usize;
    let mut truncated = false;
    let mut has_py_file = false;
    walk_overview(
        root,
        root,
        &mut visited,
        &mut truncated,
        &mut has_py_file,
        &mut files,
    );
    if files.is_empty() {
        return (None, false);
    }
    files.sort();
    let rendered = if files.len() <= OVERVIEW_MAX_FILES && !truncated {
        let mut root_files: Vec<&str> = Vec::new();
        let mut by_dir: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for f in &files {
            match f.rsplit_once('/') {
                Some((dir, name)) => by_dir.entry(dir).or_default().push(name),
                None => root_files.push(f.as_str()),
            }
        }
        let mut lines: Vec<String> = root_files.into_iter().map(str::to_owned).collect();
        for (dir, names) in by_dir {
            lines.push(format!("{dir}/: {}", names.join(", ")));
        }
        lines.join("\n")
    } else {
        let mut top: BTreeMap<String, usize> = BTreeMap::new();
        for f in &files {
            let key = match f.split_once('/') {
                Some((dir, _)) => format!("{dir}/"),
                None => f.clone(),
            };
            *top.entry(key).or_default() += 1;
        }
        let more = if truncated { "+" } else { "" };
        top.into_iter()
            .map(|(k, n)| {
                if k.ends_with('/') {
                    format!("{k} ({n}{more} files)")
                } else {
                    k
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    (Some(rendered), has_py_file)
}

fn walk_overview(
    root: &Path,
    dir: &Path,
    visited: &mut usize,
    truncated: &mut bool,
    has_py_file: &mut bool,
    out: &mut Vec<String>,
) {
    if *visited >= OVERVIEW_MAX_VISITED {
        *truncated = true;
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        if *visited >= OVERVIEW_MAX_VISITED {
            *truncated = true;
            break;
        }
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_symlink() {
            continue;
        }
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let path = entry.path();
        if ft.is_dir() {
            if OVERVIEW_SKIP_DIRS.contains(&name_str.as_ref()) {
                continue;
            }
            walk_overview(root, &path, visited, truncated, has_py_file, out);
        } else if ft.is_file() {
            *visited += 1;
            if path.extension().and_then(|e| e.to_str()) == Some("py") {
                *has_py_file = true;
            }
            if let Ok(rel) = path.strip_prefix(root) {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
}

fn fast_git_branch(root: &Path) -> Option<String> {
    let head_path = root.join(".git").join("HEAD");
    if let Ok(raw) = fs::read_to_string(&head_path) {
        let trimmed = raw.trim();
        if let Some(branch) = trimmed.strip_prefix("ref: refs/heads/") {
            if !branch.is_empty() {
                return Some(branch.to_string());
            }
        } else if !trimmed.is_empty() {
            return Some(trimmed.chars().take(12).collect());
        }
    }
    let out = carnot_sandbox::host_safe_git_command(root)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let branch = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if branch.is_empty() {
        None
    } else {
        Some(branch)
    }
}

impl Module for ExecMicroSkillsModule {
    fn name(&self) -> &str {
        "exec_microskills"
    }

    fn on_session_start(&self, config: &CarnotConfig) -> BoxFuture<'_, Result<Option<String>>> {
        let root = config.workspace_root.clone();
        let extra_skills_dirs = config.skills_dirs.clone();
        Box::pin(async move {
            let res = tokio::task::spawn_blocking(move || {
                let mut facts: Vec<String> = Vec::new();
                facts.push(format!("cwd: {}", root.display()));

                let userland = if cfg!(target_os = "macos") {
                    "macOS (BSD userland: `sed -i ''`, no GNU-only flags)"
                } else if cfg!(target_os = "windows") {
                    "Windows"
                } else if cfg!(target_os = "linux") {
                    "Linux (GNU userland)"
                } else {
                    std::env::consts::OS
                };
                facts.push(format!("platform: {userland}"));

                let (overview, has_py_file) = build_project_overview(&root);

                if root.join("Cargo.toml").exists() && has_binary_on_path("cargo") {
                    facts.push("toolchain: rust/cargo (`cargo check`, `cargo test`)".into());
                }
                if root.join("package.json").exists() {
                    if root.join("bun.lock").exists() && has_binary_on_path("bun") {
                        facts.push("toolchain: bun (`bun test`)".into());
                    } else if root.join("pnpm-lock.yaml").exists() && has_binary_on_path("pnpm") {
                        facts.push("toolchain: pnpm (`pnpm test`)".into());
                    } else if has_binary_on_path("node") {
                        facts.push("toolchain: node (`node --test`, `npm test`)".into());
                    }
                }
                if (has_py_file
                    || root.join("pyproject.toml").exists()
                    || root.join("pytest.ini").exists()
                    || root.join("setup.py").exists()
                    || root.join("requirements.txt").exists())
                    && has_binary_on_path("python3")
                {
                    let cmd = if has_binary_on_path("pytest") {
                        "pytest` or `python3 -m unittest"
                    } else {
                        "python3 -m unittest"
                    };
                    facts.push(format!("toolchain: python3 (`{cmd}`)"));
                }
                if root.join("go.mod").exists() && has_binary_on_path("go") {
                    facts.push("toolchain: go (`go test ./...`)".into());
                }

                // Fast zero-subprocess git branch probe (reads `.git/HEAD` directly in ~0.01ms; falls back safely for worktrees)
                if let Some(branch) = fast_git_branch(&root) {
                    facts.push(format!("git_branch: {branch}"));
                }

                let mut out = format!("[workspace-facts]\n{}", facts.join("\n"));
                if let Some(files_map) = overview {
                    out.push_str("\nfiles:\n");
                    out.push_str(&files_map);
                }
                if let Some(instr) = load_repo_instructions(&root) {
                    out.push_str("\n\n");
                    out.push_str(&instr);
                }
                if let Some(skills) = load_agent_skills(&root, &extra_skills_dirs) {
                    out.push_str("\n\n");
                    out.push_str(&skills);
                }
                out
            })
            .await
            .unwrap_or_default();

            if res.is_empty() {
                Ok(None)
            } else {
                Ok(Some(res))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_agent_skills_from_workspace_and_explicit_dirs() {
        let dir = std::env::temp_dir().join(format!("carnot-skills-{}", uuid::Uuid::new_v4()));
        let ws_skill = dir.join(".agents/skills/perf-audit");
        fs::create_dir_all(&ws_skill).expect("mkdir");
        fs::write(
            ws_skill.join("SKILL.md"),
            "---\nname: perf-audit\ndescription: Profile web layout shifts and latency.\n---\n# Body\n",
        )
        .expect("write");

        let extra_root =
            std::env::temp_dir().join(format!("carnot-extra-skills-{}", uuid::Uuid::new_v4()));
        let extra_skill = extra_root.join("db-migrate");
        fs::create_dir_all(&extra_skill).expect("mkdir");
        fs::write(
            extra_skill.join("SKILL.md"),
            "---\nname: db-migrate\ndescription: Safe zero-downtime migrations.\n---\n",
        )
        .expect("write");

        let rendered =
            load_agent_skills(&dir, std::slice::from_ref(&extra_root)).expect("skills found");
        assert!(rendered.contains("[agent-skills]"));
        assert!(rendered.contains("perf-audit"));
        assert!(rendered.contains("Profile web layout shifts and latency."));
        assert!(rendered.contains("db-migrate"));
        assert!(rendered.contains("Safe zero-downtime migrations."));

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&extra_root);
    }
}
