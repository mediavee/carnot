//! Safe Warm-Fork Worktree Isolator (opt-in).
//!
//! Creates a detached `git worktree add --detach` with a dedicated `CARGO_TARGET_DIR` AND a
//! clean local baseline commit (`base_sha`) capturing the parent's dirty tracked + untracked
//! files at fork time. Guarantees:
//!   - Monorepo subdirectory safe: resolves `git rev-parse --show-toplevel` + `--show-prefix`
//!     so `worker_workspace` points at the exact matching subdirectory inside the worktree and
//!     `git apply` always runs from `repo_root`.
//!   - `export_worker_diff()` returns strictly the worker's own edits since `base_sha`.
//!   - `land_into_parent(&mut self)` checks `git apply --check --binary` before mutating the
//!     parent workspace, then commits & advances `base_sha` so subsequent `land` calls on a
//!     `keep_alive` worker apply only newly made edits.
//!   - Per-user `0700` fork root (`~/.carnot/forks`) + deterministic `Drop` cleanup & prune.

use carnot_core::{CarnotError, Result};
use std::fmt::Write as _;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const MAX_UNTRACKED_COPY_BYTES: u64 = 5 * 1024 * 1024;

#[derive(Debug)]
pub enum LandOutcome {
    /// Worker made no file changes relative to its fork/land baseline.
    NoChanges,
    /// Worker diff was cleanly and atomically applied to the parent workspace.
    Applied { stat: String },
    /// `git apply --check` detected a conflict; parent workspace was NOT modified.
    Conflict { stderr: String, diff_bytes: Vec<u8> },
}

#[derive(Debug)]
pub struct IsolatedForkWorktree {
    /// Top-level git repository root (parent checkout).
    repo_root: PathBuf,
    /// Top-level detached worktree root.
    pub worktree_path: PathBuf,
    /// Working directory for the worker (`worktree_path` joined with any monorepo subdir prefix).
    pub worker_workspace: PathBuf,
    pub cargo_target_dir: PathBuf,
    /// Detached baseline commit inside the worktree containing parent HEAD + dirty state at fork
    /// time (advanced after every successful `land_into_parent`).
    pub base_sha: String,
    /// Set if a previous `land` patched the parent workspace but failed to commit the new worktree
    /// baseline; refuses subsequent `land` calls so the already-applied diff is never re-applied.
    land_checkpoint_failed: bool,
}

#[cfg(unix)]
fn is_process_alive(pid: u32) -> bool {
    if pid == std::process::id() {
        return true;
    }
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_or(true, |s| s.success())
}

#[cfg(not(unix))]
fn is_process_alive(pid: u32) -> bool {
    pid == std::process::id()
}

fn forks_base_dir() -> Result<PathBuf> {
    let base = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map_or_else(
            || std::env::temp_dir().join(format!("carnot-forks-{}", std::process::id())),
            |h| PathBuf::from(h).join(".carnot").join("forks"),
        );
    std::fs::create_dir_all(&base)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&base, std::fs::Permissions::from_mode(0o700));
    }
    Ok(base)
}

fn commit_worktree_checkpoint(worktree_path: &Path, message: &str) -> Result<String> {
    let add_out = carnot_sandbox::host_safe_git_command(worktree_path)
        .args(["add", "-A"])
        .output()?;
    if !add_out.status.success() {
        return Err(CarnotError::Session(format!(
            "failed to stage worktree checkpoint: {}",
            String::from_utf8_lossy(&add_out.stderr).trim()
        )));
    }
    let commit_out = carnot_sandbox::host_safe_git_command(worktree_path)
        .args([
            "-c",
            "user.name=carnot",
            "-c",
            "user.email=carnot@local",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "--no-verify",
            "-m",
            message,
        ])
        .output()?;
    if !commit_out.status.success() {
        return Err(CarnotError::Session(format!(
            "failed to commit worktree checkpoint: {}",
            String::from_utf8_lossy(&commit_out.stderr).trim()
        )));
    }
    let rev_out = carnot_sandbox::host_safe_git_command(worktree_path)
        .args(["rev-parse", "HEAD"])
        .output()?;
    let sha = String::from_utf8_lossy(&rev_out.stdout).trim().to_string();
    if sha.is_empty() {
        return Err(CarnotError::Session(
            "failed to resolve worktree checkpoint commit SHA".into(),
        ));
    }
    Ok(sha)
}

impl IsolatedForkWorktree {
    pub fn create(parent_workspace: &Path, fork_id: &str) -> Result<Self> {
        // Resolve git top-level root and relative monorepo subdirectory prefix
        let top_out = carnot_sandbox::host_safe_git_command(parent_workspace)
            .args(["rev-parse", "--show-toplevel"])
            .output()?;
        if !top_out.status.success() {
            return Err(CarnotError::Session(format!(
                "isolated_worktree requires a git repository ({}): {}",
                parent_workspace.display(),
                String::from_utf8_lossy(&top_out.stderr).trim()
            )));
        }
        let repo_root = PathBuf::from(String::from_utf8_lossy(&top_out.stdout).trim());

        let prefix_out = carnot_sandbox::host_safe_git_command(parent_workspace)
            .args(["rev-parse", "--show-prefix"])
            .output()?;
        let rel_prefix = if prefix_out.status.success() {
            String::from_utf8_lossy(&prefix_out.stdout)
                .trim()
                .trim_end_matches('/')
                .to_string()
        } else {
            String::new()
        };

        let base = forks_base_dir()?;
        let worktree_path = base.join(format!("wt-{fork_id}"));
        let cargo_target_dir = base.join(format!("target-{fork_id}"));
        std::fs::create_dir_all(&cargo_target_dir)?;
        let _ = std::fs::write(
            cargo_target_dir.join(".carnot-owner-pid"),
            std::process::id().to_string(),
        );

        let out = carnot_sandbox::host_safe_git_command(&repo_root)
            .args([
                "worktree",
                "add",
                "--detach",
                &worktree_path.to_string_lossy(),
                "HEAD",
            ])
            .output()?;
        if !out.status.success() {
            let _ = std::fs::remove_dir_all(&cargo_target_dir);
            return Err(CarnotError::Session(format!(
                "git worktree add failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }

        let worker_workspace = if rel_prefix.is_empty() {
            worktree_path.clone()
        } else {
            worktree_path.join(&rel_prefix)
        };

        // Guard ensures cleanup if any baseline replication step fails
        let mut guard = Self {
            repo_root: repo_root.clone(),
            worktree_path: worktree_path.clone(),
            worker_workspace,
            cargo_target_dir,
            base_sha: String::new(),
            land_checkpoint_failed: false,
        };

        // 1. Replicate parent's uncommitted tracked changes (binary-safe, verified exit status)
        let diff_out = carnot_sandbox::host_safe_git_command(&repo_root)
            .args(["diff", "HEAD", "--binary"])
            .output()?;
        if diff_out.status.success() && !diff_out.stdout.is_empty() {
            let mut child = carnot_sandbox::host_safe_git_command(&worktree_path)
                .args(["apply", "--allow-empty", "--binary"])
                .stdin(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?;
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(&diff_out.stdout)?;
            }
            let res = child.wait_with_output()?;
            if !res.status.success() {
                return Err(CarnotError::Session(format!(
                    "failed to replicate parent dirty diff into worktree: {}",
                    String::from_utf8_lossy(&res.stderr).trim()
                )));
            }
        }

        // 2. Replicate parent's untracked non-ignored files (skipping .carnot/, target/, oversized files)
        if let Ok(untracked_out) = carnot_sandbox::host_safe_git_command(&repo_root)
            .args(["ls-files", "--others", "--exclude-standard", "-z"])
            .output()
            && untracked_out.status.success()
        {
            for raw in untracked_out.stdout.split(|&b| b == 0) {
                if raw.is_empty() {
                    continue;
                }
                let rel = String::from_utf8_lossy(raw);
                if rel.starts_with(".carnot/")
                    || rel.starts_with("target/")
                    || rel.starts_with("node_modules/")
                {
                    continue;
                }
                let src = repo_root.join(rel.as_ref());
                let Ok(meta) = std::fs::symlink_metadata(&src) else {
                    continue;
                };
                if !meta.is_file() || meta.len() > MAX_UNTRACKED_COPY_BYTES {
                    continue;
                }
                let dst = worktree_path.join(rel.as_ref());
                if let Some(p) = dst.parent() {
                    let _ = std::fs::create_dir_all(p);
                }
                let _ = std::fs::copy(&src, &dst);
            }
        }

        // Ensure worker_workspace exists even if it was an untracked subdir at fork time
        if !guard.worker_workspace.exists() {
            let _ = std::fs::create_dir_all(&guard.worker_workspace);
        }

        // 3. Stage & commit baseline snapshot inside detached worktree so future `git diff <base_sha>`
        //    captures strictly the worker's own edits (never parent's pre-existing dirty tree).
        guard.base_sha = commit_worktree_checkpoint(&worktree_path, "carnot-fork-base")?;
        Ok(guard)
    }

    /// Exports strictly the worker's own changes (`git diff --binary <base_sha>`, including newly
    /// created untracked files, excluding `.carnot/`).
    pub fn export_worker_diff(&self) -> Result<Option<Vec<u8>>> {
        let _ = carnot_sandbox::host_safe_git_command(&self.worktree_path)
            .args(["add", "-N", "."])
            .output();
        let out = carnot_sandbox::host_safe_git_command(&self.worktree_path)
            .args([
                "diff",
                "--binary",
                &self.base_sha,
                "--",
                ".",
                ":(exclude).carnot",
            ])
            .output()?;
        if !out.status.success() {
            return Err(CarnotError::Session(format!(
                "git diff from worktree baseline failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        if out.stdout.is_empty() {
            Ok(None)
        } else {
            Ok(Some(out.stdout))
        }
    }

    /// Concise `git diff --stat <base_sha>` summarizing only files modified by this worker.
    pub fn changed_files_summary(&self) -> Option<String> {
        let _ = carnot_sandbox::host_safe_git_command(&self.worktree_path)
            .args(["add", "-N", "."])
            .output();
        let out = carnot_sandbox::host_safe_git_command(&self.worktree_path)
            .args([
                "diff",
                "--stat",
                &self.base_sha,
                "--",
                ".",
                ":(exclude).carnot",
            ])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if s.is_empty() { None } else { Some(s) }
    }

    /// Atomically checks and applies the worker's diff onto the parent repository root, then
    /// advances `self.base_sha` inside the worktree so subsequent `land` calls on a `keep_alive`
    /// worker only diff and apply newly made changes.
    pub fn land_into_parent(&mut self) -> Result<LandOutcome> {
        if self.land_checkpoint_failed {
            return Err(CarnotError::Session(
                "previous land applied changes to parent but failed to advance worktree baseline; do not re-land — inspect parent workspace or close this worker".into(),
            ));
        }
        let Some(diff_bytes) = self.export_worker_diff()? else {
            return Ok(LandOutcome::NoChanges);
        };
        let stat = self
            .changed_files_summary()
            .unwrap_or_else(|| format!("{} bytes", diff_bytes.len()));

        // Phase 1: Dry-run conflict check from repo_root (zero mutation on parent)
        let mut check_child = carnot_sandbox::host_safe_git_command(&self.repo_root)
            .args(["apply", "--check", "--binary"])
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        if let Some(mut stdin) = check_child.stdin.take() {
            stdin.write_all(&diff_bytes)?;
        }
        let check_out = check_child.wait_with_output()?;
        if !check_out.status.success() {
            return Ok(LandOutcome::Conflict {
                stderr: String::from_utf8_lossy(&check_out.stderr)
                    .trim()
                    .to_string(),
                diff_bytes,
            });
        }

        // Phase 2: Atomic apply from repo_root
        let mut apply_child = carnot_sandbox::host_safe_git_command(&self.repo_root)
            .args(["apply", "--binary"])
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        if let Some(mut stdin) = apply_child.stdin.take() {
            stdin.write_all(&diff_bytes)?;
        }
        let apply_out = apply_child.wait_with_output()?;
        if !apply_out.status.success() {
            return Ok(LandOutcome::Conflict {
                stderr: String::from_utf8_lossy(&apply_out.stderr)
                    .trim()
                    .to_string(),
                diff_bytes,
            });
        }

        // Phase 3: Advance worktree baseline commit so future diffs/lands are strictly incremental.
        // Parent workspace is already updated by Phase 2, so report a warning rather than Err if
        // baseline checkpointing fails.
        let mut final_stat = stat;
        match commit_worktree_checkpoint(&self.worktree_path, "carnot-land-checkpoint") {
            Ok(new_sha) => self.base_sha = new_sha,
            Err(e) => {
                self.land_checkpoint_failed = true;
                let _ = write!(
                    final_stat,
                    "\n[warning: patch was applied to parent workspace, but advancing worktree baseline failed ({e}); do not re-run `land` on this worker — inspect or close it instead]"
                );
            }
        }

        Ok(LandOutcome::Applied { stat: final_stat })
    }

    /// Fast check whether any orphaned `wt-*` fork worktree directories exist on disk.
    /// Returns `false` in microseconds with zero subprocesses when `~/.carnot/forks` is absent or empty.
    pub fn has_stale_fork_candidates() -> bool {
        let Ok(base) = forks_base_dir() else {
            return false;
        };
        let Ok(entries) = std::fs::read_dir(&base) else {
            return false;
        };
        entries.flatten().any(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|s| s.starts_with("wt-"))
        })
    }

    /// Removes orphaned fork worktrees left behind after an unclean process termination (`SIGKILL`),
    /// skipping any fork whose recorded owner PID is still alive. Returns the count removed.
    pub fn prune_stale_forks(parent_workspace: &Path) -> usize {
        let Ok(base) = forks_base_dir() else {
            return 0;
        };
        let Ok(entries) = std::fs::read_dir(&base) else {
            return 0;
        };
        let stale: Vec<(PathBuf, PathBuf)> = entries
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name();
                let fork_id = name.to_str()?.strip_prefix("wt-")?;
                let wt_path = entry.path();
                let target_path = base.join(format!("target-{fork_id}"));
                let pid_path = target_path.join(".carnot-owner-pid");
                let owner_alive = std::fs::read_to_string(&pid_path)
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .is_some_and(is_process_alive);
                (!owner_alive).then_some((wt_path, target_path))
            })
            .collect();
        if stale.is_empty() {
            return 0;
        }
        let repo_root = carnot_sandbox::host_safe_git_command(parent_workspace)
            .args(["rev-parse", "--show-toplevel"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim()));

        let mut pruned = 0usize;
        for (wt_path, target_path) in stale {
            if let Some(ref rr) = repo_root {
                let _ = carnot_sandbox::host_safe_git_command(rr)
                    .args(["worktree", "remove", "--force", &wt_path.to_string_lossy()])
                    .output();
            }
            let _ = std::fs::remove_dir_all(&wt_path);
            let _ = std::fs::remove_dir_all(&target_path);
            pruned += 1;
        }
        if let Some(ref rr) = repo_root {
            let _ = carnot_sandbox::host_safe_git_command(rr)
                .args(["worktree", "prune"])
                .output();
        }
        pruned
    }

    pub fn cleanup(self) {
        drop(self);
    }
}

impl Drop for IsolatedForkWorktree {
    fn drop(&mut self) {
        let _ = carnot_sandbox::host_safe_git_command(&self.repo_root)
            .args([
                "worktree",
                "remove",
                "--force",
                &self.worktree_path.to_string_lossy(),
            ])
            .output();
        let _ = carnot_sandbox::host_safe_git_command(&self.repo_root)
            .args(["worktree", "prune"])
            .output();
        let _ = std::fs::remove_dir_all(&self.worktree_path);
        let _ = std::fs::remove_dir_all(&self.cargo_target_dir);
    }
}
