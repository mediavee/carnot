//! Safe N3 Warm-Fork Worktree Isolator (opt-in).
//!
//! Fixes the Red-Team APFS `cp -c -R` git-corruption bug: ALWAYS creates an isolated
//! detached `git worktree add --detach` with a dedicated `CARGO_TARGET_DIR` so parallel
//! best-of-k forks never corrupt parent `.git/worktrees/<name>` or contend on Cargo locks.

use carnot_core::{CarnotError, Result};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct IsolatedForkWorktree {
    parent_repo: PathBuf,
    pub worktree_path: PathBuf,
    pub cargo_target_dir: PathBuf,
}

impl IsolatedForkWorktree {
    pub fn create(parent_repo: &Path, fork_id: &str) -> Result<Self> {
        let base = std::env::temp_dir().join("carnot-forks");
        std::fs::create_dir_all(&base)?;
        let worktree_path = base.join(format!("wt-{fork_id}"));
        let cargo_target_dir = base.join(format!("target-{fork_id}"));
        std::fs::create_dir_all(&cargo_target_dir)?;

        let out = carnot_sandbox::host_safe_git_command(parent_repo)
            .args([
                "worktree",
                "add",
                "--detach",
                &worktree_path.to_string_lossy(),
                "HEAD",
            ])
            .output()?;
        if !out.status.success() {
            return Err(CarnotError::Session(format!(
                "git worktree add failed: {}",
                String::from_utf8_lossy(&out.stderr)
            )));
        }

        // Apply any uncommitted diff from parent into the isolated fork cleanly
        if let Ok(diff_out) = carnot_sandbox::host_safe_git_command(parent_repo)
            .args(["diff", "HEAD"])
            .output()
            && diff_out.status.success()
            && !diff_out.stdout.is_empty()
            && let Ok(mut child) = carnot_sandbox::host_safe_git_command(&worktree_path)
                .args(["apply", "--allow-empty"])
                .stdin(std::process::Stdio::piped())
                .spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                use std::io::Write;
                let _ = stdin.write_all(&diff_out.stdout);
            }
            let _ = child.wait();
        }

        Ok(Self {
            parent_repo: parent_repo.to_path_buf(),
            worktree_path,
            cargo_target_dir,
        })
    }

    pub fn cleanup(self) {
        // Drop impl performs deterministic cleanup even on cancellation/panic (fixes Opus #8!)
        drop(self);
    }
}

impl Drop for IsolatedForkWorktree {
    fn drop(&mut self) {
        let _ = carnot_sandbox::host_safe_git_command(&self.parent_repo)
            .args([
                "worktree",
                "remove",
                "--force",
                &self.worktree_path.to_string_lossy(),
            ])
            .output();
        let _ = std::fs::remove_dir_all(&self.cargo_target_dir);
    }
}
