//! Safe, zero-`unsafe` Linux Landlock fallback backend for `carnot-sandbox`.
//!
//! When `bubblewrap` (`bwrap`) is unavailable or blocked by container seccomp (`CLONE_NEWUSER`
//! denied inside rootless Docker), `carnot-sandbox` falls back automatically to Linux Landlock
//! via a self-exec trampoline (`<carnot-exe> __carnot-landlock-exec <spec-json>`).
//!
//! Security properties:
//!   - Default-deny filesystem: `$HOME` is NOT granted read access, so `~/.ssh`, `~/.aws`,
//!     `~/.gnupg`, `~/.config/gh`, `~/.secrets`, `~/.claude`, `~/.codex`, etc. stay unreadable.
//!   - System binaries/libraries (`/usr`, `/bin`, `/sbin`, `/lib`, `/lib64`, `/etc`, `/opt`,
//!     `/proc`, `/dev`, `/tmp`) and configured language toolchains are READ-ONLY.
//!   - Writes are restricted to `workspace_root`, `extra_writable_roots`, verified linked git
//!     worktree dirs, and `/dev/null` + `/dev/tty`.
//!   - On Landlock ABI v4+, when `allow_network == false`, TCP bind and connect are blocked
//!     in-kernel (`AccessNet::from_all`).
//!   - On Landlock ABI v6+, abstract UNIX sockets and cross-domain signals are scoped.
//!   - Fail-closed: exits with code `126` if Landlock cannot be enforced.

use crate::SandboxPolicy;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tokio::process::Command;

/// Hidden argv[1] sentinel recognized by `try_handle_landlock_exec_arg()` before Tokio starts.
pub const LANDLOCK_EXEC_ARG: &str = "__carnot-landlock-exec";

static LANDLOCK_HELPER: OnceLock<PathBuf> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LandlockSpec {
    pub workspace_root: PathBuf,
    pub extra_writable_roots: Vec<PathBuf>,
    pub frozen_git_write_dirs: Vec<PathBuf>,
    pub allow_network: bool,
    pub cwd: PathBuf,
    pub shell_cmd: String,
}

impl LandlockSpec {
    pub fn from_policy(policy: &SandboxPolicy, cwd: &Path, shell_cmd: &str) -> Self {
        Self {
            workspace_root: policy.workspace_root.clone(),
            extra_writable_roots: policy.extra_writable_roots.clone(),
            frozen_git_write_dirs: policy.frozen_git_write_dirs.clone(),
            allow_network: policy.allow_network,
            cwd: cwd.to_path_buf(),
            shell_cmd: shell_cmd.to_string(),
        }
    }
}

/// Registers the executable used as the Landlock self-exec trampoline (typically `current_exe()`
/// registered by the `carnot` CLI binary at startup).
pub fn register_landlock_helper(exe: PathBuf) {
    let _ = LANDLOCK_HELPER.set(exe);
}

/// Returns the registered Landlock helper executable, if any.
pub fn landlock_helper() -> Option<&'static Path> {
    LANDLOCK_HELPER.get().map(PathBuf::as_path)
}

/// Must be called as the FIRST thing in the CLI binary's `main()`, before starting Tokio or
/// spawning any threads:
///   - If invoked with `argv[1] == LANDLOCK_EXEC_ARG`, applies Landlock and `exec`s the shell
///     (never returns).
///   - Otherwise registers `std::env::current_exe()` as the Landlock helper for this process.
pub fn try_handle_landlock_exec_arg() {
    let mut args = std::env::args_os();
    let _exe = args.next();
    if args.next().and_then(|a| a.into_string().ok()).as_deref() == Some(LANDLOCK_EXEC_ARG) {
        let Some(spec_json) = args.next().and_then(|a| a.into_string().ok()) else {
            eprintln!("carnot-sandbox: missing Landlock spec argument");
            std::process::exit(126);
        };
        run_landlock_exec(&spec_json);
    }
    if let Ok(exe) = std::env::current_exe() {
        register_landlock_helper(exe);
    }
}

/// Builds a `Command` that invokes the registered Landlock trampoline with the serialized policy.
pub(crate) fn build_linux_landlock_command(
    helper: &Path,
    policy: &SandboxPolicy,
    shell_cmd: &str,
    cwd: &Path,
) -> std::result::Result<Command, String> {
    let spec = LandlockSpec::from_policy(policy, cwd, shell_cmd);
    let spec_json = serde_json::to_string(&spec)
        .map_err(|e| format!("failed to serialize Landlock spec: {e}"))?;
    let mut c = Command::new(helper);
    c.arg(LANDLOCK_EXEC_ARG).arg(spec_json);
    Ok(c)
}

#[cfg(target_os = "linux")]
pub fn detect_landlock_abi() -> Option<u32> {
    static CACHED: OnceLock<Option<u32>> = OnceLock::new();
    *CACHED.get_or_init(probe_landlock_abi_uncached)
}

#[cfg(not(target_os = "linux"))]
pub fn detect_landlock_abi() -> Option<u32> {
    None
}

#[cfg(target_os = "linux")]
fn probe_landlock_abi_uncached() -> Option<u32> {
    use landlock::{
        ABI, Access, AccessFs, AccessNet, CompatLevel, Compatible, Ruleset, RulesetAttr, Scope,
    };

    for (abi, num) in [
        (ABI::V6, 6u32),
        (ABI::V5, 5),
        (ABI::V4, 4),
        (ABI::V3, 3),
        (ABI::V2, 2),
        (ABI::V1, 1),
    ] {
        let mut r = Ruleset::default()
            .set_compatibility(CompatLevel::HardRequirement)
            .handle_access(AccessFs::from_all(abi));
        if abi >= ABI::V4 {
            r = r.and_then(|rs| rs.handle_access(AccessNet::from_all(abi)));
        }
        if abi >= ABI::V6 {
            r = r.and_then(|rs| rs.scope(Scope::from_all(abi)));
        }
        if r.and_then(|rs| rs.create()).is_ok() {
            return Some(num);
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn run_landlock_exec(spec_json: &str) -> ! {
    use std::os::unix::process::CommandExt;

    let spec: LandlockSpec = match serde_json::from_str(spec_json) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("carnot-sandbox: invalid Landlock spec: {e}");
            std::process::exit(126);
        }
    };

    if let Err(e) = enforce_landlock(&spec) {
        eprintln!("carnot-sandbox: Landlock enforcement failed: {e}");
        std::process::exit(126);
    }

    if let Err(e) = std::env::set_current_dir(&spec.cwd) {
        eprintln!(
            "carnot-sandbox: failed to chdir to {}: {e}",
            spec.cwd.display()
        );
        std::process::exit(126);
    }

    let shell = crate::resolve_shell_binary();
    let err = std::process::Command::new(shell)
        .arg("-c")
        .arg(&spec.shell_cmd)
        .exec();
    eprintln!("carnot-sandbox: failed to exec {shell}: {err}");
    std::process::exit(126);
}

#[cfg(not(target_os = "linux"))]
fn run_landlock_exec(_spec_json: &str) -> ! {
    eprintln!("carnot-sandbox: Landlock is only supported on Linux");
    std::process::exit(126);
}

#[cfg(target_os = "linux")]
fn enforce_landlock(spec: &LandlockSpec) -> std::result::Result<(), String> {
    use landlock::{
        ABI, Access, AccessFs, AccessNet, CompatLevel, Compatible, Ruleset, RulesetAttr,
        RulesetCreatedAttr, RulesetStatus, Scope, path_beneath_rules,
    };

    let target_abi = ABI::V6;
    let mut ruleset = Ruleset::default()
        .set_compatibility(CompatLevel::BestEffort)
        .handle_access(AccessFs::from_all(target_abi))
        .map_err(|e| format!("handle_access(fs): {e}"))?;

    if !spec.allow_network {
        ruleset = ruleset
            .handle_access(AccessNet::from_all(target_abi))
            .map_err(|e| format!("handle_access(net): {e}"))?;
    }

    ruleset = ruleset
        .scope(Scope::from_all(target_abi))
        .map_err(|e| format!("scope: {e}"))?;

    let mut ro_paths: Vec<PathBuf> = [
        "/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt", "/proc", "/dev", "/tmp",
    ]
    .into_iter()
    .map(PathBuf::from)
    .collect();

    for env_k in ["CARGO_HOME", "RUSTUP_HOME", "GOPATH", "VIRTUAL_ENV"] {
        if let Some(v) = std::env::var_os(env_k)
            && !v.is_empty()
        {
            ro_paths.push(PathBuf::from(v));
        }
    }

    let mut rw_paths: Vec<PathBuf> =
        Vec::with_capacity(1 + spec.extra_writable_roots.len() + spec.frozen_git_write_dirs.len());
    rw_paths.push(spec.workspace_root.clone());
    rw_paths.extend(spec.extra_writable_roots.iter().cloned());
    rw_paths.extend(spec.frozen_git_write_dirs.iter().cloned());

    let dev_rw_files = [PathBuf::from("/dev/null"), PathBuf::from("/dev/tty")];
    let dev_file_access = AccessFs::from_file(target_abi) & !AccessFs::Execute;

    let created = ruleset
        .create()
        .map_err(|e| format!("create ruleset: {e}"))?
        .add_rules(path_beneath_rules(
            &ro_paths,
            AccessFs::from_read(target_abi),
        ))
        .map_err(|e| format!("add read rules: {e}"))?
        .add_rules(path_beneath_rules(
            &rw_paths,
            AccessFs::from_all(target_abi),
        ))
        .map_err(|e| format!("add write rules: {e}"))?
        .add_rules(path_beneath_rules(&dev_rw_files, dev_file_access))
        .map_err(|e| format!("add /dev rules: {e}"))?;

    let status = created
        .restrict_self()
        .map_err(|e| format!("restrict_self: {e}"))?;

    if status.ruleset == RulesetStatus::NotEnforced {
        return Err("kernel did not enforce Landlock ruleset".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn landlock_spec_roundtrips_json() {
        let policy = SandboxPolicy::for_workspace(PathBuf::from("/tmp/ws"), true, false)
            .with_extra_writable_roots(vec![PathBuf::from("/tmp/extra")]);
        let spec = LandlockSpec::from_policy(&policy, Path::new("/tmp/ws"), "echo hi");
        let raw = serde_json::to_string(&spec).expect("serialize");
        let parsed: LandlockSpec = serde_json::from_str(&raw).expect("deserialize");
        assert_eq!(parsed, spec);
        assert!(!parsed.allow_network);
    }
}
