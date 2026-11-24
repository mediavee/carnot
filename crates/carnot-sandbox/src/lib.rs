//! Hardened OS-native process sandboxing (macOS Seatbelt, Linux bwrap + Landlock fallback) and
//! strict environment allowlisting.
//!
//! Security invariants (verified against Opus 5.5 High red-team exploits C1, S1, S3, N6, N7):
//!   - `<workspace>/.git` is READ-ONLY inside Seatbelt & bwrap (closes C1: sandboxed process can
//!     NEVER write `.git/config` or `.git/hooks/*` to hijack host-side `git`!).
//!   - Linked git worktrees are verified strictly (`gd` outside workspace, `cd == gd/../..`,
//!     reciprocal `worktrees/<name>/gitdir` -> `<ws>/.git` — closes N6) and ONLY `worktrees/<name>`
//!     + `cd/{objects,refs,logs}` are writable — NEVER `cd/config` or `cd/hooks`!
//!   - Host-side git invocations use `host_safe_git_command()` (`-c core.fsmonitor=false -c core.hooksPath=/dev/null`).
//!   - Strict environment allowlist (`env_clear()` + safe baseline vars + isolated `ZDOTDIR`,
//!     no `BASH_ENV`/`ENV`) + Seatbelt/bwrap/Landlock block `~/.ssh`, `~/.aws`, `~/.gnupg`,
//!     `~/.config/gh`, `~/.secrets`, `~/.zshenv`, `~/.zshrc`, `~/.claude`, `~/.codex`,
//!     `~/.config/agentsafe`, `~/.docker`, `~/.kube` (closes S3 & N7).
//!   - Workspace-isolated `TMPDIR` (`.carnot/tmp`), network OFF by default, fail-closed if no OS sandbox.

mod landlock;

pub use landlock::{
    LANDLOCK_EXEC_ARG, detect_landlock_abi, landlock_helper, register_landlock_helper,
    try_handle_landlock_exec_arg,
};

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tokio::process::Command;

/// Safe environment variables preserved across `env_clear()`.
const ALLOWED_ENV_VARS: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "LANG",
    "COLORTERM",
    "NO_COLOR",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "GOPATH",
    "VIRTUAL_ENV",
];

#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    pub enabled: bool,
    pub workspace_root: PathBuf,
    pub extra_writable_roots: Vec<PathBuf>,
    /// Frozen at policy creation time via strict host-side verification — never `config` or `hooks`!
    frozen_git_write_dirs: Vec<PathBuf>,
    pub allow_network: bool,
    /// Sensitive env vars (e.g. Carnot's own LLM provider keys) scrubbed even when `enabled == false`.
    pub scrubbed_env_vars: Vec<String>,
}

impl SandboxPolicy {
    pub fn for_workspace(workspace_root: PathBuf, enabled: bool, allow_network: bool) -> Self {
        let canonical_ws = workspace_root
            .canonicalize()
            .unwrap_or_else(|_| workspace_root.clone());
        let frozen_git_write_dirs = resolve_verified_linked_git_dirs(&canonical_ws);
        Self {
            enabled,
            workspace_root: canonical_ws,
            extra_writable_roots: Vec::new(),
            frozen_git_write_dirs,
            allow_network,
            scrubbed_env_vars: Vec::new(),
        }
    }

    pub fn with_extra_writable_roots(mut self, roots: Vec<PathBuf>) -> Self {
        self.extra_writable_roots = roots
            .into_iter()
            .map(|r| r.canonicalize().unwrap_or(r))
            .collect();
        self
    }

    pub fn with_scrubbed_env_vars(mut self, vars: Vec<String>) -> Self {
        self.scrubbed_env_vars = vars;
        self
    }
}

/// Prefers `/bin/bash` when available so `Bash` tool invocations support standard Bash syntax
/// (`set -o pipefail`, `[[ ... ]]`, `source`, arrays, `<(...)`), falling back to `/bin/sh` on minimal images.
fn resolve_shell_binary() -> &'static str {
    if Path::new("/bin/bash").is_file() {
        "/bin/bash"
    } else {
        "/bin/sh"
    }
}

/// Builds a host-side `std::process::Command` for `git` that neutralizes repo-controlled
/// `core.fsmonitor`, `core.hooksPath`, and `core.sshCommand` (defense-in-depth for C1).
pub fn host_safe_git_command(repo_root: &Path) -> std::process::Command {
    let mut cmd = std::process::Command::new("git");
    cmd.args([
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.sshCommand=false",
        "-C",
        &repo_root.to_string_lossy(),
    ]);
    cmd
}

/// Applies strict environment allowlist (`env_clear()`) + safe `LC_*` + `extra_env` to any child process
/// (used by `Bash`, `McpStdioClient`, and `SubprocessExtension`).
pub fn apply_safe_env(cmd: &mut Command, extra_env: &BTreeMap<String, String>) {
    let mut kept: BTreeMap<String, String> = BTreeMap::new();
    for (k, v) in std::env::vars() {
        if ALLOWED_ENV_VARS.contains(&k.as_str()) || k.starts_with("LC_") {
            kept.insert(k, v);
        }
    }
    cmd.env_clear();
    for (k, v) in kept {
        cmd.env(k, v);
    }
    cmd.env("AI_AGENT", "carnot");
    // Neutralize shell startup scripts that could reload host secrets (N7)
    cmd.env("BASH_ENV", "");
    cmd.env("ENV", "");
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
}

/// Preserves the inherited runtime/container environment for `--no-sandbox` shells (keeping
/// `JAVA_HOME`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `CONDA_PREFIX`, service endpoints/tokens, etc.),
/// while still scrubbing Carnot's own LLM credential variables and shell startup hooks.
pub fn apply_unsandboxed_env(
    cmd: &mut Command,
    scrubbed_vars: &[String],
    extra_env: &BTreeMap<String, String>,
) {
    for var in scrubbed_vars {
        cmd.env_remove(var);
    }
    cmd.env("AI_AGENT", "carnot");
    cmd.env("BASH_ENV", "");
    cmd.env("ENV", "");
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
}

/// Builds a `tokio::process::Command` executing `shell_cmd` under the hardened OS sandbox.
/// Fails closed (returns `Err`) if `policy.enabled` is true and no OS sandbox is available.
pub fn build_sandboxed_shell(
    policy: &SandboxPolicy,
    shell_cmd: &str,
    extra_env: &BTreeMap<String, String>,
) -> std::result::Result<Command, String> {
    build_sandboxed_shell_in_dir(policy, shell_cmd, &policy.workspace_root, extra_env)
}

/// Summary of the active OS sandbox backend on this host (for diagnostics such as `carnot doctor`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxBackendInfo {
    pub backend: &'static str,
    pub available: bool,
    pub detail: Option<String>,
}

/// Detects the OS sandbox backend that `build_sandboxed_shell` will use on this host.
pub fn detect_sandbox_backend() -> SandboxBackendInfo {
    if cfg!(target_os = "macos") {
        let ok = Path::new("/usr/bin/sandbox-exec").is_file();
        return SandboxBackendInfo {
            backend: "seatbelt",
            available: ok,
            detail: None,
        };
    }
    if cfg!(target_os = "linux") {
        if working_bwrap_binary().is_some() {
            return SandboxBackendInfo {
                backend: "bwrap",
                available: true,
                detail: None,
            };
        }
        if let Some(abi) = detect_landlock_abi() {
            let helper_ready = landlock_helper().is_some();
            return SandboxBackendInfo {
                backend: "landlock",
                available: helper_ready,
                detail: Some(format!("ABI v{abi}")),
            };
        }
        return SandboxBackendInfo {
            backend: "none",
            available: false,
            detail: None,
        };
    }
    SandboxBackendInfo {
        backend: "none",
        available: false,
        detail: None,
    }
}

/// Same as `build_sandboxed_shell`, starting the shell in a verified working directory `cwd`.
pub fn build_sandboxed_shell_in_dir(
    policy: &SandboxPolicy,
    shell_cmd: &str,
    cwd: &Path,
    extra_env: &BTreeMap<String, String>,
) -> std::result::Result<Command, String> {
    let mut cmd = if policy.enabled {
        if cfg!(target_os = "macos") && Path::new("/usr/bin/sandbox-exec").exists() {
            build_macos_seatbelt_command(policy, shell_cmd)
        } else if cfg!(target_os = "linux")
            && let Some(bwrap_bin) = working_bwrap_binary()
        {
            build_linux_bwrap_command(bwrap_bin, policy, shell_cmd, cwd)
        } else if cfg!(target_os = "linux")
            && detect_landlock_abi().is_some()
            && let Some(helper) = landlock_helper()
        {
            landlock::build_linux_landlock_command(helper, policy, shell_cmd, cwd)?
        } else {
            return Err(
                "OS sandbox requested (`sandbox: true`) but no supported kernel sandbox (macOS sandbox-exec / Linux bwrap or Landlock) is available on this host (pass `--no-sandbox` to explicitly disable)"
                    .into(),
            );
        }
    } else {
        let mut c = Command::new(resolve_shell_binary());
        c.arg("-c").arg(shell_cmd);
        c
    };

    cmd.current_dir(cwd);
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);

    if policy.enabled {
        apply_safe_env(&mut cmd, extra_env);
        // Isolate child TMPDIR & ZDOTDIR inside the workspace (closes host /tmp writes & ~/.zshenv leak N7)
        let ws_tmp = policy.workspace_root.join(".carnot").join("tmp");
        let zdot = ws_tmp.join("zdotdir");
        let _ = std::fs::create_dir_all(&zdot);
        cmd.env("TMPDIR", &ws_tmp);
        cmd.env("ZDOTDIR", &zdot);
        // Inside an offline OS sandbox, host compiler wrapper daemons configured in `~/.cargo/config.toml`
        // (e.g. `sccache` / `cachette-sccache`) cannot reach their local sockets/network and fail `cargo test`.
        // Override them cleanly to empty unless the caller explicitly passed them in `extra_env`.
        if !policy.allow_network {
            for k in [
                "RUSTC_WRAPPER",
                "CARGO_BUILD_RUSTC_WRAPPER",
                "RUSTC_WORKSPACE_WRAPPER",
                "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
            ] {
                if !extra_env.contains_key(k) {
                    cmd.env(k, "");
                }
            }
        }
    } else {
        apply_unsandboxed_env(&mut cmd, &policy.scrubbed_env_vars, extra_env);
    }
    Ok(cmd)
}

const MACOS_DENY_SUBPATHS: &[(&str, &str)] = &[
    ("SSH", ".ssh"),
    ("AWS", ".aws"),
    ("GNUPG", ".gnupg"),
    ("GH", ".config/gh"),
    ("SECRETS", ".secrets"),
    ("CLAUDE", ".claude"),
    ("CODEX", ".codex"),
    ("AGENTSAFE", ".config/agentsafe"),
    ("DOCKER", ".docker"),
    ("KUBE", ".kube"),
    ("OP", ".config/op"),
    ("KEYCHAINS", "Library/Keychains"),
    ("GCLOUD", ".config/gcloud"),
];

const MACOS_DENY_LITERALS: &[(&str, &str)] = &[
    ("ZSHENV", ".zshenv"),
    ("ZSHRC", ".zshrc"),
    ("ZSH_HIST", ".zsh_history"),
    ("BASH_HIST", ".bash_history"),
    ("PY_HIST", ".python_history"),
    ("PSQL_HIST", ".psql_history"),
    ("HOME_ENV", ".env"),
    ("PYPIRC", ".pypirc"),
    ("NETRC", ".netrc"),
    ("GIT_CREDS", ".git-credentials"),
    ("CONFIG_GIT_CREDS", ".config/git/credentials"),
    ("NPMRC", ".npmrc"),
    ("CARGO_CREDS", ".cargo/credentials"),
    ("CARGO_CREDS_TOML", ".cargo/credentials.toml"),
    ("CARNOT_TOKEN", ".carnot/server.token"),
];

/// Resolves all candidate home directories to protect in macOS Seatbelt so overriding `$HOME`
/// (e.g. for isolated app config) NEVER disables secret-read denies on the real user home.
fn macos_protected_homes() -> Vec<PathBuf> {
    let mut homes: Vec<PathBuf> = Vec::new();
    let mut push_home = |candidate: PathBuf| {
        let canonical = candidate.canonicalize().unwrap_or(candidate);
        if !homes.contains(&canonical) {
            homes.push(canonical);
        }
    };
    if let Some(h) = std::env::var_os("HOME").filter(|s| !s.is_empty()) {
        push_home(PathBuf::from(h));
    }
    for var in ["USER", "LOGNAME"] {
        if let Ok(user) = std::env::var(var)
            && !user.is_empty()
            && !user.contains('/')
            && !user.contains("..")
        {
            let real_home = PathBuf::from("/Users").join(&user);
            if real_home.is_dir() {
                push_home(real_home);
            }
        }
    }
    if homes.is_empty() {
        homes.push(PathBuf::from("/nonexistent"));
    }
    homes
}

fn build_macos_seatbelt_command(policy: &SandboxPolicy, shell_cmd: &str) -> Command {
    let mut profile = String::from(
        "(version 1)\n\
         (deny default)\n\
         (allow process*)\n\
         (allow sysctl-read)\n\
         (allow signal (target same-sandbox))\n\
         (allow mach-lookup)\n\
         (deny mach-lookup\n\
           (global-name-prefix \"com.apple.lsd\")\n\
           (global-name-prefix \"com.apple.coreservices.launchservicesd\")\n\
           (global-name \"com.apple.SecurityServer\")\n\
           (global-name \"com.apple.securityd\"))\n\
         (allow file-read*)\n",
    );

    let mut c = Command::new("/usr/bin/sandbox-exec");
    let homes = macos_protected_homes();

    profile.push_str("(deny file-read*\n");
    for (h_idx, home) in homes.iter().enumerate() {
        for (tag, rel) in MACOS_DENY_SUBPATHS {
            let key = format!("DENY_{tag}_{h_idx}");
            profile.push_str(&format!("  (subpath (param \"{key}\"))\n"));
            c.arg("-D")
                .arg(format!("{key}={}", home.join(rel).display()));
        }
        for (tag, rel) in MACOS_DENY_LITERALS {
            let key = format!("DENY_{tag}_{h_idx}");
            profile.push_str(&format!("  (literal (param \"{key}\"))\n"));
            c.arg("-D")
                .arg(format!("{key}={}", home.join(rel).display()));
        }
    }
    profile.push_str(")\n");
    profile.push_str(
        "(allow file-write* (literal \"/dev/null\") (literal \"/dev/tty\"))\n\
         (allow file-write* (require-all (subpath (param \"WORKSPACE\")) (require-not (subpath (param \"WS_DOT_GIT\")))))\n",
    );
    if policy.allow_network {
        profile.push_str("(allow network*)\n");
    }

    c.arg("-D")
        .arg(format!("WORKSPACE={}", policy.workspace_root.display()));
    c.arg("-D").arg(format!(
        "WS_DOT_GIT={}",
        policy.workspace_root.join(".git").display()
    ));

    for (idx, gdir) in policy.frozen_git_write_dirs.iter().enumerate() {
        let key = format!("GITW_{idx}");
        profile.push_str(&format!(
            "(allow file-write* (subpath (param \"{key}\")))\n"
        ));
        c.arg("-D").arg(format!("{key}={}", gdir.display()));
    }
    for (idx, root) in policy.extra_writable_roots.iter().enumerate() {
        let key = format!("EXTRA_{idx}");
        profile.push_str(&format!(
            "(allow file-write* (subpath (param \"{key}\")))\n"
        ));
        c.arg("-D").arg(format!("{key}={}", root.display()));
    }

    c.arg("-p").arg(profile);
    c.arg(resolve_shell_binary()).arg("-c").arg(shell_cmd);
    c
}

/// Hardened Linux `bubblewrap` (`bwrap`) backend (fail-closed, zero `unsafe`! — fixes Opus #1).
///   - `--unshare-all` (isolates PID, IPC, UTS, cgroup, user, net namespaces -> parent `/proc/<pid>/environ`
///     and host Unix sockets are unreachable!) + `--share-net` ONLY when `policy.allow_network == true`.
///   - Explicit minimal read-only mounts (`/usr`, `/bin`, `/sbin`, `/lib`, `/lib64`, `/etc`, `/opt`)
///     instead of `--ro-bind / /` (prevents connecting to host D-Bus / `docker.sock` / SSH agent sockets!).
///   - Fresh `--tmpfs /run`, `--tmpfs /tmp`, and `--tmpfs $XDG_RUNTIME_DIR`.
///   - Read-only `<ws>/.git` (C1), `--new-session` (blocks TIOCSTI), `--die-with-parent`, and
///     credential file/dir masking (`~/.netrc`, `~/.git-credentials`, `~/.npmrc`, `~/.cargo/credentials.toml`).
fn resolve_bwrap_binary() -> Option<PathBuf> {
    for candidate in ["/usr/bin/bwrap", "/bin/bwrap"] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join("bwrap"))
            .find(|p| p.is_file())
    })
}

/// Probes once whether `bwrap --unshare-all` actually succeeds on this host (false inside standard
/// Docker containers where seccomp blocks `CLONE_NEWUSER`, triggering automatic Landlock fallback).
fn working_bwrap_binary() -> Option<&'static Path> {
    static CACHED: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            let bin = resolve_bwrap_binary()?;
            let ok = std::process::Command::new(&bin)
                .args([
                    "--unshare-all",
                    "--ro-bind",
                    "/",
                    "/",
                    resolve_shell_binary(),
                    "-c",
                    "true",
                ])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            ok.then_some(bin)
        })
        .as_deref()
}

fn build_linux_bwrap_command(
    bwrap_bin: &Path,
    policy: &SandboxPolicy,
    shell_cmd: &str,
    cwd: &Path,
) -> Command {
    let mut c = Command::new(bwrap_bin);
    c.args(["--unshare-all", "--new-session", "--die-with-parent"]);
    if policy.allow_network {
        c.arg("--share-net");
    }
    c.args(["--dev", "/dev", "--proc", "/proc"]);
    c.args(["--tmpfs", "/tmp", "--tmpfs", "/run"]);
    if let Some(xdg) = std::env::var_os("XDG_RUNTIME_DIR")
        && !xdg.is_empty()
    {
        c.args(["--tmpfs", &xdg.to_string_lossy()]);
    }

    // Minimal read-only OS system dirs (NEVER `--ro-bind / /`!)
    for sys_dir in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt"] {
        if Path::new(sys_dir).exists() {
            c.args(["--ro-bind", sys_dir, sys_dir]);
        }
    }

    // Read-only language toolchain directories so `cargo`, `rustup`, `go`, `venv` work inside bwrap
    // while keeping credentials strictly masked below.
    let home_opt = std::env::var_os("HOME").map(PathBuf::from);
    let mut ro_toolchains: Vec<PathBuf> = Vec::new();
    for env_k in ["CARGO_HOME", "RUSTUP_HOME", "GOPATH", "VIRTUAL_ENV"] {
        if let Some(v) = std::env::var_os(env_k)
            && !v.is_empty()
        {
            ro_toolchains.push(PathBuf::from(v));
        }
    }
    if let Some(ref home) = home_opt {
        for rel in [".cargo", ".rustup"] {
            ro_toolchains.push(home.join(rel));
        }
    }
    let mut seen_ro = std::collections::HashSet::new();
    for dir in ro_toolchains {
        if dir.is_dir() && seen_ro.insert(dir.clone()) {
            let s = dir.to_string_lossy().to_string();
            c.args(["--ro-bind", &s, &s]);
        }
    }

    let ws = policy.workspace_root.to_string_lossy().to_string();
    c.args(["--bind", &ws, &ws]);
    let dot_git = policy.workspace_root.join(".git");
    if dot_git.exists() {
        let dg = dot_git.to_string_lossy().to_string();
        c.args(["--ro-bind", &dg, &dg]);
    }
    for gdir in &policy.frozen_git_write_dirs {
        let g = gdir.to_string_lossy().to_string();
        c.args(["--bind", &g, &g]);
    }
    for root in &policy.extra_writable_roots {
        let r = root.to_string_lossy().to_string();
        c.args(["--bind", &r, &r]);
    }

    // Mask sensitive credential dirs & files under `$HOME` if `$HOME` or toolchains are exposed
    if let Some(home) = home_opt {
        for rel in [
            ".ssh",
            ".aws",
            ".gnupg",
            ".config/gh",
            ".config/gcloud",
            ".secrets",
            ".claude",
            ".codex",
            ".config/agentsafe",
            ".docker",
            ".kube",
            ".config/op",
        ] {
            let p = home.join(rel);
            if p.is_dir() {
                c.args(["--tmpfs", &p.to_string_lossy()]);
            }
        }
        for rel_file in [
            ".netrc",
            ".git-credentials",
            ".config/git/credentials",
            ".npmrc",
            ".pypirc",
            ".env",
            ".cargo/credentials.toml",
            ".cargo/credentials",
            ".carnot/server.token",
            ".zshenv",
            ".zshrc",
            ".zsh_history",
            ".bash_history",
            ".python_history",
            ".psql_history",
        ] {
            let p = home.join(rel_file);
            if p.is_file() {
                c.args(["--ro-bind", "/dev/null", &p.to_string_lossy()]);
            }
        }
    }

    let chdir = cwd.to_string_lossy().to_string();
    c.args(["--chdir", &chdir, resolve_shell_binary(), "-c", shell_cmd]);
    c
}

/// Strictly validates a linked git worktree ONCE on the host (closes S1 & N6):
///   - `gd` MUST lie OUTSIDE `canonical_ws`
///   - `gd` MUST have layout `<cd>/worktrees/<name>` (so `cd == gd/../..`)
///   - `<gd>/gitdir` MUST reciprocally resolve to `<canonical_ws>/.git`
///   - Only safe object/ref/worktree subdirs are returned — NEVER `cd/config` or `cd/hooks` (C1)!
fn resolve_verified_linked_git_dirs(canonical_ws: &Path) -> Vec<PathBuf> {
    let dot_git = canonical_ws.join(".git");
    if !dot_git.is_file() {
        return Vec::new();
    }
    let Ok(out) = host_safe_git_command(canonical_ws)
        .args(["rev-parse", "--git-dir", "--git-common-dir"])
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    let txt = String::from_utf8_lossy(&out.stdout);
    let mut lines = txt.lines();
    let (Some(gd_raw), Some(cd_raw)) = (lines.next(), lines.next()) else {
        return Vec::new();
    };
    let resolve = |raw: &str| {
        let p = PathBuf::from(raw.trim());
        let abs = if p.is_absolute() {
            p
        } else {
            canonical_ws.join(p)
        };
        abs.canonicalize().ok()
    };
    let (Some(gd), Some(cd)) = (resolve(gd_raw), resolve(cd_raw)) else {
        return Vec::new();
    };

    // N6 guard: `gd` MUST NOT be inside workspace, and `cd` MUST be `gd/../..` with `worktrees` parent
    if gd.starts_with(canonical_ws) || cd.starts_with(canonical_ws) {
        return Vec::new();
    }
    let Some(wt_parent) = gd.parent() else {
        return Vec::new();
    };
    if wt_parent.file_name().and_then(|s| s.to_str()) != Some("worktrees") {
        return Vec::new();
    }
    let Some(expected_cd) = wt_parent.parent() else {
        return Vec::new();
    };
    if expected_cd != cd {
        return Vec::new();
    }
    let Ok(back_ptr) = std::fs::read_to_string(gd.join("gitdir")) else {
        return Vec::new();
    };
    let back_path = PathBuf::from(back_ptr.trim())
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(back_ptr.trim()));
    let expected_dot_git = dot_git.canonicalize().unwrap_or(dot_git);
    if back_path != expected_dot_git {
        return Vec::new();
    }

    // C1 guard: open ONLY `gd`, `cd/objects`, `cd/refs`, `cd/logs` — NEVER `cd/config` or `cd/hooks`!
    let mut dirs = vec![gd];
    for sub in ["objects", "refs", "logs"] {
        let p = cd.join(sub);
        if p.exists() {
            dirs.push(p);
        }
    }
    dirs
}

/// Detects likely OS sandbox denial in stderr/stdout and returns a concise guidance note.
pub fn annotate_sandbox_denial(output: &str, policy: &SandboxPolicy) -> Option<String> {
    if !policy.enabled {
        return None;
    }
    let denied = output.contains("Operation not permitted")
        || output.contains("Read-only file system")
        || (!policy.allow_network
            && (output.contains("Could not resolve host")
                || output.contains("Network is unreachable")));
    if denied {
        Some(format!(
            "\n[carnot-sandbox] Writes restricted to {} (excluding .git; network: {}).",
            policy.workspace_root.display(),
            if policy.allow_network { "on" } else { "off" }
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn blocks_c1_dot_git_config_tampering_under_seatbelt() {
        let base = std::env::temp_dir().join(format!("carnot-c1-{}", uuid::Uuid::new_v4()));
        let ws = base.join("ws");
        std::fs::create_dir_all(&ws).expect("ws");
        let init = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&ws)
            .status()
            .expect("git init");
        assert!(init.success());

        let policy = SandboxPolicy::for_workspace(ws.clone(), true, false);
        let out = build_sandboxed_shell(
            &policy,
            "git config core.fsmonitor /tmp/pwn.sh",
            &BTreeMap::new(),
        )
        .expect("cmd")
        .output()
        .await
        .expect("run");
        assert!(
            !out.status.success(),
            "C1 regression: sandboxed process MUST NOT write .git/config!"
        );
        let cfg = std::fs::read_to_string(ws.join(".git/config")).expect("read config");
        assert!(!cfg.contains("fsmonitor"));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn blocks_n6_forged_commondir_in_fake_worktree() {
        let base = std::env::temp_dir().join(format!("carnot-n6-{}", uuid::Uuid::new_v4()));
        let ws = base.join("ws");
        let victim = base.join("victim_repo");
        std::fs::create_dir_all(&ws).expect("ws");
        std::fs::create_dir_all(&victim).expect("victim");
        let _ = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&victim)
            .status();

        // Forge ws/fake/worktrees/x pointing commondir -> victim/.git
        let fake_gd = ws.join("fake/worktrees/x");
        std::fs::create_dir_all(&fake_gd).expect("fake_gd");
        std::fs::write(ws.join(".git"), format!("gitdir: {}\n", fake_gd.display()))
            .expect("dotgit");
        std::fs::write(
            fake_gd.join("gitdir"),
            format!("{}\n", ws.join(".git").display()),
        )
        .expect("ptr");
        std::fs::write(fake_gd.join("HEAD"), "ref: refs/heads/main\n").expect("head");
        std::fs::write(
            fake_gd.join("commondir"),
            format!("{}\n", victim.join(".git").display()),
        )
        .expect("commondir");

        let policy = SandboxPolicy::for_workspace(ws.clone(), true, false);
        assert!(
            policy.frozen_git_write_dirs.is_empty(),
            "N6 regression: forged worktree commondir MUST be rejected!"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn scrubs_github_token_and_ssh_auth_sock_and_zshenv() {
        let dir = std::env::temp_dir().join(format!("carnot-sb-env-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let policy = SandboxPolicy::for_workspace(dir.clone(), true, false);
        let out = build_sandboxed_shell(
            &policy,
            "printf '%s|%s|%s' \"$ANTHROPIC_API_KEY\" \"$GITHUB_TOKEN\" \"$SSH_AUTH_SOCK\"",
            &BTreeMap::new(),
        )
        .expect("cmd")
        .output()
        .await
        .expect("run");
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout), "||");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn unsandboxed_preserves_runtime_env_scrubs_configured_keys_and_runs_bash() {
        let dir = std::env::temp_dir().join(format!("carnot-unsb-env-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let policy = SandboxPolicy::for_workspace(dir.clone(), false, true)
            .with_scrubbed_env_vars(vec!["CARNOT_TEST_SECRET_KEY".into()]);
        let mut extra = BTreeMap::new();
        extra.insert("CARNOT_TEST_RUNTIME_VAR".into(), "preserved_ok".into());
        let out = build_sandboxed_shell(
            &policy,
            "set -o pipefail; [[ -n \"$CARNOT_TEST_RUNTIME_VAR\" ]] && printf '%s|%s' \"$CARNOT_TEST_RUNTIME_VAR\" \"${CARNOT_TEST_SECRET_KEY:-}\"",
            &extra,
        )
        .expect("cmd")
        .output()
        .await
        .expect("run");
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout), "preserved_ok|");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn macos_seatbelt_protects_real_user_home_even_when_home_env_is_isolated() {
        let dir = std::env::temp_dir().join(format!("carnot-sb-home-{}", uuid::Uuid::new_v4()));
        let fake_home = dir.join("isolated_home");
        std::fs::create_dir_all(&fake_home).expect("mkdir");
        let policy = SandboxPolicy::for_workspace(dir.clone(), true, false);

        // Find a real sensitive file under /Users/$USER to verify Seatbelt still blocks it
        let real_user_home = std::env::var("USER")
            .ok()
            .map(|u| PathBuf::from("/Users").join(u))
            .filter(|p| p.is_dir());
        let Some(real_home) = real_user_home else {
            let _ = std::fs::remove_dir_all(&dir);
            return;
        };
        let probe = [".ssh/known_hosts", ".zshenv", ".zshrc", ".zsh_history"]
            .into_iter()
            .map(|rel| real_home.join(rel))
            .find(|p| p.is_file());
        let Some(target) = probe else {
            let _ = std::fs::remove_dir_all(&dir);
            return;
        };

        let mut extra = BTreeMap::new();
        extra.insert("HOME".to_string(), fake_home.display().to_string());
        let cmd_str = format!(
            "cat {:?} >/dev/null 2>&1 && echo READABLE || echo BLOCKED",
            target
        );
        let out = build_sandboxed_shell(&policy, &cmd_str, &extra)
            .expect("cmd")
            .output()
            .await
            .expect("run");
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            "BLOCKED",
            "Seatbelt must still block {:?} even when HOME is overridden to {:?}",
            target,
            fake_home
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
