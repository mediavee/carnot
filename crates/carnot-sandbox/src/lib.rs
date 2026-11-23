//! Hardened OS-native process sandboxing (macOS Seatbelt) and strict environment allowlisting.
//!
//! Security invariants (verified against Opus 5.5 High red-team exploits C1, S1, S3, N6, N7):
//!   - `<workspace>/.git` is READ-ONLY inside Seatbelt (closes C1: sandboxed process can NEVER
//!     write `.git/config` or `.git/hooks/*` to hijack host-side `git`!).
//!   - Linked git worktrees are verified strictly (`gd` outside workspace, `cd == gd/../..`,
//!     reciprocal `worktrees/<name>/gitdir` -> `<ws>/.git` — closes N6) and ONLY `worktrees/<name>`
//!     + `cd/{objects,refs,logs}` are writable — NEVER `cd/config` or `cd/hooks`!
//!   - Host-side git invocations use `host_safe_git_command()` (`-c core.fsmonitor=false -c core.hooksPath=/dev/null`).
//!   - Strict environment allowlist (`env_clear()` + safe baseline vars + isolated `ZDOTDIR`,
//!     no `BASH_ENV`/`ENV`) + Seatbelt denies `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.config/gh`,
//!     `~/.secrets`, `~/.zshenv`, `~/.zshrc`, `~/.claude`, `~/.codex`, `~/.config/agentsafe`,
//!     `~/.docker`, `~/.kube` (closes S3 & N7).
//!   - Workspace-isolated `TMPDIR` (`.carnot/tmp`), network OFF by default, fail-closed if no OS sandbox.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
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
            && let Some(bwrap_bin) = resolve_bwrap_binary()
        {
            build_linux_bwrap_command(&bwrap_bin, policy, shell_cmd, cwd)
        } else {
            return Err(
                "OS sandbox requested (`sandbox: true`) but no supported kernel sandbox (macOS sandbox-exec / Linux bwrap) is available on this host (pass `--no-sandbox` to explicitly disable)"
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
         (allow file-read*)\n\
         (deny file-read*\n\
           (subpath (param \"DENY_SSH\"))\n\
           (subpath (param \"DENY_AWS\"))\n\
           (subpath (param \"DENY_GNUPG\"))\n\
           (subpath (param \"DENY_GH\"))\n\
           (subpath (param \"DENY_SECRETS\"))\n\
           (literal (param \"DENY_ZSHENV\"))\n\
           (literal (param \"DENY_ZSHRC\"))\n\
           (literal (param \"DENY_ZSH_HIST\"))\n\
           (literal (param \"DENY_BASH_HIST\"))\n\
           (literal (param \"DENY_PY_HIST\"))\n\
           (literal (param \"DENY_PSQL_HIST\"))\n\
           (literal (param \"DENY_HOME_ENV\"))\n\
           (literal (param \"DENY_PYPIRC\"))\n\
           (subpath (param \"DENY_CLAUDE\"))\n\
           (subpath (param \"DENY_CODEX\"))\n\
           (subpath (param \"DENY_AGENTSAFE\"))\n\
           (subpath (param \"DENY_DOCKER\"))\n\
           (subpath (param \"DENY_KUBE\"))\n\
           (subpath (param \"DENY_OP\"))\n\
           (subpath (param \"DENY_KEYCHAINS\"))\n\
           (literal (param \"DENY_NETRC\"))\n\
           (literal (param \"DENY_GIT_CREDS\"))\n\
           (literal (param \"DENY_CONFIG_GIT_CREDS\"))\n\
           (literal (param \"DENY_NPMRC\"))\n\
           (literal (param \"DENY_CARGO_CREDS\"))\n\
           (literal (param \"DENY_CARGO_CREDS_TOML\"))\n\
           (subpath (param \"DENY_GCLOUD\"))\n\
           (literal (param \"DENY_CARNOT_TOKEN\")))\n\
         (allow file-write* (literal \"/dev/null\") (literal \"/dev/tty\"))\n\
         (allow file-write* (require-all (subpath (param \"WORKSPACE\")) (require-not (subpath (param \"WS_DOT_GIT\")))))\n",
    );
    if policy.allow_network {
        profile.push_str("(allow network*)\n");
    }

    let mut c = Command::new("/usr/bin/sandbox-exec");
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .and_then(|p| p.canonicalize().ok().or(Some(p)))
        .unwrap_or_else(|| PathBuf::from("/nonexistent"));

    c.arg("-D")
        .arg(format!("WORKSPACE={}", policy.workspace_root.display()));
    c.arg("-D").arg(format!(
        "WS_DOT_GIT={}",
        policy.workspace_root.join(".git").display()
    ));
    c.arg("-D")
        .arg(format!("DENY_SSH={}", home.join(".ssh").display()));
    c.arg("-D")
        .arg(format!("DENY_AWS={}", home.join(".aws").display()));
    c.arg("-D")
        .arg(format!("DENY_GNUPG={}", home.join(".gnupg").display()));
    c.arg("-D")
        .arg(format!("DENY_GH={}", home.join(".config/gh").display()));
    c.arg("-D")
        .arg(format!("DENY_SECRETS={}", home.join(".secrets").display()));
    c.arg("-D")
        .arg(format!("DENY_ZSHENV={}", home.join(".zshenv").display()));
    c.arg("-D")
        .arg(format!("DENY_ZSHRC={}", home.join(".zshrc").display()));
    c.arg("-D").arg(format!(
        "DENY_ZSH_HIST={}",
        home.join(".zsh_history").display()
    ));
    c.arg("-D").arg(format!(
        "DENY_BASH_HIST={}",
        home.join(".bash_history").display()
    ));
    c.arg("-D").arg(format!(
        "DENY_PY_HIST={}",
        home.join(".python_history").display()
    ));
    c.arg("-D").arg(format!(
        "DENY_PSQL_HIST={}",
        home.join(".psql_history").display()
    ));
    c.arg("-D")
        .arg(format!("DENY_HOME_ENV={}", home.join(".env").display()));
    c.arg("-D")
        .arg(format!("DENY_PYPIRC={}", home.join(".pypirc").display()));
    c.arg("-D")
        .arg(format!("DENY_CLAUDE={}", home.join(".claude").display()));
    c.arg("-D")
        .arg(format!("DENY_CODEX={}", home.join(".codex").display()));
    c.arg("-D").arg(format!(
        "DENY_AGENTSAFE={}",
        home.join(".config/agentsafe").display()
    ));
    c.arg("-D")
        .arg(format!("DENY_DOCKER={}", home.join(".docker").display()));
    c.arg("-D")
        .arg(format!("DENY_KUBE={}", home.join(".kube").display()));
    c.arg("-D")
        .arg(format!("DENY_OP={}", home.join(".config/op").display()));
    c.arg("-D").arg(format!(
        "DENY_KEYCHAINS={}",
        home.join("Library/Keychains").display()
    ));
    c.arg("-D")
        .arg(format!("DENY_NETRC={}", home.join(".netrc").display()));
    c.arg("-D").arg(format!(
        "DENY_GIT_CREDS={}",
        home.join(".git-credentials").display()
    ));
    c.arg("-D").arg(format!(
        "DENY_CONFIG_GIT_CREDS={}",
        home.join(".config/git/credentials").display()
    ));
    c.arg("-D")
        .arg(format!("DENY_NPMRC={}", home.join(".npmrc").display()));
    c.arg("-D").arg(format!(
        "DENY_CARGO_CREDS={}",
        home.join(".cargo/credentials").display()
    ));
    c.arg("-D").arg(format!(
        "DENY_CARGO_CREDS_TOML={}",
        home.join(".cargo/credentials.toml").display()
    ));
    c.arg("-D").arg(format!(
        "DENY_GCLOUD={}",
        home.join(".config/gcloud").display()
    ));
    c.arg("-D").arg(format!(
        "DENY_CARNOT_TOKEN={}",
        home.join(".carnot/server.token").display()
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
}
