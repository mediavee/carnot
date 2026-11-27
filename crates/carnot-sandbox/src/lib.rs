//! Hardened OS-native process sandboxing (macOS Seatbelt, Linux bwrap + Landlock fallback) and
//! strict environment allowlisting.
//!
//! Security invariants:
//!   - `<workspace>/.git` is READ-ONLY inside Seatbelt & bwrap (sandboxed processes can never
//!     write `.git/config` or `.git/hooks/*` to hijack host-side `git`).
//!   - Linked git worktrees are verified strictly (`gd` outside workspace, `cd == gd/../..`,
//!     reciprocal `worktrees/<name>/gitdir` -> `<ws>/.git`) and ONLY `worktrees/<name>`
//!     + `cd/{objects,refs,logs}` are writable — never `cd/config` or `cd/hooks`.
//!   - Host-side git invocations use `host_safe_git_command()` (`-c core.fsmonitor=false -c core.hooksPath=/dev/null`).
//!   - Strict environment allowlist (`env_clear()` + safe baseline vars + isolated `ZDOTDIR`,
//!     no `BASH_ENV`/`ENV`) + Seatbelt/bwrap/Landlock block `~/.ssh`, `~/.aws`, `~/.gnupg`,
//!     `~/.config/gh`, `~/.secrets`, `~/.zshenv`, `~/.zshrc`, `~/.claude`, `~/.codex`,
//!     `~/.config/agentsafe`, `~/.docker`, `~/.kube`.
//!   - Workspace-isolated `TMPDIR` (`.carnot/tmp`), network OFF by default, fail-closed if no OS sandbox.

mod landlock;

pub use landlock::{
    LANDLOCK_EXEC_ARG, detect_landlock_abi, landlock_helper, register_landlock_helper,
    try_handle_landlock_exec_arg,
};

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
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
        let canonical_ws = workspace_root.canonicalize().unwrap_or(workspace_root);
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

    #[must_use]
    pub fn with_extra_writable_roots(mut self, roots: Vec<PathBuf>) -> Self {
        self.extra_writable_roots = roots
            .into_iter()
            .map(|r| r.canonicalize().unwrap_or(r))
            .collect();
        self
    }

    #[must_use]
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
/// `core.fsmonitor`, `core.hooksPath`, and `core.sshCommand`.
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

/// Default non-interactive environment variables applied to agent subshells before `extra_env`
/// so pagers, credential prompts, and progress bars never hang or flood observation budgets.
const NON_INTERACTIVE_SHELL_ENV: &[(&str, &str)] = &[
    ("GIT_TERMINAL_PROMPT", "0"),
    ("PAGER", "cat"),
    ("MANPAGER", "cat"),
    ("PIP_PROGRESS_BAR", "off"),
    ("TQDM_DISABLE", "1"),
];

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
    // Neutralize shell startup scripts that could reload host secrets.
    cmd.env("BASH_ENV", "");
    cmd.env("ENV", "");
    for (k, v) in NON_INTERACTIVE_SHELL_ENV {
        cmd.env(k, v);
    }
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
    for (k, v) in NON_INTERACTIVE_SHELL_ENV {
        cmd.env(k, v);
    }
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
        // Isolate child TMPDIR & ZDOTDIR inside the workspace (prevents host /tmp writes and ~/.zshenv leaks).
        let ws_tmp = policy.workspace_root.join(".carnot").join("tmp");
        let zdot = ws_tmp.join("zdotdir");
        let _ = std::fs::create_dir_all(&zdot);
        cmd.env("TMPDIR", &ws_tmp);
        cmd.env("ZDOTDIR", &zdot);
        // Inside an offline OS sandbox, host compiler wrapper daemons configured in `~/.cargo/config.toml`
        // (e.g. `sccache`) cannot reach their local sockets/network and fail `cargo test`.
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

/// Portable, OS-sandboxed command specification extracted from [`build_sandboxed_shell_in_dir`]
/// so alternate spawners (such as PTY executors) reuse the exact same sandbox wrappers and env
/// scrubbing without duplicating policy logic.
#[derive(Debug, Clone)]
pub struct ShellCommandSpec {
    pub program: std::ffi::OsString,
    pub args: Vec<std::ffi::OsString>,
    pub cwd: PathBuf,
    pub clear_env: bool,
    pub envs: Vec<(std::ffi::OsString, Option<std::ffi::OsString>)>,
}

/// Builds a [`ShellCommandSpec`] with the exact same OS sandbox wrapper and environment scrubbing
/// as [`build_sandboxed_shell_in_dir`].
pub fn build_sandboxed_shell_spec_in_dir(
    policy: &SandboxPolicy,
    shell_cmd: &str,
    cwd: &Path,
    extra_env: &BTreeMap<String, String>,
) -> std::result::Result<ShellCommandSpec, String> {
    let cmd = build_sandboxed_shell_in_dir(policy, shell_cmd, cwd, extra_env)?;
    let std_cmd = cmd.as_std();
    let program = std_cmd.get_program().to_os_string();
    let args = std_cmd
        .get_args()
        .map(std::ffi::OsStr::to_os_string)
        .collect();
    let envs = std_cmd
        .get_envs()
        .map(|(k, v)| (k.to_os_string(), v.map(std::ffi::OsStr::to_os_string)))
        .collect();
    Ok(ShellCommandSpec {
        program,
        args,
        cwd: std_cmd
            .get_current_dir()
            .map_or_else(|| cwd.to_path_buf(), Path::to_path_buf),
        clear_env: policy.enabled,
        envs,
    })
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
    ("CARNOT_AUTH", ".carnot/auth.json"),
    ("CARNOT_AUTH_LOCK", ".carnot/auth.json.lock"),
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

/// Dynamic credential file paths driven by environment variables (`CARGO_HOME`, `CARNOT_AUTH_FILE`)
/// that must stay unreadable inside both Seatbelt and `bwrap` even when placed outside `$HOME`.
fn extra_sensitive_credential_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(cargo_home) = std::env::var_os("CARGO_HOME")
        && !cargo_home.is_empty()
    {
        let ch = PathBuf::from(cargo_home);
        out.push(ch.join("credentials"));
        out.push(ch.join("credentials.toml"));
    }
    if let Some(auth_file) = std::env::var_os("CARNOT_AUTH_FILE")
        && !auth_file.is_empty()
    {
        let af = PathBuf::from(auth_file);
        out.push(af.with_extension("json.lock"));
        out.push(af);
    }
    out
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
            let _ = writeln!(profile, "  (subpath (param \"{key}\"))");
            c.arg("-D")
                .arg(format!("{key}={}", home.join(rel).display()));
        }
        for (tag, rel) in MACOS_DENY_LITERALS {
            let key = format!("DENY_{tag}_{h_idx}");
            let _ = writeln!(profile, "  (literal (param \"{key}\"))");
            c.arg("-D")
                .arg(format!("{key}={}", home.join(rel).display()));
        }
    }
    for (e_idx, extra_cred) in extra_sensitive_credential_files().into_iter().enumerate() {
        let key = format!("DENY_EXTRA_CRED_{e_idx}");
        let resolved = extra_cred.canonicalize().unwrap_or(extra_cred);
        let _ = writeln!(profile, "  (literal (param \"{key}\"))");
        c.arg("-D").arg(format!("{key}={}", resolved.display()));
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
        let _ = writeln!(profile, "(allow file-write* (subpath (param \"{key}\")))");
        c.arg("-D").arg(format!("{key}={}", gdir.display()));
    }
    for (idx, root) in policy.extra_writable_roots.iter().enumerate() {
        let key = format!("EXTRA_{idx}");
        let _ = writeln!(profile, "(allow file-write* (subpath (param \"{key}\")))");
        c.arg("-D").arg(format!("{key}={}", root.display()));
    }

    c.arg("-p").arg(profile);
    c.arg(resolve_shell_binary()).arg("-c").arg(shell_cmd);
    c
}

/// Hardened Linux `bubblewrap` (`bwrap`) backend (fail-closed, zero `unsafe`):
///   - `--unshare-all` (isolates PID, IPC, UTS, cgroup, user, net namespaces so parent `/proc/<pid>/environ`
///     and host Unix sockets are unreachable) + `--share-net` only when `policy.allow_network == true`.
///   - Explicit minimal read-only mounts (`/usr`, `/bin`, `/sbin`, `/lib`, `/lib64`, `/etc`, `/opt`)
///     instead of `--ro-bind / /` (prevents connecting to host D-Bus / `docker.sock` / SSH agent sockets).
///   - Fresh `--tmpfs /run`, `--tmpfs /tmp`, and `--tmpfs $XDG_RUNTIME_DIR`.
///   - Read-only `<ws>/.git`, `--new-session` (blocks TIOCSTI), `--die-with-parent`, and
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
    static CACHED: LazyLock<Option<PathBuf>> = LazyLock::new(|| {
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
            .is_ok_and(|s| s.success());
        ok.then_some(bin)
    });
    CACHED.as_deref()
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
            ".carnot/auth.json",
            ".carnot/auth.json.lock",
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
    for extra_cred in extra_sensitive_credential_files() {
        if extra_cred.is_file() {
            c.args(["--ro-bind", "/dev/null", &extra_cred.to_string_lossy()]);
        }
    }

    let chdir = cwd.to_string_lossy().to_string();
    c.args(["--chdir", &chdir, resolve_shell_binary(), "-c", shell_cmd]);
    c
}

/// Strictly validates a linked git worktree once on the host:
///   - `gd` must lie outside `canonical_ws`
///   - `gd` must have layout `<cd>/worktrees/<name>` (so `cd == gd/../..`)
///   - `<gd>/gitdir` must reciprocally resolve to `<canonical_ws>/.git`
///   - Only safe object/ref/worktree subdirs are returned — never `cd/config` or `cd/hooks`.
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

    // Guard: `gd` and `cd` must lie outside workspace, and `cd` must be `gd/../..` under `worktrees/`.
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

    // Open only `gd`, `cd/objects`, `cd/refs`, `cd/logs` — never `cd/config` or `cd/hooks`.
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
    async fn blocks_dot_git_config_tampering_under_seatbelt() {
        let sb = detect_sandbox_backend();
        // Landlock is allow-only (cannot carve `.git` out of a writable workspace_root);
        // in-process tools still enforce `.git` write denial across all platforms.
        if !sb.available || sb.backend == "landlock" {
            return;
        }
        let base = std::env::temp_dir().join(format!("carnot-git-ro-{}", uuid::Uuid::new_v4()));
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
            "sandboxed process must not write .git/config"
        );
        let cfg = std::fs::read_to_string(ws.join(".git/config")).expect("read config");
        assert!(!cfg.contains("fsmonitor"));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn blocks_forged_commondir_in_fake_worktree() {
        let base = std::env::temp_dir().join(format!("carnot-wt-{}", uuid::Uuid::new_v4()));
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

        let policy = SandboxPolicy::for_workspace(ws, true, false);
        assert!(
            policy.frozen_git_write_dirs.is_empty(),
            "forged worktree commondir must be rejected"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn scrubs_github_token_and_ssh_auth_sock_and_zshenv() {
        if !detect_sandbox_backend().available {
            return;
        }
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
        let cmd_str = format!("cat {target:?} >/dev/null 2>&1 && echo READABLE || echo BLOCKED");
        let out = build_sandboxed_shell(&policy, &cmd_str, &extra)
            .expect("cmd")
            .output()
            .await
            .expect("run");
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            "BLOCKED",
            "Seatbelt must still block {target:?} even when HOME is overridden to {fake_home:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn non_interactive_shell_env_defaults_and_extra_env_override() {
        let dir = std::env::temp_dir().join(format!("carnot-sb-nonint-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let policy = SandboxPolicy::for_workspace(dir.clone(), false, true);

        let out = build_sandboxed_shell(
            &policy,
            "printf '%s|%s|%s|%s|%s' \"$GIT_TERMINAL_PROMPT\" \"$PAGER\" \"$MANPAGER\" \"$PIP_PROGRESS_BAR\" \"$TQDM_DISABLE\"",
            &BTreeMap::new(),
        )
        .expect("cmd")
        .output()
        .await
        .expect("run");
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout), "0|cat|cat|off|1");

        let mut extra = BTreeMap::new();
        extra.insert("PAGER".into(), "custom-pager".into());
        let out_override = build_sandboxed_shell(&policy, "printf '%s' \"$PAGER\"", &extra)
            .expect("cmd")
            .output()
            .await
            .expect("run");
        assert!(out_override.status.success());
        assert_eq!(
            String::from_utf8_lossy(&out_override.stdout),
            "custom-pager",
            "extra_env must override default PAGER"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn host_safe_git_command_neutralizes_repo_hooks_fsmonitor_and_ssh() {
        let dir = PathBuf::from("/tmp/example-repo");
        let cmd = host_safe_git_command(&dir);
        let args: Vec<String> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert!(
            args.contains(&"core.fsmonitor=false".to_string()),
            "host_safe_git_command must disable core.fsmonitor"
        );
        assert!(
            args.contains(&"core.hooksPath=/dev/null".to_string()),
            "host_safe_git_command must neutralize core.hooksPath"
        );
        assert!(
            args.contains(&"core.sshCommand=false".to_string()),
            "host_safe_git_command must disable core.sshCommand"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn blocks_symlink_write_escape_to_dot_git_and_outside_workspace() {
        struct Cleanup(PathBuf, PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
                let _ = std::fs::remove_dir_all(&self.1);
            }
        }

        let sb = detect_sandbox_backend();
        if !sb.available || sb.backend == "landlock" {
            return;
        }
        let id = uuid::Uuid::new_v4();
        let ws_base = std::env::temp_dir().join(format!("carnot-sb-escape-ws-{id}"));
        let ws = ws_base.join("ws");
        // Place the outside target outside `/tmp` (since Linux `bwrap` mounts a writable `--tmpfs /tmp`)
        // and outside `ws` so both Seatbelt and `bwrap` must block writes through the symlink.
        let outside_base = std::env::var_os("HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| std::env::temp_dir()))
            .join(format!(".carnot-sb-escape-out-{id}"));
        let outside = outside_base.join("outside.txt");
        let _cleanup = Cleanup(ws_base.clone(), outside_base.clone());

        std::fs::create_dir_all(&ws).expect("ws");
        std::fs::create_dir_all(&outside_base).expect("outside_base");
        std::fs::write(&outside, "original").expect("outside");
        let init = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&ws)
            .status()
            .expect("git init");
        assert!(init.success(), "git init must succeed");

        // Create symlinks inside workspace pointing to .git/config and outside workspace.
        let git_link = ws.join("escape_git");
        let out_link = ws.join("escape_out");
        std::os::unix::fs::symlink(ws.join(".git/config"), &git_link).expect("symlink git");
        std::os::unix::fs::symlink(&outside, &out_link).expect("symlink outside");

        let policy = SandboxPolicy::for_workspace(ws.clone(), true, false);
        let out = build_sandboxed_shell(
            &policy,
            "echo hacked >> escape_git 2>/dev/null && echo GIT_WRITABLE || echo GIT_BLOCKED; \
             echo hacked >> escape_out 2>/dev/null && echo OUT_WRITABLE || echo OUT_BLOCKED",
            &BTreeMap::new(),
        )
        .expect("cmd")
        .output()
        .await
        .expect("run");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains("GIT_BLOCKED") && stdout.contains("OUT_BLOCKED"),
            "sandbox must block symlink writes to .git/config and outside workspace, got: {stdout}"
        );
        assert_eq!(
            std::fs::read_to_string(&outside).expect("read outside"),
            "original",
            "outside file must remain untouched"
        );
    }
}
