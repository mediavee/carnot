//! Official Carnot Rust Extension: Declarative Hooks 100% compatible with Claude Code hooks
//! (`PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `SessionStart`, `PreCompact`, `Stop`).
//!
//! Supports:
//!   - Exact Claude Code `settings.json` / `hooks.json` schema (`hooks.<Event>[].{matcher, hooks[]}`)
//!   - Exact Claude Code stdin JSON payloads, exit codes (`0` JSON/text, `2` blocking error),
//!     `hookSpecificOutput.{permissionDecision, permissionDecisionReason, updatedInput, additionalContext}`,
//!     `decision: "block"`, `continue: false`, `systemMessage`, `timeout`, and `async: true`
//!   - `${CLAUDE_PLUGIN_ROOT}` / `${CLAUDE_PROJECT_DIR}` / `${CARNOT_PROJECT_DIR}` expansion
//!   - Optional Carnot `file_glob` matcher on `tool_input.file_path` / `tool_input.path`
//!   - Anti-RCE confinement: workspace `.carnot/hooks.json` and `.claude/settings*.json` are
//!     NEVER executed unless `trust_workspace_hooks` is explicitly opted in.

use carnot_core::{CarnotConfig, CarnotError, Result, ToolOutput, ToolResultBlock};
use carnot_ext::{
    BoundaryAction, BoxFuture, CompactionHookAction, CompactionHookContext, Module,
    ToolGateDecision, TurnBoundarySnapshot,
};
use carnot_sandbox::apply_safe_env;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

const DEFAULT_HOOK_TIMEOUT_SECS: u64 = 30;
const MAX_HOOK_TIMEOUT_SECS: u64 = 600;
const MAX_HOOK_OUTPUT_BYTES: usize = 64 * 1024;

/// Supported Claude Code / Carnot lifecycle hook events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HookEvent {
    PreToolUse,
    PostToolUse,
    PostToolUseFailure,
    SessionStart,
    PreCompact,
    Stop,
}

impl HookEvent {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::PostToolUseFailure => "PostToolUseFailure",
            Self::SessionStart => "SessionStart",
            Self::PreCompact => "PreCompact",
            Self::Stop => "Stop",
        }
    }

    fn from_key(key: &str) -> Option<Self> {
        match key {
            "PreToolUse" | "pre_tool_use" => Some(Self::PreToolUse),
            "PostToolUse" | "post_tool_use" => Some(Self::PostToolUse),
            "PostToolUseFailure" | "post_tool_use_failure" => Some(Self::PostToolUseFailure),
            "SessionStart" | "session_start" => Some(Self::SessionStart),
            "PreCompact" | "pre_compact" => Some(Self::PreCompact),
            "Stop" | "stop" => Some(Self::Stop),
            _ => None,
        }
    }
}

/// Discovery & security configuration for [`HooksModule`].
#[derive(Debug, Clone, Default)]
pub struct HooksConfig {
    /// Explicit `hooks.json` or `settings.json` files to load.
    pub files: Vec<PathBuf>,
    /// Load user-global `~/.carnot/hooks.json` if present (default `true`).
    pub load_carnot_user_hooks: bool,
    /// Load user-global `~/.claude/settings.json` hooks if present.
    pub load_claude_user_settings: bool,
    /// Opt-in to load workspace-local `.carnot/hooks.json`, `.claude/settings.json`, and
    /// `.claude/settings.local.json` (disabled by default to block repo RCE).
    pub trust_workspace_hooks: bool,
    /// If true, non-zero non-2 exit codes or timeouts on `PreToolUse` deny the tool call.
    pub fail_closed: bool,
}

impl HooksConfig {
    pub fn user_default() -> Self {
        Self {
            load_carnot_user_hooks: true,
            ..Self::default()
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawHookEntry {
    #[serde(default)]
    matcher: Option<String>,
    /// Carnot extension: optional glob matched against `tool_input.file_path` or `tool_input.path`.
    #[serde(default)]
    file_glob: Option<String>,
    #[serde(default)]
    hooks: Vec<RawHookHandler>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawHookHandler {
    #[serde(rename = "type", default = "default_hook_type")]
    kind: String,
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    allowed_env_vars: Vec<String>,
    #[serde(default)]
    timeout: Option<u64>,
    #[serde(rename = "async", default)]
    is_async: bool,
}

fn default_hook_type() -> String {
    "command".to_string()
}

/// Execution backend for a compiled hook (`command` or Claude Code `http` webhook).
#[derive(Debug, Clone)]
pub enum HookAction {
    Command(String),
    Http {
        url: String,
        headers: BTreeMap<String, String>,
        allowed_env_vars: Vec<String>,
    },
}

/// Compiled single command or HTTP hook rule ready for zero-overhead matching.
#[derive(Debug, Clone)]
pub struct CompiledHook {
    pub event: HookEvent,
    pub matcher_raw: String,
    matcher_re: Option<Regex>,
    file_glob_re: Option<Regex>,
    pub action: HookAction,
    pub timeout_secs: u64,
    pub is_async: bool,
    pub plugin_root: Option<PathBuf>,
    pub source: String,
}

impl CompiledHook {
    fn matches(&self, target: &str, tool_input: Option<&Value>) -> bool {
        if let Some(ref re) = self.matcher_re
            && !re.is_match(target)
        {
            return false;
        }
        if let Some(ref glob_re) = self.file_glob_re {
            let path = tool_input
                .and_then(|v| {
                    v.get("file_path")
                        .or_else(|| v.get("path"))
                        .and_then(Value::as_str)
                })
                .unwrap_or("");
            if !glob_re.is_match(path) {
                return false;
            }
        }
        true
    }
}

/// Converts a simple glob (`*.rs`, `src/**/*.ts`) to a regex without external glob crates.
fn glob_to_regex(glob: &str) -> Option<Regex> {
    let g = glob.trim();
    if g.is_empty() || g == "*" || g == "**" {
        return None;
    }
    let mut rx = String::with_capacity(g.len() * 2 + 4);
    if !g.contains('/') {
        // Basename glob like `*.rs`: match end of path or after `/`
        rx.push_str("(?:^|/)");
    } else {
        rx.push('^');
    }
    let chars: Vec<char> = g.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    rx.push_str(".*");
                    i += 2;
                    if i < chars.len() && chars[i] == '/' {
                        rx.push_str("/?");
                        i += 1;
                    }
                    continue;
                }
                rx.push_str("[^/]*");
            }
            '?' => rx.push_str("[^/]"),
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '{' | '}' | '[' | ']' | '\\' => {
                rx.push('\\');
                rx.push(chars[i]);
            }
            c => rx.push(c),
        }
        i += 1;
    }
    rx.push('$');
    Regex::new(&rx).ok()
}

/// Expands `${CLAUDE_PLUGIN_ROOT}`, `${CLAUDE_PROJECT_DIR}`, and `${CARNOT_PROJECT_DIR}`.
pub fn expand_hook_vars(raw: &str, project_dir: &Path, plugin_root: Option<&Path>) -> String {
    let proj = project_dir.to_string_lossy();
    let mut out = raw
        .replace("${CLAUDE_PROJECT_DIR}", &proj)
        .replace("${CARNOT_PROJECT_DIR}", &proj);
    if let Some(pr) = plugin_root {
        out = out.replace("${CLAUDE_PLUGIN_ROOT}", &pr.to_string_lossy());
    }
    out
}

/// Parses a Claude Code `settings.json` / `hooks.json` JSON value into compiled hook rules.
pub fn parse_hooks_value(
    val: &Value,
    project_dir: &Path,
    plugin_root: Option<&Path>,
    source: &str,
) -> Result<Vec<CompiledHook>> {
    // Support both `{ "hooks": { "PreToolUse": [...] } }` (settings.json / plugin hooks.json)
    // and flat `{ "PreToolUse": [...] }`.
    let hooks_obj = val
        .get("hooks")
        .and_then(Value::as_object)
        .or_else(|| val.as_object());
    let Some(map) = hooks_obj else {
        return Ok(Vec::new());
    };

    let mut out = Vec::new();
    for (event_key, entries_val) in map {
        let Some(event) = HookEvent::from_key(event_key) else {
            continue;
        };
        let Some(entries_arr) = entries_val.as_array() else {
            continue;
        };
        for raw_entry_val in entries_arr {
            let entry: RawHookEntry =
                serde_json::from_value(raw_entry_val.clone()).map_err(|e| CarnotError::Tool {
                    tool: "hooks".into(),
                    message: format!("invalid hook entry in {source} ({event_key}): {e}"),
                })?;
            let matcher_raw = entry.matcher.unwrap_or_default();
            let matcher_re = {
                let m = matcher_raw.trim();
                if m.is_empty() || m == "*" {
                    None
                } else {
                    Some(Regex::new(m).map_err(|e| CarnotError::Tool {
                        tool: "hooks".into(),
                        message: format!("invalid hook matcher regex `{m}` in {source}: {e}"),
                    })?)
                }
            };
            let file_glob_re = entry.file_glob.as_deref().and_then(glob_to_regex);

            for handler in entry.hooks {
                let action = if handler.kind.eq_ignore_ascii_case("command") {
                    let Some(cmd_raw) = handler.command.filter(|c| !c.trim().is_empty()) else {
                        continue;
                    };
                    HookAction::Command(expand_hook_vars(&cmd_raw, project_dir, plugin_root))
                } else if handler.kind.eq_ignore_ascii_case("http") {
                    let Some(url_raw) = handler.url.filter(|u| {
                        let t = u.trim();
                        t.starts_with("https://") || t.starts_with("http://")
                    }) else {
                        continue;
                    };
                    HookAction::Http {
                        url: expand_hook_vars(&url_raw, project_dir, plugin_root),
                        headers: handler.headers,
                        allowed_env_vars: handler.allowed_env_vars,
                    }
                } else {
                    tracing::debug!(
                        source = source,
                        hook_type = %handler.kind,
                        "skipping unsupported hook type"
                    );
                    continue;
                };
                let timeout_secs = handler
                    .timeout
                    .unwrap_or(DEFAULT_HOOK_TIMEOUT_SECS)
                    .clamp(1, MAX_HOOK_TIMEOUT_SECS);
                out.push(CompiledHook {
                    event,
                    matcher_raw: matcher_raw.clone(),
                    matcher_re: matcher_re.clone(),
                    file_glob_re: file_glob_re.clone(),
                    action,
                    timeout_secs,
                    is_async: handler.is_async,
                    plugin_root: plugin_root.map(Path::to_path_buf),
                    source: source.to_string(),
                });
            }
        }
    }
    Ok(out)
}

/// Parses a `hooks.json` or `settings.json` file on disk.
pub fn parse_hooks_file(
    path: &Path,
    project_dir: &Path,
    plugin_root: Option<&Path>,
) -> Result<Vec<CompiledHook>> {
    let raw = std::fs::read_to_string(path).map_err(|e| CarnotError::Tool {
        tool: "hooks".into(),
        message: format!("failed to read hooks file {}: {e}", path.display()),
    })?;
    let val: Value = serde_json::from_str(&raw).map_err(|e| CarnotError::Tool {
        tool: "hooks".into(),
        message: format!("invalid JSON in hooks file {}: {e}", path.display()),
    })?;
    parse_hooks_value(&val, project_dir, plugin_root, &path.display().to_string())
}

#[derive(Debug)]
struct HookExecResult {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

fn expand_allowed_env(raw: &str, allowed: &[String]) -> String {
    let mut out = String::with_capacity(raw.len());
    let bytes = raw.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() {
            if bytes[i + 1] == b'{'
                && let Some(end_rel) = raw[i + 2..].find('}')
            {
                let name = &raw[i + 2..i + 2 + end_rel];
                if allowed.iter().any(|a| a == name) {
                    out.push_str(&std::env::var(name).unwrap_or_default());
                    i += 2 + end_rel + 1;
                    continue;
                }
            } else if bytes[i + 1].is_ascii_alphabetic() || bytes[i + 1] == b'_' {
                let mut end = i + 2;
                while end < bytes.len()
                    && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_')
                {
                    end += 1;
                }
                let name = &raw[i + 1..end];
                if allowed.iter().any(|a| a == name) {
                    out.push_str(&std::env::var(name).unwrap_or_default());
                    i = end;
                    continue;
                }
            }
        }
        let ch = raw[i..].chars().next().unwrap_or('\0');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn build_hook_http_client(timeout_secs: u64) -> std::result::Result<reqwest::Client, String> {
    let make = || {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(timeout_secs))
            .redirect(reqwest::redirect::Policy::none())
            .user_agent(format!("carnot-hooks/{}", env!("CARGO_PKG_VERSION")))
    };
    if let Ok(c) = make().build() {
        return Ok(c);
    }
    let certs: Vec<reqwest::Certificate> = webpki_root_certs::TLS_SERVER_ROOT_CERTS
        .iter()
        .filter_map(|der| reqwest::Certificate::from_der(der.as_ref()).ok())
        .collect();
    make()
        .tls_certs_only(certs)
        .build()
        .map_err(|e| format!("failed to build HTTP hook client: {e}"))
}

async fn run_command_hook(
    hook: &CompiledHook,
    workspace_root: &Path,
    stdin_payload: &Value,
) -> std::result::Result<HookExecResult, String> {
    let command_str = match &hook.action {
        HookAction::Command(c) => c,
        HookAction::Http {
            url,
            headers,
            allowed_env_vars,
        } => {
            let client = build_hook_http_client(hook.timeout_secs)?;
            let mut req = client.post(url).header("content-type", "application/json");
            for (k, v) in headers {
                let expanded = expand_allowed_env(v, allowed_env_vars);
                req = req.header(k.as_str(), expanded);
            }
            if hook.is_async {
                let payload = stdin_payload.clone();
                let source = hook.source.clone();
                tokio::spawn(async move {
                    if let Err(e) = req.json(&payload).send().await {
                        tracing::warn!(source = %source, error = %e, "async HTTP hook failed");
                    }
                });
                return Ok(HookExecResult {
                    exit_code: 0,
                    stdout: String::new(),
                    stderr: String::new(),
                });
            }
            let resp = req
                .json(stdin_payload)
                .send()
                .await
                .map_err(|e| format!("HTTP hook `{}` failed: {e}", hook.source))?;
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let body_clamped: String = body.chars().take(MAX_HOOK_OUTPUT_BYTES).collect();
            return if status.is_success() {
                Ok(HookExecResult {
                    exit_code: 0,
                    stdout: body_clamped.trim().to_string(),
                    stderr: String::new(),
                })
            } else {
                Ok(HookExecResult {
                    exit_code: 1,
                    stdout: String::new(),
                    stderr: format!("HTTP {status}: {}", body_clamped.trim()),
                })
            };
        }
    };

    let mut cmd = if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(command_str);
        c
    } else {
        let mut c = Command::new("/bin/sh");
        c.arg("-c").arg(command_str);
        c
    };

    let mut extra_env = BTreeMap::new();
    let proj_str = workspace_root.to_string_lossy().to_string();
    extra_env.insert("CLAUDE_PROJECT_DIR".into(), proj_str.clone());
    extra_env.insert("CARNOT_PROJECT_DIR".into(), proj_str);
    if let Some(ref pr) = hook.plugin_root {
        extra_env.insert(
            "CLAUDE_PLUGIN_ROOT".into(),
            pr.to_string_lossy().to_string(),
        );
    }

    apply_safe_env(&mut cmd, &extra_env);
    cmd.current_dir(workspace_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let payload_bytes = serde_json::to_vec(stdin_payload).unwrap_or_else(|_| b"{}".to_vec());

    if hook.is_async {
        let timeout = Duration::from_secs(hook.timeout_secs);
        let source = hook.source.clone();
        if let Ok(mut child) = cmd.spawn() {
            tokio::spawn(async move {
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(&payload_bytes).await;
                    let _ = stdin.shutdown().await;
                }
                if tokio::time::timeout(timeout, child.wait_with_output())
                    .await
                    .is_err()
                {
                    tracing::warn!(source = %source, "async hook timed out");
                }
            });
        }
        return Ok(HookExecResult {
            exit_code: 0,
            stdout: String::new(),
            stderr: String::new(),
        });
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to spawn hook `{}`: {e}", hook.source))?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(&payload_bytes).await;
        let _ = stdin.shutdown().await;
    }

    let output = tokio::time::timeout(
        Duration::from_secs(hook.timeout_secs),
        child.wait_with_output(),
    )
    .await
    .map_err(|_| {
        format!(
            "hook `{}` timed out after {}s",
            hook.source, hook.timeout_secs
        )
    })?
    .map_err(|e| format!("hook `{}` I/O error: {e}", hook.source))?;

    let clamp = |b: Vec<u8>| -> String {
        let slice = &b[..b.len().min(MAX_HOOK_OUTPUT_BYTES)];
        String::from_utf8_lossy(slice).trim().to_string()
    };

    Ok(HookExecResult {
        exit_code: output.status.code().unwrap_or(-1),
        stdout: clamp(output.stdout),
        stderr: clamp(output.stderr),
    })
}

/// Declarative lifecycle hooks `Module` (100% Claude Code hooks compatible).
#[derive(Debug, Clone)]
pub struct HooksModule {
    workspace_root: PathBuf,
    hooks: Vec<CompiledHook>,
    fail_closed: bool,
}

impl HooksModule {
    /// Builds a `HooksModule` from pre-compiled hooks (used directly by `carnot-ext-claude-plugins`).
    pub fn from_compiled(
        workspace_root: PathBuf,
        hooks: Vec<CompiledHook>,
        fail_closed: bool,
    ) -> Self {
        Self {
            workspace_root,
            hooks,
            fail_closed,
        }
    }

    /// Discovers and compiles all configured hook files respecting workspace trust boundaries.
    pub fn discover(workspace_root: &Path, cfg: &HooksConfig) -> Result<Self> {
        let mut hooks = Vec::new();

        if cfg.load_carnot_user_hooks
            && let Some(home) = std::env::var_os("HOME")
        {
            let p = PathBuf::from(home).join(".carnot").join("hooks.json");
            if p.is_file() {
                hooks.extend(parse_hooks_file(&p, workspace_root, None)?);
            }
        }

        if cfg.load_claude_user_settings
            && let Some(home) = std::env::var_os("HOME")
        {
            let p = PathBuf::from(home).join(".claude").join("settings.json");
            if p.is_file() {
                hooks.extend(parse_hooks_file(&p, workspace_root, None)?);
            }
        }

        for explicit in &cfg.files {
            hooks.extend(parse_hooks_file(explicit, workspace_root, None)?);
        }

        if cfg.trust_workspace_hooks {
            for rel in [
                ".carnot/hooks.json",
                ".claude/settings.json",
                ".claude/settings.local.json",
            ] {
                let p = workspace_root.join(rel);
                if p.is_file() {
                    hooks.extend(parse_hooks_file(&p, workspace_root, None)?);
                }
            }
        }

        Ok(Self {
            workspace_root: workspace_root.to_path_buf(),
            hooks,
            fail_closed: cfg.fail_closed,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    pub fn hook_count(&self) -> usize {
        self.hooks.len()
    }

    pub fn extend_hooks(&mut self, extra: Vec<CompiledHook>) {
        self.hooks.extend(extra);
    }

    fn matching_hooks<'a>(
        &'a self,
        event: HookEvent,
        target: &str,
        tool_input: Option<&Value>,
    ) -> Vec<&'a CompiledHook> {
        self.hooks
            .iter()
            .filter(|h| h.event == event && h.matches(target, tool_input))
            .collect()
    }
}

impl Module for HooksModule {
    fn name(&self) -> &str {
        "hooks"
    }

    fn on_session_start(&self, config: &CarnotConfig) -> BoxFuture<'_, Result<Option<String>>> {
        let matched: Vec<CompiledHook> = self
            .matching_hooks(HookEvent::SessionStart, "startup", None)
            .into_iter()
            .cloned()
            .collect();
        if matched.is_empty() {
            return Box::pin(async { Ok(None) });
        }
        let ws = self.workspace_root.clone();
        let payload = json!({
            "hook_event_name": "SessionStart",
            "cwd": ws.display().to_string(),
            "source": "startup",
            "model": config.model,
        });
        Box::pin(async move {
            let mut contexts = Vec::new();
            for h in &matched {
                match run_command_hook(h, &ws, &payload).await {
                    Ok(res) if res.exit_code == 0 && !res.stdout.is_empty() => {
                        if let Ok(v) = serde_json::from_str::<Value>(&res.stdout) {
                            if let Some(ctx) = v
                                .pointer("/hookSpecificOutput/additionalContext")
                                .or_else(|| v.get("additionalContext"))
                                .or_else(|| v.get("systemMessage"))
                                .and_then(Value::as_str)
                                && !ctx.trim().is_empty()
                            {
                                contexts.push(ctx.to_string());
                            }
                        } else {
                            // Claude Code SessionStart injects plain stdout directly into Turn-1 context
                            contexts.push(res.stdout);
                        }
                    }
                    Ok(res) if res.exit_code != 0 => {
                        tracing::warn!(
                            source = %h.source,
                            exit_code = res.exit_code,
                            stderr = %res.stderr,
                            "SessionStart hook exited non-zero"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(source = %h.source, error = %e, "SessionStart hook failed");
                    }
                    _ => {}
                }
            }
            if contexts.is_empty() {
                Ok(None)
            } else {
                Ok(Some(contexts.join("\n\n")))
            }
        })
    }

    fn before_tool_call(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> BoxFuture<'_, Result<ToolGateDecision>> {
        let matched: Vec<CompiledHook> = self
            .matching_hooks(HookEvent::PreToolUse, tool_name, Some(input))
            .into_iter()
            .cloned()
            .collect();
        if matched.is_empty() {
            return Box::pin(async { Ok(ToolGateDecision::Allow) });
        }
        let ws = self.workspace_root.clone();
        let fail_closed = self.fail_closed;
        let tool = tool_name.to_string();
        let mut current_input = input.clone();
        Box::pin(async move {
            let mut rewritten = false;
            for h in &matched {
                let payload = json!({
                    "hook_event_name": "PreToolUse",
                    "cwd": ws.display().to_string(),
                    "tool_name": tool,
                    "tool_input": current_input,
                });
                match run_command_hook(h, &ws, &payload).await {
                    Ok(res) if res.exit_code == 2 => {
                        let reason = if res.stderr.is_empty() {
                            format!("blocked by PreToolUse hook `{}` (exit 2)", h.source)
                        } else {
                            res.stderr
                        };
                        return Ok(ToolGateDecision::Deny { reason });
                    }
                    Ok(res) if res.exit_code == 0 => {
                        if res.stdout.is_empty() {
                            continue;
                        }
                        if let Ok(v) = serde_json::from_str::<Value>(&res.stdout) {
                            if v.get("continue").and_then(Value::as_bool) == Some(false) {
                                let reason = v
                                    .get("stopReason")
                                    .and_then(Value::as_str)
                                    .unwrap_or("stopped by PreToolUse hook")
                                    .to_string();
                                return Ok(ToolGateDecision::Deny { reason });
                            }
                            let perm = v
                                .pointer("/hookSpecificOutput/permissionDecision")
                                .and_then(Value::as_str);
                            let legacy_dec = v.get("decision").and_then(Value::as_str);
                            if perm == Some("deny") || legacy_dec == Some("block") {
                                let reason = v
                                    .pointer("/hookSpecificOutput/permissionDecisionReason")
                                    .or_else(|| v.get("reason"))
                                    .and_then(Value::as_str)
                                    .unwrap_or("denied by PreToolUse hook")
                                    .to_string();
                                return Ok(ToolGateDecision::Deny { reason });
                            }
                            if let Some(updated) = v
                                .pointer("/hookSpecificOutput/updatedInput")
                                .filter(|u| u.is_object())
                            {
                                current_input = updated.clone();
                                rewritten = true;
                            }
                        }
                    }
                    Ok(res) => {
                        tracing::warn!(
                            source = %h.source,
                            exit_code = res.exit_code,
                            stderr = %res.stderr,
                            "PreToolUse hook failed"
                        );
                        if fail_closed {
                            return Ok(ToolGateDecision::Deny {
                                reason: format!(
                                    "PreToolUse hook `{}` fail-closed (exit {}): {}",
                                    h.source, res.exit_code, res.stderr
                                ),
                            });
                        }
                    }
                    Err(e) => {
                        tracing::warn!(source = %h.source, error = %e, "PreToolUse hook error");
                        if fail_closed {
                            return Ok(ToolGateDecision::Deny {
                                reason: format!("PreToolUse hook fail-closed: {e}"),
                            });
                        }
                    }
                }
            }
            if rewritten {
                Ok(ToolGateDecision::RewriteInput(current_input))
            } else {
                Ok(ToolGateDecision::Allow)
            }
        })
    }

    fn after_tool_result<'a>(
        &'a self,
        tool_name: &'a str,
        input: &'a Value,
        output: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>> {
        let event = if output.is_error {
            HookEvent::PostToolUseFailure
        } else {
            HookEvent::PostToolUse
        };
        let matched: Vec<CompiledHook> = self
            .matching_hooks(event, tool_name, Some(input))
            .into_iter()
            .cloned()
            .collect();
        if matched.is_empty() {
            return Box::pin(async { Ok(()) });
        }
        let ws = self.workspace_root.clone();
        let payload = json!({
            "hook_event_name": event.as_str(),
            "cwd": ws.display().to_string(),
            "tool_name": tool_name,
            "tool_input": input,
            "tool_response": {
                "text": output.text_only(),
                "is_error": output.is_error,
            },
        });
        Box::pin(async move {
            for h in &matched {
                match run_command_hook(h, &ws, &payload).await {
                    Ok(res) if res.exit_code == 2 => {
                        let msg = if res.stderr.is_empty() {
                            format!("[hook feedback ({})]", h.source)
                        } else {
                            format!("[hook feedback]\n{}", res.stderr)
                        };
                        output.blocks.push(ToolResultBlock::text(msg));
                    }
                    Ok(res) if res.exit_code == 0 && !res.stdout.is_empty() => {
                        if let Ok(v) = serde_json::from_str::<Value>(&res.stdout) {
                            if let Some(extra) = v
                                .pointer("/hookSpecificOutput/additionalContext")
                                .or_else(|| v.get("additionalContext"))
                                .or_else(|| v.get("systemMessage"))
                                .or_else(|| v.get("reason"))
                                .and_then(Value::as_str)
                                && !extra.trim().is_empty()
                            {
                                output.blocks.push(ToolResultBlock::text(extra));
                            }
                        } else {
                            output.blocks.push(ToolResultBlock::text(res.stdout));
                        }
                    }
                    Ok(res) if res.exit_code != 0 => {
                        tracing::warn!(
                            source = %h.source,
                            exit_code = res.exit_code,
                            stderr = %res.stderr,
                            "PostToolUse hook exited non-zero"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(source = %h.source, error = %e, "PostToolUse hook failed");
                    }
                    _ => {}
                }
            }
            Ok(())
        })
    }

    fn on_turn_boundary<'a>(
        &'a self,
        snapshot: &'a TurnBoundarySnapshot,
    ) -> BoxFuture<'a, Result<BoundaryAction>> {
        // Claude Code `Stop` hooks fire only when the assistant is about to finish (no tool calls).
        if snapshot.had_tool_call_in_turn {
            return Box::pin(async { Ok(BoundaryAction::Continue) });
        }
        let matched: Vec<CompiledHook> = self
            .matching_hooks(HookEvent::Stop, "", None)
            .into_iter()
            .cloned()
            .collect();
        if matched.is_empty() {
            return Box::pin(async { Ok(BoundaryAction::Continue) });
        }
        let ws = self.workspace_root.clone();
        let payload = json!({
            "hook_event_name": "Stop",
            "cwd": ws.display().to_string(),
            "stop_hook_active": true,
            "last_assistant_message": snapshot.last_assistant_text,
            "turn_index": snapshot.turn_index,
        });
        Box::pin(async move {
            for h in &matched {
                match run_command_hook(h, &ws, &payload).await {
                    Ok(res) if res.exit_code == 2 => {
                        let msg = if res.stderr.is_empty() {
                            "Stop hook blocked completion (exit 2).".to_string()
                        } else {
                            res.stderr
                        };
                        return Ok(BoundaryAction::AppendAdvisory {
                            signal: "stop_hook".into(),
                            message: msg,
                        });
                    }
                    Ok(res) if res.exit_code == 0 && !res.stdout.is_empty() => {
                        if let Ok(v) = serde_json::from_str::<Value>(&res.stdout) {
                            if v.get("continue").and_then(Value::as_bool) == Some(false) {
                                return Ok(BoundaryAction::Halt {
                                    reason: v
                                        .get("stopReason")
                                        .and_then(Value::as_str)
                                        .unwrap_or("halted by Stop hook")
                                        .to_string(),
                                });
                            }
                            if v.get("decision").and_then(Value::as_str) == Some("block") {
                                return Ok(BoundaryAction::AppendAdvisory {
                                    signal: "stop_hook".into(),
                                    message: v
                                        .get("reason")
                                        .and_then(Value::as_str)
                                        .unwrap_or("Stop hook requested continuation")
                                        .to_string(),
                                });
                            }
                        }
                    }
                    Ok(res) if res.exit_code != 0 => {
                        tracing::warn!(source = %h.source, exit_code = res.exit_code, "Stop hook failed");
                    }
                    Err(e) => {
                        tracing::warn!(source = %h.source, error = %e, "Stop hook error");
                    }
                    _ => {}
                }
            }
            Ok(BoundaryAction::Continue)
        })
    }

    fn before_compact<'a>(
        &'a self,
        ctx: &'a CompactionHookContext,
    ) -> BoxFuture<'a, Result<CompactionHookAction>> {
        let trigger = if ctx.mode.eq_ignore_ascii_case("manual") {
            "manual"
        } else {
            "auto"
        };
        let matched: Vec<CompiledHook> = self
            .matching_hooks(HookEvent::PreCompact, trigger, None)
            .into_iter()
            .cloned()
            .collect();
        if matched.is_empty() {
            return Box::pin(async { Ok(CompactionHookAction::Continue) });
        }
        let ws = self.workspace_root.clone();
        let payload = json!({
            "hook_event_name": "PreCompact",
            "cwd": ws.display().to_string(),
            "trigger": trigger,
            "mode": ctx.mode,
            "estimated_tokens": ctx.tokens_before,
            "message_count": ctx.message_count,
            "custom_focus": ctx.custom_focus,
        });
        Box::pin(async move {
            let mut extra_focus = Vec::new();
            for h in &matched {
                match run_command_hook(h, &ws, &payload).await {
                    Ok(res) if res.exit_code == 2 => {
                        return Ok(CompactionHookAction::Cancel {
                            reason: if res.stderr.is_empty() {
                                "cancelled by PreCompact hook (exit 2)".into()
                            } else {
                                res.stderr
                            },
                        });
                    }
                    Ok(res) if res.exit_code == 0 && !res.stdout.is_empty() => {
                        if let Ok(v) = serde_json::from_str::<Value>(&res.stdout) {
                            if v.get("decision").and_then(Value::as_str) == Some("block") {
                                return Ok(CompactionHookAction::Cancel {
                                    reason: v
                                        .get("reason")
                                        .and_then(Value::as_str)
                                        .unwrap_or("cancelled by PreCompact hook")
                                        .to_string(),
                                });
                            }
                            if let Some(f) = v
                                .pointer("/hookSpecificOutput/additionalContext")
                                .or_else(|| v.get("focus"))
                                .or_else(|| v.get("additionalContext"))
                                .and_then(Value::as_str)
                                && !f.trim().is_empty()
                            {
                                extra_focus.push(f.to_string());
                            }
                        } else {
                            extra_focus.push(res.stdout);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(source = %h.source, error = %e, "PreCompact hook error");
                    }
                    _ => {}
                }
            }
            if extra_focus.is_empty() {
                Ok(CompactionHookAction::Continue)
            } else {
                Ok(CompactionHookAction::AddFocus(extra_focus.join("\n")))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn claude_code_hooks_support_deny_rewrite_post_context_and_stop_block() {
        let dir = std::env::temp_dir().join(format!("carnot-hooks-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let hooks_json = json!({
            "hooks": {
                "SessionStart": [{
                    "matcher": "startup",
                    "hooks": [{
                        "type": "command",
                        "command": "printf 'session-ready-from-cc-hook'"
                    }]
                }],
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [{
                            "type": "command",
                            "command": "python3 -c 'import json,sys; d=json.load(sys.stdin); cmd=d[\"tool_input\"].get(\"command\",\"\"); (sys.stderr.write(\"blocked rm\"), sys.exit(2)) if \"rm -rf\" in cmd else print(json.dumps({\"hookSpecificOutput\":{\"permissionDecision\":\"allow\",\"updatedInput\":{\"command\":cmd+\" #checked\"}}}))'"
                        }]
                    }
                ],
                "PostToolUse": [
                    {
                        "matcher": "Edit|Write",
                        "file_glob": "*.rs",
                        "hooks": [{
                            "type": "command",
                            "command": "printf '{\"hookSpecificOutput\":{\"additionalContext\":\"rustfmt-ok\"}}'"
                        }]
                    }
                ],
                "Stop": [{
                    "hooks": [{
                        "type": "command",
                        "command": "printf '{\"decision\":\"block\",\"reason\":\"run tests first\"}'"
                    }]
                }]
            }
        });

        let compiled = parse_hooks_value(&hooks_json, &dir, None, "test.json").expect("parse");
        let module = HooksModule::from_compiled(dir.clone(), compiled, false);
        let cfg = CarnotConfig::new(dir.clone());

        // 1. SessionStart plain stdout injection
        let start = module.on_session_start(&cfg).await.expect("start");
        assert_eq!(start.as_deref(), Some("session-ready-from-cc-hook"));

        // 2. PreToolUse exit 2 -> Deny with stderr
        let deny = module
            .before_tool_call("Bash", &json!({"command": "rm -rf /tmp/x"}))
            .await
            .expect("deny");
        assert!(matches!(deny, ToolGateDecision::Deny { reason } if reason.contains("blocked rm")));

        // 3. PreToolUse updatedInput -> RewriteInput
        let rw = module
            .before_tool_call("Bash", &json!({"command": "echo hi"}))
            .await
            .expect("rewrite");
        assert!(
            matches!(rw, ToolGateDecision::RewriteInput(v) if v == json!({"command": "echo hi #checked"}))
        );

        // 4. PostToolUse file_glob matches `.rs`, skips `.txt`
        let mut out_rs = ToolOutput::ok_text("edited");
        module
            .after_tool_result("Edit", &json!({"file_path": "src/main.rs"}), &mut out_rs)
            .await
            .expect("post rs");
        assert!(out_rs.text_only().contains("rustfmt-ok"));

        let mut out_txt = ToolOutput::ok_text("edited");
        module
            .after_tool_result("Edit", &json!({"file_path": "notes.txt"}), &mut out_txt)
            .await
            .expect("post txt");
        assert!(!out_txt.text_only().contains("rustfmt-ok"));

        // 5. Stop hook decision=block -> AppendAdvisory
        let snap = TurnBoundarySnapshot {
            turn_index: 1,
            stop_reason: carnot_core::StopReason::EndTurn,
            total_context_tokens: 100,
            had_edit_in_segment: true,
            had_tool_call_in_turn: false,
            files_read_in_segment: vec![],
            files_edited_in_segment: vec![],
            handles_created_in_segment: vec![],
            last_assistant_text: "done".into(),
            failed_attempts_summary: vec![],
        };
        let act = module.on_turn_boundary(&snap).await.expect("stop");
        assert!(
            matches!(act, BoundaryAction::AppendAdvisory { message, .. } if message.contains("run tests first"))
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
