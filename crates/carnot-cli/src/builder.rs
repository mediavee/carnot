//! Agent builder wiring, one-shot prompt/image resolution, and workspace session discovery.

use crate::args::{Cli, Commands, Mode};
use base64::Engine as _;
use carnot::{
    CacheTtl, CarnotBuilder, ContentBlock, McpManager, McpServerConfig, ProviderRegistry,
    read_session_header,
};
use std::collections::BTreeMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "ext-codemode")]
fn has_configured_mcp_servers(cwd: &Path, trust_workspace_mcp: bool) -> bool {
    let mut candidates = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(PathBuf::from(home).join(".carnot").join("mcp.json"));
    }
    if trust_workspace_mcp {
        candidates.push(cwd.join(".carnot").join("mcp.json"));
        candidates.push(cwd.join(".mcp.json"));
    }
    candidates.into_iter().any(|p| {
        std::fs::read_to_string(&p)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| {
                v.get("mcpServers")
                    .or_else(|| v.get("servers"))
                    .and_then(serde_json::Value::as_object)
                    .map(|m| !m.is_empty())
            })
            .unwrap_or(false)
    })
}

pub(crate) async fn build_agent(
    cli: &Cli,
    cwd: PathBuf,
) -> carnot::Result<(Arc<carnot::Agent>, Arc<McpManager>)> {
    build_agent_full(cli, cwd, BTreeMap::new(), Vec::new(), None).await
}

pub(crate) async fn build_agent_with_mcp(
    cli: &Cli,
    cwd: PathBuf,
    extra_mcp: BTreeMap<String, McpServerConfig>,
) -> carnot::Result<(Arc<carnot::Agent>, Arc<McpManager>)> {
    build_agent_full(cli, cwd, extra_mcp, Vec::new(), None).await
}

pub(crate) async fn build_agent_full(
    cli: &Cli,
    cwd: PathBuf,
    extra_mcp: BTreeMap<String, McpServerConfig>,
    extra_modules: Vec<Arc<dyn carnot::Module>>,
    system_prompt_append: Option<String>,
) -> carnot::Result<(Arc<carnot::Agent>, Arc<McpManager>)> {
    let raw_ttl = cli.effective_cache_ttl();
    let ttl = CacheTtl::parse(raw_ttl).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid --cache-ttl `{raw_ttl}` (expected short|hybrid|long, or 5m|1h|24h)"),
        )
    })?;
    let reg_opt = ProviderRegistry::load_default(&cli.provider_configs).ok();
    let default_provider = cli.resolved_provider(reg_opt.as_ref());
    let (effective_provider, effective_model) = if let Some(ref m) = cli.model {
        if let Some(ref reg) = reg_opt {
            if let Some((prov, mid)) = reg.resolve_model_spec(m, cli.provider.as_deref()) {
                (prov, Some(mid))
            } else {
                (default_provider, Some(m.clone()))
            }
        } else {
            (default_provider, Some(m.clone()))
        }
    } else {
        (default_provider, None)
    };

    let sys_override = if let Some(ref spf) = cli.system_prompt_file {
        Some(std::fs::read_to_string(spf)?)
    } else {
        cli.system_prompt.clone()
    };

    let interactive_surface = matches!(cli.mode, Mode::Acp | Mode::Rpc)
        || matches!(cli.command, Some(Commands::Serve { .. }))
        || (cli.prompt.is_none()
            && cli.prompt_file.is_none()
            && cli.images.is_empty()
            && cli.command.is_none()
            && std::io::stdin().is_terminal());
    let mut b = CarnotBuilder::new(cwd.clone())
        .profile(cli.effective_profile().into())
        .model_opt(effective_model)
        .effort(cli.effective_effort().into())
        .provider_name(effective_provider)
        .cache_ttl(ttl)
        .persist_session(!cli.no_session)
        .system_prompt_override(sys_override)
        .system_prompt_append(system_prompt_append)
        .exclude_tools(cli.exclude_tools.clone())
        .load_context_files(!cli.no_context_files)
        .load_skills(!cli.no_skills)
        .disable_mcp(cli.no_mcp)
        .autonomous_gates(cli.gates.clone())
        .policy_guard(cli.policy_guard)
        .ask_user(interactive_surface)
        .trust_workspace_mcp(cli.trust_workspace_mcp)
        .trust_workspace_extensions(cli.trust_workspace_extensions);
    if let Some(ref sd) = cli.session_dir {
        b = b.session_dir(sd.clone());
    }
    for m in extra_modules {
        b = b.with_module(m);
    }

    #[cfg(feature = "ext-subagent")]
    {
        b = b.subagent(cli.subagent_enabled());
    }
    #[cfg(feature = "ext-web")]
    if cli.web_enabled() {
        b = b.web_config(carnot::WebConfig {
            mode: cli.web_mode,
            allowed_domains: cli.web_allow_domains.clone(),
            blocked_domains: cli.web_block_domains.clone(),
            ..carnot::WebConfig::default()
        });
    }
    #[cfg(feature = "ext-lsp")]
    {
        b = b.lsp_diagnostics(cli.lsp_enabled());
    }
    #[cfg(feature = "ext-hooks")]
    {
        let user_hooks_exists = std::env::var_os("HOME").is_some_and(|h| {
            PathBuf::from(h)
                .join(".carnot")
                .join("hooks.json")
                .is_file()
        });
        if !cli.hooks_files.is_empty()
            || cli.claude_hooks
            || cli.trust_workspace_hooks
            || user_hooks_exists
        {
            b = b.hooks_config(carnot::HooksConfig {
                files: cli.hooks_files.clone(),
                load_carnot_user_hooks: true,
                load_claude_user_settings: cli.claude_hooks,
                trust_workspace_hooks: cli.trust_workspace_hooks,
                fail_closed: false,
            });
        }
    }
    #[cfg(feature = "ext-claude-plugins")]
    if !cli.claude_plugins.is_empty() || cli.load_installed_claude_plugins {
        b = b.claude_plugins_config(carnot::ClaudePluginsConfig {
            plugin_dirs: cli.claude_plugins.clone(),
            load_installed_plugins: cli.load_installed_claude_plugins,
            fail_closed_hooks: false,
        });
    }
    #[cfg(feature = "ext-memory")]
    if cli.memory_enabled() {
        let mut mc = carnot::MemoryConfig::default();
        if let Some(ref dir) = cli.memory_dir {
            mc.root_dir.clone_from(dir);
        }
        b = b.memory_config(mc);
    }
    #[cfg(feature = "ext-codemode")]
    if cli.codemode_enabled() {
        let mut cm_cfg = carnot::CodeModeConfig::default();
        let bash_excluded = cli
            .exclude_tools
            .iter()
            .any(|t| t.eq_ignore_ascii_case("bash"));
        let has_mcp = !extra_mcp.is_empty()
            || (!cli.no_mcp && has_configured_mcp_servers(&cwd, cli.trust_workspace_mcp));
        // Defer `CodeMode` behind `ToolSearch` in default standalone Bash-only sessions so plain
        // shell/SWE sessions pay 0 eager prefix tokens; load eagerly when `--codemode`, MCP
        // servers, or `--exclude-tool Bash` are active.
        cm_cfg.defer_loading = !cli.codemode && !bash_excluded && !has_mcp;
        b = b.codemode_config(cm_cfg);
    }
    #[cfg(feature = "ext-goal")]
    {
        let mut gc = carnot::GoalConfig::default();
        gc.criterion.clone_from(&cli.goal);
        gc.verify_commands.clone_from(&cli.goal_cmds);
        if let Some(max_iter) = cli.goal_max_iter {
            gc.max_iterations = max_iter.max(1);
        }
        gc.verifier_model.clone_from(&cli.goal_model);
        // Always attach a lightweight GoalHandle (0 static prefix bytes; dormant until configured)
        // so interactive TUI `/goal`, RPC `set_goal`, and HTTP `POST /v1/goal` work live.
        b = b.goal_handle(carnot::GoalHandle::new(gc));
    }
    #[cfg(feature = "ext-adaptive")]
    if cli.adaptive {
        b = b.adaptive(true);
    }
    for ext in &cli.extensions {
        b = b.extension(ext.clone());
    }

    if let Some(mt) = cli.max_turns {
        b = b.max_turns(mt);
    }
    if let Some(mc) = cli.max_cost_usd {
        b = b.max_cost_usd(mc);
    }
    if let Some(ca) = cli.compact_at {
        b = b.soft_compaction_tokens(if ca == 0 { None } else { Some(ca) });
    }

    for p in &cli.provider_configs {
        b = b.provider_config(p.clone());
    }
    for p in &cli.mcp_configs {
        b = b.mcp_config(p.clone());
    }
    if !extra_mcp.is_empty() {
        b = b.extra_mcp_servers(extra_mcp);
    }
    for d in &cli.skills_dirs {
        b = b.skills_dir(d.clone());
    }
    let default_sdir = || {
        cli.session_dir.clone().unwrap_or_else(|| {
            std::env::var_os("HOME").map_or_else(
                || cwd.join(".carnot").join("sessions"),
                |h| PathBuf::from(h).join(".carnot").join("sessions"),
            )
        })
    };
    if let Some(ref res) = cli.resume {
        let p = PathBuf::from(res);
        let resolved = if p.is_file() {
            p
        } else {
            let sdir = default_sdir();
            if std::path::Path::new(res)
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("jsonl"))
            {
                sdir.join(res)
            } else {
                sdir.join(format!("{res}.jsonl"))
            }
        };
        b = b.resume_session(resolved).fork_session(cli.fork);
    } else if cli.continue_last {
        let sdir = default_sdir();
        let latest = find_latest_session_for_workspace(&sdir, &cwd).ok_or_else(|| {
            carnot::CarnotError::Session(format!(
                "no previous session found for workspace `{}` in `{}`",
                cwd.display(),
                sdir.display()
            ))
        })?;
        b = b.resume_session(latest).fork_session(cli.fork);
    }
    if cli.no_sandbox {
        b = b
            .sandbox(false)
            .extra_writable_roots(vec![PathBuf::from("/")]);
    }
    if cli.allow_network {
        b = b.sandbox_network(true);
    }
    b.build_async().await
}


/// Resolves a one-shot prompt from `--prompt-file`, `-p`, and/or piped `stdin`.
///
/// Supports combining piped stdin with an explicit `-p` instruction (e.g. `git diff | carnot -p "review"`),
/// standalone piped stdin (`cat err.log | carnot`), explicit `-p -`, and `--image` without text.
pub(crate) fn resolve_one_shot_prompt(
    cli: &Cli,
    stdin_is_tty: bool,
    read_stdin: impl FnOnce() -> std::io::Result<String>,
) -> Result<Option<String>, String> {
    if let Some(ref pf) = cli.prompt_file {
        let content = std::fs::read_to_string(pf)
            .map_err(|e| format!("reading --prompt-file {}: {e}", pf.display()))?;
        return Ok(Some(content));
    }
    if cli.prompt.as_deref() == Some("-") {
        let buf = read_stdin().map_err(|e| format!("reading prompt from stdin: {e}"))?;
        return Ok(Some(buf));
    }
    let piped = if stdin_is_tty {
        None
    } else {
        let buf = read_stdin().map_err(|e| format!("reading prompt from stdin: {e}"))?;
        if buf.trim().is_empty() {
            None
        } else {
            Some(buf)
        }
    };

    match (cli.prompt.as_deref(), piped) {
        (Some(p), Some(s)) => Ok(Some(format!("{p}\n\n{s}"))),
        (Some(p), None) => Ok(Some(p.to_string())),
        (None, Some(s)) => Ok(Some(s)),
        (None, None) if !cli.images.is_empty() => Ok(Some("Analyze the attached image(s).".into())),
        (None, None) => Ok(None),
    }
}

const MAX_CLI_IMAGE_BYTES: usize = 20 * 1024 * 1024;

pub(crate) fn detect_image_mime(path: &std::path::Path, bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("image/png");
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some("image/jpeg");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => Some("image/png"),
        Some("jpg" | "jpeg") => Some("image/jpeg"),
        Some("gif") => Some("image/gif"),
        Some("webp") => Some("image/webp"),
        _ => None,
    }
}

pub(crate) fn load_image_blocks(paths: &[PathBuf]) -> Result<Vec<ContentBlock>, String> {
    let mut out = Vec::with_capacity(paths.len());
    for p in paths {
        let meta = std::fs::metadata(p)
            .map_err(|e| format!("cannot stat image `{}`: {e}", p.display()))?;
        if meta.len() > MAX_CLI_IMAGE_BYTES as u64 {
            return Err(format!(
                "image `{}` exceeds 20 MiB limit ({} bytes)",
                p.display(),
                meta.len()
            ));
        }
        let bytes =
            std::fs::read(p).map_err(|e| format!("cannot read image `{}`: {e}", p.display()))?;
        let media_type = detect_image_mime(p, &bytes).ok_or_else(|| {
            format!(
                "unsupported image format `{}` (expected png, jpeg, webp, or gif)",
                p.display()
            )
        })?;
        let data_base64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        out.push(ContentBlock::Image {
            media_type: media_type.to_string(),
            data_base64,
        });
    }
    Ok(out)
}

/// Finds the most recently modified session JSONL in `session_dir` belonging to `cwd`.
pub(crate) fn find_latest_session_for_workspace(
    session_dir: &std::path::Path,
    cwd: &std::path::Path,
) -> Option<PathBuf> {
    let target_ws = std::fs::canonicalize(cwd).unwrap_or_else(|_| cwd.to_path_buf());
    let entries = std::fs::read_dir(session_dir).ok()?;
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") || !path.is_file() {
            continue;
        }
        let Some(hdr) = read_session_header(&path) else {
            continue;
        };
        let Some(ws) = hdr.workspace_root else {
            continue;
        };
        let ws_path = PathBuf::from(&ws);
        let ws_canon = std::fs::canonicalize(&ws_path).unwrap_or(ws_path);
        if ws_canon != target_ws {
            continue;
        }
        let mtime = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(std::time::UNIX_EPOCH);
        let is_newer = match &best {
            None => true,
            Some((best_t, best_p)) => mtime > *best_t || (mtime == *best_t && path > *best_p),
        };
        if is_newer {
            best = Some((mtime, path));
        }
    }
    best.map(|(_, p)| p)
}

