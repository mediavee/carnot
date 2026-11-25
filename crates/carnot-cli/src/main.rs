//! `carnot` CLI binary: headless print/json, ACP stdio, Host RPC stdio, HTTP/SSE API server,
//! MCP 2026-07-28 / 2025-11-25 / Legacy SSE 2024-11-05 management (`carnot mcp list|inspect`),
//! prefix-invariance CI audit, production diagnostics (`carnot doctor`), session resume,
//! budget/timeout guardrails, and graceful signal handling.

mod config;
mod repl;

use base64::Engine as _;
use carnot::{
    AcpHost, AcpModelInfo, AgentEvent, CacheTtl, CarnotBuilder, CasStore, ContentBlock,
    HostRpcBridge, HostRpcModule, McpManager, McpServerConfig, Profile, ProviderRegistry,
    ReasoningEffort, ResolvedAcpModel, ResolvedRpcModel, RpcCommandInfo, RpcHost, RpcModelInfo,
    assert_prefix_invariance, detect_sandbox_backend, discover_dropin_specs, load_mcp_configs,
    parse_acp_mcp_servers, read_host_registration, read_session_header, serve_acp_stdio_with_host,
    serve_http, serve_rpc_stdio_with_extension, serve_rpc_stdio_with_host,
    try_handle_landlock_exec_arg,
};
use carnot_ext::microskills::discover_agent_skills;
use carnot_ext::prefix_guard::static_system_prompt;
use carnot_tools::canonical_tool_specs;
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::json;
use std::collections::BTreeMap;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Mode {
    Text,
    Json,
    Rpc,
    Acp,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliProfile {
    Minimal,
    Standard,
    Carnot,
}

impl From<CliProfile> for Profile {
    fn from(p: CliProfile) -> Self {
        match p {
            CliProfile::Minimal => Profile::Minimal,
            CliProfile::Standard => Profile::Standard,
            CliProfile::Carnot => Profile::Carnot,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl From<CliEffort> for ReasoningEffort {
    fn from(e: CliEffort) -> Self {
        match e {
            CliEffort::Low => ReasoningEffort::Low,
            CliEffort::Medium => ReasoningEffort::Medium,
            CliEffort::High => ReasoningEffort::High,
            CliEffort::Xhigh => ReasoningEffort::Xhigh,
            CliEffort::Max => ReasoningEffort::Max,
        }
    }
}

#[derive(Debug, Clone, Parser)]
#[command(
    name = "carnot",
    version,
    about = "Carnot: Fast, safe coding-agent harness in Rust"
)]
struct Cli {
    /// One-shot prompt text
    #[arg(short = 'p', long)]
    prompt: Option<String>,

    /// Read prompt from a file path (avoids exposing prompt text in /proc/<pid>/cmdline or `ARG_MAX` limits)
    #[arg(long = "prompt-file")]
    prompt_file: Option<PathBuf>,

    /// Attach image file(s) (`png`, `jpg`, `webp`, `gif`) to a one-shot prompt (repeatable)
    #[arg(short = 'i', long = "image")]
    images: Vec<PathBuf>,

    /// Execution surface mode
    #[arg(long, value_enum, default_value_t = Mode::Text)]
    mode: Mode,

    /// Use minimalist line-oriented plain REPL instead of the full-screen TUI
    #[arg(long)]
    plain: bool,

    /// Module profile preset (defaults to `carnot`, or `~/.carnot/config.toml`)
    #[arg(long, value_enum)]
    profile: Option<CliProfile>,

    /// Provider name or alias from providers.json (auto-detects first configured provider if omitted)
    #[arg(long)]
    provider: Option<String>,

    /// Model identifier (defaults to the selected provider's catalog `default_model`)
    #[arg(short = 'm', long)]
    model: Option<String>,

    /// Explicit `providers.json` override/plugin catalog path(s) (repeatable)
    #[arg(long = "provider-config")]
    provider_configs: Vec<PathBuf>,

    /// Reasoning effort level (low, medium, high, xhigh, max; defaults to `high` or `~/.carnot/config.toml`)
    #[arg(long, value_enum)]
    effort: Option<CliEffort>,

    /// Prompt cache retention tier (short, hybrid, or long; aliases: 5m, 1h, 24h)
    #[arg(long, alias = "cache-retention")]
    cache_ttl: Option<String>,

    /// Workspace directory (defaults to current directory)
    #[arg(short = 'C', long)]
    cwd: Option<PathBuf>,

    /// Resume the most recent session for this workspace
    #[arg(short = 'c', long = "continue", conflicts_with = "resume")]
    continue_last: bool,

    /// Resume an existing session JSONL file (or session ID under ~/.carnot/sessions)
    #[arg(long)]
    resume: Option<String>,

    /// With `--resume` or `--continue`: fork into a brand-new session file without modifying the original
    #[arg(long)]
    fork: bool,

    /// Disable loading global or workspace `.mcp.json` servers
    #[arg(long)]
    no_mcp: bool,

    /// Explicit `.mcp.json` config path(s) (repeatable)
    #[arg(long = "mcp-config")]
    mcp_configs: Vec<PathBuf>,

    /// Explicit Agent Skills directory path(s) containing `<name>/SKILL.md` (repeatable)
    #[arg(long = "skills-dir")]
    skills_dirs: Vec<PathBuf>,

    /// Explicitly trust workspace-local `.mcp.json` (disabled by default to prevent `MCPoison` RCE)
    #[arg(long)]
    trust_workspace_mcp: bool,

    /// Drop-in file extension path(s) (executable script/binary speaking `carnot-ext/1` JSON-RPC; repeatable)
    #[arg(long = "ext")]
    extensions: Vec<PathBuf>,

    /// Explicitly trust workspace-local `.carnot/extensions/` (disabled by default to prevent RCE)
    #[arg(long = "trust-workspace-ext")]
    trust_workspace_extensions: bool,

    /// Enable compiled-in Rust `carnot-ext-web` extension (enabled by default in standalone `carnot` profile)
    #[cfg(feature = "ext-web")]
    #[arg(long)]
    web: bool,

    /// Disable compiled-in `carnot-ext-web` extension
    #[cfg(feature = "ext-web")]
    #[arg(long = "no-web")]
    no_web: bool,

    /// Web extension mode: `auto` (prefer native provider search/fetch, fallback local), `native`, or `local`
    #[cfg(feature = "ext-web")]
    #[arg(long = "web-mode", default_value = "auto")]
    web_mode: carnot::WebMode,

    /// Restrict web search/fetch to specific hostname(s) (repeatable, e.g. `docs.rs`)
    #[cfg(feature = "ext-web")]
    #[arg(long = "web-allow-domain")]
    web_allow_domains: Vec<String>,

    /// Block web search/fetch from specific hostname(s) (repeatable)
    #[cfg(feature = "ext-web")]
    #[arg(long = "web-block-domain")]
    web_block_domains: Vec<String>,

    /// Enable compiled-in Rust `carnot-ext-lsp` extension (enabled by default in standalone `carnot` profile)
    #[cfg(feature = "ext-lsp")]
    #[arg(long)]
    lsp: bool,

    /// Disable compiled-in `carnot-ext-lsp` extension
    #[cfg(feature = "ext-lsp")]
    #[arg(long = "no-lsp")]
    no_lsp: bool,

    /// Enable disposable-context `Subagent` delegation extension (enabled by default in standalone `carnot` profile)
    #[cfg(feature = "ext-subagent")]
    #[arg(long)]
    subagent: bool,

    /// Disable compiled-in `Subagent` delegation extension
    #[cfg(feature = "ext-subagent")]
    #[arg(long = "no-subagent")]
    no_subagent: bool,

    /// Declarative Claude Code / Carnot hooks JSON file(s) (repeatable, requires `--features ext-hooks`)
    #[cfg(feature = "ext-hooks")]
    #[arg(long = "hooks-file")]
    hooks_files: Vec<PathBuf>,

    /// Also load user-global Claude Code hooks from `~/.claude/settings.json`
    #[cfg(feature = "ext-hooks")]
    #[arg(long = "claude-hooks")]
    claude_hooks: bool,

    /// Explicitly trust workspace-local `.carnot/hooks.json` and `.claude/settings*.json` (disabled by default to prevent RCE)
    #[cfg(feature = "ext-hooks")]
    #[arg(long = "trust-workspace-hooks")]
    trust_workspace_hooks: bool,

    /// Claude Code plugin directory path(s) (equivalent to `--plugin-dir`, repeatable, requires `--features ext-claude-plugins`)
    #[cfg(feature = "ext-claude-plugins")]
    #[arg(long = "claude-plugin")]
    claude_plugins: Vec<PathBuf>,

    /// Auto-discover installed enabled Claude Code plugins from `~/.claude/plugins/installed_plugins.json`
    #[cfg(feature = "ext-claude-plugins")]
    #[arg(long = "load-installed-claude-plugins")]
    load_installed_claude_plugins: bool,

    /// Enable cross-session persistent memory extension (enabled by default in standalone `carnot` profile)
    #[cfg(feature = "ext-memory")]
    #[arg(long)]
    memory: bool,

    /// Disable cross-session persistent memory extension
    #[cfg(feature = "ext-memory")]
    #[arg(long = "no-memory")]
    no_memory: bool,

    /// Custom root directory for persistent memory store (defaults to `~/.carnot/memory`)
    #[cfg(feature = "ext-memory")]
    #[arg(long = "memory-dir")]
    memory_dir: Option<PathBuf>,

    /// Enable enterprise `PolicyGuard` (command/path guardrails + secret redaction)
    #[arg(long)]
    policy_guard: bool,

    /// Explicitly disable OS kernel sandbox AND workspace filesystem confinement (fail-closed by default)
    #[arg(long)]
    no_sandbox: bool,

    /// Allow outbound network inside the OS sandbox (disabled by default)
    #[arg(long)]
    allow_network: bool,

    /// Do not persist session JSONL to disk
    #[arg(long)]
    no_session: bool,

    /// Explicit directory for session JSONL files (defaults to ~/.carnot/sessions)
    #[arg(long = "session-dir")]
    session_dir: Option<PathBuf>,

    /// Enable bidirectional host-extension handshake (`host/register`) on `--mode rpc`
    #[arg(long = "host-extension")]
    host_extension: bool,

    /// Override the static system prompt (frozen before Turn 1 for prompt-cache invariance)
    #[arg(long = "system-prompt")]
    system_prompt: Option<String>,

    /// Read static system prompt override from a file path (avoids `ARG_MAX` limits)
    #[arg(long = "system-prompt-file")]
    system_prompt_file: Option<PathBuf>,

    /// Exclude built-in tools by name before Turn 1 (comma-separated or repeatable, e.g. `Bash,Edit,Write`)
    #[arg(long = "exclude-tools", value_delimiter = ',')]
    exclude_tools: Vec<String>,

    /// Disable automatic Turn-1 `AGENTS.md`/`CLAUDE.md` discovery
    #[arg(long = "no-context-files")]
    no_context_files: bool,

    /// Disable automatic Turn-1 `SKILL.md` discovery
    #[arg(long = "no-skills")]
    no_skills: bool,

    /// Maximum agent turns before halting (default: 500)
    #[arg(long = "max-turns")]
    max_turns: Option<u64>,

    /// Hard USD cost cap circuit-breaker for this session (e.g. --max-cost-usd 2.50)
    #[arg(long = "max-cost-usd")]
    max_cost_usd: Option<f64>,

    /// Soft compaction threshold in estimated tokens (default: 220000; 0 disables soft compaction)
    #[arg(long = "compact-at")]
    compact_at: Option<u64>,

    /// Overall wall-clock timeout in seconds for one-shot runs (cancels cleanly on expiry)
    #[arg(long = "timeout")]
    timeout_secs: Option<u64>,

    /// Autonomous verification gate command(s) (repeatable)
    #[arg(long = "gate")]
    gates: Vec<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Clone, Subcommand)]
enum Commands {
    /// Start embedded HTTP/1.1 + SSE API server (Bearer auth + CSRF protection)
    Serve {
        #[arg(long, default_value = "127.0.0.1:7744")]
        addr: String,
    },
    /// Verify byte-identical static prefix invariance across dates, worktrees, and sessions (CI gate)
    AuditPrefix,
    /// Production environment & readiness healthcheck (auth status without secrets, sandbox, MCP, skills)
    Doctor,
    /// List active modules and canonical core tool schemas
    Modules,
    /// List registered wire-protocol adapters & data-driven providers from providers.json
    Providers,
    /// List available catalog models or resolve a model reference / family alias (e.g. `anthropic/sonnet`)
    Models {
        /// Output structured JSON instead of a human table
        #[arg(long)]
        json: bool,
        /// Resolve a model reference (`provider/model`, bare id, or family alias) to canonical `provider/id`
        #[arg(long)]
        resolve: Option<String>,
        /// Include unconfigured providers in listing
        #[arg(long)]
        all: bool,
    },
    /// Manage & inspect configured MCP 2026-07-28 servers and deferred tools
    Mcp {
        #[command(subcommand)]
        action: McpCommands,
    },
    /// Authenticate with OAuth 2.0 providers (PKCE, Device Code, `OpenRouter` PKCE) & manage `~/.carnot/auth.json`
    Auth {
        #[command(subcommand)]
        action: AuthCommands,
    },
}

#[derive(Debug, Clone, Subcommand)]
enum McpCommands {
    /// List configured MCP servers & tool counts (uses zero-cold-start schema cache when warm)
    List,
    /// Inspect all discovered MCP tool specs (minified & canonicalized)
    Inspect,
}

#[derive(Debug, Clone, Subcommand)]
enum AuthCommands {
    /// Log in to a provider via OAuth 2.0 (PKCE, Device Code, or `OpenRouter` PKCE) or store a token securely (`0600`)
    Login {
        /// Provider name or alias from providers.json (e.g. `openrouter`, `gemini`, `anthropic`, `openai`)
        provider: String,
        /// Use RFC 8628 Device Authorization Grant instead of browser loopback PKCE
        #[arg(long)]
        device: bool,
        /// Headless/SSH mode: paste redirected callback URL or authorization code manually instead of binding localhost
        #[arg(long)]
        manual: bool,
        /// Do not auto-open the system browser (print URL only)
        #[arg(long = "no-browser")]
        no_browser: bool,
        /// Explicit localhost callback port for loopback PKCE redirect
        #[arg(long)]
        port: Option<u16>,
        /// Override OAuth `client_id` for this login
        #[arg(long = "client-id")]
        client_id: Option<String>,
        /// Store a Bearer access token read from stdin into `~/.carnot/auth.json` (`0600`)
        #[arg(long = "token-stdin")]
        token_stdin: bool,
    },
    /// Show secret-safe authentication & OAuth token status across all configured providers
    Status,
    /// Force-refresh a stored OAuth token immediately using its `refresh_token`
    Refresh {
        /// Provider name or alias from providers.json
        provider: String,
    },
    /// Remove stored OAuth credentials for a provider from `~/.carnot/auth.json`
    Logout {
        /// Provider name or alias from providers.json
        provider: String,
    },
}

fn init_tracing_if_requested() {
    if std::env::var_os("RUST_LOG").is_some() {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_writer(std::io::stderr)
            .init();
    }
}

fn main() {
    try_handle_landlock_exec_arg();
    init_tracing_if_requested();

    let mut cli = Cli::parse();
    if !cli.host_extension {
        match config::load_user_config() {
            Ok(Some((path, cfg))) => {
                if let Err(e) = cli.apply_user_config(&cfg) {
                    eprintln!("config error ({}): {e}", path.display());
                    std::process::exit(2);
                }
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("config error: {e}");
                std::process::exit(2);
            }
        }
    }
    let cwd = cli
        .cwd
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    // Fast synchronous subcommands run with ZERO Tokio runtime threads (~2ms cold start).
    if let Some(ref cmd) = cli.command {
        match cmd {
            Commands::AuditPrefix => {
                match assert_prefix_invariance(|_date, _cwd, _sid| {
                    (static_system_prompt().to_string(), canonical_tool_specs())
                }) {
                    Ok(hash) => {
                        let out = json!({
                            "status": "pass",
                            "prefix_blake3": hash,
                            "static_prompt_chars": static_system_prompt().len(),
                            "tools_count": canonical_tool_specs().len(),
                        });
                        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
                        return;
                    }
                    Err(e) => {
                        eprintln!("PREFIX INVARIANT FAILED: {e}");
                        std::process::exit(1);
                    }
                }
            }
            Commands::Modules => {
                let eff_profile = cli.effective_profile();
                let sw = carnot::ModuleSwitches::from_profile(eff_profile.into());
                let out = json!({
                    "profile": format!("{eff_profile:?}").to_lowercase(),
                    "modules": sw,
                    "canonical_tools": canonical_tool_specs().iter().map(|t| &t.name).collect::<Vec<_>>(),
                    "compiled_extensions": compiled_extensions(),
                    "optional_modules": ["subagent", "policy_guard", "mcp_2026_07_28", "dropin_extensions"],
                });
                println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
                return;
            }
            Commands::Providers => match ProviderRegistry::load_default(&cli.provider_configs) {
                Ok(reg) => {
                    let out = json!({
                        "registered_api_adapters": reg.registered_apis(),
                        "providers": reg.summaries(),
                    });
                    println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
                    return;
                }
                Err(e) => {
                    eprintln!("providers error: {e}");
                    std::process::exit(1);
                }
            },
            Commands::Models { json, resolve, all } => {
                match ProviderRegistry::load_default(&cli.provider_configs) {
                    Ok(reg) => {
                        if let Some(reference) = resolve {
                            if let Some((prov, model)) = reg.resolve_model_spec(reference, None) {
                                println!("{prov}/{model}");
                                return;
                            }
                            eprintln!("models error: could not resolve `{reference}`");
                            std::process::exit(1);
                        }
                        let models = list_rpc_models(&reg, *all);
                        if *json {
                            let out = json!({ "models": models });
                            println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
                        } else if models.is_empty() {
                            println!(
                                "No models available (configure provider credentials first or pass --all)."
                            );
                        } else {
                            println!("{:<20} {:<38} {:>10}", "provider", "model", "context");
                            for m in models {
                                let ctx = m
                                    .context_window
                                    .map_or_else(|| "-".into(), |c| format!("{}k", c / 1000));
                                println!("{:<20} {:<38} {:>10}", m.provider, m.id, ctx);
                            }
                        }
                        return;
                    }
                    Err(e) => {
                        eprintln!("models error: {e}");
                        std::process::exit(1);
                    }
                }
            }
            _ => {}
        }
    }

    // Use a single-threaded Tokio runtime for CLI/ACP/RPC/Doctor/MCP (zero extra pthread stacks or RSS),
    // and a compact multi-threaded runtime for the multi-client HTTP server (`carnot serve`).
    let is_serve = matches!(cli.command, Some(Commands::Serve { .. }));
    let rt = if is_serve {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
    } else {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
    }
    .expect("tokio runtime");

    let exit_code = rt.block_on(async_main(cli, cwd));
    // All async cleanup (MCP shutdown, session flush, abort) has already finished inside `async_main`;
    // detach any blocking `tokio::io::stdin()` reader thread so SIGTERM shutdown never hangs on drop.
    rt.shutdown_background();
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

async fn async_main(cli: Cli, cwd: PathBuf) -> i32 {
    if let Some(ref cmd) = cli.command {
        match cmd {
            Commands::Doctor => {
                return run_doctor(&cli, &cwd);
            }
            Commands::Mcp { action } => {
                if let Err(e) = run_mcp_cmd(&cli, &cwd, action).await {
                    eprintln!("mcp error: {e}");
                    return 1;
                }
                return 0;
            }
            Commands::Auth { action } => {
                return run_auth_cmd(&cli, action).await;
            }
            Commands::Serve { addr } => {
                if let Ok(sock) = addr.parse::<std::net::SocketAddr>()
                    && !sock.ip().is_loopback()
                {
                    eprintln!(
                        "warning: binding carnot HTTP server to non-loopback address `{addr}`; ensure network isolation (Host/Origin checks only allow loopback)."
                    );
                }
                let (agent, mcp) = match build_agent(&cli, cwd).await {
                    Ok(pair) => pair,
                    Err(e) => {
                        eprintln!("error: {e}");
                        return 1;
                    }
                };
                let res = serve_http(agent, addr).await;
                mcp.shutdown_all().await;
                if let Err(e) = res {
                    eprintln!("server error: {e}");
                    return 1;
                }
                return 0;
            }
            Commands::AuditPrefix
            | Commands::Modules
            | Commands::Providers
            | Commands::Models { .. } => return 0,
        }
    }

    // In `--mode rpc --host-extension`, perform the `host/register` handshake BEFORE building
    // the Agent so host-supplied tools & static system-prompt sections freeze before Turn 1.
    if matches!(cli.mode, Mode::Rpc) && cli.host_extension {
        let pending = match read_host_registration().await {
            Ok(p) => p,
            Err(e) => {
                eprintln!("rpc host-extension handshake error: {e}");
                return 1;
            }
        };
        let timeout = Duration::from_millis(pending.registration.timeout_ms.max(1_000));
        let bridge = Arc::new(HostRpcBridge::new(Arc::clone(&pending.stdout), timeout));
        let sys_append = pending.registration.system_prompt_append.clone();
        let host_mod: Arc<dyn carnot::Module> = Arc::new(HostRpcModule::from_registration(
            pending.registration.clone(),
            Arc::clone(&bridge),
        ));
        let (agent, mcp) = match build_agent_full(
            &cli,
            cwd.clone(),
            BTreeMap::new(),
            vec![host_mod],
            sys_append,
        )
        .await
        {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!("error: {e}");
                return 1;
            }
        };
        let rpc_host: Arc<dyn RpcHost> = Arc::new(CliRpcHost {
            cli: cli.clone(),
            cwd,
        });
        let code =
            match serve_rpc_stdio_with_extension(agent, Some(rpc_host), bridge, pending).await {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("rpc error: {e}");
                    1
                }
            };
        mcp.shutdown_all().await;
        return code;
    }

    let (agent, mcp) = match build_agent(&cli, cwd.clone()).await {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };

    let code = match cli.mode {
        Mode::Acp => {
            let host = Arc::new(CliAcpHost {
                cli: cli.clone(),
                session_mcps: tokio::sync::Mutex::new(Vec::new()),
            });
            let rc = match serve_acp_stdio_with_host(agent, Arc::clone(&host) as Arc<dyn AcpHost>)
                .await
            {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("acp error: {e}");
                    1
                }
            };
            host.shutdown_session_mcps().await;
            rc
        }
        Mode::Rpc => {
            let rpc_host: Arc<dyn RpcHost> = Arc::new(CliRpcHost {
                cli: cli.clone(),
                cwd,
            });
            match serve_rpc_stdio_with_host(agent, Some(rpc_host)).await {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("rpc error: {e}");
                    1
                }
            }
        }
        Mode::Text | Mode::Json => {
            let stdin_is_tty = std::io::stdin().is_terminal();
            let prompt = match resolve_one_shot_prompt(&cli, stdin_is_tty, || {
                use std::io::Read;
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf)?;
                Ok(buf)
            }) {
                Ok(Some(p)) => p,
                Ok(None) if matches!(cli.mode, Mode::Text) && stdin_is_tty => {
                    let want_plain = cli.plain
                        || !std::io::stdout().is_terminal()
                        || std::env::var_os("CARNOT_PLAIN").is_some()
                        || matches!(std::env::var("TERM").ok().as_deref(), Some("dumb"));
                    let reg_opt = ProviderRegistry::load_default(&cli.provider_configs).ok();
                    let default_provider = cli.resolved_provider(reg_opt.as_ref());
                    #[cfg(feature = "tui")]
                    let rc = if want_plain {
                        repl::run_repl(agent, default_provider, cli.provider_configs.clone()).await
                    } else {
                        carnot_tui::run_tui(
                            agent,
                            carnot_tui::TuiLaunchInfo {
                                default_provider,
                                provider_configs: cli.provider_configs.clone(),
                                active_extensions: active_extension_badges(&cli),
                            },
                        )
                        .await
                    };
                    #[cfg(not(feature = "tui"))]
                    let rc = {
                        let _ = want_plain;
                        repl::run_repl(agent, default_provider, cli.provider_configs.clone()).await
                    };
                    mcp.shutdown_all().await;
                    return rc;
                }
                Ok(None) => {
                    eprintln!(
                        "usage: carnot [-p \"<prompt>\" | --prompt-file <path>] [--image <path>] [-c] | carnot doctor | carnot audit-prefix | carnot providers | carnot models [--all] | carnot auth login|status | carnot modules | carnot --mode acp|rpc | carnot serve | carnot mcp list"
                    );
                    mcp.shutdown_all().await;
                    return 2;
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    mcp.shutdown_all().await;
                    return 2;
                }
            };

            let image_blocks = match load_image_blocks(&cli.images) {
                Ok(blocks) => blocks,
                Err(e) => {
                    eprintln!("image error: {e}");
                    mcp.shutdown_all().await;
                    return 2;
                }
            };

            let is_json = matches!(cli.mode, Mode::Json);
            let mut ev_stream = agent.subscribe_from(0);
            let ev_task = tokio::spawn(async move {
                use std::io::Write as _;
                loop {
                    let env = ev_stream.next().await;
                    if is_json {
                        if let Ok(line) = serde_json::to_string(&env)
                            && writeln!(std::io::stdout(), "{line}").is_err()
                        {
                            break;
                        }
                    } else if let AgentEvent::TextDelta { delta } = &env.event {
                        let mut out = std::io::stdout();
                        if write!(out, "{delta}").is_err() {
                            break;
                        }
                        let _ = out.flush();
                    }
                    if matches!(
                        env.event,
                        AgentEvent::RunSettled { .. } | AgentEvent::Error { .. }
                    ) {
                        break;
                    }
                }
            });

            // Graceful SIGINT / SIGTERM cancellation + optional wall-clock timeout
            let agent_sig = Arc::clone(&agent);
            let sig_code = Arc::new(std::sync::atomic::AtomicI32::new(0));
            let sig_code_bg = Arc::clone(&sig_code);
            let sig_task = tokio::spawn(async move {
                let code = wait_termination_signal().await;
                sig_code_bg.store(code, std::sync::atomic::Ordering::SeqCst);
                agent_sig.abort().await;
            });

            let run_fut = async {
                if image_blocks.is_empty() {
                    agent.prompt(prompt).await
                } else {
                    let mut blocks = Vec::with_capacity(1 + image_blocks.len());
                    blocks.push(ContentBlock::Text { text: prompt });
                    blocks.extend(image_blocks);
                    agent.prompt_blocks(blocks).await
                }
            };
            let run_res = if let Some(secs) = cli.timeout_secs {
                if let Ok(res) =
                    tokio::time::timeout(Duration::from_secs(secs.max(1)), run_fut).await
                {
                    res
                } else {
                    agent.abort().await;
                    Err(carnot::CarnotError::Tool {
                        tool: "cli".into(),
                        message: format!("session timed out after {secs}s"),
                    })
                }
            } else {
                run_fut.await
            };

            sig_task.abort();
            let got_sig = sig_code.load(std::sync::atomic::Ordering::SeqCst);
            match run_res {
                Ok(outcome) => {
                    let _ = tokio::time::timeout(Duration::from_millis(250), ev_task).await;
                    if !is_json {
                        println!();
                    }
                    if got_sig != 0 {
                        got_sig
                    } else if let Some(ref reason) = outcome.halted_reason {
                        eprintln!("halted: {reason}");
                        3
                    } else {
                        0
                    }
                }
                Err(e) => {
                    // Allow JSON event printer to flush `AgentEvent::Error` before exiting!
                    let _ = tokio::time::timeout(Duration::from_millis(250), ev_task).await;
                    if got_sig != 0 {
                        eprintln!("interrupted (signal exit {got_sig})");
                        got_sig
                    } else {
                        eprintln!("error: {e}");
                        1
                    }
                }
            }
        }
    };
    mcp.shutdown_all().await;
    code
}

async fn wait_termination_signal() -> i32 {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = signal(SignalKind::terminate()).ok();
        tokio::select! {
            _ = tokio::signal::ctrl_c() => 130,
            () = async {
                if let Some(ref mut s) = sigterm {
                    s.recv().await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => 143,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        130
    }
}

fn compiled_extensions() -> Vec<&'static str> {
    let items: &[&str] = &[
        #[cfg(feature = "ext-web")]
        "carnot-ext-web",
        #[cfg(feature = "ext-lsp")]
        "carnot-ext-lsp",
        #[cfg(feature = "ext-subagent")]
        "carnot-ext-subagent",
        #[cfg(feature = "ext-hooks")]
        "carnot-ext-hooks",
        #[cfg(feature = "ext-claude-plugins")]
        "carnot-ext-claude-plugins",
        #[cfg(feature = "ext-memory")]
        "carnot-ext-memory",
    ];
    items.to_vec()
}

#[cfg(feature = "tui")]
fn active_extension_badges(cli: &Cli) -> Vec<String> {
    let mut out = Vec::new();
    #[cfg(feature = "ext-web")]
    if cli.web_enabled() {
        out.push("web".to_string());
    }
    #[cfg(feature = "ext-memory")]
    if cli.memory_enabled() {
        out.push("memory".to_string());
    }
    #[cfg(feature = "ext-subagent")]
    if cli.subagent_enabled() {
        out.push("subagent".to_string());
    }
    #[cfg(feature = "ext-lsp")]
    if cli.lsp_enabled() {
        out.push("lsp".to_string());
    }
    #[cfg(feature = "ext-hooks")]
    if !cli.hooks_files.is_empty() || cli.claude_hooks || cli.trust_workspace_hooks {
        out.push("hooks".to_string());
    }
    #[cfg(feature = "ext-claude-plugins")]
    if !cli.claude_plugins.is_empty() || cli.load_installed_claude_plugins {
        out.push("plugins".to_string());
    }
    if !cli.extensions.is_empty() || cli.trust_workspace_extensions {
        out.push("ext".to_string());
    }
    if cli.policy_guard {
        out.push("policy".to_string());
    }
    out
}

impl Cli {
    fn effective_profile(&self) -> CliProfile {
        self.profile.unwrap_or(CliProfile::Carnot)
    }

    fn effective_effort(&self) -> CliEffort {
        self.effort.unwrap_or(CliEffort::High)
    }

    fn effective_cache_ttl(&self) -> &str {
        self.cache_ttl.as_deref().unwrap_or("hybrid")
    }

    /// Merges `~/.carnot/config.toml` defaults into `Cli` without overriding explicit CLI flags.
    fn apply_user_config(&mut self, cfg: &config::UserConfig) -> Result<(), String> {
        if self.provider.is_none() {
            self.provider.clone_from(&cfg.provider);
        }
        if self.model.is_none() {
            self.model.clone_from(&cfg.model);
        }
        if self.effort.is_none()
            && let Some(ref e) = cfg.effort
        {
            self.effort = Some(CliEffort::from_str(e, true).map_err(|_| {
                format!("invalid effort `{e}` (expected low|medium|high|xhigh|max)")
            })?);
        }
        if self.cache_ttl.is_none()
            && let Some(ref ttl) = cfg.cache_ttl
        {
            if CacheTtl::parse(ttl).is_none() {
                return Err(format!(
                    "invalid cache_ttl `{ttl}` (expected short|hybrid|long, or 5m|1h|24h)"
                ));
            }
            self.cache_ttl = Some(ttl.clone());
        }
        if self.profile.is_none()
            && let Some(ref p) = cfg.profile
        {
            self.profile = Some(CliProfile::from_str(p, true).map_err(|_| {
                format!("invalid profile `{p}` (expected minimal|standard|carnot)")
            })?);
        }
        if !self.plain && cfg.plain == Some(true) {
            self.plain = true;
        }
        if !self.no_sandbox && cfg.sandbox == Some(false) {
            self.no_sandbox = true;
        }
        if !self.allow_network && cfg.allow_network == Some(true) {
            self.allow_network = true;
        }
        if !self.no_session && cfg.persist_session == Some(false) {
            self.no_session = true;
        }
        if self.session_dir.is_none() {
            self.session_dir.clone_from(&cfg.session_dir);
        }
        if self.max_turns.is_none() {
            self.max_turns = cfg.max_turns;
        }
        if self.max_cost_usd.is_none() {
            self.max_cost_usd = cfg.max_cost_usd;
        }
        if self.compact_at.is_none() {
            self.compact_at = cfg.compact_at;
        }
        if self.timeout_secs.is_none() {
            self.timeout_secs = cfg.timeout_secs;
        }
        if self.exclude_tools.is_empty()
            && let Some(ref tools) = cfg.exclude_tools
        {
            self.exclude_tools.clone_from(tools);
        }
        if !self.no_mcp && cfg.extensions.mcp == Some(false) {
            self.no_mcp = true;
        }
        #[cfg(feature = "ext-web")]
        match cfg.extensions.web {
            Some(false) if !self.web => self.no_web = true,
            Some(true) if !self.no_web => self.web = true,
            _ => {}
        }
        #[cfg(feature = "ext-lsp")]
        match cfg.extensions.lsp {
            Some(false) if !self.lsp => self.no_lsp = true,
            Some(true) if !self.no_lsp => self.lsp = true,
            _ => {}
        }
        #[cfg(feature = "ext-subagent")]
        match cfg.extensions.subagent {
            Some(false) if !self.subagent => self.no_subagent = true,
            Some(true) if !self.no_subagent => self.subagent = true,
            _ => {}
        }
        #[cfg(feature = "ext-memory")]
        {
            match cfg.extensions.memory {
                Some(false) if !self.memory => self.no_memory = true,
                Some(true) if !self.no_memory => self.memory = true,
                _ => {}
            }
            if self.memory_dir.is_none() {
                self.memory_dir.clone_from(&cfg.extensions.memory_dir);
            }
        }
        Ok(())
    }

    /// Standalone `carnot` (not `--host-extension`) on the default `carnot` profile enables
    /// standard zero-prefix-cost extensions out of the box (opt out with `--no-*` or `--profile minimal`).
    #[cfg(any(
        feature = "ext-web",
        feature = "ext-lsp",
        feature = "ext-subagent",
        feature = "ext-memory"
    ))]
    fn default_batteries(&self) -> bool {
        !self.host_extension && matches!(self.effective_profile(), CliProfile::Carnot)
    }

    #[cfg(feature = "ext-web")]
    fn web_enabled(&self) -> bool {
        !self.no_web && (self.web || self.default_batteries())
    }

    #[cfg(feature = "ext-lsp")]
    fn lsp_enabled(&self) -> bool {
        !self.no_lsp && (self.lsp || self.default_batteries())
    }

    #[cfg(feature = "ext-subagent")]
    fn subagent_enabled(&self) -> bool {
        !self.no_subagent && (self.subagent || self.default_batteries())
    }

    #[cfg(feature = "ext-memory")]
    fn memory_enabled(&self) -> bool {
        !self.no_memory && (self.memory || self.memory_dir.is_some() || self.default_batteries())
    }

    /// Resolves `--provider`, or auto-selects the first configured cloud provider before falling
    /// back to `"anthropic"`.
    fn resolved_provider(&self, reg: Option<&ProviderRegistry>) -> String {
        const PREFERRED: &[&str] = &[
            "anthropic",
            "openai",
            "openai-responses",
            "gemini",
            "grok",
            "muse",
            "mistral",
            "opencode",
            "opencode-go",
            "openrouter",
            "deepseek",
        ];
        if let Some(ref explicit) = self.provider {
            return explicit.clone();
        }
        if let Some(r) = reg {
            let summaries = r.summaries();
            for pref in PREFERRED {
                if summaries.iter().any(|s| s.id == *pref && s.configured) {
                    return (*pref).to_string();
                }
            }
            if let Some(first) = summaries.iter().find(|s| {
                s.configured
                    && s.api != "faux"
                    && !matches!(s.id.as_str(), "ollama" | "vllm" | "lmstudio")
            }) {
                return first.id.clone();
            }
        }
        "anthropic".to_string()
    }
}

fn run_doctor(cli: &Cli, cwd: &std::path::Path) -> i32 {
    let prefix = assert_prefix_invariance(|_date, _cwd, _sid| {
        (static_system_prompt().to_string(), canonical_tool_specs())
    });
    let reg = ProviderRegistry::load_default(&cli.provider_configs);
    let (provider_status, selected_model, context_window) = match &reg {
        Ok(r) => {
            let target_prov = cli.resolved_provider(Some(r));
            let canonical_id = r
                .catalog()
                .find_provider(&target_prov)
                .map_or_else(|| target_prov.clone(), |(id, _)| id.to_string());
            let summary = r
                .summaries()
                .into_iter()
                .find(|s| s.id.eq_ignore_ascii_case(&canonical_id));
            let model = cli.model.clone().unwrap_or_else(|| {
                r.catalog()
                    .find_provider(&target_prov)
                    .and_then(|(_, e)| e.default_model.clone())
                    .unwrap_or_else(|| "claude-sonnet-5-5".to_string())
            });
            let ctx = r.context_window_for_model(&model).unwrap_or(200_000);
            let st = match summary {
                Some(s) => json!({
                    "id": s.id,
                    "api": s.api,
                    "configured": s.configured,
                    "auth_source": s.auth_source,
                    "base_url": s.base_url,
                }),
                None => json!({ "id": target_prov, "error": "unknown provider" }),
            };
            (st, model, ctx)
        }
        Err(e) => (
            json!({ "error": e.to_string() }),
            cli.model.clone().unwrap_or_default(),
            200_000,
        ),
    };

    let sb = detect_sandbox_backend();
    let landlock_warn = (sb.backend == "landlock").then_some(
        "Landlock is additive allow-only and cannot make `<workspace>/.git` read-only for Bash commands (in-process Edit/Write still block `.git`); install `bwrap` on Linux for strict `.git` RO isolation.",
    );
    let sandbox_backend = match (sb.detail, landlock_warn) {
        (Some(detail), Some(warning)) => json!({
            "backend": sb.backend,
            "available": sb.available,
            "detail": detail,
            "warning": warning,
        }),
        (Some(detail), None) => json!({
            "backend": sb.backend,
            "available": sb.available,
            "detail": detail,
        }),
        (None, Some(warning)) => json!({
            "backend": sb.backend,
            "available": sb.available,
            "warning": warning,
        }),
        (None, None) => json!({
            "backend": sb.backend,
            "available": sb.available,
        }),
    };

    let mcp_cfgs = load_mcp_configs(cwd, &cli.mcp_configs, cli.trust_workspace_mcp);
    let mcp_info = match mcp_cfgs {
        Ok(cfgs) => json!({ "configured_servers": cfgs.keys().collect::<Vec<_>>() }),
        Err(e) => json!({ "error": e.to_string() }),
    };

    let skills = discover_agent_skills(cwd, &cli.skills_dirs);
    let skills_names: Vec<&str> = skills.iter().map(|(n, _, _)| n.as_str()).collect();
    let dropin_exts = discover_dropin_specs(cwd, &cli.extensions, cli.trust_workspace_extensions);
    let active_config = config::user_config_path()
        .filter(|(p, _)| p.is_file())
        .map(|(p, _)| p);

    let out = json!({
        "status": if prefix.is_ok() { "ok" } else { "degraded" },
        "version": env!("CARGO_PKG_VERSION"),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "workspace": cwd,
        "config_file": active_config,
        "prefix_invariance": prefix.as_ref().map_or_else(|e| json!({"pass": false, "error": e.to_string()}), |h| json!({"pass": true, "blake3": h})),
        "provider": provider_status,
        "model": selected_model,
        "context_window": context_window,
        "sandbox": sandbox_backend,
        "mcp": mcp_info,
        "extensions": {
            "compiled": compiled_extensions(),
            "dropin": dropin_exts.iter().map(|e| json!({"name": e.name, "source": e.source})).collect::<Vec<_>>(),
        },
        "skills": { "count": skills_names.len(), "names": skills_names },
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    i32::from(prefix.is_err())
}

async fn run_mcp_cmd(cli: &Cli, cwd: &std::path::Path, action: &McpCommands) -> carnot::Result<()> {
    let t0 = Instant::now();
    let cfgs = load_mcp_configs(cwd, &cli.mcp_configs, cli.trust_workspace_mcp)?;
    let cas = Arc::new(CasStore::open(cwd)?);
    let mgr = McpManager::from_configs(cfgs, cwd.to_path_buf(), cas, true, 4096, 1400).await?;
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    match action {
        McpCommands::List => {
            let out = json!({
                "protocol_version": carnot::MCP_PROTOCOL_VERSION,
                "load_ms": (load_ms * 100.0).round() / 100.0,
                "servers": mgr.summaries(),
            });
            println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
        }
        McpCommands::Inspect => {
            let specs: Vec<_> = mgr.tools().iter().map(|t| t.spec()).collect();
            let out = json!({
                "protocol_version": carnot::MCP_PROTOCOL_VERSION,
                "load_ms": (load_ms * 100.0).round() / 100.0,
                "tools_count": specs.len(),
                "tools": specs,
            });
            println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
        }
    }
    mgr.shutdown_all().await;
    Ok(())
}

#[derive(Debug)]
struct CliAcpHost {
    cli: Cli,
    session_mcps: tokio::sync::Mutex<Vec<Arc<McpManager>>>,
}

impl CliAcpHost {
    async fn shutdown_session_mcps(&self) {
        let managers: Vec<Arc<McpManager>> = {
            let mut guard = self.session_mcps.lock().await;
            std::mem::take(&mut *guard)
        };
        for mgr in managers {
            mgr.shutdown_all().await;
        }
    }
}

impl AcpHost for CliAcpHost {
    fn available_models(&self) -> Vec<AcpModelInfo> {
        let Ok(reg) = ProviderRegistry::load_default(&self.cli.provider_configs) else {
            return Vec::new();
        };
        let cat = reg.catalog();
        reg.available_models(false)
            .into_iter()
            .map(|m| {
                let desc = cat
                    .find_provider(&m.provider)
                    .map(|(_, e)| {
                        if e.description.is_empty() {
                            m.provider.clone()
                        } else {
                            e.description.clone()
                        }
                    })
                    .or(Some(m.provider));
                AcpModelInfo {
                    id: m.id,
                    name: m.label,
                    description: desc,
                }
            })
            .collect()
    }

    fn resolve_model(&self, model_id: &str) -> carnot::Result<ResolvedAcpModel> {
        let reg = ProviderRegistry::load_default(&self.cli.provider_configs)?;
        let (prov_id, resolved_id) = reg.resolve_model_spec(model_id, None).unwrap_or_else(|| {
            (
                self.cli.resolved_provider(Some(&reg)),
                model_id.trim().to_string(),
            )
        });
        let provider = reg.build_provider(&prov_id, Some(&resolved_id))?;
        let context_window = reg.context_window_for_model(&resolved_id);
        let max_output_tokens = reg.max_output_tokens_for_model(&resolved_id);
        Ok(ResolvedAcpModel {
            provider,
            context_window,
            max_output_tokens,
        })
    }

    fn create_agent_for_cwd(
        &self,
        cwd: PathBuf,
        model: Option<String>,
        effort: Option<ReasoningEffort>,
        mcp_servers: Vec<serde_json::Value>,
    ) -> carnot_ext::BoxFuture<'_, carnot::Result<Arc<carnot::Agent>>> {
        let mut eff_cli = self.cli.clone();
        if let Some(m) = model {
            eff_cli.model = Some(m);
        }
        if let Some(e) = effort {
            eff_cli.effort = Some(match e {
                ReasoningEffort::Low => CliEffort::Low,
                ReasoningEffort::Medium => CliEffort::Medium,
                ReasoningEffort::High => CliEffort::High,
                ReasoningEffort::Xhigh => CliEffort::Xhigh,
                ReasoningEffort::Max => CliEffort::Max,
            });
        }
        Box::pin(async move {
            let extra_mcp = parse_acp_mcp_servers(&mcp_servers, &cwd);
            let (agent, session_mcp) = build_agent_with_mcp(&eff_cli, cwd, extra_mcp).await?;
            self.session_mcps.lock().await.push(session_mcp);
            Ok(agent)
        })
    }
}

#[derive(Debug)]
struct CliRpcHost {
    cli: Cli,
    cwd: PathBuf,
}

fn list_rpc_models(reg: &ProviderRegistry, include_unconfigured: bool) -> Vec<RpcModelInfo> {
    let cat = reg.catalog();
    reg.available_models(!include_unconfigured)
        .into_iter()
        .map(|m| {
            let entry = cat.find_model(&m.id);
            let supports_effort = entry.and_then(|e| e.supports_effort).unwrap_or(false);
            let thinking_mode = entry.and_then(|e| e.thinking_mode.as_deref());
            let effort_preserves_cache = supports_effort && thinking_mode != Some("budget");
            RpcModelInfo {
                provider: m.provider,
                id: m.id.clone(),
                label: Some(m.label),
                context_window: reg.context_window_for_model(&m.id),
                max_output_tokens: reg.max_output_tokens_for_model(&m.id),
                effort_preserves_cache,
            }
        })
        .collect()
}

impl RpcHost for CliRpcHost {
    fn list_models(&self) -> Vec<RpcModelInfo> {
        let Ok(reg) = ProviderRegistry::load_default(&self.cli.provider_configs) else {
            return Vec::new();
        };
        list_rpc_models(&reg, false)
    }

    fn resolve_model(
        &self,
        model_spec: &str,
        provider_override: Option<&str>,
    ) -> carnot::Result<ResolvedRpcModel> {
        let reg = ProviderRegistry::load_default(&self.cli.provider_configs)?;
        let (prov_id, model_id) = reg
            .resolve_model_spec(model_spec, provider_override)
            .unwrap_or_else(|| {
                (
                    self.cli.resolved_provider(Some(&reg)),
                    model_spec.trim().to_string(),
                )
            });
        let provider = reg.build_provider(&prov_id, Some(&model_id))?;
        let context_window = reg.context_window_for_model(&model_id);
        let max_output_tokens = reg.max_output_tokens_for_model(&model_id);
        Ok(ResolvedRpcModel {
            provider_id: prov_id,
            model_id,
            provider,
            context_window,
            max_output_tokens,
        })
    }

    fn list_commands(&self) -> Vec<RpcCommandInfo> {
        discover_agent_skills(&self.cwd, &self.cli.skills_dirs)
            .into_iter()
            .map(|(name, desc, _)| RpcCommandInfo {
                name,
                description: if desc.is_empty() { None } else { Some(desc) },
            })
            .collect()
    }
}

async fn build_agent(
    cli: &Cli,
    cwd: PathBuf,
) -> carnot::Result<(Arc<carnot::Agent>, Arc<McpManager>)> {
    build_agent_full(cli, cwd, BTreeMap::new(), Vec::new(), None).await
}

async fn build_agent_with_mcp(
    cli: &Cli,
    cwd: PathBuf,
    extra_mcp: BTreeMap<String, McpServerConfig>,
) -> carnot::Result<(Arc<carnot::Agent>, Arc<McpManager>)> {
    build_agent_full(cli, cwd, extra_mcp, Vec::new(), None).await
}

async fn build_agent_full(
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

async fn run_auth_cmd(cli: &Cli, action: &AuthCommands) -> i32 {
    let catalog = match carnot::ProviderCatalog::load_layered(&cli.provider_configs) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("auth catalog error: {e}");
            return 1;
        }
    };
    let Some(store) = carnot::AuthStore::open_default() else {
        eprintln!("auth error: could not determine HOME directory for ~/.carnot/auth.json");
        return 1;
    };

    match action {
        AuthCommands::Status => {
            let stored = store.list().unwrap_or_default();
            let summaries = catalog.summaries();
            let mut rows: Vec<serde_json::Value> = Vec::with_capacity(summaries.len());
            for s in summaries {
                let entry_opt = catalog.find_provider(&s.id).map(|(_, e)| e);
                let entry_oauth = entry_opt.and_then(|e| e.oauth.as_ref());
                let profile = entry_opt.map_or(s.id.as_str(), |e| e.oauth_profile(&s.id));
                let tok = stored.get(profile);
                rows.push(json!({
                    "provider": s.id,
                    "oauth_profile": profile,
                    "api": s.api,
                    "configured": s.configured,
                    "auth_source": s.auth_source,
                    "oauth_flow": entry_oauth.map(|o| format!("{:?}", o.flow).to_lowercase()),
                    "oauth_stored": tok.is_some(),
                    "oauth_expired": tok.map(carnot::OAuthStoredToken::is_expired),
                    "oauth_expires_in_secs": tok.and_then(carnot::OAuthStoredToken::seconds_until_expiry),
                    "oauth_refreshable": tok.map(|t| t.can_refresh(entry_oauth)),
                }));
            }
            let out = json!({
                "store_path": store.path(),
                "providers": rows,
            });
            println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
            0
        }
        AuthCommands::Login {
            provider,
            device,
            manual,
            no_browser,
            port,
            client_id,
            token_stdin,
        } => {
            let (canonical_id, entry) = if let Some((id, e)) = catalog.find_provider(provider) {
                (id.to_string(), e.clone())
            } else {
                eprintln!("auth error: unknown provider `{provider}` (run `carnot providers`)");
                return 1;
            };
            let profile = entry.oauth_profile(&canonical_id).to_string();

            if *token_stdin {
                let mut buf = String::new();
                if let Err(e) = std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf) {
                    eprintln!("auth error reading token from stdin: {e}");
                    return 1;
                }
                match carnot::save_manual_token(&profile, &buf, &store) {
                    Ok(_) => {
                        println!(
                            "Stored Bearer token for `{canonical_id}` (profile `{profile}`) in {} (mode 0600).",
                            store.path().display()
                        );
                        return 0;
                    }
                    Err(e) => {
                        eprintln!("auth error: {e}");
                        return 1;
                    }
                }
            }

            let Some(oauth_cfg) = entry.oauth.as_ref() else {
                eprintln!(
                    "Provider `{canonical_id}` has no OAuth 2.0 flow configured in providers.json.\n\
                     Either add an `oauth` block in ~/.carnot/providers.json (standard PKCE / Device Code),\n\
                     or store a Bearer token securely with: printf '%s' \"$TOKEN\" | carnot auth login {canonical_id} --token-stdin"
                );
                return 1;
            };

            let opts = carnot::OAuthLoginOptions {
                force_device_flow: *device,
                manual_callback: *manual,
                open_browser: !*no_browser,
                port_override: *port,
                client_id_override: client_id.clone(),
                timeout: Duration::from_secs(300),
            };

            let res = carnot::login_with_oauth(
                &profile,
                oauth_cfg,
                &store,
                &opts,
                |prompt| match prompt {
                    carnot::OAuthPrompt::OpenUrl {
                        url,
                        callback_url,
                        manual_paste,
                    } => {
                        eprintln!("Authenticate `{canonical_id}` by opening:\n  {url}");
                        if manual_paste {
                            if callback_url.is_empty() {
                                eprintln!(
                                    "After approving, paste the displayed authorization code below and press Enter:"
                                );
                            } else {
                                eprintln!(
                                    "After approving, paste the full callback URL (starting with {callback_url}) or code below and press Enter:"
                                );
                            }
                        } else {
                            eprintln!("Waiting for OAuth callback on {callback_url} ...");
                        }
                    }
                    carnot::OAuthPrompt::DeviceCode {
                        verification_uri,
                        verification_uri_complete,
                        user_code,
                    } => {
                        eprintln!("Authenticate `{canonical_id}` via Device Code:");
                        eprintln!("  URL:  {verification_uri}");
                        eprintln!("  Code: {user_code}");
                        if let Some(full) = verification_uri_complete {
                            eprintln!("  Direct link: {full}");
                        }
                        eprintln!("Waiting for approval ...");
                    }
                },
                || {
                    let mut line = String::new();
                    std::io::stdin()
                        .read_line(&mut line)
                        .map_err(carnot::CarnotError::Io)?;
                    Ok(line)
                },
            )
            .await;

            match res {
                Ok(tok) => {
                    let expiry = tok
                        .seconds_until_expiry()
                        .map(|s| format!(" (expires in {s}s)"))
                        .unwrap_or_default();
                    let refreshable = if tok.refresh_token.is_some() {
                        ", auto-refresh enabled"
                    } else {
                        ""
                    };
                    println!(
                        "Authenticated `{canonical_id}` (profile `{profile}`) and saved to {}{expiry}{refreshable}.",
                        store.path().display()
                    );
                    0
                }
                Err(e) => {
                    eprintln!("auth login failed: {e}");
                    1
                }
            }
        }
        AuthCommands::Refresh { provider } => {
            let (canonical_id, entry) = if let Some((id, e)) = catalog.find_provider(provider) {
                (id.to_string(), e.clone())
            } else {
                eprintln!("auth error: unknown provider `{provider}`");
                return 1;
            };
            let profile = entry.oauth_profile(&canonical_id).to_string();
            match carnot::refresh_stored_token(&profile, entry.oauth.as_ref(), &store).await {
                Ok(tok) => {
                    let expiry = tok
                        .seconds_until_expiry()
                        .map(|s| format!(" (valid for {s}s)"))
                        .unwrap_or_default();
                    println!(
                        "Refreshed OAuth token for `{canonical_id}` (profile `{profile}`){expiry}."
                    );
                    0
                }
                Err(e) => {
                    eprintln!("auth refresh failed: {e}");
                    1
                }
            }
        }
        AuthCommands::Logout { provider } => {
            let profile = catalog.find_provider(provider).map_or_else(
                || provider.clone(),
                |(id, e)| e.oauth_profile(id).to_string(),
            );
            match store.remove(&profile) {
                Ok(true) => {
                    println!(
                        "Removed stored credentials for `{profile}` from {}.",
                        store.path().display()
                    );
                    0
                }
                Ok(false) => {
                    println!("No stored credentials found for `{profile}`.");
                    0
                }
                Err(e) => {
                    eprintln!("auth logout failed: {e}");
                    1
                }
            }
        }
    }
}

/// Resolves a one-shot prompt from `--prompt-file`, `-p`, and/or piped `stdin`.
///
/// Supports combining piped stdin with an explicit `-p` instruction (e.g. `git diff | carnot -p "review"`),
/// standalone piped stdin (`cat err.log | carnot`), explicit `-p -`, and `--image` without text.
fn resolve_one_shot_prompt(
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

fn detect_image_mime(path: &std::path::Path, bytes: &[u8]) -> Option<&'static str> {
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

fn load_image_blocks(paths: &[PathBuf]) -> Result<Vec<ContentBlock>, String> {
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
fn find_latest_session_for_workspace(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_prompt_combines_flag_and_piped_stdin() {
        let cli = Cli::try_parse_from(["carnot", "-p", "review this diff"]).expect("parse");
        let combined = resolve_one_shot_prompt(&cli, false, || Ok("diff --git a/x b/x".into()))
            .expect("ok")
            .expect("some");
        assert_eq!(combined, "review this diff\n\ndiff --git a/x b/x");

        let tty = resolve_one_shot_prompt(&cli, true, || unreachable!("should not read tty"))
            .expect("ok")
            .expect("some");
        assert_eq!(tty, "review this diff");
    }

    #[test]
    fn resolve_prompt_reads_standalone_piped_stdin() {
        let cli = Cli::try_parse_from(["carnot"]).expect("parse");
        let piped = resolve_one_shot_prompt(&cli, false, || Ok("fix cargo test\n".into()))
            .expect("ok")
            .expect("some");
        assert_eq!(piped, "fix cargo test\n");

        let empty_tty = resolve_one_shot_prompt(&cli, true, || unreachable!()).expect("ok");
        assert!(empty_tty.is_none());
    }

    #[test]
    fn cli_flags_override_user_config_values() {
        let mut cli = Cli::try_parse_from(["carnot", "--provider", "openai", "--effort", "low"])
            .expect("cli");
        let cfg = config::UserConfig {
            provider: Some("anthropic".into()),
            model: Some("claude-sonnet-5-5".into()),
            effort: Some("max".into()),
            cache_ttl: Some("5m".into()),
            sandbox: Some(false),
            ..Default::default()
        };
        cli.apply_user_config(&cfg).expect("apply");
        assert_eq!(cli.provider.as_deref(), Some("openai"));
        assert_eq!(cli.model.as_deref(), Some("claude-sonnet-5-5"));
        assert!(matches!(cli.effective_effort(), CliEffort::Low));
        assert_eq!(cli.effective_cache_ttl(), "5m");
        assert!(cli.no_sandbox);
    }

    #[test]
    fn detect_image_mime_recognizes_magic_bytes() {
        let png = b"\x89PNG\r\n\x1a\nrest";
        assert_eq!(
            detect_image_mime(std::path::Path::new("blob.bin"), png),
            Some("image/png")
        );
        let jpg = &[0xFF, 0xD8, 0xFF, 0xE0];
        assert_eq!(
            detect_image_mime(std::path::Path::new("blob"), jpg),
            Some("image/jpeg")
        );
        assert_eq!(
            detect_image_mime(std::path::Path::new("notes.txt"), b"hello"),
            None
        );
    }
}
