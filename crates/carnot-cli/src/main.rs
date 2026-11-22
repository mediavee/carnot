//! `carnot` CLI binary: headless print/json, ACP stdio, Host RPC stdio, HTTP/SSE API server,
//! MCP 2026-07-28 / 2025-11-25 / Legacy SSE 2024-11-05 management (`carnot mcp list|inspect`),
//! prefix-invariance CI audit, production diagnostics (`carnot doctor`), session resume,
//! budget/timeout guardrails, graceful signal handling, and micro-benchmarks.

mod bench;
mod repl;

use carnot::{
    AcpHost, AcpModelInfo, AgentEvent, CacheTtl, CarnotBuilder, CasStore, McpManager, Profile,
    ProviderRegistry, ReasoningEffort, ResolvedAcpModel, assert_prefix_invariance,
    load_mcp_configs, serve_acp_stdio_with_host, serve_http, serve_rpc_stdio,
};
use carnot_ext::microskills::discover_agent_skills;
use carnot_ext::prefix_guard::static_system_prompt;
use carnot_tools::canonical_tool_specs;
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::json;
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
    about = "Carnot: Next-Generation Thermodynamic Coding Agent Harness in Rust"
)]
struct Cli {
    /// One-shot prompt text
    #[arg(short = 'p', long)]
    prompt: Option<String>,

    /// Read prompt from a file path (avoids exposing prompt text in /proc/<pid>/cmdline or ARG_MAX limits)
    #[arg(long = "prompt-file")]
    prompt_file: Option<PathBuf>,

    /// Execution surface mode
    #[arg(long, value_enum, default_value_t = Mode::Text)]
    mode: Mode,

    /// Module profile preset
    #[arg(long, value_enum, default_value_t = CliProfile::Carnot)]
    profile: CliProfile,

    /// Provider name or alias from providers.json (run `carnot providers` to list all)
    #[arg(long, default_value = "anthropic")]
    provider: String,

    /// Model identifier (defaults to the selected provider's catalog default_model)
    #[arg(short = 'm', long)]
    model: Option<String>,

    /// Explicit `providers.json` override/plugin catalog path(s) (repeatable)
    #[arg(long = "provider-config")]
    provider_configs: Vec<PathBuf>,

    /// Reasoning effort level (low, medium, high, max)
    #[arg(long, value_enum, default_value_t = CliEffort::High)]
    effort: CliEffort,

    /// Prompt cache TTL (hybrid, 5m, or 1h)
    #[arg(long, default_value = "hybrid")]
    cache_ttl: String,

    /// Workspace directory (defaults to current directory)
    #[arg(short = 'C', long)]
    cwd: Option<PathBuf>,

    /// Resume an existing session JSONL file (or session ID under ~/.carnot/sessions)
    #[arg(long)]
    resume: Option<String>,

    /// Explicit `.mcp.json` config path(s) (repeatable)
    #[arg(long = "mcp-config")]
    mcp_configs: Vec<PathBuf>,

    /// Explicit Agent Skills directory path(s) containing `<name>/SKILL.md` (repeatable)
    #[arg(long = "skills-dir")]
    skills_dirs: Vec<PathBuf>,

    /// Explicitly trust workspace-local `.mcp.json` (disabled by default to prevent MCPoison RCE)
    #[arg(long)]
    trust_workspace_mcp: bool,

    /// Enable opt-in SSRF-safe `WebFetch` module
    #[arg(long)]
    web: bool,

    /// Enable opt-in Tier-1 LSP diagnostics module (`rust-analyzer`/`tsserver`/`pyright`/`gopls`)
    #[arg(long)]
    lsp: bool,

    /// Enable disposable-context `Subagent` delegation module
    #[arg(long)]
    subagent: bool,

    /// Enable enterprise `PolicyGuard` (command/path guardrails + secret redaction)
    #[arg(long)]
    policy_guard: bool,

    /// Explicitly disable OS kernel sandbox (fail-closed by default)
    #[arg(long)]
    no_sandbox: bool,

    /// Allow outbound network inside the OS sandbox (disabled by default)
    #[arg(long)]
    allow_network: bool,

    /// Do not persist session JSONL to disk
    #[arg(long)]
    no_session: bool,

    /// Maximum agent turns before halting (default: 500)
    #[arg(long = "max-turns")]
    max_turns: Option<u64>,

    /// Hard USD cost cap circuit-breaker for this session (e.g. --max-cost-usd 2.50)
    #[arg(long = "max-cost-usd")]
    max_cost_usd: Option<f64>,

    /// Soft thermodynamic compaction threshold in estimated tokens (default: 220000; 0 disables soft compaction)
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
    /// List active modules and canonical LID tool schemas
    Modules,
    /// List registered wire-protocol adapters & data-driven providers from providers.json
    Providers,
    /// Manage & inspect configured MCP 2026-07-28 servers and deferred tools
    Mcp {
        #[command(subcommand)]
        action: McpCommands,
    },
    /// Run reproducible multi-suite benchmarks (micro, mcp 2026-07-28 vs 2025-11-25, replay ablation)
    Bench {
        /// Repository path to benchmark against (defaults to current directory)
        #[arg(default_value = ".")]
        repo: PathBuf,
        /// Suite to run: all, micro, mcp, or replay
        #[arg(long, default_value = "all")]
        suite: String,
    },
}

#[derive(Debug, Clone, Subcommand)]
enum McpCommands {
    /// List configured MCP servers & tool counts (uses zero-cold-start schema cache when warm)
    List,
    /// Inspect all discovered MCP tool specs (minified & canonicalized)
    Inspect,
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
    init_tracing_if_requested();

    let cli = Cli::parse();
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
                let sw = carnot::ModuleSwitches::from_profile(cli.profile.into());
                let out = json!({
                    "profile": format!("{:?}", cli.profile).to_lowercase(),
                    "modules": sw,
                    "canonical_tools": canonical_tool_specs().iter().map(|t| &t.name).collect::<Vec<_>>(),
                    "optional_modules": ["web_fetch", "lsp_diagnostics", "subagent", "policy_guard", "mcp_2026_07_28"],
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
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

async fn async_main(cli: Cli, cwd: PathBuf) -> i32 {
    if let Some(ref cmd) = cli.command {
        match cmd {
            Commands::Doctor => {
                return run_doctor(&cli, &cwd).await;
            }
            Commands::Mcp { action } => {
                if let Err(e) = run_mcp_cmd(&cli, &cwd, action).await {
                    eprintln!("mcp error: {e}");
                    return 1;
                }
                return 0;
            }
            Commands::Bench { repo, suite } => {
                let s = match suite.to_ascii_lowercase().as_str() {
                    "micro" => bench::BenchSuite::Micro,
                    "mcp" => bench::BenchSuite::Mcp,
                    "replay" => bench::BenchSuite::Replay,
                    _ => bench::BenchSuite::All,
                };
                bench::run_bench_suite(repo, s).await;
                return 0;
            }
            Commands::Serve { addr } => {
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
            Commands::AuditPrefix | Commands::Modules | Commands::Providers => return 0,
        }
    }

    let (agent, mcp) = match build_agent(&cli, cwd).await {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };

    let code = match cli.mode {
        Mode::Acp => {
            let host = Arc::new(CliAcpHost { cli: cli.clone() });
            match serve_acp_stdio_with_host(agent, host).await {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("acp error: {e}");
                    1
                }
            }
        }
        Mode::Rpc => match serve_rpc_stdio(agent).await {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("rpc error: {e}");
                1
            }
        },
        Mode::Text | Mode::Json => {
            let prompt = if let Some(ref pf) = cli.prompt_file {
                match std::fs::read_to_string(pf) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("error reading --prompt-file {}: {e}", pf.display());
                        mcp.shutdown_all().await;
                        return 2;
                    }
                }
            } else if cli.prompt.as_deref() == Some("-") {
                use std::io::Read;
                let mut buf = String::new();
                if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
                    eprintln!("error reading prompt from stdin: {e}");
                    mcp.shutdown_all().await;
                    return 2;
                }
                buf
            } else if let Some(p) = cli.prompt {
                p
            } else if matches!(cli.mode, Mode::Text) && std::io::stdin().is_terminal() {
                let rc = repl::run_repl(agent).await;
                mcp.shutdown_all().await;
                return rc;
            } else {
                eprintln!(
                    "usage: carnot [-p \"<prompt>\" | --prompt-file <path>] | carnot doctor | carnot --mode acp|rpc | carnot serve | carnot mcp list | carnot bench"
                );
                mcp.shutdown_all().await;
                return 2;
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

            let run_res = if let Some(secs) = cli.timeout_secs {
                match tokio::time::timeout(Duration::from_secs(secs.max(1)), agent.prompt(prompt))
                    .await
                {
                    Ok(res) => res,
                    Err(_) => {
                        agent.abort().await;
                        Err(carnot::CarnotError::Tool {
                            tool: "cli".into(),
                            message: format!("session timed out after {secs}s"),
                        })
                    }
                }
            } else {
                agent.prompt(prompt).await
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
            _ = async {
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

async fn run_doctor(cli: &Cli, cwd: &std::path::Path) -> i32 {
    let prefix = assert_prefix_invariance(|_date, _cwd, _sid| {
        (static_system_prompt().to_string(), canonical_tool_specs())
    });
    let reg = ProviderRegistry::load_default(&cli.provider_configs);
    let (provider_status, selected_model, context_window) = match &reg {
        Ok(r) => {
            let canonical_id = r
                .catalog()
                .find_provider(&cli.provider)
                .map(|(id, _)| id.to_string())
                .unwrap_or_else(|| cli.provider.clone());
            let summary = r
                .summaries()
                .into_iter()
                .find(|s| s.id.eq_ignore_ascii_case(&canonical_id));
            let model = cli.model.clone().unwrap_or_else(|| {
                r.catalog()
                    .find_provider(&cli.provider)
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
                None => json!({ "id": cli.provider, "error": "unknown provider" }),
            };
            (st, model, ctx)
        }
        Err(e) => (
            json!({ "error": e.to_string() }),
            cli.model.clone().unwrap_or_default(),
            200_000,
        ),
    };

    let sandbox_backend = if cfg!(target_os = "macos") {
        let ok = std::path::Path::new("/usr/bin/sandbox-exec").is_file();
        json!({ "backend": "seatbelt", "available": ok })
    } else if cfg!(target_os = "linux") {
        let ok = std::process::Command::new("bwrap")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        json!({ "backend": "bwrap", "available": ok })
    } else {
        json!({ "backend": "none", "available": false })
    };

    let mcp_cfgs = load_mcp_configs(cwd, &cli.mcp_configs, cli.trust_workspace_mcp);
    let mcp_info = match mcp_cfgs {
        Ok(cfgs) => json!({ "configured_servers": cfgs.keys().collect::<Vec<_>>() }),
        Err(e) => json!({ "error": e.to_string() }),
    };

    let skills = discover_agent_skills(cwd, &cli.skills_dirs);
    let skills_names: Vec<&str> = skills.iter().map(|(n, _, _)| n.as_str()).collect();

    let out = json!({
        "status": if prefix.is_ok() { "ok" } else { "degraded" },
        "version": env!("CARGO_PKG_VERSION"),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "workspace": cwd,
        "prefix_invariance": prefix.as_ref().map(|h| json!({"pass": true, "blake3": h})).unwrap_or_else(|e| json!({"pass": false, "error": e.to_string()})),
        "provider": provider_status,
        "model": selected_model,
        "context_window": context_window,
        "sandbox": sandbox_backend,
        "mcp": mcp_info,
        "skills": { "count": skills_names.len(), "names": skills_names },
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    if prefix.is_ok() { 0 } else { 1 }
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
}

impl AcpHost for CliAcpHost {
    fn available_models(&self) -> Vec<AcpModelInfo> {
        let Ok(reg) = ProviderRegistry::load_default(&self.cli.provider_configs) else {
            return Vec::new();
        };
        let cat = reg.catalog();
        let has_labeled = cat
            .models
            .values()
            .any(|e| e.provider.is_some() && e.label.is_some());
        // Advertise models that explicitly declare a provider (skipping unlabeled catch-all prefix
        // entries when labeled models are present), ordered by optional `order` then ID.
        let mut items: Vec<(u32, AcpModelInfo)> = cat
            .models
            .iter()
            .filter_map(|(id, entry)| {
                let prov_id = entry.provider.as_deref()?;
                if has_labeled && entry.label.is_none() {
                    return None;
                }
                // Only advertise models whose declared provider exists in the catalog
                let (_, prov_entry) = cat.find_provider(prov_id)?;
                let name = entry.label.clone().unwrap_or_else(|| id.clone());
                let desc = if !prov_entry.description.is_empty() {
                    Some(prov_entry.description.clone())
                } else {
                    Some(prov_id.to_string())
                };
                Some((
                    entry.order.unwrap_or(500),
                    AcpModelInfo {
                        id: id.clone(),
                        name,
                        description: desc,
                    },
                ))
            })
            .collect();
        items.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.id.cmp(&b.1.id)));
        items.into_iter().map(|(_, info)| info).collect()
    }

    fn resolve_model(&self, model_id: &str) -> carnot::Result<ResolvedAcpModel> {
        let reg = ProviderRegistry::load_default(&self.cli.provider_configs)?;
        let prov_id = reg
            .catalog()
            .provider_for_model(model_id)
            .unwrap_or(&self.cli.provider);
        let provider = reg.build_provider(prov_id, Some(model_id))?;
        let context_window = reg.context_window_for_model(model_id);
        let max_output_tokens = reg.max_output_tokens_for_model(model_id);
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
    ) -> carnot_ext::BoxFuture<'_, carnot::Result<Arc<carnot::Agent>>> {
        let mut eff_cli = self.cli.clone();
        if let Some(m) = model {
            eff_cli.model = Some(m);
        }
        if let Some(e) = effort {
            eff_cli.effort = match e {
                ReasoningEffort::Low => CliEffort::Low,
                ReasoningEffort::Medium => CliEffort::Medium,
                ReasoningEffort::High => CliEffort::High,
                ReasoningEffort::Xhigh => CliEffort::Xhigh,
                ReasoningEffort::Max => CliEffort::Max,
            };
        }
        Box::pin(async move {
            let (agent, _mcp) = build_agent(&eff_cli, cwd).await?;
            Ok(agent)
        })
    }
}

async fn build_agent(
    cli: &Cli,
    cwd: PathBuf,
) -> carnot::Result<(Arc<carnot::Agent>, Arc<McpManager>)> {
    let ttl = match cli.cache_ttl.to_ascii_lowercase().as_str() {
        "5m" | "short" => CacheTtl::Short5m,
        "1h" | "long" => CacheTtl::Long1h,
        _ => CacheTtl::Hybrid,
    };
    let effective_provider = if let Some(ref m) = cli.model
        && cli.provider == "anthropic"
        && let Ok(reg) = ProviderRegistry::load_default(&cli.provider_configs)
        && let Some(routed) = reg.catalog().provider_for_model(m)
    {
        routed.to_string()
    } else {
        cli.provider.clone()
    };
    let mut b = CarnotBuilder::new(cwd.clone())
        .profile(cli.profile.into())
        .model_opt(cli.model.clone())
        .effort(cli.effort.into())
        .provider_name(effective_provider)
        .cache_ttl(ttl)
        .persist_session(!cli.no_session)
        .autonomous_gates(cli.gates.clone())
        .web_fetch(cli.web)
        .lsp_diagnostics(cli.lsp)
        .subagent(cli.subagent)
        .policy_guard(cli.policy_guard)
        .trust_workspace_mcp(cli.trust_workspace_mcp);

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
    for d in &cli.skills_dirs {
        b = b.skills_dir(d.clone());
    }
    if let Some(ref res) = cli.resume {
        let p = PathBuf::from(res);
        let resolved = if p.is_file() {
            p
        } else {
            let sdir = std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join(".carnot").join("sessions"))
                .unwrap_or_else(|| cwd.join(".carnot").join("sessions"));
            if res.ends_with(".jsonl") {
                sdir.join(res)
            } else {
                sdir.join(format!("{res}.jsonl"))
            }
        };
        b = b.resume_session(resolved);
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
