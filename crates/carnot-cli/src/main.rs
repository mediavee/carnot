//! `carnot` CLI binary: headless print/json, ACP stdio, Host RPC stdio, HTTP/SSE API server,
//! MCP 2026-07-28 / 2025-11-25 / Legacy SSE 2024-11-05 management (`carnot mcp list|inspect`),
//! prefix-invariance CI audit, production diagnostics (`carnot doctor`), session resume,
//! budget/timeout guardrails, and graceful signal handling.

mod args;
mod auth_cmd;
mod builder;
mod config;
mod doctor;
mod hosts;
mod repl;

#[cfg(feature = "tui")]
use crate::args::active_extension_badges;
use crate::args::{Cli, Commands, Mode, compiled_extensions};
use crate::auth_cmd::run_auth_cmd;
use crate::builder::{
    build_agent, build_agent_full, load_image_blocks, resolve_one_shot_prompt,
};
use crate::doctor::{run_doctor, run_mcp_cmd};
use crate::hosts::{CliAcpHost, CliRpcHost, list_rpc_models};
use carnot::{
    AcpHost, AgentEvent, ContentBlock, HostRpcBridge, HostRpcModule, ProviderRegistry, RpcHost,
    assert_prefix_invariance, read_host_registration, serve_acp_stdio_with_host, serve_http,
    serve_rpc_stdio_with_extension, serve_rpc_stdio_with_host, try_handle_landlock_exec_arg,
};
use carnot_ext::prefix_guard::static_system_prompt;
use carnot_tools::canonical_tool_specs;
use clap::Parser;
use serde_json::json;
use std::collections::BTreeMap;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::CliEffort;
    use crate::builder::detect_image_mime;

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
