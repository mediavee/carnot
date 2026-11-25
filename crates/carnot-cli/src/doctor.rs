//! Diagnostics (`carnot doctor`) and MCP server inspection (`carnot mcp list|inspect`).

use crate::args::{Cli, McpCommands, compiled_extensions};
use crate::config;
use carnot::{
    CasStore, McpManager, ProviderRegistry, assert_prefix_invariance, detect_sandbox_backend,
    discover_dropin_specs, load_mcp_configs,
};
use carnot_ext::microskills::discover_agent_skills;
use carnot_ext::prefix_guard::static_system_prompt;
use carnot_tools::canonical_tool_specs;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

pub(crate) fn run_doctor(cli: &Cli, cwd: &std::path::Path) -> i32 {
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

pub(crate) async fn run_mcp_cmd(cli: &Cli, cwd: &std::path::Path, action: &McpCommands) -> carnot::Result<()> {
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

