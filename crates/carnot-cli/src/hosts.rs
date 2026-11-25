//! ACP and Host-RPC runtime host adapters (`CliAcpHost`, `CliRpcHost`).

use crate::args::{Cli, CliEffort};
use crate::builder::build_agent_with_mcp;
use carnot::{
    AcpHost, AcpModelInfo, McpManager, ProviderRegistry, ReasoningEffort, ResolvedAcpModel,
    ResolvedRpcModel, RpcCommandInfo, RpcHost, RpcModelInfo, parse_acp_mcp_servers,
};
use carnot_ext::microskills::discover_agent_skills;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct CliAcpHost {

    pub(crate) cli: Cli,
    pub(crate) session_mcps: tokio::sync::Mutex<Vec<Arc<McpManager>>>,
}

impl CliAcpHost {
    pub(crate) async fn shutdown_session_mcps(&self) {
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
pub(crate) struct CliRpcHost {
    pub(crate) cli: Cli,
    pub(crate) cwd: PathBuf,
}

pub(crate) fn list_rpc_models(reg: &ProviderRegistry, include_unconfigured: bool) -> Vec<RpcModelInfo> {
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

