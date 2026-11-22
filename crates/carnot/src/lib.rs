//! Ergonomic Rust SDK facade for embedding the Carnot next-generation agent harness.
//!
//! ```no_run
//! use carnot::{CarnotBuilder, Profile};
//!
//! # async fn demo() -> carnot::Result<()> {
//! let agent = CarnotBuilder::new(std::env::current_dir()?)
//!     .profile(Profile::Carnot)
//!     .model("claude-sonnet-5-5")
//!     .build()?;
//! let outcome = agent.prompt("Run tests and summarize any failures").await?;
//! println!("{}", outcome.final_text);
//! # Ok(())
//! # }
//! ```

pub use carnot_agent::{Agent, EventLog, EventStream, RunOutcome, SubagentModule};
pub use carnot_ai::{
    Provider, anthropic::AnthropicProvider, catalog::ProviderCatalog, catalog::ProviderSummary,
    catalog::ResolvedProviderSpec, faux::FauxProvider, faux::ScriptedTurn, google::GoogleProvider,
    openai::OpenAiProvider, openai_responses::OpenAiResponsesProvider, registry::ApiAdapter,
    registry::ProviderRegistry, retry::ResilientProvider, subprocess::SubprocessProvider,
};
pub use carnot_cas::{
    CasHandle, CasStore, WorktreeHasher, shape_command_output, shape_text_output,
};
pub use carnot_core::{
    AgentEvent, CacheTtl, CarnotConfig, CarnotError, EventEnvelope, Message, ModelPricing,
    ModuleSwitches, Profile, ReasoningEffort, Result, StopReason, ThermodynamicMetrics,
    ToolEffects, ToolOutput, ToolSpec, Usage, canonicalize_json,
};
pub use carnot_ext::{
    BoundaryAction, CompactionHookAction, CompactionHookContext, Module, PermissionBroker,
    ToolGateDecision, TurnBoundarySnapshot, daemon::MaxwellDaemonModule, lsp::LspDiagnosticsModule,
    microskills::ExecMicroSkillsModule, policy::PolicyGuardModule,
    prefix_guard::assert_prefix_invariance, rewind::RewindReceiptModule,
    subprocess::SubprocessExtension, warm_fork::IsolatedForkWorktree, web::WebFetchModule,
};
pub use carnot_mcp::{
    CachedServerSchema, CachedToolEntry, HttpServerConfig, MCP_LEGACY_PROTOCOL_VERSION,
    MCP_PROTOCOL_VERSION, McpClient, McpConfigFile, McpManager, McpServerConfig, McpServerSummary,
    ProtocolEra, StdioServerConfig, encode_mcp_header_value, load_mcp_configs,
};
pub use carnot_protocol::{
    acp::serve_acp_stdio, http::serve_http, http::serve_http_with_token, rpc::serve_rpc_stdio,
};
pub use carnot_tools::{Tool, ToolRegistry};

use std::path::PathBuf;
use std::sync::Arc;

/// Fluent builder assembling a Carnot `Agent` with custom providers, MCP 2026-07-28 / 2025-11-25 /
/// Legacy SSE 2024-11-05 servers, Agent Skills, modules (`Subagent`, `WebFetch`, `LspDiagnostics`,
/// `PolicyGuard`), extensions, and session resume.
#[derive(Debug)]
pub struct CarnotBuilder {
    config: CarnotConfig,
    explicit_model: bool,
    provider: Option<Arc<dyn Provider>>,
    extra_adapters: Vec<Arc<dyn ApiAdapter>>,
    extra_modules: Vec<Arc<dyn Module>>,
    extra_tools: Vec<Arc<dyn Tool>>,
    persist_session: bool,
    resume_session: Option<PathBuf>,
    policy_guard: bool,
}

impl CarnotBuilder {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            config: CarnotConfig::new(workspace_root),
            explicit_model: false,
            provider: None,
            extra_adapters: Vec::new(),
            extra_modules: Vec::new(),
            extra_tools: Vec::new(),
            persist_session: true,
            resume_session: None,
            policy_guard: false,
        }
    }

    pub fn profile(mut self, profile: Profile) -> Self {
        self.config.modules = ModuleSwitches::from_profile(profile);
        self
    }

    pub fn config(mut self, config: CarnotConfig) -> Self {
        self.config = config;
        self.explicit_model = true;
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.model = model.into();
        self.explicit_model = true;
        self
    }

    pub fn model_opt(mut self, model: Option<String>) -> Self {
        if let Some(m) = model {
            self.config.model = m;
            self.explicit_model = true;
        }
        self
    }

    pub fn max_turns(mut self, max_turns: u64) -> Self {
        self.config.max_turns = max_turns.max(1);
        self
    }

    pub fn max_cost_usd(mut self, max_cost_usd: f64) -> Self {
        if max_cost_usd.is_finite() && max_cost_usd > 0.0 {
            self.config.max_cost_usd = Some(max_cost_usd);
        }
        self
    }

    pub fn soft_compaction_tokens(mut self, tokens: Option<u64>) -> Self {
        self.config.soft_compaction_tokens = tokens.filter(|&t| t > 0);
        self
    }

    pub fn effort(mut self, effort: ReasoningEffort) -> Self {
        self.config.effort = effort;
        self
    }

    pub fn sandbox(mut self, enabled: bool) -> Self {
        self.config.modules.sandbox = enabled;
        self
    }

    pub fn sandbox_network(mut self, allow: bool) -> Self {
        self.config.modules.sandbox_network = allow;
        self
    }

    pub fn web_fetch(mut self, enabled: bool) -> Self {
        self.config.modules.web_fetch = enabled;
        self
    }

    pub fn lsp_diagnostics(mut self, enabled: bool) -> Self {
        self.config.modules.lsp_diagnostics = enabled;
        self
    }

    pub fn subagent(mut self, enabled: bool) -> Self {
        self.config.modules.subagent = enabled;
        self
    }

    pub fn policy_guard(mut self, enabled: bool) -> Self {
        self.policy_guard = enabled;
        self
    }

    pub fn mcp_config(mut self, path: PathBuf) -> Self {
        self.config.mcp_config_paths.push(path);
        self
    }

    pub fn skills_dir(mut self, path: PathBuf) -> Self {
        self.config.skills_dirs.push(path);
        self
    }

    pub fn trust_workspace_mcp(mut self, trust: bool) -> Self {
        self.config.trust_workspace_mcp = trust;
        self
    }

    pub fn resume_session(mut self, path: PathBuf) -> Self {
        self.resume_session = Some(path);
        self
    }

    pub fn extra_writable_roots(mut self, roots: Vec<PathBuf>) -> Self {
        self.config.extra_writable_roots = roots;
        self
    }

    pub fn provider_name(mut self, provider: impl Into<String>) -> Self {
        self.config.provider = provider.into();
        self
    }

    pub fn provider_config(mut self, path: PathBuf) -> Self {
        self.config.provider_config_paths.push(path);
        self
    }

    /// Registers a custom Rust wire-protocol plugin (`ApiAdapter`) by protocol `api_id`.
    pub fn with_api_adapter(mut self, adapter: Arc<dyn ApiAdapter>) -> Self {
        self.extra_adapters.push(adapter);
        self
    }

    pub fn cache_ttl(mut self, ttl: CacheTtl) -> Self {
        self.config.cache_ttl = ttl;
        self
    }

    pub fn persist_session(mut self, persist: bool) -> Self {
        self.persist_session = persist;
        self
    }

    pub fn autonomous_gates(mut self, gates: Vec<String>) -> Self {
        self.config.autonomous_gates = gates;
        self
    }

    pub fn provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn with_module(mut self, module: Arc<dyn Module>) -> Self {
        self.extra_modules.push(module);
        self
    }

    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.extra_tools.push(tool);
        self
    }

    /// Synchronous build (no MCP server connections; suitable for embedded Rust callers & tests).
    pub fn build(self) -> Result<Arc<Agent>> {
        let cas = Arc::new(CasStore::open(&self.config.workspace_root)?);
        self.build_with_mcp(cas, Arc::new(McpManager::empty()))
    }

    /// Async build loading all validated `.mcp.json` servers (with zero-cold-start lazy disk cache!).
    pub async fn build_async(self) -> Result<(Arc<Agent>, Arc<McpManager>)> {
        let cas = Arc::new(CasStore::open(&self.config.workspace_root)?);
        let cfgs = load_mcp_configs(
            &self.config.workspace_root,
            &self.config.mcp_config_paths,
            self.config.trust_workspace_mcp,
        )?;
        let mcp = if cfgs.is_empty() {
            Arc::new(McpManager::empty())
        } else {
            Arc::new(
                McpManager::from_configs(
                    cfgs,
                    self.config.workspace_root.clone(),
                    Arc::clone(&cas),
                    self.config.modules.cas_handles,
                    self.config.cas_spill_threshold_bytes,
                    self.config.cas_preview_max_bytes,
                )
                .await?,
            )
        };
        let agent = self.build_with_mcp(cas, Arc::clone(&mcp))?;
        Ok((agent, mcp))
    }

    fn build_with_mcp(mut self, cas: Arc<CasStore>, mcp: Arc<McpManager>) -> Result<Arc<Agent>> {
        let mut registry = ProviderRegistry::load_default(&self.config.provider_config_paths)?;
        for adapter in self.extra_adapters {
            registry.register_adapter(adapter);
        }
        if !self.explicit_model
            && let Some((_, entry)) = registry.catalog().find_provider(&self.config.provider)
            && let Some(ref default_model) = entry.default_model
        {
            self.config.model = default_model.clone();
        }
        if let Some(ctx_win) = registry.context_window_for_model(&self.config.model) {
            self.config.context_window = ctx_win;
        }
        if let Some(max_out) = registry.max_output_tokens_for_model(&self.config.model) {
            self.config.max_output_tokens = max_out;
        }
        if self.config.scrubbed_env_vars.is_empty() {
            self.config.scrubbed_env_vars = registry.catalog().secret_env_vars();
        }
        // Pre-authorize the deterministic read-only workspace CAS fallback root without touching disk at startup.
        let fallback_cas_root = CasStore::fallback_root(&self.config.workspace_root);
        if !self
            .config
            .extra_writable_roots
            .iter()
            .any(|r| r == &fallback_cas_root)
        {
            self.config.extra_writable_roots.push(fallback_cas_root);
        }
        let model_pricing = registry.pricing_for_model(&self.config.model);

        let mut tools = ToolRegistry::with_canonical_lid_tools(&self.config, Arc::clone(&cas));
        for t in mcp.tools() {
            tools.register(t);
        }
        for t in self.extra_tools {
            tools.register(t);
        }

        let provider: Arc<dyn Provider> = match self.provider {
            Some(p) => p,
            None => registry.build_provider(&self.config.provider, Some(&self.config.model))?,
        };

        let mut modules: Vec<Arc<dyn Module>> = Vec::new();
        let sw = &self.config.modules;
        let policy_mod: Option<Arc<dyn Module>> = self
            .policy_guard
            .then(|| Arc::new(PolicyGuardModule::new()) as Arc<dyn Module>);

        if sw.exec_microskills {
            modules.push(Arc::new(ExecMicroSkillsModule::new()));
        }
        if sw.maxwell_daemon {
            modules.push(Arc::new(MaxwellDaemonModule::new(
                self.config.workspace_root.clone(),
                self.config.idle_turn_circuit_breaker,
            )));
        }
        if sw.rewind_receipt {
            modules.push(Arc::new(RewindReceiptModule::new(
                self.config.workspace_root.clone(),
                model_pricing,
                self.config.cache_ttl,
                self.config.rewind_min_heat_tokens,
                self.config.rewind_max_per_session,
                self.config.rewind_assumed_reread_frac,
            )));
        }
        if sw.web_fetch {
            modules.push(Arc::new(WebFetchModule::new(
                Arc::clone(&cas),
                sw.cas_handles,
                self.config.cas_spill_threshold_bytes,
                self.config.cas_preview_max_bytes,
            )?));
        }
        if sw.lsp_diagnostics {
            modules.push(Arc::new(LspDiagnosticsModule::new(
                self.config.workspace_root.clone(),
                400,
            )));
        }
        if sw.subagent {
            // Propagate ALL custom modules + PolicyGuard into child subagents (Opus #2!)
            let mut inherited_for_child = self.extra_modules.clone();
            if let Some(ref pm) = policy_mod {
                inherited_for_child.push(Arc::clone(pm));
            }
            modules.push(Arc::new(SubagentModule::new(
                self.config.clone(),
                Arc::clone(&provider),
                inherited_for_child,
            )));
        }
        modules.extend(self.extra_modules);
        // Always run PolicyGuard LAST so `RewriteInput` from earlier modules can never bypass it!
        if let Some(pm) = policy_mod {
            modules.push(pm);
        }

        let agent = if let Some(ref resume_path) = self.resume_session {
            Agent::resume(resume_path, self.config, provider, tools, modules)?
        } else {
            Agent::new(self.config, provider, tools, modules, self.persist_session)?
        };
        Ok(Arc::new(agent))
    }
}
