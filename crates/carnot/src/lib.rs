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

pub use carnot_agent::{Agent, EventLog, EventStream, RunOutcome};
pub use carnot_ai::{
    Provider,
    anthropic::AnthropicProvider,
    catalog::ProviderCatalog,
    catalog::ProviderSummary,
    catalog::ResolvedProviderSpec,
    faux::FauxProvider,
    faux::ScriptedTurn,
    google::GoogleProvider,
    oauth::{
        AuthStore, OAuthConfig, OAuthFlow, OAuthLoginOptions, OAuthPrompt, OAuthStoredToken,
        OAuthTokenManager, login_with_oauth, refresh_stored_token, save_manual_token,
    },
    openai::OpenAiProvider,
    openai_responses::OpenAiResponsesProvider,
    registry::ApiAdapter,
    registry::AvailableModel,
    registry::ProviderRegistry,
    registry::ResolvedModelTarget,
    retry::ResilientProvider,
    subprocess::SubprocessProvider,
};
pub use carnot_cas::{
    CasHandle, CasStore, redact_secrets, shape_command_output, shape_command_output_with_redaction,
    shape_text_output,
};
pub use carnot_core::{
    AgentEvent, CacheTtl, CarnotConfig, CarnotError, ContextBreakdown, EventEnvelope, Message,
    ModelPricing, ModuleSwitches, NativeWebConfig, Profile, ProviderWebCapabilities,
    RateLimitWindow, ReasoningEffort, Result, StopReason, ThermodynamicMetrics, ToolEffects,
    ToolOutput, ToolSpec, Usage, UserQuestion, UserQuestionAnswer, UserQuestionOption,
    canonicalize_json,
};
pub use carnot_ext::{
    BoundaryAction, CompactionHookAction, CompactionHookContext, Module, PermissionBroker,
    PromptInjection, ToolGateDecision, TurnBoundarySnapshot, UserQuestionBroker,
    ask_user::{AskUserModule, SharedQuestionBroker},
    dropin::{
        DropinExtension, DropinExtensionSpec, FailurePolicy, discover_dropin_specs,
        load_dropin_extensions,
    },
    microskills::ExecMicroSkillsModule,
    policy::PolicyGuardModule,
    prefix_guard::assert_prefix_invariance,
    warm_fork::{IsolatedForkWorktree, LandOutcome},
};
#[cfg(feature = "ext-claude-plugins")]
pub use carnot_ext_claude_plugins::{
    ClaudePluginsConfig, ClaudePluginsModule, LoadedClaudePlugin, PluginAsset,
    discover_installed_claude_plugin_dirs,
};
#[cfg(feature = "ext-hooks")]
pub use carnot_ext_hooks::{
    CompiledHook, HookEvent, HooksConfig, HooksModule, parse_hooks_file, parse_hooks_value,
};
#[cfg(feature = "ext-lsp")]
pub use carnot_ext_lsp::{LspDiagnosticsModule, LspLang};
#[cfg(feature = "ext-memory")]
pub use carnot_ext_memory::{MemoryConfig, MemoryEntry, MemoryKind, MemoryModule, MemoryScope};
#[cfg(feature = "ext-subagent")]
pub use carnot_ext_subagent::SubagentModule;
#[cfg(feature = "ext-web")]
pub use carnot_ext_web::{ResolvedWebExtension, WebConfig, WebMode, WebModule};
pub use carnot_mcp::{
    CachedServerSchema, CachedToolEntry, HttpServerConfig, MCP_LEGACY_PROTOCOL_VERSION,
    MCP_PROTOCOL_VERSION, McpClient, McpConfigFile, McpManager, McpServerConfig, McpServerSummary,
    ProtocolEra, StdioServerConfig, encode_mcp_header_value, load_mcp_configs,
    parse_acp_mcp_servers,
};
pub use carnot_protocol::{
    acp::AcpHost,
    acp::AcpModelInfo,
    acp::ResolvedAcpModel,
    acp::serve_acp_stdio,
    acp::serve_acp_stdio_with_host,
    host_ext::{HostRegistration, HostRpcBridge, HostRpcModule, HostToolDecl},
    http::serve_http,
    http::serve_http_with_token,
    interaction::ChannelInteractionBroker,
    rpc::{
        PendingHostSession, ResolvedRpcModel, RpcCommandInfo, RpcHost, RpcModelInfo,
        read_host_registration, serve_rpc_stdio, serve_rpc_stdio_with_extension,
        serve_rpc_stdio_with_host,
    },
};
pub use carnot_sandbox::{
    SandboxBackendInfo, detect_sandbox_backend, try_handle_landlock_exec_arg,
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
    extra_mcp_servers: std::collections::BTreeMap<String, McpServerConfig>,
    disable_mcp: bool,
    persist_session: bool,
    resume_session: Option<PathBuf>,
    fork_session: bool,
    policy_guard: bool,
    #[cfg(feature = "ext-web")]
    web_config: Option<WebConfig>,
    #[cfg(feature = "ext-lsp")]
    lsp_enabled: bool,
    #[cfg(feature = "ext-hooks")]
    hooks_config: Option<HooksConfig>,
    #[cfg(feature = "ext-claude-plugins")]
    claude_plugins_config: Option<ClaudePluginsConfig>,
    #[cfg(feature = "ext-memory")]
    memory_config: Option<MemoryConfig>,
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
            extra_mcp_servers: std::collections::BTreeMap::new(),
            disable_mcp: false,
            persist_session: true,
            resume_session: None,
            fork_session: false,
            policy_guard: false,
            #[cfg(feature = "ext-web")]
            web_config: None,
            #[cfg(feature = "ext-lsp")]
            lsp_enabled: false,
            #[cfg(feature = "ext-hooks")]
            hooks_config: None,
            #[cfg(feature = "ext-claude-plugins")]
            claude_plugins_config: None,
            #[cfg(feature = "ext-memory")]
            memory_config: None,
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

    #[cfg(feature = "ext-subagent")]
    pub fn subagent(mut self, enabled: bool) -> Self {
        self.config.modules.subagent = enabled;
        self
    }

    pub fn policy_guard(mut self, enabled: bool) -> Self {
        self.policy_guard = enabled;
        self
    }

    /// Registers the deferred interactive `AskUser` tool (`defer_loading: true`, 0 static tokens).
    pub fn ask_user(mut self, enabled: bool) -> Self {
        self.config.enable_ask_user = enabled;
        self
    }

    pub fn mcp_config(mut self, path: PathBuf) -> Self {
        self.config.mcp_config_paths.push(path);
        self
    }

    /// Merges programmatically supplied MCP server configs (e.g. from ACP `session/new` `mcpServers`).
    pub fn extra_mcp_servers(
        mut self,
        servers: std::collections::BTreeMap<String, McpServerConfig>,
    ) -> Self {
        self.extra_mcp_servers.extend(servers);
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

    /// Adds an out-of-process drop-in file extension path (loaded via JSON-RPC stdio in `build_async`).
    pub fn extension(mut self, path: PathBuf) -> Self {
        self.config.extension_paths.push(path);
        self
    }

    /// Opts into loading workspace-local `.carnot/extensions/` (disabled by default to block RCE).
    pub fn trust_workspace_extensions(mut self, trust: bool) -> Self {
        self.config.trust_workspace_extensions = trust;
        self
    }

    /// Enables or disables the official web extension in `WebMode::Auto` (native server tools when
    /// supported by the active provider, automatic fallback to local `WebFetch`/`WebSearch`).
    #[cfg(feature = "ext-web")]
    pub fn web_fetch(mut self, enabled: bool) -> Self {
        self.web_config = enabled.then(WebConfig::default);
        self
    }

    /// Enables the official web extension with a specific mode (`Auto`, `Native`, or `Local`).
    #[cfg(feature = "ext-web")]
    pub fn web_mode(mut self, mode: WebMode) -> Self {
        let mut cfg = self.web_config.unwrap_or_default();
        cfg.mode = mode;
        self.web_config = Some(cfg);
        self
    }

    /// Configures the official web extension (mode, domain policy, token/use caps).
    #[cfg(feature = "ext-web")]
    pub fn web_config(mut self, cfg: WebConfig) -> Self {
        self.web_config = Some(cfg);
        self
    }

    #[cfg(feature = "ext-lsp")]
    pub fn lsp_diagnostics(mut self, enabled: bool) -> Self {
        self.lsp_enabled = enabled;
        self
    }

    /// Configures Claude Code-compatible declarative hooks (`PreToolUse`, `PostToolUse`, `SessionStart`, `PreCompact`, `Stop`).
    #[cfg(feature = "ext-hooks")]
    pub fn hooks_config(mut self, cfg: HooksConfig) -> Self {
        self.hooks_config = Some(cfg);
        self
    }

    /// Adds a Claude Code plugin directory (equivalent to Claude Code `--plugin-dir`).
    #[cfg(feature = "ext-claude-plugins")]
    pub fn claude_plugin(mut self, dir: PathBuf) -> Self {
        self.claude_plugins_config
            .get_or_insert_with(ClaudePluginsConfig::default)
            .plugin_dirs
            .push(dir);
        self
    }

    /// Configures Claude Code plugin discovery (explicit dirs and/or `~/.claude/plugins/installed_plugins.json`).
    #[cfg(feature = "ext-claude-plugins")]
    pub fn claude_plugins_config(mut self, cfg: ClaudePluginsConfig) -> Self {
        self.claude_plugins_config = Some(cfg);
        self
    }

    /// Enables or disables cross-session persistent memory (`MemoryModule`) with default `~/.carnot/memory` storage.
    #[cfg(feature = "ext-memory")]
    pub fn memory(mut self, enabled: bool) -> Self {
        self.memory_config = enabled.then(MemoryConfig::default);
        self
    }

    /// Configures cross-session persistent memory (`MemoryModule`).
    #[cfg(feature = "ext-memory")]
    pub fn memory_config(mut self, cfg: MemoryConfig) -> Self {
        self.memory_config = Some(cfg);
        self
    }

    pub fn resume_session(mut self, path: PathBuf) -> Self {
        self.resume_session = Some(path);
        self
    }

    /// When paired with `resume_session(...)`, forks the resumed session into a brand-new JSONL
    /// session file so the original session stays untouched while keeping the warm cache prefix.
    pub fn fork_session(mut self, enabled: bool) -> Self {
        self.fork_session = enabled;
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

    /// Sets an explicit session JSONL directory (defaults to `~/.carnot/sessions`).
    pub fn session_dir(mut self, dir: PathBuf) -> Self {
        self.config.session_dir = dir;
        self
    }

    /// Overrides the default static system prompt (frozen before Turn 1 for cache invariance).
    pub fn system_prompt_override(mut self, prompt: Option<String>) -> Self {
        self.config.static_system_prompt_override = prompt;
        self
    }

    /// Appends a frozen static section to the system prompt before Turn 1.
    pub fn system_prompt_append(mut self, section: Option<String>) -> Self {
        self.config.static_system_prompt_append = section;
        self
    }

    /// Excludes named built-in tools (case-insensitive) from `ToolRegistry` before Turn 1.
    pub fn exclude_tools(mut self, excluded: Vec<String>) -> Self {
        self.config.excluded_tools = excluded;
        self
    }

    /// Controls whether Turn-1 micro-skills auto-load `AGENTS.md`/`CLAUDE.md` (default `true`).
    pub fn load_context_files(mut self, enabled: bool) -> Self {
        self.config.load_context_files = enabled;
        self
    }

    /// Controls whether Turn-1 micro-skills auto-discover `SKILL.md` directories (default `true`).
    pub fn load_skills(mut self, enabled: bool) -> Self {
        self.config.load_skills = enabled;
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

    /// Disables loading all global/workspace `.mcp.json` servers for this session.
    pub fn disable_mcp(mut self, disable: bool) -> Self {
        self.disable_mcp = disable;
        self
    }

    /// Synchronous build (no MCP server connections; suitable for embedded Rust callers & tests).
    pub fn build(self) -> Result<Arc<Agent>> {
        let cas = Arc::new(CasStore::open(&self.config.workspace_root)?);
        self.build_with_mcp(cas, Arc::new(McpManager::empty()))
    }

    /// Async build loading all validated `.mcp.json` servers and drop-in file extensions.
    pub async fn build_async(mut self) -> Result<(Arc<Agent>, Arc<McpManager>)> {
        let cas = Arc::new(CasStore::open(&self.config.workspace_root)?);
        let mut cfgs = if self.disable_mcp {
            std::collections::BTreeMap::new()
        } else {
            load_mcp_configs(
                &self.config.workspace_root,
                &self.config.mcp_config_paths,
                self.config.trust_workspace_mcp,
            )?
        };
        if !self.disable_mcp {
            cfgs.extend(self.extra_mcp_servers.clone());
        }
        #[cfg(feature = "ext-claude-plugins")]
        if let Some(plugin_cfg) = self.claude_plugins_config.take() {
            let plugin_mod = ClaudePluginsModule::load(&self.config.workspace_root, &plugin_cfg)?;
            if !plugin_mod.is_empty() {
                cfgs.extend(plugin_mod.mcp_servers().clone());
                self.extra_modules.push(Arc::new(plugin_mod));
            }
        }
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
        let dropins = load_dropin_extensions(
            &self.config.workspace_root,
            &self.config.extension_paths,
            self.config.trust_workspace_extensions,
        )
        .await?;
        self.extra_modules.extend(dropins);
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
        #[cfg(feature = "ext-web")]
        if let Some(ref web_cfg) = self.web_config {
            let caps = provider.web_capabilities(&self.config.model);
            let resolved = WebModule::resolve(
                web_cfg,
                caps,
                Arc::clone(&cas),
                sw.cas_handles,
                self.config.cas_spill_threshold_bytes,
                self.config.cas_preview_max_bytes,
            )?;
            self.config.native_web = resolved.native_web;
            if !resolved.module.tools().is_empty() {
                modules.push(Arc::new(resolved.module));
            }
        }
        #[cfg(feature = "ext-lsp")]
        if self.lsp_enabled {
            modules.push(Arc::new(LspDiagnosticsModule::new(
                self.config.workspace_root.clone(),
                400,
            )));
        }
        #[cfg(feature = "ext-memory")]
        if let Some(mem_cfg) = self.memory_config {
            modules.push(Arc::new(MemoryModule::open(
                &self.config.workspace_root,
                mem_cfg,
            )));
        }
        #[cfg(feature = "ext-hooks")]
        let mut hooks_mod: Option<Arc<dyn Module>> = None;
        #[cfg(feature = "ext-hooks")]
        if let Some(ref hooks_cfg) = self.hooks_config {
            let hm = HooksModule::discover(&self.config.workspace_root, hooks_cfg)?;
            if !hm.is_empty() {
                let hm: Arc<dyn Module> = Arc::new(hm);
                modules.push(Arc::clone(&hm));
                hooks_mod = Some(hm);
            }
        }
        #[cfg(feature = "ext-claude-plugins")]
        let mut claude_plugins_mod: Option<Arc<dyn Module>> = None;
        #[cfg(feature = "ext-claude-plugins")]
        if let Some(ref plugin_cfg) = self.claude_plugins_config {
            let pm = ClaudePluginsModule::load(&self.config.workspace_root, plugin_cfg)?;
            if !pm.is_empty() {
                let pm: Arc<dyn Module> = Arc::new(pm);
                modules.push(Arc::clone(&pm));
                claude_plugins_mod = Some(pm);
            }
        }
        #[cfg(feature = "ext-subagent")]
        let subagent_mod: Option<Arc<SubagentModule>> = if sw.subagent {
            // Propagate Hooks, Claude plugins, custom modules, and PolicyGuard into child workers
            // so fail-closed PreToolUse hooks and policy gates can NEVER be bypassed via workers!
            let mut inherited_for_child: Vec<Arc<dyn Module>> = Vec::new();
            #[cfg(feature = "ext-hooks")]
            if let Some(ref hm) = hooks_mod {
                inherited_for_child.push(Arc::clone(hm));
            }
            #[cfg(feature = "ext-claude-plugins")]
            if let Some(ref pm) = claude_plugins_mod {
                inherited_for_child.push(Arc::clone(pm));
            }
            inherited_for_child.extend(self.extra_modules.iter().cloned());
            if let Some(ref pm) = policy_mod {
                inherited_for_child.push(Arc::clone(pm));
            }
            let sm = Arc::new(
                SubagentModule::new(
                    self.config.clone(),
                    Arc::clone(&provider),
                    inherited_for_child,
                )
                .with_provider_configs(self.config.provider_config_paths.clone()),
            );
            modules.push(sm.clone());
            Some(sm)
        } else {
            None
        };
        modules.extend(self.extra_modules);
        // Always run PolicyGuard LAST so `RewriteInput` from earlier modules can never bypass it!
        if let Some(pm) = policy_mod {
            modules.push(pm);
        }

        let agent = if let Some(ref resume_path) = self.resume_session {
            if self.fork_session {
                Agent::resume_forked(
                    resume_path,
                    self.config,
                    provider,
                    tools,
                    modules,
                    self.persist_session,
                )?
            } else {
                Agent::resume(resume_path, self.config, provider, tools, modules)?
            }
        } else {
            Agent::new(self.config, provider, tools, modules, self.persist_session)?
        };
        let descs = mcp.server_descriptions();
        if !descs.is_empty() {
            agent.set_deferred_server_descriptions(descs);
        }
        let agent = Arc::new(agent);
        #[cfg(feature = "ext-subagent")]
        if let Some(ref sm) = subagent_mod {
            sm.attach_parent(&agent);
        }
        Ok(agent)
    }
}
