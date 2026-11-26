//! CLI argument definitions, profiles, layered config merging, and extension flags.

use crate::config;
use carnot::{CacheTtl, Profile, ProviderRegistry, ReasoningEffort};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum Mode {
    Text,
    Json,
    Rpc,
    Acp,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum CliProfile {
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
pub(crate) enum CliEffort {
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
pub(crate) struct Cli {
    /// One-shot prompt text
    #[arg(short = 'p', long)]
    pub(crate) prompt: Option<String>,

    /// Read prompt from a file path (avoids exposing prompt text in /proc/<pid>/cmdline or `ARG_MAX` limits)
    #[arg(long = "prompt-file")]
    pub(crate) prompt_file: Option<PathBuf>,

    /// Attach image file(s) (`png`, `jpg`, `webp`, `gif`) to a one-shot prompt (repeatable)
    #[arg(short = 'i', long = "image")]
    pub(crate) images: Vec<PathBuf>,

    /// Execution surface mode
    #[arg(long, value_enum, default_value_t = Mode::Text)]
    pub(crate) mode: Mode,

    /// Use minimalist line-oriented plain REPL instead of the full-screen TUI
    #[arg(long)]
    pub(crate) plain: bool,

    /// Module profile preset (defaults to `carnot`, or `~/.carnot/config.toml`)
    #[arg(long, value_enum)]
    pub(crate) profile: Option<CliProfile>,

    /// Provider name or alias from providers.json (auto-detects first configured provider if omitted)
    #[arg(long)]
    pub(crate) provider: Option<String>,

    /// Model identifier (defaults to the selected provider's catalog `default_model`)
    #[arg(short = 'm', long)]
    pub(crate) model: Option<String>,

    /// Explicit `providers.json` override/plugin catalog path(s) (repeatable)
    #[arg(long = "provider-config")]
    pub(crate) provider_configs: Vec<PathBuf>,

    /// Reasoning effort level (low, medium, high, xhigh, max; defaults to `high` or `~/.carnot/config.toml`)
    #[arg(long, value_enum)]
    pub(crate) effort: Option<CliEffort>,

    /// Prompt cache retention tier (short, hybrid, or long; aliases: 5m, 1h, 24h)
    #[arg(long, alias = "cache-retention")]
    pub(crate) cache_ttl: Option<String>,

    /// Workspace directory (defaults to current directory)
    #[arg(short = 'C', long)]
    pub(crate) cwd: Option<PathBuf>,

    /// Resume the most recent session for this workspace
    #[arg(short = 'c', long = "continue", conflicts_with = "resume")]
    pub(crate) continue_last: bool,

    /// Resume an existing session JSONL file (or session ID under ~/.carnot/sessions)
    #[arg(long)]
    pub(crate) resume: Option<String>,

    /// With `--resume` or `--continue`: fork into a brand-new session file without modifying the original
    #[arg(long)]
    pub(crate) fork: bool,

    /// Disable loading global or workspace `.mcp.json` servers
    #[arg(long)]
    pub(crate) no_mcp: bool,

    /// Explicit `.mcp.json` config path(s) (repeatable)
    #[arg(long = "mcp-config")]
    pub(crate) mcp_configs: Vec<PathBuf>,

    /// Explicit Agent Skills directory path(s) containing `<name>/SKILL.md` (repeatable)
    #[arg(long = "skills-dir")]
    pub(crate) skills_dirs: Vec<PathBuf>,

    /// Explicitly trust workspace-local `.mcp.json` (disabled by default to prevent `MCPoison` RCE)
    #[arg(long)]
    pub(crate) trust_workspace_mcp: bool,

    /// Drop-in file extension path(s) (executable script/binary speaking `carnot-ext/1` JSON-RPC; repeatable)
    #[arg(long = "ext")]
    pub(crate) extensions: Vec<PathBuf>,

    /// Explicitly trust workspace-local `.carnot/extensions/` (disabled by default to prevent RCE)
    #[arg(long = "trust-workspace-ext")]
    pub(crate) trust_workspace_extensions: bool,

    /// Enable compiled-in Rust `carnot-ext-web` extension (enabled by default in standalone `carnot` profile)
    #[cfg(feature = "ext-web")]
    #[arg(long)]
    pub(crate) web: bool,

    /// Disable compiled-in `carnot-ext-web` extension
    #[cfg(feature = "ext-web")]
    #[arg(long = "no-web")]
    pub(crate) no_web: bool,

    /// Web extension mode: `auto` (prefer native provider search/fetch, fallback local), `native`, or `local`
    #[cfg(feature = "ext-web")]
    #[arg(long = "web-mode", default_value = "auto")]
    pub(crate) web_mode: carnot::WebMode,

    /// Restrict web search/fetch to specific hostname(s) (repeatable, e.g. `docs.rs`)
    #[cfg(feature = "ext-web")]
    #[arg(long = "web-allow-domain")]
    pub(crate) web_allow_domains: Vec<String>,

    /// Block web search/fetch from specific hostname(s) (repeatable)
    #[cfg(feature = "ext-web")]
    #[arg(long = "web-block-domain")]
    pub(crate) web_block_domains: Vec<String>,

    /// Enable compiled-in Rust `carnot-ext-lsp` extension (enabled by default in standalone `carnot` profile)
    #[cfg(feature = "ext-lsp")]
    #[arg(long)]
    pub(crate) lsp: bool,

    /// Disable compiled-in `carnot-ext-lsp` extension
    #[cfg(feature = "ext-lsp")]
    #[arg(long = "no-lsp")]
    pub(crate) no_lsp: bool,

    /// Enable disposable-context `Subagent` delegation extension (enabled by default in standalone `carnot` profile)
    #[cfg(feature = "ext-subagent")]
    #[arg(long)]
    pub(crate) subagent: bool,

    /// Disable compiled-in `Subagent` delegation extension
    #[cfg(feature = "ext-subagent")]
    #[arg(long = "no-subagent")]
    pub(crate) no_subagent: bool,

    /// Declarative Claude Code / Carnot hooks JSON file(s) (repeatable, requires `--features ext-hooks`)
    #[cfg(feature = "ext-hooks")]
    #[arg(long = "hooks-file")]
    pub(crate) hooks_files: Vec<PathBuf>,

    /// Also load user-global Claude Code hooks from `~/.claude/settings.json`
    #[cfg(feature = "ext-hooks")]
    #[arg(long = "claude-hooks")]
    pub(crate) claude_hooks: bool,

    /// Explicitly trust workspace-local `.carnot/hooks.json` and `.claude/settings*.json` (disabled by default to prevent RCE)
    #[cfg(feature = "ext-hooks")]
    #[arg(long = "trust-workspace-hooks")]
    pub(crate) trust_workspace_hooks: bool,

    /// Claude Code plugin directory path(s) (equivalent to `--plugin-dir`, repeatable, requires `--features ext-claude-plugins`)
    #[cfg(feature = "ext-claude-plugins")]
    #[arg(long = "claude-plugin")]
    pub(crate) claude_plugins: Vec<PathBuf>,

    /// Auto-discover installed enabled Claude Code plugins from `~/.claude/plugins/installed_plugins.json`
    #[cfg(feature = "ext-claude-plugins")]
    #[arg(long = "load-installed-claude-plugins")]
    pub(crate) load_installed_claude_plugins: bool,

    /// Enable cross-session persistent memory extension (enabled by default in standalone `carnot` profile)
    #[cfg(feature = "ext-memory")]
    #[arg(long)]
    pub(crate) memory: bool,

    /// Disable cross-session persistent memory extension
    #[cfg(feature = "ext-memory")]
    #[arg(long = "no-memory")]
    pub(crate) no_memory: bool,

    /// Custom root directory for persistent memory store (defaults to `~/.carnot/memory`)
    #[cfg(feature = "ext-memory")]
    #[arg(long = "memory-dir")]
    pub(crate) memory_dir: Option<PathBuf>,

    /// Enable isolated `QuickJS` `CodeMode` tool-orchestration extension (enabled by default in standalone `carnot` profile)
    #[cfg(feature = "ext-codemode")]
    #[arg(long)]
    pub(crate) codemode: bool,

    /// Disable isolated `QuickJS` `CodeMode` extension
    #[cfg(feature = "ext-codemode")]
    #[arg(long = "no-codemode")]
    pub(crate) no_codemode: bool,

    /// Natural-language goal criterion evaluated by an independent virgin-context verifier (`carnot-ext-goal`)
    #[cfg(feature = "ext-goal")]
    #[arg(long = "goal")]
    pub(crate) goal: Option<String>,

    /// Deterministic sandboxed verification shell command(s) for goal mode (repeatable, e.g. `cargo test`)
    #[cfg(feature = "ext-goal")]
    #[arg(long = "goal-cmd")]
    pub(crate) goal_cmds: Vec<String>,

    /// Maximum autonomous continuation iterations for `--goal` / `--goal-cmd` (default: 8)
    #[cfg(feature = "ext-goal")]
    #[arg(long = "goal-max-iter")]
    pub(crate) goal_max_iter: Option<u32>,

    /// Optional model/provider override for the independent goal verifier (e.g. `gemini/gemini-2.5-flash`)
    #[cfg(feature = "ext-goal")]
    #[arg(long = "goal-model")]
    pub(crate) goal_model: Option<String>,

    /// Enable enterprise `PolicyGuard` (command/path guardrails + secret redaction)
    #[arg(long)]
    pub(crate) policy_guard: bool,

    /// Explicitly disable OS kernel sandbox AND workspace filesystem confinement (fail-closed by default)
    #[arg(long)]
    pub(crate) no_sandbox: bool,

    /// Allow outbound network inside the OS sandbox (disabled by default)
    #[arg(long)]
    pub(crate) allow_network: bool,

    /// Do not persist session JSONL to disk
    #[arg(long)]
    pub(crate) no_session: bool,

    /// Explicit directory for session JSONL files (defaults to ~/.carnot/sessions)
    #[arg(long = "session-dir")]
    pub(crate) session_dir: Option<PathBuf>,

    /// Enable bidirectional host-extension handshake (`host/register`) on `--mode rpc`
    #[arg(long = "host-extension")]
    pub(crate) host_extension: bool,

    /// Override the static system prompt (frozen before Turn 1 for prompt-cache invariance)
    #[arg(long = "system-prompt")]
    pub(crate) system_prompt: Option<String>,

    /// Read static system prompt override from a file path (avoids `ARG_MAX` limits)
    #[arg(long = "system-prompt-file")]
    pub(crate) system_prompt_file: Option<PathBuf>,

    /// Exclude built-in tools by name before Turn 1 (comma-separated or repeatable, e.g. `Bash,Edit,Write`)
    #[arg(long = "exclude-tools", value_delimiter = ',')]
    pub(crate) exclude_tools: Vec<String>,

    /// Disable automatic Turn-1 `AGENTS.md`/`CLAUDE.md` discovery
    #[arg(long = "no-context-files")]
    pub(crate) no_context_files: bool,

    /// Disable automatic Turn-1 `SKILL.md` discovery
    #[arg(long = "no-skills")]
    pub(crate) no_skills: bool,

    /// Maximum agent turns before halting (default: 500)
    #[arg(long = "max-turns")]
    pub(crate) max_turns: Option<u64>,

    /// Hard USD cost cap circuit-breaker for this session (e.g. --max-cost-usd 2.50)
    #[arg(long = "max-cost-usd")]
    pub(crate) max_cost_usd: Option<f64>,

    /// Soft compaction threshold in estimated tokens (default: 220000; 0 disables soft compaction)
    #[arg(long = "compact-at")]
    pub(crate) compact_at: Option<u64>,

    /// Overall wall-clock timeout in seconds for one-shot runs (cancels cleanly on expiry)
    #[arg(long = "timeout")]
    pub(crate) timeout_secs: Option<u64>,

    /// Autonomous verification gate command(s) (repeatable)
    #[arg(long = "gate")]
    pub(crate) gates: Vec<String>,

    #[command(subcommand)]
    pub(crate) command: Option<Commands>,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum Commands {
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
pub(crate) enum McpCommands {
    /// List configured MCP servers & tool counts (uses zero-cold-start schema cache when warm)
    List,
    /// Inspect all discovered MCP tool specs (minified & canonicalized)
    Inspect,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum AuthCommands {
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

pub(crate) fn compiled_extensions() -> Vec<&'static str> {
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
        #[cfg(feature = "ext-codemode")]
        "carnot-ext-codemode",
        #[cfg(feature = "ext-goal")]
        "carnot-ext-goal",
    ];
    items.to_vec()
}

#[cfg(feature = "tui")]
pub(crate) fn active_extension_badges(cli: &Cli) -> Vec<String> {
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
    #[cfg(feature = "ext-codemode")]
    if cli.codemode_enabled() {
        out.push("codemode".to_string());
    }
    #[cfg(feature = "ext-goal")]
    if cli.goal.is_some() || !cli.goal_cmds.is_empty() {
        out.push("goal".to_string());
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
    pub(crate) fn effective_profile(&self) -> CliProfile {
        self.profile.unwrap_or(CliProfile::Carnot)
    }

    pub(crate) fn effective_effort(&self) -> CliEffort {
        self.effort.unwrap_or(CliEffort::High)
    }

    pub(crate) fn effective_cache_ttl(&self) -> &str {
        self.cache_ttl.as_deref().unwrap_or("hybrid")
    }

    /// Merges `~/.carnot/config.toml` defaults into `Cli` without overriding explicit CLI flags.
    pub(crate) fn apply_user_config(&mut self, cfg: &config::UserConfig) -> Result<(), String> {
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
        #[cfg(feature = "ext-codemode")]
        match cfg.extensions.codemode {
            Some(false) if !self.codemode => self.no_codemode = true,
            Some(true) if !self.no_codemode => self.codemode = true,
            _ => {}
        }
        Ok(())
    }

    /// Standalone `carnot` (not `--host-extension`) on the default `carnot` profile enables
    /// standard zero-prefix-cost extensions out of the box (opt out with `--no-*` or `--profile minimal`).
    #[cfg(any(
        feature = "ext-web",
        feature = "ext-lsp",
        feature = "ext-subagent",
        feature = "ext-memory",
        feature = "ext-codemode"
    ))]
    pub(crate) fn default_batteries(&self) -> bool {
        !self.host_extension && matches!(self.effective_profile(), CliProfile::Carnot)
    }

    #[cfg(feature = "ext-web")]
    pub(crate) fn web_enabled(&self) -> bool {
        !self.no_web && (self.web || self.default_batteries())
    }

    #[cfg(feature = "ext-lsp")]
    pub(crate) fn lsp_enabled(&self) -> bool {
        !self.no_lsp && (self.lsp || self.default_batteries())
    }

    #[cfg(feature = "ext-subagent")]
    pub(crate) fn subagent_enabled(&self) -> bool {
        !self.no_subagent && (self.subagent || self.default_batteries())
    }

    #[cfg(feature = "ext-memory")]
    pub(crate) fn memory_enabled(&self) -> bool {
        !self.no_memory && (self.memory || self.memory_dir.is_some() || self.default_batteries())
    }

    #[cfg(feature = "ext-codemode")]
    pub(crate) fn codemode_enabled(&self) -> bool {
        !self.no_codemode && (self.codemode || self.default_batteries())
    }

    /// Resolves `--provider`, or auto-selects the first configured cloud provider before falling
    /// back to `"anthropic"`.
    pub(crate) fn resolved_provider(&self, reg: Option<&ProviderRegistry>) -> String {
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
