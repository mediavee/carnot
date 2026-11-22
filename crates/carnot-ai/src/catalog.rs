//! Declarative 3-layer Provider & Model data catalog (`providers.json`).
//!
//! Separation of concerns (Opus Phase 5 contract):
//!   1. `api`      = wire protocol adapter (Rust code registered by protocol name ONLY)
//!   2. `provider` = pure data (`base_url`, `auth_chain`, `headers`, `betas`, `compat`, `usage_map`, `body_patch`)
//!   3. `model`    = pure data (`pricing`, `effort_map`, capability flags, `compat_override`)
//!
//! Default catalog is embedded from `catalog/providers.json` via `include_str!` and can be
//! layered/overridden at runtime by `~/.carnot/providers.json` and explicit `--provider-config`.

use crate::compat::json_merge_patch;
use carnot_core::{CarnotError, ModelPricing, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Embedded default provider + model catalog (pure data, zero hardcoded Rust provider branches).
pub const EMBEDDED_PROVIDERS_JSON: &str = include_str!("../catalog/providers.json");

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderCatalog {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderEntry>,
    #[serde(default)]
    pub models: BTreeMap<String, ModelEntry>,
}

fn default_schema_version() -> u32 {
    1
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderEntry {
    /// Wire protocol adapter ID (e.g. `anthropic-messages`, `openai-completions`,
    /// `openai-responses`, `google-generative`, `google-cloudcode`, `subprocess-jsonl`, `faux`).
    pub api: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub fallback_base_urls: Vec<String>,
    #[serde(default)]
    pub base_url_envs: Vec<String>,
    #[serde(default)]
    pub project_env: Option<String>,
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default)]
    pub auth_chain: Vec<AuthEntry>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub betas: Vec<String>,
    #[serde(default)]
    pub cache_diagnosis_beta: Option<String>,
    #[serde(default)]
    pub native_tool_search: bool,
    #[serde(default)]
    pub compat: CompatFlags,
    #[serde(default)]
    pub usage_map: UsageMap,
    #[serde(default)]
    pub body_patch: Option<Value>,
    #[serde(default)]
    pub subprocess: Option<SubprocessProviderConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AuthScheme {
    Bearer,
    XApiKey,
    Header,
    Query,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthEntry {
    pub scheme: AuthScheme,
    #[serde(default)]
    pub env: Option<String>,
    #[serde(default)]
    pub header_name: Option<String>,
    #[serde(default)]
    pub query_param: Option<String>,
    /// Optional secret command executed lazily (e.g. `agentsafe get ...`). NEVER stores raw secrets in JSON!
    #[serde(default)]
    pub cmd: Option<Vec<String>>,
}

#[derive(Clone)]
pub enum ResolvedAuth {
    None,
    Bearer {
        token: String,
        source: String,
    },
    Header {
        name: String,
        value: String,
        source: String,
    },
    Query {
        param: String,
        value: String,
        source: String,
    },
}

impl std::fmt::Debug for ResolvedAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => f.write_str("ResolvedAuth::None"),
            Self::Bearer { source, .. } => f
                .debug_struct("ResolvedAuth::Bearer")
                .field("token", &"<redacted>")
                .field("source", source)
                .finish(),
            Self::Header { name, source, .. } => f
                .debug_struct("ResolvedAuth::Header")
                .field("name", name)
                .field("value", &"<redacted>")
                .field("source", source)
                .finish(),
            Self::Query { param, source, .. } => f
                .debug_struct("ResolvedAuth::Query")
                .field("param", param)
                .field("value", &"<redacted>")
                .field("source", source)
                .finish(),
        }
    }
}

fn percent_encode_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

impl ResolvedAuth {
    /// Appends URL-encoded query-parameter credentials when `ResolvedAuth::Query` is active.
    pub fn apply_to_url(&self, url: &str) -> String {
        match self {
            Self::Query { param, value, .. } => {
                let sep = if url.contains('?') { '&' } else { '?' };
                let p = percent_encode_component(param);
                let v = percent_encode_component(value);
                format!("{url}{sep}{p}={v}")
            }
            _ => url.to_string(),
        }
    }

    pub fn apply_to_request(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self {
            Self::None | Self::Query { .. } => req,
            Self::Bearer { token, .. } => req.bearer_auth(token),
            Self::Header { name, value, .. } => req.header(name.as_str(), value.as_str()),
        }
    }

    pub fn source_label(&self) -> &str {
        match self {
            Self::None => "none",
            Self::Bearer { source, .. }
            | Self::Header { source, .. }
            | Self::Query { source, .. } => source,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubprocessProviderConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default = "default_cancel_kill_ms")]
    pub cancel_kill_timeout_ms: u64,
}

fn default_cancel_kill_ms() -> u64 {
    2_000
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompatFlags {
    #[serde(default)]
    pub max_tokens_field: Option<String>,
    #[serde(default)]
    pub supports_developer_role: Option<bool>,
    #[serde(default)]
    pub supports_reasoning_effort: Option<bool>,
    #[serde(default)]
    pub reasoning_object_effort: Option<bool>,
    #[serde(default)]
    pub requires_reasoning_content_on_assistant_messages: Option<bool>,
    #[serde(default)]
    pub supports_prompt_cache_key: Option<bool>,
    #[serde(default)]
    pub send_stream_options: Option<bool>,
    #[serde(default)]
    pub session_id_header: Option<String>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub include_encrypted_reasoning: Option<bool>,
    /// Gemini/CloudCode thinking style: `"level"` (Gemini 3/4) or `"budget"` (Gemini 2.5).
    #[serde(default)]
    pub thinking_style: Option<String>,
    #[serde(default)]
    pub include_thoughts: Option<bool>,
    #[serde(default)]
    pub sanitize_tool_schemas: Option<bool>,
    #[serde(default)]
    pub envelope_request_type: Option<String>,
    #[serde(default)]
    pub envelope_user_agent: Option<String>,
    #[serde(default)]
    pub system_preamble: Option<String>,
}

impl CompatFlags {
    pub fn merge_override(&self, ov: &CompatFlags) -> Self {
        Self {
            max_tokens_field: ov
                .max_tokens_field
                .clone()
                .or_else(|| self.max_tokens_field.clone()),
            supports_developer_role: ov.supports_developer_role.or(self.supports_developer_role),
            supports_reasoning_effort: ov
                .supports_reasoning_effort
                .or(self.supports_reasoning_effort),
            reasoning_object_effort: ov.reasoning_object_effort.or(self.reasoning_object_effort),
            requires_reasoning_content_on_assistant_messages: ov
                .requires_reasoning_content_on_assistant_messages
                .or(self.requires_reasoning_content_on_assistant_messages),
            supports_prompt_cache_key: ov
                .supports_prompt_cache_key
                .or(self.supports_prompt_cache_key),
            send_stream_options: ov.send_stream_options.or(self.send_stream_options),
            session_id_header: ov
                .session_id_header
                .clone()
                .or_else(|| self.session_id_header.clone()),
            store: ov.store.or(self.store),
            include_encrypted_reasoning: ov
                .include_encrypted_reasoning
                .or(self.include_encrypted_reasoning),
            thinking_style: ov
                .thinking_style
                .clone()
                .or_else(|| self.thinking_style.clone()),
            include_thoughts: ov.include_thoughts.or(self.include_thoughts),
            sanitize_tool_schemas: ov.sanitize_tool_schemas.or(self.sanitize_tool_schemas),
            envelope_request_type: ov
                .envelope_request_type
                .clone()
                .or_else(|| self.envelope_request_type.clone()),
            envelope_user_agent: ov
                .envelope_user_agent
                .clone()
                .or_else(|| self.envelope_user_agent.clone()),
            system_preamble: ov
                .system_preamble
                .clone()
                .or_else(|| self.system_preamble.clone()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMap {
    #[serde(default = "default_input_ptr")]
    pub input_ptr: String,
    #[serde(default = "default_output_ptr")]
    pub output_ptr: String,
    #[serde(default = "default_cache_read_ptr")]
    pub cache_read_ptr: Option<String>,
    #[serde(default = "default_reasoning_ptr")]
    pub reasoning_ptr: Option<String>,
    #[serde(default = "default_true")]
    pub cached_included_in_input: bool,
    #[serde(default = "default_true")]
    pub reasoning_included_in_output: bool,
}

fn default_input_ptr() -> String {
    "/prompt_tokens".into()
}
fn default_output_ptr() -> String {
    "/completion_tokens".into()
}
fn default_cache_read_ptr() -> Option<String> {
    Some("/prompt_tokens_details/cached_tokens".into())
}
fn default_reasoning_ptr() -> Option<String> {
    Some("/completion_tokens_details/reasoning_tokens".into())
}
fn default_true() -> bool {
    true
}

impl Default for UsageMap {
    fn default() -> Self {
        Self {
            input_ptr: default_input_ptr(),
            output_ptr: default_output_ptr(),
            cache_read_ptr: default_cache_read_ptr(),
            reasoning_ptr: default_reasoning_ptr(),
            cached_included_in_input: true,
            reasoning_included_in_output: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelEntry {
    #[serde(default)]
    pub api: Option<String>,
    #[serde(default)]
    pub pricing: Option<ModelPricing>,
    #[serde(default)]
    pub context_window: Option<u64>,
    #[serde(default)]
    pub supports_mid_system_role: Option<bool>,
    #[serde(default)]
    pub supports_effort: Option<bool>,
    #[serde(default)]
    pub supports_xhigh: Option<bool>,
    #[serde(default)]
    pub needs_adaptive_thinking: Option<bool>,
    /// Anthropic extended thinking mode from data catalog: `"adaptive"`, `"budget"`, or `"none"`.
    #[serde(default)]
    pub thinking_mode: Option<String>,
    #[serde(default)]
    pub native_tool_search: Option<bool>,
    #[serde(default)]
    pub compat_override: Option<CompatFlags>,
    #[serde(default)]
    pub effort_map: BTreeMap<String, Value>,
    #[serde(default)]
    pub body_patch: Option<Value>,
}

/// Fully resolved, concrete specification passed to an `ApiAdapter::build`.
#[derive(Debug, Clone)]
pub struct ResolvedProviderSpec {
    pub provider_id: String,
    pub api: String,
    pub base_url: String,
    pub fallback_base_urls: Vec<String>,
    pub project_id: Option<String>,
    pub auth: ResolvedAuth,
    pub headers: BTreeMap<String, String>,
    pub betas: Vec<String>,
    pub cache_diagnosis_beta: Option<String>,
    pub native_tool_search: bool,
    pub compat: CompatFlags,
    pub usage_map: UsageMap,
    pub body_patch: Option<Value>,
    pub subprocess: Option<SubprocessProviderConfig>,
    pub catalog_models: BTreeMap<String, ModelEntry>,
}

impl ResolvedProviderSpec {
    pub fn model_entry(&self, model: &str) -> Option<&ModelEntry> {
        self.catalog_models.get(model).or_else(|| {
            // Prefix match (e.g. `claude-sonnet-5-5-20260901` matches `claude-sonnet-5-5`)
            self.catalog_models
                .iter()
                .filter(|(k, _)| model.starts_with(k.as_str()))
                .max_by_key(|(k, _)| k.len())
                .map(|(_, v)| v)
        })
    }

    pub fn effective_compat(&self, model: &str) -> CompatFlags {
        match self
            .model_entry(model)
            .and_then(|m| m.compat_override.as_ref())
        {
            Some(ov) => self.compat.merge_override(ov),
            None => self.compat.clone(),
        }
    }

    pub fn effective_pricing(&self, model: &str) -> ModelPricing {
        self.model_entry(model)
            .and_then(|m| m.pricing)
            .unwrap_or_else(|| ModelPricing::for_model(model))
    }

    pub fn apply_body_patches(&self, model: &str, body: &mut Value) {
        if let Some(ref patch) = self.body_patch {
            json_merge_patch(body, patch);
        }
        if let Some(patch) = self.model_entry(model).and_then(|m| m.body_patch.as_ref()) {
            json_merge_patch(body, patch);
        }
    }
}

/// Public summary emitted by `carnot providers` (NEVER includes secret values!).
#[derive(Debug, Clone, Serialize)]
pub struct ProviderSummary {
    pub id: String,
    pub api: String,
    pub aliases: Vec<String>,
    pub description: String,
    pub base_url: String,
    pub default_model: Option<String>,
    pub auth_env_vars: Vec<String>,
    pub configured: bool,
    pub auth_source: String,
}

impl ProviderCatalog {
    /// Loads the embedded default data catalog (`catalog/providers.json`), parsed once per process.
    pub fn embedded() -> Result<Self> {
        static EMBEDDED: std::sync::OnceLock<std::result::Result<ProviderCatalog, String>> =
            std::sync::OnceLock::new();
        let res = EMBEDDED.get_or_init(|| {
            serde_json::from_str(EMBEDDED_PROVIDERS_JSON)
                .map_err(|e| format!("failed to parse embedded providers.json: {e}"))
        });
        match res {
            Ok(cat) => Ok(cat.clone()),
            Err(msg) => Err(CarnotError::Provider {
                provider: "catalog".into(),
                message: msg.clone(),
                retryable: false,
                retry_after_ms: None,
            }),
        }
    }

    /// Loads embedded defaults, overlays `~/.carnot/providers.json` if present, then overlays
    /// any explicit `extra_paths` (e.g. `--provider-config`).
    pub fn load_layered(extra_paths: &[PathBuf]) -> Result<Self> {
        let mut catalog = Self::embedded()?;
        if let Some(home) = std::env::var_os("HOME") {
            let user_path = PathBuf::from(home).join(".carnot").join("providers.json");
            if user_path.is_file() {
                catalog.merge_file(&user_path)?;
            }
        }
        for p in extra_paths {
            catalog.merge_file(p)?;
        }
        Ok(catalog)
    }

    pub fn merge_file(&mut self, path: &Path) -> Result<()> {
        let raw = std::fs::read_to_string(path).map_err(|e| CarnotError::Provider {
            provider: "catalog".into(),
            message: format!("failed to read provider catalog {}: {e}", path.display()),
            retryable: false,
            retry_after_ms: None,
        })?;
        let overlay: ProviderCatalog =
            serde_json::from_str(&raw).map_err(|e| CarnotError::Provider {
                provider: "catalog".into(),
                message: format!("invalid JSON in provider catalog {}: {e}", path.display()),
                retryable: false,
                retry_after_ms: None,
            })?;
        self.merge(overlay);
        Ok(())
    }

    pub fn merge(&mut self, other: ProviderCatalog) {
        for (k, v) in other.providers {
            self.providers.insert(k, v);
        }
        for (k, v) in other.models {
            self.models.insert(k, v);
        }
    }

    /// Returns all unique credential environment variable names declared across the provider catalog.
    /// Used to scrub Carnot's own LLM credentials from unsandboxed child shells without hardcoding names.
    pub fn secret_env_vars(&self) -> Vec<String> {
        let mut set = std::collections::BTreeSet::new();
        for prov in self.providers.values() {
            for auth in &prov.auth_chain {
                if let Some(ref env_name) = auth.env
                    && !env_name.is_empty()
                {
                    set.insert(env_name.clone());
                }
            }
        }
        set.into_iter().collect()
    }

    /// Resolves a provider name OR alias (case-insensitive) to `(canonical_id, &ProviderEntry)`.
    pub fn find_provider(&self, name_or_alias: &str) -> Option<(&str, &ProviderEntry)> {
        let needle = name_or_alias.trim().to_ascii_lowercase();
        if let Some((k, v)) = self.providers.get_key_value(&needle) {
            return Some((k.as_str(), v));
        }
        for (k, v) in &self.providers {
            if k.eq_ignore_ascii_case(&needle)
                || v.aliases.iter().any(|a| a.eq_ignore_ascii_case(&needle))
            {
                return Some((k.as_str(), v));
            }
        }
        None
    }

    /// Prefix-aware model lookup matching `ResolvedProviderSpec::model_entry`.
    pub fn find_model(&self, model: &str) -> Option<&ModelEntry> {
        self.models.get(model).or_else(|| {
            self.models
                .iter()
                .filter(|(k, _)| model.starts_with(k.as_str()))
                .max_by_key(|(k, _)| k.len())
                .map(|(_, v)| v)
        })
    }

    /// Resolves base_url, auth credentials, project_id, and model override for a provider.
    pub fn resolve_spec(
        &self,
        name_or_alias: &str,
        model_hint: Option<&str>,
    ) -> Result<ResolvedProviderSpec> {
        let (canonical_id, entry) =
            self.find_provider(name_or_alias)
                .ok_or_else(|| CarnotError::Provider {
                    provider: name_or_alias.to_string(),
                    message: format!(
                        "unknown provider `{name_or_alias}` (configured providers: {})",
                        self.providers
                            .keys()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    retryable: false,
                    retry_after_ms: None,
                })?;

        let mut base_url = entry.base_url.clone();
        for env_name in &entry.base_url_envs {
            if let Ok(val) = std::env::var(env_name)
                && !val.trim().is_empty()
            {
                base_url = val.trim().to_string();
                break;
            }
        }

        let project_id = entry.project_env.as_deref().and_then(|env_name| {
            std::env::var(env_name)
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        });

        let auth = resolve_auth_chain(canonical_id, &entry.auth_chain)?;

        // Model-level API override (prefix-aware, consistent with `model_entry`!)
        let model_entry = model_hint.and_then(|m| self.find_model(m));
        let api = model_entry
            .and_then(|me| me.api.clone())
            .unwrap_or_else(|| entry.api.clone());

        let native_tool_search = model_entry
            .and_then(|me| me.native_tool_search)
            .unwrap_or(entry.native_tool_search);

        Ok(ResolvedProviderSpec {
            provider_id: canonical_id.to_string(),
            api,
            base_url: base_url.trim_end_matches('/').to_string(),
            fallback_base_urls: entry
                .fallback_base_urls
                .iter()
                .map(|u| u.trim_end_matches('/').to_string())
                .collect(),
            project_id,
            auth,
            headers: entry.headers.clone(),
            betas: entry.betas.clone(),
            cache_diagnosis_beta: entry.cache_diagnosis_beta.clone(),
            native_tool_search,
            compat: entry.compat.clone(),
            usage_map: entry.usage_map.clone(),
            body_patch: entry.body_patch.clone(),
            subprocess: entry.subprocess.clone(),
            catalog_models: self.models.clone(),
        })
    }

    /// Produces a secret-safe summary of all registered providers for `carnot providers`.
    pub fn summaries(&self) -> Vec<ProviderSummary> {
        self.providers
            .iter()
            .map(|(id, entry)| {
                let mut base_url = entry.base_url.clone();
                for env_name in &entry.base_url_envs {
                    if let Ok(val) = std::env::var(env_name)
                        && !val.trim().is_empty()
                    {
                        base_url = val.trim().to_string();
                        break;
                    }
                }
                let auth_env_vars: Vec<String> = entry
                    .auth_chain
                    .iter()
                    .filter_map(|a| a.env.clone())
                    .collect();
                let (configured, auth_source) = match probe_auth_chain_env_only(&entry.auth_chain) {
                    Some(src) => (true, src),
                    None => (false, "missing".into()),
                };
                ProviderSummary {
                    id: id.clone(),
                    api: entry.api.clone(),
                    aliases: entry.aliases.clone(),
                    description: entry.description.clone(),
                    base_url,
                    default_model: entry.default_model.clone(),
                    auth_env_vars,
                    configured,
                    auth_source,
                }
            })
            .collect()
    }
}

fn probe_auth_chain_env_only(chain: &[AuthEntry]) -> Option<String> {
    if chain.is_empty() {
        return Some("none".into());
    }
    for entry in chain {
        if matches!(entry.scheme, AuthScheme::None) {
            return Some("none".into());
        }
        if let Some(ref env_name) = entry.env
            && std::env::var(env_name)
                .map(|v| !v.trim().is_empty())
                .unwrap_or(false)
        {
            return Some(format!("env:{env_name}"));
        }
        if entry.cmd.as_ref().is_some_and(|c| !c.is_empty()) {
            return Some("cmd".into());
        }
    }
    None
}

fn resolve_auth_chain(provider_id: &str, chain: &[AuthEntry]) -> Result<ResolvedAuth> {
    if chain.is_empty() {
        return Ok(ResolvedAuth::None);
    }
    let mut tried_envs: Vec<String> = Vec::new();
    for entry in chain {
        if matches!(entry.scheme, AuthScheme::None) {
            return Ok(ResolvedAuth::None);
        }
        let secret = if let Some(ref env_name) = entry.env {
            tried_envs.push(env_name.clone());
            std::env::var(env_name)
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .map(|v| (v, format!("env:{env_name}")))
        } else {
            None
        }
        .or_else(|| {
            let cmd = entry.cmd.as_ref()?;
            let (prog, args) = cmd.split_first()?;
            let out = std::process::Command::new(prog).args(args).output().ok()?;
            if !out.status.success() {
                return None;
            }
            let token = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if token.is_empty() {
                None
            } else {
                Some((token, format!("cmd:{prog}")))
            }
        });

        if let Some((value, source)) = secret {
            tracing::debug!(
                provider = %provider_id,
                auth_source = %source,
                "resolved provider credentials"
            );
            return Ok(match entry.scheme {
                AuthScheme::Bearer => ResolvedAuth::Bearer {
                    token: value,
                    source,
                },
                AuthScheme::XApiKey => ResolvedAuth::Header {
                    name: "x-api-key".into(),
                    value,
                    source,
                },
                AuthScheme::Header => ResolvedAuth::Header {
                    name: entry
                        .header_name
                        .clone()
                        .unwrap_or_else(|| "x-api-key".into()),
                    value,
                    source,
                },
                AuthScheme::Query => ResolvedAuth::Query {
                    param: entry.query_param.clone().unwrap_or_else(|| "key".into()),
                    value,
                    source,
                },
                AuthScheme::None => ResolvedAuth::None,
            });
        }
    }

    Err(CarnotError::Provider {
        provider: provider_id.to_string(),
        message: format!(
            "missing credentials for provider `{provider_id}`; set one of: {}",
            tried_envs.join(", ")
        ),
        retryable: false,
        retry_after_ms: None,
    })
}
