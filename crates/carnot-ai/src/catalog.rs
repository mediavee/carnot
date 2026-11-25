//! Declarative 3-layer Provider & Model data catalog (`providers.json`).
//!
//! Separation of concerns:
//!   1. `api`      = wire protocol adapter (Rust code registered by protocol name ONLY)
//!   2. `provider` = pure data (`base_url`, `auth_chain`, `headers`, `betas`, `compat`, `usage_map`, `body_patch`)
//!   3. `model`    = pure data (`pricing`, `effort_map`, capability flags, `compat_override`)
//!
//! Default catalog is embedded from `catalog/providers.json` via `include_str!` and can be
//! layered/overridden at runtime by `~/.carnot/providers.json` and explicit `--provider-config`.

use crate::compat::json_merge_patch;
use crate::oauth::{AuthStore, OAuthConfig, OAuthTokenManager};
use carnot_core::{CarnotError, ModelPricing, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    /// Optional static URL query parameters appended to every request (e.g. `api-version`).
    #[serde(default)]
    pub query_params: BTreeMap<String, String>,
    #[serde(default)]
    pub betas: Vec<String>,
    #[serde(default)]
    pub cache_diagnosis_beta: Option<String>,
    #[serde(default)]
    pub native_tool_search: Option<bool>,
    #[serde(default)]
    pub compat: CompatFlags,
    #[serde(default)]
    pub usage_map: UsageMap,
    #[serde(default)]
    pub body_patch: Option<Value>,
    #[serde(default)]
    pub subprocess: Option<SubprocessProviderConfig>,
    /// Optional standards-compliant OAuth 2.0 configuration (`carnot auth login <provider>`).
    #[serde(default)]
    pub oauth: Option<OAuthConfig>,
    /// Declarative per-model-prefix routing rules for multi-protocol gateways and model-generation splits.
    #[serde(default)]
    pub model_routes: Vec<ModelRouteRule>,
}

/// Declarative per-model-prefix routing override inside a provider entry.
/// Matched longest-prefix-first against the target model ID (case-insensitive, delimiter-aware).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelRouteRule {
    #[serde(default)]
    pub prefixes: Vec<String>,
    #[serde(default)]
    pub api: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub auth_chain: Vec<AuthEntry>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub betas: Option<Vec<String>>,
    #[serde(default)]
    pub native_tool_search: Option<bool>,
    #[serde(default)]
    pub compat: Option<CompatFlags>,
    #[serde(default)]
    pub usage_map: Option<UsageMap>,
    #[serde(default)]
    pub body_patch: Option<Value>,
}

impl ProviderEntry {
    /// Returns the OAuth token store profile name for this provider (`auth_chain[].profile` if
    /// configured on an `oauth` entry, otherwise `canonical_id`).
    pub fn oauth_profile<'a>(&'a self, canonical_id: &'a str) -> &'a str {
        self.auth_chain
            .iter()
            .find(|a| matches!(a.scheme, AuthScheme::Oauth))
            .and_then(|a| a.profile.as_deref())
            .filter(|p| !p.trim().is_empty())
            .unwrap_or(canonical_id)
    }

    /// Returns the most specific `ModelRouteRule` matching `model` (longest matching prefix wins).
    pub fn match_model_route(&self, model: &str) -> Option<&ModelRouteRule> {
        let mut best: Option<(usize, &ModelRouteRule)> = None;
        for rule in &self.model_routes {
            for prefix in &rule.prefixes {
                if is_route_prefix_match(model, prefix)
                    && best.is_none_or(|(len, _)| prefix.len() > len)
                {
                    best = Some((prefix.len(), rule));
                }
            }
        }
        best.map(|(_, rule)| rule)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AuthScheme {
    Bearer,
    XApiKey,
    Header,
    Query,
    Oauth,
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
    /// Optional OAuth token profile name in `~/.carnot/auth.json` (defaults to canonical provider ID).
    #[serde(default)]
    pub profile: Option<String>,
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
    OAuth {
        manager: Arc<OAuthTokenManager>,
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
            Self::OAuth { source, .. } => f
                .debug_struct("ResolvedAuth::OAuth")
                .field("token", &"<redacted>")
                .field("source", source)
                .finish(),
        }
    }
}

fn percent_encode_component(s: &str) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            let _ = write!(out, "%{b:02X}");
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
            Self::None | Self::Query { .. } | Self::OAuth { .. } => req,
            Self::Bearer { token, .. } => req.bearer_auth(token),
            Self::Header { name, value, .. } => req.header(name.as_str(), value.as_str()),
        }
    }

    /// Async auth application that supports proactive & compare-and-swap OAuth token refresh.
    /// Returns the modified `RequestBuilder` and (for OAuth) the exact access token attached so
    /// an HTTP `401` retry can pass it as `rejected_token` to deduplicate concurrent refreshes.
    pub async fn apply_to_request_async(
        &self,
        req: reqwest::RequestBuilder,
        rejected_token: Option<&str>,
    ) -> Result<(reqwest::RequestBuilder, Option<String>)> {
        match self {
            Self::OAuth { manager, .. } => {
                let token = manager.access_token(rejected_token).await?;
                Ok((req.bearer_auth(&token), Some(token)))
            }
            other => Ok((other.apply_to_request(req), None)),
        }
    }

    /// True if this credential can be refreshed (e.g., OAuth with a stored `refresh_token`).
    pub fn supports_refresh(&self) -> bool {
        match self {
            Self::OAuth { manager, .. } => manager.supports_refresh(),
            _ => false,
        }
    }

    pub fn source_label(&self) -> &str {
        match self {
            Self::None => "none",
            Self::Bearer { source, .. }
            | Self::Header { source, .. }
            | Self::Query { source, .. }
            | Self::OAuth { source, .. } => source,
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
    /// `OpenAI` 24-hour prompt cache retention (`"prompt_cache_retention": "24h"`).
    #[serde(default)]
    pub supports_prompt_cache_retention: Option<bool>,
    /// `OpenAI` / Responses `xhigh` reasoning effort support.
    #[serde(default)]
    pub supports_xhigh_reasoning: Option<bool>,
    /// Anthropic `eager_input_streaming: true` on `Edit`/`Write` (default true on official Anthropic).
    #[serde(default)]
    pub supports_eager_input_streaming: Option<bool>,
    /// Anthropic signed-thinking requirement (default true on Claude; set false on unsigned gateways).
    #[serde(default)]
    pub requires_thinking_signature: Option<bool>,
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
    /// Provider + model natively supports server-side web search (Anthropic `web_search`,
    /// `OpenAI` Responses `web_search`, Gemini `googleSearch`).
    #[serde(default)]
    pub supports_native_web_search: Option<bool>,
    /// Provider + model natively supports server-side URL fetch (Anthropic `web_fetch`,
    /// `OpenAI` Responses `web_search`, Gemini `urlContext`).
    #[serde(default)]
    pub supports_native_web_fetch: Option<bool>,
    /// Optional server-side web search tool type override (Anthropic default: `web_search_20250305`).
    #[serde(default)]
    pub native_web_search_type: Option<String>,
    /// Optional server-side web fetch tool type override (Anthropic default: `web_fetch_20250910`).
    #[serde(default)]
    pub native_web_fetch_type: Option<String>,
    /// Optional beta header sent ONLY when native web fetch is active (Anthropic: `web-fetch-2025-09-10`).
    #[serde(default)]
    pub native_web_fetch_beta: Option<String>,
    /// OpenAI-compatible gateway supports Anthropic/Gemini `cache_control` breakpoints inside
    /// content-part arrays (e.g., `OpenRouter` routing to `anthropic/*` or `google/gemini*`).
    #[serde(default)]
    pub supports_cache_control: Option<bool>,
    /// Optional model-ID prefixes restricting when `supports_cache_control` is emitted (when empty/unset,
    /// applies to all models on that provider).
    #[serde(default)]
    pub cache_control_model_prefixes: Option<Vec<String>>,
    /// Emit `"ttl": "1h"` on static/Turn-1 `cache_control` breakpoints for `anthropic/*` models
    /// when `CacheTtl::Long | CacheTtl::Hybrid` is active.
    #[serde(default)]
    pub supports_cache_control_1h_ttl: Option<bool>,
    /// Optional wire-format constraint for tool call IDs (`"alnum9"` for strict 9-char alphanumeric
    /// IDs, `"sanitized"` for `[A-Za-z0-9_-]{1,64}`).
    #[serde(default)]
    pub tool_call_id_format: Option<String>,
    /// Emit `""` instead of `null` for empty assistant `content` on tool-calling messages.
    #[serde(default)]
    pub empty_assistant_content_as_empty_string: Option<bool>,
}

impl CompatFlags {
    #[must_use]
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
            supports_prompt_cache_retention: ov
                .supports_prompt_cache_retention
                .or(self.supports_prompt_cache_retention),
            supports_xhigh_reasoning: ov
                .supports_xhigh_reasoning
                .or(self.supports_xhigh_reasoning),
            supports_eager_input_streaming: ov
                .supports_eager_input_streaming
                .or(self.supports_eager_input_streaming),
            requires_thinking_signature: ov
                .requires_thinking_signature
                .or(self.requires_thinking_signature),
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
            supports_native_web_search: ov
                .supports_native_web_search
                .or(self.supports_native_web_search),
            supports_native_web_fetch: ov
                .supports_native_web_fetch
                .or(self.supports_native_web_fetch),
            native_web_search_type: ov
                .native_web_search_type
                .clone()
                .or_else(|| self.native_web_search_type.clone()),
            native_web_fetch_type: ov
                .native_web_fetch_type
                .clone()
                .or_else(|| self.native_web_fetch_type.clone()),
            native_web_fetch_beta: ov
                .native_web_fetch_beta
                .clone()
                .or_else(|| self.native_web_fetch_beta.clone()),
            supports_cache_control: ov.supports_cache_control.or(self.supports_cache_control),
            cache_control_model_prefixes: ov
                .cache_control_model_prefixes
                .clone()
                .or_else(|| self.cache_control_model_prefixes.clone()),
            supports_cache_control_1h_ttl: ov
                .supports_cache_control_1h_ttl
                .or(self.supports_cache_control_1h_ttl),
            tool_call_id_format: ov
                .tool_call_id_format
                .clone()
                .or_else(|| self.tool_call_id_format.clone()),
            empty_assistant_content_as_empty_string: ov
                .empty_assistant_content_as_empty_string
                .or(self.empty_assistant_content_as_empty_string),
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
    #[serde(default)]
    pub cache_write_ptr: Option<String>,
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
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default attribute requires fn() -> Option<String>"
)]
fn default_cache_read_ptr() -> Option<String> {
    Some("/prompt_tokens_details/cached_tokens".into())
}
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default attribute requires fn() -> Option<String>"
)]
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
            cache_write_ptr: None,
            reasoning_ptr: default_reasoning_ptr(),
            cached_included_in_input: true,
            reasoning_included_in_output: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Optional provider ID in `providers.json` that serves this model (enables automatic
    /// multi-provider model routing in CLI & ACP model pickers).
    #[serde(default)]
    pub provider: Option<String>,
    /// Optional human-friendly display label (used by ACP model pickers such as T3 Code).
    #[serde(default)]
    pub label: Option<String>,
    /// Optional ordering hint (lower appears first in ACP model pickers).
    #[serde(default)]
    pub order: Option<u32>,
    #[serde(default)]
    pub api: Option<String>,
    #[serde(default)]
    pub pricing: Option<ModelPricing>,
    #[serde(default)]
    pub context_window: Option<u64>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
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
    pub query_params: BTreeMap<String, String>,
    pub betas: Vec<String>,
    pub cache_diagnosis_beta: Option<String>,
    pub native_tool_search: bool,
    pub compat: CompatFlags,
    pub usage_map: UsageMap,
    pub body_patch: Option<Value>,
    pub subprocess: Option<SubprocessProviderConfig>,
    pub catalog_models: BTreeMap<String, ModelEntry>,
}

fn is_model_prefix_match(model: &str, prefix: &str) -> bool {
    if !model.starts_with(prefix) {
        return false;
    }
    // Require end-of-string or a real delimiter so e.g. `gpt-5.4` matches `gpt-5.4-20260901`
    // and `opencode-go` matches `opencode-go/deepseek-v4.1-flash`.
    matches!(
        model.as_bytes().get(prefix.len()),
        None | Some(b'-' | b'.' | b':' | b'@' | b'/')
    )
}

fn is_route_prefix_match(model: &str, prefix: &str) -> bool {
    if prefix.is_empty() || model.len() < prefix.len() {
        return false;
    }
    let Some(head) = model.get(..prefix.len()) else {
        return false;
    };
    if !head.eq_ignore_ascii_case(prefix) {
        return false;
    }
    if matches!(prefix.as_bytes().last(), Some(b'-' | b'/' | b'.' | b':')) {
        return true;
    }
    matches!(
        model.as_bytes().get(prefix.len()),
        None | Some(b'-' | b'.' | b':' | b'@' | b'/' | b'_')
    )
}

impl ResolvedProviderSpec {
    pub fn model_entry(&self, model: &str) -> Option<&ModelEntry> {
        self.catalog_models.get(model).or_else(|| {
            // Boundary-aware prefix match (e.g. `claude-sonnet-5-5-20260901` matches `claude-sonnet-5-5`)
            self.catalog_models
                .iter()
                .filter(|(k, _)| is_model_prefix_match(model, k.as_str()))
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

    /// Appends configured static `query_params` (e.g., `api-version`) and any query-based auth.
    pub fn build_url(&self, raw_url: &str) -> String {
        let mut out = raw_url.to_string();
        for (k, v) in &self.query_params {
            let sep = if out.contains('?') { '&' } else { '?' };
            out.push(sep);
            out.push_str(&percent_encode_component(k));
            out.push('=');
            out.push_str(&percent_encode_component(v));
        }
        self.auth.apply_to_url(&out)
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
    pub oauth_supported: bool,
}

impl ProviderCatalog {
    /// Loads the embedded default data catalog (`catalog/providers.json`), parsed once per process.
    pub fn embedded() -> Result<Self> {
        static EMBEDDED: std::sync::LazyLock<std::result::Result<ProviderCatalog, String>> =
            std::sync::LazyLock::new(|| {
                serde_json::from_str(EMBEDDED_PROVIDERS_JSON)
                    .map_err(|e| format!("failed to parse embedded providers.json: {e}"))
            });
        match &*EMBEDDED {
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
        for (k, mut v) in other.providers {
            if let Some(base) = self.providers.get(&k)
                && base.api == v.api
            {
                if v.aliases.is_empty() {
                    v.aliases.clone_from(&base.aliases);
                }
                if v.description.is_empty() {
                    v.description.clone_from(&base.description);
                }
                if v.base_url.is_empty() {
                    v.base_url.clone_from(&base.base_url);
                }
                if v.fallback_base_urls.is_empty() {
                    v.fallback_base_urls.clone_from(&base.fallback_base_urls);
                }
                if v.base_url_envs.is_empty() {
                    v.base_url_envs.clone_from(&base.base_url_envs);
                }
                if v.project_env.is_none() {
                    v.project_env.clone_from(&base.project_env);
                }
                if v.default_model.is_none() {
                    v.default_model.clone_from(&base.default_model);
                }
                if v.auth_chain.is_empty() {
                    v.auth_chain.clone_from(&base.auth_chain);
                }
                if !base.headers.is_empty() {
                    let mut merged_headers = base.headers.clone();
                    merged_headers.extend(v.headers);
                    v.headers = merged_headers;
                }
                if !base.query_params.is_empty() {
                    let mut merged_qp = base.query_params.clone();
                    merged_qp.extend(v.query_params);
                    v.query_params = merged_qp;
                }
                if v.betas.is_empty() {
                    v.betas.clone_from(&base.betas);
                }
                if v.cache_diagnosis_beta.is_none() {
                    v.cache_diagnosis_beta
                        .clone_from(&base.cache_diagnosis_beta);
                }
                v.native_tool_search = v.native_tool_search.or(base.native_tool_search);
                v.compat = base.compat.merge_override(&v.compat);
                if v.body_patch.is_none() {
                    v.body_patch.clone_from(&base.body_patch);
                }
                if v.subprocess.is_none() {
                    v.subprocess.clone_from(&base.subprocess);
                }
                if v.oauth.is_none() {
                    v.oauth.clone_from(&base.oauth);
                }
                if v.model_routes.is_empty() && v.base_url == base.base_url {
                    v.model_routes.clone_from(&base.model_routes);
                }
            }
            self.providers.insert(k, v);
        }
        for (k, mut v) in other.models {
            if let Some(base) = self.models.get(&k) {
                v.provider = v.provider.or_else(|| base.provider.clone());
                v.label = v.label.or_else(|| base.label.clone());
                v.order = v.order.or(base.order);
                v.api = v.api.or_else(|| base.api.clone());
                v.pricing = v.pricing.or(base.pricing);
                v.context_window = v.context_window.or(base.context_window);
                v.max_output_tokens = v.max_output_tokens.or(base.max_output_tokens);
                v.supports_mid_system_role =
                    v.supports_mid_system_role.or(base.supports_mid_system_role);
                v.supports_effort = v.supports_effort.or(base.supports_effort);
                v.supports_xhigh = v.supports_xhigh.or(base.supports_xhigh);
                v.needs_adaptive_thinking =
                    v.needs_adaptive_thinking.or(base.needs_adaptive_thinking);
                v.thinking_mode = v.thinking_mode.or_else(|| base.thinking_mode.clone());
                v.native_tool_search = v.native_tool_search.or(base.native_tool_search);
                v.compat_override = match (&base.compat_override, &v.compat_override) {
                    (Some(b), Some(o)) => Some(b.merge_override(o)),
                    (Some(b), None) => Some(b.clone()),
                    (None, o) => o.clone(),
                };
                if v.effort_map.is_empty() {
                    v.effort_map = base.effort_map.clone();
                }
                if v.body_patch.is_none() {
                    v.body_patch.clone_from(&base.body_patch);
                }
            }
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
            for route in &prov.model_routes {
                for auth in &route.auth_chain {
                    if let Some(ref env_name) = auth.env
                        && !env_name.is_empty()
                    {
                        set.insert(env_name.clone());
                    }
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

    /// Boundary-aware prefix model lookup matching `ResolvedProviderSpec::model_entry`.
    pub fn find_model(&self, model: &str) -> Option<&ModelEntry> {
        self.models.get(model).or_else(|| {
            self.models
                .iter()
                .filter(|(k, _)| is_model_prefix_match(model, k.as_str()))
                .max_by_key(|(k, _)| k.len())
                .map(|(_, v)| v)
        })
    }

    /// Resolves the target provider ID declared for `model` (exact or boundary-prefix match).
    pub fn provider_for_model(&self, model: &str) -> Option<&str> {
        self.find_model(model)?.provider.as_deref()
    }

    /// Resolves the effective wire protocol adapter ID for `(provider, model)` considering
    /// model-level `api` overrides and declarative `model_routes` (longest-prefix-first).
    pub fn effective_api(&self, name_or_alias: &str, model: &str) -> Option<String> {
        let (_, entry) = self.find_provider(name_or_alias)?;
        let model_entry = self.find_model(model);
        if let Some(api) = model_entry.and_then(|me| me.api.as_ref()) {
            return Some(api.clone());
        }
        if let Some(api) = entry.match_model_route(model).and_then(|r| r.api.as_ref()) {
            return Some(api.clone());
        }
        Some(entry.api.clone())
    }

    /// Resolves `base_url`, auth credentials, `project_id`, model routes, and model overrides for a provider.
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

        let effective_model = model_hint
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .or(entry.default_model.as_deref());
        let route = effective_model.and_then(|m| entry.match_model_route(m));

        let mut base_url = route
            .and_then(|r| r.base_url.as_ref())
            .filter(|u| !u.is_empty())
            .cloned()
            .unwrap_or_else(|| entry.base_url.clone());
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

        let auth_chain = route
            .filter(|r| !r.auth_chain.is_empty())
            .map_or(entry.auth_chain.as_slice(), |r| r.auth_chain.as_slice());
        let auth = resolve_auth_chain(canonical_id, auth_chain, entry.oauth.as_ref())?;

        // Model-level API override > provider model_route API > provider default API
        let model_entry = effective_model.and_then(|m| self.find_model(m));
        let api = model_entry
            .and_then(|me| me.api.clone())
            .or_else(|| route.and_then(|r| r.api.clone()))
            .unwrap_or_else(|| entry.api.clone());

        let native_tool_search = model_entry
            .and_then(|me| me.native_tool_search)
            .or_else(|| route.and_then(|r| r.native_tool_search))
            .or(entry.native_tool_search)
            .unwrap_or(false);

        let mut headers = entry.headers.clone();
        if let Some(r) = route {
            headers.extend(r.headers.clone());
        }

        let betas = route
            .and_then(|r| r.betas.clone())
            .unwrap_or_else(|| entry.betas.clone());

        let compat = match route.and_then(|r| r.compat.as_ref()) {
            Some(rc) => entry.compat.merge_override(rc),
            None => entry.compat.clone(),
        };

        let usage_map = route
            .and_then(|r| r.usage_map.clone())
            .unwrap_or_else(|| entry.usage_map.clone());

        let body_patch = match (&entry.body_patch, route.and_then(|r| r.body_patch.as_ref())) {
            (Some(base), Some(ov)) => {
                let mut merged = base.clone();
                json_merge_patch(&mut merged, ov);
                Some(merged)
            }
            (None, Some(ov)) => Some(ov.clone()),
            (Some(base), None) => Some(base.clone()),
            (None, None) => None,
        };

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
            headers,
            query_params: entry.query_params.clone(),
            betas,
            cache_diagnosis_beta: entry.cache_diagnosis_beta.clone(),
            native_tool_search,
            compat,
            usage_map,
            body_patch,
            subprocess: entry.subprocess.clone(),
            catalog_models: self.models.clone(),
        })
    }

    /// Produces a secret-safe summary of all registered providers for `carnot providers`.
    pub fn summaries(&self) -> Vec<ProviderSummary> {
        let stored_tokens = AuthStore::open_default()
            .and_then(|s| s.list().ok())
            .unwrap_or_default();
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
                let (configured, auth_source) = match probe_auth_chain(
                    id,
                    &entry.auth_chain,
                    entry.oauth.as_ref(),
                    &stored_tokens,
                ) {
                    Some(src) => (true, src),
                    None => (false, "missing".into()),
                };
                let oauth_supported = entry.oauth.is_some()
                    || entry
                        .auth_chain
                        .iter()
                        .any(|a| matches!(a.scheme, AuthScheme::Oauth));
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
                    oauth_supported,
                }
            })
            .collect()
    }
}

fn probe_auth_chain(
    provider_id: &str,
    chain: &[AuthEntry],
    oauth_cfg: Option<&OAuthConfig>,
    stored_tokens: &BTreeMap<String, crate::oauth::OAuthStoredToken>,
) -> Option<String> {
    if chain.is_empty() {
        return Some("none".into());
    }
    for entry in chain {
        if matches!(entry.scheme, AuthScheme::None) {
            return Some("none".into());
        }
        if matches!(entry.scheme, AuthScheme::Oauth) {
            let profile = entry.profile.as_deref().unwrap_or(provider_id);
            if let Some(tok) = stored_tokens.get(profile)
                && (!tok.is_expired() || tok.can_refresh(oauth_cfg))
            {
                return Some(format!("oauth:{profile}"));
            }
            continue;
        }
        if let Some(ref env_name) = entry.env
            && std::env::var(env_name).is_ok_and(|v| !v.trim().is_empty())
        {
            return Some(format!("env:{env_name}"));
        }
        if entry.cmd.as_ref().is_some_and(|c| !c.is_empty()) {
            return Some("cmd".into());
        }
    }
    None
}

fn resolve_auth_chain(
    provider_id: &str,
    chain: &[AuthEntry],
    oauth_cfg: Option<&OAuthConfig>,
) -> Result<ResolvedAuth> {
    if chain.is_empty() {
        return Ok(ResolvedAuth::None);
    }
    let mut tried_hints: Vec<String> = Vec::new();
    for entry in chain {
        if matches!(entry.scheme, AuthScheme::None) {
            return Ok(ResolvedAuth::None);
        }
        if matches!(entry.scheme, AuthScheme::Oauth) {
            let profile = entry.profile.as_deref().unwrap_or(provider_id);
            tried_hints.push(format!("`carnot auth login {profile}`"));
            if let Some(store) = AuthStore::open_default()
                && let Ok(Some(tok)) = store.get(profile)
                && (!tok.is_expired() || tok.can_refresh(oauth_cfg))
            {
                let source = format!("oauth:{profile}");
                tracing::debug!(
                    provider = %provider_id,
                    auth_source = %source,
                    "resolved OAuth provider credentials"
                );
                let manager = OAuthTokenManager::shared(profile, tok, oauth_cfg.cloned(), store);
                return Ok(ResolvedAuth::OAuth { manager, source });
            }
            continue;
        }
        let secret = if let Some(ref env_name) = entry.env {
            tried_hints.push(env_name.clone());
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
                AuthScheme::Oauth | AuthScheme::None => ResolvedAuth::None,
            });
        }
    }

    Err(CarnotError::Provider {
        provider: provider_id.to_string(),
        message: format!(
            "missing credentials for provider `{provider_id}`; set one of: {}",
            tried_hints.join(", ")
        ),
        retryable: false,
        retry_after_ms: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_preserves_builtin_compat_and_native_flags_on_partial_override() {
        let mut cat = ProviderCatalog::embedded().expect("embedded catalog");
        let overlay: ProviderCatalog = serde_json::from_str(
            r#"{
              "providers": {
                "anthropic": {
                  "api": "anthropic-messages",
                  "base_url": "https://gateway.example.internal",
                  "auth_chain": [{ "scheme": "bearer", "env": "CUSTOM_GW_TOKEN" }]
                }
              }
            }"#,
        )
        .expect("parse overlay");
        cat.merge(overlay);

        let (_, ant) = cat.find_provider("anthropic").expect("anthropic");
        assert_eq!(ant.base_url, "https://gateway.example.internal");
        assert_eq!(ant.compat.supports_native_web_search, Some(true));
        assert_eq!(ant.compat.supports_native_web_fetch, Some(true));
        assert_eq!(
            ant.compat.native_web_fetch_beta.as_deref(),
            Some("web-fetch-2025-09-10")
        );
        assert_eq!(ant.native_tool_search, Some(true));
        assert_eq!(
            ant.headers.get("anthropic-version").map(String::as_str),
            Some("2023-06-01")
        );

        // Explicit false in overlay still overrides inherited true
        let disable_overlay: ProviderCatalog = serde_json::from_str(
            r#"{
              "providers": {
                "anthropic": {
                  "api": "anthropic-messages",
                  "native_tool_search": false,
                  "compat": { "supports_native_web_search": false }
                }
              }
            }"#,
        )
        .expect("parse disable overlay");
        cat.merge(disable_overlay);

        let (_, ant2) = cat.find_provider("anthropic").expect("anthropic");
        assert_eq!(ant2.compat.supports_native_web_search, Some(false));
        assert_eq!(ant2.compat.supports_native_web_fetch, Some(true));
        assert_eq!(ant2.native_tool_search, Some(false));
    }

    #[test]
    fn merge_drops_upstream_model_routes_when_overlay_replaces_base_url() {
        let mut cat = ProviderCatalog::embedded().expect("embedded catalog");
        // Partial override without changing base_url keeps model_routes
        let partial: ProviderCatalog = serde_json::from_str(
            r#"{
              "providers": {
                "opencode": {
                  "api": "openai-completions",
                  "default_model": "kimi-k3.1-pro"
                }
              }
            }"#,
        )
        .expect("partial overlay");
        cat.merge(partial);
        assert!(
            !cat.find_provider("opencode")
                .expect("opencode")
                .1
                .model_routes
                .is_empty()
        );

        // Redirecting base_url to a custom proxy must NOT inherit upstream model_routes
        let proxy: ProviderCatalog = serde_json::from_str(
            r#"{
              "providers": {
                "opencode-go": {
                  "api": "openai-completions",
                  "base_url": "https://proxy.example.internal/v1"
                }
              }
            }"#,
        )
        .expect("proxy overlay");
        cat.merge(proxy);
        let (_, go) = cat.find_provider("opencode-go").expect("opencode-go");
        assert_eq!(go.base_url, "https://proxy.example.internal/v1");
        assert!(go.model_routes.is_empty());
    }
}
