//! Modular, zero-hardcode multi-provider streaming wire layer:
//!   - 3-layer separation: `api` (code adapters) / `provider` (data in `providers.json`) / `model` (data)
//!   - 7 registered protocol adapters: `anthropic-messages`, `openai-completions`, `openai-responses`,
//!     `google-generative`, `google-cloudcode`, `subprocess-jsonl`, `faux`
//!   - Out-of-the-box declarative catalog for Anthropic, OpenAI, OpenAI Responses, Grok (xAI),
//!     Muse (Meta/Llama), Gemini, Antigravity (Cloud Code), OpenRouter, DeepSeek, Ollama, Faux
//!   - Origin-aware cross-provider history replay (zero 400s on foreign signatures/opaques)
//!   - Full multimodal image support (incl. tool-result bridge) & declarative `UsageMap`

pub mod anthropic;
pub mod catalog;
pub mod compat;
pub mod faux;
pub mod google;
pub mod oauth;
pub mod openai;
pub mod openai_responses;
pub mod registry;
pub mod retry;
pub mod subprocess;

#[cfg(test)]
mod conformance_tests;

use carnot_core::{
    CacheTtl, Message, NativeWebConfig, ProviderWebCapabilities, RateLimitWindow, ReasoningEffort,
    Result, StopReason, ToolSpec, Usage,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Build an HTTP client using the OS certificate store, falling back to the
/// compiled-in Mozilla WebPKI root certificates when running inside minimal
/// containers (e.g., Alpine / Bun / distroless) that ship without system CA bundles.
pub fn build_http_client() -> reqwest::Client {
    build_http_client_for_url("https://")
}

/// URL-aware HTTP client builder: avoids OS root-store / macOS Keychain initialization on
/// plaintext `http://` endpoints (local proxies, Ollama, vLLM, tests), while preserving full
/// OS-store + `webpki-root-certs` fallback for all `https://` endpoints.
pub fn build_http_client_for_url(base_url: &str) -> reqwest::Client {
    let base = || {
        reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .tcp_keepalive(std::time::Duration::from_secs(30))
    };
    if base_url.trim_start().starts_with("http://")
        && let Ok(client) = base().tls_certs_only(Vec::new()).build()
    {
        return client;
    }
    if let Ok(client) = base().build() {
        return client;
    }
    let certs: Vec<reqwest::Certificate> = webpki_root_certs::TLS_SERVER_ROOT_CERTS
        .iter()
        .filter_map(|der| reqwest::Certificate::from_der(der.as_ref()).ok())
        .collect();
    base()
        .tls_certs_only(certs)
        .build()
        .expect("reqwest client with bundled webpki roots should always build")
}

/// Outgoing provider request carrying exact cache-breakpoint metadata (`Arc<Message>` zero-copy).
#[derive(Debug, Clone)]
pub struct ProviderRequest {
    pub model: String,
    pub static_system_prompt: String,
    pub tools: Vec<ToolSpec>,
    pub prefix_hash: String,
    /// Stable session identifier used to isolate per-session provider cache keys (`prompt_cache_key`,
    /// `session_id_header`) so concurrent sessions with the same static prefix never collide.
    pub session_id: String,
    pub messages: Vec<Arc<Message>>,
    pub checkpoint_index: Option<usize>,
    pub cache_ttl: CacheTtl,
    pub effort: ReasoningEffort,
    pub max_output_tokens: u32,
    /// When `true` (used by one-shot cache-warm compaction calls), reuse the warm prompt cache
    /// up to the previous turn's boundary without paying a `1.25x`/`2.0x` cache-write surcharge
    /// on the discarded tail + compaction prompt.
    pub skip_tail_cache_write: bool,
    /// Optional provider-native server-side web search & fetch configuration.
    pub native_web: NativeWebConfig,
    /// Ephemeral per-turn tail notes attached AFTER all provider prompt-cache breakpoints so
    /// transient reminders NEVER invalidate or re-bill cached conversation prefixes.
    pub ephemeral_tail_notes: Vec<String>,
}

/// Incremental streaming event from any model provider.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ThinkingDelta(String),
    /// Emitted when a complete `ToolUse` finishes streaming (carries `invalid_json` if malformed).
    ToolCallReady {
        id: String,
        name: String,
        input: serde_json::Value,
        invalid_json: Option<String>,
    },
    Done {
        message: Message,
        usage: Usage,
        stop_reason: StopReason,
        rate_limits: Vec<RateLimitWindow>,
    },
    Error {
        message: String,
        retryable: bool,
    },
    /// Mid-stream context-window overflow signaled over SSE (routed directly to emergency compaction).
    ContextOverflow(String),
}

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub trait Provider: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;

    /// True if this provider natively supports `defer_loading: true` + server-side tool search.
    /// When false, `carnot-agent` enables the cache-invariant local `ToolSearch` + `CallDeferredTool` fallback.
    fn supports_native_tool_search(&self) -> bool {
        false
    }

    /// Declares whether this provider + model natively supports server-side web search and/or fetch.
    fn web_capabilities(&self, _model: &str) -> ProviderWebCapabilities {
        ProviderWebCapabilities::default()
    }

    /// True when changing `ProviderRequest::effort` preserves the provider's prompt cache for
    /// `model` (e.g. Anthropic `output_config.effort` on adaptive/native models vs legacy
    /// `thinking.budget_tokens` models where changing budget invalidates message caches).
    fn effort_preserves_cache(&self, _model: &str) -> bool {
        false
    }

    fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<mpsc::Receiver<StreamEvent>>>;
}

/// Sends a JSON POST request applying static `query_params`, provider headers, live auth
/// (including proactive OAuth refresh), and a single reactive retry on HTTP `401 Unauthorized`
/// when credentials are refreshable.
pub(crate) async fn send_authorized_json_post(
    client: &reqwest::Client,
    spec: &crate::catalog::ResolvedProviderSpec,
    raw_url: &str,
    extra_headers: &[(String, String)],
    body: &serde_json::Value,
    timeout: std::time::Duration,
) -> Result<reqwest::Response> {
    let url = spec.build_url(raw_url);
    let mut rejected_token: Option<String> = None;
    for attempt in 0..2 {
        let mut rb = client.post(&url).header("content-type", "application/json");
        let used_token;
        (rb, used_token) = spec
            .auth
            .apply_to_request_async(rb, rejected_token.as_deref())
            .await?;
        for (k, v) in &spec.headers {
            rb = rb.header(k.as_str(), v.as_str());
        }
        for (k, v) in extra_headers {
            rb = rb.header(k.as_str(), v.as_str());
        }

        let resp = tokio::time::timeout(timeout, rb.json(body).send())
            .await
            .map_err(|_| carnot_core::CarnotError::Provider {
                provider: spec.provider_id.clone(),
                message: format!("HTTP response headers timeout ({}s)", timeout.as_secs()),
                retryable: true,
                retry_after_ms: None,
            })?
            .map_err(|e| carnot_core::CarnotError::Provider {
                provider: spec.provider_id.clone(),
                message: e.without_url().to_string(),
                retryable: true,
                retry_after_ms: None,
            })?;

        if attempt == 0
            && resp.status() == reqwest::StatusCode::UNAUTHORIZED
            && spec.auth.supports_refresh()
        {
            tracing::debug!(
                provider = %spec.provider_id,
                "received HTTP 401; forcing single-flight OAuth token refresh and retrying once"
            );
            rejected_token = used_token;
            continue;
        }
        return Ok(resp);
    }
    unreachable!("loop always returns on attempt 1")
}
