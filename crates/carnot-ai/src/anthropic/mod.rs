//! Anthropic Messages wire protocol adapter (`api = "anthropic-messages"`).
//!
//! Zero hardcoded provider names or URLs! Auth scheme (`Bearer` vs `x-api-key` — never both),
//! base URL, headers, `betas`, model capabilities, pricing, and `body_patch` all come from
//! `ResolvedProviderSpec` (`providers.json`).

mod request;
mod stream;
#[cfg(test)]
mod tests;

pub use stream::find_sse_frame_end;

use self::stream::{SseParseState, handle_sse_frame};
use crate::catalog::{ProviderCatalog, ResolvedAuth, ResolvedProviderSpec};
use crate::compat::{
    is_context_overflow, is_retryable_status, parse_rate_limit_windows, parse_retry_after_ms,
};
use crate::registry::ApiAdapter;
use crate::{BoxFuture, Provider, ProviderRequest, StreamEvent};
use carnot_core::{
    CarnotError, ContentBlock, ContentOrigin, Message, ProviderWebCapabilities, Result, Role,
};
use futures_util::StreamExt;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const API_ID: &str = "anthropic-messages";
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(300);
const HTTP_HEADERS_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug)]
pub struct AnthropicMessagesAdapter;

impl ApiAdapter for AnthropicMessagesAdapter {
    fn api_id(&self) -> &'static str {
        API_ID
    }

    fn build(&self, spec: &ResolvedProviderSpec) -> Result<Arc<dyn Provider>> {
        let p = AnthropicProvider::from_spec(spec.clone());
        if tokio::runtime::Handle::try_current().is_ok() {
            p.spawn_prewarm();
        }
        Ok(Arc::new(p))
    }
}

fn messages_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{base}/messages")
    } else {
        format!("{base}/v1/messages")
    }
}

/// Anthropic Messages API provider with hybrid cache TTLs, cache diagnostics, and TLS pre-warm.
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    client: reqwest::Client,
    spec: ResolvedProviderSpec,
    cache_diagnosis: bool,
    last_message_id: Arc<std::sync::Mutex<Option<String>>>,
}

impl AnthropicProvider {
    /// Convenience constructor for direct Rust embedding & tests using embedded catalog metadata.
    pub fn new(api_key: String) -> Self {
        let base_url = ProviderCatalog::embedded()
            .ok()
            .and_then(|c| {
                c.find_provider("anthropic")
                    .map(|(_, e)| e.base_url.clone())
            })
            .unwrap_or_default();
        Self::with_base_url(api_key, base_url)
    }

    pub fn with_base_url(api_key: String, mut base_url: String) -> Self {
        while base_url.ends_with('/') {
            base_url.pop();
        }
        let catalog = ProviderCatalog::embedded().unwrap_or_default();
        let mut headers = BTreeMap::new();
        headers.insert("anthropic-version".into(), "2023-06-01".into());
        let spec = ResolvedProviderSpec {
            provider_id: "anthropic".into(),
            api: API_ID.into(),
            base_url,
            fallback_base_urls: Vec::new(),
            project_id: None,
            auth: ResolvedAuth::Header {
                name: "x-api-key".into(),
                value: api_key,
                source: "direct".into(),
            },
            headers,
            query_params: BTreeMap::new(),
            betas: Vec::new(),
            cache_diagnosis_beta: Some("cache-diagnosis-2026-04-07".into()),
            native_tool_search: true,
            compat: crate::catalog::CompatFlags::default(),
            usage_map: crate::catalog::UsageMap::default(),
            body_patch: None,
            subprocess: None,
            catalog_models: catalog.models,
        };
        Self::from_spec(spec)
    }

    pub fn from_spec(spec: ResolvedProviderSpec) -> Self {
        let client = crate::build_http_client_for_url(&spec.base_url);
        let cache_diagnosis = std::env::var("CARNOT_CACHE_DIAGNOSIS")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        Self {
            client,
            spec,
            cache_diagnosis,
            last_message_id: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    #[must_use]
    pub fn with_cache_diagnosis(mut self, enabled: bool) -> Self {
        self.cache_diagnosis = enabled;
        self
    }

    /// Background non-blocking DNS+TLS pool pre-warming toward remote HTTPS endpoints.
    pub fn spawn_prewarm(&self) {
        if !self.spec.base_url.starts_with("https://") {
            return;
        }
        let client = self.client.clone();
        let url = messages_url(&self.spec.base_url);
        tokio::spawn(async move {
            let _ = tokio::time::timeout(Duration::from_secs(3), client.head(&url).send()).await;
        });
    }

    fn supports_mid_system_role(&self, model: &str) -> bool {
        self.spec
            .model_entry(model)
            .and_then(|m| m.supports_mid_system_role)
            .unwrap_or(false)
    }

    fn supports_effort(&self, model: &str) -> bool {
        self.spec
            .model_entry(model)
            .and_then(|m| m.supports_effort)
            .unwrap_or(false)
    }

    fn supports_xhigh(&self, model: &str) -> bool {
        self.spec
            .model_entry(model)
            .and_then(|m| m.supports_xhigh)
            .unwrap_or(false)
    }

    fn thinking_mode<'a>(&'a self, model: &str) -> Option<&'a str> {
        let entry = self.spec.model_entry(model)?;
        if let Some(ref mode) = entry.thinking_mode {
            return Some(mode.as_str());
        }
        if entry.needs_adaptive_thinking == Some(true) {
            return Some("adaptive");
        }
        None
    }
}

impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        &self.spec.provider_id
    }

    fn supports_native_tool_search(&self) -> bool {
        self.spec.native_tool_search
    }

    fn web_capabilities(&self, model: &str) -> ProviderWebCapabilities {
        let compat = self.spec.effective_compat(model);
        ProviderWebCapabilities {
            native_search: compat.supports_native_web_search.unwrap_or(false),
            native_fetch: compat.supports_native_web_fetch.unwrap_or(false),
        }
    }

    fn effort_preserves_cache(&self, model: &str) -> bool {
        self.supports_effort(model) && self.thinking_mode(model) != Some("budget")
    }

    fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<mpsc::Receiver<StreamEvent>>> {
        Box::pin(async move {
            let raw_url = messages_url(&self.spec.base_url);
            let body = self.build_body(&request);
            let mut extra_headers: Vec<(String, String)> = Vec::new();
            if !self.spec.headers.contains_key("anthropic-version") {
                extra_headers.push(("anthropic-version".into(), "2023-06-01".into()));
            }
            // Data-driven betas (empty by default so gateways/Bedrock/Vertex never 400!)
            let mut betas = self.spec.betas.clone();
            let compat = self.spec.effective_compat(&request.model);
            if request.native_web.fetch
                && compat.supports_native_web_fetch.unwrap_or(false)
                && let Some(ref wf_beta) = compat.native_web_fetch_beta
                && !betas.iter().any(|b| b == wf_beta)
            {
                betas.push(wf_beta.clone());
            }
            if self.cache_diagnosis
                && let Some(ref diag_beta) = self.spec.cache_diagnosis_beta
                && !betas.iter().any(|b| b == diag_beta)
            {
                betas.push(diag_beta.clone());
            }
            if !betas.is_empty() {
                extra_headers.push(("anthropic-beta".into(), betas.join(",")));
            }

            let resp = crate::send_authorized_json_post(
                &self.client,
                &self.spec,
                &raw_url,
                &extra_headers,
                &body,
                HTTP_HEADERS_TIMEOUT,
            )
            .await?;

            let status = resp.status();
            if !status.is_success() {
                let retry_after_ms = parse_retry_after_ms(resp.headers());
                let text = resp.text().await.unwrap_or_default();
                if is_context_overflow(Some(status.as_u16()), &text) {
                    return Err(CarnotError::ContextOverflow(text));
                }
                return Err(CarnotError::Provider {
                    provider: self.spec.provider_id.clone(),
                    message: format!("HTTP {status}: {text}"),
                    retryable: is_retryable_status(status.as_u16()),
                    retry_after_ms,
                });
            }

            let rate_limits = parse_rate_limit_windows(resp.headers());
            let (tx, rx) = mpsc::channel(64);
            let pricing = self.spec.effective_pricing(&request.model);
            let default_ttl = request.cache_ttl;
            let last_msg_id = Arc::clone(&self.last_message_id);
            let origin = ContentOrigin::new(API_ID, &self.spec.provider_id, &request.model);

            tokio::spawn(async move {
                let mut stream = resp.bytes_stream();
                let mut raw_buf: Vec<u8> = Vec::with_capacity(8192);
                let mut state = SseParseState::default();

                loop {
                    let next_chunk = tokio::select! {
                        () = cancel.cancelled() => {
                            let _ = tx.send(StreamEvent::Error { message: "cancelled".into(), retryable: false }).await;
                            return;
                        }
                        res = tokio::time::timeout(STREAM_IDLE_TIMEOUT, stream.next()) => res,
                    };

                    let Ok(chunk_opt) = next_chunk else {
                        let _ = tx
                            .send(StreamEvent::Error {
                                message: "stream idle timeout (300s)".into(),
                                retryable: true,
                            })
                            .await;
                        return;
                    };

                    let Some(chunk) = chunk_opt else { break };
                    let Ok(bytes) = chunk else {
                        let _ = tx
                            .send(StreamEvent::Error {
                                message: "stream read error".into(),
                                retryable: true,
                            })
                            .await;
                        return;
                    };

                    raw_buf.extend_from_slice(&bytes);
                    while let Some((end, delim_len)) = find_sse_frame_end(&raw_buf) {
                        let frame_bytes: Vec<u8> = raw_buf.drain(..end + delim_len).collect();
                        let frame_str = String::from_utf8_lossy(&frame_bytes[..end]);
                        if let Some((err_msg, retryable)) =
                            handle_sse_frame(&frame_str, &mut state, default_ttl, &tx).await
                        {
                            let ev = if is_context_overflow(None, &err_msg) {
                                StreamEvent::ContextOverflow(err_msg)
                            } else {
                                StreamEvent::Error {
                                    message: err_msg,
                                    retryable,
                                }
                            };
                            let _ = tx.send(ev).await;
                            return;
                        }
                    }
                }

                // Flush any final SSE frame not terminated by trailing blank line on EOF
                if !raw_buf.is_empty() {
                    let tail = String::from_utf8_lossy(&raw_buf);
                    if !tail.trim().is_empty()
                        && let Some((err_msg, retryable)) =
                            handle_sse_frame(&tail, &mut state, default_ttl, &tx).await
                    {
                        let ev = if is_context_overflow(None, &err_msg) {
                            StreamEvent::ContextOverflow(err_msg)
                        } else {
                            StreamEvent::Error {
                                message: err_msg,
                                retryable,
                            }
                        };
                        let _ = tx.send(ev).await;
                        return;
                    }
                }

                // EOF without `message_stop` is a truncated stream error, never a silent success.
                if !state.saw_message_stop {
                    let _ = tx
                        .send(StreamEvent::Error {
                            message: "stream terminated before message_stop".into(),
                            retryable: true,
                        })
                        .await;
                    return;
                }

                if let Some(mid) = state.message_id
                    && let Ok(mut g) = last_msg_id.lock()
                {
                    *g = Some(mid);
                }
                state.usage.compute_cost(pricing);
                let content: Vec<ContentBlock> = state
                    .blocks
                    .into_values()
                    .map(|b| b.finish(&origin))
                    .collect();
                let _ = tx
                    .send(StreamEvent::Done {
                        message: Message {
                            role: Role::Assistant,
                            content,
                        },
                        usage: state.usage,
                        stop_reason: state.stop_reason,
                        rate_limits,
                    })
                    .await;
            });

            Ok(rx)
        })
    }
}
