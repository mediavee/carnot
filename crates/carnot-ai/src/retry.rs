use crate::{BoxFuture, Provider, ProviderRequest, StreamEvent};
use carnot_core::{CarnotError, Result};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Exponential backoff retry + ordered provider failover wrapper.
#[derive(Debug)]
pub struct ResilientProvider {
    providers: Vec<Arc<dyn Provider>>,
    max_attempts_per_provider: u32,
    base_backoff_ms: u64,
}

impl ResilientProvider {
    pub fn new(providers: Vec<Arc<dyn Provider>>) -> Self {
        Self {
            providers,
            max_attempts_per_provider: 5,
            base_backoff_ms: 1_500,
        }
    }

    pub fn with_retries(mut self, attempts: u32, base_backoff_ms: u64) -> Self {
        self.max_attempts_per_provider = attempts.max(1);
        self.base_backoff_ms = base_backoff_ms;
        self
    }
}

impl Provider for ResilientProvider {
    fn name(&self) -> &str {
        self.providers
            .first()
            .map(|p| p.name())
            .unwrap_or("resilient")
    }

    fn supports_native_tool_search(&self) -> bool {
        self.providers
            .first()
            .map(|p| p.supports_native_tool_search())
            .unwrap_or(false)
    }

    fn web_capabilities(&self, model: &str) -> carnot_core::ProviderWebCapabilities {
        self.providers
            .first()
            .map(|p| p.web_capabilities(model))
            .unwrap_or_default()
    }

    fn effort_preserves_cache(&self, model: &str) -> bool {
        self.providers
            .first()
            .map(|p| p.effort_preserves_cache(model))
            .unwrap_or(false)
    }

    fn rolling_cache_ttl_ms(&self, model: &str, ttl: carnot_core::CacheTtl) -> u64 {
        self.providers
            .first()
            .map(|p| p.rolling_cache_ttl_ms(model, ttl))
            .unwrap_or(match ttl {
                carnot_core::CacheTtl::Long => 3_610_000,
                carnot_core::CacheTtl::Short | carnot_core::CacheTtl::Hybrid => 310_000,
            })
    }

    fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<mpsc::Receiver<StreamEvent>>> {
        Box::pin(async move {
            const MAX_RETRY_DELAY_MS: u64 = 60_000;
            let mut last_err: Option<CarnotError> = None;
            for provider in &self.providers {
                for attempt in 0..self.max_attempts_per_provider {
                    if cancel.is_cancelled() {
                        return Err(CarnotError::Cancelled);
                    }
                    match provider.stream(request.clone(), cancel.clone()).await {
                        Ok(rx) => return Ok(rx),
                        Err(CarnotError::Provider {
                            retryable: true,
                            retry_after_ms,
                            provider: prov,
                            message,
                        }) => {
                            let is_last_attempt = attempt + 1 >= self.max_attempts_per_provider;
                            // Fail fast if server explicitly demands a wait longer than our cap
                            // (e.g. 1-hour quota block) instead of burning 5x60s retries.
                            let exceeds_cap =
                                retry_after_ms.is_some_and(|ms| ms > MAX_RETRY_DELAY_MS);
                            last_err = Some(CarnotError::Provider {
                                provider: prov.clone(),
                                message: message.clone(),
                                retryable: true,
                                retry_after_ms,
                            });
                            if is_last_attempt || exceeds_cap {
                                tracing::warn!(
                                    provider = %prov,
                                    attempt,
                                    ?retry_after_ms,
                                    %message,
                                    "transient provider error exhausted retry budget or exceeded max delay; failing over"
                                );
                                break;
                            }
                            let shift = 1u64.checked_shl(attempt.min(10)).unwrap_or(1024);
                            let base_delay = retry_after_ms
                                .unwrap_or_else(|| self.base_backoff_ms.saturating_mul(shift))
                                .min(MAX_RETRY_DELAY_MS);
                            // Proportional +-20% jitter desynchronizes 429 retry storms cleanly
                            let nanos = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.subsec_nanos() as u64)
                                .unwrap_or(0);
                            let jitter_band = (base_delay / 5).max(1);
                            let jitter_offset = (nanos
                                ^ (attempt as u64).wrapping_mul(0x9e37_79b9))
                                % (jitter_band * 2);
                            let delay = base_delay
                                .saturating_sub(jitter_band)
                                .saturating_add(jitter_offset)
                                .clamp(100, MAX_RETRY_DELAY_MS);
                            tracing::warn!(
                                provider = %prov,
                                attempt,
                                delay_ms = delay,
                                %message,
                                "transient provider error; retrying"
                            );
                            tokio::select! {
                                _ = cancel.cancelled() => return Err(CarnotError::Cancelled),
                                _ = tokio::time::sleep(Duration::from_millis(delay)) => {}
                            }
                        }
                        Err(other) => return Err(other),
                    }
                }
            }
            Err(last_err.unwrap_or_else(|| CarnotError::Provider {
                provider: "resilient".into(),
                message: "no providers configured".into(),
                retryable: false,
                retry_after_ms: None,
            }))
        })
    }
}
