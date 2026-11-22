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
            max_attempts_per_provider: 3,
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

    fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<mpsc::Receiver<StreamEvent>>> {
        Box::pin(async move {
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
                            let shift = 1u64.checked_shl(attempt.min(10)).unwrap_or(1024);
                            let nanos = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.subsec_nanos() as u64)
                                .unwrap_or(0);
                            let jitter = (nanos ^ (attempt as u64).wrapping_mul(0x9e37_79b9)) % 400;
                            let delay = retry_after_ms
                                .unwrap_or_else(|| {
                                    self.base_backoff_ms
                                        .saturating_mul(shift)
                                        .saturating_add(jitter)
                                })
                                .min(30_000);
                            tracing::warn!(
                                provider = %prov,
                                attempt,
                                delay_ms = delay,
                                %message,
                                "transient provider error; retrying"
                            );
                            last_err = Some(CarnotError::Provider {
                                provider: prov,
                                message,
                                retryable: true,
                                retry_after_ms,
                            });
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
