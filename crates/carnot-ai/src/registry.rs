//! Protocol-indexed `ApiAdapter` trait and `ProviderRegistry` (zero hardcoded provider names!).
//!
//! Any wire protocol (`anthropic-messages`, `openai-completions`, `openai-responses`,
//! `google-generative`, `google-cloudcode`, `subprocess-jsonl`, `faux`, or a custom Rust plugin)
//! registers itself strictly by `api_id()`. Concrete providers (`anthropic`, `openai`, `grok`,
//! `muse`, `gemini`, `antigravity`, `deepseek`, `openrouter`, `ollama`, or user-defined entries)
//! live purely as data in `providers.json`.

use crate::Provider;
use crate::anthropic::AnthropicMessagesAdapter;
use crate::catalog::{ProviderCatalog, ProviderSummary, ResolvedProviderSpec};
use crate::faux::FauxApiAdapter;
use crate::google::{GoogleCloudCodeAdapter, GoogleGenerativeAdapter};
use crate::openai::OpenAiCompletionsAdapter;
use crate::openai_responses::OpenAiResponsesAdapter;
use crate::retry::ResilientProvider;
use crate::subprocess::SubprocessJsonlAdapter;
use carnot_core::{CarnotError, ModelPricing, Result};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Pluggable wire-protocol adapter interface.
pub trait ApiAdapter: Send + Sync + std::fmt::Debug {
    /// Canonical wire protocol identifier (e.g. `"openai-completions"`, `"google-generative"`).
    fn api_id(&self) -> &'static str;

    /// Instantiates a concrete `Provider` from a resolved data catalog specification.
    fn build(&self, spec: &ResolvedProviderSpec) -> Result<Arc<dyn Provider>>;
}

/// Unified registry mapping `provider_name` (via data catalog) -> `api_id` -> `ApiAdapter` -> `Provider`.
#[derive(Debug)]
pub struct ProviderRegistry {
    catalog: ProviderCatalog,
    adapters: BTreeMap<String, Arc<dyn ApiAdapter>>,
}

impl ProviderRegistry {
    /// Builds a registry populated with all built-in wire protocol adapters and the layered
    /// data catalog (embedded defaults + `~/.carnot/providers.json` + explicit `extra_configs`).
    pub fn load_default(extra_configs: &[PathBuf]) -> Result<Self> {
        let catalog = ProviderCatalog::load_layered(extra_configs)?;
        Ok(Self::from_catalog(catalog))
    }

    pub fn from_catalog(catalog: ProviderCatalog) -> Self {
        let mut reg = Self {
            catalog,
            adapters: BTreeMap::new(),
        };
        reg.register_adapter(Arc::new(AnthropicMessagesAdapter));
        reg.register_adapter(Arc::new(OpenAiCompletionsAdapter));
        reg.register_adapter(Arc::new(OpenAiResponsesAdapter));
        reg.register_adapter(Arc::new(GoogleGenerativeAdapter));
        reg.register_adapter(Arc::new(GoogleCloudCodeAdapter));
        reg.register_adapter(Arc::new(SubprocessJsonlAdapter));
        reg.register_adapter(Arc::new(FauxApiAdapter));
        reg
    }

    /// Registers a custom Rust wire-protocol plugin (overwrites any existing adapter for `api_id`).
    pub fn register_adapter(&mut self, adapter: Arc<dyn ApiAdapter>) {
        self.adapters.insert(adapter.api_id().to_string(), adapter);
    }

    pub fn catalog(&self) -> &ProviderCatalog {
        &self.catalog
    }

    pub fn registered_apis(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }

    pub fn summaries(&self) -> Vec<ProviderSummary> {
        self.catalog.summaries()
    }

    /// Resolves exact pricing from the catalog (prefix-aware, falling back to `ModelPricing::for_model`).
    pub fn pricing_for_model(&self, model: &str) -> ModelPricing {
        self.catalog
            .find_model(model)
            .and_then(|m| m.pricing)
            .unwrap_or_else(|| ModelPricing::for_model(model))
    }

    /// Resolves the model's declared context window from `providers.json` when available.
    pub fn context_window_for_model(&self, model: &str) -> Option<u64> {
        self.catalog
            .find_model(model)
            .and_then(|m| m.context_window)
    }

    pub fn max_output_tokens_for_model(&self, model: &str) -> Option<u32> {
        self.catalog
            .find_model(model)
            .and_then(|m| m.max_output_tokens)
    }

    /// Resolves a provider by name or alias and wraps it in `ResilientProvider` retry logic.
    pub fn build_provider(
        &self,
        provider_name: &str,
        model_hint: Option<&str>,
    ) -> Result<Arc<dyn Provider>> {
        let spec = self.catalog.resolve_spec(provider_name, model_hint)?;
        let raw = self.build_from_spec(&spec)?;
        if spec.api == "faux" {
            Ok(raw)
        } else {
            Ok(Arc::new(ResilientProvider::new(vec![raw])))
        }
    }

    /// Builds an unwrapped provider directly from a `ResolvedProviderSpec`.
    pub fn build_from_spec(&self, spec: &ResolvedProviderSpec) -> Result<Arc<dyn Provider>> {
        let adapter = self
            .adapters
            .get(&spec.api)
            .ok_or_else(|| CarnotError::Provider {
                provider: spec.provider_id.clone(),
                message: format!(
                    "no ApiAdapter registered for api `{}` (available: {})",
                    spec.api,
                    self.registered_apis().join(", ")
                ),
                retryable: false,
                retry_after_ms: None,
            })?;
        adapter.build(spec)
    }
}
