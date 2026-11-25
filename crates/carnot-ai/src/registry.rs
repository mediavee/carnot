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
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

/// Engine-level summary of a model discovered from the layered `ProviderCatalog`
/// (`embedded` + `~/.carnot/providers.json` + `--provider-config`).
/// Zero hardcoded model lists: shared by ACP, TUI model picker, Subagent discovery, and SDK callers.
#[derive(Debug, Clone, Serialize)]
pub struct AvailableModel {
    pub id: String,
    pub label: String,
    pub provider: String,
    pub api: String,
    pub configured: bool,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u32>,
}

/// Resolved provider + concrete model specification ready to run an agent or subagent.
#[derive(Debug)]
pub struct ResolvedModelTarget {
    pub provider_id: String,
    pub model_id: String,
    pub provider: Arc<dyn Provider>,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u32>,
}

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

    /// Lists models from the layered catalog (`embedded` + `~/.carnot/providers.json` +
    /// `--provider-config`), ordered by catalog `order` then ID.
    ///
    /// When `configured_only` is `true`, only returns models whose provider currently has
    /// valid credentials configured (excluding test-only `faux` providers).
    pub fn available_models(&self, configured_only: bool) -> Vec<AvailableModel> {
        let summaries = self.catalog.summaries();
        let configured_map: BTreeMap<String, (String, bool)> = summaries
            .iter()
            .map(|s| (s.id.clone(), (s.api.clone(), s.configured)))
            .collect();

        let has_labeled = self
            .catalog
            .models
            .values()
            .any(|e| e.provider.is_some() && e.label.is_some());

        let mut seen = BTreeSet::new();
        let mut items: Vec<(u32, AvailableModel)> = Vec::new();

        for (id, entry) in &self.catalog.models {
            let Some(prov_id) = entry.provider.as_deref() else {
                continue;
            };
            if has_labeled && entry.label.is_none() {
                continue;
            }
            let Some((canon_prov, _)) = self.catalog.find_provider(prov_id) else {
                continue;
            };
            let Some((api, is_cfg)) = configured_map.get(canon_prov) else {
                continue;
            };
            if configured_only && (!*is_cfg || api == "faux") {
                continue;
            }
            if !seen.insert((canon_prov.to_string(), id.clone())) {
                continue;
            }
            let effective_api = self
                .catalog
                .effective_api(canon_prov, id)
                .unwrap_or_else(|| api.clone());
            items.push((
                entry.order.unwrap_or(500),
                AvailableModel {
                    id: id.clone(),
                    label: entry.label.clone().unwrap_or_else(|| id.clone()),
                    provider: canon_prov.to_string(),
                    api: effective_api,
                    configured: *is_cfg,
                    context_window: entry.context_window,
                    max_output_tokens: entry.max_output_tokens,
                },
            ));
        }

        // Also include each configured provider's `default_model` if no labeled entry covered it
        // (e.g. user-added custom gateway providers in `~/.carnot/providers.json`!).
        for s in &summaries {
            if configured_only && (!s.configured || s.api == "faux") {
                continue;
            }
            let Some(ref def_model) = s.default_model else {
                continue;
            };
            if seen.contains(&(s.id.clone(), def_model.clone())) {
                continue;
            }
            let already_has_provider = items.iter().any(|(_, m)| m.provider == s.id);
            if already_has_provider {
                continue;
            }
            seen.insert((s.id.clone(), def_model.clone()));
            let me = self.catalog.find_model(def_model);
            let effective_api = self
                .catalog
                .effective_api(&s.id, def_model)
                .unwrap_or_else(|| s.api.clone());
            items.push((
                900,
                AvailableModel {
                    id: def_model.clone(),
                    label: me
                        .and_then(|m| m.label.clone())
                        .unwrap_or_else(|| def_model.clone()),
                    provider: s.id.clone(),
                    api: effective_api,
                    configured: s.configured,
                    context_window: me.and_then(|m| m.context_window),
                    max_output_tokens: me.and_then(|m| m.max_output_tokens),
                },
            ));
        }

        items.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.id.cmp(&b.1.id)));
        items.into_iter().map(|(_, m)| m).collect()
    }

    /// Resolves a user-supplied model reference (bare model ID, `provider/model`, or family alias
    /// like `sonnet` / `anthropic/opus`) against the layered catalog, preferring configured
    /// providers over unconfigured ones.
    pub fn resolve_model_spec(
        &self,
        reference: &str,
        provider_override: Option<&str>,
    ) -> Option<(String, String)> {
        let trimmed = reference.trim();
        if trimmed.is_empty() {
            return None;
        }
        let cat = &self.catalog;
        let (wanted_prov, raw_model) = if let Some(ov) = provider_override {
            let canon = cat
                .find_provider(ov)
                .map_or_else(|| ov.to_string(), |(id, _)| id.to_string());
            (Some(canon), trimmed)
        } else if let Some((lhs, rhs)) = trimmed.split_once('/')
            && let Some((canon, _)) = cat.find_provider(lhs)
        {
            (Some(canon.to_string()), rhs)
        } else {
            (None, trimmed)
        };

        let all_models = self.available_models(false);
        // 1. Exact (provider, id) match
        if let Some(ref wp) = wanted_prov
            && let Some(found) = all_models
                .iter()
                .find(|m| m.provider == *wp && m.id == raw_model)
        {
            return Some((found.provider.clone(), found.id.clone()));
        }
        // 2. Bare exact id match (prefer configured providers over unconfigured ones)
        if wanted_prov.is_none() {
            if let Some(found) = all_models
                .iter()
                .find(|m| m.configured && m.id == raw_model)
            {
                return Some((found.provider.clone(), found.id.clone()));
            }
            if let Some(found) = all_models.iter().find(|m| m.id == raw_model) {
                return Some((found.provider.clone(), found.id.clone()));
            }
        }
        // 3. Family alias resolution (e.g. `anthropic/sonnet` or `opus` -> newest version; prefer configured)
        let mut best: Option<(bool, Vec<u64>, String, String)> = None;
        for m in &all_models {
            if let Some(ref wp) = wanted_prov
                && m.provider != *wp
            {
                continue;
            }
            if let Some(ver) = family_version_tuple(&m.id, raw_model) {
                let better = best
                    .as_ref()
                    .is_none_or(|(bcfg, bv, _, _)| (m.configured, &ver) > (*bcfg, bv));
                if better {
                    best = Some((m.configured, ver, m.provider.clone(), m.id.clone()));
                }
            }
        }
        if let Some((_, _, prov, id)) = best {
            return Some((prov, id));
        }
        // 4. Fallback: explicit provider + literal model id (for arbitrary/new models not in catalog)
        if let Some(wp) = wanted_prov {
            return Some((wp, raw_model.to_string()));
        }
        cat.provider_for_model(raw_model)
            .map(|p| (p.to_string(), raw_model.to_string()))
    }

    /// Resolves `(provider_hint, model_hint)` against the layered catalog and builds a ready-to-use
    /// `Provider` target. Supports:
    ///   - `model` only -> auto-routes provider via `catalog.provider_for_model(model)` (fallback: `fallback_provider`)
    ///   - `provider` only -> selects that provider's `default_model`
    ///   - both `provider` + `model` -> explicit override
    pub fn resolve_and_build(
        &self,
        provider_hint: Option<&str>,
        model_hint: Option<&str>,
        fallback_provider: &str,
    ) -> Result<ResolvedModelTarget> {
        let model_trimmed = model_hint.map(str::trim).filter(|s| !s.is_empty());
        let prov_trimmed = provider_hint.map(str::trim).filter(|s| !s.is_empty());

        let selected_prov = if let Some(p) = prov_trimmed {
            p
        } else if let Some(m) = model_trimmed
            && let Some(routed) = self.catalog.provider_for_model(m)
        {
            routed
        } else {
            fallback_provider
        };

        let (canonical_prov, prov_entry) =
            self.catalog
                .find_provider(selected_prov)
                .ok_or_else(|| CarnotError::Provider {
                    provider: selected_prov.to_string(),
                    message: format!("unknown provider `{selected_prov}` in catalog"),
                    retryable: false,
                    retry_after_ms: None,
                })?;

        let resolved_model = if let Some(m) = model_trimmed {
            m.to_string()
        } else if let Some(ref def) = prov_entry.default_model {
            def.clone()
        } else {
            return Err(CarnotError::Provider {
                provider: canonical_prov.to_string(),
                message: format!(
                    "provider `{canonical_prov}` has no default_model; pass `model` explicitly"
                ),
                retryable: false,
                retry_after_ms: None,
            });
        };

        let provider = self.build_provider(canonical_prov, Some(&resolved_model))?;
        let context_window = self.context_window_for_model(&resolved_model);
        let max_output_tokens = self.max_output_tokens_for_model(&resolved_model);

        Ok(ResolvedModelTarget {
            provider_id: canonical_prov.to_string(),
            model_id: resolved_model,
            provider,
            context_window,
            max_output_tokens,
        })
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

fn family_version_tuple(model_id: &str, family: &str) -> Option<Vec<u64>> {
    let parts: Vec<&str> = model_id.split('-').collect();
    if parts.len() < 3 {
        return None;
    }
    let maker = parts[0];
    let name = parts[1];
    if name != family || !maker.is_ascii() || !maker.bytes().all(|b| b.is_ascii_lowercase()) {
        return None;
    }
    let mut ver = Vec::new();
    for p in &parts[2..] {
        if p.len() >= 8 || !p.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        ver.push(p.parse::<u64>().ok()?);
    }
    (!ver.is_empty()).then_some(ver)
}
