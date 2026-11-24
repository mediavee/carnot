use crate::catalog::ResolvedProviderSpec;
use crate::registry::ApiAdapter;
use crate::{BoxFuture, Provider, ProviderRequest, StreamEvent};

/// Protocol adapter registering `"faux"` through the standard `ProviderRegistry`.
#[derive(Debug)]
pub struct FauxApiAdapter;

impl ApiAdapter for FauxApiAdapter {
    fn api_id(&self) -> &'static str {
        "faux"
    }

    fn build(&self, _spec: &ResolvedProviderSpec) -> Result<Arc<dyn Provider>> {
        if let Ok(raw) = std::env::var("CARNOT_FAUX_SCRIPT")
            && !raw.trim().is_empty()
        {
            let trimmed = raw.trim();
            let json_text = if trimmed.starts_with('[') {
                trimmed.to_string()
            } else {
                std::fs::read_to_string(trimmed)?
            };
            let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_text).map_err(|e| {
                carnot_core::CarnotError::Provider {
                    provider: "faux".into(),
                    message: format!("invalid CARNOT_FAUX_SCRIPT JSON: {e}"),
                    retryable: false,
                    retry_after_ms: None,
                }
            })?;
            let mut turns = Vec::with_capacity(parsed.len());
            for item in parsed {
                let mut blocks = Vec::new();
                if let Some(thinking) = item.get("thinking").and_then(|v| v.as_str())
                    && !thinking.is_empty()
                {
                    blocks.push(ContentBlock::Thinking {
                        thinking: thinking.to_string(),
                        signature: None,
                        origin: None,
                    });
                }
                if let Some(text) = item.get("text").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    blocks.push(ContentBlock::Text {
                        text: text.to_string(),
                    });
                }
                let mut had_tools = false;
                if let Some(calls) = item.get("tool_calls").and_then(|v| v.as_array()) {
                    for (idx, tc) in calls.iter().enumerate() {
                        let id = tc
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(ToString::to_string)
                            .unwrap_or_else(|| format!("faux-call-{idx}"));
                        let name = tc
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let input = tc
                            .get("input")
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!({}));
                        if !name.is_empty() {
                            had_tools = true;
                            blocks.push(ContentBlock::ToolUse {
                                id,
                                name,
                                input,
                                invalid_json: None,
                                provider_meta: None,
                                origin: None,
                            });
                        }
                    }
                }
                let stop_reason = match item.get("stop_reason").and_then(|v| v.as_str()) {
                    Some("tool_use") => StopReason::ToolUse,
                    Some("max_tokens") => StopReason::MaxTokens,
                    Some("refusal") => StopReason::Refusal,
                    Some("end_turn") => StopReason::EndTurn,
                    _ if had_tools => StopReason::ToolUse,
                    _ => StopReason::EndTurn,
                };
                turns.push(ScriptedTurn {
                    blocks,
                    stop_reason,
                    output_tokens: item
                        .get("output_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(24),
                });
            }
            if !turns.is_empty() {
                return Ok(Arc::new(FauxProvider::new(turns)));
            }
        }
        Ok(Arc::new(FauxProvider::new(vec![ScriptedTurn::text(
            "Carnot faux provider ready.",
        )])))
    }
}
use carnot_core::{
    CacheTtl, ContentBlock, Message, ModelPricing, ProviderWebCapabilities, Result, Role,
    StopReason, Usage,
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Scripted reply returned by `FauxProvider` for deterministic, zero-cost E2E tests.
#[derive(Debug, Clone)]
pub struct ScriptedTurn {
    pub blocks: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub output_tokens: u64,
}

impl ScriptedTurn {
    pub fn text(s: impl Into<String>) -> Self {
        let txt = s.into();
        let ot = (txt.len() as u64 / 4).max(16);
        Self {
            blocks: vec![ContentBlock::Text { text: txt }],
            stop_reason: StopReason::EndTurn,
            output_tokens: ot,
        }
    }

    pub fn tool_calls(calls: Vec<(&str, &str, serde_json::Value)>) -> Self {
        let blocks = calls
            .into_iter()
            .map(|(id, name, input)| ContentBlock::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input,
                invalid_json: None,
                provider_meta: None,
                origin: None,
            })
            .collect();
        Self {
            blocks,
            stop_reason: StopReason::ToolUse,
            output_tokens: 90,
        }
    }
}

/// Deterministic local provider simulating exact Anthropic prefix-cache accounting & streaming.
#[derive(Debug, Clone)]
pub struct FauxProvider {
    turns: Arc<Mutex<VecDeque<ScriptedTurn>>>,
    seen_requests: Arc<Mutex<Vec<ProviderRequest>>>,
    cached_prefix_tokens: Arc<Mutex<u64>>,
    last_prefix_hash: Arc<Mutex<Option<String>>>,
    web_caps: ProviderWebCapabilities,
}

impl FauxProvider {
    pub fn new(turns: Vec<ScriptedTurn>) -> Self {
        Self {
            turns: Arc::new(Mutex::new(turns.into())),
            seen_requests: Arc::new(Mutex::new(Vec::new())),
            cached_prefix_tokens: Arc::new(Mutex::new(0)),
            last_prefix_hash: Arc::new(Mutex::new(None)),
            web_caps: ProviderWebCapabilities::default(),
        }
    }

    pub fn with_web_capabilities(mut self, caps: ProviderWebCapabilities) -> Self {
        self.web_caps = caps;
        self
    }

    pub fn recorded_requests(&self) -> Vec<ProviderRequest> {
        self.seen_requests
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }
}

impl Provider for FauxProvider {
    fn name(&self) -> &str {
        "faux"
    }

    fn web_capabilities(&self, _model: &str) -> ProviderWebCapabilities {
        self.web_caps
    }

    fn stream(
        &self,
        request: ProviderRequest,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<mpsc::Receiver<StreamEvent>>> {
        Box::pin(async move {
            let tools_bytes: usize = request
                .tools
                .iter()
                .filter(|t| !t.defer_loading)
                .map(|t| {
                    t.name.len()
                        + t.description.len()
                        + serde_json::to_vec(&t.input_schema)
                            .map(|b| b.len())
                            .unwrap_or(0)
                })
                .sum();
            let static_est: u64 =
                ((request.static_system_prompt.len() + tools_bytes) as u64 / 4).max(1);
            let est_ctx: u64 = static_est
                + request
                    .messages
                    .iter()
                    .map(|m| m.estimated_tokens())
                    .sum::<u64>();

            if let Ok(mut seen) = self.seen_requests.lock() {
                seen.push(request.clone());
            }

            // Simulate realistic prefix cache hits & writes!
            let (cache_read, cache_write) = {
                let mut prev_h = self
                    .last_prefix_hash
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let mut prev_tok = self
                    .cached_prefix_tokens
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let same_static = prev_h.as_deref() == Some(request.prefix_hash.as_str());
                let cr = if same_static {
                    (*prev_tok).min(est_ctx)
                } else {
                    0
                };
                let cw = est_ctx.saturating_sub(cr);
                *prev_h = Some(request.prefix_hash.clone());
                *prev_tok = est_ctx;
                (cr, cw)
            };

            let next_turn = self
                .turns
                .lock()
                .ok()
                .and_then(|mut q| q.pop_front())
                .unwrap_or_else(|| ScriptedTurn::text("Done."));

            let (tx, rx) = mpsc::channel(32);
            let pricing = ModelPricing::for_model(&request.model);
            let ttl = request.cache_ttl;

            tokio::spawn(async move {
                for b in &next_turn.blocks {
                    match b {
                        ContentBlock::Text { text } => {
                            let _ = tx.send(StreamEvent::TextDelta(text.clone())).await;
                        }
                        ContentBlock::Thinking { thinking, .. } => {
                            let _ = tx.send(StreamEvent::ThinkingDelta(thinking.clone())).await;
                        }
                        ContentBlock::ToolUse {
                            id,
                            name,
                            input,
                            invalid_json,
                            ..
                        } => {
                            let _ = tx
                                .send(StreamEvent::ToolCallReady {
                                    id: id.clone(),
                                    name: name.clone(),
                                    input: input.clone(),
                                    invalid_json: invalid_json.clone(),
                                })
                                .await;
                        }
                        _ => {}
                    }
                }
                let (cw_5m, cw_1h) = match ttl {
                    CacheTtl::Short5m => (cache_write, 0),
                    CacheTtl::Long1h => (0, cache_write),
                    // Hybrid: static prefix (exact prompt+tools size on Turn 1) billed at 1h, rolling turns at 5m!
                    CacheTtl::Hybrid => {
                        if cache_read == 0 {
                            let static_tok = static_est.min(cache_write);
                            (cache_write.saturating_sub(static_tok), static_tok)
                        } else {
                            (cache_write, 0)
                        }
                    }
                };
                let mut usage = Usage {
                    input_tokens: 0,
                    output_tokens: next_turn.output_tokens,
                    reasoning_tokens: 0,
                    cache_read_tokens: cache_read,
                    cache_write_5m_tokens: cw_5m,
                    cache_write_1h_tokens: cw_1h,
                    cost_usd: 0.0,
                };
                usage.compute_cost(pricing);
                let _ = tx
                    .send(StreamEvent::Done {
                        message: Message {
                            role: Role::Assistant,
                            content: next_turn.blocks,
                        },
                        usage,
                        stop_reason: next_turn.stop_reason,
                    })
                    .await;
            });

            Ok(rx)
        })
    }
}
