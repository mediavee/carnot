//! Anthropic Messages wire protocol adapter (`api = "anthropic-messages"`).
//!
//! Zero hardcoded provider names or URLs! Auth scheme (`Bearer` vs `x-api-key` — never both),
//! base URL, headers, `betas`, model capabilities, pricing, and `body_patch` all come from
//! `ResolvedProviderSpec` (`providers.json`).

use crate::catalog::{ProviderCatalog, ResolvedAuth, ResolvedProviderSpec};
use crate::compat::{can_replay_origin, is_retryable_status, parse_retry_after_ms};
use crate::registry::ApiAdapter;
use crate::{BoxFuture, Provider, ProviderRequest, StreamEvent};
use carnot_core::{
    CacheTtl, CarnotError, ContentBlock, ContentOrigin, Message, ReasoningEffort, Result, Role,
    StopReason, ToolResultBlock, Usage,
};
use futures::StreamExt;
use serde_json::{Value, json};
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

/// Hardened Anthropic Messages API provider (A1–A10 + Hybrid TTL + cache-diagnosis + TLS pre-warm).
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

    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        let catalog = ProviderCatalog::embedded().unwrap_or_default();
        let mut headers = BTreeMap::new();
        headers.insert("anthropic-version".into(), "2023-06-01".into());
        let spec = ResolvedProviderSpec {
            provider_id: "anthropic".into(),
            api: API_ID.into(),
            base_url: base_url.trim_end_matches('/').to_string(),
            fallback_base_urls: Vec::new(),
            project_id: None,
            auth: ResolvedAuth::Header {
                name: "x-api-key".into(),
                value: api_key,
                source: "direct".into(),
            },
            headers,
            betas: Vec::new(),
            cache_diagnosis_beta: Some("cache-diagnosis-2026-04-07".into()),
            native_tool_search: true,
            compat: Default::default(),
            usage_map: Default::default(),
            body_patch: None,
            subprocess: None,
            catalog_models: catalog.models,
        };
        Self::from_spec(spec)
    }

    pub fn from_spec(spec: ResolvedProviderSpec) -> Self {
        let client = crate::build_http_client_for_url(&spec.base_url);
        let cache_diagnosis = std::env::var("CARNOT_CACHE_DIAGNOSIS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        Self {
            client,
            spec,
            cache_diagnosis,
            last_message_id: Arc::new(std::sync::Mutex::new(None)),
        }
    }

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
        let url = format!("{}/v1/messages", self.spec.base_url);
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

    /// Builds the Anthropic JSON body with:
    ///   - 4-breakpoint cache planning (Hybrid 1h static / 5m rolling tail!) placed strictly on
    ///     the last NON-thinking block (A9)
    ///   - Native `defer_loading` + server `tool_search_tool_regex_20251119` when deferred tools exist
    ///   - `eager_input_streaming: true` on `Edit` and `Write` for lower TTFT on large edits
    ///   - Origin-aware replay: strips foreign `Thinking` signatures / foreign `Opaque` blocks!
    ///   - Valid Anthropic role alternation & native `role: "system"` placement (A6)
    ///   - `output_config.effort` + declarative `body_patch`
    pub fn build_body(&self, req: &ProviderRequest) -> Value {
        let cache_ctrl = |ttl: CacheTtl| match ttl {
            CacheTtl::Short5m => json!({"type": "ephemeral", "ttl": "5m"}),
            CacheTtl::Long1h | CacheTtl::Hybrid => json!({"type": "ephemeral", "ttl": "1h"}),
        };
        let static_cc = cache_ctrl(req.cache_ttl.static_prefix_tier());
        let conv_cc = cache_ctrl(req.cache_ttl.conversation_tier());

        let has_deferred = req.tools.iter().any(|t| t.defer_loading);
        let mut tools_json: Vec<Value> =
            Vec::with_capacity(req.tools.len() + usize::from(has_deferred));
        if has_deferred {
            tools_json.push(json!({
                "type": "tool_search_tool_regex_20251119",
                "name": "tool_search_tool_regex",
            }));
        }
        for t in &req.tools {
            let mut obj = json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            });
            if let Some(m) = obj.as_object_mut() {
                if t.defer_loading {
                    m.insert("defer_loading".into(), json!(true));
                }
                if matches!(t.name.as_str(), "Edit" | "Write") {
                    m.insert("eager_input_streaming".into(), json!(true));
                }
            }
            tools_json.push(obj);
        }
        // BP1: last non-deferred tool (static prefix tier -> 1h in Hybrid!)
        if let Some(last_non_deferred) = tools_json
            .iter_mut()
            .rev()
            .find(|v| v.get("defer_loading").and_then(Value::as_bool) != Some(true))
            && let Some(obj) = last_non_deferred.as_object_mut()
        {
            obj.insert("cache_control".to_string(), static_cc.clone());
        }

        // BP2: static system prompt (static prefix tier -> 1h in Hybrid!)
        let system_blocks = vec![json!({
            "type": "text",
            "text": req.static_system_prompt,
            "cache_control": static_cc,
        })];

        let allow_sys_role = self.supports_mid_system_role(&req.model);
        let mut wire_messages: Vec<(String, Vec<Value>, bool)> =
            Vec::with_capacity(req.messages.len());

        for (idx, msg) in req.messages.iter().enumerate() {
            let is_cp = req.checkpoint_index == Some(idx);
            let blocks = encode_blocks(&msg.content, &self.spec.provider_id, &req.model);
            if blocks.is_empty() {
                continue;
            }

            if msg.role == Role::System {
                let prev_is_user = wire_messages
                    .last()
                    .map(|(r, _, _)| r == "user")
                    .unwrap_or(false);
                let next_ok = req
                    .messages
                    .get(idx + 1)
                    .map(|m| m.role == Role::Assistant)
                    .unwrap_or(true);
                if allow_sys_role && prev_is_user && next_ok && !wire_messages.is_empty() {
                    wire_messages.push(("system".into(), blocks, is_cp));
                } else if let Some((r, prev_blocks, prev_cp)) = wire_messages.last_mut() {
                    if r == "user" {
                        prev_blocks.extend(blocks);
                        *prev_cp = *prev_cp || is_cp;
                    } else {
                        wire_messages.push(("user".into(), blocks, is_cp));
                    }
                } else {
                    wire_messages.push(("user".into(), blocks, is_cp));
                }
                continue;
            }

            let role_str = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => unreachable!(),
            };

            if let Some((prev_role, prev_blocks, prev_cp)) = wire_messages.last_mut()
                && prev_role == role_str
            {
                if role_str == "user" {
                    let mut merged = Vec::with_capacity(prev_blocks.len() + blocks.len());
                    merged.append(prev_blocks);
                    merged.extend(blocks);
                    let (trs, others): (Vec<Value>, Vec<Value>) =
                        merged.into_iter().partition(|b| {
                            b.get("type").and_then(Value::as_str) == Some("tool_result")
                        });
                    prev_blocks.extend(trs);
                    prev_blocks.extend(others);
                } else {
                    prev_blocks.extend(blocks);
                }
                *prev_cp = *prev_cp || is_cp;
                continue;
            }
            wire_messages.push((role_str.into(), blocks, is_cp));
        }

        let last_idx = wire_messages.len().saturating_sub(1);
        let final_messages: Vec<Value> = wire_messages
            .into_iter()
            .enumerate()
            .map(|(idx, (role, mut blocks, is_cp))| {
                let want_bp = (is_cp && idx != last_idx) || idx == last_idx;
                if want_bp {
                    // A9: attach cache_control ONLY to the last NON-thinking block!
                    if let Some(target) = blocks.iter_mut().rev().find(|b| {
                        !matches!(
                            b.get("type").and_then(Value::as_str),
                            Some("thinking" | "redacted_thinking")
                        )
                    }) && let Some(obj) = target.as_object_mut()
                    {
                        obj.insert("cache_control".into(), conv_cc.clone());
                    }
                }
                json!({"role": role, "content": blocks})
            })
            .collect();

        let mut body = json!({
            "model": req.model,
            "max_tokens": req.max_output_tokens,
            "stream": true,
            "system": system_blocks,
            "tools": tools_json,
            "messages": final_messages,
        });

        if let Some(m) = body.as_object_mut() {
            if self.supports_effort(&req.model) {
                let effort_str = match req.effort {
                    ReasoningEffort::Low => "low",
                    ReasoningEffort::Medium => "medium",
                    ReasoningEffort::High => "high",
                    ReasoningEffort::Xhigh => {
                        if self.supports_xhigh(&req.model) {
                            "xhigh"
                        } else {
                            "high"
                        }
                    }
                    ReasoningEffort::Max => "max",
                };
                m.insert("output_config".into(), json!({ "effort": effort_str }));
            }
            match self.thinking_mode(&req.model) {
                Some("adaptive") => {
                    m.insert("thinking".into(), json!({ "type": "adaptive" }));
                }
                Some("budget") => {
                    let desired: u32 = match req.effort {
                        ReasoningEffort::Low => 2_048,
                        ReasoningEffort::Medium => 6_144,
                        ReasoningEffort::High => 12_288,
                        ReasoningEffort::Xhigh => 20_480,
                        ReasoningEffort::Max => 31_999,
                    };
                    let max_allowed = req.max_output_tokens.saturating_sub(1);
                    if max_allowed >= 1_024 {
                        let budget = desired.clamp(1_024, max_allowed);
                        m.insert(
                            "thinking".into(),
                            json!({ "type": "enabled", "budget_tokens": budget }),
                        );
                    }
                }
                _ => {}
            }
            if self.cache_diagnosis
                && let Ok(guard) = self.last_message_id.lock()
                && let Some(ref prev_id) = *guard
            {
                m.insert(
                    "diagnostics".into(),
                    json!({ "previous_message_id": prev_id }),
                );
            }
        }
        self.spec.apply_body_patches(&req.model, &mut body);
        body
    }
}

fn encode_blocks(blocks: &[ContentBlock], provider_id: &str, model: &str) -> Vec<Value> {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(json!({"type": "text", "text": text})),
            ContentBlock::Thinking {
                thinking,
                signature,
                origin,
            } => {
                // Only replay signed thinking from THIS exact Anthropic `(api, provider, model)`
                // (legacy untagged blocks with a signature are also allowed for session resume).
                let ok = can_replay_origin(
                    origin.as_ref(),
                    API_ID,
                    provider_id,
                    model,
                    signature.is_some(),
                );
                if !ok {
                    return None;
                }
                let mut o = json!({"type": "thinking", "thinking": thinking});
                if let Some(sig) = signature
                    && let Some(m) = o.as_object_mut()
                {
                    m.insert("signature".into(), json!(sig));
                }
                Some(o)
            }
            ContentBlock::ToolUse {
                id, name, input, ..
            } => Some(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            })),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                let inner: Vec<Value> = content
                    .iter()
                    .map(|tb| match tb {
                        ToolResultBlock::Text { text } => json!({"type": "text", "text": text}),
                        ToolResultBlock::Image {
                            media_type,
                            data_base64,
                        } => json!({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data_base64,
                            }
                        }),
                    })
                    .collect();
                Some(json!({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": inner,
                    "is_error": is_error,
                }))
            }
            ContentBlock::Image {
                media_type,
                data_base64,
            } => Some(json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data_base64,
                }
            })),
            ContentBlock::Opaque { raw, origin } => {
                // Never replay foreign opaque blocks (e.g. OpenAI Responses encrypted reasoning)!
                let is_legacy_anthropic = origin.is_none()
                    && matches!(
                        raw.get("type").and_then(Value::as_str),
                        Some("redacted_thinking" | "server_tool_use" | "web_search_tool_result")
                    );
                if can_replay_origin(
                    origin.as_ref(),
                    API_ID,
                    provider_id,
                    model,
                    is_legacy_anthropic,
                ) {
                    Some(raw.clone())
                } else {
                    None
                }
            }
        })
        .collect()
}

/// Finds the next SSE frame delimiter (`\n\n` or `\r\n\r\n`) in raw bytes (A1 & A2 fix).
pub fn find_sse_frame_end(buf: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i + 1 < buf.len() {
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some((i, 2));
        }
        if i + 3 < buf.len()
            && buf[i] == b'\r'
            && buf[i + 1] == b'\n'
            && buf[i + 2] == b'\r'
            && buf[i + 3] == b'\n'
        {
            return Some((i, 4));
        }
        i += 1;
    }
    None
}

impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        &self.spec.provider_id
    }

    fn supports_native_tool_search(&self) -> bool {
        self.spec.native_tool_search
    }

    fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<mpsc::Receiver<StreamEvent>>> {
        Box::pin(async move {
            let url = self
                .spec
                .auth
                .apply_to_url(&format!("{}/v1/messages", self.spec.base_url));
            let body = self.build_body(&request);
            let mut req_builder = self
                .client
                .post(&url)
                .header("content-type", "application/json");
            // Explicit auth from data catalog (Bearer OR x-api-key, NEVER both!)
            req_builder = self.spec.auth.apply_to_request(req_builder);
            for (k, v) in &self.spec.headers {
                req_builder = req_builder.header(k.as_str(), v.as_str());
            }
            if !self.spec.headers.contains_key("anthropic-version") {
                req_builder = req_builder.header("anthropic-version", "2023-06-01");
            }
            // Data-driven betas (empty by default so gateways/Bedrock/Vertex never 400!)
            let mut betas = self.spec.betas.clone();
            if self.cache_diagnosis
                && let Some(ref diag_beta) = self.spec.cache_diagnosis_beta
                && !betas.iter().any(|b| b == diag_beta)
            {
                betas.push(diag_beta.clone());
            }
            if !betas.is_empty() {
                req_builder = req_builder.header("anthropic-beta", betas.join(","));
            }

            let send_fut = req_builder.json(&body).send();
            let resp = tokio::time::timeout(HTTP_HEADERS_TIMEOUT, send_fut)
                .await
                .map_err(|_| CarnotError::Provider {
                    provider: self.spec.provider_id.clone(),
                    message: "HTTP response headers timeout (300s)".into(),
                    retryable: true,
                    retry_after_ms: None,
                })?
                .map_err(|e| CarnotError::Provider {
                    provider: self.spec.provider_id.clone(),
                    message: e.without_url().to_string(),
                    retryable: true,
                    retry_after_ms: None,
                })?;

            let status = resp.status();
            if !status.is_success() {
                let retry_after_ms = parse_retry_after_ms(resp.headers());
                let text = resp.text().await.unwrap_or_default();
                let low = text.to_ascii_lowercase();
                if low.contains("prompt is too long")
                    || low.contains("context length")
                    || low.contains("too many tokens")
                    || low.contains("maximum context")
                {
                    return Err(CarnotError::ContextOverflow(text));
                }
                return Err(CarnotError::Provider {
                    provider: self.spec.provider_id.clone(),
                    message: format!("HTTP {status}: {text}"),
                    retryable: is_retryable_status(status.as_u16()),
                    retry_after_ms,
                });
            }

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
                        _ = cancel.cancelled() => {
                            let _ = tx.send(StreamEvent::Error { message: "cancelled".into(), retryable: false }).await;
                            return;
                        }
                        res = tokio::time::timeout(STREAM_IDLE_TIMEOUT, stream.next()) => res,
                    };

                    let chunk_opt = match next_chunk {
                        Ok(opt) => opt,
                        Err(_) => {
                            let _ = tx
                                .send(StreamEvent::Error {
                                    message: "stream idle timeout (300s)".into(),
                                    retryable: true,
                                })
                                .await;
                            return;
                        }
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
                            let _ = tx
                                .send(StreamEvent::Error {
                                    message: err_msg,
                                    retryable,
                                })
                                .await;
                            return;
                        }
                    }
                }

                // A4: EOF without `message_stop` is a truncated stream error, NEVER a silent success!
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
                    .filter_map(|b| b.finish(&origin))
                    .collect();
                let _ = tx
                    .send(StreamEvent::Done {
                        message: Message {
                            role: Role::Assistant,
                            content,
                        },
                        usage: state.usage,
                        stop_reason: state.stop_reason,
                    })
                    .await;
            });

            Ok(rx)
        })
    }
}

#[derive(Default)]
struct SseParseState {
    blocks: BTreeMap<u64, BlockBuilder>,
    usage: Usage,
    stop_reason: StopReason,
    saw_message_stop: bool,
    message_id: Option<String>,
}

enum BlockBuilder {
    Text(String),
    Thinking {
        text: String,
        sig: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        json_buf: String,
    },
    Opaque {
        raw: Value,
        json_buf: String,
    },
}

impl BlockBuilder {
    fn finish(self, origin: &ContentOrigin) -> Option<ContentBlock> {
        match self {
            Self::Text(text) => Some(ContentBlock::Text { text }),
            Self::Thinking { text, sig } => Some(ContentBlock::Thinking {
                thinking: text,
                signature: sig,
                origin: Some(origin.clone()),
            }),
            Self::ToolUse { id, name, json_buf } => {
                let trimmed = json_buf.trim();
                let (input, invalid_json) = if trimmed.is_empty() {
                    (json!({}), None)
                } else {
                    match serde_json::from_str::<Value>(trimmed) {
                        Ok(v) => (v, None),
                        Err(e) => (
                            json!({}),
                            Some(format!("malformed tool JSON ({e}): {trimmed}")),
                        ),
                    }
                };
                Some(ContentBlock::ToolUse {
                    id,
                    name,
                    input,
                    invalid_json,
                    provider_meta: None,
                    origin: Some(origin.clone()),
                })
            }
            Self::Opaque { mut raw, json_buf } => {
                let trimmed = json_buf.trim();
                if !trimmed.is_empty()
                    && let Ok(parsed) = serde_json::from_str::<Value>(trimmed)
                    && let Some(obj) = raw.as_object_mut()
                {
                    obj.insert("input".into(), parsed);
                }
                Some(ContentBlock::Opaque {
                    raw,
                    origin: Some(origin.clone()),
                })
            }
        }
    }
}

async fn handle_sse_frame(
    frame: &str,
    state: &mut SseParseState,
    default_ttl: CacheTtl,
    tx: &mpsc::Sender<StreamEvent>,
) -> Option<(String, bool)> {
    let mut data_parts: Vec<&str> = Vec::new();
    for line in frame.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(rest) = line.strip_prefix("data:") {
            data_parts.push(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    if data_parts.is_empty() {
        return None;
    }
    let joined = data_parts.join("\n");
    let Ok(v) = serde_json::from_str::<Value>(&joined) else {
        return None;
    };

    match v.get("type").and_then(Value::as_str) {
        Some("error") => {
            let err_type = v
                .pointer("/error/type")
                .and_then(Value::as_str)
                .unwrap_or("api_error");
            let msg = v
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("stream error");
            let retryable = matches!(
                err_type,
                "overloaded_error" | "rate_limit_error" | "api_error"
            );
            return Some((format!("{err_type}: {msg}"), retryable));
        }
        Some("message_stop") => {
            state.saw_message_stop = true;
        }
        Some("message_start") => {
            if let Some(mid) = v.pointer("/message/id").and_then(Value::as_str) {
                state.message_id = Some(mid.to_string());
            }
            if let Some(diag) = v
                .get("diagnostics")
                .or_else(|| v.pointer("/message/diagnostics"))
            {
                tracing::info!(
                    target: "carnot::cache_diagnosis",
                    diagnostics = %diag,
                    "anthropic_cache_diagnosis"
                );
            }
            if let Some(u) = v.pointer("/message/usage") {
                state.usage.input_tokens +=
                    u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
                state.usage.cache_read_tokens += u
                    .get("cache_read_input_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let cw_5m = u
                    .pointer("/cache_creation/ephemeral_5m_input_tokens")
                    .and_then(Value::as_u64);
                let cw_1h = u
                    .pointer("/cache_creation/ephemeral_1h_input_tokens")
                    .and_then(Value::as_u64);
                if cw_5m.is_some() || cw_1h.is_some() {
                    state.usage.cache_write_5m_tokens += cw_5m.unwrap_or(0);
                    state.usage.cache_write_1h_tokens += cw_1h.unwrap_or(0);
                } else {
                    let cw = u
                        .get("cache_creation_input_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    match default_ttl.conversation_tier() {
                        CacheTtl::Short5m | CacheTtl::Hybrid => {
                            state.usage.cache_write_5m_tokens += cw;
                        }
                        CacheTtl::Long1h => state.usage.cache_write_1h_tokens += cw,
                    }
                }
            }
        }
        Some("content_block_start") => {
            let idx = v.get("index").and_then(Value::as_u64).unwrap_or(0);
            if let Some(cb) = v.get("content_block") {
                let builder = match cb.get("type").and_then(Value::as_str) {
                    Some("text") => BlockBuilder::Text(
                        cb.get("text")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                    ),
                    Some("thinking") => BlockBuilder::Thinking {
                        text: cb
                            .get("thinking")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        sig: cb
                            .get("signature")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    },
                    Some("tool_use") => BlockBuilder::ToolUse {
                        id: cb
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        name: cb
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        json_buf: String::new(),
                    },
                    _ => BlockBuilder::Opaque {
                        raw: cb.clone(),
                        json_buf: String::new(),
                    },
                };
                state.blocks.insert(idx, builder);
            }
        }
        Some("content_block_delta") => {
            let idx = v.get("index").and_then(Value::as_u64).unwrap_or(0);
            let delta = v.get("delta")?;
            if let Some(b) = state.blocks.get_mut(&idx) {
                match (b, delta.get("type").and_then(Value::as_str)) {
                    (BlockBuilder::Text(s), Some("text_delta")) => {
                        if let Some(t) = delta.get("text").and_then(Value::as_str) {
                            s.push_str(t);
                            let _ = tx.send(StreamEvent::TextDelta(t.to_string())).await;
                        }
                    }
                    (BlockBuilder::Thinking { text, .. }, Some("thinking_delta")) => {
                        if let Some(t) = delta.get("thinking").and_then(Value::as_str) {
                            text.push_str(t);
                            let _ = tx.send(StreamEvent::ThinkingDelta(t.to_string())).await;
                        }
                    }
                    (BlockBuilder::Thinking { sig, .. }, Some("signature_delta")) => {
                        if let Some(s) = delta.get("signature").and_then(Value::as_str) {
                            *sig = Some(s.to_string());
                        }
                    }
                    (BlockBuilder::ToolUse { json_buf, .. }, Some("input_json_delta"))
                    | (BlockBuilder::Opaque { json_buf, .. }, Some("input_json_delta")) => {
                        if let Some(p) = delta.get("partial_json").and_then(Value::as_str) {
                            json_buf.push_str(p);
                        }
                    }
                    _ => {}
                }
            }
        }
        Some("content_block_stop") => {
            let idx = v.get("index").and_then(Value::as_u64).unwrap_or(0);
            if let Some(BlockBuilder::ToolUse { id, name, json_buf }) = state.blocks.get(&idx) {
                let trimmed = json_buf.trim();
                let (input, invalid_json) = if trimmed.is_empty() {
                    (json!({}), None)
                } else {
                    match serde_json::from_str::<Value>(trimmed) {
                        Ok(val) => (val, None),
                        Err(e) => (
                            json!({}),
                            Some(format!("malformed tool JSON ({e}): {trimmed}")),
                        ),
                    }
                };
                let _ = tx
                    .send(StreamEvent::ToolCallReady {
                        id: id.clone(),
                        name: name.clone(),
                        input,
                        invalid_json,
                    })
                    .await;
            }
        }
        Some("message_delta") => {
            if let Some(sr) = v.pointer("/delta/stop_reason").and_then(Value::as_str) {
                state.stop_reason = match sr {
                    "tool_use" => StopReason::ToolUse,
                    "max_tokens" => StopReason::MaxTokens,
                    "refusal" => StopReason::Refusal,
                    "pause_turn" => StopReason::PauseTurn,
                    "model_context_window_exceeded" => StopReason::ContextWindowExceeded,
                    _ => StopReason::EndTurn,
                };
            }
            if let Some(ot) = v.pointer("/usage/output_tokens").and_then(Value::as_u64) {
                state.usage.output_tokens = ot;
            }
            if let Some(tt) = v
                .pointer("/usage/output_tokens_details/thinking_tokens")
                .and_then(Value::as_u64)
            {
                state.usage.reasoning_tokens = tt;
            }
        }
        _ => {}
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use carnot_core::ToolSpec;

    #[test]
    fn places_four_cache_breakpoints_and_skips_thinking_blocks() {
        let p = AnthropicProvider::new("test-key".into());
        let req = ProviderRequest {
            model: "claude-opus-5-5".into(),
            static_system_prompt: "STATIC".into(),
            tools: vec![ToolSpec {
                name: "Read".into(),
                description: "Read file".into(),
                input_schema: json!({"type": "object"}),
                defer_loading: false,
            }],
            prefix_hash: "abc".into(),
            messages: vec![
                Arc::new(Message::user_text("cp")),
                Arc::new(Message::system_note("receipt")),
            ],
            checkpoint_index: Some(0),
            cache_ttl: CacheTtl::Long1h,
            effort: ReasoningEffort::High,
            max_output_tokens: 32_768,
        };
        let body = p.build_body(&req);
        assert!(body.pointer("/tools/0/cache_control").is_some());
        assert!(body.pointer("/system/0/cache_control").is_some());
        assert!(
            body.pointer("/messages/0/content/0/cache_control")
                .is_some()
        );
        assert!(
            body.pointer("/messages/1/content/0/cache_control")
                .is_some()
        );
        assert_eq!(
            body.pointer("/messages/1/role").and_then(Value::as_str),
            Some("system")
        );
    }

    #[test]
    fn sse_frame_finder_handles_crlf_and_lf() {
        assert_eq!(find_sse_frame_end(b"data: {}\n\nrest"), Some((8, 2)));
        assert_eq!(find_sse_frame_end(b"data: {}\r\n\r\nrest"), Some((8, 4)));
    }
}
