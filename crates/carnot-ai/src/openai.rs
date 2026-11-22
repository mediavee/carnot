//! Declarative OpenAI Chat Completions wire protocol adapter (`api = "openai-completions"`).
//!
//! Zero hardcoded provider names, hostnames, or `detect_compat` URL heuristics!
//! Driven 100% by `CompatFlags`, `UsageMap`, `headers`, and `body_patch` from `providers.json`.
//!
//! Full fidelity (Opus Phase 5 fixes):
//! - Multimodal user images (`image_url` data URLs) AND automatic `user` bridge message for
//!   images returned inside `ToolResult` (since the `tool` role only accepts text!).
//! - Streams `reasoning_content` / `reasoning` deltas -> `ThinkingDelta` + persists `Thinking`
//!   with `origin`, and replays `reasoning_content` on assistant messages when configured.
//! - Declarative `max_tokens_field` (`max_completion_tokens` vs `max_tokens`), `developer` role,
//!   `reasoning_effort` vs `reasoning: { effort }`, `prompt_cache_key`, `session_id_header`, `body_patch`.
//! - Accurate declarative `UsageMap` (handles OpenAI, xAI, DeepSeek top-level cache hit, OpenRouter).
//! - `Retry-After` parsing, retries on `408|429|500|502|503|504|529`, and non-retryable stream error classification.

use crate::catalog::{ProviderCatalog, ResolvedAuth, ResolvedProviderSpec};
use crate::compat::{
    apply_usage_map, can_replay_origin, is_retryable_status, parse_retry_after_ms,
};
use crate::registry::ApiAdapter;
use crate::{BoxFuture, Provider, ProviderRequest, StreamEvent};
use carnot_core::{
    CarnotError, ContentBlock, ContentOrigin, Message, ReasoningEffort, Result, Role, StopReason,
    ToolResultBlock, Usage,
};
use futures::StreamExt;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const API_ID: &str = "openai-completions";
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug)]
pub struct OpenAiCompletionsAdapter;

impl ApiAdapter for OpenAiCompletionsAdapter {
    fn api_id(&self) -> &'static str {
        API_ID
    }

    fn build(&self, spec: &ResolvedProviderSpec) -> Result<Arc<dyn Provider>> {
        Ok(Arc::new(OpenAiProvider::from_spec(spec.clone())))
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    client: reqwest::Client,
    spec: ResolvedProviderSpec,
}

impl OpenAiProvider {
    pub fn new(api_key: String) -> Self {
        let base_url = ProviderCatalog::embedded()
            .ok()
            .and_then(|c| c.find_provider("openai").map(|(_, e)| e.base_url.clone()))
            .unwrap_or_default();
        Self::with_base_url(api_key, base_url, "openai".into())
    }

    pub fn with_base_url(api_key: String, base_url: String, provider_label: String) -> Self {
        let catalog = ProviderCatalog::embedded().unwrap_or_default();
        let entry = catalog
            .find_provider(&provider_label)
            .map(|(_, e)| e.clone());
        let spec = ResolvedProviderSpec {
            provider_id: provider_label,
            api: API_ID.into(),
            base_url: base_url.trim_end_matches('/').to_string(),
            fallback_base_urls: Vec::new(),
            project_id: None,
            auth: ResolvedAuth::Bearer {
                token: api_key,
                source: "direct".into(),
            },
            headers: entry
                .as_ref()
                .map(|e| e.headers.clone())
                .unwrap_or_default(),
            betas: Vec::new(),
            cache_diagnosis_beta: None,
            native_tool_search: false,
            compat: entry.as_ref().map(|e| e.compat.clone()).unwrap_or_default(),
            usage_map: entry
                .as_ref()
                .map(|e| e.usage_map.clone())
                .unwrap_or_default(),
            body_patch: entry.as_ref().and_then(|e| e.body_patch.clone()),
            subprocess: None,
            catalog_models: catalog.models,
        };
        Self::from_spec(spec)
    }

    pub fn from_spec(spec: ResolvedProviderSpec) -> Self {
        let client = crate::build_http_client_for_url(&spec.base_url);
        Self { client, spec }
    }

    pub fn build_body(&self, req: &ProviderRequest) -> Value {
        let compat = self.spec.effective_compat(&req.model);
        let sys_role = if compat.supports_developer_role.unwrap_or(false) {
            "developer"
        } else {
            "system"
        };

        let mut messages = vec![json!({
            "role": sys_role,
            "content": req.static_system_prompt,
        })];

        for msg in &req.messages {
            match msg.role {
                Role::System => {
                    let txt = msg.text_concat();
                    if !txt.is_empty() {
                        messages.push(json!({"role": sys_role, "content": txt}));
                    }
                }
                Role::User => {
                    let mut pushed_tool = false;
                    let mut tool_bridge_images: Vec<Value> = Vec::new();
                    for b in &msg.content {
                        if let ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } = b
                        {
                            let mut text_parts = Vec::new();
                            for tb in content {
                                match tb {
                                    ToolResultBlock::Text { text } => {
                                        text_parts.push(text.as_str())
                                    }
                                    ToolResultBlock::Image {
                                        media_type,
                                        data_base64,
                                    } => {
                                        text_parts
                                            .push("[image attached in following user message]");
                                        tool_bridge_images.push(json!({
                                            "type": "image_url",
                                            "image_url": {
                                                "url": format!("data:{media_type};base64,{data_base64}")
                                            }
                                        }));
                                    }
                                }
                            }
                            messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": text_parts.join("\n"),
                            }));
                            pushed_tool = true;
                        }
                    }

                    // Bridge message for tool-result images (`role: "tool"` cannot carry image_url!)
                    if !tool_bridge_images.is_empty() {
                        let mut parts = vec![json!({
                            "type": "text",
                            "text": "Image(s) returned by the preceding tool call:"
                        })];
                        parts.extend(tool_bridge_images);
                        messages.push(json!({"role": "user", "content": parts}));
                    }

                    let has_user_image = msg
                        .content
                        .iter()
                        .any(|b| matches!(b, ContentBlock::Image { .. }));
                    if has_user_image {
                        let mut parts: Vec<Value> = Vec::new();
                        for b in &msg.content {
                            match b {
                                ContentBlock::Text { text } if !text.is_empty() => {
                                    parts.push(json!({"type": "text", "text": text}));
                                }
                                ContentBlock::Image {
                                    media_type,
                                    data_base64,
                                } => {
                                    parts.push(json!({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": format!("data:{media_type};base64,{data_base64}")
                                        }
                                    }));
                                }
                                _ => {}
                            }
                        }
                        if !parts.is_empty() {
                            messages.push(json!({"role": "user", "content": parts}));
                        }
                    } else {
                        let txt = msg.text_concat();
                        if !txt.is_empty() || !pushed_tool {
                            messages.push(json!({"role": "user", "content": txt}));
                        }
                    }
                }
                Role::Assistant => {
                    let txt = msg.text_concat();
                    let reasoning_txt: String = if compat
                        .requires_reasoning_content_on_assistant_messages
                        .unwrap_or(false)
                    {
                        msg.content
                            .iter()
                            .filter_map(|b| match b {
                                ContentBlock::Thinking {
                                    thinking, origin, ..
                                } if can_replay_origin(
                                    origin.as_ref(),
                                    API_ID,
                                    &self.spec.provider_id,
                                    &req.model,
                                    true,
                                ) =>
                                {
                                    Some(thinking.as_str())
                                }
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    } else {
                        String::new()
                    };

                    let tool_calls: Vec<Value> = msg
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::ToolUse {
                                id, name, input, ..
                            } => Some(json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": input.to_string(),
                                }
                            })),
                            _ => None,
                        })
                        .collect();

                    let mut obj = json!({
                        "role": "assistant",
                        "content": if txt.is_empty() { Value::Null } else { json!(txt) }
                    });
                    if let Some(m) = obj.as_object_mut() {
                        if !reasoning_txt.is_empty() {
                            m.insert("reasoning_content".into(), json!(reasoning_txt));
                        }
                        if !tool_calls.is_empty() {
                            m.insert("tool_calls".into(), json!(tool_calls));
                        }
                    }
                    messages.push(obj);
                }
            }
        }

        let tools: Vec<Value> = req
            .tools
            .iter()
            .filter(|t| !t.defer_loading)
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                })
            })
            .collect();

        let max_tokens_key = compat
            .max_tokens_field
            .as_deref()
            .unwrap_or("max_completion_tokens");
        let mut body = json!({
            "model": req.model,
            "stream": true,
            "messages": messages,
        });

        if let Some(m) = body.as_object_mut() {
            m.insert(max_tokens_key.into(), json!(req.max_output_tokens));
            if compat.send_stream_options.unwrap_or(true) {
                m.insert("stream_options".into(), json!({"include_usage": true}));
            }
            if compat.supports_prompt_cache_key.unwrap_or(false) {
                m.insert(
                    "prompt_cache_key".into(),
                    json!(&req.prefix_hash[..16.min(req.prefix_hash.len())]),
                );
            }
            let effort_str = match req.effort {
                ReasoningEffort::Low => "low",
                ReasoningEffort::Medium => "medium",
                ReasoningEffort::High | ReasoningEffort::Xhigh | ReasoningEffort::Max => "high",
            };
            if compat.supports_reasoning_effort.unwrap_or(false) {
                m.insert("reasoning_effort".into(), json!(effort_str));
            } else if compat.reasoning_object_effort.unwrap_or(false) {
                m.insert("reasoning".into(), json!({ "effort": effort_str }));
            }
            if !tools.is_empty() {
                m.insert("tools".into(), json!(tools));
            }
        }

        self.spec.apply_body_patches(&req.model, &mut body);
        body
    }
}

impl Provider for OpenAiProvider {
    fn name(&self) -> &str {
        &self.spec.provider_id
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
                .apply_to_url(&format!("{}/chat/completions", self.spec.base_url));
            let body = self.build_body(&request);
            let compat = self.spec.effective_compat(&request.model);

            let mut rb = self
                .client
                .post(&url)
                .header("content-type", "application/json");
            rb = self.spec.auth.apply_to_request(rb);
            for (k, v) in &self.spec.headers {
                rb = rb.header(k.as_str(), v.as_str());
            }
            if let Some(ref hdr) = compat.session_id_header {
                rb = rb.header(
                    hdr.as_str(),
                    &request.prefix_hash[..16.min(request.prefix_hash.len())],
                );
            }

            let resp = tokio::time::timeout(Duration::from_secs(90), rb.json(&body).send())
                .await
                .map_err(|_| CarnotError::Provider {
                    provider: self.spec.provider_id.clone(),
                    message: "HTTP response headers timeout (90s)".into(),
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
                let lower = text.to_ascii_lowercase();
                if lower.contains("context_length_exceeded")
                    || lower.contains("maximum context length")
                    || lower.contains("prompt is too long")
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
            let usage_map = self.spec.usage_map.clone();
            let origin = ContentOrigin::new(API_ID, &self.spec.provider_id, &request.model);

            tokio::spawn(async move {
                let mut stream = resp.bytes_stream();
                let mut raw_buf: Vec<u8> = Vec::with_capacity(8192);
                let mut text_acc = String::new();
                let mut reasoning_acc = String::new();
                let mut tool_calls: BTreeMap<usize, (String, String, String)> = BTreeMap::new();
                let mut usage = Usage::default();
                let mut stop_reason = StopReason::EndTurn;
                let mut saw_done_or_finish = false;

                'stream_loop: loop {
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            let _ = tx.send(StreamEvent::Error { message: "cancelled".into(), retryable: false }).await;
                            return;
                        }
                        chunk = tokio::time::timeout(STREAM_IDLE_TIMEOUT, stream.next()) => {
                            let Ok(chunk_opt) = chunk else {
                                let _ = tx.send(StreamEvent::Error { message: "stream idle timeout (120s)".into(), retryable: true }).await;
                                return;
                            };
                            let Some(chunk) = chunk_opt else { break 'stream_loop };
                            let Ok(bytes) = chunk else {
                                let _ = tx.send(StreamEvent::Error { message: "stream read error".into(), retryable: true }).await;
                                return;
                            };
                            raw_buf.extend_from_slice(&bytes);
                            while let Some(pos) = raw_buf.iter().position(|&b| b == b'\n') {
                                let line_bytes: Vec<u8> = raw_buf.drain(..=pos).collect();
                                let line = String::from_utf8_lossy(&line_bytes).trim().to_string();
                                let Some(data) = line.strip_prefix("data:") else { continue };
                                let data = data.trim();
                                if data == "[DONE]" {
                                    saw_done_or_finish = true;
                                    break 'stream_loop;
                                }
                                let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
                                if let Some(err) = v.get("error") {
                                    let err_type = err
                                        .get("type")
                                        .or_else(|| err.get("code"))
                                        .and_then(Value::as_str)
                                        .unwrap_or("");
                                    let retryable = !matches!(
                                        err_type,
                                        "invalid_request_error"
                                            | "authentication_error"
                                            | "permission_error"
                                            | "context_length_exceeded"
                                            | "insufficient_quota"
                                    );
                                    let _ = tx.send(StreamEvent::Error {
                                        message: err.to_string(),
                                        retryable,
                                    }).await;
                                    return;
                                }
                                if let Some(u) = v.get("usage")
                                    && u.is_object()
                                {
                                    apply_usage_map(u, &usage_map, &mut usage);
                                }
                                if let Some(choice) = v.pointer("/choices/0") {
                                    if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
                                        saw_done_or_finish = true;
                                        stop_reason = match fr {
                                            "tool_calls" => StopReason::ToolUse,
                                            "length" => StopReason::MaxTokens,
                                            "content_filter" => StopReason::Refusal,
                                            _ => StopReason::EndTurn,
                                        };
                                    }
                                    if let Some(delta) = choice.get("delta") {
                                        if let Some(r) = delta
                                            .get("reasoning_content")
                                            .or_else(|| delta.get("reasoning"))
                                            .and_then(Value::as_str)
                                            && !r.is_empty()
                                        {
                                            reasoning_acc.push_str(r);
                                            let _ = tx
                                                .send(StreamEvent::ThinkingDelta(r.to_string()))
                                                .await;
                                        }
                                        if let Some(t) = delta.get("content").and_then(Value::as_str)
                                            && !t.is_empty()
                                        {
                                            text_acc.push_str(t);
                                            let _ = tx.send(StreamEvent::TextDelta(t.to_string())).await;
                                        }
                                        if let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array) {
                                            for tc in tcs {
                                                let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                                                let entry = tool_calls.entry(idx).or_insert_with(|| {
                                                    (String::new(), String::new(), String::new())
                                                });
                                                if let Some(id) = tc.get("id").and_then(Value::as_str)
                                                    && !id.is_empty()
                                                {
                                                    entry.0 = id.to_string();
                                                }
                                                if let Some(n) = tc.pointer("/function/name").and_then(Value::as_str) {
                                                    entry.1.push_str(n);
                                                }
                                                if let Some(a) = tc.pointer("/function/arguments").and_then(Value::as_str) {
                                                    entry.2.push_str(a);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if !saw_done_or_finish {
                    let _ = tx
                        .send(StreamEvent::Error {
                            message: "stream ended without finish_reason or [DONE]".into(),
                            retryable: true,
                        })
                        .await;
                    return;
                }

                let mut blocks = Vec::new();
                if !reasoning_acc.is_empty() {
                    blocks.push(ContentBlock::Thinking {
                        thinking: reasoning_acc,
                        signature: None,
                        origin: Some(origin.clone()),
                    });
                }
                if !text_acc.is_empty() {
                    blocks.push(ContentBlock::Text { text: text_acc });
                }
                for (_, (id, name, args)) in tool_calls {
                    let trimmed = args.trim();
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
                    let _ = tx
                        .send(StreamEvent::ToolCallReady {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                            invalid_json: invalid_json.clone(),
                        })
                        .await;
                    blocks.push(ContentBlock::ToolUse {
                        id,
                        name,
                        input,
                        invalid_json,
                        provider_meta: None,
                        origin: Some(origin.clone()),
                    });
                }
                usage.compute_cost(pricing);
                let _ = tx
                    .send(StreamEvent::Done {
                        message: Message {
                            role: Role::Assistant,
                            content: blocks,
                        },
                        usage,
                        stop_reason,
                    })
                    .await;
            });

            Ok(rx)
        })
    }
}
