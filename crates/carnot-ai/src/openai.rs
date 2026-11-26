//! Declarative `OpenAI` Chat Completions wire protocol adapter (`api = "openai-completions"`).
//!
//! Zero hardcoded provider names, hostnames, or `detect_compat` URL heuristics!
//! Driven 100% by `CompatFlags`, `UsageMap`, `headers`, and `body_patch` from `providers.json`.
//!
//! Protocol fidelity:
//! - Multimodal user images (`image_url` data URLs) AND automatic `user` bridge message for
//!   images returned inside `ToolResult` (since the `tool` role only accepts text!).
//! - Streams `reasoning_content` / `reasoning` deltas -> `ThinkingDelta` + persists `Thinking`
//!   with `origin`, and replays `reasoning_content` on assistant messages when configured.
//! - Declarative `max_tokens_field` (`max_completion_tokens` vs `max_tokens`), `developer` role,
//!   `reasoning_effort` vs `reasoning: { effort }`, `prompt_cache_key`, `session_id_header`, `body_patch`.
//! - Accurate declarative `UsageMap` (handles `OpenAI`, xAI, `DeepSeek` top-level cache hit, `OpenRouter`).
//! - `Retry-After` parsing, retries on `408|429|500|502|503|504|529`, and non-retryable stream error classification.

use crate::catalog::{ProviderCatalog, ResolvedAuth, ResolvedProviderSpec};
use crate::compat::{
    apply_usage_map, can_replay_origin, format_tool_call_id, is_context_overflow,
    is_retryable_status, parse_retry_after_ms, repair_json, session_cache_key,
};
use crate::registry::ApiAdapter;
use crate::{BoxFuture, Provider, ProviderRequest, StreamEvent};
use carnot_core::{
    CacheTtl, CarnotError, ContentBlock, ContentOrigin, Message, ReasoningEffort, Result, Role,
    StopReason, ToolResultBlock, Usage,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const API_ID: &str = "openai-completions";
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(300);
const HTTP_HEADERS_TIMEOUT: Duration = Duration::from_secs(300);
static SYNTHETIC_CALL_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

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

    pub fn with_base_url(api_key: String, mut base_url: String, provider_label: String) -> Self {
        while base_url.ends_with('/') {
            base_url.pop();
        }
        let catalog = ProviderCatalog::embedded().unwrap_or_default();
        let entry = catalog
            .find_provider(&provider_label)
            .map(|(_, e)| e.clone());
        let spec = ResolvedProviderSpec {
            provider_id: provider_label,
            api: API_ID.into(),
            base_url,
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
            query_params: entry
                .as_ref()
                .map(|e| e.query_params.clone())
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
        // Track wire index of the last emitted wire message for each `req.messages` index,
        // so `checkpoint_index` and Turn-1 (`idx == 0`) map accurately to wire positions.
        let mut msg_end_wire_idx: Vec<Option<usize>> = Vec::with_capacity(req.messages.len());

        for msg in &req.messages {
            let before_len = messages.len();
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
                                        text_parts.push(text.as_str());
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
                            let wire_tool_call_id = format_tool_call_id(
                                tool_use_id,
                                compat.tool_call_id_format.as_deref(),
                            );
                            messages.push(json!({
                                "role": "tool",
                                "tool_call_id": wire_tool_call_id,
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
                                "id": format_tool_call_id(id, compat.tool_call_id_format.as_deref()),
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": input.to_string(),
                                }
                            })),
                            _ => None,
                        })
                        .collect();

                    let empty_content = if compat
                        .empty_assistant_content_as_empty_string
                        .unwrap_or(false)
                    {
                        Value::String(String::new())
                    } else {
                        Value::Null
                    };
                    let mut obj = json!({
                        "role": "assistant",
                        "content": if txt.is_empty() { empty_content } else { json!(txt) }
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
            if messages.len() > before_len {
                msg_end_wire_idx.push(Some(messages.len() - 1));
            } else {
                msg_end_wire_idx.push(None);
            }
        }

        if should_inject_cache_control(&compat, &req.model) {
            apply_openai_cache_control(
                &mut messages,
                &msg_end_wire_idx,
                req.checkpoint_index,
                req.cache_ttl,
                req.skip_tail_cache_write,
                &compat,
                &req.model,
            );
        }

        // Append ephemeral per-turn notes AFTER cache breakpoints so transient notes stay uncached!
        for note in &req.ephemeral_tail_notes {
            if !note.trim().is_empty() {
                messages.push(json!({"role": "user", "content": note}));
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
                    json!(session_cache_key(&req.prefix_hash, &req.session_id)),
                );
            }
            if compat.supports_prompt_cache_retention.unwrap_or(false)
                && matches!(req.cache_ttl, CacheTtl::Long | CacheTtl::Hybrid)
            {
                m.insert("prompt_cache_retention".into(), json!("24h"));
            }
            let allow_xhigh = compat.supports_xhigh_reasoning.unwrap_or(false);
            let effort_str = match req.effort {
                ReasoningEffort::Low => "low",
                ReasoningEffort::Medium => "medium",
                ReasoningEffort::High => "high",
                ReasoningEffort::Xhigh | ReasoningEffort::Max => {
                    if allow_xhigh {
                        "xhigh"
                    } else {
                        "high"
                    }
                }
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

    fn rolling_cache_ttl_ms(&self, model: &str, ttl: CacheTtl) -> u64 {
        let compat = self.spec.effective_compat(model);
        if compat.supports_prompt_cache_retention.unwrap_or(false)
            && matches!(ttl, CacheTtl::Hybrid | CacheTtl::Long)
        {
            86_410_000
        } else {
            match ttl {
                CacheTtl::Long => 3_610_000,
                CacheTtl::Short | CacheTtl::Hybrid => 310_000,
            }
        }
    }

    fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<mpsc::Receiver<StreamEvent>>> {
        Box::pin(async move {
            let raw_url = format!("{}/chat/completions", self.spec.base_url);
            let body = self.build_body(&request);
            let compat = self.spec.effective_compat(&request.model);

            let mut extra_headers: Vec<(String, String)> = Vec::new();
            if let Some(ref hdr) = compat.session_id_header {
                extra_headers.push((
                    hdr.clone(),
                    session_cache_key(&request.prefix_hash, &request.session_id),
                ));
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
                let mut stream_eof = false;

                'stream_loop: loop {
                    if stream_eof {
                        break 'stream_loop;
                    }
                    tokio::select! {
                        () = cancel.cancelled() => {
                            let _ = tx.send(StreamEvent::Error { message: "cancelled".into(), retryable: false }).await;
                            return;
                        }
                        chunk = tokio::time::timeout(STREAM_IDLE_TIMEOUT, stream.next()) => {
                            let Ok(chunk_opt) = chunk else {
                                let _ = tx.send(StreamEvent::Error { message: "stream idle timeout (300s)".into(), retryable: true }).await;
                                return;
                            };
                            match chunk_opt {
                                Some(Ok(bytes)) => raw_buf.extend_from_slice(&bytes),
                                Some(Err(_)) => {
                                    let _ = tx.send(StreamEvent::Error { message: "stream read error".into(), retryable: true }).await;
                                    return;
                                }
                                None => {
                                    stream_eof = true;
                                    if raw_buf.is_empty() {
                                        break 'stream_loop;
                                    }
                                    raw_buf.push(b'\n');
                                }
                            }
                        }
                    };
                    while let Some(pos) = raw_buf.iter().position(|&b| b == b'\n') {
                        let line_bytes: Vec<u8> = raw_buf.drain(..=pos).collect();
                        let line = String::from_utf8_lossy(&line_bytes).trim().to_string();
                        let Some(data) = line.strip_prefix("data:") else {
                            continue;
                        };
                        let data = data.trim();
                        if data == "[DONE]" {
                            saw_done_or_finish = true;
                            break 'stream_loop;
                        }
                        let Ok(v) = serde_json::from_str::<Value>(data) else {
                            continue;
                        };
                        if let Some(err) = v.get("error") {
                            let err_str = err.to_string();
                            if is_context_overflow(None, &err_str) {
                                let _ = tx.send(StreamEvent::ContextOverflow(err_str)).await;
                                return;
                            }
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
                            let _ = tx
                                .send(StreamEvent::Error {
                                    message: err_str,
                                    retryable,
                                })
                                .await;
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
                                    let _ =
                                        tx.send(StreamEvent::ThinkingDelta(r.to_string())).await;
                                }
                                if let Some(content) = delta.get("content") {
                                    if let Some(t) = content.as_str() {
                                        if !t.is_empty() {
                                            text_acc.push_str(t);
                                            let _ = tx
                                                .send(StreamEvent::TextDelta(t.to_string()))
                                                .await;
                                        }
                                    } else if let Some(parts) = content.as_array() {
                                        // Structured content blocks (e.g. `type: "thinking"` + `type: "text"`).
                                        for part in parts {
                                            match part.get("type").and_then(Value::as_str) {
                                                Some("thinking") => {
                                                    if let Some(inner) = part
                                                        .get("thinking")
                                                        .and_then(Value::as_array)
                                                    {
                                                        for item in inner {
                                                            if let Some(rt) = item
                                                                .get("text")
                                                                .and_then(Value::as_str)
                                                                && !rt.is_empty()
                                                            {
                                                                reasoning_acc.push_str(rt);
                                                                let _ = tx
                                                                    .send(
                                                                        StreamEvent::ThinkingDelta(
                                                                            rt.to_string(),
                                                                        ),
                                                                    )
                                                                    .await;
                                                            }
                                                        }
                                                    } else if let Some(rt) = part
                                                        .get("thinking")
                                                        .or_else(|| part.get("text"))
                                                        .and_then(Value::as_str)
                                                        && !rt.is_empty()
                                                    {
                                                        reasoning_acc.push_str(rt);
                                                        let _ = tx
                                                            .send(StreamEvent::ThinkingDelta(
                                                                rt.to_string(),
                                                            ))
                                                            .await;
                                                    }
                                                }
                                                Some("text") => {
                                                    if let Some(t) =
                                                        part.get("text").and_then(Value::as_str)
                                                        && !t.is_empty()
                                                    {
                                                        text_acc.push_str(t);
                                                        let _ = tx
                                                            .send(StreamEvent::TextDelta(
                                                                t.to_string(),
                                                            ))
                                                            .await;
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                                if let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array)
                                {
                                    for (pos_in_chunk, tc) in tcs.iter().enumerate() {
                                        let idx = match tc.get("index").and_then(Value::as_u64) {
                                            Some(i) => i as usize,
                                            None => {
                                                if tc
                                                    .get("id")
                                                    .and_then(Value::as_str)
                                                    .is_some_and(|s| !s.is_empty())
                                                {
                                                    tool_calls
                                                        .keys()
                                                        .next_back()
                                                        .map_or(0, |k| k + 1)
                                                } else {
                                                    tool_calls
                                                        .keys()
                                                        .next_back()
                                                        .copied()
                                                        .unwrap_or(pos_in_chunk)
                                                }
                                            }
                                        };
                                        let entry = tool_calls.entry(idx).or_insert_with(|| {
                                            (String::new(), String::new(), String::new())
                                        });
                                        if let Some(id) = tc.get("id").and_then(Value::as_str)
                                            && !id.is_empty()
                                        {
                                            entry.0 = id.to_string();
                                        }
                                        if let Some(n) =
                                            tc.pointer("/function/name").and_then(Value::as_str)
                                        {
                                            entry.1.push_str(n);
                                        }
                                        if let Some(a) = tc
                                            .pointer("/function/arguments")
                                            .and_then(Value::as_str)
                                        {
                                            entry.2.push_str(a);
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

                if !tool_calls.is_empty() && stop_reason == StopReason::EndTurn {
                    stop_reason = StopReason::ToolUse;
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
                for (_, (raw_id, name, args)) in tool_calls {
                    let id = if raw_id.is_empty() {
                        format!(
                            "call_oai_{}",
                            SYNTHETIC_CALL_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        )
                    } else {
                        raw_id
                    };
                    let trimmed = args.trim();
                    let (input, invalid_json) = if trimmed.is_empty() {
                        (json!({}), None)
                    } else if let Some(v) = repair_json(trimmed) {
                        (v, None)
                    } else {
                        let err = serde_json::from_str::<Value>(trimmed)
                            .err()
                            .map_or_else(|| "invalid JSON".into(), |e| e.to_string());
                        (
                            json!({}),
                            Some(format!("malformed tool JSON ({err}): {trimmed}")),
                        )
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
                        rate_limits: Vec::new(),
                    })
                    .await;
            });

            Ok(rx)
        })
    }
}

fn should_inject_cache_control(compat: &crate::catalog::CompatFlags, model: &str) -> bool {
    if !compat.supports_cache_control.unwrap_or(false) {
        return false;
    }
    let Some(ref prefixes) = compat.cache_control_model_prefixes else {
        return true;
    };
    if prefixes.is_empty() {
        return true;
    }
    let lower = model.to_ascii_lowercase();
    prefixes
        .iter()
        .any(|p| lower.starts_with(&p.to_ascii_lowercase()))
}

fn cache_control_value(ttl_1h: bool) -> Value {
    if ttl_1h {
        json!({"type": "ephemeral", "ttl": "1h"})
    } else {
        json!({"type": "ephemeral"})
    }
}

/// Attaches `cache_control` to the final text/part of a Chat Completions message, converting plain
/// string `content` into multipart `[{"type":"text","text":...,"cache_control":...}]` as required
/// by `OpenRouter` / Anthropic-compatible gateways. Returns `false` if message has no cacheable text.
fn attach_cache_control_to_message(msg: &mut Value, cc: &Value) -> bool {
    let Some(obj) = msg.as_object_mut() else {
        return false;
    };
    let Some(content) = obj.get_mut("content") else {
        return false;
    };
    match content {
        Value::String(s) => {
            if s.is_empty() {
                return false;
            }
            let text = std::mem::take(s);
            *content = json!([{
                "type": "text",
                "text": text,
                "cache_control": cc,
            }]);
            true
        }
        Value::Array(parts) => {
            // Attach to the FINAL content part so images (`[text, image_url]`) stay inside the cached prefix.
            if let Some(part_obj) = parts.last_mut().and_then(Value::as_object_mut) {
                part_obj.insert("cache_control".into(), cc.clone());
                return true;
            }
            false
        }
        _ => false,
    }
}

/// Applies up to 4 strictly monotonic `cache_control` breakpoints across `messages`:
/// - BP1: static system prompt (`messages[0]`, 1h when supported)
/// - BP2: Turn-1 initial user/context message (1h when supported)
/// - BP3: `checkpoint_index` message (rolling TTL)
/// - BP4: final cacheable tail message (rolling TTL; skipped during compaction when `skip_tail_cache_write` is true)
fn apply_openai_cache_control(
    messages: &mut [Value],
    msg_end_wire_idx: &[Option<usize>],
    checkpoint_index: Option<usize>,
    cache_ttl: CacheTtl,
    skip_tail_cache_write: bool,
    compat: &crate::catalog::CompatFlags,
    model: &str,
) {
    if messages.is_empty() {
        return;
    }
    let allow_1h = compat.supports_cache_control_1h_ttl.unwrap_or(false)
        && model.to_ascii_lowercase().starts_with("anthropic/");
    let static_cc =
        cache_control_value(allow_1h && matches!(cache_ttl, CacheTtl::Long | CacheTtl::Hybrid));
    let rolling_cc = cache_control_value(allow_1h && cache_ttl == CacheTtl::Long);

    let mut used: Vec<usize> = Vec::with_capacity(4);

    // BP1: static system prompt
    if attach_cache_control_to_message(&mut messages[0], &static_cc) {
        used.push(0);
    }

    // BP2: Turn-1 context message (AGENTS.md + workspace-facts + goal)
    if let Some(&Some(turn1_wire)) = msg_end_wire_idx.first()
        && turn1_wire < messages.len()
        && !used.contains(&turn1_wire)
        && attach_cache_control_to_message(&mut messages[turn1_wire], &static_cc)
    {
        used.push(turn1_wire);
    }

    // BP3: stable checkpoint boundary (walk backward if exact checkpoint message has `content: null`,
    // e.g. an assistant message containing only `tool_calls`, so BP3 is never silently wasted)
    if let Some(cp_idx) = checkpoint_index
        && let Some(&Some(cp_wire)) = msg_end_wire_idx.get(cp_idx)
        && cp_wire < messages.len()
        && used.len() < 4
    {
        for idx in (1..=cp_wire).rev() {
            if used.contains(&idx) {
                break;
            }
            if attach_cache_control_to_message(&mut messages[idx], &rolling_cc) {
                used.push(idx);
                break;
            }
        }
    }

    // BP4: rolling tail (omitted when `skip_tail_cache_write` is active during compaction)
    if !skip_tail_cache_write && used.len() < 4 {
        for idx in (1..messages.len()).rev() {
            if used.contains(&idx) {
                break;
            }
            if attach_cache_control_to_message(&mut messages[idx], &rolling_cc) {
                used.push(idx);
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use carnot_core::{NativeWebConfig, ReasoningEffort};
    use std::sync::Arc;

    fn openrouter_provider() -> OpenAiProvider {
        let catalog = ProviderCatalog::embedded().expect("embedded catalog");
        let (_, entry) = catalog.find_provider("openrouter").expect("openrouter");
        let spec = ResolvedProviderSpec {
            provider_id: "openrouter".into(),
            api: API_ID.into(),
            base_url: entry.base_url.clone(),
            fallback_base_urls: Vec::new(),
            project_id: None,
            auth: ResolvedAuth::Bearer {
                token: "test".into(),
                source: "test".into(),
            },
            headers: entry.headers.clone(),
            query_params: entry.query_params.clone(),
            betas: Vec::new(),
            cache_diagnosis_beta: None,
            native_tool_search: false,
            compat: entry.compat.clone(),
            usage_map: entry.usage_map.clone(),
            body_patch: entry.body_patch.clone(),
            subprocess: None,
            catalog_models: catalog.models,
        };
        OpenAiProvider::from_spec(spec)
    }

    fn make_req(
        model: &str,
        messages: Vec<Message>,
        cp: Option<usize>,
        skip_tail: bool,
    ) -> ProviderRequest {
        ProviderRequest {
            model: model.into(),
            static_system_prompt: "You are Carnot.".into(),
            tools: Vec::new(),
            prefix_hash: "abcdef0123456789".into(),
            session_id: "sess-1".into(),
            messages: messages.into_iter().map(Arc::new).collect(),
            checkpoint_index: cp,
            cache_ttl: CacheTtl::Hybrid,
            effort: ReasoningEffort::High,
            max_output_tokens: 4096,
            skip_tail_cache_write: skip_tail,
            native_web: NativeWebConfig::default(),
            ephemeral_tail_notes: Vec::new(),
        }
    }

    #[test]
    fn openrouter_injects_4_monotonic_cache_control_breakpoints_for_claude() {
        let p = openrouter_provider();
        let body = p.build_body(&make_req(
            "anthropic/claude-sonnet-4.5",
            vec![
                Message::user_text("Turn 1 AGENTS.md + goal"),
                Message::assistant_text("Working on it"),
                Message::user_text("Checkpoint turn"),
                Message::assistant_text("Done step"),
                Message::user_text("Latest user turn"),
            ],
            Some(2),
            false,
        ));
        let serialized = body.to_string();
        assert_eq!(serialized.matches("\"cache_control\"").count(), 4);
        assert_eq!(
            body.pointer("/messages/0/content/0/cache_control/ttl"),
            Some(&json!("1h"))
        );
        assert_eq!(
            body.pointer("/messages/1/content/0/cache_control/ttl"),
            Some(&json!("1h"))
        );
        assert_eq!(
            body.pointer("/messages/3/content/0/cache_control"),
            Some(&json!({"type": "ephemeral"}))
        );
        assert_eq!(
            body.pointer("/messages/5/content/0/cache_control"),
            Some(&json!({"type": "ephemeral"}))
        );
    }

    #[test]
    fn openrouter_uses_plain_ephemeral_for_gemini_and_skips_openai_models() {
        let p = openrouter_provider();
        let gemini_body = p.build_body(&make_req(
            "google/gemini-2.5-pro",
            vec![
                Message::user_text("Turn 1"),
                Message::assistant_text("ok"),
                Message::user_text("Turn 2"),
            ],
            None,
            false,
        ));
        assert_eq!(
            gemini_body.pointer("/messages/0/content/0/cache_control"),
            Some(&json!({"type": "ephemeral"}))
        );
        assert!(
            gemini_body
                .pointer("/messages/0/content/0/cache_control/ttl")
                .is_none()
        );

        let gpt_body = p.build_body(&make_req(
            "openai/gpt-5.4",
            vec![Message::user_text("Turn 1")],
            None,
            false,
        ));
        assert!(!gpt_body.to_string().contains("cache_control"));
    }

    #[test]
    fn openrouter_skip_tail_cache_write_omits_bp4_during_compaction() {
        let p = openrouter_provider();
        let body = p.build_body(&make_req(
            "anthropic/claude-sonnet-4.5",
            vec![
                Message::user_text("Turn 1"),
                Message::assistant_text("Step 1"),
                Message::user_text("Compaction prompt"),
            ],
            Some(1),
            true,
        ));
        let serialized = body.to_string();
        assert_eq!(serialized.matches("\"cache_control\"").count(), 3);
        assert!(
            body.pointer("/messages/3/content")
                .and_then(Value::as_str)
                .is_some()
        );
    }
}
