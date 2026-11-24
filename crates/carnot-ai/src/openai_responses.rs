//! Stateless OpenAI Responses API adapter (`api = "openai-responses"`).
//!
//! Implements Opus Phase 5 contract:
//! - Stateless by default (`store: false`, `include: ["reasoning.encrypted_content"]`) — never uses
//!   `previous_response_id` so ZDR, failover, and session resume stay 100% deterministic.
//! - Verbatim replay of origin-matched `Opaque` reasoning items (`type: "reasoning"`).
//! - Multimodal `input_image` data URLs on user messages & tool-output bridge.
//! - Declarative `prompt_cache_key`, `reasoning.effort`, `body_patch`, and `Retry-After`.

use crate::anthropic::find_sse_frame_end;
use crate::catalog::ResolvedProviderSpec;
use crate::compat::{
    can_replay_origin, is_context_overflow, is_retryable_status, parse_retry_after_ms, repair_json,
    session_cache_key,
};
use crate::registry::ApiAdapter;
use crate::{BoxFuture, Provider, ProviderRequest, StreamEvent};
use carnot_core::{
    CacheTtl, CarnotError, ContentBlock, ContentOrigin, Message, ProviderWebCapabilities,
    ReasoningEffort, Result, Role, StopReason, ToolResultBlock, Usage,
};
use futures::StreamExt;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const API_ID: &str = "openai-responses";
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(300);
const HTTP_HEADERS_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug)]
pub struct OpenAiResponsesAdapter;

impl ApiAdapter for OpenAiResponsesAdapter {
    fn api_id(&self) -> &'static str {
        API_ID
    }

    fn build(&self, spec: &ResolvedProviderSpec) -> Result<Arc<dyn Provider>> {
        Ok(Arc::new(OpenAiResponsesProvider::from_spec(spec.clone())))
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiResponsesProvider {
    client: reqwest::Client,
    spec: ResolvedProviderSpec,
}

impl OpenAiResponsesProvider {
    pub fn from_spec(spec: ResolvedProviderSpec) -> Self {
        let client = crate::build_http_client_for_url(&spec.base_url);
        Self { client, spec }
    }

    pub fn build_body(&self, req: &ProviderRequest) -> Value {
        let compat = self.spec.effective_compat(&req.model);
        let mut input_items: Vec<Value> = Vec::new();

        for msg in &req.messages {
            match msg.role {
                Role::System => {
                    let txt = msg.text_concat();
                    if !txt.is_empty() {
                        input_items.push(json!({
                            "role": "developer",
                            "content": [{ "type": "input_text", "text": txt }]
                        }));
                    }
                }
                Role::User => {
                    let mut user_parts: Vec<Value> = Vec::new();
                    let mut tool_bridge_images: Vec<Value> = Vec::new();
                    for b in &msg.content {
                        match b {
                            ContentBlock::Text { text } if !text.is_empty() => {
                                user_parts.push(json!({ "type": "input_text", "text": text }));
                            }
                            ContentBlock::Image {
                                media_type,
                                data_base64,
                            } => {
                                user_parts.push(json!({
                                    "type": "input_image",
                                    "image_url": format!("data:{media_type};base64,{data_base64}")
                                }));
                            }
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                ..
                            } => {
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
                                                .push("[image attached in following user turn]");
                                            tool_bridge_images.push(json!({
                                                "type": "input_image",
                                                "image_url": format!("data:{media_type};base64,{data_base64}")
                                            }));
                                        }
                                    }
                                }
                                input_items.push(json!({
                                    "type": "function_call_output",
                                    "call_id": tool_use_id,
                                    "output": text_parts.join("\n"),
                                }));
                            }
                            _ => {}
                        }
                    }
                    if !tool_bridge_images.is_empty() {
                        let mut parts = vec![json!({
                            "type": "input_text",
                            "text": "Image(s) returned by the preceding tool call:"
                        })];
                        parts.extend(tool_bridge_images);
                        input_items.push(json!({ "role": "user", "content": parts }));
                    }
                    if !user_parts.is_empty() {
                        input_items.push(json!({ "role": "user", "content": user_parts }));
                    }
                }
                Role::Assistant => {
                    let mut text_parts: Vec<Value> = Vec::new();
                    for b in &msg.content {
                        match b {
                            ContentBlock::Opaque { raw, origin }
                                if can_replay_origin(
                                    origin.as_ref(),
                                    API_ID,
                                    &self.spec.provider_id,
                                    &req.model,
                                    false,
                                ) =>
                            {
                                input_items.push(raw.clone());
                            }
                            ContentBlock::Text { text } if !text.is_empty() => {
                                text_parts.push(json!({
                                    "type": "output_text",
                                    "text": text,
                                    "annotations": []
                                }));
                            }
                            ContentBlock::ToolUse {
                                id, name, input, ..
                            } => {
                                if !text_parts.is_empty() {
                                    input_items.push(json!({
                                        "type": "message",
                                        "role": "assistant",
                                        "content": std::mem::take(&mut text_parts),
                                    }));
                                }
                                input_items.push(json!({
                                    "type": "function_call",
                                    "call_id": id,
                                    "name": name,
                                    "arguments": input.to_string(),
                                }));
                            }
                            _ => {}
                        }
                    }
                    if !text_parts.is_empty() {
                        input_items.push(json!({
                            "type": "message",
                            "role": "assistant",
                            "content": text_parts,
                        }));
                    }
                }
            }
        }

        for note in &req.ephemeral_tail_notes {
            if !note.trim().is_empty() {
                input_items.push(json!({
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": note}],
                }));
            }
        }

        let mut tools: Vec<Value> = req
            .tools
            .iter()
            .filter(|t| !t.defer_loading)
            .map(|t| {
                json!({
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                })
            })
            .collect();

        let want_native_web = (req.native_web.search || req.native_web.fetch)
            && (compat.supports_native_web_search.unwrap_or(false)
                || compat.supports_native_web_fetch.unwrap_or(false));
        if want_native_web {
            let mut ws = json!({ "type": "web_search" });
            if let Some(m) = ws.as_object_mut() {
                if !req.native_web.allowed_domains.is_empty() {
                    m.insert(
                        "filters".into(),
                        json!({ "allowed_domains": req.native_web.allowed_domains }),
                    );
                }
                if let Some(ref sz) = req.native_web.search_context_size
                    && matches!(sz.as_str(), "low" | "medium" | "high")
                {
                    m.insert("search_context_size".into(), json!(sz));
                }
            }
            tools.push(ws);
        }

        let store = compat.store.unwrap_or(false);
        let mut body = json!({
            "model": req.model,
            "instructions": req.static_system_prompt,
            "input": input_items,
            "max_output_tokens": req.max_output_tokens,
            "stream": true,
            "store": store,
        });

        if let Some(m) = body.as_object_mut() {
            if !tools.is_empty() {
                m.insert("tools".into(), json!(tools));
            }
            if compat.include_encrypted_reasoning.unwrap_or(true) {
                m.insert("include".into(), json!(["reasoning.encrypted_content"]));
            }
            if compat.supports_prompt_cache_key.unwrap_or(true) {
                m.insert(
                    "prompt_cache_key".into(),
                    json!(session_cache_key(&req.prefix_hash, &req.session_id)),
                );
            }
            if compat.supports_prompt_cache_retention.unwrap_or(false)
                && matches!(req.cache_ttl, CacheTtl::Long1h | CacheTtl::Hybrid)
            {
                m.insert("prompt_cache_retention".into(), json!("24h"));
            }
            if compat.supports_reasoning_effort.unwrap_or(true) {
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
                m.insert("reasoning".into(), json!({ "effort": effort_str }));
            }
        }

        self.spec.apply_body_patches(&req.model, &mut body);
        body
    }
}

impl Provider for OpenAiResponsesProvider {
    fn name(&self) -> &str {
        &self.spec.provider_id
    }

    fn web_capabilities(&self, model: &str) -> ProviderWebCapabilities {
        let compat = self.spec.effective_compat(model);
        ProviderWebCapabilities {
            native_search: compat.supports_native_web_search.unwrap_or(false),
            native_fetch: compat.supports_native_web_fetch.unwrap_or(false),
        }
    }

    fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<mpsc::Receiver<StreamEvent>>> {
        Box::pin(async move {
            let raw_url = format!("{}/responses", self.spec.base_url);
            let body = self.build_body(&request);
            let mut extra_headers: Vec<(String, String)> = Vec::new();
            let compat = self.spec.effective_compat(&request.model);
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
            let origin = ContentOrigin::new(API_ID, &self.spec.provider_id, &request.model);

            tokio::spawn(async move {
                let mut stream = resp.bytes_stream();
                let mut raw_buf: Vec<u8> = Vec::with_capacity(8192);
                let mut text_acc = String::new();
                let mut thinking_acc = String::new();
                let mut opaque_items: Vec<Value> = Vec::new();
                // output_index -> (call_id, name, args_buf)
                let mut fn_calls: BTreeMap<u64, (String, String, String)> = BTreeMap::new();
                let mut usage = Usage::default();
                let mut stop_reason = StopReason::EndTurn;
                let mut saw_completed = false;

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
                                    message: "stream idle timeout (120s)".into(),
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
                        let mut data_parts: Vec<&str> = Vec::new();
                        for line in frame_str.lines() {
                            let line = line.trim_end_matches('\r');
                            if let Some(rest) = line.strip_prefix("data:") {
                                data_parts.push(rest.strip_prefix(' ').unwrap_or(rest));
                            }
                        }
                        if data_parts.is_empty() {
                            continue;
                        }
                        let joined = data_parts.join("\n");
                        if joined.trim() == "[DONE]" {
                            saw_completed = true;
                            continue;
                        }
                        let Ok(v) = serde_json::from_str::<Value>(&joined) else {
                            continue;
                        };
                        match v.get("type").and_then(Value::as_str) {
                            Some("response.output_text.delta") => {
                                if let Some(d) = v.get("delta").and_then(Value::as_str) {
                                    text_acc.push_str(d);
                                    let _ = tx.send(StreamEvent::TextDelta(d.to_string())).await;
                                }
                            }
                            Some("response.reasoning_summary_text.delta")
                            | Some("response.reasoning_text.delta") => {
                                if let Some(d) = v.get("delta").and_then(Value::as_str) {
                                    thinking_acc.push_str(d);
                                    let _ =
                                        tx.send(StreamEvent::ThinkingDelta(d.to_string())).await;
                                }
                            }
                            Some("response.output_item.added") => {
                                let idx =
                                    v.get("output_index").and_then(Value::as_u64).unwrap_or(0);
                                if let Some(item) = v.get("item")
                                    && item.get("type").and_then(Value::as_str)
                                        == Some("function_call")
                                {
                                    let cid = item
                                        .get("call_id")
                                        .or_else(|| item.get("id"))
                                        .and_then(Value::as_str)
                                        .unwrap_or("")
                                        .to_string();
                                    let name = item
                                        .get("name")
                                        .and_then(Value::as_str)
                                        .unwrap_or("")
                                        .to_string();
                                    fn_calls.insert(idx, (cid, name, String::new()));
                                }
                            }
                            Some("response.function_call_arguments.delta") => {
                                let idx =
                                    v.get("output_index").and_then(Value::as_u64).unwrap_or(0);
                                if let Some(d) = v.get("delta").and_then(Value::as_str)
                                    && let Some(entry) = fn_calls.get_mut(&idx)
                                {
                                    entry.2.push_str(d);
                                }
                            }
                            Some("response.output_item.done") => {
                                let idx =
                                    v.get("output_index").and_then(Value::as_u64).unwrap_or(0);
                                if let Some(item) = v.get("item") {
                                    match item.get("type").and_then(Value::as_str) {
                                        Some("reasoning" | "web_search_call") => {
                                            opaque_items.push(item.clone());
                                        }
                                        Some("function_call") => {
                                            let cid = item
                                                .get("call_id")
                                                .or_else(|| item.get("id"))
                                                .and_then(Value::as_str)
                                                .unwrap_or("")
                                                .to_string();
                                            let name = item
                                                .get("name")
                                                .and_then(Value::as_str)
                                                .unwrap_or("")
                                                .to_string();
                                            let args = item
                                                .get("arguments")
                                                .and_then(Value::as_str)
                                                .map(str::to_string)
                                                .unwrap_or_else(|| {
                                                    fn_calls
                                                        .get(&idx)
                                                        .map(|e| e.2.clone())
                                                        .unwrap_or_default()
                                                });
                                            fn_calls.insert(
                                                idx,
                                                (cid.clone(), name.clone(), args.clone()),
                                            );
                                            let (input, invalid_json) = parse_tool_args_json(&args);
                                            let _ = tx
                                                .send(StreamEvent::ToolCallReady {
                                                    id: cid,
                                                    name,
                                                    input,
                                                    invalid_json,
                                                })
                                                .await;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            Some("response.completed") | Some("response.incomplete") => {
                                saw_completed = true;
                                if let Some(resp_obj) = v.get("response") {
                                    if let Some(u) = resp_obj.get("usage") {
                                        let in_tok = u
                                            .get("input_tokens")
                                            .and_then(Value::as_u64)
                                            .unwrap_or(0);
                                        let cached = u
                                            .pointer("/input_tokens_details/cached_tokens")
                                            .and_then(Value::as_u64)
                                            .unwrap_or(0);
                                        usage.cache_read_tokens = cached;
                                        usage.input_tokens = in_tok.saturating_sub(cached);
                                        usage.output_tokens = u
                                            .get("output_tokens")
                                            .and_then(Value::as_u64)
                                            .unwrap_or(0);
                                        usage.reasoning_tokens = u
                                            .pointer("/output_tokens_details/reasoning_tokens")
                                            .and_then(Value::as_u64)
                                            .unwrap_or(0);
                                    }
                                    if resp_obj.get("status").and_then(Value::as_str)
                                        == Some("incomplete")
                                    {
                                        let reason = resp_obj
                                            .pointer("/incomplete_details/reason")
                                            .and_then(Value::as_str)
                                            .unwrap_or("");
                                        stop_reason = if reason == "content_filter" {
                                            StopReason::Refusal
                                        } else {
                                            StopReason::MaxTokens
                                        };
                                    }
                                }
                            }
                            Some("response.failed") | Some("error") => {
                                let code = v
                                    .pointer("/response/error/code")
                                    .or_else(|| v.pointer("/error/code"))
                                    .and_then(Value::as_str)
                                    .unwrap_or("api_error");
                                let msg = v
                                    .pointer("/response/error/message")
                                    .or_else(|| v.pointer("/error/message"))
                                    .and_then(Value::as_str)
                                    .unwrap_or("responses stream error");
                                let full_msg = format!("{code}: {msg}");
                                if is_context_overflow(None, &full_msg) {
                                    let _ = tx.send(StreamEvent::ContextOverflow(full_msg)).await;
                                    return;
                                }
                                let retryable = !matches!(
                                    code,
                                    "invalid_request_error"
                                        | "authentication_error"
                                        | "context_length_exceeded"
                                        | "insufficient_quota"
                                );
                                let _ = tx
                                    .send(StreamEvent::Error {
                                        message: full_msg,
                                        retryable,
                                    })
                                    .await;
                                return;
                            }
                            _ => {}
                        }
                    }
                }

                if !saw_completed {
                    let _ = tx
                        .send(StreamEvent::Error {
                            message: "responses stream terminated before response.completed".into(),
                            retryable: true,
                        })
                        .await;
                    return;
                }

                let mut blocks: Vec<ContentBlock> = Vec::new();
                for raw in opaque_items {
                    blocks.push(ContentBlock::Opaque {
                        raw,
                        origin: Some(origin.clone()),
                    });
                }
                if !thinking_acc.is_empty() {
                    blocks.push(ContentBlock::Thinking {
                        thinking: thinking_acc,
                        signature: None,
                        origin: Some(origin.clone()),
                    });
                }
                if !text_acc.is_empty() {
                    blocks.push(ContentBlock::Text { text: text_acc });
                }
                if !fn_calls.is_empty() && stop_reason == StopReason::EndTurn {
                    stop_reason = StopReason::ToolUse;
                }
                for (_, (id, name, args)) in fn_calls {
                    let (input, invalid_json) = parse_tool_args_json(&args);
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

fn parse_tool_args_json(args: &str) -> (Value, Option<String>) {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        (json!({}), None)
    } else if let Some(v) = repair_json(trimmed) {
        (v, None)
    } else {
        let err = serde_json::from_str::<Value>(trimmed)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_else(|| "invalid JSON".into());
        (
            json!({}),
            Some(format!("malformed tool JSON ({err}): {trimmed}")),
        )
    }
}
