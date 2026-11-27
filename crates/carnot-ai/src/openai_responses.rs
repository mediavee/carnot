//! Stateless `OpenAI` Responses API adapter (`api = "openai-responses"`).
//!
//! Protocol contract:
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
use futures_util::StreamExt;
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
        let sys_role = if compat.supports_developer_role.unwrap_or(true) {
            "developer"
        } else {
            "system"
        };
        let mut input_items: Vec<Value> = Vec::new();

        for msg in &req.messages {
            match msg.role {
                Role::System => {
                    let txt = msg.text_concat();
                    if !txt.is_empty() {
                        input_items.push(json!({
                            "role": sys_role,
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
                    "role": sys_role,
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
                && matches!(req.cache_ttl, CacheTtl::Long | CacheTtl::Hybrid)
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
                let mut stream_eof = false;

                loop {
                    if stream_eof {
                        break;
                    }
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
                    match chunk_opt {
                        Some(Ok(bytes)) => raw_buf.extend_from_slice(&bytes),
                        Some(Err(_)) => {
                            let _ = tx
                                .send(StreamEvent::Error {
                                    message: "stream read error".into(),
                                    retryable: true,
                                })
                                .await;
                            return;
                        }
                        None => {
                            stream_eof = true;
                            if raw_buf.is_empty() {
                                break;
                            }
                            raw_buf.extend_from_slice(b"\n\n");
                        }
                    }

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
                            Some(
                                "response.reasoning_summary_text.delta"
                                | "response.reasoning_text.delta",
                            ) => {
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
                                                .map_or_else(
                                                    || {
                                                        fn_calls
                                                            .get(&idx)
                                                            .map(|e| e.2.clone())
                                                            .unwrap_or_default()
                                                    },
                                                    str::to_string,
                                                );
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
                            Some("response.completed" | "response.incomplete") => {
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
                            Some("response.failed" | "error") => {
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
            .map_or_else(|| "invalid JSON".into(), |e| e.to_string());
        (
            json!({}),
            Some(format!("malformed tool JSON ({err}): {trimmed}")),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{CompatFlags, ResolvedAuth, UsageMap};
    use carnot_core::{NativeWebConfig, ToolSpec};
    use std::collections::BTreeMap;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn make_responses_spec(base_url: &str) -> ResolvedProviderSpec {
        let compat = CompatFlags {
            store: Some(false),
            include_encrypted_reasoning: Some(true),
            supports_reasoning_effort: Some(true),
            supports_prompt_cache_key: Some(true),
            supports_prompt_cache_retention: Some(true),
            ..CompatFlags::default()
        };
        ResolvedProviderSpec {
            provider_id: "openai-responses".into(),
            api: API_ID.into(),
            base_url: base_url.into(),
            fallback_base_urls: vec![],
            project_id: None,
            auth: ResolvedAuth::Bearer {
                token: "test-key".into(),
                source: "test".into(),
            },
            headers: BTreeMap::new(),
            query_params: BTreeMap::new(),
            betas: vec![],
            cache_diagnosis_beta: None,
            native_tool_search: false,
            compat,
            usage_map: UsageMap::default(),
            body_patch: None,
            subprocess: None,
            catalog_models: BTreeMap::new(),
        }
    }

    #[test]
    fn build_body_sets_stateless_store_cache_retention_and_filters_foreign_opaque_origins() {
        let provider = OpenAiResponsesProvider::from_spec(make_responses_spec(
            "https://mock-responses.example.internal/v1",
        ));
        let same_origin = ContentOrigin::new(API_ID, "openai-responses", "gpt-5.4");
        let foreign_origin =
            ContentOrigin::new("anthropic-messages", "anthropic", "claude-opus-5-5");

        let req = ProviderRequest {
            model: "gpt-5.4".into(),
            static_system_prompt: "STATIC_PROMPT".into(),
            tools: vec![ToolSpec {
                name: "Read".into(),
                description: "Read file".into(),
                input_schema: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
                defer_loading: false,
            }],
            prefix_hash: "prefix123".into(),
            session_id: "sess-1".into(),
            messages: vec![
                Arc::new(Message::user_text("inspect main.rs")),
                Arc::new(Message {
                    role: Role::Assistant,
                    content: vec![
                        ContentBlock::Opaque {
                            raw: json!({"type": "reasoning", "id": "rs_same", "encrypted_content": "enc_ok"}),
                            origin: Some(same_origin),
                        },
                        ContentBlock::Opaque {
                            raw: json!({"type": "redacted_thinking", "data": "foreign"}),
                            origin: Some(foreign_origin),
                        },
                        ContentBlock::ToolUse {
                            id: "call_1".into(),
                            name: "Read".into(),
                            input: json!({"path": "src/main.rs"}),
                            invalid_json: None,
                            provider_meta: None,
                            origin: None,
                        },
                    ],
                }),
                Arc::new(Message {
                    role: Role::User,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "call_1".into(),
                        content: vec![ToolResultBlock::text("fn main() {}")],
                        is_error: false,
                    }],
                }),
            ],
            checkpoint_index: None,
            max_output_tokens: 4096,
            effort: ReasoningEffort::High,
            cache_ttl: CacheTtl::Long,
            skip_tail_cache_write: false,
            native_web: NativeWebConfig::default(),
            ephemeral_tail_notes: vec![],
        };

        let body = provider.build_body(&req);
        assert_eq!(
            body["store"],
            json!(false),
            "stateless store=false must be set"
        );
        assert_eq!(
            body["instructions"],
            json!("STATIC_PROMPT"),
            "instructions must hold static system prompt"
        );
        assert_eq!(
            body["include"],
            json!(["reasoning.encrypted_content"]),
            "encrypted reasoning include must be enabled"
        );
        assert_eq!(
            body["prompt_cache_retention"],
            json!("24h"),
            "CacheTtl::Long must request 24h prompt_cache_retention"
        );
        assert!(
            body["prompt_cache_key"]
                .as_str()
                .is_some_and(|s| !s.is_empty()),
            "prompt_cache_key must be set"
        );

        let input = body["input"].as_array().expect("input array");
        assert!(
            input
                .iter()
                .any(|item| item.get("id") == Some(&json!("rs_same"))),
            "same-origin encrypted reasoning item must be replayed"
        );
        assert!(
            !input
                .iter()
                .any(|item| item.get("type") == Some(&json!("redacted_thinking"))),
            "foreign-origin opaque block must be stripped"
        );
        assert!(
            input
                .iter()
                .any(|item| item.get("type") == Some(&json!("function_call"))
                    && item.get("name") == Some(&json!("Read"))),
            "tool call must serialize as function_call"
        );
        assert!(
            input.iter().any(
                |item| item.get("type") == Some(&json!("function_call_output"))
                    && item.get("output") == Some(&json!("fn main() {}"))
            ),
            "tool result must serialize as function_call_output"
        );
    }

    #[test]
    fn parse_tool_args_json_repairs_or_flags_malformed_json() {
        let (empty_v, empty_err) = parse_tool_args_json("   ");
        assert_eq!(empty_v, json!({}), "empty args must yield empty object");
        assert!(empty_err.is_none(), "empty args must not flag invalid_json");

        let (valid_v, valid_err) = parse_tool_args_json(r#"{"path":"a.rs"}"#);
        assert_eq!(valid_v["path"], json!("a.rs"), "valid JSON must parse");
        assert!(valid_err.is_none(), "valid JSON must not flag invalid_json");

        let (_bad_v, bad_err) = parse_tool_args_json("{not valid json at all!!!");
        assert!(
            bad_err.is_some_and(|m| m.contains("malformed tool JSON")),
            "unrepairable JSON must populate invalid_json"
        );
    }

    #[tokio::test]
    async fn stream_parses_sse_deltas_encrypted_reasoning_and_tool_calls() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");

        let sse_body = concat!(
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"encrypted_content\":\"enc_abc\"}}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Checking file.\"}\n\n",
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"call_id\":\"fc_1\",\"name\":\"Read\"}}\n\n",
            "event: response.function_call_arguments.delta\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":1,\"delta\":\"{\\\"path\\\":\\\"lib.rs\\\"}\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1000,\"output_tokens\":42,\"input_tokens_details\":{\"cached_tokens\":800},\"output_tokens_details\":{\"reasoning_tokens\":12}}}}\n\n"
        );
        let http_resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse_body}",
            sse_body.len()
        );
        let srv = tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut req_buf = [0u8; 4096];
                let _ = sock.read(&mut req_buf).await;
                let _ = sock.write_all(http_resp.as_bytes()).await;
            }
        });

        let provider =
            OpenAiResponsesProvider::from_spec(make_responses_spec(&format!("http://{addr}/v1")));
        let req = ProviderRequest {
            model: "gpt-5.4".into(),
            static_system_prompt: "SYS".into(),
            tools: vec![],
            prefix_hash: "p".into(),
            session_id: "s".into(),
            messages: vec![Arc::new(Message::user_text("hi"))],
            checkpoint_index: None,
            max_output_tokens: 1024,
            effort: ReasoningEffort::Medium,
            cache_ttl: CacheTtl::Short,
            skip_tail_cache_write: false,
            native_web: NativeWebConfig::default(),
            ephemeral_tail_notes: vec![],
        };

        let mut rx = provider
            .stream(req, CancellationToken::new())
            .await
            .expect("stream");
        let mut done_ev = None;
        let mut saw_text = String::new();
        while let Some(ev) = rx.recv().await {
            match ev {
                StreamEvent::TextDelta(t) => saw_text.push_str(&t),
                StreamEvent::Done {
                    message,
                    usage,
                    stop_reason,
                    ..
                } => {
                    done_ev = Some((message, usage, stop_reason));
                }
                _ => {}
            }
        }
        let (msg, usage, stop_reason) = done_ev.expect("must emit Done event");
        assert_eq!(saw_text, "Checking file.", "text delta must be streamed");
        assert_eq!(
            stop_reason,
            StopReason::ToolUse,
            "function_call must set StopReason::ToolUse"
        );
        assert_eq!(usage.cache_read_tokens, 800, "cached_tokens must be mapped");
        assert_eq!(usage.input_tokens, 200, "uncached input = 1000 - 800");
        assert_eq!(
            usage.reasoning_tokens, 12,
            "reasoning_tokens must be mapped"
        );
        assert!(
            msg.content
                .iter()
                .any(|b| matches!(b, ContentBlock::Opaque { raw, .. } if raw.get("id") == Some(&json!("rs_1")))),
            "encrypted reasoning item must be preserved as Opaque"
        );
        assert!(
            msg.content.iter().any(|b| matches!(
                b,
                ContentBlock::ToolUse { id, name, input, invalid_json: None, .. }
                    if id == "fc_1" && name == "Read" && input["path"] == json!("lib.rs")
            )),
            "function_call must be reconstructed cleanly"
        );
        let _ = srv.await;
    }
}
