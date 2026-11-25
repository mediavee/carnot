//! Google Gemini (`api = "google-generative"`) and Cloud Code Envelope (`api = "google-cloudcode"`)
//! wire protocol adapters.
//!
//! Zero hardcoded provider names, URLs, or User-Agent strings! All endpoints, fallback URLs,
//! headers, system preambles, and thinking styles come strictly from `providers.json`.
//!
//! Protocol fidelity guarantees:
//! - Captures & replays `thoughtSignature` on the EXACT `ToolUse` part (via `provider_meta`)
//!   and on `Thinking` parts; injects `skip_thought_signature_validator` ONLY for unsigned
//!   cross-model function calls.
//! - Mutual exclusion of `thinkingLevel` (Gemini 3/4) vs `thinkingBudget` (Gemini 2.5) — never both!
//! - Accurate usage accounting: `cachedContentTokenCount` subtracted from `promptTokenCount`;
//!   `output_tokens = candidatesTokenCount + thoughtsTokenCount`.
//! - Inline multimodal images (`inlineData`) in both user messages and tool results.
//! - Deterministic synthetic tool call IDs generated once when a provider omits `call.id`.
//! - Cloud Code envelope (`{ model, project, requestType, userAgent, requestId, request }`),
//!   `chunk.response` unwrapping, Gemini schema sanitizer, and `fallback_base_urls` failover on 403/404.

use crate::anthropic::find_sse_frame_end;
use crate::catalog::ResolvedProviderSpec;
use crate::compat::{
    can_replay_origin, is_context_overflow, is_retryable_status, parse_retry_after_ms,
    sanitize_gemini_schema,
};
use crate::registry::ApiAdapter;
use crate::{BoxFuture, Provider, ProviderRequest, StreamEvent};
use carnot_core::{
    CarnotError, ContentBlock, ContentOrigin, Message, ProviderWebCapabilities, ReasoningEffort,
    Result, Role, StopReason, ToolResultBlock, Usage,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const API_GENERATIVE: &str = "google-generative";
const API_CLOUDCODE: &str = "google-cloudcode";
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(300);
const HTTP_HEADERS_TIMEOUT: Duration = Duration::from_secs(300);
const SKIP_THOUGHT_SIG_SENTINEL: &str = "skip_thought_signature_validator";

static SYNTHETIC_CALL_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct GoogleGenerativeAdapter;

impl ApiAdapter for GoogleGenerativeAdapter {
    fn api_id(&self) -> &'static str {
        API_GENERATIVE
    }

    fn build(&self, spec: &ResolvedProviderSpec) -> Result<Arc<dyn Provider>> {
        Ok(Arc::new(GoogleProvider::from_spec(spec.clone(), false)))
    }
}

#[derive(Debug)]
pub struct GoogleCloudCodeAdapter;

impl ApiAdapter for GoogleCloudCodeAdapter {
    fn api_id(&self) -> &'static str {
        API_CLOUDCODE
    }

    fn build(&self, spec: &ResolvedProviderSpec) -> Result<Arc<dyn Provider>> {
        Ok(Arc::new(GoogleProvider::from_spec(spec.clone(), true)))
    }
}

#[derive(Debug, Clone)]
pub struct GoogleProvider {
    client: reqwest::Client,
    spec: ResolvedProviderSpec,
    cloudcode_envelope: bool,
}

impl GoogleProvider {
    pub fn from_spec(spec: ResolvedProviderSpec, cloudcode_envelope: bool) -> Self {
        let client = crate::build_http_client_for_url(&spec.base_url);
        Self {
            client,
            spec,
            cloudcode_envelope,
        }
    }

    fn api_id(&self) -> &'static str {
        if self.cloudcode_envelope {
            API_CLOUDCODE
        } else {
            API_GENERATIVE
        }
    }

    /// Builds the inner Gemini `GenerateContent` request payload.
    pub fn build_inner_body(&self, req: &ProviderRequest) -> Value {
        let compat = self.spec.effective_compat(&req.model);
        let api_id = self.api_id();

        // Map tool_use_id -> function name so `functionResponse.name` always matches its call.
        let mut id_to_name: HashMap<String, String> = HashMap::new();
        for msg in &req.messages {
            for b in &msg.content {
                if let ContentBlock::ToolUse { id, name, .. } = b {
                    id_to_name.insert(id.clone(), name.clone());
                }
            }
        }

        // Static system prompt (+ optional provider data-driven preamble, kept constant for cache!)
        let mut system_parts: Vec<Value> = Vec::new();
        if let Some(ref preamble) = compat.system_preamble
            && !preamble.is_empty()
        {
            system_parts.push(json!({ "text": preamble }));
        }
        system_parts.push(json!({ "text": req.static_system_prompt }));

        let mut contents: Vec<Value> = Vec::new();
        for msg in &req.messages {
            let role = match msg.role {
                Role::Assistant => "model",
                Role::User | Role::System => "user",
            };
            let mut parts: Vec<Value> = Vec::new();
            let mut saw_first_fn_call_in_turn = false;

            for b in &msg.content {
                match b {
                    ContentBlock::Text { text } if !text.is_empty() => {
                        parts.push(json!({ "text": text }));
                    }
                    ContentBlock::Thinking {
                        thinking,
                        signature,
                        origin,
                    } => {
                        if can_replay_origin(
                            origin.as_ref(),
                            api_id,
                            &self.spec.provider_id,
                            &req.model,
                            false,
                        ) {
                            let mut part = json!({
                                "thought": true,
                                "text": thinking,
                            });
                            if let Some(sig) = signature
                                && let Some(obj) = part.as_object_mut()
                            {
                                obj.insert("thoughtSignature".into(), json!(sig));
                            }
                            parts.push(part);
                        }
                    }
                    ContentBlock::Image {
                        media_type,
                        data_base64,
                    } => {
                        parts.push(json!({
                            "inlineData": {
                                "mimeType": media_type,
                                "data": data_base64,
                            }
                        }));
                    }
                    ContentBlock::ToolUse {
                        id,
                        name,
                        input,
                        provider_meta,
                        origin,
                        ..
                    } => {
                        let same_origin = can_replay_origin(
                            origin.as_ref(),
                            api_id,
                            &self.spec.provider_id,
                            &req.model,
                            false,
                        );
                        let mut part = json!({
                            "functionCall": {
                                "id": id,
                                "name": name,
                                "args": input,
                            }
                        });
                        if let Some(obj) = part.as_object_mut() {
                            if same_origin {
                                if let Some(sig) = provider_meta
                                    .as_ref()
                                    .and_then(|m| m.get("thoughtSignature"))
                                    .and_then(Value::as_str)
                                {
                                    obj.insert("thoughtSignature".into(), json!(sig));
                                }
                            } else if !saw_first_fn_call_in_turn {
                                // Cross-model unsigned functionCall sentinel (never overwrite real sigs!)
                                obj.insert(
                                    "thoughtSignature".into(),
                                    json!(SKIP_THOUGHT_SIG_SENTINEL),
                                );
                            }
                        }
                        saw_first_fn_call_in_turn = true;
                        parts.push(part);
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        let fn_name = id_to_name
                            .get(tool_use_id)
                            .cloned()
                            .unwrap_or_else(|| "tool".to_string());
                        let mut text_out = String::new();
                        let mut inline_imgs: Vec<Value> = Vec::new();
                        for tb in content {
                            match tb {
                                ToolResultBlock::Text { text } => {
                                    if !text_out.is_empty() {
                                        text_out.push('\n');
                                    }
                                    text_out.push_str(text);
                                }
                                ToolResultBlock::Image {
                                    media_type,
                                    data_base64,
                                } => {
                                    inline_imgs.push(json!({
                                        "inlineData": {
                                            "mimeType": media_type,
                                            "data": data_base64,
                                        }
                                    }));
                                }
                            }
                        }
                        let resp_obj = if *is_error {
                            json!({ "error": text_out })
                        } else {
                            json!({ "output": text_out })
                        };
                        parts.push(json!({
                            "functionResponse": {
                                "id": tool_use_id,
                                "name": fn_name,
                                "response": resp_obj,
                            }
                        }));
                        parts.extend(inline_imgs);
                    }
                    ContentBlock::Opaque { .. } | ContentBlock::Text { .. } => {
                        // Drop foreign opaques (e.g. Anthropic `redacted_thinking`, OpenAI `reasoning`)
                        // and empty text blocks.
                    }
                }
            }

            if parts.is_empty() {
                continue;
            }
            // Coalesce consecutive identical roles (`user` + `user` or `model` + `model`)
            if let Some(last) = contents.last_mut()
                && last.get("role").and_then(Value::as_str) == Some(role)
                && let Some(prev_parts) = last.get_mut("parts").and_then(Value::as_array_mut)
            {
                prev_parts.extend(parts);
            } else {
                contents.push(json!({ "role": role, "parts": parts }));
            }
        }

        for note in &req.ephemeral_tail_notes {
            if !note.trim().is_empty() {
                if let Some(last) = contents.last_mut()
                    && last.get("role").and_then(Value::as_str) == Some("user")
                    && let Some(prev_parts) = last.get_mut("parts").and_then(Value::as_array_mut)
                {
                    prev_parts.push(json!({ "text": note }));
                } else {
                    contents.push(json!({ "role": "user", "parts": [{ "text": note }] }));
                }
            }
        }

        let sanitize = compat
            .sanitize_tool_schemas
            .unwrap_or(self.cloudcode_envelope);
        let decls: Vec<Value> = req
            .tools
            .iter()
            .filter(|t| !t.defer_loading)
            .map(|t| {
                if sanitize {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "parameters": sanitize_gemini_schema(&t.input_schema),
                    })
                } else {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "parametersJsonSchema": t.input_schema,
                    })
                }
            })
            .collect();

        let mut gen_cfg = json!({
            "maxOutputTokens": req.max_output_tokens,
        });
        if let Some(gc) = gen_cfg.as_object_mut() {
            let include_thoughts = compat.include_thoughts.unwrap_or(true);
            match compat.thinking_style.as_deref() {
                Some("level") => {
                    let level = match req.effort {
                        ReasoningEffort::Low => "LOW",
                        ReasoningEffort::Medium => "MEDIUM",
                        ReasoningEffort::High | ReasoningEffort::Xhigh | ReasoningEffort::Max => {
                            "HIGH"
                        }
                    };
                    gc.insert(
                        "thinkingConfig".into(),
                        json!({
                            "includeThoughts": include_thoughts,
                            "thinkingLevel": level,
                        }),
                    );
                }
                Some("budget") => {
                    let desired: u32 = match req.effort {
                        ReasoningEffort::Low => 2_048,
                        ReasoningEffort::Medium => 8_192,
                        ReasoningEffort::High => 16_384,
                        ReasoningEffort::Xhigh | ReasoningEffort::Max => 24_576,
                    };
                    let max_budget = req.max_output_tokens.saturating_sub(1_024).max(1_024);
                    let budget = desired.min(max_budget);
                    gc.insert(
                        "thinkingConfig".into(),
                        json!({
                            "includeThoughts": include_thoughts,
                            "thinkingBudget": budget,
                        }),
                    );
                }
                _ => {}
            }
        }

        let mut inner = json!({
            "systemInstruction": { "parts": system_parts },
            "contents": contents,
            "generationConfig": gen_cfg,
        });
        let mut tools_arr: Vec<Value> = Vec::new();
        if !decls.is_empty() {
            tools_arr.push(json!({ "functionDeclarations": decls }));
        }
        if req.native_web.search && compat.supports_native_web_search.unwrap_or(false) {
            tools_arr.push(json!({ "googleSearch": {} }));
        }
        if req.native_web.fetch && compat.supports_native_web_fetch.unwrap_or(false) {
            tools_arr.push(json!({ "urlContext": {} }));
        }
        if !tools_arr.is_empty()
            && let Some(obj) = inner.as_object_mut()
        {
            obj.insert("tools".into(), Value::Array(tools_arr));
        }
        inner
    }

    /// Builds the final wire body (either direct Gemini payload or Cloud Code envelope).
    pub fn build_body(&self, req: &ProviderRequest) -> Value {
        let compat = self.spec.effective_compat(&req.model);
        let inner = self.build_inner_body(req);
        let mut body = if self.cloudcode_envelope {
            let prefix_short: String = req.prefix_hash.chars().take(8).collect();
            let req_id = format!(
                "carnot-{prefix_short}-{}",
                SYNTHETIC_CALL_SEQ.fetch_add(1, Ordering::Relaxed)
            );
            let mut env = json!({
                "model": req.model,
                "request": inner,
                "requestType": compat.envelope_request_type.as_deref().unwrap_or("agent"),
                "userAgent": compat.envelope_user_agent.as_deref().unwrap_or("carnot"),
                "requestId": req_id,
            });
            if let Some(ref proj) = self.spec.project_id
                && let Some(obj) = env.as_object_mut()
            {
                obj.insert("project".into(), json!(proj));
            }
            env
        } else {
            inner
        };
        self.spec.apply_body_patches(&req.model, &mut body);
        body
    }

    fn endpoint_urls(&self, model: &str) -> Vec<String> {
        let mut bases = vec![self.spec.base_url.clone()];
        bases.extend(self.spec.fallback_base_urls.iter().cloned());
        bases
            .into_iter()
            .filter(|b| !b.is_empty())
            .map(|base| {
                let b = base.trim_end_matches('/');
                if self.cloudcode_envelope {
                    format!("{b}/v1internal:streamGenerateContent?alt=sse")
                } else if b.ends_with("/v1beta") || b.ends_with("/v1") || b.ends_with("/v1alpha") {
                    format!("{b}/models/{model}:streamGenerateContent?alt=sse")
                } else {
                    format!("{b}/v1beta/models/{model}:streamGenerateContent?alt=sse")
                }
            })
            .collect()
    }
}

impl Provider for GoogleProvider {
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
            let body = self.build_body(&request);
            let urls = self.endpoint_urls(&request.model);
            if urls.is_empty() {
                return Err(CarnotError::Provider {
                    provider: self.spec.provider_id.clone(),
                    message: "no base_url configured".into(),
                    retryable: false,
                    retry_after_ms: None,
                });
            }

            let mut resp_opt: Option<reqwest::Response> = None;
            let mut last_err: Option<CarnotError> = None;

            for (idx, raw_url) in urls.iter().enumerate() {
                let send_res = match crate::send_authorized_json_post(
                    &self.client,
                    &self.spec,
                    raw_url,
                    &[],
                    &body,
                    HTTP_HEADERS_TIMEOUT,
                )
                .await
                {
                    Ok(r) => r,
                    Err(err) => {
                        if idx + 1 < urls.len() {
                            tracing::warn!(
                                provider = %self.spec.provider_id,
                                "endpoint error; trying fallback_base_urls"
                            );
                            last_err = Some(err);
                            continue;
                        }
                        return Err(err);
                    }
                };

                let status = send_res.status();
                if status.is_success() {
                    resp_opt = Some(send_res);
                    break;
                }
                let retry_after_ms = parse_retry_after_ms(send_res.headers());
                let text = send_res.text().await.unwrap_or_default();
                if is_context_overflow(Some(status.as_u16()), &text) {
                    return Err(CarnotError::ContextOverflow(text));
                }

                // Fallback to next configured endpoint (e.g. daily -> prod) on 403 / 404!
                if matches!(status.as_u16(), 403 | 404) && idx + 1 < urls.len() {
                    tracing::warn!(
                        provider = %self.spec.provider_id,
                        status = status.as_u16(),
                        "endpoint returned 403/404; trying fallback_base_urls"
                    );
                    last_err = Some(CarnotError::Provider {
                        provider: self.spec.provider_id.clone(),
                        message: format!("HTTP {status}: {text}"),
                        retryable: false,
                        retry_after_ms,
                    });
                    continue;
                }

                return Err(CarnotError::Provider {
                    provider: self.spec.provider_id.clone(),
                    message: format!("HTTP {status}: {text}"),
                    retryable: is_retryable_status(status.as_u16()),
                    retry_after_ms,
                });
            }

            let Some(resp) = resp_opt else {
                return Err(last_err.unwrap_or_else(|| CarnotError::Provider {
                    provider: self.spec.provider_id.clone(),
                    message: "all endpoints failed".into(),
                    retryable: false,
                    retry_after_ms: None,
                }));
            };

            let (tx, rx) = mpsc::channel(64);
            let pricing = self.spec.effective_pricing(&request.model);
            let origin = ContentOrigin::new(self.api_id(), &self.spec.provider_id, &request.model);
            let unwrap_envelope = self.cloudcode_envelope;

            tokio::spawn(async move {
                let mut stream = resp.bytes_stream();
                let mut raw_buf: Vec<u8> = Vec::with_capacity(8192);
                let mut text_acc = String::new();
                let mut thinking_acc = String::new();
                let mut thinking_sig: Option<String> = None;
                let mut tool_blocks: Vec<ContentBlock> = Vec::new();
                let mut grounding_sources: Vec<(String, String)> = Vec::new();
                let mut usage = Usage::default();
                let mut stop_reason = StopReason::EndTurn;
                let mut saw_any_candidate = false;
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
                        let Ok(raw_v) = serde_json::from_str::<Value>(&joined) else {
                            continue;
                        };
                        if let Some(err) = raw_v.get("error") {
                            let err_str = err.to_string();
                            if is_context_overflow(None, &err_str) {
                                let _ = tx.send(StreamEvent::ContextOverflow(err_str)).await;
                                return;
                            }
                            let code = err.get("code").and_then(Value::as_u64).unwrap_or(500);
                            let retryable = is_retryable_status(code as u16);
                            let _ = tx
                                .send(StreamEvent::Error {
                                    message: err_str,
                                    retryable,
                                })
                                .await;
                            return;
                        }
                        let v = if unwrap_envelope {
                            raw_v.get("response").unwrap_or(&raw_v)
                        } else {
                            &raw_v
                        };

                        if let Some(um) = v.get("usageMetadata") {
                            let prompt_tok = um
                                .get("promptTokenCount")
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            let cached_tok = um
                                .get("cachedContentTokenCount")
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            let cand_tok = um
                                .get("candidatesTokenCount")
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            let thought_tok = um
                                .get("thoughtsTokenCount")
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            usage.cache_read_tokens = cached_tok;
                            usage.input_tokens = prompt_tok.saturating_sub(cached_tok);
                            usage.output_tokens = cand_tok.saturating_add(thought_tok);
                            usage.reasoning_tokens = thought_tok;
                        }

                        if let Some(cand) = v.pointer("/candidates/0") {
                            saw_any_candidate = true;
                            if let Some(fr) = cand.get("finishReason").and_then(Value::as_str) {
                                if fr == "MALFORMED_FUNCTION_CALL" {
                                    let _ = tx
                                        .send(StreamEvent::Error {
                                            message: "provider returned MALFORMED_FUNCTION_CALL"
                                                .into(),
                                            retryable: true,
                                        })
                                        .await;
                                    return;
                                }
                                stop_reason = match fr {
                                    "MAX_TOKENS" => StopReason::MaxTokens,
                                    "SAFETY" | "RECITATION" | "PROHIBITED_CONTENT"
                                    | "BLOCKLIST" => StopReason::Refusal,
                                    _ => StopReason::EndTurn,
                                };
                            }
                            if let Some(chunks) = cand
                                .pointer("/groundingMetadata/groundingChunks")
                                .and_then(Value::as_array)
                            {
                                for chunk in chunks {
                                    if grounding_sources.len() >= 10 {
                                        break;
                                    }
                                    if let Some(uri) =
                                        chunk.pointer("/web/uri").and_then(Value::as_str)
                                        && !uri.is_empty()
                                        && !grounding_sources.iter().any(|(_, u)| u == uri)
                                    {
                                        let title = chunk
                                            .pointer("/web/title")
                                            .and_then(Value::as_str)
                                            .filter(|t| !t.trim().is_empty())
                                            .unwrap_or(uri);
                                        grounding_sources
                                            .push((title.to_string(), uri.to_string()));
                                    }
                                }
                            }
                            if let Some(parts) =
                                cand.pointer("/content/parts").and_then(Value::as_array)
                            {
                                for part in parts {
                                    let is_thought =
                                        part.get("thought").and_then(Value::as_bool) == Some(true);
                                    let part_sig = part
                                        .get("thoughtSignature")
                                        .and_then(Value::as_str)
                                        .map(str::to_string);

                                    if let Some(fc) = part.get("functionCall") {
                                        let name = fc
                                            .get("name")
                                            .and_then(Value::as_str)
                                            .unwrap_or("")
                                            .to_string();
                                        let id = fc
                                            .get("id")
                                            .and_then(Value::as_str)
                                            .filter(|s| !s.is_empty())
                                            .map_or_else(
                                                || {
                                                    let seq = SYNTHETIC_CALL_SEQ
                                                        .fetch_add(1, Ordering::Relaxed);
                                                    format!("call_{name}_{seq}")
                                                },
                                                str::to_string,
                                            );
                                        let input =
                                            fc.get("args").cloned().unwrap_or_else(|| json!({}));
                                        let provider_meta =
                                            part_sig.map(|sig| json!({ "thoughtSignature": sig }));
                                        let _ = tx
                                            .send(StreamEvent::ToolCallReady {
                                                id: id.clone(),
                                                name: name.clone(),
                                                input: input.clone(),
                                                invalid_json: None,
                                            })
                                            .await;
                                        tool_blocks.push(ContentBlock::ToolUse {
                                            id,
                                            name,
                                            input,
                                            invalid_json: None,
                                            provider_meta,
                                            origin: Some(origin.clone()),
                                        });
                                    } else if let Some(txt) =
                                        part.get("text").and_then(Value::as_str)
                                    {
                                        if is_thought {
                                            thinking_acc.push_str(txt);
                                            if let Some(sig) = part_sig {
                                                thinking_sig = Some(sig);
                                            }
                                            let _ = tx
                                                .send(StreamEvent::ThinkingDelta(txt.to_string()))
                                                .await;
                                        } else {
                                            text_acc.push_str(txt);
                                            // Gemini 3 attaches thoughtSignature to the final text part on non-tool turns
                                            if let Some(sig) = part_sig {
                                                thinking_sig = Some(sig);
                                            }
                                            let _ = tx
                                                .send(StreamEvent::TextDelta(txt.to_string()))
                                                .await;
                                        }
                                    } else if is_thought && let Some(sig) = part_sig {
                                        thinking_sig = Some(sig);
                                    }
                                }
                            }
                        }
                    }
                }

                if !saw_any_candidate {
                    let _ = tx
                        .send(StreamEvent::Error {
                            message: "stream ended without any candidate response".into(),
                            retryable: true,
                        })
                        .await;
                    return;
                }

                let missing_sources: Vec<&(String, String)> = grounding_sources
                    .iter()
                    .filter(|(_, uri)| !text_acc.contains(uri.as_str()))
                    .collect();
                if !missing_sources.is_empty() && tool_blocks.is_empty() {
                    use std::fmt::Write as _;
                    let mut footer = String::from("\n\nSources:\n");
                    for (title, uri) in missing_sources {
                        let _ = writeln!(footer, "- [{title}]({uri})");
                    }
                    text_acc.push_str(&footer);
                    let _ = tx.send(StreamEvent::TextDelta(footer)).await;
                }

                let mut blocks: Vec<ContentBlock> = Vec::new();
                if !thinking_acc.is_empty() || thinking_sig.is_some() {
                    blocks.push(ContentBlock::Thinking {
                        thinking: thinking_acc,
                        signature: thinking_sig,
                        origin: Some(origin.clone()),
                    });
                }
                if !text_acc.is_empty() {
                    blocks.push(ContentBlock::Text { text: text_acc });
                }
                if !tool_blocks.is_empty() && stop_reason == StopReason::EndTurn {
                    stop_reason = StopReason::ToolUse;
                }
                blocks.extend(tool_blocks);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{CompatFlags, ResolvedAuth, UsageMap};
    use std::collections::BTreeMap;

    fn make_spec(base_url: &str) -> ResolvedProviderSpec {
        ResolvedProviderSpec {
            provider_id: "gemini".into(),
            api: "google-generative".into(),
            base_url: base_url.into(),
            fallback_base_urls: vec![],
            project_id: None,
            auth: ResolvedAuth::None,
            headers: BTreeMap::new(),
            query_params: BTreeMap::new(),
            betas: vec![],
            cache_diagnosis_beta: None,
            native_tool_search: false,
            compat: CompatFlags::default(),
            usage_map: UsageMap::default(),
            body_patch: None,
            subprocess: None,
            catalog_models: BTreeMap::new(),
        }
    }

    #[test]
    fn endpoint_urls_normalizes_bare_host_and_preserves_v1beta() {
        let p_bare = GoogleProvider::from_spec(make_spec("https://gateway.example.com"), false);
        assert_eq!(
            p_bare.endpoint_urls("gemini-3-pro"),
            vec![
                "https://gateway.example.com/v1beta/models/gemini-3-pro:streamGenerateContent?alt=sse"
            ]
        );

        let p_v1beta =
            GoogleProvider::from_spec(make_spec("https://api.example.com/v1beta/"), false);
        assert_eq!(
            p_v1beta.endpoint_urls("gemini-3-pro"),
            vec![
                "https://api.example.com/v1beta/models/gemini-3-pro:streamGenerateContent?alt=sse"
            ]
        );
    }
}
