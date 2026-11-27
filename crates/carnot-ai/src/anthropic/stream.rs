//! Anthropic Messages SSE frame framing and incremental block assembly.

use crate::StreamEvent;
use crate::compat::repair_json;
use carnot_core::{CacheTtl, ContentBlock, ContentOrigin, StopReason, Usage};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use tokio::sync::mpsc;

/// Finds the next SSE frame delimiter (`\n\n` or `\r\n\r\n`) in raw bytes.
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

#[derive(Default)]
pub(super) struct SseParseState {
    pub(super) blocks: BTreeMap<u64, BlockBuilder>,
    pub(super) usage: Usage,
    pub(super) stop_reason: StopReason,
    pub(super) saw_message_stop: bool,
    pub(super) message_id: Option<String>,
}

pub(super) enum BlockBuilder {
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
    pub(super) fn finish(self, origin: &ContentOrigin) -> ContentBlock {
        match self {
            Self::Text(text) => ContentBlock::Text { text },
            Self::Thinking { text, sig } => ContentBlock::Thinking {
                thinking: text,
                signature: sig,
                origin: Some(origin.clone()),
            },
            Self::ToolUse { id, name, json_buf } => {
                let trimmed = json_buf.trim();
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
                ContentBlock::ToolUse {
                    id,
                    name,
                    input,
                    invalid_json,
                    provider_meta: None,
                    origin: Some(origin.clone()),
                }
            }
            Self::Opaque { mut raw, json_buf } => {
                let trimmed = json_buf.trim();
                if !trimmed.is_empty()
                    && let Ok(parsed) = serde_json::from_str::<Value>(trimmed)
                    && let Some(obj) = raw.as_object_mut()
                {
                    obj.insert("input".into(), parsed);
                }
                ContentBlock::Opaque {
                    raw,
                    origin: Some(origin.clone()),
                }
            }
        }
    }
}

pub(super) async fn handle_sse_frame(
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
                        CacheTtl::Short | CacheTtl::Hybrid => {
                            state.usage.cache_write_5m_tokens += cw;
                        }
                        CacheTtl::Long => state.usage.cache_write_1h_tokens += cw,
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
                            sig.get_or_insert_with(String::new).push_str(s);
                        }
                    }
                    (
                        BlockBuilder::ToolUse { json_buf, .. }
                        | BlockBuilder::Opaque { json_buf, .. },
                        Some("input_json_delta"),
                    ) => {
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
                } else if let Some(val) = repair_json(trimmed) {
                    (val, None)
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
