//! Anthropic Messages wire request body construction and content-block encoding.

use super::{API_ID, AnthropicProvider};
use crate::ProviderRequest;
use crate::compat::{can_replay_origin, normalize_tool_call_id};
use carnot_core::{CacheTtl, ContentBlock, ReasoningEffort, Role, ToolResultBlock};
use serde_json::{Value, json};

impl AnthropicProvider {
    /// Builds the Anthropic JSON body with:
    ///   - 4-breakpoint cache planning (Hybrid 1h static / 5m rolling tail) placed strictly on
    ///     the last non-thinking block
    ///   - Native `defer_loading` + server `tool_search_tool_regex_20251119` when deferred tools exist
    ///   - `eager_input_streaming: true` on `Edit` and `Write` for lower TTFT on large edits
    ///   - Origin-aware replay: strips foreign `Thinking` signatures / foreign `Opaque` blocks
    ///   - Valid Anthropic role alternation & native `role: "system"` placement
    ///   - `output_config.effort` + declarative `body_patch`
    pub fn build_body(&self, req: &ProviderRequest) -> Value {
        let cache_ctrl = |ttl: CacheTtl| match ttl {
            CacheTtl::Short => json!({"type": "ephemeral", "ttl": "5m"}),
            CacheTtl::Long | CacheTtl::Hybrid => json!({"type": "ephemeral", "ttl": "1h"}),
        };
        let static_cc = cache_ctrl(req.cache_ttl.static_prefix_tier());
        let conv_cc = cache_ctrl(req.cache_ttl.conversation_tier());

        let has_deferred = req.tools.iter().any(|t| t.defer_loading);
        let compat = self.spec.effective_compat(&req.model);
        let want_native_search =
            req.native_web.search && compat.supports_native_web_search.unwrap_or(false);
        let want_native_fetch =
            req.native_web.fetch && compat.supports_native_web_fetch.unwrap_or(false);
        let mut tools_json: Vec<Value> = Vec::with_capacity(
            req.tools.len()
                + usize::from(has_deferred)
                + usize::from(want_native_search)
                + usize::from(want_native_fetch),
        );
        if has_deferred {
            tools_json.push(json!({
                "type": "tool_search_tool_regex_20251119",
                "name": "tool_search_tool_regex",
            }));
        }
        if want_native_search {
            let mut ws = json!({
                "type": compat
                    .native_web_search_type
                    .as_deref()
                    .unwrap_or("web_search_20250305"),
                "name": "web_search",
                "max_uses": req.native_web.max_uses.unwrap_or(5).max(1),
            });
            if let Some(m) = ws.as_object_mut() {
                if !req.native_web.allowed_domains.is_empty() {
                    m.insert(
                        "allowed_domains".into(),
                        json!(req.native_web.allowed_domains),
                    );
                } else if !req.native_web.blocked_domains.is_empty() {
                    m.insert(
                        "blocked_domains".into(),
                        json!(req.native_web.blocked_domains),
                    );
                }
            }
            tools_json.push(ws);
        }
        if want_native_fetch {
            let mut wf = json!({
                "type": compat
                    .native_web_fetch_type
                    .as_deref()
                    .unwrap_or("web_fetch_20250910"),
                "name": "web_fetch",
                "max_uses": req.native_web.max_uses.unwrap_or(5).max(1),
                "max_content_tokens": req.native_web.max_content_tokens.unwrap_or(12_000).max(512),
            });
            if let Some(m) = wf.as_object_mut() {
                if !req.native_web.allowed_domains.is_empty() {
                    m.insert(
                        "allowed_domains".into(),
                        json!(req.native_web.allowed_domains),
                    );
                } else if !req.native_web.blocked_domains.is_empty() {
                    m.insert(
                        "blocked_domains".into(),
                        json!(req.native_web.blocked_domains),
                    );
                }
            }
            tools_json.push(wf);
        }
        let allow_eager = compat.supports_eager_input_streaming.unwrap_or(true);
        let require_sig = compat.requires_thinking_signature.unwrap_or(true);
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
                if allow_eager && matches!(t.name.as_str(), "Edit" | "Write") {
                    m.insert("eager_input_streaming".into(), json!(true));
                }
            }
            tools_json.push(obj);
        }
        // BP1 (static prefix tier -> 1h in Hybrid): single breakpoint on static system prompt
        // caches `tools` + `static_system_prompt` together (since both are frozen in Carnot),
        // freeing a full cache_control slot for Turn-1 (AGENTS.md + workspace-facts + goal) on 1h!
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
            let blocks = encode_blocks(
                &msg.content,
                &self.spec.provider_id,
                &req.model,
                require_sig,
            );
            if blocks.is_empty() {
                continue;
            }

            if msg.role == Role::System {
                let prev_is_user = wire_messages.last().is_some_and(|(r, _, _)| r == "user");
                let next_ok = req
                    .messages
                    .get(idx + 1)
                    .is_none_or(|m| m.role == Role::Assistant);
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
        let rolling_bp_idx = if req.skip_tail_cache_write {
            // One-shot compaction call: hit the warm prefix cached on the previous turn
            // (`last_idx - 2`) without paying a 1.25x/2.0x cache-write surcharge on the
            // discarded tail + compaction prompt.
            last_idx.checked_sub(2)
        } else {
            Some(last_idx)
        };
        let msg_count = wire_messages.len();
        let mut final_messages: Vec<Value> = wire_messages
            .into_iter()
            .enumerate()
            .map(|(idx, (role, mut blocks, is_cp))| {
                // 4-slot monotonic TTL ladder (1h -> 1h -> 5m -> 5m in Hybrid, max 4 blocks total):
                //   BP1: system[0] (`static_cc` = 1h) — covers tools + static system prompt
                //   BP2: messages[0] (`static_cc` = 1h) — covers Turn-1 (AGENTS.md + workspace-facts + goal)
                //   BP3: checkpoint_index > 0 (`conv_cc` = 5m)
                //   BP4: rolling_bp_idx > 0 (`conv_cc` = 5m)
                let cc_for_msg: Option<&Value> = if idx == 0 {
                    if req.skip_tail_cache_write && msg_count == 1 {
                        None
                    } else {
                        Some(&static_cc)
                    }
                } else if (is_cp && Some(idx) != rolling_bp_idx && idx != last_idx)
                    || Some(idx) == rolling_bp_idx
                {
                    Some(&conv_cc)
                } else {
                    None
                };
                if let Some(cc) = cc_for_msg {
                    // Attach cache_control only to the last non-thinking / non-server-tool-use block.
                    if let Some(target) = blocks.iter_mut().rev().find(|b| {
                        !matches!(
                            b.get("type").and_then(Value::as_str),
                            Some("thinking" | "redacted_thinking" | "server_tool_use")
                        )
                    }) && let Some(obj) = target.as_object_mut()
                    {
                        obj.insert("cache_control".into(), cc.clone());
                    }
                }
                json!({"role": role, "content": blocks})
            })
            .collect();

        // Append ephemeral per-turn notes AFTER all `cache_control` breakpoints are placed so
        // BP2/BP4 always sit strictly on persisted blocks and transient notes stay uncached!
        if !req.ephemeral_tail_notes.is_empty() {
            let note_blocks: Vec<Value> = req
                .ephemeral_tail_notes
                .iter()
                .filter(|n| !n.trim().is_empty())
                .map(|n| json!({"type": "text", "text": n}))
                .collect();
            if !note_blocks.is_empty() {
                let append_to_last_user = final_messages
                    .last()
                    .and_then(|m| m.get("role"))
                    .and_then(Value::as_str)
                    == Some("user");
                if append_to_last_user
                    && let Some(arr) = final_messages
                        .last_mut()
                        .and_then(|m| m.get_mut("content"))
                        .and_then(Value::as_array_mut)
                {
                    arr.extend(note_blocks);
                } else {
                    final_messages.push(json!({"role": "user", "content": note_blocks}));
                }
            }
        }

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

fn encode_blocks(
    blocks: &[ContentBlock],
    provider_id: &str,
    model: &str,
    require_thinking_signature: bool,
) -> Vec<Value> {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(json!({"type": "text", "text": text})),
            ContentBlock::Thinking {
                thinking,
                signature,
                origin,
            } => {
                // Prevent HTTP 400 after MaxTokens mid-thinking truncation: Claude requires a
                // cryptographic signature on every replayed thinking block, while unsigned
                // Anthropic-compatible gateways can set `requires_thinking_signature: false`.
                if require_thinking_signature && signature.is_none() {
                    return None;
                }
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
                "id": normalize_tool_call_id(id),
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
                    "tool_use_id": normalize_tool_call_id(tool_use_id),
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
                        Some(
                            "redacted_thinking"
                                | "server_tool_use"
                                | "web_search_tool_result"
                                | "web_fetch_tool_result"
                        )
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
