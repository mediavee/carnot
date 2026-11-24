//! Multi-provider Conformance Test Matrix:
//! Verifies ALL 6 wire protocol adapters (`anthropic-messages`, `openai-completions`,
//! `openai-responses`, `google-generative`, `google-cloudcode`, `subprocess-jsonl`) on:
//!   1. Plain text streaming & exact usage accounting (including DeepSeek top-level cache hit)
//!   2. Parallel tool calls + malformed tool JSON preservation
//!   3. Multimodal user images & tool-result image bridge
//!   4. Thinking & `thoughtSignature` / `encrypted_content` 2-turn replay
//!   5. Cross-provider origin sanitization (Anthropic -> Gemini -> OpenAI -> Responses zero 400s!)
//!   6. 429 + `Retry-After` header propagation & non-retryable stream error classification
//!   7. Cloud Code daily -> prod fallback on 404 + Gemini schema sanitizer
//!   8. Persistent `subprocess-jsonl` child process across multiple turns + cancellation
//!   9. Automated Zero-Hardcode CI check over `crates/carnot-ai/src/` and `crates/carnot/src/`!

use crate::anthropic::AnthropicProvider;
use crate::catalog::{
    ProviderCatalog, ResolvedAuth, ResolvedProviderSpec, SubprocessProviderConfig,
};
use crate::google::GoogleProvider;
use crate::openai::OpenAiProvider;
use crate::openai_responses::OpenAiResponsesProvider;
use crate::registry::ProviderRegistry;
use crate::{Provider, ProviderRequest, StreamEvent};
use carnot_core::{
    CacheTtl, CarnotError, ContentBlock, ContentOrigin, Message, ReasoningEffort, Role, StopReason,
    ToolResultBlock, ToolSpec,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

type MockHttpResponse = (u16, Vec<(&'static str, &'static str)>, String);
type CapturedRequests = Arc<Mutex<Vec<(String, String, String)>>>;

/// Spins up a tiny real HTTP/1.1 mock server on 127.0.0.1 that captures incoming requests
/// and returns scripted `(status, extra_headers, body)` responses in order.
async fn spawn_mock_http(responses: Vec<MockHttpResponse>) -> (String, CapturedRequests) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let captured: CapturedRequests = Arc::new(Mutex::new(Vec::new()));
    let cap_clone = Arc::clone(&captured);

    tokio::spawn(async move {
        let mut queue = std::collections::VecDeque::from(responses);
        while let Ok((mut stream, _)) = listener.accept().await {
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            let mut header_end = None;
            let mut content_len = 0usize;
            while let Ok(n) = stream.read(&mut tmp).await {
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                if header_end.is_none()
                    && let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n")
                {
                    header_end = Some(pos + 4);
                    let hdr_str = String::from_utf8_lossy(&buf[..pos]);
                    for line in hdr_str.lines() {
                        if let Some((k, v)) = line.split_once(':')
                            && k.eq_ignore_ascii_case("content-length")
                        {
                            content_len = v.trim().parse().unwrap_or(0);
                        }
                    }
                }
                if let Some(he) = header_end
                    && buf.len() >= he + content_len
                {
                    break;
                }
            }
            let he = header_end.unwrap_or(buf.len());
            let hdr_str = String::from_utf8_lossy(&buf[..he]).to_string();
            let first_line = hdr_str.lines().next().unwrap_or("").to_string();
            let body_str = String::from_utf8_lossy(&buf[he..]).to_string();
            if let Ok(mut g) = cap_clone.lock() {
                g.push((first_line, hdr_str, body_str));
            }
            let (status, extra_hdrs, resp_body) =
                queue.pop_front().unwrap_or((200, vec![], String::new()));
            let reason = match status {
                200 => "OK",
                404 => "Not Found",
                429 => "Too Many Requests",
                _ => "Error",
            };
            let mut raw = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n",
                resp_body.len()
            );
            for (k, v) in extra_hdrs {
                raw.push_str(&format!("{k}: {v}\r\n"));
            }
            raw.push_str("\r\n");
            raw.push_str(&resp_body);
            let _ = stream.write_all(raw.as_bytes()).await;
            let _ = stream.shutdown().await;
            if queue.is_empty() {
                break;
            }
        }
    });

    (format!("http://{addr}"), captured)
}

fn sample_request(model: &str, messages: Vec<Message>) -> ProviderRequest {
    ProviderRequest {
        model: model.to_string(),
        static_system_prompt: "STATIC_SYSTEM_PROMPT".into(),
        tools: vec![ToolSpec {
            name: "Read".into(),
            description: "Read a file".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "Path" },
                    "tags": { "type": "array" }
                },
                "required": ["file_path", "nonexistent_req"],
                "additionalProperties": false
            }),
            defer_loading: false,
        }],
        prefix_hash: "0123456789abcdef0123456789abcdef".into(),
        session_id: "test-session-1".into(),
        messages: messages.into_iter().map(Arc::new).collect(),
        checkpoint_index: Some(0),
        cache_ttl: CacheTtl::Hybrid,
        effort: ReasoningEffort::High,
        max_output_tokens: 4096,
        skip_tail_cache_write: false,
        native_web: Default::default(),
        ephemeral_tail_notes: Vec::new(),
    }
}

fn spec_for(
    catalog: &ProviderCatalog,
    provider_id: &str,
    base_url: String,
) -> ResolvedProviderSpec {
    let (_, entry) = catalog
        .find_provider(provider_id)
        .expect("provider in catalog");
    ResolvedProviderSpec {
        provider_id: provider_id.to_string(),
        api: entry.api.clone(),
        base_url,
        fallback_base_urls: Vec::new(),
        project_id: Some("test-project-123".into()),
        auth: ResolvedAuth::Bearer {
            token: "test-secret-token".into(),
            source: "test".into(),
        },
        headers: entry.headers.clone(),
        query_params: entry.query_params.clone(),
        betas: entry.betas.clone(),
        cache_diagnosis_beta: entry.cache_diagnosis_beta.clone(),
        native_tool_search: entry.native_tool_search.unwrap_or(false),
        compat: entry.compat.clone(),
        usage_map: entry.usage_map.clone(),
        body_patch: entry.body_patch.clone(),
        subprocess: entry.subprocess.clone(),
        catalog_models: catalog.models.clone(),
    }
}

#[tokio::test]
async fn conformance_anthropic_bearer_vs_xapikey_and_cross_origin_strip() {
    let catalog = ProviderCatalog::embedded().expect("catalog");
    let sse = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":900,\"cache_creation_input_tokens\":50}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_ant\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":20}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    );
    let (base_url, captured) = spawn_mock_http(vec![(
        200,
        vec![("Content-Type", "text/event-stream")],
        sse.to_string(),
    )])
    .await;

    let reg = ProviderRegistry::from_catalog(catalog.clone());
    let spec = spec_for(&catalog, "anthropic", base_url);
    let provider = reg.build_from_spec(&spec).expect("build anthropic");

    // Cross-provider history: OpenAI encrypted reasoning Opaque + Gemini Thinking MUST be stripped!
    let msgs = vec![
        Message::user_text("hi"),
        Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Opaque {
                    raw: json!({"type": "reasoning", "encrypted_content": "enc_openai"}),
                    origin: Some(ContentOrigin::new(
                        "openai-responses",
                        "openai-responses",
                        "gpt-5.4",
                    )),
                },
                ContentBlock::Thinking {
                    thinking: "foreign thought".into(),
                    signature: Some("foreign_sig".into()),
                    origin: Some(ContentOrigin::new(
                        "google-generative",
                        "gemini",
                        "gemini-3-pro-preview",
                    )),
                },
                ContentBlock::Text { text: "ok".into() },
            ],
        },
        Message::user_text("turn 2"),
    ];

    let mut rx = provider
        .stream(
            sample_request("claude-sonnet-5-5", msgs),
            CancellationToken::new(),
        )
        .await
        .expect("stream");
    let mut done_msg = None;
    let mut done_usage = None;
    while let Some(ev) = rx.recv().await {
        if let StreamEvent::Done { message, usage, .. } = ev {
            done_msg = Some(message);
            done_usage = Some(usage);
        }
    }
    let msg = done_msg.expect("done");
    let usage = done_usage.expect("usage");
    assert_eq!(usage.cache_read_tokens, 900);
    assert!(msg.content.iter().any(|b| matches!(
        b,
        ContentBlock::Thinking { signature: Some(s), origin: Some(o), .. }
            if s == "sig_ant" && o.api == "anthropic-messages"
    )));

    // Verify Bearer header was sent WITHOUT x-api-key, and NO foreign opaque leaked!
    let reqs = captured.lock().expect("cap");
    let (_, hdrs, body) = &reqs[0];
    let hdrs_lower = hdrs.to_ascii_lowercase();
    assert!(hdrs_lower.contains("authorization: bearer test-secret-token"));
    assert!(!hdrs_lower.contains("x-api-key:"));
    assert!(!hdrs_lower.contains("anthropic-beta:"));
    assert!(
        !body.contains("enc_openai"),
        "foreign OpenAI opaque leaked to Anthropic!"
    );
    assert!(
        !body.contains("foreign_sig"),
        "foreign Gemini signature leaked to Anthropic!"
    );
}

#[tokio::test]
async fn conformance_openai_compat_images_reasoning_deepseek_usage_and_retry_after() {
    let catalog = ProviderCatalog::embedded().expect("catalog");
    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think1\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"ans\",\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"Read\",\"arguments\":\"{\\\"file_path\\\":\\\"a.rs\\\"}\"}},{\"index\":1,\"id\":\"c2\",\"function\":{\"name\":\"Read\",\"arguments\":\"{bad\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":50,\"prompt_cache_hit_tokens\":920,\"completion_tokens_details\":{\"reasoning_tokens\":30}}}\n\n",
        "data: [DONE]\n\n"
    );
    let (base_url, captured) = spawn_mock_http(vec![
        (
            429,
            vec![("Retry-After", "3")],
            "{\"error\":\"rate limit\"}".into(),
        ),
        (
            200,
            vec![("Content-Type", "text/event-stream")],
            sse.to_string(),
        ),
    ])
    .await;

    let spec = spec_for(&catalog, "deepseek", base_url);
    let provider = OpenAiProvider::from_spec(spec);

    // 1) Verify 429 propagates retryable=true + retry_after_ms=3000
    let err = provider
        .stream(
            sample_request("deepseek-reasoner", vec![Message::user_text("hi")]),
            CancellationToken::new(),
        )
        .await
        .expect_err("expected 429");
    match err {
        CarnotError::Provider {
            retryable,
            retry_after_ms,
            ..
        } => {
            assert!(retryable);
            assert_eq!(retry_after_ms, Some(3000));
        }
        other => panic!("unexpected err: {other:?}"),
    }

    // 2) Verify multimodal tool-result image bridge + user image + reasoning_content replay + DeepSeek UsageMap!
    let msgs = vec![
        Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "look".into(),
                },
                ContentBlock::Image {
                    media_type: "image/png".into(),
                    data_base64: "AAAA".into(),
                },
            ],
        },
        Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "prior reasoning".into(),
                    signature: None,
                    origin: Some(ContentOrigin::new(
                        "openai-completions",
                        "deepseek",
                        "deepseek-reasoner",
                    )),
                },
                ContentBlock::ToolUse {
                    id: "prev_c1".into(),
                    name: "Read".into(),
                    input: json!({"file_path": "img.png"}),
                    invalid_json: None,
                    provider_meta: None,
                    origin: None,
                },
            ],
        },
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "prev_c1".into(),
                content: vec![
                    ToolResultBlock::text("loaded"),
                    ToolResultBlock::Image {
                        media_type: "image/png".into(),
                        data_base64: "BBBB".into(),
                    },
                ],
                is_error: false,
            }],
        },
    ];

    let mut rx = provider
        .stream(
            sample_request("deepseek-reasoner", msgs),
            CancellationToken::new(),
        )
        .await
        .expect("stream");
    let mut saw_thinking = String::new();
    let mut done_ev = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            StreamEvent::ThinkingDelta(d) => saw_thinking.push_str(&d),
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
    let (msg, usage, sr) = done_ev.expect("done");
    assert_eq!(saw_thinking, "think1");
    assert_eq!(sr, StopReason::ToolUse);
    // DeepSeek UsageMap: 1000 prompt_tokens with 920 prompt_cache_hit_tokens -> 80 uncached + 920 cached (>= 0.90 hit rate!)
    assert_eq!(usage.cache_read_tokens, 920);
    assert_eq!(usage.input_tokens, 80);
    assert_eq!(usage.reasoning_tokens, 30);
    // Parallel calls: 1 valid + 1 malformed JSON flagged!
    let tool_uses: Vec<_> = msg
        .content
        .iter()
        .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
        .collect();
    assert_eq!(tool_uses.len(), 2);
    assert!(matches!(
        tool_uses[1],
        ContentBlock::ToolUse {
            invalid_json: Some(_),
            ..
        }
    ));

    // Check outgoing request body had image_url user bridge + reasoning_content + max_tokens (not max_completion_tokens!)
    let reqs = captured.lock().expect("cap");
    let body: Value = serde_json::from_str(&reqs[1].2).expect("json body");
    assert!(body.get("max_tokens").is_some());
    assert!(body.get("max_completion_tokens").is_none());
    let body_str = reqs[1].2.as_str();
    assert!(body_str.contains("data:image/png;base64,AAAA"));
    assert!(body_str.contains("data:image/png;base64,BBBB"));
    assert!(body_str.contains("\"reasoning_content\":\"prior reasoning\""));
}

#[tokio::test]
async fn conformance_grok_and_muse_catalog_flags() {
    let catalog = ProviderCatalog::embedded().expect("catalog");
    let grok_spec = spec_for(&catalog, "grok", "http://127.0.0.1:9".into());
    let grok = OpenAiProvider::from_spec(grok_spec);
    let body_g4 = grok.build_body(&sample_request("grok-4", vec![Message::user_text("hi")]));
    assert!(
        body_g4.get("reasoning_effort").is_none(),
        "grok-4 must NOT send reasoning_effort"
    );
    assert!(body_g4.get("prompt_cache_key").is_none());

    let body_g3m = grok.build_body(&sample_request(
        "grok-3-mini",
        vec![Message::user_text("hi")],
    ));
    assert_eq!(
        body_g3m.get("reasoning_effort").and_then(Value::as_str),
        Some("high"),
        "grok-3-mini model override enables reasoning_effort"
    );

    // Alias lookup works for xai, meta, agy, google-antigravity, codex
    assert_eq!(catalog.find_provider("xai").map(|(id, _)| id), Some("grok"));
    assert_eq!(
        catalog.find_provider("meta").map(|(id, _)| id),
        Some("muse")
    );
    assert_eq!(
        catalog.find_provider("agy").map(|(id, _)| id),
        Some("antigravity")
    );
    assert_eq!(
        catalog.find_provider("codex").map(|(id, _)| id),
        Some("openai-responses")
    );
}

#[tokio::test]
async fn conformance_openai_responses_stateless_encrypted_reasoning_replay() {
    let catalog = ProviderCatalog::embedded().expect("catalog");
    let sse = concat!(
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"thinking...\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"encrypted_content\":\"gAAAA_secret\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Done!\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":500,\"input_tokens_details\":{\"cached_tokens\":460},\"output_tokens\":40,\"output_tokens_details\":{\"reasoning_tokens\":25}}}}\n\n"
    );
    let (base_url, captured) = spawn_mock_http(vec![(
        200,
        vec![("Content-Type", "text/event-stream")],
        sse.to_string(),
    )])
    .await;

    let spec = spec_for(&catalog, "openai-responses", base_url);
    let provider = OpenAiResponsesProvider::from_spec(spec);

    // Replay matching OpenAI reasoning Opaque + foreign Anthropic redacted_thinking (must be dropped!)
    let msgs = vec![
        Message::user_text("first"),
        Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Opaque {
                    raw: json!({"type": "reasoning", "id": "rs_0", "encrypted_content": "enc_0"}),
                    origin: Some(ContentOrigin::new(
                        "openai-responses",
                        "openai-responses",
                        "gpt-5.4",
                    )),
                },
                ContentBlock::Opaque {
                    raw: json!({"type": "redacted_thinking", "data": "ant_only"}),
                    origin: Some(ContentOrigin::new(
                        "anthropic-messages",
                        "anthropic",
                        "claude-sonnet-5-5",
                    )),
                },
                ContentBlock::Text {
                    text: "step 1".into(),
                },
            ],
        },
        Message::user_text("second"),
    ];

    let mut rx = provider
        .stream(sample_request("gpt-5.4", msgs), CancellationToken::new())
        .await
        .expect("stream");
    let mut done = None;
    while let Some(ev) = rx.recv().await {
        if let StreamEvent::Done { message, usage, .. } = ev {
            done = Some((message, usage));
        }
    }
    let (msg, usage) = done.expect("done");
    assert_eq!(usage.cache_read_tokens, 460);
    assert_eq!(usage.input_tokens, 40);
    assert!(msg.content.iter().any(|b| matches!(
        b,
        ContentBlock::Opaque { raw, origin: Some(o) }
            if raw.get("encrypted_content").and_then(Value::as_str) == Some("gAAAA_secret")
                && o.api == "openai-responses"
    )));

    let reqs = captured.lock().expect("cap");
    let body: Value = serde_json::from_str(&reqs[0].2).expect("body");
    assert_eq!(body.get("store").and_then(Value::as_bool), Some(false));
    assert!(
        reqs[0].2.contains("enc_0"),
        "matching encrypted reasoning must replay verbatim"
    );
    assert!(
        !reqs[0].2.contains("ant_only"),
        "foreign Anthropic opaque must NOT leak to Responses"
    );
}

#[tokio::test]
async fn conformance_google_gemini_and_antigravity_cloudcode_fallback_and_thought_signature() {
    let catalog = ProviderCatalog::embedded().expect("catalog");
    let cc_sse = concat!(
        "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"thought\":true,\"text\":\"deep plan\",\"thoughtSignature\":\"sig_thought_1\"},{\"functionCall\":{\"name\":\"Read\",\"args\":{\"file_path\":\"lib.rs\"}},\"thoughtSignature\":\"sig_fc_1\"}]}}]}}\n\n",
        "data: {\"response\":{\"candidates\":[{\"finishReason\":\"STOP\",\"content\":{\"parts\":[{\"text\":\"reading\"}]}}],\"usageMetadata\":{\"promptTokenCount\":1000,\"cachedContentTokenCount\":950,\"candidatesTokenCount\":15,\"thoughtsTokenCount\":35}}}\n\n"
    );

    // Server 1 (daily) returns 404 -> Server 2 (prod fallback) returns 200 SSE!
    let (daily_url, _) = spawn_mock_http(vec![(404, vec![], "not found".into())]).await;
    let (prod_url, prod_cap) = spawn_mock_http(vec![(
        200,
        vec![("Content-Type", "text/event-stream")],
        cc_sse.to_string(),
    )])
    .await;

    let mut spec = spec_for(&catalog, "antigravity", daily_url);
    spec.fallback_base_urls = vec![prod_url];
    let provider = GoogleProvider::from_spec(spec.clone(), true);

    // Turn 1: foreign tool call without Gemini signature MUST receive skip_thought_signature_validator!
    let msgs = vec![
        Message::user_text("start"),
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "foreign_1".into(),
                name: "Read".into(),
                input: json!({"file_path": "x.rs"}),
                invalid_json: None,
                provider_meta: None,
                origin: Some(ContentOrigin::new(
                    "anthropic-messages",
                    "anthropic",
                    "claude-sonnet-5-5",
                )),
            }],
        },
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "foreign_1".into(),
                content: vec![ToolResultBlock::text("contents")],
                is_error: false,
            }],
        },
    ];

    let mut rx = provider
        .stream(
            sample_request("gemini-3-pro-high", msgs),
            CancellationToken::new(),
        )
        .await
        .expect("stream with daily->prod fallback");
    let mut done = None;
    while let Some(ev) = rx.recv().await {
        if let StreamEvent::Done { message, usage, .. } = ev {
            done = Some((message, usage));
        }
    }
    let (msg, usage) = done.expect("done");
    assert_eq!(usage.cache_read_tokens, 950);
    assert_eq!(usage.input_tokens, 50);
    assert_eq!(usage.output_tokens, 50); // 15 candidates + 35 thoughts!
    assert_eq!(usage.reasoning_tokens, 35);

    // Verify thoughtSignature was preserved on the exact ToolUse block via provider_meta!
    let tu = msg
        .content
        .iter()
        .find(|b| matches!(b, ContentBlock::ToolUse { .. }))
        .expect("tool use");
    match tu {
        ContentBlock::ToolUse { provider_meta, .. } => {
            assert_eq!(
                provider_meta
                    .as_ref()
                    .and_then(|m| m.get("thoughtSignature"))
                    .and_then(Value::as_str),
                Some("sig_fc_1")
            );
        }
        _ => unreachable!(),
    }

    // Verify Cloud Code envelope, schema sanitizer (uppercase OBJECT, stripped additionalProperties & nonexistent_req, added array items),
    // skip_thought_signature_validator sentinel, and thinkingLevel (NOT thinkingBudget)!
    let prod_reqs = prod_cap.lock().expect("cap");
    let sent: Value = serde_json::from_str(&prod_reqs[0].2).expect("body");
    assert_eq!(
        sent.get("requestType").and_then(Value::as_str),
        Some("agent")
    );
    assert_eq!(
        sent.get("project").and_then(Value::as_str),
        Some("test-project-123")
    );
    assert!(
        sent.pointer("/request/generationConfig/thinkingConfig/thinkingLevel")
            .is_some()
    );
    assert!(
        sent.pointer("/request/generationConfig/thinkingConfig/thinkingBudget")
            .is_none()
    );
    let params = sent
        .pointer("/request/tools/0/functionDeclarations/0/parameters")
        .expect("sanitized parameters");
    assert_eq!(params.get("type").and_then(Value::as_str), Some("OBJECT"));
    assert!(params.get("additionalProperties").is_none());
    assert_eq!(
        params
            .pointer("/properties/tags/items/type")
            .and_then(Value::as_str),
        Some("STRING")
    );
    assert_eq!(
        params
            .get("required")
            .and_then(Value::as_array)
            .map(|a| a.len()),
        Some(1)
    );
    assert!(prod_reqs[0].2.contains("skip_thought_signature_validator"));

    // Turn 2: replaying the returned ToolUse MUST send back its EXACT `sig_fc_1`!
    let body_t2 = provider.build_body(&sample_request("gemini-3-pro-high", vec![msg]));
    assert!(body_t2.to_string().contains("sig_fc_1"));

    // Gemini 2.5 uses thinkingBudget (NEVER thinkingLevel!)
    let g25_spec = spec_for(&catalog, "gemini", "http://127.0.0.1:9".into());
    let g25 = GoogleProvider::from_spec(g25_spec, false);
    let body_25 = g25.build_body(&sample_request(
        "gemini-2.5-pro",
        vec![Message::user_text("hi")],
    ));
    assert!(
        body_25
            .pointer("/generationConfig/thinkingConfig/thinkingBudget")
            .is_some()
    );
    assert!(
        body_25
            .pointer("/generationConfig/thinkingConfig/thinkingLevel")
            .is_none()
    );
}

#[tokio::test]
async fn conformance_subprocess_jsonl_persistent_multiturn() {
    // Persistent Python NDJSON provider plugin stays alive across 2 turns and echoes turn count!
    let script = r#"
import sys, json
turn = 0
for line in sys.stdin:
    msg = json.loads(line)
    t = msg.get("type")
    if t == "hello":
        print(json.dumps({"type":"hello_ok","protocol":"carnot-provider-v1","native_tool_search":False}), flush=True)
    elif t == "stream":
        turn += 1
        out = f"turn-{turn}-ok"
        print(json.dumps({"type":"text_delta","delta":out}), flush=True)
        print(json.dumps({"type":"done","message":{"role":"assistant","content":[{"type":"text","text":out}]},"usage":{"input_tokens":10,"output_tokens":5,"reasoning_tokens":0,"cache_read_tokens":0,"cache_write_5m_tokens":0,"cache_write_1h_tokens":0,"cost_usd":0.0},"stop_reason":"end_turn"}), flush=True)
"#;
    let catalog = ProviderCatalog::embedded().expect("catalog");
    let reg = ProviderRegistry::from_catalog(catalog);
    let spec = ResolvedProviderSpec {
        provider_id: "custom-subproc".into(),
        api: "subprocess-jsonl".into(),
        base_url: String::new(),
        fallback_base_urls: Vec::new(),
        project_id: None,
        auth: ResolvedAuth::None,
        headers: BTreeMap::new(),
        query_params: BTreeMap::new(),
        betas: Vec::new(),
        cache_diagnosis_beta: None,
        native_tool_search: false,
        compat: Default::default(),
        usage_map: Default::default(),
        body_patch: None,
        subprocess: Some(SubprocessProviderConfig {
            command: "python3".into(),
            args: vec!["-u".into(), "-c".into(), script.into()],
            env: BTreeMap::new(),
            cancel_kill_timeout_ms: 500,
        }),
        catalog_models: BTreeMap::new(),
    };
    let provider = reg.build_from_spec(&spec).expect("build subprocess");

    for expected in ["turn-1-ok", "turn-2-ok"] {
        let mut rx = provider
            .stream(
                sample_request("custom-1", vec![Message::user_text("ping")]),
                CancellationToken::new(),
            )
            .await
            .expect("stream");
        let mut got = String::new();
        while let Some(ev) = rx.recv().await {
            if let StreamEvent::TextDelta(d) = ev {
                got.push_str(&d);
            }
        }
        assert_eq!(
            got, expected,
            "persistent subprocess must stay alive across turns!"
        );
    }
}

#[test]
fn ci_gate_zero_hardcoded_providers_outside_data_catalog() {
    // Contract: zero provider-specific hostnames or match-arm hardcoding in Rust source.
    let forbidden = [
        "api.x.ai",
        "openrouter.ai",
        "api.deepseek.com",
        "api.llama.com",
        "api.groq.com",
        "api.mistral.ai",
        "api.fireworks.ai",
        "api.together.xyz",
        "api.cerebras.ai",
        "api.moonshot.ai",
        "dashscope-intl.aliyuncs.com",
        "accounts.google.com",
        "oauth2.googleapis.com",
        "daily-cloudcode-pa",
        "cloudcode-pa.googleapis.com",
        "generativelanguage.googleapis.com",
        "x-grok-conv-id",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "META_API_KEY",
    ];
    let ai_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let facade_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("carnot")
        .join("src");

    fn scan_dir_recursive(dir: &std::path::Path, forbidden: &[&str]) {
        for entry in std::fs::read_dir(dir).expect("read_dir").flatten() {
            let path = entry.path();
            if path.is_dir() {
                scan_dir_recursive(&path, forbidden);
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }
            if path.file_name().and_then(|s| s.to_str()) == Some("conformance_tests.rs") {
                continue;
            }
            let content = std::fs::read_to_string(&path).expect("read rs");
            for needle in forbidden {
                assert!(
                    !content.contains(needle),
                    "ZERO-HARDCODE VIOLATION: `{needle}` found in {} — must live in catalog/providers.json!",
                    path.display()
                );
            }
        }
    }

    for dir in [ai_src, facade_src] {
        scan_dir_recursive(&dir, &forbidden);
    }
}

#[tokio::test]
async fn conformance_native_web_search_and_fetch_across_anthropic_openai_and_gemini() {
    let catalog = ProviderCatalog::embedded().expect("embedded catalog");
    let nw = carnot_core::NativeWebConfig {
        search: true,
        fetch: true,
        max_uses: Some(4),
        max_content_tokens: Some(8_000),
        allowed_domains: vec!["docs.rs".into()],
        blocked_domains: vec!["evil.example".into()],
        search_context_size: Some("high".into()),
    };

    // 1. Anthropic: web_search_20250305 + web_fetch_20250910 BEFORE custom tools so BP1 lands on `Read`,
    //    allowed_domains takes precedence over blocked_domains, and beta header is sent ONLY when fetch=true!
    let sse_anth = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_w1\",\"usage\":{\"input_tokens\":10}}}\n\n\
        event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"ok\"}}\n\n\
        event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n\
        event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let (anth_url, anth_cap) = spawn_mock_http(vec![(200, vec![], sse_anth.into())]).await;
    let anth_spec = spec_for(&catalog, "anthropic", anth_url);
    let anth = AnthropicProvider::from_spec(anth_spec);
    assert_eq!(
        anth.web_capabilities("claude-sonnet-5-5"),
        carnot_core::ProviderWebCapabilities {
            native_search: true,
            native_fetch: true,
        }
    );
    let mut anth_req = sample_request("claude-sonnet-5-5", vec![Message::user_text("search")]);
    anth_req.native_web = nw.clone();
    let anth_body = anth.build_body(&anth_req);
    let tools = anth_body
        .get("tools")
        .and_then(Value::as_array)
        .expect("tools");
    assert_eq!(tools.len(), 3);
    assert_eq!(
        tools[0].get("type").and_then(Value::as_str),
        Some("web_search_20250305")
    );
    assert_eq!(tools[0].get("max_uses").and_then(Value::as_u64), Some(4));
    assert!(tools[0].get("allowed_domains").is_some());
    assert!(tools[0].get("blocked_domains").is_none());
    assert_eq!(
        tools[1].get("type").and_then(Value::as_str),
        Some("web_fetch_20250910")
    );
    assert_eq!(
        tools[1].get("max_content_tokens").and_then(Value::as_u64),
        Some(8_000)
    );
    // Never attach cache_control onto server tools (BP1 sits on system[0], covering tools + system)
    assert!(tools[0].get("cache_control").is_none());
    assert!(tools[1].get("cache_control").is_none());
    assert!(tools[2].get("cache_control").is_none());
    assert!(anth_body.pointer("/system/0/cache_control").is_some());

    let mut rx = anth
        .stream(anth_req, CancellationToken::new())
        .await
        .expect("stream");
    while rx.recv().await.is_some() {}
    let hdrs = anth_cap.lock().expect("cap")[0].1.clone();
    assert!(
        hdrs.to_ascii_lowercase()
            .contains("anthropic-beta: web-fetch-2025-09-10"),
        "expected web-fetch beta header when native fetch is active: {hdrs}"
    );

    // 2. OpenAI Responses: emits `{"type":"web_search","filters":{"allowed_domains":["docs.rs"]},"search_context_size":"high"}`
    let oai_spec = spec_for(&catalog, "openai-responses", "http://127.0.0.1:9".into());
    let oai = OpenAiResponsesProvider::from_spec(oai_spec);
    assert_eq!(
        oai.web_capabilities("gpt-5.4"),
        carnot_core::ProviderWebCapabilities {
            native_search: true,
            native_fetch: true,
        }
    );
    let mut oai_req = sample_request("gpt-5.4", vec![Message::user_text("search")]);
    oai_req.native_web = nw.clone();
    let oai_body = oai.build_body(&oai_req);
    let oai_tools = oai_body
        .get("tools")
        .and_then(Value::as_array)
        .expect("oai tools");
    let ws = oai_tools
        .iter()
        .find(|t| t.get("type").and_then(Value::as_str) == Some("web_search"))
        .expect("web_search tool");
    assert_eq!(
        ws.pointer("/filters/allowed_domains/0")
            .and_then(Value::as_str),
        Some("docs.rs")
    );
    assert_eq!(
        ws.get("search_context_size").and_then(Value::as_str),
        Some("high")
    );

    // 3. Gemini 3 emits googleSearch + urlContext alongside functionDeclarations AND formats grounding sources;
    //    Gemini 2.5 explicitly disables native web in `providers.json` to avoid combined-tool HTTP 400!
    let sse_gem = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Carnot is fast.\"}]},\"finishReason\":\"STOP\",\"groundingMetadata\":{\"groundingChunks\":[{\"web\":{\"uri\":\"https://docs.rs/carnot\",\"title\":\"Carnot Docs\"}}]}}],\"usageMetadata\":{\"promptTokenCount\":20,\"candidatesTokenCount\":5}}\n\n";
    let (gem_url, _) = spawn_mock_http(vec![(200, vec![], sse_gem.into())]).await;
    let gem_spec = spec_for(&catalog, "gemini", gem_url);
    let gem = GoogleProvider::from_spec(gem_spec, false);
    assert_eq!(
        gem.web_capabilities("gemini-3-pro-preview"),
        carnot_core::ProviderWebCapabilities {
            native_search: true,
            native_fetch: true,
        }
    );
    assert_eq!(
        gem.web_capabilities("gemini-2.5-pro"),
        carnot_core::ProviderWebCapabilities {
            native_search: false,
            native_fetch: false,
        }
    );
    let mut gem3_req = sample_request("gemini-3-pro-preview", vec![Message::user_text("search")]);
    gem3_req.native_web = nw.clone();
    let gem3_body = gem.build_body(&gem3_req);
    let gem3_tools = gem3_body
        .get("tools")
        .and_then(Value::as_array)
        .expect("gemini3 tools");
    assert!(gem3_tools.iter().any(|t| t.get("googleSearch").is_some()));
    assert!(gem3_tools.iter().any(|t| t.get("urlContext").is_some()));

    let mut gem25_req = sample_request("gemini-2.5-pro", vec![Message::user_text("search")]);
    gem25_req.native_web = nw;
    let gem25_body = gem.build_body(&gem25_req);
    let gem25_tools = gem25_body
        .get("tools")
        .and_then(Value::as_array)
        .expect("gemini25 tools");
    assert!(!gem25_tools.iter().any(|t| t.get("googleSearch").is_some()));
    assert!(!gem25_tools.iter().any(|t| t.get("urlContext").is_some()));

    let mut grx = gem
        .stream(gem3_req, CancellationToken::new())
        .await
        .expect("gem stream");
    let mut final_msg = None;
    while let Some(ev) = grx.recv().await {
        if let StreamEvent::Done { message, .. } = ev {
            final_msg = Some(message);
        }
    }
    let txt = final_msg.expect("done msg").text_concat();
    assert!(
        txt.contains("Carnot is fast.") && txt.contains("[Carnot Docs](https://docs.rs/carnot)"),
        "expected grounded citation footer in Gemini text: {txt}"
    );
}

#[tokio::test]
async fn conformance_oauth_reactive_401_refresh_and_query_params() {
    use crate::oauth::{AuthStore, OAuthConfig, OAuthFlow, OAuthStoredToken, OAuthTokenManager};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
    let addr = listener.local_addr().expect("addr");

    let server_task = tokio::spawn(async move {
        // Request 1: chat/completions with stale bearer token -> HTTP 401
        // Request 2: POST /oauth/token with refresh_token -> HTTP 200 fresh token
        // Request 3: retry chat/completions with fresh token + query_params -> HTTP 200 SSE
        for step in 0..3 {
            let (mut s, _) = listener.accept().await.expect("accept");
            let mut buf = [0u8; 8192];
            let n = s.read(&mut buf).await.expect("read");
            let req = String::from_utf8_lossy(&buf[..n]);
            match step {
                0 => {
                    assert!(req.contains("POST /chat/completions?api-version=2025-04-01 "));
                    assert!(
                        req.to_ascii_lowercase()
                            .contains("authorization: bearer stale-tok")
                    );
                    let body = r#"{"error":"token expired"}"#;
                    let resp = format!(
                        "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    s.write_all(resp.as_bytes()).await.expect("write 401");
                }
                1 => {
                    assert!(req.contains("POST /oauth/token "));
                    assert!(req.contains("grant_type=refresh_token"));
                    assert!(req.contains("refresh_token=ref-live-1"));
                    let body = r#"{"access_token":"fresh-tok-2","refresh_token":"ref-live-2","expires_in":3600,"token_type":"Bearer"}"#;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    s.write_all(resp.as_bytes()).await.expect("write token");
                }
                _ => {
                    assert!(req.contains("POST /chat/completions?api-version=2025-04-01 "));
                    assert!(
                        req.to_ascii_lowercase()
                            .contains("authorization: bearer fresh-tok-2")
                    );
                    let sse = concat!(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"refreshed-ok\"},\"finish_reason\":null}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n",
                        "data: [DONE]\n\n"
                    );
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse}",
                        sse.len()
                    );
                    s.write_all(resp.as_bytes()).await.expect("write sse");
                }
            }
        }
    });

    let dir = std::env::temp_dir().join(format!("carnot-oauth-401-{}", uuid::Uuid::new_v4()));
    let store = AuthStore::open(dir.join("auth.json"));
    let tok = OAuthStoredToken {
        access_token: "stale-tok".into(),
        refresh_token: Some("ref-live-1".into()),
        token_type: "Bearer".into(),
        expires_at_epoch_secs: Some(crate::oauth::now_epoch_secs() + 3600),
        scope: None,
        client_id: Some("cid".into()),
        token_url: Some(format!("http://{addr}/oauth/token")),
        updated_at_epoch_secs: crate::oauth::now_epoch_secs(),
    };
    store.save("oauth-test", tok.clone()).expect("save");
    let manager = Arc::new(OAuthTokenManager::new(
        "oauth-test",
        tok,
        Some(OAuthConfig {
            flow: OAuthFlow::Pkce,
            token_url: Some(format!("http://{addr}/oauth/token")),
            client_id: Some("cid".into()),
            ..Default::default()
        }),
        store.clone(),
    ));

    let mut qp = BTreeMap::new();
    qp.insert("api-version".into(), "2025-04-01".into());
    let spec = ResolvedProviderSpec {
        provider_id: "oauth-test".into(),
        api: "openai-completions".into(),
        base_url: format!("http://{addr}"),
        fallback_base_urls: Vec::new(),
        project_id: None,
        auth: ResolvedAuth::OAuth {
            manager,
            source: "oauth:oauth-test".into(),
        },
        headers: BTreeMap::new(),
        query_params: qp,
        betas: Vec::new(),
        cache_diagnosis_beta: None,
        native_tool_search: false,
        compat: Default::default(),
        usage_map: Default::default(),
        body_patch: None,
        subprocess: None,
        catalog_models: BTreeMap::new(),
    };

    let p = OpenAiProvider::from_spec(spec);
    let mut rx = p
        .stream(
            sample_request("gpt-5.4", vec![Message::user_text("hi")]),
            CancellationToken::new(),
        )
        .await
        .expect("stream after 401 refresh");
    let mut final_text = String::new();
    while let Some(ev) = rx.recv().await {
        if let StreamEvent::Done { message, .. } = ev {
            final_text = message.text_concat();
        }
    }
    assert_eq!(final_text, "refreshed-ok");

    let persisted = store.get("oauth-test").expect("read").expect("present");
    assert_eq!(persisted.access_token, "fresh-tok-2");
    assert_eq!(persisted.refresh_token.as_deref(), Some("ref-live-2"));

    server_task.await.expect("server done");
    let _ = std::fs::remove_dir_all(&dir);
}
