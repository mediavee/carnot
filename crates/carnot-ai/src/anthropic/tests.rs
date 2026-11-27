use super::*;
use carnot_core::{CacheTtl, ContentBlock, ReasoningEffort, ToolResultBlock, ToolSpec};
use serde_json::{Value, json};

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
        session_id: "sess-1".into(),
        messages: vec![
            Arc::new(Message::user_text("cp")),
            Arc::new(Message::system_note("receipt")),
        ],
        checkpoint_index: Some(0),
        cache_ttl: CacheTtl::Long,
        effort: ReasoningEffort::High,
        max_output_tokens: 32_768,
        skip_tail_cache_write: false,
        native_web: carnot_core::NativeWebConfig::default(),
        ephemeral_tail_notes: Vec::new(),
    };
    let body = p.build_body(&req);
    // BP1 on system[0] covers tools + static system prompt; BP2 on messages[0] covers Turn-1;
    // BP3/4 on rolling messages. Never attach cache_control onto tools directly.
    assert!(body.pointer("/tools/0/cache_control").is_none());
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
fn hybrid_4slot_ladder_promotes_turn1_to_1h_with_monotonic_ttls() {
    let p = AnthropicProvider::new("test-key".into());
    let req = ProviderRequest {
        model: "claude-sonnet-5-5".into(),
        static_system_prompt: "STATIC".into(),
        tools: vec![ToolSpec {
            name: "Read".into(),
            description: "Read file".into(),
            input_schema: json!({"type": "object"}),
            defer_loading: false,
        }],
        prefix_hash: "abc".into(),
        session_id: "sess-1".into(),
        messages: vec![
            Arc::new(Message::user_text("turn1 workspace facts + AGENTS.md")),
            Arc::new(Message::assistant_text("a0")),
            Arc::new(Message::user_text("u1_checkpoint")),
            Arc::new(Message::assistant_text("a1")),
            Arc::new(Message::user_text("u2_tail")),
        ],
        checkpoint_index: Some(2),
        cache_ttl: CacheTtl::Hybrid,
        effort: ReasoningEffort::High,
        max_output_tokens: 32_768,
        skip_tail_cache_write: false,
        native_web: carnot_core::NativeWebConfig::default(),
        ephemeral_tail_notes: Vec::new(),
    };
    let body = p.build_body(&req);
    // Exactly 4 breakpoints with strictly monotonic TTLs: 1h -> 1h -> 5m -> 5m
    assert_eq!(
        body.pointer("/system/0/cache_control/ttl")
            .and_then(Value::as_str),
        Some("1h")
    );
    assert_eq!(
        body.pointer("/messages/0/content/0/cache_control/ttl")
            .and_then(Value::as_str),
        Some("1h")
    );
    assert_eq!(
        body.pointer("/messages/2/content/0/cache_control/ttl")
            .and_then(Value::as_str),
        Some("5m")
    );
    assert_eq!(
        body.pointer("/messages/4/content/0/cache_control/ttl")
            .and_then(Value::as_str),
        Some("5m")
    );
    // Count total cache_control blocks across entire payload (Anthropic hard max = 4)
    let s = serde_json::to_string(&body).expect("json");
    assert_eq!(s.matches("\"cache_control\"").count(), 4);
}

#[test]
fn skip_tail_cache_write_keeps_warm_prefix_without_writing_dead_tail() {
    let p = AnthropicProvider::new("test-key".into());
    let req = ProviderRequest {
        model: "claude-sonnet-5-5".into(),
        static_system_prompt: "STATIC".into(),
        tools: vec![],
        prefix_hash: "abc".into(),
        session_id: "sess-1".into(),
        messages: vec![
            Arc::new(Message::user_text("u0")),
            Arc::new(Message::assistant_text("a0")),
            Arc::new(Message::user_text("u1_warm")),
            Arc::new(Message::assistant_text("a1_new")),
            Arc::new(Message::user_text("u2_compact_prompt")),
        ],
        checkpoint_index: Some(0),
        cache_ttl: CacheTtl::Hybrid,
        effort: ReasoningEffort::Low,
        max_output_tokens: 64_000,
        skip_tail_cache_write: true,
        native_web: carnot_core::NativeWebConfig::default(),
        ephemeral_tail_notes: Vec::new(),
    };
    let body = p.build_body(&req);
    assert!(
        body.pointer("/messages/0/content/0/cache_control")
            .is_some()
    );
    assert!(
        body.pointer("/messages/2/content/0/cache_control")
            .is_some()
    );
    assert!(
        body.pointer("/messages/4/content/0/cache_control")
            .is_none()
    );
}

#[test]
fn exhaustive_cache_control_invariants_never_violate_anthropic_rules() {
    let p = AnthropicProvider::new("test-key".into());
    let tool_sets: Vec<(Vec<ToolSpec>, carnot_core::NativeWebConfig)> = vec![
        (vec![], carnot_core::NativeWebConfig::default()),
        (
            vec![
                ToolSpec {
                    name: "Read".into(),
                    description: "Read".into(),
                    input_schema: json!({"type": "object"}),
                    defer_loading: false,
                },
                ToolSpec {
                    name: "mcp__srv__tool".into(),
                    description: "Deferred".into(),
                    input_schema: json!({"type": "object"}),
                    defer_loading: true,
                },
            ],
            carnot_core::NativeWebConfig {
                search: true,
                fetch: true,
                ..Default::default()
            },
        ),
    ];

    // Diverse message shapes: plain user, thinking+tool_use assistant, tool_result user,
    // mid-conversation Role::System notes, and redacted_thinking-only assistant edge cases.
    let sample_pool: Vec<Arc<Message>> = vec![
        Arc::new(Message::user_text("turn 1 AGENTS.md + workspace facts")),
        Arc::new(Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "plan".into(),
                    signature: Some("sig1".into()),
                    origin: Some(ContentOrigin::new(
                        "anthropic-messages",
                        "anthropic",
                        "claude-sonnet-5-5",
                    )),
                },
                ContentBlock::ToolUse {
                    id: "tu_1".into(),
                    name: "Read".into(),
                    input: json!({"file_path": "a.rs"}),
                    invalid_json: None,
                    provider_meta: None,
                    origin: None,
                },
            ],
        }),
        Arc::new(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "tu_1".into(),
                content: vec![ToolResultBlock::text("fn main() {}")],
                is_error: false,
            }],
        }),
        Arc::new(Message::system_note("[worker-update untrusted]\nping")),
        Arc::new(Message::assistant_text("step 2 answer")),
        Arc::new(Message::user_text("follow-up user turn")),
        Arc::new(Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Opaque {
                    raw: json!({"type": "redacted_thinking", "data": "abc"}),
                    origin: Some(ContentOrigin::new(
                        "anthropic-messages",
                        "anthropic",
                        "claude-sonnet-5-5",
                    )),
                },
                ContentBlock::Text {
                    text: "visible reply".into(),
                },
            ],
        }),
        Arc::new(Message::user_text("final compaction prompt")),
    ];

    let mut checked_cases = 0usize;
    for (tools, native_web) in &tool_sets {
        for ttl in [CacheTtl::Short, CacheTtl::Long, CacheTtl::Hybrid] {
            for skip_tail in [false, true] {
                for len in 0..=sample_pool.len() {
                    let msgs = sample_pool[..len].to_vec();
                    let mut cps: Vec<Option<usize>> = vec![None];
                    for cp in 0..=len {
                        cps.push(Some(cp));
                    }
                    for cp in cps {
                        let req = ProviderRequest {
                            model: "claude-sonnet-5-5".into(),
                            static_system_prompt: "STATIC".into(),
                            tools: tools.clone(),
                            prefix_hash: "ph".into(),
                            session_id: "sess".into(),
                            messages: msgs.clone(),
                            checkpoint_index: cp,
                            cache_ttl: ttl,
                            effort: ReasoningEffort::High,
                            max_output_tokens: 32_768,
                            skip_tail_cache_write: skip_tail,
                            native_web: native_web.clone(),
                            ephemeral_tail_notes: Vec::new(),
                        };
                        let body = p.build_body(&req);

                        // Collect every cache_control in Anthropic wire evaluation order:
                        // tools -> system -> messages
                        let mut seen_ttls: Vec<String> = Vec::new();
                        if let Some(t_arr) = body.get("tools").and_then(Value::as_array) {
                            for t in t_arr {
                                assert!(
                                    t.get("cache_control").is_none(),
                                    "no tool (server or custom) should carry cache_control"
                                );
                            }
                        }
                        if let Some(s_arr) = body.get("system").and_then(Value::as_array) {
                            for sb in s_arr {
                                if let Some(cc) = sb.get("cache_control") {
                                    seen_ttls.push(
                                        cc.get("ttl")
                                            .and_then(Value::as_str)
                                            .unwrap_or("5m")
                                            .to_string(),
                                    );
                                }
                            }
                        }
                        if let Some(m_arr) = body.get("messages").and_then(Value::as_array) {
                            for m in m_arr {
                                if let Some(c_arr) = m.get("content").and_then(Value::as_array) {
                                    for blk in c_arr {
                                        if let Some(cc) = blk.get("cache_control") {
                                            let btype = blk
                                                .get("type")
                                                .and_then(Value::as_str)
                                                .unwrap_or("");
                                            assert!(
                                                !matches!(
                                                    btype,
                                                    "thinking"
                                                        | "redacted_thinking"
                                                        | "server_tool_use"
                                                ),
                                                "cache_control must NEVER land on {btype}"
                                            );
                                            seen_ttls.push(
                                                cc.get("ttl")
                                                    .and_then(Value::as_str)
                                                    .unwrap_or("5m")
                                                    .to_string(),
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        // Invariant 1: Anthropic hard maximum of 4 cache_control blocks
                        assert!(
                            seen_ttls.len() <= 4,
                            "exceeded 4 cache_control blocks ({seen_ttls:?}) for ttl={ttl:?} skip={skip_tail} len={len} cp={cp:?}"
                        );
                        // Invariant 2: strictly non-increasing TTL order (1h NEVER after 5m!)
                        let mut saw_5m = false;
                        for t in &seen_ttls {
                            if t == "5m" {
                                saw_5m = true;
                            } else if t == "1h" {
                                assert!(
                                    !saw_5m,
                                    "INVALID Anthropic TTL order (1h after 5m): {seen_ttls:?} for ttl={ttl:?} skip={skip_tail} len={len} cp={cp:?}"
                                );
                            }
                        }
                        checked_cases += 1;
                    }
                }
            }
        }
    }
    assert!(checked_cases >= 500, "checked {checked_cases} combinations");
}

#[test]
fn effort_preserves_cache_follows_catalog_thinking_mode() {
    let p = AnthropicProvider::new("test-key".into());
    assert!(p.effort_preserves_cache("claude-opus-5-5"));
    assert!(p.effort_preserves_cache("claude-sonnet-5-5"));
    assert!(!p.effort_preserves_cache("unknown-budget-model"));
}

#[test]
fn sse_frame_finder_handles_crlf_and_lf() {
    assert_eq!(find_sse_frame_end(b"data: {}\n\nrest"), Some((8, 2)));
    assert_eq!(find_sse_frame_end(b"data: {}\r\n\r\nrest"), Some((8, 4)));
}

#[test]
fn ephemeral_tail_notes_never_capture_cache_breakpoints_on_turn1_or_multiturn() {
    let p = AnthropicProvider::new("test-key".into());
    // Turn 1: BP2 (1h) must stay on persisted user prompt, ephemeral note uncached after it!
    let req_t1 = ProviderRequest {
        model: "claude-sonnet-5-5".into(),
        static_system_prompt: "STATIC".into(),
        tools: Vec::new(),
        prefix_hash: "abc".into(),
        session_id: "sess-1".into(),
        messages: vec![Arc::new(Message::user_text("persisted turn 1 goal"))],
        checkpoint_index: Some(0),
        cache_ttl: CacheTtl::Hybrid,
        effort: ReasoningEffort::High,
        max_output_tokens: 4096,
        skip_tail_cache_write: false,
        native_web: carnot_core::NativeWebConfig::default(),
        ephemeral_tail_notes: vec!["[ephemeral] unattended turn".into()],
    };
    let b1 = p.build_body(&req_t1);
    assert!(b1.pointer("/messages/0/content/0/cache_control").is_some());
    assert_eq!(
        b1.pointer("/messages/0/content/1/text")
            .and_then(Value::as_str),
        Some("[ephemeral] unattended turn")
    );
    assert!(
        b1.pointer("/messages/0/content/1/cache_control").is_none(),
        "ephemeral tail note must NEVER receive cache_control on Turn 1"
    );

    // Multi-turn: BP4 (5m) must sit on persisted tool result / user turn, NOT ephemeral note!
    let req_t2 = ProviderRequest {
        messages: vec![
            Arc::new(Message::user_text("persisted turn 1 goal")),
            Arc::new(Message::assistant_text("step 1")),
            Arc::new(Message::user_text("persisted turn 2")),
        ],
        ..req_t1
    };
    let b2 = p.build_body(&req_t2);
    // Persisted Turn-1 block is byte-identical between Turn 1 and Turn 2!
    assert_eq!(
        b1.pointer("/messages/0/content/0"),
        b2.pointer("/messages/0/content/0")
    );
    assert!(b2.pointer("/messages/2/content/0/cache_control").is_some());
    assert!(
        b2.pointer("/messages/2/content/1/cache_control").is_none(),
        "ephemeral tail note must NEVER capture rolling BP4 on multi-turn runs"
    );
}
