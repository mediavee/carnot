use super::*;
use carnot_ai::faux::{FauxProvider, ScriptedTurn};
use carnot_cas::CasStore;
use carnot_core::{CarnotConfig, EventEnvelope};
use carnot_tools::ToolRegistry;

#[test]
fn resolve_confined_session_path_blocks_traversal() {
    let dir = std::env::temp_dir().join(format!("carnot-acp-test-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let valid = dir.join("sess-123.jsonl");
    std::fs::write(&valid, "{}\n").expect("write");

    assert!(resolve_confined_session_path(&dir, "sess-123").is_ok());
    assert!(resolve_confined_session_path(&dir, "../etc/passwd").is_err());
    assert!(resolve_confined_session_path(&dir, "/etc/passwd").is_err());
    assert!(resolve_confined_session_path(&dir, "sub/sess-123").is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn config_options_advertises_model_and_thought_level_in_t3_format() {
    let models = vec![
        AcpModelInfo {
            id: "claude-sonnet-5-5".into(),
            name: "Claude Sonnet 5.5".into(),
            description: Some("Anthropic".into()),
        },
        AcpModelInfo {
            id: "gpt-6.1-sol".into(),
            name: "GPT 6.1 Sol".into(),
            description: Some("OpenAI Responses".into()),
        },
    ];
    let (cfg_opts, models_state) = build_config_options(
        "claude-sonnet-5-5",
        ReasoningEffort::High,
        CacheTtl::Hybrid,
        &models,
    );
    let arr = cfg_opts.as_array().expect("array");
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0]["category"], "model");
    assert_eq!(arr[2]["id"], "cache_ttl");
    assert_eq!(arr[2]["currentValue"], "hybrid");
    assert_eq!(arr[0]["currentValue"], "claude-sonnet-5-5");
    assert_eq!(arr[1]["category"], "thought_level");
    assert_eq!(arr[1]["currentValue"], "high");
    assert_eq!(models_state["currentModelId"], "claude-sonnet-5-5");
    assert_eq!(
        models_state["availableModels"].as_array().expect("m").len(),
        2
    );
}

#[test]
fn extracts_multimodal_images_text_and_embedded_resources() {
    let params = json!({
        "prompt": [
            { "type": "text", "text": "Inspect this screenshot" },
            { "type": "image", "mimeType": "image/png", "data": "iVBORw0KGgo=" },
            { "type": "resource", "resource": { "uri": "file:///a.rs", "text": "fn main() {}" } }
        ]
    });
    let blocks = extract_acp_prompt_blocks(&params);
    assert_eq!(blocks.len(), 3);
    assert!(matches!(blocks[0], ContentBlock::Text { .. }));
    assert!(matches!(
        &blocks[1],
        ContentBlock::Image { media_type, data_base64 }
            if media_type == "image/png" && data_base64 == "iVBORw0KGgo="
    ));
    assert!(matches!(blocks[2], ContentBlock::Text { .. }));
}

#[test]
fn emits_rich_acp_tool_call_with_kind_raw_input_locations_and_output() {
    let start_env = EventEnvelope {
        seq: 10,
        event: AgentEvent::ToolStart {
            call_id: "call_1".into(),
            tool: "Edit".into(),
            eager: false,
            input: json!({ "file_path": "src/lib.rs", "start_line": 42, "end_line": 45 }),
        },
    };
    let frame = acp_update_frame("sess-1", &start_env).expect("tool_call frame");
    let upd = &frame["params"]["update"];
    assert_eq!(upd["sessionUpdate"], "tool_call");
    assert_eq!(upd["kind"], "edit");
    assert_eq!(upd["title"], "Edit src/lib.rs");
    assert_eq!(upd["rawInput"]["file_path"], "src/lib.rs");
    assert_eq!(upd["locations"][0]["path"], "src/lib.rs");
    assert_eq!(upd["locations"][0]["line"], 42);

    let end_env = EventEnvelope {
        seq: 11,
        event: AgentEvent::ToolEnd {
            call_id: "call_1".into(),
            tool: "Edit".into(),
            is_error: false,
            duration_ms: 14,
            output_preview: "Edited src/lib.rs (1 replacement(s)).".into(),
        },
    };
    let end_frame = acp_update_frame("sess-1", &end_env).expect("tool_call_update");
    let eupd = &end_frame["params"]["update"];
    assert_eq!(eupd["sessionUpdate"], "tool_call_update");
    assert_eq!(eupd["status"], "completed");
    assert_eq!(eupd["rawOutput"], "Edited src/lib.rs (1 replacement(s)).");
    assert_eq!(
        eupd["content"][0]["content"]["text"],
        "Edited src/lib.rs (1 replacement(s))."
    );
}

#[tokio::test]
async fn multi_session_routing_and_session_list_real_cwd() {
    let base = std::env::temp_dir().join(format!("carnot-acp-multi-{}", uuid::Uuid::new_v4()));
    let ws_a = base.join("ws_a");
    let ws_b = base.join("ws_b");
    let sdir = base.join("sessions");
    std::fs::create_dir_all(&ws_a).expect("ws_a");
    std::fs::create_dir_all(&ws_b).expect("ws_b");

    let mk_agent = |ws: PathBuf| {
        let mut cfg = CarnotConfig::new(ws.clone());
        cfg.session_dir = sdir.clone();
        cfg.modules.sandbox = false;
        let cas = Arc::new(CasStore::open(&ws).expect("cas"));
        let tools = ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let faux = Arc::new(FauxProvider::new(vec![ScriptedTurn::text("ok")]));
        Arc::new(Agent::new(cfg, faux, tools, Vec::new(), true).expect("agent"))
    };

    let ag_a = mk_agent(ws_a.clone());
    let ag_b = mk_agent(ws_b.clone());
    let sid_a = ag_a.session_id().await;
    let sid_b = ag_b.session_id().await;

    let mut map = HashMap::new();
    map.insert(sid_a.clone(), ag_a);
    map.insert(sid_b.clone(), ag_b);

    let (resolved_b, _) =
        resolve_session_agent(&map, &sid_a, &json!({"sessionId": sid_b})).expect("route b");
    assert_eq!(resolved_b, sid_b);
    assert!(resolve_session_agent(&map, &sid_a, &json!({"sessionId": "missing"})).is_err());

    // Verify `list_recent_sessions` reports the real persisted workspace_root (`ws_a` / `ws_b`),
    // NEVER `~/.carnot/sessions`!
    let listed = list_recent_sessions(&sdir, &ws_a, 10);
    assert_eq!(listed.len(), 2);
    let cwds: Vec<&str> = listed
        .iter()
        .filter_map(|v| v.get("cwd").and_then(Value::as_str))
        .collect();
    assert!(cwds.contains(&ws_a.display().to_string().as_str()));
    assert!(cwds.contains(&ws_b.display().to_string().as_str()));

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn resolve_session_agent_isolates_multiple_sessions_and_rejects_unknown() {
    let dir1 = std::env::temp_dir().join(format!("carnot-acp-s1-{}", uuid::Uuid::new_v4()));
    let dir2 = std::env::temp_dir().join(format!("carnot-acp-s2-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir1).expect("mkdir 1");
    std::fs::create_dir_all(&dir2).expect("mkdir 2");

    let mk = |dir: &std::path::Path| -> Arc<Agent> {
        let mut cfg = CarnotConfig::new(dir.to_path_buf());
        cfg.modules.sandbox = false;
        let cas = Arc::new(CasStore::open(dir).expect("cas"));
        let tools = ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let faux = Arc::new(FauxProvider::new(vec![ScriptedTurn::text("ok")]));
        Arc::new(Agent::new(cfg, faux, tools, Vec::new(), false).expect("agent"))
    };

    let a1 = mk(&dir1);
    let a2 = mk(&dir2);
    let mut sessions = HashMap::new();
    sessions.insert("sess-alpha".to_string(), Arc::clone(&a1));
    sessions.insert("sess-beta".to_string(), Arc::clone(&a2));

    let (sid1, got1) =
        resolve_session_agent(&sessions, "sess-alpha", &json!({"sessionId": "sess-beta"}))
            .expect("explicit sessionId must route to sess-beta");
    assert_eq!(sid1, "sess-beta", "resolved id must match explicit param");
    assert!(
        Arc::ptr_eq(&got1, &a2),
        "explicit sessionId must return exact agent instance"
    );

    let (sid_def, got_def) = resolve_session_agent(&sessions, "sess-alpha", &json!({}))
        .expect("missing sessionId must fall back to default");
    assert_eq!(sid_def, "sess-alpha", "fallback id must match default");
    assert!(
        Arc::ptr_eq(&got_def, &a1),
        "fallback must return default session agent"
    );

    let err = resolve_session_agent(
        &sessions,
        "sess-alpha",
        &json!({"sessionId": "nonexistent"}),
    )
    .expect_err("unknown sessionId must return JSON-RPC error");
    assert_eq!(err.0, -32004, "unknown session error code must be -32004");

    let _ = std::fs::remove_dir_all(&dir1);
    let _ = std::fs::remove_dir_all(&dir2);
}

#[test]
fn extract_acp_prompt_blocks_handles_text_images_and_ignores_malformed_entries() {
    use carnot_core::ContentBlock;

    let params = json!({
        "prompt": [
            { "type": "text", "text": "Inspect this screenshot" },
            { "type": "image", "mimeType": "image/png", "data": "iVBORw0KGgo=" },
            { "type": "image", "mimeType": "image/png", "data": "" },
            { "type": "unknown_kind", "foo": "bar" }
        ]
    });
    let blocks = extract_acp_prompt_blocks(&params);
    assert_eq!(
        blocks.len(),
        2,
        "valid text + non-empty image should be extracted while ignoring empty/unknown blocks"
    );
    assert!(
        matches!(&blocks[0], ContentBlock::Text { text } if text == "Inspect this screenshot"),
        "first block should be text"
    );
    assert!(
        matches!(
            &blocks[1],
            ContentBlock::Image { media_type, data_base64 }
                if media_type == "image/png" && data_base64 == "iVBORw0KGgo="
        ),
        "second block should be image"
    );
}

#[test]
fn map_acp_stop_reason_covers_all_terminal_reasons() {
    assert_eq!(map_acp_stop_reason(None), "end_turn", "None -> end_turn");
    assert_eq!(
        map_acp_stop_reason(Some("stopped: MaxTokens")),
        "max_tokens",
        "MaxTokens -> max_tokens"
    );
    assert_eq!(
        map_acp_stop_reason(Some("stopped: Refusal")),
        "refusal",
        "Refusal -> refusal"
    );
    assert_eq!(
        map_acp_stop_reason(Some("max_turns limit reached")),
        "max_turn_requests",
        "max_turns -> max_turn_requests"
    );
    assert_eq!(
        map_acp_stop_reason(Some("cost budget exceeded")),
        "max_turn_requests",
        "cost budget -> max_turn_requests"
    );
}

#[test]
fn resolve_confined_session_path_blocks_null_bytes_and_dot_segments() {
    let dir = std::env::temp_dir().join(format!("carnot-acp-sec-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    assert!(
        resolve_confined_session_path(&dir, "..").is_err(),
        "dotdot must be rejected"
    );
    assert!(
        resolve_confined_session_path(&dir, "a/../b").is_err(),
        "nested traversal must be rejected"
    );
    assert!(
        resolve_confined_session_path(&dir, "").is_err(),
        "empty session path must be rejected"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
