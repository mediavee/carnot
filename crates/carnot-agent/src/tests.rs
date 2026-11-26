use super::*;
use crate::context::{
    current_utc_date_ymd, estimate_context_breakdown, format_utc_date_ymd, is_repetition_dominated,
};
use carnot_ai::faux::{FauxProvider, ScriptedTurn};
use carnot_cas::CasStore;
use carnot_core::{ContentBlock, Role};
use serde_json::json;
use std::fmt::Write as _;

#[tokio::test]
async fn follow_up_waits_for_turn_settle_and_runs_in_fifo_order() {
    let dir = std::env::temp_dir().join(format!("carnot-followup-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(dir.join("note.txt"), "hello").expect("write");
    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    let cas = Arc::new(CasStore::open(&dir).expect("cas"));
    let tools = ToolRegistry::with_canonical_lid_tools(&cfg, cas);
    let faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![("c1", "Read", json!({"file_path": "note.txt"}))]),
        ScriptedTurn::text("first done"),
        ScriptedTurn::text("second done"),
    ]));
    let agent = Agent::new(cfg, faux, tools, Vec::new(), false).expect("agent");

    agent.follow_up("now do follow-up task").await;
    let out = agent.prompt("start main task").await.expect("prompt");
    assert_eq!(out.final_text, "second done");
    assert_eq!(out.metrics.turns, 3);

    // Verify follow-up was NOT injected into the tool_result message (index 2),
    // and instead arrived as its own User message AFTER "first done" (index 4)!
    let msgs = agent.active_messages().await;
    assert_eq!(msgs.len(), 6);
    assert!(msgs[2].has_tool_result());
    assert!(!msgs[2].text_concat().contains("now do follow-up task"));
    assert_eq!(msgs[3].text_concat(), "first done");
    assert_eq!(msgs[4].text_concat(), "now do follow-up task");
    assert_eq!(msgs[5].text_concat(), "second done");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn repetition_guard_catches_degenerate_loops_only() {
    let loop_text = "This is a stuck repeated line in output.\n".repeat(20);
    assert!(is_repetition_dominated(&loop_text));
    let mut normal_text = String::with_capacity(30 * 64);
    for i in 0..30 {
        let _ = writeln!(
            normal_text,
            "Unique diagnostic line number {i} with distinct payload {i:#x}"
        );
    }
    assert!(!is_repetition_dominated(&normal_text));
}

#[tokio::test]
async fn context_breakdown_tracks_categories_scales_to_total_and_clears_after_compaction() {
    let dir = std::env::temp_dir().join(format!("carnot-breakdown-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(dir.join("note.txt"), "hello from tool result\n").expect("write");
    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    let cas = Arc::new(CasStore::open(&dir).expect("cas"));
    let tools = ToolRegistry::with_canonical_lid_tools(&cfg, cas);
    let faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![("c1", "Read", json!({"file_path": "note.txt"}))]),
        ScriptedTurn::text("all done"),
        ScriptedTurn::text("compacted summary"),
        ScriptedTurn::text("after compaction"),
    ]));
    let agent = Agent::new(cfg, faux, tools, Vec::new(), false).expect("agent");

    // Before the first provider turn, no provider anchor exists -> breakdown is None.
    assert!(agent.context_breakdown().await.is_none());

    agent.prompt("inspect note.txt").await.expect("prompt");
    let (total, breakdown) = agent.current_context_with_breakdown().await;
    let b = breakdown.expect("breakdown after provider turn");
    assert!(total > 0);
    assert_eq!(
        b.total(),
        total,
        "breakdown must sum to exact context total"
    );
    assert!(b.estimated);
    assert!(b.system > 0);
    assert!(b.tools > 0);
    assert!(b.messages > 0);
    assert!(b.tool_results > 0);

    // Immediately after compaction, provider anchor is invalidated -> breakdown is None.
    agent.compact_now().await.expect("compact");
    assert!(agent.context_breakdown().await.is_none());

    // On the next provider turn, breakdown resumes and matches the new exact total.
    agent.prompt("continue").await.expect("post-compact prompt");
    let (total2, breakdown2) = agent.current_context_with_breakdown().await;
    let b2 = breakdown2.expect("breakdown restored after next turn");
    assert_eq!(b2.total(), total2);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn estimate_context_breakdown_skips_deferred_tools_and_classifies_memory_markers() {
    let active_tool = ToolSpec {
        name: "Read".into(),
        description: "Read a file from disk.".into(),
        input_schema: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        defer_loading: false,
    };
    let deferred_tool = ToolSpec {
        name: "mcp__huge__tool".into(),
        description: "x".repeat(4000),
        input_schema: json!({"type": "object", "description": "y".repeat(4000)}),
        defer_loading: true,
    };
    let msg = Arc::new(Message::user_text(
        "[project-instructions:AGENTS.md]\nFollow repo rules.\n\nHello user prompt",
    ));
    let raw = estimate_context_breakdown(
        "Static system prompt text.",
        &[active_tool, deferred_tool],
        &[msg],
    );
    assert!(raw.system > 0);
    // Deferred tool with 8 KiB description must contribute 0 tokens!
    assert!(raw.tools < 100);
    assert!(raw.memory > 0);
    assert!(raw.messages > 0);
    assert_eq!(raw.tool_results, 0);
}

#[test]
fn utc_date_formatting_matches_known_gregorian_dates() {
    assert_eq!(format_utc_date_ymd(0), "1970-01-01");
    assert_eq!(format_utc_date_ymd(11_016 * 86_400), "2000-02-29");
    assert_eq!(format_utc_date_ymd(19_782 * 86_400), "2024-02-29");
    assert_eq!(format_utc_date_ymd(19_783 * 86_400), "2024-03-01");
    assert_eq!(format_utc_date_ymd(20_782 * 86_400), "2026-11-25");
}

#[tokio::test]
async fn sparse_date_note_emitted_once_preserved_across_compaction_and_absent_from_static_prefix() {
    let dir = std::env::temp_dir().join(format!("carnot-date-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    let cas = Arc::new(CasStore::open(&dir).expect("cas"));
    let tools = ToolRegistry::with_canonical_lid_tools(&cfg, cas);
    let faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::text("turn 1 ok"),
        ScriptedTurn::text("turn 2 ok"),
        ScriptedTurn::text("compacted summary"),
        ScriptedTurn::text("turn 3 ok"),
    ]));
    let agent = Agent::new(
        cfg,
        Arc::clone(&faux) as Arc<dyn Provider>,
        tools,
        Vec::new(),
        false,
    )
    .expect("agent");

    let today = current_utc_date_ymd();
    let expected_tag = format!("[date: {today} UTC]");

    agent.prompt("first question").await.expect("p1");
    agent.prompt("second question").await.expect("p2");

    let reqs = faux.recorded_requests();
    // Static system prompt must NEVER carry the calendar date (preserves BLAKE3 prefix invariance).
    assert!(
        !reqs[0].static_system_prompt.contains("[date:"),
        "static system prompt must stay date-free"
    );
    // Turn 1 user message carries `[date: YYYY-MM-DD UTC]`; Turn 2 on the same day does NOT repeat it.
    let msgs_after_2 = agent.active_messages().await;
    assert!(
        msgs_after_2[0].text_concat().starts_with(&expected_tag),
        "turn 1 should start with {expected_tag}, got: {}",
        msgs_after_2[0].text_concat()
    );
    assert!(
        !msgs_after_2[2].text_concat().contains("[date:"),
        "turn 2 on same day must not repeat date tag, got: {}",
        msgs_after_2[2].text_concat()
    );

    // Compaction carries `[date: YYYY-MM-DD UTC]` in its summary footer so post-compaction turns stay sparse.
    agent.compact_now().await.expect("compact");
    let msgs_post_compact = agent.active_messages().await;
    assert_eq!(
        last_date_tag_in_messages(&msgs_post_compact).as_deref(),
        Some(today.as_str()),
        "compaction summary must preserve today's date tag"
    );

    agent.prompt("third question").await.expect("p3");
    let msgs_after_3 = agent.active_messages().await;
    let last_user = msgs_after_3
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .expect("last user");
    assert!(
        !last_user.text_concat().contains("[date:"),
        "turn after compaction on same day must not duplicate date tag"
    );

    // Assistant text containing `[date: ...]` MUST be ignored (cannot spoof session date!).
    let spoofed_msgs = vec![
        Arc::new(Message::user_text("[date: 2020-01-01 UTC]\n\nold")),
        Arc::new(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: "[date: 2099-12-31 UTC]".into(),
            }],
        }),
    ];
    assert_eq!(
        last_date_tag_in_messages(&spoofed_msgs).as_deref(),
        Some("2020-01-01")
    );

    let _ = std::fs::remove_dir_all(&dir);
}
