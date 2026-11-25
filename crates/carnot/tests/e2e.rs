//! End-to-end integration tests verifying every Carnot architectural & security invariant:
//!   1. Eager read-only execution + disk-coherent Read (unchanged stub, no CAS spill loop on Read)
//!   2. Atomic Edit + silent Tier-0 tree-sitter syntax check + CRLF & mode 0755 preservation
//!   3. Atomic 256-bit BLAKE3 CAS File Handle spill (>4 KiB Bash output)
//!   4. Verified green-boundary Rewind-Receipt + Preserved-Thinking prefix preservation + read-tracker reset
//!   5. Strict path confinement (`Read`/`Edit`/`Write` cannot escape workspace root)
//!   6. Effect-based parallel disjoint `Edit`/`Write` wave + gate-before-eager `RewriteInput`
//!      + malformed tool JSON safe error without execution
//!   7. Hardened Bash: adaptive yield, live partial output, and background grandchild (`sleep 5 &`) no-hang
//!   8. Lossless monotone-`seq` `EventLog` replay + cross-turn prefix monotonicity

use carnot::{
    CarnotBuilder, CarnotConfig, FauxProvider, Module, Profile, Result, ScriptedTurn, StopReason,
    ToolGateDecision,
};
use carnot_core::ContentBlock;
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::fs;
use std::sync::Arc;
use std::time::Instant;

fn msgs_json(msgs: &[Arc<carnot::Message>]) -> String {
    let refs: Vec<&carnot::Message> = msgs.iter().map(Arc::as_ref).collect();
    serde_json::to_string(&refs).unwrap_or_default()
}

#[derive(Debug)]
struct DenyRmRfAndRewriteReadModule;

impl Module for DenyRmRfAndRewriteReadModule {
    fn name(&self) -> &'static str {
        "deny_rm_rf_and_rewrite"
    }

    fn before_tool_call(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> carnot_ext::BoxFuture<'_, Result<ToolGateDecision>> {
        let decision = if tool_name == "Bash"
            && input
                .get("command")
                .and_then(Value::as_str)
                .is_some_and(|c| c.contains("rm -rf /"))
        {
            ToolGateDecision::Deny {
                reason: "blocked dangerous rm -rf /".into(),
            }
        } else if tool_name == "Read"
            && input.get("file_path").and_then(Value::as_str) == Some("alias_lib.rs")
        {
            ToolGateDecision::RewriteInput(json!({"file_path": "lib.rs"}))
        } else {
            ToolGateDecision::Allow
        };
        Box::pin(async move { Ok(decision) })
    }
}

#[tokio::test]
async fn e2e_read_edit_cas_spill_prefix_monotonicity_and_permission_gate() {
    let dir = std::env::temp_dir().join(format!("carnot-e2e-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(
        dir.join("lib.rs"),
        "pub fn add(a: i32, b: i32) -> i32 {\n    a - b\n}\n",
    )
    .expect("write lib.rs");

    // Generate a big log (>6 KiB): Read returns lines directly (no CAS loop!), Bash spills to .carnot/h/
    let mut big_log = String::new();
    for i in 0..200 {
        let _ = writeln!(
            big_log,
            "line {i:03}: verbose diagnostic padding content here"
        );
    }
    fs::write(dir.join("debug.log"), &big_log).expect("write log");

    let faux = Arc::new(FauxProvider::new(vec![
        // Turn 1: Eager Read of alias_lib.rs (rewritten by gate -> lib.rs!) + Read debug.log (clean, no spill!)
        // + Bash cat debug.log (spills to .carnot/h/!)
        ScriptedTurn::tool_calls(vec![
            ("c1", "Read", json!({"file_path": "alias_lib.rs"})),
            ("c2", "Read", json!({"file_path": "debug.log"})),
            ("c2b", "Bash", json!({"command": "cat debug.log"})),
        ]),
        // Turn 2: Re-read unchanged lib.rs (returns stub!) + try blocked rm -rf / (denied by Module!)
        ScriptedTurn::tool_calls(vec![
            ("c3", "Read", json!({"file_path": "lib.rs"})),
            ("c4", "Bash", json!({"command": "rm -rf /"})),
        ]),
        // Turn 3: Fix lib.rs via atomic Edit
        ScriptedTurn::tool_calls(vec![(
            "c5",
            "Edit",
            json!({
                "file_path": "lib.rs",
                "old_string": "a - b",
                "new_string": "a + b"
            }),
        )]),
        // Turn 4: Run a verification command
        ScriptedTurn::tool_calls(vec![(
            "c6",
            "Bash",
            json!({"command": "python3 -m unittest --help > /dev/null && echo 'OK 1 test passed'"}),
        )]),
        // Turn 5: Final answer
        ScriptedTurn::text("Fixed `add` in `lib.rs` and verified."),
    ]));

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules = carnot::ModuleSwitches::from_profile(Profile::Carnot);
    cfg.modules.sandbox = false; // keep CI-portable

    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .with_module(Arc::new(DenyRmRfAndRewriteReadModule))
        .persist_session(true)
        .build()
        .expect("build agent");

    let outcome = agent.prompt("Fix add() in lib.rs").await.expect("run");
    assert!(outcome.final_text.contains("Fixed `add`"));
    assert_eq!(outcome.metrics.turns, 5);
    assert!(outcome.metrics.eager_tools_started >= 2);
    assert!(outcome.metrics.cas_handles_created >= 1);

    // Verify lib.rs was atomically edited
    let updated = fs::read_to_string(dir.join("lib.rs")).expect("read updated");
    assert!(updated.contains("a + b"));

    let reqs = faux.recorded_requests();
    assert_eq!(reqs.len(), 5);

    // Verify gate-before-eager RewriteInput worked: alias_lib.rs returned lib.rs contents!
    let req2_txt = msgs_json(&reqs[1].messages);
    assert!(req2_txt.contains("pub fn add(a: i32, b: i32)"));
    // Verify Read(debug.log) returned line 199 directly (no CAS handle loop on Read).
    assert!(req2_txt.contains("line 199: verbose diagnostic padding content here"));
    // Verify Bash(cat debug.log) spilled to `.carnot/h/`
    assert!(req2_txt.contains(".carnot/h/"));

    // Verify strict append-only prefix monotonicity across all 5 turns.
    for t in 1..reqs.len() {
        let prev = &reqs[t - 1].messages;
        let curr = &reqs[t].messages;
        assert!(curr.len() > prev.len());
        for i in 0..prev.len() {
            assert_eq!(prev[i], curr[i], "prefix mutated at turn {t} msg {i}");
        }
        assert_eq!(reqs[t].prefix_hash, reqs[0].prefix_hash);
    }

    // Verify permission gate denied `rm -rf /` & unchanged stub fired on Turn 2
    let req3_text = msgs_json(&reqs[2].messages);
    assert!(req3_text.contains("blocked dangerous rm -rf /"));
    assert!(req3_text.contains("[unchanged:"));

    // Verify EventLog recorded strictly monotone sequence numbers and rich ToolStart/ToolEnd payloads.
    let replayed = agent.event_log().replay_after(0);
    assert!(replayed.len() >= 10);
    for pair in replayed.windows(2) {
        assert_eq!(pair[1].seq, pair[0].seq + 1);
    }
    assert!(replayed.iter().any(|e| matches!(
        &e.event,
        carnot::AgentEvent::ToolStart { tool, input, .. }
            if tool == "Edit" && input.get("file_path").and_then(|v| v.as_str()) == Some("lib.rs")
    )));
    assert!(replayed.iter().any(|e| matches!(
        &e.event,
        carnot::AgentEvent::ToolEnd { tool, output_preview, .. }
            if tool == "Edit" && output_preview.contains("Edited lib.rs")
    )));

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_multimodal_prompt_blocks_forwards_images_verbatim() {
    let dir = std::env::temp_dir().join(format!("carnot-mm-e2e-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");
    let faux = Arc::new(FauxProvider::new(vec![ScriptedTurn::text(
        "I see the attached PNG image.",
    )]));
    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .persist_session(false)
        .build()
        .expect("build");

    let out = agent
        .prompt_blocks(vec![
            carnot_core::ContentBlock::Text {
                text: "Describe this UI bug".into(),
            },
            carnot_core::ContentBlock::Image {
                media_type: "image/png".into(),
                data_base64: "iVBORw0KGgo=".into(),
            },
        ])
        .await
        .expect("prompt_blocks");
    assert_eq!(out.final_text, "I see the attached PNG image.");
    let reqs = faux.recorded_requests();
    assert_eq!(reqs.len(), 1);
    let has_img = reqs[0].messages[0].content.iter().any(|b| {
        matches!(
            b,
            carnot_core::ContentBlock::Image { media_type, data_base64 }
                if media_type == "image/png" && data_base64 == "iVBORw0KGgo="
        )
    });
    assert!(
        has_img,
        "Image block must be preserved verbatim in Turn 1 user message"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_path_confinement_crlf_and_permissions() {
    let dir = std::env::temp_dir().join(format!("carnot-s2-e2e-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    // Create a CRLF file and an executable 0755 script inside workspace
    fs::write(dir.join("win.txt"), "hello\r\nworld\r\n").expect("write crlf");
    let script_path = dir.join("run.sh");
    fs::write(&script_path, "#!/bin/sh\necho v1\n").expect("write sh");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&script_path, fs::Permissions::from_mode(0o755)).expect("chmod");
    }

    let escape_target =
        std::env::temp_dir().join(format!("carnot-should-not-exist-{}", uuid::Uuid::new_v4()));

    let faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![
            (
                "e1",
                "Write",
                json!({"file_path": escape_target.to_string_lossy(), "content": "pwn"}),
            ),
            ("e2", "Read", json!({"file_path": "/etc/hosts"})),
            (
                "e3",
                "Edit",
                json!({"file_path": "win.txt", "old_string": "hello\nworld", "new_string": "hi\nworld"}),
            ),
            (
                "e4",
                "Edit",
                json!({"file_path": "run.sh", "old_string": "v1", "new_string": "v2"}),
            ),
        ]),
        ScriptedTurn::text("Done."),
    ]));

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .persist_session(false)
        .build()
        .expect("build");

    let _ = agent
        .prompt("Test security and file fidelity")
        .await
        .expect("run");

    // Escape target must not exist.
    assert!(!escape_target.exists(), "Write escaped workspace root!");
    let req2_txt = msgs_json(&faux.recorded_requests()[1].messages);
    assert!(req2_txt.contains("outside allowed workspace root"));

    // CRLF line endings preserved.
    assert_eq!(
        fs::read_to_string(dir.join("win.txt")).expect("read win"),
        "hi\r\nworld\r\n"
    );

    // Executable 0755 permissions preserved.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&script_path)
            .expect("meta")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o755, "atomic_write lost executable mode");
    }
    assert_eq!(
        fs::read_to_string(&script_path).expect("read sh"),
        "#!/bin/sh\necho v2\n"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_parallel_disjoint_edits_and_invalid_json_guard() {
    let dir = std::env::temp_dir().join(format!("carnot-i2-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(dir.join("a.rs"), "pub const A: i32 = 1;\n").expect("write a");
    fs::write(dir.join("b.rs"), "pub const B: i32 = 2;\n").expect("write b");

    let faux = Arc::new(FauxProvider::new(vec![
        // Turn 1: 3 disjoint writes in one turn (runs in 1 parallel wave!) + 1 malformed JSON call!
        ScriptedTurn {
            blocks: vec![
                ContentBlock::ToolUse {
                    id: "w1".into(),
                    name: "Edit".into(),
                    input: json!({"file_path": "a.rs", "old_string": "= 1;", "new_string": "= 10;"}),
                    invalid_json: None,
                    provider_meta: None,
                    origin: None,
                },
                ContentBlock::ToolUse {
                    id: "w2".into(),
                    name: "Edit".into(),
                    input: json!({"file_path": "b.rs", "old_string": "= 2;", "new_string": "= 20;"}),
                    invalid_json: None,
                    provider_meta: None,
                    origin: None,
                },
                ContentBlock::ToolUse {
                    id: "w3".into(),
                    name: "Write".into(),
                    input: json!({"file_path": "c.rs", "content": "pub const C: i32 = 30;\n"}),
                    invalid_json: None,
                    provider_meta: None,
                    origin: None,
                },
                ContentBlock::ToolUse {
                    id: "bad1".into(),
                    name: "Write".into(),
                    input: json!({}),
                    invalid_json: Some("unexpected EOF while parsing JSON".into()),
                    provider_meta: None,
                    origin: None,
                },
            ],
            stop_reason: StopReason::ToolUse,
            output_tokens: 120,
        },
        ScriptedTurn::text("Parallel edits applied."),
    ]));

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .persist_session(false)
        .build()
        .expect("build");

    let out = agent
        .prompt("Update a, b, c in parallel")
        .await
        .expect("run");
    assert_eq!(out.metrics.parallel_edits_executed, 3);
    assert_eq!(
        fs::read_to_string(dir.join("a.rs")).expect("read a"),
        "pub const A: i32 = 10;\n"
    );
    assert_eq!(
        fs::read_to_string(dir.join("b.rs")).expect("read b"),
        "pub const B: i32 = 20;\n"
    );
    assert_eq!(
        fs::read_to_string(dir.join("c.rs")).expect("read c"),
        "pub const C: i32 = 30;\n"
    );

    let req2_txt = msgs_json(&faux.recorded_requests()[1].messages);
    assert!(req2_txt.contains("Malformed JSON arguments"));

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_adaptive_bash_yield_live_partial_and_bg_grandchild_no_hang() {
    let dir = std::env::temp_dir().join(format!("carnot-yield-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    let faux = Arc::new(FauxProvider::new(vec![
        // 1) Backgrounded grandchild holding stdout open must not hang Bash for 5s.
        ScriptedTurn::tool_calls(vec![(
            "bg0",
            "Bash",
            json!({"command": "echo HI_FAST; sleep 5 &"}),
        )]),
        // 2) Slow command with live partial output -> yields `job-1` with STEP_1 already visible.
        ScriptedTurn::tool_calls(vec![(
            "b1",
            "Bash",
            json!({"command": "echo STEP_1; sleep 0.3; echo STEP_2"}),
        )]),
        // 3) Collect `job-1`
        ScriptedTurn::tool_calls(vec![("b2", "Bash", json!({"job": "job-1", "timeout": 5}))]),
        ScriptedTurn::text("Background job completed."),
    ]));

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    cfg.bash_yield_ms = 120;

    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .persist_session(false)
        .build()
        .expect("build");

    let t0 = Instant::now();
    let out = agent
        .prompt("Run bg grandchild and slow build")
        .await
        .expect("run");
    let elapsed = t0.elapsed();
    assert!(
        elapsed.as_secs_f64() < 3.5,
        "Bash hung waiting on backgrounded grandchild: {elapsed:?}"
    );
    assert!(out.final_text.contains("completed"));

    let reqs = faux.recorded_requests();
    let req2_txt = msgs_json(&reqs[1].messages);
    assert!(req2_txt.contains("HI_FAST"));
    let req3_txt = msgs_json(&reqs[2].messages);
    assert!(req3_txt.contains("yielded after 120ms -> job-1"));
    assert!(req3_txt.contains("STEP_1"));
    let req4_txt = msgs_json(&reqs[3].messages);
    assert!(req4_txt.contains("STEP_2"));

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_max_tokens_pairs_truncated_tool_use_so_session_stays_usable() {
    let dir = std::env::temp_dir().join(format!("carnot-n2-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    let faux = Arc::new(FauxProvider::new(vec![
        // Prompt 1: model hits MaxTokens while emitting a ToolUse -> must NOT execute it AND must pair is_error ToolResult!
        ScriptedTurn {
            blocks: vec![ContentBlock::ToolUse {
                id: "trunc-1".into(),
                name: "Write".into(),
                input: json!({"file_path": "should_not_exist.txt", "content": "nope"}),
                invalid_json: None,
                provider_meta: None,
                origin: None,
            }],
            stop_reason: StopReason::MaxTokens,
            output_tokens: 32_768,
        },
        // Prompt 2: follow-up turn MUST see paired ToolResult for `trunc-1`!
        ScriptedTurn::text("Recovered cleanly after MaxTokens."),
    ]));

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    cfg.max_tokens_continuations = 0;
    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .persist_session(false)
        .build()
        .expect("build");

    let r1 = agent.prompt("First prompt").await.expect("run 1");
    assert!(
        r1.halted_reason
            .as_deref()
            .unwrap_or("")
            .contains("MaxTokens")
    );
    assert!(!dir.join("should_not_exist.txt").exists());

    let r2 = agent.prompt("Continue").await.expect("run 2");
    assert!(r2.final_text.contains("Recovered cleanly"));

    let reqs = faux.recorded_requests();
    assert_eq!(reqs.len(), 2);
    let req2_txt = msgs_json(&reqs[1].messages);
    assert!(req2_txt.contains("\"tool_use_id\":\"trunc-1\""));
    assert!(req2_txt.contains("\"is_error\":true"));

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_max_tokens_auto_continues_and_recovers() {
    let dir = std::env::temp_dir().join(format!("carnot-maxtok-auto-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    let faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn {
            blocks: vec![ContentBlock::ToolUse {
                id: "trunc-auto-1".into(),
                name: "Write".into(),
                input: json!({"file_path": "should_not_exist.txt", "content": "partial"}),
                invalid_json: None,
                provider_meta: None,
                origin: None,
            }],
            stop_reason: StopReason::MaxTokens,
            output_tokens: 32_768,
        },
        ScriptedTurn::text("Completed cleanly after continuation."),
    ]));

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .persist_session(false)
        .build()
        .expect("build");

    let res = agent.prompt("Write output").await.expect("run");
    assert!(res.halted_reason.is_none());
    assert!(res.final_text.contains("Completed cleanly"));
    assert!(!dir.join("should_not_exist.txt").exists());
    assert_eq!(faux.recorded_requests().len(), 2);

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_multimodal_read_agents_md_and_policy_redaction() {
    let dir = std::env::temp_dir().join(format!("carnot-mm-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(dir.join("AGENTS.md"), "Always prefer zero-copy slices.").expect("agents md");
    fs::write(dir.join(".env"), "TOP_SECRET=1").expect("env");
    fs::write(
        dir.join("notes.txt"),
        "token=sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456\n",
    )
    .expect("notes");
    // Valid PNG magic header bytes
    fs::write(
        dir.join("pixel.png"),
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01",
    )
    .expect("png");

    let faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![
            ("r1", "Read", json!({"file_path": "pixel.png"})),
            ("r2", "Read", json!({"file_path": ".env"})),
            ("r3", "Read", json!({"file_path": "notes.txt"})),
        ]),
        ScriptedTurn::text("Inspected image and notes safely."),
    ]));

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .policy_guard(true)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .persist_session(false)
        .build()
        .expect("build");

    let out = agent.prompt("Inspect files").await.expect("run");
    assert!(out.final_text.contains("safely"));

    let reqs = faux.recorded_requests();
    // AGENTS.md injected in Turn-1 user prompt, NEVER in static system prompt!
    assert!(!reqs[0].static_system_prompt.contains("zero-copy slices"));
    assert!(
        reqs[0].messages[0]
            .text_concat()
            .contains("[project-instructions:AGENTS.md]")
    );

    let req2_txt = msgs_json(&reqs[1].messages);
    // Multimodal PNG returned as Image block!
    assert!(req2_txt.contains("\"type\":\"image\""));
    assert!(req2_txt.contains("\"media_type\":\"image/png\""));
    // PolicyGuard blocked `.env` and redacted `sk-ant-...`!
    assert!(req2_txt.contains("protected by PolicyGuard"));
    assert!(req2_txt.contains("[REDACTED:ANTHROPIC_KEY]"));
    assert!(!req2_txt.contains("sk-ant-api03-"));

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_subagent_shares_parent_prefix_cache_and_isolates_context() {
    let dir = std::env::temp_dir().join(format!("carnot-sub-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(dir.join("data.txt"), "magic_answer=42\n").expect("write");

    // Shared FauxProvider serves turns in order:
    //   Turn 1 (parent): calls Subagent
    //   Turn 2 (child):  calls Read(data.txt)
    //   Turn 3 (child):  returns concise answer "42"
    //   Turn 4 (parent): receives [subagent-report] and finishes!
    let faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![(
            "s1",
            "Subagent",
            json!({"task": "Read data.txt and report the magic answer"}),
        )]),
        ScriptedTurn::tool_calls(vec![("c1", "Read", json!({"file_path": "data.txt"}))]),
        ScriptedTurn::text("The magic answer is 42."),
        ScriptedTurn::text("Subagent confirmed: 42."),
    ]));

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    cfg.modules.subagent = true;
    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .persist_session(false)
        .build()
        .expect("build");

    let out = agent.prompt("Delegate lookup").await.expect("run");
    assert!(out.final_text.contains("42"));

    let reqs = faux.recorded_requests();
    assert_eq!(reqs.len(), 4);
    // Parent (req 0) and Child (req 1) share BYTE-IDENTICAL static prefix hash!
    assert_eq!(
        reqs[0].prefix_hash, reqs[1].prefix_hash,
        "child subagent MUST share parent's static prefix hash for Turn-1 cache hits!"
    );
    let parent_final_ctx = msgs_json(&reqs[3].messages);
    assert!(parent_final_ctx.contains("[subagent-report"));
    assert!(parent_final_ctx.contains("The magic answer is 42."));

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_tool_search_fallback_and_session_resume() {
    #[derive(Debug)]
    struct DeferredDemoTool;
    impl carnot::Tool for DeferredDemoTool {
        fn spec(&self) -> carnot::ToolSpec {
            carnot::ToolSpec {
                name: "mcp__crm__lookup_customer".into(),
                description: "Lookup customer record by ID".into(),
                input_schema: json!({"type":"object","required":["id"],"properties":{"id":{"type":"string"}}}),
                defer_loading: true,
            }
        }
        fn execute(
            &self,
            input: Value,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> carnot_tools::BoxFuture<'_, Result<carnot::ToolOutput>> {
            Box::pin(async move {
                let id = input.get("id").and_then(Value::as_str).unwrap_or("?");
                Ok(carnot::ToolOutput::ok_text(format!(
                    "Customer {id}: Acme Corp"
                )))
            })
        }
    }

    let dir = std::env::temp_dir().join(format!("carnot-ts-{}", uuid::Uuid::new_v4()));
    let sdir = dir.join("sessions");
    fs::create_dir_all(&dir).expect("mkdir");

    let faux1 = Arc::new(FauxProvider::new(vec![
        // FauxProvider has supports_native_tool_search() == false -> ToolSearch + CallDeferredTool auto-registered!
        ScriptedTurn::tool_calls(vec![("ts1", "ToolSearch", json!({"query": "customer"}))]),
        ScriptedTurn::tool_calls(vec![(
            "cd1",
            "CallDeferredTool",
            json!({"name": "mcp__crm__lookup_customer", "arguments": {"id": "C-99"}}),
        )]),
        ScriptedTurn::text("Customer C-99 is Acme Corp."),
    ]));

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.session_dir = sdir.clone();
    cfg.modules.sandbox = false;
    let agent1 = CarnotBuilder::new(dir.clone())
        .config(cfg.clone())
        .with_tool(Arc::new(DeferredDemoTool))
        .provider(Arc::clone(&faux1) as Arc<dyn carnot::Provider>)
        .persist_session(true)
        .build()
        .expect("build1");

    let sid = agent1.session_id().await;
    let out1 = agent1.prompt("Find customer C-99").await.expect("run1");
    assert!(out1.final_text.contains("Acme Corp"));

    // Verify prefix_hash remained 100% invariant across ToolSearch & CallDeferredTool!
    let reqs1 = faux1.recorded_requests();
    assert_eq!(reqs1.len(), 3);
    assert_eq!(reqs1[0].prefix_hash, reqs1[1].prefix_hash);
    assert_eq!(reqs1[1].prefix_hash, reqs1[2].prefix_hash);
    assert!(msgs_json(&reqs1[2].messages).contains("Customer C-99: Acme Corp"));

    // Now resume the persisted session on a fresh Agent and verify history carried over!
    let session_file = sdir.join(format!("{sid}.jsonl"));
    let faux2 = Arc::new(FauxProvider::new(vec![ScriptedTurn::text(
        "Resumed session remembers Acme Corp.",
    )]));
    let agent2 = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .resume_session(session_file)
        .provider(Arc::clone(&faux2) as Arc<dyn carnot::Provider>)
        .build()
        .expect("build2");
    let out2 = agent2.prompt("Who was that customer?").await.expect("run2");
    assert!(out2.final_text.contains("Resumed session"));
    let reqs2 = faux2.recorded_requests();
    assert!(
        msgs_json(&reqs2[0].messages).contains("Customer C-99 is Acme Corp."),
        "resumed agent MUST carry forward prior session messages!"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_subagent_inherits_policy_guard_and_cannot_bypass() {
    let dir = std::env::temp_dir().join(format!("carnot-subpol-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(dir.join(".env"), "SECRET_KEY=top-secret-value\n").expect("write");

    let faux = Arc::new(FauxProvider::new(vec![
        // Turn 1 (parent): delegates .env read to Subagent
        ScriptedTurn::tool_calls(vec![(
            "s1",
            "Subagent",
            json!({"task": "Read .env and report its contents"}),
        )]),
        // Turn 2 (child): tries to Read .env -> MUST be denied by inherited PolicyGuardModule!
        ScriptedTurn::tool_calls(vec![("c1", "Read", json!({"file_path": ".env"}))]),
        // Turn 3 (child): reports denial
        ScriptedTurn::text("Policy blocked reading .env."),
        // Turn 4 (parent): concludes
        ScriptedTurn::text("Subagent was blocked by PolicyGuard."),
    ]));

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    cfg.modules.subagent = true;
    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .policy_guard(true)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .persist_session(false)
        .build()
        .expect("build");

    let out = agent
        .prompt("Try reading .env via subagent")
        .await
        .expect("run");
    assert!(out.final_text.contains("PolicyGuard"));

    let reqs = faux.recorded_requests();
    assert_eq!(reqs.len(), 4);
    let child_ctx_after_read = msgs_json(&reqs[2].messages);
    assert!(
        child_ctx_after_read.contains("Denied:"),
        "Subagent MUST inherit PolicyGuardModule and deny .env read! got: {child_ctx_after_read}"
    );
    assert!(
        !child_ctx_after_read.contains("top-secret-value"),
        "Secret content from .env MUST NOT leak into Subagent context!"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_root_max_cost_usd_halts_cleanly() {
    let dir = std::env::temp_dir().join(format!("carnot-costcap-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(dir.join("note.txt"), "hello\n").expect("write");

    let faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![("t1", "Read", json!({"file_path": "note.txt"}))]),
        ScriptedTurn::text("Should not reach turn 2"),
    ]));

    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .max_cost_usd(0.0001)
        .persist_session(false)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .build()
        .expect("build");

    let out = agent.prompt("Read note.txt").await.expect("run");
    assert!(
        out.halted_reason
            .as_deref()
            .is_some_and(|r| r.contains("session cost budget reached")),
        "expected root max_cost_usd halt, got: {:?}",
        out.halted_reason
    );
    assert_eq!(out.metrics.turns, 1);

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_cache_warm_semantic_compaction_preserves_full_goal_and_bash_cwd() {
    let dir = std::env::temp_dir().join(format!("carnot-compact-{}", uuid::Uuid::new_v4()));
    let sub = dir.join("nested_pkg");
    fs::create_dir_all(&sub).expect("mkdir");
    fs::write(sub.join("marker.txt"), "inside-nested\n").expect("write");

    // 1,400-char user goal whose critical final constraint lies past char 1,200 (old 600-char cap would lose it!)
    let long_goal = format!(
        "Inspect nested_pkg.\n{}\nCRITICAL_FINAL_TOLERANCE=1e-12_OUTPUT=/app/result.json",
        "Specification detail line for verification.\n".repeat(30)
    );
    assert!(long_goal.len() > 1_200);

    let faux = Arc::new(FauxProvider::new(vec![
        // Turn 1: Bash with confined `cwd` inside workspace
        ScriptedTurn::tool_calls(vec![(
            "b1",
            "Bash",
            json!({"command": "cat marker.txt", "cwd": "nested_pkg"}),
        )]),
        // Turn 2: Bash attempting escape via `cwd` MUST be denied!
        ScriptedTurn::tool_calls(vec![(
            "b2",
            "Bash",
            json!({"command": "pwd", "cwd": "../../etc"}),
        )]),
        ScriptedTurn::text("Initial exploration complete."),
        // Turn 4 (manual compact_now cache-warm semantic LLM handoff!)
        ScriptedTurn::text(
            "## 1. Exact Constraints & Edge Cases\nCRITICAL_FINAL_TOLERANCE=1e-12_OUTPUT=/app/result.json\n\
             ## 2. Verified Workspace State & Measurements\nVerified nested_pkg/marker.txt contains inside-nested.\n\
             ## 6. Immediate Next Step\nFinish cleanly.",
        ),
        // Turn 5: Follow-up prompt after compaction sees BOTH anchored full goal AND semantic handoff
        ScriptedTurn::text("Continued seamlessly with all constraints intact."),
    ]));

    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .build()
        .expect("build");

    let mut ev_stream = agent.subscribe();
    let _ = agent.prompt(long_goal).await.expect("run 1");

    // Trigger manual cache-warm semantic compaction
    let (before, after, semantic) = agent.compact_now().await.expect("compact_now");
    assert!(semantic, "expected semantic LLM handoff to succeed");
    assert!(before > 0 && after > 0);

    // Run follow-up turn after compaction
    let out2 = agent.prompt("Now finish").await.expect("run 2");
    assert!(out2.final_text.contains("Continued seamlessly"));

    let reqs = faux.recorded_requests();
    assert_eq!(reqs.len(), 5);
    // Compaction call (reqs[3]) MUST share the EXACT same byte-identical static prefix hash!
    assert_eq!(reqs[0].prefix_hash, reqs[3].prefix_hash);
    // Verify Bash(cwd="nested_pkg") succeeded and Bash(cwd="../../etc") was rejected!
    let turn2_ctx = msgs_json(&reqs[1].messages);
    assert!(turn2_ctx.contains("inside-nested"));
    let turn3_ctx = msgs_json(&reqs[2].messages);
    assert!(turn3_ctx.contains("Invalid Bash cwd"));

    // Verify post-compaction context (reqs[4]) retains the FULL verbatim final constraint (>char 1200) AND semantic handoff!
    let post_compact_ctx = msgs_json(&reqs[4].messages);
    assert!(
        post_compact_ctx.contains("[carnot:anchored-objective]"),
        "missing anchored objective header"
    );
    assert!(
        post_compact_ctx.contains("CRITICAL_FINAL_TOLERANCE=1e-12_OUTPUT=/app/result.json"),
        "full user objective past char 1200 MUST survive compaction verbatim!"
    );
    assert!(
        post_compact_ctx.contains("[carnot:task-handoff]"),
        "missing semantic LLM handoff section"
    );

    // Verify CompactionApplied event was emitted in EventLog
    let mut saw_compact_ev = false;
    while let Ok(env) =
        tokio::time::timeout(std::time::Duration::from_millis(50), ev_stream.next()).await
    {
        if matches!(
            env.event,
            carnot::AgentEvent::CompactionApplied {
                semantic_llm: true,
                ..
            }
        ) {
            saw_compact_ev = true;
            break;
        }
    }
    assert!(saw_compact_ev, "expected CompactionApplied event");

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_background_subagent_spawn_list_collect_and_cost_rollup() {
    use carnot_ai::{Provider, ProviderRequest, StreamEvent};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    /// Deterministic router separating parent vs background child requests so async scheduling is 100% race-free.
    #[derive(Debug)]
    struct SplitProvider {
        parent: Arc<FauxProvider>,
        child: Arc<FauxProvider>,
    }

    impl Provider for SplitProvider {
        fn name(&self) -> &'static str {
            "split-faux"
        }
        fn stream(
            &self,
            req: ProviderRequest,
            cancel: CancellationToken,
        ) -> carnot_ai::BoxFuture<'_, Result<mpsc::Receiver<StreamEvent>>> {
            let is_child = req
                .messages
                .first()
                .is_some_and(|m| m.text_concat().contains("CHILD_TASK"));
            let target = if is_child {
                Arc::clone(&self.child)
            } else {
                Arc::clone(&self.parent)
            };
            Box::pin(async move { target.stream(req, cancel).await })
        }
    }

    let dir = std::env::temp_dir().join(format!("carnot-bgsub-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(dir.join("work.txt"), "parent_file=ok\n").expect("write");

    let parent_faux = Arc::new(FauxProvider::new(vec![
        // Turn 1: parent spawns 2 non-blocking background subagents in parallel!
        ScriptedTurn::tool_calls(vec![
            (
                "bg1",
                "Subagent",
                json!({"task": "CHILD_TASK alpha", "background": true}),
            ),
            (
                "bg2",
                "Subagent",
                json!({"task": "CHILD_TASK beta", "background": true}),
            ),
        ]),
        // Turn 2: parent keeps working (Read + list) while children run in background!
        ScriptedTurn::tool_calls(vec![
            ("r1", "Read", json!({"file_path": "work.txt"})),
            ("l1", "Subagent", json!({"list": true})),
        ]),
        // Turn 3: parent waits & collects both background subagents in parallel!
        ScriptedTurn::tool_calls(vec![
            ("w1", "Subagent", json!({"job": "sub-1", "wait": true})),
            ("w2", "Subagent", json!({"job": "sub-2", "wait": true})),
        ]),
        // Turn 4: final answer
        ScriptedTurn::text("Collected both background subagents."),
    ]));

    let mut c1 = ScriptedTurn::text("Alpha result ready.");
    c1.output_tokens = 2_000;
    let mut c2 = ScriptedTurn::text("Beta result ready.");
    c2.output_tokens = 3_000;
    let child_faux = Arc::new(FauxProvider::new(vec![c1, c2]));

    let router = Arc::new(SplitProvider {
        parent: Arc::clone(&parent_faux),
        child: Arc::clone(&child_faux),
    });

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    cfg.modules.subagent = true;
    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .provider(router)
        .persist_session(false)
        .build()
        .expect("build");

    let out = agent
        .prompt("Run parallel background work")
        .await
        .expect("run");
    assert_eq!(out.final_text, "Collected both background subagents.");

    let preqs = parent_faux.recorded_requests();
    assert_eq!(preqs.len(), 4);
    // Turn 2 received immediate non-blocking handles (`sub-1`, `sub-2`)
    let t2_ctx = msgs_json(&preqs[1].messages);
    assert!(t2_ctx.contains("sub-1") && t2_ctx.contains("sub-2"));
    // Turn 3 received `work.txt` AND `list: true` output
    let t3_ctx = msgs_json(&preqs[2].messages);
    assert!(t3_ctx.contains("parent_file=ok"));
    assert!(t3_ctx.contains("CHILD_TASK"));
    // Turn 4 received BOTH subagent reports
    let t4_ctx = msgs_json(&preqs[3].messages);
    assert!(t4_ctx.contains("Alpha result ready."));
    assert!(t4_ctx.contains("Beta result ready."));

    // Parent metrics MUST include exact single-counted rollup of both children's spend!
    // Parent alone is ~$0.001; 5,000 child output tokens add ~$0.0625 -> total ~$0.0636.
    assert!(
        out.metrics.total_usage.cost_usd >= 0.05 && out.metrics.total_usage.cost_usd < 0.09,
        "expected single-counted rolled-up child cost (~$0.064), got ${:.6}",
        out.metrics.total_usage.cost_usd
    );

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_before_compact_module_hook_adds_focus_and_can_cancel() {
    use carnot::{CompactionHookAction, CompactionHookContext, Module};
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Debug)]
    struct CompactHookModule {
        cancel_next: Arc<AtomicBool>,
    }

    impl Module for CompactHookModule {
        fn name(&self) -> &'static str {
            "compact_hook_test"
        }

        fn before_compact<'a>(
            &'a self,
            _ctx: &'a CompactionHookContext,
        ) -> carnot_ext::BoxFuture<'a, Result<CompactionHookAction>> {
            let cancel = self.cancel_next.load(Ordering::Relaxed);
            Box::pin(async move {
                if cancel {
                    Ok(CompactionHookAction::Cancel {
                        reason: "busy transaction".into(),
                    })
                } else {
                    Ok(CompactionHookAction::AddFocus(
                        "HOOK_FOCUS_PRESERVE_SQL_SCHEMA_V9".into(),
                    ))
                }
            })
        }
    }

    let dir = std::env::temp_dir().join(format!("carnot-cmphook-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");
    let cancel_flag = Arc::new(AtomicBool::new(true));

    let faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::text("initial reply"),
        ScriptedTurn::text("second reply"),
        ScriptedTurn::text(
            "## 1. Exact Constraints & Edge Cases\n- Preserved SQL schema v9 migration ordering and foreign keys.\n## 6. Immediate Next Step\n- Run integration tests.",
        ),
    ]));

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    cfg.modules.compaction = true;
    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .with_module(Arc::new(CompactHookModule {
            cancel_next: Arc::clone(&cancel_flag),
        }))
        .persist_session(false)
        .build()
        .expect("build");

    let _ = agent.prompt("Initial prompt").await.expect("p1");
    let _ = agent.prompt("Second prompt").await.expect("p2");

    // 1. When hook returns `Cancel`, manual compaction is vetoed without calling the provider!
    let (b1, a1, used1) = agent.compact_now().await.expect("vetoed compact");
    assert_eq!(b1, a1);
    assert!(!used1);
    assert_eq!(faux.recorded_requests().len(), 2);

    // 2. When hook returns `AddFocus`, its instruction is injected into the semantic compaction request!
    cancel_flag.store(false, Ordering::Relaxed);
    let (_, _, used2) = agent.compact_now().await.expect("focused compact");
    assert!(used2);
    let reqs = faux.recorded_requests();
    assert_eq!(reqs.len(), 3);
    let compact_req_txt = msgs_json(&reqs[2].messages);
    assert!(
        compact_req_txt.contains("HOOK_FOCUS_PRESERVE_SQL_SCHEMA_V9"),
        "expected hook focus in compaction prompt, got: {compact_req_txt}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_deferred_directory_turn1_and_select_tool_search() {
    use carnot::{Tool, ToolOutput, ToolSpec};

    #[derive(Debug)]
    struct DeferredPingTool;

    impl Tool for DeferredPingTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "mcp__t3-code__preview_status".into(),
                description: "Inspect preview tab state".into(),
                input_schema: json!({"type": "object", "properties": {}}),
                defer_loading: true,
            }
        }

        fn execute(
            &self,
            _input: serde_json::Value,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> carnot_tools::BoxFuture<'_, carnot::Result<ToolOutput>> {
            Box::pin(async { Ok(ToolOutput::ok_text("preview-ready")) })
        }
    }

    let dir = std::env::temp_dir().join(format!("carnot-defdir-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    let faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![(
            "ts1",
            "ToolSearch",
            json!({"query": "select:mcp__t3-code__preview_status"}),
        )]),
        ScriptedTurn::tool_calls(vec![(
            "cd1",
            "CallDeferredTool",
            json!({"name": "mcp__t3-code__preview_status", "arguments": {}}),
        )]),
        ScriptedTurn::text("Preview checked."),
    ]));

    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .with_tool(Arc::new(DeferredPingTool))
        .build()
        .expect("build");

    let mut descs = std::collections::BTreeMap::new();
    descs.insert("t3-code".into(), "T3 Code workspace & preview tools".into());
    agent.set_deferred_server_descriptions(descs);

    let out = agent.prompt("Check preview").await.expect("run");
    assert_eq!(out.final_text, "Preview checked.");

    let reqs = faux.recorded_requests();
    assert_eq!(reqs.len(), 3);
    // Zero leak into static system prompt!
    assert!(!reqs[0].static_system_prompt.contains("t3-code"));
    assert!(
        !reqs[0]
            .static_system_prompt
            .contains("available-deferred-tools")
    );
    // Turn-1 user message advertises the compact MCP server summary!
    let turn1_txt = msgs_json(&reqs[0].messages);
    assert!(
        turn1_txt.contains("[available-deferred-tools]")
            && turn1_txt.contains("MCP `t3-code` (1 tool): T3 Code workspace & preview tools"),
        "expected Turn-1 deferred directory, got: {turn1_txt}"
    );
    // ToolSearch select: + CallDeferredTool succeeded!
    let turn3_txt = msgs_json(&reqs[2].messages);
    assert!(turn3_txt.contains("preview-ready"));

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(all(feature = "ext-web", feature = "ext-lsp"))]
#[tokio::test]
async fn e2e_official_rust_extensions_preserve_static_prefix_invariance() {
    let dir = std::env::temp_dir().join(format!("carnot-ext-inv-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    let base_faux = Arc::new(FauxProvider::new(vec![ScriptedTurn::text("ok")]));
    let base_agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .provider(Arc::clone(&base_faux) as Arc<dyn carnot::Provider>)
        .build()
        .expect("build base");
    base_agent.prompt("ping").await.expect("base prompt");

    let ext_faux = Arc::new(FauxProvider::new(vec![ScriptedTurn::text("ok")]));
    let ext_agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .web_fetch(true)
        .lsp_diagnostics(true)
        .provider(Arc::clone(&ext_faux) as Arc<dyn carnot::Provider>)
        .build()
        .expect("build with ext-web + ext-lsp");
    ext_agent.prompt("ping").await.expect("ext prompt");

    let base_req = &base_faux.recorded_requests()[0];
    let ext_reqs = ext_faux.recorded_requests();
    assert_eq!(ext_reqs.len(), 1);

    // Static system prompt stays byte-identical even with official extensions enabled!
    assert_eq!(
        base_req.static_system_prompt, ext_reqs[0].static_system_prompt,
        "official extensions must not mutate the static system prompt"
    );

    // On non-native providers with a small non-MCP extension set (<=6), tools (`WebFetch`,
    // `WebSearch`, `LspQuery`) are promoted directly for 1-step calls without `ToolSearch` /
    // `CallDeferredTool` overhead, while remaining deferred natively on Anthropic.
    let eager_names: Vec<&str> = ext_reqs[0]
        .tools
        .iter()
        .filter(|t| !t.defer_loading)
        .map(|t| t.name.as_str())
        .collect();
    assert!(eager_names.contains(&"WebFetch"));
    assert!(eager_names.contains(&"WebSearch"));
    assert!(eager_names.contains(&"LspQuery"));
    assert!(!eager_names.contains(&"ToolSearch"));
    assert!(!eager_names.contains(&"CallDeferredTool"));

    // Now test `WebMode::Auto` with a native-web-capable provider AND server `PauseTurn` continuations:
    //   1. Native search & fetch flags are enabled on `ProviderRequest::native_web`
    //   2. Zero duplicate local `WebFetch`/`WebSearch` tools are registered
    //   3. Four consecutive `StopReason::PauseTurn` turns do NOT trip `idle_turn_circuit_breaker` (= 3)!
    let native_faux = Arc::new(
        FauxProvider::new(vec![
            ScriptedTurn {
                blocks: vec![ContentBlock::Opaque {
                    raw: json!({"type": "server_tool_use", "id": "srv_1", "name": "web_search"}),
                    origin: None,
                }],
                stop_reason: StopReason::PauseTurn,
                output_tokens: 20,
            },
            ScriptedTurn {
                blocks: vec![ContentBlock::Opaque {
                    raw: json!({"type": "server_tool_use", "id": "srv_2", "name": "web_fetch"}),
                    origin: None,
                }],
                stop_reason: StopReason::PauseTurn,
                output_tokens: 20,
            },
            ScriptedTurn {
                blocks: vec![ContentBlock::Opaque {
                    raw: json!({"type": "server_tool_use", "id": "srv_3", "name": "web_search"}),
                    origin: None,
                }],
                stop_reason: StopReason::PauseTurn,
                output_tokens: 20,
            },
            ScriptedTurn {
                blocks: vec![ContentBlock::Opaque {
                    raw: json!({"type": "server_tool_use", "id": "srv_4", "name": "web_fetch"}),
                    origin: None,
                }],
                stop_reason: StopReason::PauseTurn,
                output_tokens: 20,
            },
            ScriptedTurn::text("Synthesized after 4 native web pauses."),
        ])
        .with_web_capabilities(carnot::ProviderWebCapabilities {
            native_search: true,
            native_fetch: true,
        }),
    );

    let native_agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .web_mode(carnot::WebMode::Auto)
        .provider(Arc::clone(&native_faux) as Arc<dyn carnot::Provider>)
        .build()
        .expect("build auto native web");
    let out = native_agent.prompt("Research docs").await.expect("run");
    assert_eq!(out.final_text, "Synthesized after 4 native web pauses.");
    assert!(
        out.halted_reason.is_none(),
        "PauseTurn must not trip idle breaker"
    );

    let nreqs = native_faux.recorded_requests();
    assert_eq!(nreqs.len(), 5);
    assert!(nreqs[0].native_web.search && nreqs[0].native_web.fetch);
    assert!(
        !nreqs[0]
            .tools
            .iter()
            .any(|t| matches!(t.name.as_str(), "WebFetch" | "WebSearch")),
        "auto mode must not register duplicate local web tools when provider handles both natively"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(all(
    feature = "ext-memory",
    feature = "ext-hooks",
    feature = "ext-claude-plugins"
))]
#[tokio::test]
async fn e2e_memory_and_claude_plugin_preserve_static_prefix_and_enforce_hooks() {
    let dir = std::env::temp_dir().join(format!("carnot-ext-suite-{}", uuid::Uuid::new_v4()));
    let mem_root = std::env::temp_dir().join(format!("carnot-ext-mem-{}", uuid::Uuid::new_v4()));
    let plug_dir = std::env::temp_dir().join(format!("carnot-ext-plug-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("ws");
    fs::create_dir_all(plug_dir.join(".claude-plugin")).expect("plug meta");
    fs::create_dir_all(plug_dir.join("hooks")).expect("plug hooks");
    fs::create_dir_all(plug_dir.join("skills/release")).expect("plug skills");

    fs::write(
        plug_dir.join(".claude-plugin/plugin.json"),
        json!({"name": "cc-demo", "version": "1.0.0"}).to_string(),
    )
    .expect("plugin.json");
    fs::write(
        plug_dir.join("skills/release/SKILL.md"),
        "---\nname: release\ndescription: Release workflow checklist\n---\nCheck changelog.\n",
    )
    .expect("skill");
    fs::write(
        plug_dir.join("hooks/hooks.json"),
        json!({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "Bash",
                    "hooks": [{
                        "type": "command",
                        "command": "python3 -c 'import json,sys; d=json.load(sys.stdin); (sys.stderr.write(\"cc-hook-denied\"), sys.exit(2)) if \"forbidden\" in d[\"tool_input\"].get(\"command\",\"\") else sys.exit(0)'"
                    }]
                }]
            }
        })
        .to_string(),
    )
    .expect("hooks.json");

    // Session 1: Save a memory via deferred CallDeferredTool + try forbidden Bash (blocked by CC plugin hook!)
    let faux1 = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![
            (
                "m1",
                "CallDeferredTool",
                json!({
                    "name": "MemorySave",
                    "arguments": {
                        "id": "deploy-rule",
                        "kind": "pitfall",
                        "title": "Always run clippy all-features",
                        "content": "Always verify --all-features before shipping.",
                        "pinned": true
                    }
                }),
            ),
            ("b1", "Bash", json!({"command": "echo forbidden"})),
        ]),
        ScriptedTurn::text("Saved memory and respected plugin hook."),
    ]));

    let agent1 = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .memory_config(carnot::MemoryConfig {
            root_dir: mem_root.clone(),
            max_turn1_chars: 1800,
            max_entries_per_scope: 50,
        })
        .claude_plugin(plug_dir.clone())
        .provider(Arc::clone(&faux1) as Arc<dyn carnot::Provider>)
        .build()
        .expect("build session 1");

    let out1 = agent1
        .prompt("Save rule and run check")
        .await
        .expect("run 1");
    assert_eq!(out1.final_text, "Saved memory and respected plugin hook.");
    let reqs1 = faux1.recorded_requests();
    let turn2_txt = msgs_json(&reqs1[1].messages);
    assert!(
        turn2_txt.contains("Saved memory `deploy-rule`"),
        "{turn2_txt}"
    );
    assert!(turn2_txt.contains("cc-hook-denied"), "{turn2_txt}");

    // Session 2: Fresh agent in same repo -> receives persisted memory + CC plugin skill in Turn-1 ONLY,
    //            while static_system_prompt stays 100% byte-identical to Session 1!
    let faux2 = Arc::new(FauxProvider::new(vec![ScriptedTurn::text("Got it.")]));
    let agent2 = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .memory_config(carnot::MemoryConfig {
            root_dir: mem_root.clone(),
            max_turn1_chars: 1800,
            max_entries_per_scope: 50,
        })
        .claude_plugin(plug_dir.clone())
        .provider(Arc::clone(&faux2) as Arc<dyn carnot::Provider>)
        .build()
        .expect("build session 2");

    agent2.prompt("Next session").await.expect("run 2");
    let reqs2 = faux2.recorded_requests();
    assert_eq!(reqs1[0].static_system_prompt, reqs2[0].static_system_prompt);
    assert!(!reqs2[0].static_system_prompt.contains("deploy-rule"));
    assert!(!reqs2[0].static_system_prompt.contains("cc-demo"));
    let s2_turn1 = msgs_json(&reqs2[0].messages);
    assert!(
        s2_turn1.contains("deploy-rule"),
        "expected persisted memory in Turn-1: {s2_turn1}"
    );
    assert!(
        s2_turn1.contains("[cc-demo:skill] `release`"),
        "expected CC plugin skill in Turn-1: {s2_turn1}"
    );

    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&mem_root);
    let _ = fs::remove_dir_all(&plug_dir);
}

#[tokio::test]
async fn e2e_ask_user_channel_broker_and_allow_always_permission_flow() {
    let dir = std::env::temp_dir().join(format!("carnot-askperm-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    let faux = Arc::new(FauxProvider::new(vec![
        // Turn 1: Ask the user a clarifying question via deferred `AskUser`
        ScriptedTurn::tool_calls(vec![(
            "ask1",
            "CallDeferredTool",
            json!({
                "name": "AskUser",
                "arguments": {
                    "question": "Use SQLite or Postgres?",
                    "options": ["SQLite", "Postgres"]
                }
            }),
        )]),
        // Turn 2: Run two Bash commands (first prompts for approval -> user picks allow_always -> second runs immediately!)
        ScriptedTurn::tool_calls(vec![("b1", "Bash", json!({"command": "echo first"}))]),
        ScriptedTurn::tool_calls(vec![("b2", "Bash", json!({"command": "echo second"}))]),
        ScriptedTurn::text("All done with user answers and permissions."),
    ]));

    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .ask_user(true)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .build()
        .expect("build");

    let broker = carnot::ChannelInteractionBroker::new(Arc::clone(&agent));
    agent
        .set_question_broker(Arc::clone(&broker) as Arc<dyn carnot::UserQuestionBroker>)
        .await;
    agent
        .set_permission_broker(Arc::clone(&broker) as Arc<dyn carnot::PermissionBroker>)
        .await;

    // Background responder mimicking an ACP/RPC/HTTP UI client
    let mut evs = agent.subscribe();
    let br_bg = Arc::clone(&broker);
    let perm_prompts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let perm_prompts_bg = Arc::clone(&perm_prompts);
    let responder = tokio::spawn(async move {
        loop {
            let env = evs.next().await;
            match env.event {
                carnot::AgentEvent::UserQuestionRequested {
                    request_id,
                    questions,
                } => {
                    let qid = questions[0].id.clone();
                    br_bg
                        .respond_question(
                            &request_id,
                            vec![carnot::UserQuestionAnswer {
                                id: qid,
                                selected: vec!["SQLite".into()],
                                text: None,
                            }],
                        )
                        .await;
                }
                carnot::AgentEvent::PermissionRequested { request_id, .. } => {
                    perm_prompts_bg.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    br_bg
                        .respond_permission(&request_id, "allow_always", None)
                        .await;
                }
                carnot::AgentEvent::RunSettled { .. } => break,
                _ => {}
            }
        }
    });

    let out = agent.prompt("Setup DB").await.expect("run");
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), responder).await;

    assert_eq!(
        out.final_text,
        "All done with user answers and permissions."
    );
    // `allow_always` cached after first Bash call -> only 1 permission prompt for 2 Bash calls!
    assert_eq!(perm_prompts.load(std::sync::atomic::Ordering::SeqCst), 1);

    let reqs = faux.recorded_requests();
    assert!(!reqs[0].static_system_prompt.contains("AskUser"));
    let turn2_txt = msgs_json(&reqs[1].messages);
    assert!(turn2_txt.contains("SQLite"), "{turn2_txt}");
    let turn4_txt = msgs_json(&reqs[3].messages);
    assert!(turn4_txt.contains("second"), "{turn4_txt}");

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_mid_tool_abort_persists_paired_tool_result_and_keeps_session_usable() {
    let dir = std::env::temp_dir().join(format!("carnot-abort-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    let faux = Arc::new(FauxProvider::new(vec![
        // Prompt 1: starts a slow Bash command that will be aborted mid-flight
        ScriptedTurn::tool_calls(vec![("slow1", "Bash", json!({"command": "sleep 10"}))]),
        // Prompt 2 (after abort): clean recovery turn on the EXACT same session!
        ScriptedTurn::text("Recovered cleanly after mid-tool abort."),
    ]));

    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .build()
        .expect("build");

    // Abort as soon as ToolStart fires for `slow1`
    let mut evs = agent.subscribe();
    let ag_abort = Arc::clone(&agent);
    let aborter = tokio::spawn(async move {
        loop {
            let env = evs.next().await;
            if let carnot::AgentEvent::ToolStart { call_id, .. } = env.event
                && call_id == "slow1"
            {
                tokio::time::sleep(std::time::Duration::from_millis(60)).await;
                ag_abort.abort().await;
                break;
            }
        }
    });

    let err = agent
        .prompt("Run slow task")
        .await
        .expect_err("should cancel mid-tool");
    let _ = aborter.await;
    assert!(
        matches!(err, carnot::CarnotError::Cancelled),
        "got: {err:?}"
    );

    // Now run a follow-up prompt on the same session: every prior `tool_use` MUST have a paired `tool_result`!
    let out2 = agent
        .prompt("Continue")
        .await
        .expect("second prompt must succeed");
    assert_eq!(out2.final_text, "Recovered cleanly after mid-tool abort.");

    let reqs = faux.recorded_requests();
    assert_eq!(reqs.len(), 2);
    let has_paired_result = reqs[1].messages.iter().any(|m| {
        m.content.iter().any(|b| {
            matches!(
                b,
                ContentBlock::ToolResult {
                    tool_use_id,
                    is_error: true,
                    ..
                } if tool_use_id == "slow1"
            )
        })
    });
    assert!(
        has_paired_result,
        "aborted tool_use `slow1` must have a paired is_error ToolResult before next turn"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_session_fork_and_resume_fork_preserve_original_and_prefix_hash() {
    let dir = std::env::temp_dir().join(format!("carnot-fork-sess-{}", uuid::Uuid::new_v4()));
    let sdir = dir.join(".carnot").join("sessions");
    fs::create_dir_all(&sdir).expect("mkdir");

    let faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::text("Original reply."),
        ScriptedTurn::text("Forked branch reply."),
    ]));
    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.session_dir = sdir.clone();
    cfg.modules.sandbox = false;

    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .persist_session(true)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .build()
        .expect("build");

    let orig_id = agent.session_id().await;
    let orig_prefix = agent.prefix_hash().await;
    agent.prompt("Remember ALPHA_42").await.expect("turn 1");

    let orig_path = sdir.join(format!("{orig_id}.jsonl"));
    let orig_bytes_before = fs::read(&orig_path).expect("read original session");

    // Live fork switches agent onto a brand-new JSONL file with identical prefix_hash!
    let (fork_id, fork_path) = agent.fork_session().await.expect("fork session");
    assert_ne!(fork_id, orig_id);
    assert_eq!(agent.prefix_hash().await, orig_prefix);
    let fork_path = fork_path.expect("forked session path");
    assert!(fork_path.is_file());

    agent.prompt("Branch follow-up").await.expect("turn 2");

    // Original JSONL on disk MUST remain completely untouched!
    let orig_bytes_after = fs::read(&orig_path).expect("re-read original session");
    assert_eq!(orig_bytes_before, orig_bytes_after);

    // Forked turn 2 saw ALPHA_42 from the inherited history!
    let reqs = faux.recorded_requests();
    assert_eq!(reqs.len(), 2);
    assert!(msgs_json(&reqs[1].messages).contains("ALPHA_42"));

    // Also test `--resume <path> --fork` via CarnotBuilder::resume_session().fork_session(true)
    let faux2 = Arc::new(FauxProvider::new(vec![ScriptedTurn::text("Resumed fork.")]));
    let mut cfg2 = CarnotConfig::new(dir.clone());
    cfg2.session_dir = sdir.clone();
    cfg2.modules.sandbox = false;
    let agent2 = CarnotBuilder::new(dir.clone())
        .config(cfg2)
        .resume_session(orig_path.clone())
        .fork_session(true)
        .persist_session(true)
        .provider(Arc::clone(&faux2) as Arc<dyn carnot::Provider>)
        .build()
        .expect("resume forked");
    let rf_id = agent2.session_id().await;
    assert_ne!(rf_id, orig_id);
    agent2
        .prompt("Second branch from original")
        .await
        .expect("resumed fork prompt");
    assert!(msgs_json(&faux2.recorded_requests()[0].messages).contains("ALPHA_42"));
    assert_eq!(
        fs::read(&orig_path).expect("original still untouched"),
        orig_bytes_before
    );

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_subagent_fork_vs_fresh_and_cross_provider_registry_routing() {
    let dir = std::env::temp_dir().join(format!("carnot-subfork-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    // Custom provider catalog testing engine-driven `ProviderRegistry` cross-provider routing
    // (zero hardcoded model names!)
    let custom_cat_path = dir.join("providers.json");
    fs::write(
        &custom_cat_path,
        json!({
            "schema_version": 1,
            "providers": {
                "alt-faux": {
                    "api": "faux",
                    "description": "Alternate mock provider for cross-provider subagent test",
                    "default_model": "alt-model-v2",
                    "auth_chain": [{ "scheme": "none" }]
                }
            },
            "models": {
                "alt-model-v2": {
                    "provider": "alt-faux",
                    "label": "Alt Model v2",
                    "context_window": 128_000,
                    "max_output_tokens": 8192
                }
            }
        })
        .to_string(),
    )
    .expect("write custom providers.json");

    // Verify `ProviderRegistry::available_models` discovers our custom model dynamically!
    let reg = carnot::ProviderRegistry::load_default(std::slice::from_ref(&custom_cat_path))
        .expect("load registry");
    assert!(
        reg.available_models(false)
            .iter()
            .any(|m| m.id == "alt-model-v2" && m.provider == "alt-faux")
    );

    // Parent uses shared `FauxProvider`:
    //   Turn 1: establishes anchor "SESSION_ANCHOR_99"
    //   Turn 2: parent invokes Subagent with `fork: true` -> child turn consumes next scripted turn
    //   Turn 3 (child forked): replies "Child saw fork!"
    //   Turn 4 (parent): invokes Subagent with `fork: false` -> fresh child consumes next turn
    //   Turn 5 (child fresh): replies "Child fresh!"
    //   Turn 6 (parent): invokes Subagent cross-provider `model: "alt-model-v2"` (`alt-faux`!)
    //   Turn 7 (parent): final text
    let faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::text("Noted SESSION_ANCHOR_99."),
        ScriptedTurn::tool_calls(vec![(
            "sf1",
            "Subagent",
            json!({
                "task": "Verify what anchor we discussed",
                "fork": true,
                "effort": "low"
            }),
        )]),
        ScriptedTurn::text("Child saw fork!"),
        ScriptedTurn::tool_calls(vec![(
            "sf2",
            "Subagent",
            json!({
                "task": "Fresh check",
                "fork": false
            }),
        )]),
        ScriptedTurn::text("Child fresh!"),
        ScriptedTurn::tool_calls(vec![(
            "sf3",
            "Subagent",
            json!({
                "task": "Cross-provider check",
                "model": "alt-model-v2",
                "effort": "medium"
            }),
        )]),
        ScriptedTurn::text("All subagent modes verified."),
    ]));

    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .subagent(true)
        .provider_config(custom_cat_path.clone())
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .build()
        .expect("build");

    agent
        .prompt("Remember SESSION_ANCHOR_99")
        .await
        .expect("turn 1");
    let out = agent
        .prompt("Run forked, fresh, and cross-provider subagents")
        .await
        .expect("turn 2");
    assert_eq!(out.final_text, "All subagent modes verified.");

    let reqs = faux.recorded_requests();
    // reqs[0]=parent T1, reqs[1]=parent T2, reqs[2]=forked child, reqs[3]=parent after fork,
    // reqs[4]=fresh child, reqs[5]=parent after fresh, reqs[6]=parent after cross-provider `alt-faux`
    assert_eq!(reqs.len(), 7, "expected 7 faux requests");

    // 1. Forked child (reqs[2]) MUST see parent's Turn-1 anchor AND have ZERO orphan `sf1` tool_use!
    let forked_child_msgs = msgs_json(&reqs[2].messages);
    assert!(
        forked_child_msgs.contains("SESSION_ANCHOR_99"),
        "forked subagent must inherit parent conversation: {forked_child_msgs}"
    );
    assert!(
        !forked_child_msgs.contains("\"id\":\"sf1\""),
        "forked subagent must strip parent's in-flight tool_use `sf1`"
    );
    assert_eq!(
        reqs[0].static_system_prompt, reqs[2].static_system_prompt,
        "forked subagent must share byte-identical static prefix"
    );
    assert_eq!(reqs[2].effort, carnot::ReasoningEffort::Low);

    // 2. Fresh child (reqs[4]) MUST NOT see parent's anchor!
    let fresh_child_msgs = msgs_json(&reqs[4].messages);
    assert!(
        !fresh_child_msgs.contains("SESSION_ANCHOR_99"),
        "fresh subagent must NOT inherit parent history"
    );

    // 3. Cross-provider subagent (sf3 -> `alt-faux` / `alt-model-v2`) ran on its own resolved provider!
    let after_cross_msgs = msgs_json(&reqs[6].messages);
    assert!(
        after_cross_msgs.contains("provider=alt-faux model=alt-model-v2"),
        "expected cross-provider subagent report for alt-faux/alt-model-v2: {after_cross_msgs}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_orchestration_dirty_parent_with_untracked_file_lands_worker_diff_cleanly() {
    let dir = std::env::temp_dir().join(format!("carnot-orch-land-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    // Initialize a real git repo with committed file + dirty tracked edit + untracked file!
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .current_dir(&dir)
            .args(args)
            .output()
            .expect("git");
        assert!(
            out.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    };
    git(&["init", "-q"]);
    git(&["config", "user.name", "test"]);
    git(&["config", "user.email", "test@local"]);
    git(&["config", "commit.gpgsign", "false"]);
    fs::write(dir.join("base.txt"), "line1\nline2\nline3\n").expect("write base");
    git(&["add", "base.txt"]);
    git(&["commit", "-q", "-m", "init"]);

    // Parent dirty state: modify base.txt (uncommitted) + add untracked parent_draft.txt
    fs::write(dir.join("base.txt"), "line1-parent-dirty\nline2\nline3\n").expect("dirty base");
    fs::write(dir.join("parent_draft.txt"), "PARENT_UNTRACKED_OK").expect("untracked");

    // Scripted turns on shared FauxProvider:
    //   0: Parent spawns named keep_alive isolated_worktree worker `impl` (sync)
    //   1: Worker reads `parent_draft.txt` and writes `worker_feature.txt`
    //   2: Worker completes Turn 1
    //   3: Parent calls `Subagent { job: "impl", land: true }`
    //   4: Parent sends multi-turn follow-up via `SendMessage { to: "impl", message: "Second pass" }`
    //      and waits for `impl` via `Subagent { jobs: ["impl"], wait: true, wait_mode: "all" }`
    //   5: Worker handles follow-up turn (same warm context!)
    //   6: Parent closes `impl` and returns final text
    let faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![(
            "sp1",
            "Subagent",
            json!({
                "name": "impl",
                "task": "Read parent_draft.txt and create worker_feature.txt",
                "isolated_worktree": true,
                "keep_alive": true,
                "fork": true
            }),
        )]),
        ScriptedTurn::tool_calls(vec![
            ("wr1", "Read", json!({"file_path": "parent_draft.txt"})),
            (
                "ww1",
                "Write",
                json!({"file_path": "worker_feature.txt", "content": "WORKER_FEATURE_V1\n"}),
            ),
        ]),
        ScriptedTurn::text("Created worker_feature.txt after reading parent_draft.txt."),
        ScriptedTurn::tool_calls(vec![(
            "ln1",
            "Subagent",
            json!({"job": "impl", "land": true}),
        )]),
        ScriptedTurn::tool_calls(vec![(
            "sm1",
            "SendMessage",
            json!({"to": "impl", "message": "Confirm warm follow-up"}),
        )]),
        ScriptedTurn::text("Warm follow-up confirmed!"),
        ScriptedTurn::tool_calls(vec![(
            "wt1",
            "Subagent",
            json!({"jobs": ["impl"], "wait": true, "wait_mode": "all"}),
        )]),
        ScriptedTurn::tool_calls(vec![(
            "kl1",
            "Subagent",
            json!({"job": "impl", "kill": true}),
        )]),
        ScriptedTurn::text("Landed, followed up, and cleaned up."),
    ]));

    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .subagent(true)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .build()
        .expect("build");

    let out = agent
        .prompt("Orchestrate isolated worker and land cleanly")
        .await
        .expect("run");
    assert_eq!(out.final_text, "Landed, followed up, and cleaned up.");

    // 1. Worker's file landed cleanly into parent workspace!
    assert_eq!(
        fs::read_to_string(dir.join("worker_feature.txt")).expect("landed file"),
        "WORKER_FEATURE_V1\n"
    );
    // 2. Parent's pre-existing dirty edit & untracked file were preserved untouched!
    assert_eq!(
        fs::read_to_string(dir.join("base.txt")).expect("parent dirty base"),
        "line1-parent-dirty\nline2\nline3\n"
    );
    assert_eq!(
        fs::read_to_string(dir.join("parent_draft.txt")).expect("parent untracked"),
        "PARENT_UNTRACKED_OK"
    );

    // 3. Verify byte-identical static prefix (`prefix_hash` + `static_system_prompt` + `tools`)
    //    between parent (req 0) and forked worker (req 1)!
    let reqs = faux.recorded_requests();
    assert_eq!(reqs[0].prefix_hash, reqs[1].prefix_hash);
    assert_eq!(reqs[0].static_system_prompt, reqs[1].static_system_prompt);
    assert_eq!(reqs[0].tools.len(), reqs[1].tools.len());

    // 4. Verify worker actually read `PARENT_UNTRACKED_OK` inside its isolated worktree!
    assert!(msgs_json(&reqs[2].messages).contains("PARENT_UNTRACKED_OK"));

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_orchestration_headless_waits_for_running_worker_and_delegates_permissions() {
    use carnot::{PermissionBroker, Provider, ToolGateDecision};

    #[derive(Debug)]
    struct TaggingBroker {
        seen_calls: std::sync::Mutex<Vec<(String, Option<String>)>>,
    }

    impl PermissionBroker for TaggingBroker {
        fn request_permission(
            &self,
            _call_id: &str,
            tool_name: &str,
            input: &serde_json::Value,
        ) -> carnot_ext::BoxFuture<'_, carnot::Result<ToolGateDecision>> {
            let worker_tag = input
                .get("_carnot_worker")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            self.seen_calls
                .lock()
                .expect("lock")
                .push((tool_name.to_string(), worker_tag.clone()));
            let deny = worker_tag.is_some() && tool_name == "Bash";
            Box::pin(async move {
                if deny {
                    Ok(ToolGateDecision::Deny {
                        reason: "parent-broker-denied-worker-bash".into(),
                    })
                } else {
                    Ok(ToolGateDecision::Allow)
                }
            })
        }
    }

    /// Deterministic race-free router separating parent turns from background worker turns.
    #[derive(Debug)]
    struct RoleRoutedProvider {
        parent: Arc<FauxProvider>,
        worker: Arc<FauxProvider>,
    }

    impl Provider for RoleRoutedProvider {
        fn name(&self) -> &'static str {
            "faux"
        }

        fn stream(
            &self,
            req: carnot_ai::ProviderRequest,
            cancel: tokio_util::sync::CancellationToken,
        ) -> carnot_ai::BoxFuture<
            '_,
            carnot::Result<tokio::sync::mpsc::Receiver<carnot_ai::StreamEvent>>,
        > {
            let is_worker = req.messages.first().is_some_and(|m| {
                m.content.iter().any(|b| match b {
                    ContentBlock::Text { text } => text.contains("Run security check"),
                    _ => false,
                })
            });
            let target = if is_worker {
                Arc::clone(&self.worker)
            } else {
                Arc::clone(&self.parent)
            };
            Box::pin(async move {
                if is_worker {
                    // Slight delay ensures parent hits `on_turn_boundary` auto-wait while worker is live!
                    tokio::time::sleep(std::time::Duration::from_millis(40)).await;
                }
                target.stream(req, cancel).await
            })
        }
    }

    let dir = std::env::temp_dir().join(format!("carnot-orch-wait-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(dir.join("notes.txt"), "READ_OK_77").expect("write");

    let parent_faux = Arc::new(FauxProvider::new(vec![
        // 0: Parent spawns `sec` in background
        ScriptedTurn::tool_calls(vec![(
            "bg1",
            "Subagent",
            json!({
                "name": "sec",
                "task": "Run security check",
                "background": true,
                "keep_alive": false
            }),
        )]),
        // 1: Parent emits plain text without waiting -> triggers `on_turn_boundary` auto-wait!
        ScriptedTurn::text("Waiting on sec..."),
        // 2: Parent woken reactively by `on_turn_boundary` with `SEC_WORKER_DONE_42`!
        ScriptedTurn::text("Final answer after reactive wake: SEC_WORKER_DONE_42."),
    ]));
    let worker_faux = Arc::new(FauxProvider::new(vec![
        // 0: Worker `sec` attempts:
        //    - Read (delegated with verbatim tool_name == "Read" & _carnot_worker == "sec")
        //    - Bash (delegated with verbatim tool_name == "Bash" & _carnot_worker == "sec" -> denied!)
        //    - P2P SendMessage (blocked by hub-and-spoke)
        //    - Subagent{list:true} & Subagent{wait:true} (blocked unconditionally for workers!)
        ScriptedTurn::tool_calls(vec![
            ("wr1", "Read", json!({"file_path": "notes.txt"})),
            ("wb1", "Bash", json!({"command": "echo nope"})),
            (
                "wp2p",
                "SendMessage",
                json!({"to": "other", "message": "peer ping"}),
            ),
            ("wlist", "Subagent", json!({"list": true})),
            ("wwait", "Subagent", json!({"wait": true})),
        ]),
        // 1: Worker `sec` settles with its final report
        ScriptedTurn::text("SEC_WORKER_DONE_42"),
    ]));

    let routed = Arc::new(RoleRoutedProvider {
        parent: Arc::clone(&parent_faux),
        worker: Arc::clone(&worker_faux),
    });
    let broker = Arc::new(TaggingBroker {
        seen_calls: std::sync::Mutex::new(Vec::new()),
    });
    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .subagent(true)
        .provider(routed)
        .build()
        .expect("build");
    agent
        .set_permission_broker(Arc::clone(&broker) as Arc<dyn carnot::PermissionBroker>)
        .await;

    let out = agent
        .prompt("Run background security check")
        .await
        .expect("run");
    assert_eq!(
        out.final_text,
        "Final answer after reactive wake: SEC_WORKER_DONE_42."
    );

    // 1. Verify parent's PermissionBroker received verbatim tool names (`"Bash"`, `"Read"`)
    //    with `_carnot_worker: "sec"` in input (never mangled `"[worker:sec] Bash"`).
    let seen = broker.seen_calls.lock().expect("lock").clone();
    assert!(
        seen.iter()
            .any(|(t, w)| t == "Bash" && w.as_deref() == Some("sec")),
        "expected verbatim Bash with _carnot_worker=sec: {seen:?}"
    );
    assert!(
        seen.iter()
            .any(|(t, w)| t == "Read" && w.as_deref() == Some("sec")),
        "expected verbatim Read with _carnot_worker=sec: {seen:?}"
    );
    assert!(
        !seen.iter().any(|(t, _)| t.starts_with("[worker:")),
        "tool_name must never be mangled with [worker:...]: {seen:?}"
    );

    // 2. Verify Read succeeded, Bash was denied, P2P SendMessage was denied, and ALL Subagent
    //    management actions (`list`, `wait`) were denied inside the worker!
    let worker_reqs = worker_faux.recorded_requests();
    let worker_json: String = worker_reqs.iter().map(|r| msgs_json(&r.messages)).collect();
    assert!(worker_json.contains("READ_OK_77"));
    assert!(worker_json.contains("parent-broker-denied-worker-bash"));
    assert!(worker_json.contains("hub-and-spoke"));
    assert!(worker_json.contains("workers cannot call `Subagent`"));

    // 3. Verify parent was woken reactively via untrusted `Role::System` note (never `[user-steer]`)!
    let parent_reqs = parent_faux.recorded_requests();
    assert_eq!(parent_reqs.len(), 3);
    let last_req = &parent_reqs[2];
    assert!(msgs_json(&last_req.messages).contains("SEC_WORKER_DONE_42"));
    assert!(
        last_req.messages.iter().any(|m| {
            m.role == carnot_core::Role::System
                && m.text_concat().contains("[worker-update untrusted]")
                && m.text_concat().contains("SEC_WORKER_DONE_42")
        }),
        "worker report must arrive as untrusted Role::System note"
    );
    assert!(
        !msgs_json(&last_req.messages).contains("[user-steer]"),
        "worker updates must NEVER be tagged [user-steer]"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_orchestration_second_land_and_monorepo_subdir() {
    use carnot::Provider;
    use std::process::Command;

    #[derive(Debug)]
    struct MonorepoSplit {
        parent: Arc<FauxProvider>,
        worker: Arc<FauxProvider>,
    }
    impl Provider for MonorepoSplit {
        fn name(&self) -> &'static str {
            "faux"
        }
        fn stream(
            &self,
            req: carnot_ai::ProviderRequest,
            cancel: tokio_util::sync::CancellationToken,
        ) -> carnot_ai::BoxFuture<
            '_,
            carnot::Result<tokio::sync::mpsc::Receiver<carnot_ai::StreamEvent>>,
        > {
            let is_worker = req
                .messages
                .first()
                .is_some_and(|m| m.text_concat().contains("MONOREPO_API_WORKER"));
            let target = if is_worker {
                Arc::clone(&self.worker)
            } else {
                Arc::clone(&self.parent)
            };
            Box::pin(async move { target.stream(req, cancel).await })
        }
    }

    let repo_dir = std::env::temp_dir().join(format!("carnot-monorepo-{}", uuid::Uuid::new_v4()));
    let pkg_dir = repo_dir.join("packages").join("api");
    fs::create_dir_all(&pkg_dir).expect("mkdir");

    let run_git = |args: &[&str]| {
        let out = Command::new("git")
            .current_dir(&repo_dir)
            .args(args)
            .output()
            .expect("git");
        assert!(
            out.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    };
    run_git(&["init"]);
    run_git(&["config", "user.name", "test"]);
    run_git(&["config", "user.email", "test@local"]);
    run_git(&["config", "commit.gpgsign", "false"]);
    fs::write(pkg_dir.join("base.txt"), "base-v1\n").expect("write");
    run_git(&["add", "."]);
    run_git(&["commit", "-m", "init"]);

    let parent_faux = Arc::new(FauxProvider::new(vec![
        // 0: Parent spawns `keep_alive` isolated worker `api`
        ScriptedTurn::tool_calls(vec![(
            "s1",
            "Subagent",
            json!({
                "name": "api",
                "task": "MONOREPO_API_WORKER write v1 file",
                "keep_alive": true,
                "isolated_worktree": true
            }),
        )]),
        // 1: Parent calls 1st `land`
        ScriptedTurn::tool_calls(vec![(
            "l1",
            "Subagent",
            json!({"job": "api", "land": true}),
        )]),
        // 2: Parent sends wave 2 follow-up via SendMessage
        ScriptedTurn::tool_calls(vec![(
            "m1",
            "SendMessage",
            json!({"to": "api", "message": "Add b.txt and update a.txt"}),
        )]),
        // 3: Parent waits for `api`
        ScriptedTurn::tool_calls(vec![(
            "wait2",
            "Subagent",
            json!({"job": "api", "wait": true}),
        )]),
        // 4: Parent calls 2nd `land` (MUST succeed cleanly because base_sha advanced!)
        ScriptedTurn::tool_calls(vec![(
            "l2",
            "Subagent",
            json!({"job": "api", "land": true}),
        )]),
        // 5: Parent calls 3rd `land` with no new edits -> NoChanges
        ScriptedTurn::tool_calls(vec![(
            "l3",
            "Subagent",
            json!({"job": "api", "land": true}),
        )]),
        // 6: Parent kills worker
        ScriptedTurn::tool_calls(vec![(
            "k1",
            "Subagent",
            json!({"job": "api", "kill": true}),
        )]),
        // 7: Final answer
        ScriptedTurn::text("Both monorepo lands succeeded cleanly."),
    ]));
    let worker_faux = Arc::new(FauxProvider::new(vec![
        // 0: Worker writes `a.txt` inside its `packages/api` workspace
        ScriptedTurn::tool_calls(vec![(
            "w1",
            "Write",
            json!({"file_path": "a.txt", "content": "wave1\n"}),
        )]),
        // 1: Worker finishes turn 1
        ScriptedTurn::text("Wave 1 ready."),
        // 2: Worker updates `a.txt` and creates `b.txt`
        ScriptedTurn::tool_calls(vec![
            (
                "w2",
                "Write",
                json!({"file_path": "a.txt", "content": "wave1\nwave2\n"}),
            ),
            (
                "w3",
                "Write",
                json!({"file_path": "b.txt", "content": "second-land-ok\n"}),
            ),
        ]),
        // 3: Worker finishes turn 2
        ScriptedTurn::text("Wave 2 ready."),
    ]));

    let agent = CarnotBuilder::new(pkg_dir.clone())
        .sandbox(false)
        .persist_session(false)
        .subagent(true)
        .provider(Arc::new(MonorepoSplit {
            parent: Arc::clone(&parent_faux),
            worker: Arc::clone(&worker_faux),
        }))
        .build()
        .expect("build");

    let out = agent
        .prompt("Run two-wave monorepo land")
        .await
        .expect("run");
    assert_eq!(out.final_text, "Both monorepo lands succeeded cleanly.");

    // Verify BOTH waves landed in the exact monorepo subdirectory!
    assert_eq!(
        fs::read_to_string(pkg_dir.join("a.txt")).expect("read a.txt"),
        "wave1\nwave2\n"
    );
    assert_eq!(
        fs::read_to_string(pkg_dir.join("b.txt")).expect("read b.txt"),
        "second-land-ok\n"
    );
    let all_json: String = parent_faux
        .recorded_requests()
        .iter()
        .map(|r| msgs_json(&r.messages))
        .collect();
    assert!(
        !all_json.contains("Conflict checking worker"),
        "second land must not conflict!"
    );
    assert!(all_json.contains("has no changes to land"));

    let _ = fs::remove_dir_all(&repo_dir);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_orchestration_worker_message_single_delivery_and_untrusted() {
    use carnot::Provider;

    #[derive(Debug)]
    struct Split {
        parent: Arc<FauxProvider>,
        worker: Arc<FauxProvider>,
    }
    impl Provider for Split {
        fn name(&self) -> &'static str {
            "faux"
        }
        fn stream(
            &self,
            req: carnot_ai::ProviderRequest,
            cancel: tokio_util::sync::CancellationToken,
        ) -> carnot_ai::BoxFuture<
            '_,
            carnot::Result<tokio::sync::mpsc::Receiver<carnot_ai::StreamEvent>>,
        > {
            let is_worker = req
                .messages
                .first()
                .is_some_and(|m| m.text_concat().contains("PING_COORDINATOR_TASK"));
            let target = if is_worker {
                Arc::clone(&self.worker)
            } else {
                Arc::clone(&self.parent)
            };
            Box::pin(async move {
                if is_worker {
                    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
                }
                target.stream(req, cancel).await
            })
        }
    }

    let dir = std::env::temp_dir().join(format!("carnot-single-msg-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    let parent_faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![(
            "s1",
            "Subagent",
            json!({
                "name": "pinger",
                "task": "PING_COORDINATOR_TASK",
                "background": true,
                "keep_alive": false
            }),
        )]),
        ScriptedTurn::text("Waiting on pinger..."),
        ScriptedTurn::text("Acknowledged worker ping, waiting for completion..."),
        ScriptedTurn::text("Done receiving single message."),
    ]));
    let worker_faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![(
            "m1",
            "SendMessage",
            json!({"to": "coordinator", "message": "UNIQUE_WORKER_PING_999"}),
        )]),
        ScriptedTurn::text("Pinger finished."),
    ]));

    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .subagent(true)
        .provider(Arc::new(Split {
            parent: Arc::clone(&parent_faux),
            worker: Arc::clone(&worker_faux),
        }))
        .build()
        .expect("build");

    let out = agent.prompt("Coordinate pinger").await.expect("run");
    assert_eq!(out.final_text, "Done receiving single message.");

    let final_msgs = agent.active_messages().await;
    let ping_hits: Vec<_> = final_msgs
        .iter()
        .filter(|m| m.text_concat().contains("UNIQUE_WORKER_PING_999"))
        .collect();
    assert_eq!(
        ping_hits.len(),
        1,
        "worker SendMessage must be delivered EXACTLY ONCE, got {}: {ping_hits:?}",
        ping_hits.len()
    );
    assert_eq!(
        ping_hits[0].role,
        carnot_core::Role::System,
        "worker message must have Role::System, never Role::User"
    );
    let text = ping_hits[0].text_concat();
    assert!(text.contains("[worker-update untrusted]"));
    assert!(!text.contains("[user-steer]"));

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_orchestration_fleet_drop_acyclic_and_cleans_worktree() {
    use carnot_ext_subagent::SubagentModule;
    use std::process::Command;

    let repo_dir = std::env::temp_dir().join(format!("carnot-acyclic-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&repo_dir).expect("mkdir");
    for args in [
        &["init"][..],
        &["config", "user.name", "test"],
        &["config", "user.email", "test@local"],
        &["config", "commit.gpgsign", "false"],
    ] {
        assert!(
            Command::new("git")
                .current_dir(&repo_dir)
                .args(args)
                .output()
                .expect("git")
                .status
                .success()
        );
    }
    fs::write(repo_dir.join("seed.txt"), "seed\n").expect("write");
    assert!(
        Command::new("git")
            .current_dir(&repo_dir)
            .args(["add", "."])
            .output()
            .expect("git")
            .status
            .success()
    );
    assert!(
        Command::new("git")
            .current_dir(&repo_dir)
            .args(["commit", "-m", "init"])
            .output()
            .expect("git")
            .status
            .success()
    );

    let faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![(
            "s1",
            "Subagent",
            json!({
                "name": "keeper",
                "task": "Say hi",
                "keep_alive": true,
                "isolated_worktree": true
            }),
        )]),
        ScriptedTurn::text("Worker hi."),
        ScriptedTurn::text("Parent done leaving keeper alive."),
    ]));

    let mut cfg = CarnotConfig::new(repo_dir.clone());
    cfg.modules.sandbox = false;
    let sub_mod = Arc::new(SubagentModule::new(
        cfg.clone(),
        Arc::clone(&faux) as Arc<dyn carnot::Provider>,
        Vec::new(),
    ));
    let weak_fleet = Arc::downgrade(&sub_mod.fleet().expect("fleet alive"));
    let cas = Arc::new(carnot::CasStore::open(&repo_dir).expect("cas"));
    let tools = carnot::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
    let agent = Arc::new(
        carnot::Agent::new(
            cfg,
            faux,
            tools,
            vec![Arc::clone(&sub_mod) as Arc<dyn carnot::Module>],
            false,
        )
        .expect("agent"),
    );
    sub_mod.attach_parent(&agent);

    let out = agent.prompt("Spawn keep_alive worker").await.expect("run");
    assert_eq!(out.final_text, "Parent done leaving keeper alive.");

    // Capture the live worktree path while keeper is still idle & alive
    let wt_path = {
        let fleet = weak_fleet.upgrade().expect("fleet still alive");
        let w = fleet.get("keeper").expect("keeper present");
        let guard = w.worktree.lock().expect("wt lock");
        let wt = guard.as_ref().expect("worktree active");
        assert!(wt.worktree_path.exists());
        wt.worktree_path.clone()
    };

    // Drop coordinator references -> NO Arc cycle, Fleet::drop() MUST run and delete worktree!
    drop(agent);
    drop(sub_mod);

    assert!(
        weak_fleet.upgrade().is_none(),
        "Arc cycle detected: Fleet was NOT dropped after coordinator Agent dropped!"
    );
    assert!(
        !wt_path.exists(),
        "worktree {} was not cleaned up by Fleet::drop()",
        wt_path.display()
    );

    let _ = fs::remove_dir_all(&repo_dir);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_orchestration_unified_fleet_budget_parent_plus_worker_and_sync_error() {
    use carnot::Provider;

    #[derive(Debug)]
    struct BudgetRouter {
        parent: Arc<FauxProvider>,
        worker: Arc<FauxProvider>,
        fail_sync_worker: bool,
    }
    impl Provider for BudgetRouter {
        fn name(&self) -> &'static str {
            "faux"
        }
        fn stream(
            &self,
            req: carnot_ai::ProviderRequest,
            cancel: tokio_util::sync::CancellationToken,
        ) -> carnot_ai::BoxFuture<
            '_,
            carnot::Result<tokio::sync::mpsc::Receiver<carnot_ai::StreamEvent>>,
        > {
            let is_worker = req
                .messages
                .first()
                .is_some_and(|m| m.text_concat().contains("BUDGET_WORKER"));
            if is_worker && self.fail_sync_worker && req.messages.len() > 1 {
                return Box::pin(async {
                    Err(carnot_core::CarnotError::Provider {
                        provider: "faux".into(),
                        message: "simulated worker turn-2 failure".into(),
                        retryable: false,
                        retry_after_ms: None,
                    })
                });
            }
            let target = if is_worker {
                Arc::clone(&self.worker)
            } else {
                Arc::clone(&self.parent)
            };
            Box::pin(async move { target.stream(req, cancel).await })
        }
    }

    // Part A: Combined parent ($0.04) + background worker ($0.025) crosses $0.05 cap -> halts & kills fleet!
    let dir_a = std::env::temp_dir().join(format!("carnot-budget-a-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir_a).expect("mkdir");

    let mut p_turn0 = ScriptedTurn::tool_calls(vec![(
        "s1",
        "Subagent",
        json!({
            "name": "spender",
            "task": "BUDGET_WORKER spend",
            "background": true,
            "keep_alive": true
        }),
    )]);
    // 3,200 output tokens on default faux pricing (~$0.0125 / 1k -> ~$0.040), UNDER $0.05 cap alone!
    p_turn0.output_tokens = 3_200;
    let p_turn1 = ScriptedTurn::text("Should halt on combined budget!");

    let mut w_turn0 = ScriptedTurn::text("Worker spent $0.025");
    w_turn0.output_tokens = 2_000; // ~$0.025 -> combined ~$0.065 >= $0.05!

    let mut cfg = CarnotConfig::new(dir_a.clone());
    cfg.modules.sandbox = false;
    cfg.modules.subagent = true;
    cfg.max_cost_usd = Some(0.05);

    let agent = CarnotBuilder::new(dir_a.clone())
        .config(cfg)
        .persist_session(false)
        .provider(Arc::new(BudgetRouter {
            parent: Arc::new(FauxProvider::new(vec![p_turn0, p_turn1])),
            worker: Arc::new(FauxProvider::new(vec![w_turn0])),
            fail_sync_worker: false,
        }))
        .build()
        .expect("build");

    let out = agent.prompt("Test combined cap").await.expect("run");
    assert!(
        out.halted_reason
            .as_deref()
            .is_some_and(|r| r.contains("cost budget")),
        "expected combined parent+worker budget halt, got {:?}",
        out.halted_reason
    );
    assert!(
        out.metrics.total_usage.cost_usd >= 0.05,
        "expected combined cost >= $0.05, got ${:.4}",
        out.metrics.total_usage.cost_usd
    );
    let _ = fs::remove_dir_all(&dir_a);

    // Part B: Failed sync one-shot subagent STILL counts its turn-1 cost into parent metrics (single-counted!)
    let dir_b = std::env::temp_dir().join(format!("carnot-budget-b-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir_b).expect("mkdir");
    fs::write(dir_b.join("f.txt"), "hello").expect("write");

    let p_sync = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![(
            "sync1",
            "Subagent",
            json!({"task": "BUDGET_WORKER sync fail"}),
        )]),
        ScriptedTurn::text("Handled worker error."),
    ]));
    let mut w_sync0 = ScriptedTurn::tool_calls(vec![("r1", "Read", json!({"file_path": "f.txt"}))]);
    w_sync0.output_tokens = 2_400; // ~$0.030 spent BEFORE failing on turn 2!

    let mut cfg_b = CarnotConfig::new(dir_b.clone());
    cfg_b.modules.sandbox = false;
    cfg_b.modules.subagent = true;
    let agent_b = CarnotBuilder::new(dir_b.clone())
        .config(cfg_b)
        .persist_session(false)
        .provider(Arc::new(BudgetRouter {
            parent: p_sync,
            worker: Arc::new(FauxProvider::new(vec![w_sync0])),
            fail_sync_worker: true,
        }))
        .build()
        .expect("build");

    let out_b = agent_b
        .prompt("Run failing sync worker")
        .await
        .expect("run");
    assert_eq!(out_b.final_text, "Handled worker error.");
    assert!(
        out_b.metrics.total_usage.cost_usd >= 0.025 && out_b.metrics.total_usage.cost_usd < 0.055,
        "failed sync subagent cost must be single-counted in parent metrics (~$0.03), got ${:.4}",
        out_b.metrics.total_usage.cost_usd
    );
    let _ = fs::remove_dir_all(&dir_b);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_orchestration_wait_any_blocks_with_idle_keep_alive_worker() {
    use carnot::Provider;
    use std::time::Instant;

    #[derive(Debug)]
    struct DelayedSlowWorker {
        parent: Arc<FauxProvider>,
        w1: Arc<FauxProvider>,
        w2: Arc<FauxProvider>,
    }
    impl Provider for DelayedSlowWorker {
        fn name(&self) -> &'static str {
            "faux"
        }
        fn stream(
            &self,
            req: carnot_ai::ProviderRequest,
            cancel: tokio_util::sync::CancellationToken,
        ) -> carnot_ai::BoxFuture<
            '_,
            carnot::Result<tokio::sync::mpsc::Receiver<carnot_ai::StreamEvent>>,
        > {
            let first = req
                .messages
                .first()
                .map(|m| m.text_concat())
                .unwrap_or_default();
            if first.contains("FAST_W1") {
                let target = Arc::clone(&self.w1);
                return Box::pin(async move { target.stream(req, cancel).await });
            }
            if first.contains("SLOW_W2") {
                let target = Arc::clone(&self.w2);
                return Box::pin(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(160)).await;
                    target.stream(req, cancel).await
                });
            }
            let target = Arc::clone(&self.parent);
            Box::pin(async move { target.stream(req, cancel).await })
        }
    }

    let dir = std::env::temp_dir().join(format!("carnot-wait-any-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    let parent_faux = Arc::new(FauxProvider::new(vec![
        // 0: Spawn fast keep_alive worker w1 synchronously so it finishes & sits Idle (report already read)
        ScriptedTurn::tool_calls(vec![(
            "s1",
            "Subagent",
            json!({"name": "w1", "task": "FAST_W1", "keep_alive": true}),
        )]),
        // 1: Spawn slow background worker w2
        ScriptedTurn::tool_calls(vec![(
            "s2",
            "Subagent",
            json!({"name": "w2", "task": "SLOW_W2", "background": true, "keep_alive": false}),
        )]),
        // 2: Call fleet-wide wait(any) while w1 is Idle and w2 is Running!
        //    MUST block until w2 finishes instead of spinning immediately on idle w1!
        ScriptedTurn::tool_calls(vec![(
            "wa",
            "Subagent",
            json!({"wait": true, "wait_mode": "any", "timeout": 5}),
        )]),
        // 3: Final answer
        ScriptedTurn::text("Waited properly for w2."),
    ]));

    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .subagent(true)
        .provider(Arc::new(DelayedSlowWorker {
            parent: Arc::clone(&parent_faux),
            w1: Arc::new(FauxProvider::new(vec![ScriptedTurn::text("W1_DONE")])),
            w2: Arc::new(FauxProvider::new(vec![ScriptedTurn::text(
                "W2_SLOW_DONE_88",
            )])),
        }))
        .build()
        .expect("build");

    let t0 = Instant::now();
    let out = agent
        .prompt("Test wait(any) with idle w1")
        .await
        .expect("run");
    let elapsed = t0.elapsed();

    assert_eq!(out.final_text, "Waited properly for w2.");
    assert!(
        elapsed >= std::time::Duration::from_millis(130),
        "wait(any) returned too fast ({elapsed:?}) — spun on already-idle w1!"
    );
    let preqs = parent_faux.recorded_requests();
    assert!(
        msgs_json(&preqs[3].messages).contains("W2_SLOW_DONE_88"),
        "wait(any) must collect slow w2's report"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_resume_forked_does_not_mutate_original_truncated_session() {
    use std::io::Write;

    let dir = std::env::temp_dir().join(format!("carnot-fork-ro-{}", uuid::Uuid::new_v4()));
    let sess_dir = dir.join("sessions");
    fs::create_dir_all(&sess_dir).expect("mkdir");

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.session_dir = sess_dir.clone();
    cfg.modules.sandbox = false;

    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg.clone())
        .persist_session(true)
        .provider(Arc::new(FauxProvider::new(vec![ScriptedTurn::text(
            "Original reply.",
        )])))
        .build()
        .expect("build");
    let orig_sid = agent.session_id().await;
    let _ = agent.prompt("Original prompt").await.expect("run");
    drop(agent);

    let orig_path = sess_dir.join(format!("{orig_sid}.jsonl"));
    {
        let mut f = fs::OpenOptions::new()
            .append(true)
            .open(&orig_path)
            .expect("open");
        f.write_all(b"{\"type\":\"message\",\"truncated_crash_tail")
            .expect("write");
    }
    let bytes_before = fs::read(&orig_path).expect("read before");

    let cas = Arc::new(carnot::CasStore::open(&dir).expect("cas"));
    let tools = carnot::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
    let forked = carnot::Agent::resume_forked(
        &orig_path,
        cfg,
        Arc::new(FauxProvider::new(vec![ScriptedTurn::text("Forked reply.")])),
        tools,
        Vec::new(),
        true,
    )
    .expect("resume_forked");

    // Original file on disk MUST remain 100% byte-identical (read-only load)!
    let bytes_after = fs::read(&orig_path).expect("read after");
    assert_eq!(
        bytes_before, bytes_after,
        "resume_forked must NEVER truncate or mutate the original session file"
    );
    assert_ne!(forked.session_id().await, orig_sid);
    let msgs = forked.active_messages().await;
    assert!(
        msgs.iter()
            .any(|m| m.text_concat().contains("Original reply."))
    );

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_orchestration_user_steer_wakes_wait_and_worker_halted_reason_surfaced() {
    use carnot::Provider;
    use std::time::Instant;

    #[derive(Debug)]
    struct MultiRouter {
        parent: Arc<FauxProvider>,
        slow_worker: Arc<FauxProvider>,
        sync_worker: Arc<FauxProvider>,
    }
    impl Provider for MultiRouter {
        fn name(&self) -> &'static str {
            "faux"
        }
        fn stream(
            &self,
            req: carnot_ai::ProviderRequest,
            cancel: tokio_util::sync::CancellationToken,
        ) -> carnot_ai::BoxFuture<
            '_,
            carnot::Result<tokio::sync::mpsc::Receiver<carnot_ai::StreamEvent>>,
        > {
            let first = req
                .messages
                .first()
                .map(|m| m.text_concat())
                .unwrap_or_default();
            if first.contains("SLOW_STEER_WORKER") {
                let target = Arc::clone(&self.slow_worker);
                return Box::pin(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(800)).await;
                    target.stream(req, cancel).await
                });
            }
            if first.contains("SYNC_HALT_WORKER") {
                let target = Arc::clone(&self.sync_worker);
                return Box::pin(async move { target.stream(req, cancel).await });
            }
            let target = Arc::clone(&self.parent);
            Box::pin(async move { target.stream(req, cancel).await })
        }
    }

    let dir = std::env::temp_dir().join(format!("carnot-steer-wake-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(dir.join("x.txt"), "hello").expect("write");

    let parent_faux = Arc::new(FauxProvider::new(vec![
        // 0: Spawn slow worker with max_turns=1 that will hit max_turns if it calls a tool
        ScriptedTurn::tool_calls(vec![(
            "s1",
            "Subagent",
            json!({
                "name": "slow",
                "task": "SLOW_STEER_WORKER",
                "background": true,
                "keep_alive": false,
                "max_turns": 1
            }),
        )]),
        // 1: Parent emits plain text -> enters `on_turn_boundary` reactive wait!
        ScriptedTurn::text("Waiting on slow..."),
        // 2: Woken EARLY by human `steer()` -> kills `slow`!
        ScriptedTurn::tool_calls(vec![(
            "k1",
            "Subagent",
            json!({"job": "slow", "kill": true}),
        )]),
        // 3: Spawn quick sync worker with max_turns=1 that calls a tool -> halts on max_turns!
        ScriptedTurn::tool_calls(vec![(
            "s2",
            "Subagent",
            json!({"task": "SYNC_HALT_WORKER", "max_turns": 1}),
        )]),
        // 4: Final answer
        ScriptedTurn::text("Steer woke early and saw halted_reason."),
    ]));
    // Sync worker's 1 turn calls Read (so max_turns=1 triggers halted_reason="max_turns reached (1)")
    let worker_faux = Arc::new(FauxProvider::new(vec![ScriptedTurn::tool_calls(vec![(
        "r1",
        "Read",
        json!({"file_path": "x.txt"}),
    )])]));

    let agent = Arc::new(
        CarnotBuilder::new(dir.clone())
            .sandbox(false)
            .persist_session(false)
            .subagent(true)
            .provider(Arc::new(MultiRouter {
                parent: Arc::clone(&parent_faux),
                slow_worker: Arc::new(FauxProvider::new(vec![ScriptedTurn::text("too late")])),
                sync_worker: worker_faux,
            }))
            .build()
            .expect("build"),
    );

    // Send a human steer 40ms in while parent is blocked in `on_turn_boundary` waiting on 800ms worker!
    let steer_agent = Arc::clone(&agent);
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        steer_agent.steer("Cancel slow worker now!").await;
    });

    let t0 = Instant::now();
    let out = agent
        .prompt("Test steer wakeup and halted_reason")
        .await
        .expect("run");
    let elapsed = t0.elapsed();

    assert_eq!(out.final_text, "Steer woke early and saw halted_reason.");
    // Must wake WELL before the 800ms slow worker would have finished!
    assert!(
        elapsed < std::time::Duration::from_millis(450),
        "human steer did not wake on_turn_boundary immediately (took {elapsed:?})"
    );
    let preqs = parent_faux.recorded_requests();
    let all_parent = msgs_json(&preqs[4].messages);
    assert!(
        all_parent.contains("[user-steer]"),
        "human steer must still carry [user-steer]"
    );
    assert!(
        all_parent.contains("max_turns reached (1)") && all_parent.contains("[subagent-halted]"),
        "worker halted_reason must be surfaced in subagent report: {all_parent}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(all(feature = "ext-subagent", feature = "ext-hooks"))]
#[tokio::test]
async fn e2e_orchestration_pre_tool_use_hook_applies_inside_worker() {
    use carnot::{HooksConfig, Provider};

    #[derive(Debug)]
    struct HookSplit {
        parent: Arc<FauxProvider>,
        worker: Arc<FauxProvider>,
    }
    impl Provider for HookSplit {
        fn name(&self) -> &'static str {
            "faux"
        }
        fn stream(
            &self,
            req: carnot_ai::ProviderRequest,
            cancel: tokio_util::sync::CancellationToken,
        ) -> carnot_ai::BoxFuture<
            '_,
            carnot::Result<tokio::sync::mpsc::Receiver<carnot_ai::StreamEvent>>,
        > {
            let is_worker = req
                .messages
                .first()
                .is_some_and(|m| m.text_concat().contains("HOOK_TEST_WORKER"));
            let target = if is_worker {
                Arc::clone(&self.worker)
            } else {
                Arc::clone(&self.parent)
            };
            Box::pin(async move { target.stream(req, cancel).await })
        }
    }

    let dir = std::env::temp_dir().join(format!("carnot-worker-hook-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");
    let hooks_file = dir.join("hooks.json");
    let marker = dir.join("should_not_exist.txt");
    fs::write(
        &hooks_file,
        serde_json::to_string(&json!({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "Bash",
                    "hooks": [{
                        "type": "command",
                        "command": "echo 'blocked-by-pre-tool-use-hook' >&2; exit 2"
                    }]
                }]
            }
        }))
        .expect("json"),
    )
    .expect("write hooks");

    let parent_faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![(
            "s1",
            "Subagent",
            json!({"name": "hooked", "task": "HOOK_TEST_WORKER run bash"}),
        )]),
        ScriptedTurn::text("Worker hook verified."),
    ]));
    let worker_faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![(
            "b1",
            "Bash",
            json!({"command": format!("echo pwned > {}", marker.display())}),
        )]),
        ScriptedTurn::text("Bash was denied by hook."),
    ]));

    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .subagent(true)
        .hooks_config(HooksConfig {
            files: vec![hooks_file],
            load_carnot_user_hooks: false,
            load_claude_user_settings: false,
            trust_workspace_hooks: false,
            fail_closed: true,
        })
        .provider(Arc::new(HookSplit {
            parent: parent_faux,
            worker: Arc::clone(&worker_faux),
        }))
        .build()
        .expect("build");

    let out = agent.prompt("Verify worker hooks").await.expect("run");
    assert_eq!(out.final_text, "Worker hook verified.");
    assert!(
        !marker.exists(),
        "PreToolUse hook failed to block Bash inside worker!"
    );
    let worker_reqs = worker_faux.recorded_requests();
    assert!(
        msgs_json(&worker_reqs[1].messages).contains("blocked-by-pre-tool-use-hook"),
        "expected hook denial reason in worker tool_result"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_prompt_with_cancel_aborts_before_provider_call() {
    let dir = std::env::temp_dir().join(format!("carnot-precancel-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    let faux = Arc::new(FauxProvider::new(vec![ScriptedTurn::text(
        "should never run",
    )]));
    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .provider(Arc::clone(&faux) as Arc<dyn carnot::Provider>)
        .build()
        .expect("build");

    let pre_cancelled = tokio_util::sync::CancellationToken::new();
    pre_cancelled.cancel();

    let res = agent
        .prompt_with_cancel("Must abort immediately", pre_cancelled)
        .await;
    assert!(
        matches!(res, Err(carnot_core::CarnotError::Cancelled)),
        "expected Err(Cancelled), got {res:?}"
    );
    assert!(
        faux.recorded_requests().is_empty(),
        "pre-cancelled prompt_with_cancel must make zero provider requests"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(all(feature = "ext-subagent", feature = "ext-memory", feature = "ext-web"))]
#[tokio::test]
async fn e2e_forked_worker_shares_exact_prefix_hash_and_cache_key_with_extensions() {
    use carnot::{MemoryConfig, Provider, WebConfig};

    #[derive(Debug)]
    struct ForkExtSplit {
        parent: Arc<FauxProvider>,
        worker: Arc<FauxProvider>,
    }
    impl Provider for ForkExtSplit {
        fn name(&self) -> &'static str {
            "faux"
        }
        fn stream(
            &self,
            req: carnot_ai::ProviderRequest,
            cancel: tokio_util::sync::CancellationToken,
        ) -> carnot_ai::BoxFuture<
            '_,
            carnot::Result<tokio::sync::mpsc::Receiver<carnot_ai::StreamEvent>>,
        > {
            let is_worker = req
                .messages
                .last()
                .is_some_and(|m| m.text_concat().contains("FORK_EXT_WORKER_CHECK"))
                || req
                    .messages
                    .iter()
                    .any(|m| m.text_concat().contains("FORK_EXT_WORKER_CHECK"));
            let target = if is_worker {
                Arc::clone(&self.worker)
            } else {
                Arc::clone(&self.parent)
            };
            Box::pin(async move { target.stream(req, cancel).await })
        }
    }

    let dir = std::env::temp_dir().join(format!("carnot-fork-ext-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    let parent_faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![(
            "s1",
            "Subagent",
            json!({
                "name": "forked_ext",
                "task": "FORK_EXT_WORKER_CHECK",
                "fork": true
            }),
        )]),
        ScriptedTurn::text("Fork with extensions matched cache prefix."),
    ]));
    let worker_faux = Arc::new(FauxProvider::new(vec![
        // Worker tries calling coordinator-only mirrored MemorySearch -> clean refusal!
        ScriptedTurn::tool_calls(vec![("m1", "MemorySearch", json!({"query": "test"}))]),
        ScriptedTurn::text("Worker saw mirrored refusal and finished."),
    ]));

    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .subagent(true)
        .memory_config(MemoryConfig::default())
        .web_config(WebConfig::default())
        .provider(Arc::new(ForkExtSplit {
            parent: Arc::clone(&parent_faux),
            worker: Arc::clone(&worker_faux),
        }))
        .build()
        .expect("build");

    let out = agent
        .prompt("Test warm fork with extensions")
        .await
        .expect("run");
    assert_eq!(out.final_text, "Fork with extensions matched cache prefix.");
    assert_eq!(out.metrics.cache_breaks, 0);

    let preqs = parent_faux.recorded_requests();
    let wreqs = worker_faux.recorded_requests();
    assert_eq!(preqs.len(), 2);
    assert_eq!(wreqs.len(), 2);

    // Parent and forked worker MUST have 100% byte-identical wire prefix + inherited cache key!
    assert_eq!(preqs[0].prefix_hash, wreqs[0].prefix_hash);
    assert_eq!(preqs[0].static_system_prompt, wreqs[0].static_system_prompt);
    assert_eq!(
        preqs[0].session_id, wreqs[0].session_id,
        "forked worker must inherit parent cache_key_id for OpenAI prompt_cache_key"
    );
    assert_eq!(
        serde_json::to_vec(&preqs[0].tools).expect("p tools"),
        serde_json::to_vec(&wreqs[0].tools).expect("w tools"),
        "forked worker must mirror exact canonicalized parent tool specs even with Web+Memory!"
    );
    assert_eq!(
        serde_json::to_vec(&preqs[0].native_web).expect("p web"),
        serde_json::to_vec(&wreqs[0].native_web).expect("w web"),
    );

    // Mirrored coordinator tool MUST refuse execution cleanly inside worker
    assert!(
        msgs_json(&wreqs[1].messages).contains("only available on the coordinator"),
        "expected coordinator-only tool refusal in worker"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_runtime_prefix_freeze_and_live_worker_updated_events() {
    use carnot::Provider;

    #[derive(Debug)]
    struct EvSplit {
        parent: Arc<FauxProvider>,
        worker: Arc<FauxProvider>,
    }
    impl Provider for EvSplit {
        fn name(&self) -> &'static str {
            "faux"
        }
        fn stream(
            &self,
            req: carnot_ai::ProviderRequest,
            cancel: tokio_util::sync::CancellationToken,
        ) -> carnot_ai::BoxFuture<
            '_,
            carnot::Result<tokio::sync::mpsc::Receiver<carnot_ai::StreamEvent>>,
        > {
            let is_worker = req
                .messages
                .first()
                .is_some_and(|m| m.text_concat().contains("LIVE_EVENT_WORKER"));
            let target = if is_worker {
                Arc::clone(&self.worker)
            } else {
                Arc::clone(&self.parent)
            };
            Box::pin(async move { target.stream(req, cancel).await })
        }
    }

    let dir = std::env::temp_dir().join(format!("carnot-ev-live-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    let parent_faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![(
            "s1",
            "Subagent",
            json!({
                "name": "obs",
                "task": "LIVE_EVENT_WORKER",
                "background": true,
                "keep_alive": true
            }),
        )]),
        ScriptedTurn::tool_calls(vec![(
            "w1",
            "Subagent",
            json!({"job": "obs", "wait": true}),
        )]),
        ScriptedTurn::tool_calls(vec![(
            "k1",
            "Subagent",
            json!({"job": "obs", "kill": true}),
        )]),
        ScriptedTurn::text("Observed all worker lifecycle events."),
    ]));
    let worker_faux = Arc::new(FauxProvider::new(vec![ScriptedTurn::text("obs done")]));

    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .subagent(true)
        .provider(Arc::new(EvSplit {
            parent: parent_faux,
            worker: worker_faux,
        }))
        .build()
        .expect("build");

    let mut ev_stream = agent.subscribe();
    let collect = tokio::spawn(async move {
        let mut saw_prefix_audit = false;
        let mut cache_breaks = 0usize;
        let mut worker_states: Vec<String> = Vec::new();
        loop {
            let env = ev_stream.next().await;
            match env.event {
                carnot::AgentEvent::PrefixAudit { stable, .. } => {
                    assert!(stable);
                    saw_prefix_audit = true;
                }
                carnot::AgentEvent::CacheBreakDetected { .. } => {
                    cache_breaks += 1;
                }
                carnot::AgentEvent::WorkerUpdated { worker, state, .. } => {
                    if worker == "obs" {
                        worker_states.push(state);
                    }
                }
                carnot::AgentEvent::RunSettled { .. } | carnot::AgentEvent::Error { .. } => break,
                _ => {}
            }
        }
        (saw_prefix_audit, cache_breaks, worker_states)
    });

    let out = agent
        .prompt("Run live worker events test")
        .await
        .expect("run");
    assert_eq!(out.final_text, "Observed all worker lifecycle events.");
    assert_eq!(out.metrics.cache_breaks, 0);

    let (saw_audit, breaks, states) = collect.await.expect("join");
    assert!(saw_audit, "expected initial PrefixAudit event");
    assert_eq!(
        breaks, 0,
        "expected zero unexpected CacheBreakDetected events"
    );
    assert!(
        states.iter().any(|s| s == "running"),
        "expected WorkerUpdated running event: {states:?}"
    );
    assert!(
        states.iter().any(|s| s == "idle"),
        "expected WorkerUpdated idle event: {states:?}"
    );
    assert!(
        states.iter().any(|s| s == "closed"),
        "expected WorkerUpdated closed event: {states:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_adversarial_strict_schema_empty_defaults_on_all_core_tools() {
    use carnot::Provider;

    #[derive(Debug)]
    struct StrictSplit {
        parent: Arc<FauxProvider>,
        worker: Arc<FauxProvider>,
    }
    impl Provider for StrictSplit {
        fn name(&self) -> &'static str {
            "faux"
        }
        fn stream(
            &self,
            req: carnot_ai::ProviderRequest,
            cancel: tokio_util::sync::CancellationToken,
        ) -> carnot_ai::BoxFuture<
            '_,
            carnot::Result<tokio::sync::mpsc::Receiver<carnot_ai::StreamEvent>>,
        > {
            let is_worker = req
                .messages
                .first()
                .is_some_and(|m| m.text_concat().contains("STRICT_SCHEMA_WORKER"));
            let target = if is_worker {
                Arc::clone(&self.worker)
            } else {
                Arc::clone(&self.parent)
            };
            Box::pin(async move { target.stream(req, cancel).await })
        }
    }

    let dir = std::env::temp_dir().join(format!("carnot-strict-tools-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");
    fs::write(
        dir.join("app.rs"),
        "fn alpha() -> i32 { 1 }\nfn beta() -> i32 { 2 }\n",
    )
    .expect("write");

    // Simulate OpenAI strict-schema calls where EVERY optional field is filled with "", 0, [], false!
    let parent_faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![
            (
                "r1",
                "Read",
                json!({"file_path": "app.rs", "offset": 0, "limit": 0, "outline": false, "symbol": ""}),
            ),
            (
                "g1",
                "Grep",
                json!({"pattern": "fn ", "path": "", "ext": "", "glob": "", "case_insensitive": false, "limit": 0}),
            ),
            (
                "e1",
                "Edit",
                json!({"file_path": "app.rs", "edits": [], "old_string": "1", "new_string": "42", "start_line": 0, "end_line": 0, "replace_all": false}),
            ),
            (
                "b1",
                "Bash",
                json!({"command": "echo strict-bash-ok", "cwd": "", "job": "", "kill": false, "timeout": 0}),
            ),
            (
                "s1",
                "Subagent",
                json!({
                    "task": "STRICT_SCHEMA_WORKER check",
                    "name": "strict_w",
                    "provider": "",
                    "model": "",
                    "effort": "",
                    "fork": false,
                    "keep_alive": false,
                    "max_turns": 0,
                    "isolated_worktree": false,
                    "background": false,
                    "job": "",
                    "jobs": [],
                    "wait": false,
                    "wait_mode": "",
                    "timeout": 0,
                    "land": false,
                    "kill": false,
                    "list": false
                }),
            ),
        ]),
        ScriptedTurn::text("All strict-schema default calls succeeded."),
    ]));
    let worker_faux = Arc::new(FauxProvider::new(vec![ScriptedTurn::text(
        "STRICT_WORKER_OK",
    )]));

    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .subagent(true)
        .provider(Arc::new(StrictSplit {
            parent: Arc::clone(&parent_faux),
            worker: worker_faux,
        }))
        .build()
        .expect("build");

    let out = agent
        .prompt("Run strict schema gauntlet")
        .await
        .expect("run");
    assert_eq!(out.final_text, "All strict-schema default calls succeeded.");

    // Verify EVERY single tool succeeded with non-empty output (none errored or returned 0 lines/1 match)
    let turn2_ctx = msgs_json(&parent_faux.recorded_requests()[1].messages);
    assert!(
        turn2_ctx.contains("fn alpha()"),
        "Read with offset=0/limit=0 failed: {turn2_ctx}"
    );
    assert!(
        turn2_ctx.contains("fn beta()"),
        "Grep with path=\"\"/ext=\"\"/limit=0 failed: {turn2_ctx}"
    );
    assert!(
        turn2_ctx.contains("Edited app.rs"),
        "Edit with edits=[] + old_string failed: {turn2_ctx}"
    );
    assert!(
        turn2_ctx.contains("strict-bash-ok"),
        "Bash with job=\"\"/timeout=0 failed: {turn2_ctx}"
    );
    assert!(
        turn2_ctx.contains("STRICT_WORKER_OK"),
        "Subagent with job=\"\"/jobs=[]/max_turns=0 failed: {turn2_ctx}"
    );
    assert_eq!(
        fs::read_to_string(dir.join("app.rs")).expect("read"),
        "fn alpha() -> i32 { 42 }\nfn beta() -> i32 { 2 }\n"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "ext-subagent")]
#[tokio::test]
async fn e2e_adversarial_nested_fork_of_fork_and_mid_tool_sanitization() {
    use carnot::Provider;

    #[derive(Debug)]
    struct NestedSplit {
        parent: Arc<FauxProvider>,
        worker: Arc<FauxProvider>,
    }
    impl Provider for NestedSplit {
        fn name(&self) -> &'static str {
            "faux"
        }
        fn stream(
            &self,
            req: carnot_ai::ProviderRequest,
            cancel: tokio_util::sync::CancellationToken,
        ) -> carnot_ai::BoxFuture<
            '_,
            carnot::Result<tokio::sync::mpsc::Receiver<carnot_ai::StreamEvent>>,
        > {
            let is_worker = req
                .messages
                .last()
                .is_some_and(|m| m.text_concat().contains("NESTED_FORK_WORKER"));
            let target = if is_worker {
                Arc::clone(&self.worker)
            } else {
                Arc::clone(&self.parent)
            };
            Box::pin(async move { target.stream(req, cancel).await })
        }
    }

    let dir = std::env::temp_dir().join(format!("carnot-nested-fork-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("mkdir");

    let parent_faux = Arc::new(FauxProvider::new(vec![
        // Session A turn 1
        ScriptedTurn::text("Session A answer."),
        // Session B (fork of A) spawns worker C with fork: true (fork of a fork, mid-tool_use!)
        ScriptedTurn::tool_calls(vec![(
            "s1",
            "Subagent",
            json!({
                "name": "grandchild_fork",
                "task": "NESTED_FORK_WORKER",
                "fork": true
            }),
        )]),
        ScriptedTurn::text("Session B completed with forked worker C."),
    ]));
    let worker_faux = Arc::new(FauxProvider::new(vec![ScriptedTurn::text(
        "Worker C saw Session A answer with zero orphan tool_use.",
    )]));

    let agent = CarnotBuilder::new(dir.clone())
        .sandbox(false)
        .persist_session(false)
        .subagent(true)
        .provider(Arc::new(NestedSplit {
            parent: Arc::clone(&parent_faux),
            worker: Arc::clone(&worker_faux),
        }))
        .build()
        .expect("build");

    let sid_a = agent.session_id().await;
    let out_a = agent.prompt("Prompt on session A").await.expect("run A");
    assert_eq!(out_a.final_text, "Session A answer.");

    // Fork A -> B
    let (sid_b, _) = agent.fork_session().await.expect("fork A->B");
    assert_ne!(sid_a, sid_b);

    let out_b = agent
        .prompt("Now spawn forked worker C")
        .await
        .expect("run B");
    assert_eq!(
        out_b.final_text,
        "Session B completed with forked worker C."
    );
    assert_eq!(out_b.metrics.cache_breaks, 0);

    let preqs = parent_faux.recorded_requests();
    let wreqs = worker_faux.recorded_requests();
    assert_eq!(preqs.len(), 3);
    assert_eq!(wreqs.len(), 1);

    // Transitive cache_key_id inheritance: A == B == C (for OpenAI prompt_cache_key!)
    assert_eq!(preqs[0].session_id, preqs[1].session_id);
    assert_eq!(preqs[0].session_id, wreqs[0].session_id);
    // Byte-identical prefix_hash across A, B, and C!
    assert_eq!(preqs[0].prefix_hash, preqs[1].prefix_hash);
    assert_eq!(preqs[0].prefix_hash, wreqs[0].prefix_hash);

    // Worker C must inherit Session A's history AND contain ZERO orphan trailing tool_use blocks!
    let w_msgs = &wreqs[0].messages;
    assert!(msgs_json(w_msgs).contains("Session A answer."));
    for (i, m) in w_msgs.iter().enumerate() {
        if m.role == carnot_core::Role::Assistant && m.has_tool_use() {
            assert!(
                w_msgs
                    .get(i + 1)
                    .is_some_and(|next| next.role == carnot_core::Role::User),
                "orphan tool_use found in forked worker messages at index {i}!"
            );
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "ext-memory")]
#[tokio::test]
async fn e2e_adversarial_resume_with_different_tools_emits_unstable_prefix_audit() {
    use carnot::MemoryConfig;

    let dir = std::env::temp_dir().join(format!("carnot-resume-drift-{}", uuid::Uuid::new_v4()));
    let sess_dir = dir.join("sessions");
    fs::create_dir_all(&sess_dir).expect("mkdir");

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.session_dir = sess_dir.clone();
    cfg.modules.sandbox = false;

    // 1. Create original session WITH `--memory`
    let orig = CarnotBuilder::new(dir.clone())
        .config(cfg.clone())
        .persist_session(true)
        .memory_config(MemoryConfig::default())
        .provider(Arc::new(FauxProvider::new(vec![ScriptedTurn::text(
            "Saved with memory.",
        )])))
        .build()
        .expect("build orig");
    let orig_sid = orig.session_id().await;
    let orig_ph = orig.prefix_hash().await;
    let _ = orig.prompt("Hello").await.expect("run orig");
    drop(orig);

    // 2. Resume the SAME session WITHOUT `--memory` -> tool set drifted!
    let orig_path = sess_dir.join(format!("{orig_sid}.jsonl"));
    let resumed = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .resume_session(orig_path)
        .provider(Arc::new(FauxProvider::new(vec![ScriptedTurn::text(
            "Resumed without memory.",
        )])))
        .build()
        .expect("build resumed");

    // Prefix hash MUST have been refreshed to match the actual resumed tool set
    let resumed_ph = resumed.prefix_hash().await;
    assert_ne!(
        orig_ph, resumed_ph,
        "resumed session with different tools must update prefix_hash"
    );

    // PrefixAudit { stable: false } MUST have been emitted onto EventLog!
    let mut stream = resumed.subscribe_from(0);
    let mut saw_unstable_audit = false;
    for _ in 0..4 {
        let Ok(env) =
            tokio::time::timeout(std::time::Duration::from_millis(50), stream.next()).await
        else {
            break;
        };
        if let carnot::AgentEvent::PrefixAudit { stable, note, .. } = env.event
            && !stable
            && note
                .as_deref()
                .is_some_and(|n| n.contains("different tool set"))
        {
            saw_unstable_audit = true;
            break;
        }
    }
    assert!(
        saw_unstable_audit,
        "expected PrefixAudit {{ stable: false }} when resuming with a changed tool set"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(feature = "ext-codemode")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_codemode_batches_tools_enforces_policy_guard_and_preserves_prefix_invariance() {
    let dir1 = std::env::temp_dir().join(format!("carnot-cm-e2e-1-{}", uuid::Uuid::new_v4()));
    let dir2 = std::env::temp_dir().join(format!("carnot-cm-e2e-2-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir1).expect("mkdir 1");
    fs::create_dir_all(&dir2).expect("mkdir 2");
    fs::write(dir1.join("a.txt"), "alpha=10\n").expect("write a");
    fs::write(dir1.join("b.txt"), "beta=20\n").expect("write b");
    fs::write(dir1.join("secret_cfg.txt"), "cfg=important_content\n").expect("write secret_cfg");
    fs::write(
        dir1.join("multi.txt"),
        "k0=old\nk1=old\nk2=old\nk3=old\nk4=old\nk5=old\nk6=old\nk7=old\n",
    )
    .expect("write multi");
    fs::write(dir1.join(".env"), "TOP_SECRET=never_expose_me\n").expect("write env");

    let faux1 = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::tool_calls(vec![(
            "cm1",
            "CodeMode",
            json!({
                "code": r#"
                    const [a, b, cfg] = await Promise.all([
                        tools.Read({ file_path: "a.txt" }),
                        tools.Read({ file_path: "b.txt" }),
                        tools.Read({ file_path: "secret_cfg.txt" }),
                    ]);
                    // 8 concurrent Edits to the SAME file must serialize without losing any update!
                    await Promise.all([0,1,2,3,4,5,6,7].map(i =>
                        tools.Edit({ file_path: "multi.txt", old_string: `k${i}=old`, new_string: `k${i}=new` })
                    ));
                    let blocked = false;
                    try {
                        await tools.Read({ file_path: ".env" });
                    } catch (e) {
                        blocked = String(e.message).includes("denied");
                    }
                    store("done", true);
                    return { a_ok: a.includes("alpha=10"), b_ok: b.includes("beta=20"), cfg_len: cfg.length, env_blocked: blocked };
                "#
            }),
        )]),
        // Turn 2: Model reads `secret_cfg.txt` directly -> MUST return real content, NOT `[unchanged]` stub!
        ScriptedTurn::tool_calls(vec![(
            "r_cfg",
            "Read",
            json!({"file_path": "secret_cfg.txt"}),
        )]),
        ScriptedTurn::text("Batched safely via CodeMode."),
    ]));

    let mut cfg1 = CarnotConfig::new(dir1.clone());
    cfg1.modules.sandbox = false;
    let agent1 = CarnotBuilder::new(dir1.clone())
        .config(cfg1)
        .provider(Arc::clone(&faux1) as Arc<dyn carnot::Provider>)
        .persist_session(false)
        .policy_guard(true)
        .codemode(true)
        .build()
        .expect("build agent1");

    let out = agent1
        .prompt("Inspect a.txt, b.txt, multi.txt, and check .env policy via CodeMode")
        .await
        .expect("prompt");
    assert_eq!(out.final_text, "Batched safely via CodeMode.");

    // All 8 parallel edits inside CodeMode MUST survive on disk!
    let multi_disk = fs::read_to_string(dir1.join("multi.txt")).expect("read multi");
    for i in 0..8 {
        assert!(
            multi_disk.contains(&format!("k{i}=new")),
            "lost parallel CodeMode edit k{i}! disk:\n{multi_disk}"
        );
    }

    let reqs = faux1.recorded_requests();
    assert_eq!(reqs.len(), 3, "expected 3 provider turns");
    let turn2_json = msgs_json(&reqs[1].messages);
    assert!(
        turn2_json.contains("\\\"a_ok\\\":true")
            && turn2_json.contains("\\\"b_ok\\\":true")
            && turn2_json.contains("\\\"env_blocked\\\":true"),
        "CodeMode should batch reads and enforce PolicyGuard on .env, got:\n{turn2_json}"
    );
    assert!(
        !turn2_json.contains("never_expose_me"),
        "secret from .env must NEVER leak through CodeMode"
    );
    // Turn 3 must see actual content of `secret_cfg.txt`, proving CodeMode Read did not poison ReadTracker!
    let turn3_json = msgs_json(&reqs[2].messages);
    assert!(
        turn3_json.contains("cfg=important_content") && !turn3_json.contains("[unchanged:"),
        "Read after CodeMode Read must return real content, not unchanged stub! got:\n{turn3_json}"
    );

    // Cross-CWD prefix invariance with `codemode(true)` must hold byte-for-byte.
    let mut cfg2 = CarnotConfig::new(dir2.clone());
    cfg2.modules.sandbox = false;
    let agent2 = CarnotBuilder::new(dir2.clone())
        .config(cfg2)
        .provider(Arc::new(FauxProvider::new(vec![ScriptedTurn::text("ok")])))
        .persist_session(false)
        .policy_guard(true)
        .codemode(true)
        .build()
        .expect("build agent2");
    assert_eq!(
        agent1.prefix_hash().await,
        agent2.prefix_hash().await,
        "CodeMode must keep static prefix BLAKE3 hash identical across workspaces"
    );

    // Dropping Agent must be completely acyclic (no Arc leak via CodeModeModule).
    let weak = Arc::downgrade(&agent1);
    drop(agent1);
    assert!(
        weak.upgrade().is_none(),
        "Agent must drop cleanly with zero Arc reference cycles"
    );

    let _ = fs::remove_dir_all(&dir1);
    let _ = fs::remove_dir_all(&dir2);
}

#[cfg(feature = "ext-goal")]
#[tokio::test]
async fn e2e_goal_deterministic_and_independent_verifier_loop_preserves_prefix_invariance() {
    let dir1 = std::env::temp_dir().join(format!("carnot-goal-1-{}", uuid::Uuid::new_v4()));
    let dir2 = std::env::temp_dir().join(format!("carnot-goal-2-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir1).expect("mkdir 1");
    fs::create_dir_all(&dir2).expect("mkdir 2");
    fs::write(dir1.join("answer.txt"), "wrong\n").expect("seed answer.txt");

    // Worker tries to stop prematurely on Turn 1 (fails deterministic check -> 0 verifier calls),
    // then reads & fixes answer.txt on Turns 2-4, then verifier rejects once (asks for comment),
    // then worker updates answer.txt and verifier returns `met`.
    let worker_faux = Arc::new(FauxProvider::new(vec![
        // Attempt 1: premature stop -> deterministic check `grep -q "^42$" answer.txt` fails
        ScriptedTurn::text("I think it's already done."),
        // Attempt 2: worker reads answer.txt, writes 42, then stops
        ScriptedTurn::tool_calls(vec![("r1", "Read", json!({"file_path": "answer.txt"}))]),
        ScriptedTurn::tool_calls(vec![(
            "w1",
            "Write",
            json!({"file_path": "answer.txt", "content": "42\n"}),
        )]),
        ScriptedTurn::text("Fixed answer.txt to 42."),
        // Attempt 3 (after independent verifier says `not_met`): add marker.txt and stop
        ScriptedTurn::tool_calls(vec![(
            "w2",
            "Write",
            json!({"file_path": "marker.txt", "content": "verified\n"}),
        )]),
        ScriptedTurn::text("Added marker.txt as requested."),
    ]));

    // Independent virgin-context verifier: only invoked AFTER deterministic check passes!
    let verifier_faux = Arc::new(FauxProvider::new(vec![
        ScriptedTurn::text(
            r#"{"status":"not_met","feedback":"Also create marker.txt containing verified."}"#,
        ),
        ScriptedTurn::text(
            r#"{"status":"met","feedback":"answer.txt is 42 and marker.txt is present."}"#,
        ),
    ]));

    let goal_cfg = carnot::GoalConfig {
        criterion: Some("Ensure answer.txt contains 42 and marker.txt contains verified.".into()),
        verify_commands: vec!["grep -q '^42$' answer.txt".into()],
        max_iterations: 6,
        stagnation_limit: 3,
        ..carnot::GoalConfig::default()
    };
    let handle = carnot::GoalHandle::new(goal_cfg);

    let mut cfg1 = CarnotConfig::new(dir1.clone());
    cfg1.modules.sandbox = false;
    let agent1 = CarnotBuilder::new(dir1.clone())
        .config(cfg1)
        .provider(Arc::clone(&worker_faux) as Arc<dyn carnot::Provider>)
        .persist_session(false)
        .goal_handle(handle.clone())
        .goal_verifier_provider(
            Arc::clone(&verifier_faux) as Arc<dyn carnot::Provider>,
            "faux-verifier",
        )
        .build()
        .expect("build goal agent");

    // Baseline agent WITHOUT GoalModule in a different workspace to prove 0-byte static prefix overhead!
    let mut cfg2 = CarnotConfig::new(dir2.clone());
    cfg2.modules.sandbox = false;
    let agent_plain = CarnotBuilder::new(dir2.clone())
        .config(cfg2)
        .provider(Arc::new(FauxProvider::new(vec![ScriptedTurn::text("ok")])))
        .persist_session(false)
        .build()
        .expect("build plain agent");

    assert_eq!(
        agent1.prefix_hash().await,
        agent_plain.prefix_hash().await,
        "GoalModule must add ZERO bytes to the static cached system prompt prefix"
    );

    let out = agent1
        .prompt("Make answer.txt equal 42 and create marker.txt.")
        .await
        .expect("goal prompt");
    assert!(out.final_text.contains("Added marker.txt"));

    let status = handle.status();
    assert_eq!(status.status, carnot::GoalStatus::Met);
    assert_eq!(
        status.iterations, 2,
        "expected 1 deterministic retry + 1 verifier retry"
    );

    // Verifier MUST only have been called twice (skipped when deterministic check failed on Attempt 1),
    // with 0 tools, 1 clean user message, and skip_tail_cache_write = true.
    let vreqs = verifier_faux.recorded_requests();
    assert_eq!(
        vreqs.len(),
        2,
        "verifier should be skipped on deterministic check failure"
    );
    for vr in &vreqs {
        assert!(vr.tools.is_empty(), "verifier must have zero tools");
        assert_eq!(
            vr.messages.len(),
            1,
            "verifier must run in virgin single-turn context"
        );
        assert!(
            vr.skip_tail_cache_write,
            "verifier must set skip_tail_cache_write = true"
        );
    }

    // Worker must have received the `[carnot:goal]` dynamic user prompt note & continuation messages.
    let wreqs = worker_faux.recorded_requests();
    assert_eq!(wreqs.len(), 6);
    assert!(msgs_json(&wreqs[0].messages).contains("[carnot:goal]"));
    assert!(msgs_json(&wreqs[1].messages).contains("Deterministic verification command failed"));
    assert!(msgs_json(&wreqs[4].messages).contains("Also create marker.txt"));

    // Acyclic drop check.
    let weak = Arc::downgrade(&agent1);
    drop(agent1);
    assert!(
        weak.upgrade().is_none(),
        "Agent with GoalModule must drop cleanly with zero Arc cycles"
    );

    let _ = fs::remove_dir_all(&dir1);
    let _ = fs::remove_dir_all(&dir2);
}
