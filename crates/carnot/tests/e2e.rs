//! End-to-end integration tests verifying every Carnot architectural & security invariant:
//!   1. Eager read-only execution + disk-coherent Read (unchanged stub, no CAS spill loop on Read)
//!   2. Atomic Edit + silent Tier-0 tree-sitter syntax check + CRLF & mode 0755 preservation (F5/F6)
//!   3. Atomic 256-bit BLAKE3 CAS File Handle spill (>4 KiB Bash output)
//!   4. Verified green-boundary Rewind-Receipt + Preserved-Thinking prefix preservation + read-tracker reset
//!   5. Strict S2 path confinement (`Read`/`Edit`/`Write` cannot escape workspace root)
//!   6. Opus I2 effect-based parallel disjoint `Edit`/`Write` wave + gate-before-eager `RewriteInput`
//!      + malformed tool JSON safe error without execution
//!   7. Hardened Bash: adaptive yield, live partial output, and background grandchild (`sleep 5 &`) no-hang (B1–B6)
//!   8. Opus I4 lossless monotone-`seq` `EventLog` replay + cross-turn prefix monotonicity (I1)

use carnot::{
    CarnotBuilder, CarnotConfig, FauxProvider, Module, Profile, Result, ScriptedTurn, StopReason,
    ToolGateDecision,
};
use carnot_core::ContentBlock;
use serde_json::{Value, json};
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
    fn name(&self) -> &str {
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
                .map(|c| c.contains("rm -rf /"))
                .unwrap_or(false)
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
async fn e2e_read_edit_cas_spill_rewind_and_permission_gate() {
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
        big_log.push_str(&format!(
            "line {i:03}: verbose diagnostic padding content here\n"
        ));
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
        // Turn 4: Run a strict verified test command
        ScriptedTurn::tool_calls(vec![(
            "c6",
            "Bash",
            json!({"command": "python3 -m unittest --help > /dev/null && echo 'OK 1 test passed'"}),
        )]),
        // Turn 5: Final answer after Rewind-Receipt!
        ScriptedTurn::text("Fixed `add` in `lib.rs` and verified."),
    ]));

    let mut cfg = CarnotConfig::new(dir.clone());
    cfg.modules = carnot::ModuleSwitches::from_profile(Profile::Carnot);
    cfg.modules.sandbox = false; // keep CI-portable
    cfg.rewind_min_heat_tokens = 200; // trigger Rewind-Receipt in compact test

    let agent = CarnotBuilder::new(dir.clone())
        .config(cfg)
        .provider(faux.clone())
        .with_module(Arc::new(DenyRmRfAndRewriteReadModule))
        .persist_session(true)
        .build()
        .expect("build agent");

    let outcome = agent.prompt("Fix add() in lib.rs").await.expect("run");
    assert!(outcome.final_text.contains("Fixed `add`"));
    assert_eq!(outcome.metrics.turns, 5);
    assert!(outcome.metrics.eager_tools_started >= 2);
    assert!(outcome.metrics.cas_handles_created >= 1);
    assert_eq!(
        outcome.metrics.rewinds_applied, 1,
        "expected verified green-boundary Rewind-Receipt to fire once"
    );
    assert!(outcome.metrics.heat_shed_tokens > 0);

    // Verify lib.rs was atomically edited
    let updated = fs::read_to_string(dir.join("lib.rs")).expect("read updated");
    assert!(updated.contains("a + b"));

    let reqs = faux.recorded_requests();
    assert_eq!(reqs.len(), 5);

    // Verify gate-before-eager RewriteInput worked: alias_lib.rs returned lib.rs contents!
    let req2_txt = msgs_json(&reqs[1].messages);
    assert!(req2_txt.contains("pub fn add(a: i32, b: i32)"));
    // Verify Read(debug.log) returned line 199 directly (F1: no CAS handle loop on Read!)
    assert!(req2_txt.contains("line 199: verbose diagnostic padding content here"));
    // Verify Bash(cat debug.log) spilled to `.carnot/h/`
    assert!(req2_txt.contains(".carnot/h/"));

    // Verify I1 prefix monotonicity across turns 1..4 (before rewind)
    for t in 1..4 {
        let prev = &reqs[t - 1].messages;
        let curr = &reqs[t].messages;
        assert!(curr.len() > prev.len());
        for i in 0..prev.len() {
            assert_eq!(
                prev[i], curr[i],
                "prefix mutated before rewind at turn {t} msg {i}"
            );
        }
    }

    // Verify request #5 received the rewound, lean context with `[carnot:rewind-receipt]`!
    let last_req = &reqs[4];
    assert_eq!(last_req.messages.len(), 2);
    assert!(
        last_req.messages[1]
            .text_concat()
            .contains("[carnot:rewind-receipt]")
    );
    for r in &reqs {
        assert_eq!(r.prefix_hash, reqs[0].prefix_hash);
    }

    // Verify permission gate denied `rm -rf /` & unchanged stub fired on Turn 2
    let req3_text = msgs_json(&reqs[2].messages);
    assert!(req3_text.contains("blocked dangerous rm -rf /"));
    assert!(req3_text.contains("[unchanged:"));

    // Verify I4 EventLog recorded strictly monotone sequence numbers
    let replayed = agent.event_log().replay_after(0);
    assert!(replayed.len() >= 10);
    for pair in replayed.windows(2) {
        assert_eq!(pair[1].seq, pair[0].seq + 1);
    }

    let _ = fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn e2e_s2_path_confinement_crlf_and_permissions() {
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
        .provider(faux.clone())
        .persist_session(false)
        .build()
        .expect("build");

    let _ = agent
        .prompt("Test security and file fidelity")
        .await
        .expect("run");

    // S2: escape target MUST NOT exist!
    assert!(!escape_target.exists(), "Write escaped workspace root!");
    let req2_txt = msgs_json(&faux.recorded_requests()[1].messages);
    assert!(req2_txt.contains("outside allowed workspace root"));

    // F5: CRLF preserved!
    assert_eq!(
        fs::read_to_string(dir.join("win.txt")).expect("read win"),
        "hi\r\nworld\r\n"
    );

    // F6: 0755 permissions preserved!
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
async fn e2e_i2_parallel_disjoint_edits_and_invalid_json_guard() {
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
        .provider(faux.clone())
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
        // 1) Backgrounded grandchild holding stdout open MUST NOT hang Bash for 5s (B3)!
        ScriptedTurn::tool_calls(vec![(
            "bg0",
            "Bash",
            json!({"command": "echo HI_FAST; sleep 5 &"}),
        )]),
        // 2) Slow command with live partial output -> yields `job-1` with STEP_1 already visible (B2)!
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
        .provider(faux.clone())
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
async fn e2e_n2_max_tokens_pairs_truncated_tool_use_so_session_stays_usable() {
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
        .provider(faux.clone())
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
        .provider(faux.clone())
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
        .provider(faux.clone())
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
        .provider(faux.clone())
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
    let dir = std::env::temp_dir().join(format!("carnot-ts-{}", uuid::Uuid::new_v4()));
    let sdir = dir.join("sessions");
    fs::create_dir_all(&dir).expect("mkdir");

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
        .provider(faux1.clone())
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
        .provider(faux2.clone())
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
        .provider(faux.clone())
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
        .provider(faux.clone())
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
        .provider(faux.clone())
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
        fn name(&self) -> &str {
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
                .map(|m| m.text_concat().contains("CHILD_TASK"))
                .unwrap_or(false);
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
        out.metrics.total_usage.cost_usd >= 0.05,
        "expected rolled-up child cost >= $0.05, got ${:.6}",
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
        fn name(&self) -> &str {
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
        .provider(faux.clone())
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
        .provider(faux.clone())
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
