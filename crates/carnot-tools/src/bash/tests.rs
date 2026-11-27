use super::*;

#[tokio::test]
async fn repeated_spilled_bash_output_deduplicates_and_resets_after_clear() {
    let dir = std::env::temp_dir().join(format!("carnot-bash-dedup-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let mut cfg = carnot_core::CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    cfg.cas_spill_threshold_bytes = 256;
    cfg.cas_preview_max_bytes = 128;

    let cas = Arc::new(CasStore::open(&dir).expect("cas"));
    let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
    let bash = reg.get("Bash").expect("Bash");

    let cmd =
        "python3 -c \"print('\\n'.join(f'entry-{i:04d}-payload-abcdefghij' for i in range(40)))\"";
    let first = bash
        .execute(json!({"command": cmd}), CancellationToken::new())
        .await
        .expect("first run");
    assert!(!first.is_error, "first run failed: {}", first.text_only());
    assert_eq!(first.details["deduplicated"], false, "first not deduped");
    assert!(
        first.text_only().contains(".carnot/h/"),
        "expected CAS handle: {}",
        first.text_only()
    );

    let second = bash
        .execute(json!({"command": cmd}), CancellationToken::new())
        .await
        .expect("second run");
    assert!(
        !second.is_error,
        "second run failed: {}",
        second.text_only()
    );
    assert_eq!(second.details["deduplicated"], true, "second deduped");
    assert!(
        second
            .text_only()
            .contains("identical output to earlier run"),
        "got: {}",
        second.text_only()
    );
    assert!(
        second.text_only().len() < first.text_only().len(),
        "dedup receipt must be shorter"
    );

    // Compaction / rewind calls `reset_read_tracker()` which must clear seen CAS hashes too!
    reg.reset_read_tracker();
    let third = bash
        .execute(json!({"command": cmd}), CancellationToken::new())
        .await
        .expect("third run after reset");
    assert_eq!(third.details["deduplicated"], false, "reset clears dedup");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn spilled_bash_output_redacts_secrets_before_writing_to_cas_disk() {
    let dir = std::env::temp_dir().join(format!("carnot-bash-redact-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let mut cfg = carnot_core::CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    cfg.cas_spill_threshold_bytes = 256;
    cfg.cas_preview_max_bytes = 128;
    cfg.redact_cas_secrets = true;

    let cas = Arc::new(CasStore::open(&dir).expect("cas"));
    let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
    let bash = reg.get("Bash").expect("Bash");

    let secret = "sk-ant-api03-secretvalue1234567890abcdef";
    let cmd = format!(
        "python3 -c \"print('TOKEN={secret}'); print('\\n'.join(f'line-{{i:04d}}-padding-abcdef' for i in range(40)))\""
    );
    let out = bash
        .execute(json!({"command": cmd}), CancellationToken::new())
        .await
        .expect("run");
    assert!(!out.is_error, "run failed: {}", out.text_only());
    assert!(
        !out.text_only().contains(secret),
        "secret leaked in preview"
    );
    let rel = out.details["handle"].as_str().expect("spilled CAS handle");
    let on_disk = std::fs::read_to_string(dir.join(rel)).expect("read CAS blob");
    assert!(
        !on_disk.contains(secret),
        "raw secret must NEVER be written to .carnot/h/ on disk"
    );
    assert!(
        on_disk.contains("[REDACTED:ANTHROPIC_KEY]"),
        "expected redaction marker"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn live_progress_sink_receives_incremental_bash_output() {
    let dir = std::env::temp_dir().join(format!("carnot-bash-stream-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let mut cfg = carnot_core::CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;

    let cas = Arc::new(CasStore::open(&dir).expect("cas"));
    let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
    let bash = reg.get("Bash").expect("Bash");

    let updates: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_cb = Arc::clone(&updates);
    let sink: ToolProgressSink = Arc::new(move |msg: String| {
        updates_cb.lock().expect("lock").push(msg);
    });

    let out = bash
        .execute_with_progress(
            "call-stream-1",
            json!({"command": "echo step-one; sleep 0.2; echo step-two"}),
            CancellationToken::new(),
            Some(sink),
        )
        .await
        .expect("run");
    assert!(!out.is_error, "command failed: {}", out.text_only());
    let seen = updates.lock().expect("lock").clone();
    assert!(
        !seen.is_empty(),
        "expected at least one live progress update"
    );
    assert!(
        seen.iter().any(|m| m.contains("step-one")),
        "expected step-one in live updates: {seen:?}"
    );
    assert!(
        seen.iter().any(|m| m.contains("step-two")),
        "expected step-two in live updates: {seen:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn pty_mode_provides_real_tty_and_accepts_interactive_stdin() {
    let dir = std::env::temp_dir().join(format!("carnot-bash-pty-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let mut cfg = carnot_core::CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    cfg.bash_yield_ms = 250;

    let cas = Arc::new(CasStore::open(&dir).expect("cas"));
    let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
    let bash = reg.get("Bash").expect("Bash");

    // 1. Verify PTY allocates a real TTY (`test -t 1`).
    let tty_out = bash
        .execute(
            json!({"command": "test -t 1 && echo IS_TTY || echo NOT_TTY", "pty": true}),
            CancellationToken::new(),
        )
        .await
        .expect("pty tty check");
    assert!(
        !tty_out.is_error,
        "tty check failed: {}",
        tty_out.text_only()
    );
    assert!(
        tty_out.text_only().contains("IS_TTY"),
        "expected real TTY in pty mode, got: {}",
        tty_out.text_only()
    );
    assert_eq!(tty_out.details["pty"], true, "pty flag in details");

    // 2. Interactive command yields while waiting for stdin, then completes once stdin is sent.
    let yielded = bash
        .execute(
            json!({"command": "printf 'PROMPT> '; read line; echo \"GOT:$line\"", "pty": true}),
            CancellationToken::new(),
        )
        .await
        .expect("pty yield");
    assert_eq!(yielded.details["yielded"], true, "expected yielded PTY job");
    let job_id = yielded.details["job_id"]
        .as_str()
        .expect("job_id")
        .to_string();

    let done = bash
        .execute(
            json!({"job": job_id, "stdin": "hello-pty\n", "timeout": 5}),
            CancellationToken::new(),
        )
        .await
        .expect("pty stdin reply");
    assert!(!done.is_error, "pty reply failed: {}", done.text_only());
    assert!(
        done.text_only().contains("GOT:hello-pty"),
        "expected interactive echo, got: {}",
        done.text_only()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn timeout_kills_process_group_via_native_syscall() {
    let dir = std::env::temp_dir().join(format!("carnot-bash-kill-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let mut cfg = carnot_core::CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;

    let cas = Arc::new(CasStore::open(&dir).expect("cas"));
    let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
    let bash = reg.get("Bash").expect("Bash");

    let out = bash
        .execute(
            json!({"command": "sleep 30", "timeout": 1}),
            CancellationToken::new(),
        )
        .await
        .expect("timeout run");
    assert!(out.is_error, "timeout must mark error");
    assert_eq!(out.details["exit_code"], 124, "timeout exit code 124");

    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn pty_mode_works_under_macos_seatbelt_sandbox() {
    let dir = std::env::temp_dir().join(format!("carnot-bash-pty-sb-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let mut cfg = carnot_core::CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = true;
    cfg.modules.sandbox_network = false;

    let cas = Arc::new(CasStore::open(&dir).expect("cas"));
    let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
    let bash = reg.get("Bash").expect("Bash");

    let out = bash
        .execute(
            json!({"command": "test -t 1 && echo SEATBELT_PTY_OK", "pty": true}),
            CancellationToken::new(),
        )
        .await
        .expect("sandboxed pty run");
    assert!(!out.is_error, "sandboxed PTY failed: {}", out.text_only());
    assert!(
        out.text_only().contains("SEATBELT_PTY_OK"),
        "expected sandboxed PTY output, got: {}",
        out.text_only()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn live_progress_sink_redacts_secrets_before_emitting() {
    let dir =
        std::env::temp_dir().join(format!("carnot-bash-live-redact-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let mut cfg = carnot_core::CarnotConfig::new(dir.clone());
    cfg.modules.sandbox = false;
    cfg.redact_cas_secrets = true;

    let cas = Arc::new(CasStore::open(&dir).expect("cas"));
    let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
    let bash = reg.get("Bash").expect("Bash");

    let updates: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let updates_cb = Arc::clone(&updates);
    let sink: ToolProgressSink = Arc::new(move |msg: String| {
        updates_cb.lock().expect("lock").push(msg);
    });

    let secret = "sk-ant-api03-liveprogresssecretvalue1234567890";
    let cmd = format!("printf 'TOKEN=%s\\n' '{secret}'; sleep 0.2; echo done");
    let out = bash
        .execute_with_progress(
            "call-live-redact",
            json!({"command": cmd}),
            CancellationToken::new(),
            Some(sink),
        )
        .await
        .expect("run");
    assert!(!out.is_error, "command failed: {}", out.text_only());
    assert!(
        !out.text_only().contains(secret),
        "final output must redact secret"
    );
    let seen = updates.lock().expect("lock").clone();
    assert!(!seen.is_empty(), "expected live progress updates");
    for msg in &seen {
        assert!(
            !msg.contains(secret),
            "raw secret leaked into live ToolUpdate: {msg}"
        );
    }
    assert!(
        seen.iter().any(|m| m.contains("[REDACTED:ANTHROPIC_KEY]")),
        "expected redacted marker in live updates: {seen:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
