//! End-to-end CLI integration tests for production subcommands (`doctor`, `audit-prefix`,
//! `providers`, `modules`, `--version`, and `--prompt-file` error handling).

use std::process::Command;

fn carnot_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_carnot"))
}

#[test]
fn cli_audit_prefix_and_doctor_return_valid_json() {
    let audit = carnot_bin()
        .arg("audit-prefix")
        .output()
        .expect("run audit-prefix");
    assert!(audit.status.success(), "audit-prefix failed: {audit:?}");
    let audit_val: serde_json::Value =
        serde_json::from_slice(&audit.stdout).expect("valid audit-prefix JSON");
    assert_eq!(audit_val["status"], "pass");
    assert_eq!(audit_val["tools_count"], 5);

    let doctor = carnot_bin().arg("doctor").output().expect("run doctor");
    assert!(doctor.status.success(), "doctor failed: {doctor:?}");
    let doc_val: serde_json::Value =
        serde_json::from_slice(&doctor.stdout).expect("valid doctor JSON");
    assert_eq!(doc_val["status"], "ok");
    assert_eq!(doc_val["prefix_invariance"]["pass"], true);
    assert!(doc_val["sandbox"]["backend"].is_string());
}

#[test]
fn cli_providers_and_modules_succeed() {
    let prov = carnot_bin()
        .arg("providers")
        .output()
        .expect("run providers");
    assert!(prov.status.success());
    let pv: serde_json::Value = serde_json::from_slice(&prov.stdout).expect("valid providers JSON");
    assert!(
        pv["registered_api_adapters"]
            .as_array()
            .is_some_and(|a| a.len() >= 6)
    );

    let mods = carnot_bin().arg("modules").output().expect("run modules");
    assert!(mods.status.success());
    let mv: serde_json::Value = serde_json::from_slice(&mods.stdout).expect("valid modules JSON");
    assert_eq!(mv["profile"], "carnot");
}

#[test]
fn cli_missing_prompt_file_exits_with_code_2() {
    let out = carnot_bin()
        .args([
            "--no-session",
            "--provider",
            "faux",
            "--prompt-file",
            "/nonexistent/carnot-missing-prompt-12345.txt",
        ])
        .output()
        .expect("run missing prompt-file");
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn cli_acp_stdio_end_to_end_initialize_new_prompt_and_close() {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = carnot_bin()
        .args([
            "--mode",
            "acp",
            "--provider",
            "faux",
            "--no-sandbox",
            "--no-session",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn acp");

    {
        let stdin = child.stdin.as_mut().expect("stdin");
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":1}}}}"#
        )
        .expect("init");
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":2,"method":"session/new","params":{{}}}}"#
        )
        .expect("new");
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{{"prompt":[{{"type":"text","text":"ping"}},{{"type":"image","mimeType":"image/png","data":"iVBORw0KGgo="}}]}}}}"#
        )
        .expect("prompt");
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":4,"method":"session/close","params":{{}}}}"#
        )
        .expect("close");
    }

    let out = child.wait_with_output().expect("wait acp");
    assert!(out.status.success(), "acp exited with {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let frames: Vec<serde_json::Value> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid JSON-RPC frame"))
        .collect();

    let init_resp = frames
        .iter()
        .find(|f| f.get("id") == Some(&serde_json::json!(1)))
        .expect("init response");
    assert_eq!(init_resp["result"]["agentInfo"]["name"], "carnot");
    assert_eq!(
        init_resp["result"]["agentCapabilities"]["promptCapabilities"]["image"],
        true
    );

    let new_resp = frames
        .iter()
        .find(|f| f.get("id") == Some(&serde_json::json!(2)))
        .expect("session/new response");
    assert!(new_resp["result"]["sessionId"].is_string());
    assert!(new_resp["result"]["configOptions"].is_array());

    let prompt_resp = frames
        .iter()
        .find(|f| f.get("id") == Some(&serde_json::json!(3)))
        .expect("session/prompt response");
    assert_eq!(prompt_resp["result"]["stopReason"], "end_turn");

    assert!(
        frames.iter().any(|f| {
            f.get("method").and_then(|m| m.as_str()) == Some("session/update")
                && f.pointer("/params/update/sessionUpdate")
                    .and_then(|v| v.as_str())
                    == Some("agent_message_chunk")
        }),
        "expected streamed agent_message_chunk frame"
    );
}

#[test]
fn cli_rpc_stdio_end_to_end_ping_state_prompt_and_undo() {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = carnot_bin()
        .args([
            "--mode",
            "rpc",
            "--provider",
            "faux",
            "--no-sandbox",
            "--no-session",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn rpc");

    {
        let stdin = child.stdin.as_mut().expect("stdin");
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"ping","params":{{}}}}"#
        )
        .expect("ping");
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":2,"method":"get_state","params":{{}}}}"#
        )
        .expect("get_state");
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":3,"method":"undo","params":{{}}}}"#
        )
        .expect("undo");
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":4,"method":"prompt","params":{{"message":"hello rpc"}}}}"#
        )
        .expect("prompt");
    }

    let out = child.wait_with_output().expect("wait rpc");
    assert!(out.status.success(), "rpc exited with {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let frames: Vec<serde_json::Value> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("valid JSON-RPC frame"))
        .collect();

    let ping = frames
        .iter()
        .find(|f| f.get("id") == Some(&serde_json::json!(1)))
        .expect("ping resp");
    assert_eq!(ping["result"]["pong"], true);

    let state = frames
        .iter()
        .find(|f| f.get("id") == Some(&serde_json::json!(2)))
        .expect("state resp");
    assert!(state["result"]["prefix_hash"].is_string());

    let undo = frames
        .iter()
        .find(|f| f.get("id") == Some(&serde_json::json!(3)))
        .expect("undo resp");
    assert!(undo["result"]["reverted"].is_null());

    let prompt = frames
        .iter()
        .find(|f| f.get("id") == Some(&serde_json::json!(4)))
        .expect("prompt resp");
    assert!(prompt["result"]["final_text"].is_string());
}

#[test]
fn cli_dropin_file_extension_loads_in_doctor_and_executes_prompt() {
    let dir = std::env::temp_dir().join(format!("carnot-cli-ext-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let script = dir.join("demo_ext.py");
    std::fs::write(
        &script,
        r#"#!/usr/bin/env python3
import json, sys

for line in sys.stdin:
    req = json.loads(line)
    rid = req.get("id")
    m = req.get("method")
    if m == "ext/init":
        res = {
            "protocol": "carnot-ext/1",
            "hooks": ["before_tool"],
            "tools": [{
                "name": "demo_ping",
                "description": "Returns pong",
                "input_schema": {"type": "object", "properties": {}}
            }]
        }
    elif m == "ext/before_tool":
        res = {"decision": "allow"}
    elif m == "ext/tool_call":
        res = {"text": "pong", "is_error": False}
    else:
        res = {}
    if rid is not None:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": res}) + "\n")
        sys.stdout.flush()
"#,
    )
    .expect("write dropin script");

    let doctor = carnot_bin()
        .args(["--cwd"])
        .arg(&dir)
        .args(["--ext"])
        .arg(&script)
        .arg("doctor")
        .output()
        .expect("run doctor with --ext");
    assert!(doctor.status.success(), "doctor failed: {doctor:?}");
    let doc_val: serde_json::Value =
        serde_json::from_slice(&doctor.stdout).expect("valid doctor JSON");
    let dropins = doc_val["extensions"]["dropin"]
        .as_array()
        .expect("extensions.dropin array");
    assert!(
        dropins
            .iter()
            .any(|d| d["name"] == "demo_ext" && d["source"] == "cli"),
        "expected demo_ext in doctor output: {doc_val}"
    );

    let run = carnot_bin()
        .args(["--cwd"])
        .arg(&dir)
        .args(["--ext"])
        .arg(&script)
        .args([
            "--provider",
            "faux",
            "--no-sandbox",
            "--no-session",
            "-p",
            "hello dropin",
        ])
        .output()
        .expect("run faux prompt with --ext");
    assert!(run.status.success(), "prompt with --ext failed: {run:?}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_auth_login_status_and_logout_with_custom_profile() {
    use std::io::Write;
    use std::process::Stdio;

    let dir = std::env::temp_dir().join(format!("carnot-cli-auth-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let auth_file = dir.join("auth.json");
    let custom_cfg = dir.join("providers.json");
    std::fs::write(
        &custom_cfg,
        r#"{
          "providers": {
            "custom-oauth-gw": {
              "api": "openai-completions",
              "base_url": "https://gw.example.test/v1",
              "auth_chain": [{ "scheme": "oauth", "profile": "shared-gw-profile" }]
            }
          }
        }"#,
    )
    .expect("write custom providers.json");

    // 1. Login via --token-stdin stores under custom `profile` ("shared-gw-profile")
    let mut child = carnot_bin()
        .env("CARNOT_AUTH_FILE", &auth_file)
        .args(["--provider-config"])
        .arg(&custom_cfg)
        .args(["auth", "login", "custom-oauth-gw", "--token-stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn auth login");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(b"bearer-secret-xyz\n")
        .expect("write token");
    let login_out = child.wait_with_output().expect("wait login");
    assert!(login_out.status.success(), "login failed: {login_out:?}");

    // 2. `carnot auth status` reports configured via oauth:shared-gw-profile without leaking secret!
    let status_out = carnot_bin()
        .env("CARNOT_AUTH_FILE", &auth_file)
        .args(["--provider-config"])
        .arg(&custom_cfg)
        .args(["auth", "status"])
        .output()
        .expect("auth status");
    assert!(status_out.status.success());
    let stdout_str = String::from_utf8_lossy(&status_out.stdout);
    assert!(
        !stdout_str.contains("bearer-secret-xyz"),
        "secret must NEVER appear in auth status output"
    );
    let v: serde_json::Value = serde_json::from_str(&stdout_str).expect("valid JSON");
    let row = v["providers"]
        .as_array()
        .expect("providers array")
        .iter()
        .find(|p| p["provider"] == "custom-oauth-gw")
        .expect("custom provider row");
    assert_eq!(row["oauth_profile"], "shared-gw-profile");
    assert_eq!(row["configured"], true);
    assert_eq!(row["auth_source"], "oauth:shared-gw-profile");
    assert_eq!(row["oauth_stored"], true);

    // 3. `carnot auth logout` removes the custom profile cleanly
    let logout_out = carnot_bin()
        .env("CARNOT_AUTH_FILE", &auth_file)
        .args(["--provider-config"])
        .arg(&custom_cfg)
        .args(["auth", "logout", "custom-oauth-gw"])
        .output()
        .expect("auth logout");
    assert!(logout_out.status.success());

    let status_after = carnot_bin()
        .env("CARNOT_AUTH_FILE", &auth_file)
        .args(["--provider-config"])
        .arg(&custom_cfg)
        .args(["auth", "status"])
        .output()
        .expect("auth status after logout");
    let v2: serde_json::Value = serde_json::from_slice(&status_after.stdout).expect("valid JSON");
    let row2 = v2["providers"]
        .as_array()
        .expect("providers array")
        .iter()
        .find(|p| p["provider"] == "custom-oauth-gw")
        .expect("custom provider row");
    assert_eq!(row2["configured"], false);
    assert_eq!(row2["oauth_stored"], false);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_models_list_and_resolve_family_alias() {
    let dir = std::env::temp_dir().join(format!("carnot-cli-models-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");

    let list_out = carnot_bin()
        .env("HOME", &dir)
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .args(["models", "--json", "--all"])
        .output()
        .expect("carnot models --json --all");
    assert!(
        list_out.status.success(),
        "models list failed: {list_out:?}"
    );
    let v: serde_json::Value = serde_json::from_slice(&list_out.stdout).expect("valid JSON");
    let models = v["models"].as_array().expect("models array");
    assert!(
        models
            .iter()
            .any(|m| m["provider"] == "anthropic" && m["id"] == "claude-sonnet-5-5"),
        "expected claude-sonnet-5-5 in models catalog"
    );

    let res_out = carnot_bin()
        .env("HOME", &dir)
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .args(["models", "--resolve", "anthropic/sonnet"])
        .output()
        .expect("carnot models --resolve");
    assert!(res_out.status.success(), "resolve failed: {res_out:?}");
    let resolved = String::from_utf8_lossy(&res_out.stdout).trim().to_string();
    assert_eq!(resolved, "anthropic/claude-sonnet-5-5");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_rpc_host_extension_bidirectional_handshake_tools_hooks_and_controls() {
    use std::io::{BufRead, BufReader, Write};
    use std::process::Stdio;

    let dir = std::env::temp_dir().join(format!("carnot-cli-hostext-{}", std::process::id()));
    let sessions_dir = dir.join("sessions");
    let _ = std::fs::create_dir_all(&sessions_dir);

    let mut child = carnot_bin()
        .args([
            "--cwd",
            &dir.to_string_lossy(),
            "--provider",
            "faux",
            "--mode",
            "rpc",
            "--host-extension",
            "--no-sandbox",
            "--no-context-files",
            "--no-skills",
            "--exclude-tools",
            "Bash,Write",
            "--system-prompt",
            "Custom frozen host system prompt.",
            "--session-dir",
            &sessions_dir.to_string_lossy(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn carnot rpc --host-extension");

    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    let mut send = |v: serde_json::Value| {
        writeln!(stdin, "{v}").expect("write jsonrpc");
        stdin.flush().expect("flush");
    };

    // 1. Handshake: host/register with custom tool + hooks
    send(serde_json::json!({
        "jsonrpc": "2.0",
        "id": "reg-1",
        "method": "host/register",
        "params": {
            "tools": [{
                "name": "host_echo",
                "description": "Echoes a message from the host",
                "input_schema": {
                    "type": "object",
                    "properties": { "text": { "type": "string" } },
                    "required": ["text"]
                }
            }],
            "system_prompt_append": "Frozen host appendix.",
            "hooks": ["gate", "before_prompt", "turn_end", "before_compact"]
        }
    }));

    let mut line = String::new();
    let reg_resp: serde_json::Value = loop {
        line.clear();
        assert!(reader.read_line(&mut line).expect("read") > 0);
        let v: serde_json::Value = serde_json::from_str(&line).expect("json");
        if v.get("id") == Some(&serde_json::json!("reg-1")) {
            break v;
        }
    };
    let active_tools = reg_resp["result"]["tools"]["active"]
        .as_array()
        .expect("active tools");
    let tool_names: Vec<&str> = active_tools.iter().filter_map(|v| v.as_str()).collect();
    assert!(
        tool_names.contains(&"host_echo"),
        "missing host_echo: {tool_names:?}"
    );
    assert!(
        !tool_names.contains(&"Bash"),
        "Bash should be excluded: {tool_names:?}"
    );
    assert!(
        !tool_names.contains(&"Write"),
        "Write should be excluded: {tool_names:?}"
    );
    assert!(
        reg_resp["result"]["session_file"]
            .as_str()
            .is_some_and(|p| p.contains("sessions")),
        "expected session_file inside --session-dir: {reg_resp}"
    );

    // 2. append_entry + get_entries + get_session_stats + set_effort
    send(serde_json::json!({
        "jsonrpc": "2.0",
        "id": "app-1",
        "method": "append_entry",
        "params": { "kind": "host_snapshot", "data": { "v": 1 } }
    }));
    send(serde_json::json!({
        "jsonrpc": "2.0",
        "id": "eff-1",
        "method": "set_effort",
        "params": { "effort": "medium", "scoped": false }
    }));
    send(serde_json::json!({
        "jsonrpc": "2.0",
        "id": "prompt-1",
        "method": "prompt",
        "params": { "message": "hello bidirectional host" }
    }));

    let mut saw_before_prompt = false;
    let mut saw_turn_end = false;
    let mut saw_prompt_done = false;
    let mut saw_append_ok = false;
    let mut saw_effort_ok = false;

    while !saw_prompt_done {
        line.clear();
        let n = reader.read_line(&mut line).expect("read");
        assert!(n > 0, "EOF before prompt-1 completed");
        let v: serde_json::Value = serde_json::from_str(&line).expect("json");

        // Handle engine -> host callbacks!
        if let Some(method) = v.get("method").and_then(|m| m.as_str()) {
            let cid = v.get("id").cloned();
            match method {
                "host/before_prompt" => {
                    saw_before_prompt = true;
                    send(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": cid,
                        "result": {
                            "injections": [{ "text": "[host-context] injected cleanly", "ephemeral": false }]
                        }
                    }));
                }
                "host/turn_end" => {
                    saw_turn_end = true;
                    send(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": cid,
                        "result": { "action": "continue" }
                    }));
                }
                "host/gate" => {
                    send(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": cid,
                        "result": { "decision": "allow" }
                    }));
                }
                _ => {}
            }
            continue;
        }

        if v.get("id") == Some(&serde_json::json!("app-1")) {
            assert!(v["result"]["id"].is_string());
            saw_append_ok = true;
        } else if v.get("id") == Some(&serde_json::json!("eff-1")) {
            assert_eq!(v["result"]["effort"], "medium");
            saw_effort_ok = true;
        } else if v.get("id") == Some(&serde_json::json!("prompt-1")) {
            assert!(v["result"]["final_text"].is_string(), "prompt error: {v}");
            assert_eq!(v["result"]["metrics"]["cache_breaks"], 0);
            saw_prompt_done = true;
        }
    }

    assert!(saw_append_ok);
    assert!(saw_effort_ok);
    assert!(saw_before_prompt, "expected host/before_prompt callback");
    assert!(saw_turn_end, "expected host/turn_end callback");

    drop(stdin);
    let status = child.wait().expect("wait");
    assert!(status.success(), "carnot exited with {status}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_rpc_host_extension_scripted_tool_calls_ephemeral_cache_invariance_and_continue_run() {
    use std::io::{BufRead, BufReader, Write};
    use std::process::Stdio;

    fn write_line(stdin: &mut std::process::ChildStdin, raw: &str) {
        stdin.write_all(raw.as_bytes()).expect("write");
        stdin.write_all(b"\n").expect("nl");
        stdin.flush().expect("flush");
    }

    let dir = std::env::temp_dir().join(format!("carnot-cli-scripted-{}", std::process::id()));
    let sessions_dir = dir.join("sessions");
    let _ = std::fs::create_dir_all(&sessions_dir);

    // 3 scripted turns:
    //   Turn 1: invokes host tool `host_echo` (allowed) + built-in `Bash` (denied by host gate)
    //   Turn 2: returns text, host/turn_end requests `continue_run` once
    //   Turn 3: final settled answer
    let script = serde_json::json!([
        {
            "text": "Calling tools.",
            "tool_calls": [
                {"id": "tc-echo", "name": "host_echo", "input": {"msg": "ping"}},
                {"id": "tc-bash", "name": "Bash", "input": {"command": "echo forbidden"}}
            ]
        },
        {"text": "First draft."},
        {"text": "Final answer after continue_run."}
    ]);

    let mut child = carnot_bin()
        .env("CARNOT_FAUX_SCRIPT", script.to_string())
        .args([
            "--cwd",
            &dir.to_string_lossy(),
            "--provider",
            "faux",
            "--mode",
            "rpc",
            "--host-extension",
            "--no-mcp",
            "--no-sandbox",
            "--no-context-files",
            "--no-skills",
            "--system-prompt",
            "Frozen host prompt.",
            "--session-dir",
            &sessions_dir.to_string_lossy(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scripted carnot");

    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    let mut send = |val: serde_json::Value| {
        let s = serde_json::to_string(&val).expect("serialize");
        stdin.write_all(s.as_bytes()).expect("write");
        stdin.write_all(b"\n").expect("nl");
        stdin.flush().expect("flush");
    };

    send(serde_json::json!({
        "jsonrpc": "2.0",
        "id": "reg-1",
        "method": "host/register",
        "params": {
            "tools": [{
                "name": "host_echo",
                "description": "Echo msg",
                "input_schema": {
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                    "required": ["msg"],
                    "additionalProperties": false
                }
            }],
            "hooks": ["gate", "tool_result", "before_prompt", "turn_end"]
        }
    }));

    let mut line = String::new();
    let reg_v: serde_json::Value = loop {
        line.clear();
        assert!(reader.read_line(&mut line).expect("read") > 0);
        let v: serde_json::Value = serde_json::from_str(&line).expect("reg json");
        if v.get("id") == Some(&serde_json::json!("reg-1")) {
            break v;
        }
    };
    let reg_tools = reg_v["result"]["tools"]["active"]
        .as_array()
        .expect("active tools");
    assert!(reg_tools.iter().any(|t| t == "host_echo"));

    // Send malformed JSON first to verify -32700 parse error
    {
        drop(send);
        write_line(&mut stdin, "{bad json");
    }
    let mut send = |val: serde_json::Value| {
        let s = serde_json::to_string(&val).expect("serialize");
        write_line(&mut stdin, &s);
    };

    // Send prompt with `ack: true`
    send(serde_json::json!({
        "jsonrpc": "2.0",
        "id": "p-ack",
        "method": "prompt",
        "params": { "message": "go", "ack": true }
    }));

    let mut saw_parse_error = false;
    let mut saw_ack = false;
    let mut saw_gate_call_ids = Vec::new();
    let mut saw_host_tool_call = false;
    let mut saw_tool_result_hook = false;
    let mut saw_continue_run_sent = false;
    let mut saw_cache_break = false;
    let final_text: String;

    loop {
        line.clear();
        let n = reader.read_line(&mut line).expect("read");
        assert!(n > 0, "unexpected EOF before run_settled");
        let v: serde_json::Value = serde_json::from_str(&line).expect("valid json");

        if v.get("error").is_some() && v["error"]["code"] == -32700 {
            saw_parse_error = true;
            continue;
        }
        if v.get("id") == Some(&serde_json::json!("p-ack")) {
            assert_eq!(v["result"]["accepted"], true);
            saw_ack = true;
            continue;
        }
        if v.get("method") == Some(&serde_json::json!("event")) {
            let ev = &v["params"];
            if ev["event"] == "cache_break_detected" {
                saw_cache_break = true;
            }
            if ev["event"] == "run_settled" {
                final_text = ev["final_text"].as_str().unwrap_or("").to_string();
                break;
            }
            continue;
        }

        if let Some(method) = v.get("method").and_then(|m| m.as_str()) {
            let cid = v["id"].clone();
            let params = &v["params"];
            match method {
                "host/before_prompt" => {
                    send(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": cid,
                        "result": {
                            "injections": [
                                {"text": "[durable] remember X", "ephemeral": false},
                                {"text": "[ephemeral] stay concise", "ephemeral": true}
                            ]
                        }
                    }));
                }
                "host/gate" => {
                    let call_id = params["call_id"].as_str().unwrap_or("").to_string();
                    saw_gate_call_ids.push(call_id);
                    if params["tool"] == "Bash" {
                        send(serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": cid,
                            "result": {"decision": "deny", "reason": "bash blocked"}
                        }));
                    } else {
                        send(serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": cid,
                            "result": {"decision": "allow"}
                        }));
                    }
                }
                "host/tool_call" => {
                    assert_eq!(params["call_id"], "tc-echo");
                    assert_eq!(params["name"], "host_echo");
                    assert_eq!(params["input"]["msg"], "ping");
                    saw_host_tool_call = true;
                    send(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": cid,
                        "result": {
                            "content": [{"type": "text", "text": "pong:ping"}],
                            "is_error": false
                        }
                    }));
                }
                "host/tool_result" => {
                    saw_tool_result_hook = true;
                    send(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": cid,
                        "result": {"append_text": "[checked]"}
                    }));
                }
                "host/turn_end" => {
                    assert!(
                        params.get("stop_reason").is_some(),
                        "missing stop_reason: {params}"
                    );
                    if params["stop_reason"] == "end_turn" && !saw_continue_run_sent {
                        saw_continue_run_sent = true;
                        send(serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": cid,
                            "result": {
                                "action": "continue_run",
                                "signal": "open_items",
                                "message": "Finish remaining item."
                            }
                        }));
                    } else {
                        send(serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": cid,
                            "result": {"action": "continue"}
                        }));
                    }
                }
                _ => {}
            }
        }
    }

    assert!(saw_parse_error, "expected -32700 parse error");
    assert!(saw_ack, "expected immediate ack: true response");
    assert!(saw_gate_call_ids.iter().any(|c| c == "tc-echo"));
    assert!(saw_gate_call_ids.iter().any(|c| c == "tc-bash"));
    assert!(saw_host_tool_call, "expected host/tool_call execution");
    assert!(saw_tool_result_hook, "expected host/tool_result hook");
    assert!(saw_continue_run_sent, "expected continue_run from turn_end");
    assert!(
        !saw_cache_break,
        "ephemeral prompt injection must NEVER trigger CacheBreakDetected across multi-tool turns!"
    );
    assert_eq!(final_text, "Final answer after continue_run.");

    drop(stdin);
    let status = child.wait().expect("wait");
    assert!(status.success(), "carnot exited with {status}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_user_config_toml_applies_defaults_and_flags_override() {
    let dir = std::env::temp_dir().join(format!("carnot-cfg-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    let cfg_path = dir.join("config.toml");
    std::fs::write(
        &cfg_path,
        r#"
provider = "openai"
model = "gpt-5.4"
effort = "medium"
profile = "minimal"
cache_ttl = "5m"
"#,
    )
    .expect("write config");

    // Config sets profile = "minimal" and model = "gpt-5.4".
    let mods = carnot_bin()
        .env("CARNOT_CONFIG", &cfg_path)
        .arg("modules")
        .output()
        .expect("modules");
    assert!(mods.status.success(), "modules failed: {mods:?}");
    let mv: serde_json::Value = serde_json::from_slice(&mods.stdout).expect("json");
    assert_eq!(mv["profile"], "minimal");

    // Explicit CLI flag overrides config.toml.
    let mods_override = carnot_bin()
        .env("CARNOT_CONFIG", &cfg_path)
        .args(["--profile", "standard", "modules"])
        .output()
        .expect("modules override");
    assert!(mods_override.status.success());
    let mvo: serde_json::Value = serde_json::from_slice(&mods_override.stdout).expect("json");
    assert_eq!(mvo["profile"], "standard");

    // `doctor` reflects config model & reports config_file.
    let doc = carnot_bin()
        .env("CARNOT_CONFIG", &cfg_path)
        .arg("doctor")
        .output()
        .expect("doctor");
    assert!(doc.status.success());
    let dv: serde_json::Value = serde_json::from_slice(&doc.stdout).expect("json");
    assert_eq!(dv["model"], "gpt-5.4");
    assert_eq!(
        dv["config_file"].as_str(),
        Some(cfg_path.to_string_lossy().as_ref())
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cli_continue_resumes_latest_workspace_session_and_piped_stdin_plus_image_work() {
    use std::io::Write;
    use std::process::Stdio;

    let dir = std::env::temp_dir().join(format!("carnot-unix-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let ws = dir.join("ws");
    let sdir = dir.join("sessions");
    std::fs::create_dir_all(&ws).expect("mkdir ws");
    std::fs::create_dir_all(&sdir).expect("mkdir sdir");

    let ws_s = ws.to_str().expect("ws utf8");
    let sdir_s = sdir.to_str().expect("sdir utf8");

    // 1. `-c` fails cleanly when no session exists yet for this workspace.
    let no_sess = carnot_bin()
        .args([
            "-C",
            ws_s,
            "--session-dir",
            sdir_s,
            "--provider",
            "faux",
            "--no-sandbox",
            "-c",
            "-p",
            "continue please",
        ])
        .output()
        .expect("run -c empty");
    assert_eq!(no_sess.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&no_sess.stderr).contains("no previous session found"),
        "stderr: {}",
        String::from_utf8_lossy(&no_sess.stderr)
    );

    // 2. Run a first turn with piped stdin + `-p` + `--image` (1x1 valid PNG).
    let png_path = dir.join("pixel.png");
    let png_bytes: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
        0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0xD7, 0x63, 0xF8,
        0xCF, 0xC0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D, 0xB0, 0x00, 0x00, 0x00,
        0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];
    std::fs::write(&png_path, png_bytes).expect("write png");

    let mut child = carnot_bin()
        .args([
            "-C",
            ws_s,
            "--session-dir",
            sdir_s,
            "--provider",
            "faux",
            "--no-sandbox",
            "--mode",
            "json",
            "--image",
            png_path.to_str().expect("png utf8"),
            "-p",
            "inspect piped diff and screenshot",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn first turn");
    child
        .stdin
        .take()
        .expect("stdin pipe")
        .write_all(b"diff --git a/a.rs b/a.rs\n+fn hi() {}\n")
        .expect("write stdin");
    let out1 = child.wait_with_output().expect("wait first turn");
    assert!(
        out1.status.success(),
        "first turn failed: {}",
        String::from_utf8_lossy(&out1.stderr)
    );

    let session_files: Vec<_> = std::fs::read_dir(&sdir)
        .expect("read sdir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .collect();
    assert_eq!(session_files.len(), 1, "expected 1 persisted session");
    let raw_session = std::fs::read_to_string(&session_files[0]).expect("read session");
    assert!(raw_session.contains("inspect piped diff and screenshot"));
    assert!(raw_session.contains("diff --git a/a.rs b/a.rs"));
    assert!(raw_session.contains("image/png"));

    // 3. `-c` resumes that exact session (no second file created).
    let out2 = carnot_bin()
        .args([
            "-C",
            ws_s,
            "--session-dir",
            sdir_s,
            "--provider",
            "faux",
            "--no-sandbox",
            "--mode",
            "json",
            "-c",
            "-p",
            "follow up turn",
        ])
        .output()
        .expect("run -c");
    assert!(
        out2.status.success(),
        "-c failed: {}",
        String::from_utf8_lossy(&out2.stderr)
    );
    let session_files_after: Vec<_> = std::fs::read_dir(&sdir)
        .expect("read sdir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .collect();
    assert_eq!(
        session_files_after.len(),
        1,
        "-c should append to existing session"
    );
    let raw_after = std::fs::read_to_string(&session_files_after[0]).expect("read session");
    assert!(raw_after.contains("follow up turn"));

    // 4. Unsupported `--image` exits with code 2.
    let bad_img = dir.join("notes.txt");
    std::fs::write(&bad_img, "not an image").expect("write txt");
    let out_bad = carnot_bin()
        .args([
            "--no-session",
            "--provider",
            "faux",
            "--no-sandbox",
            "--image",
            bad_img.to_str().expect("bad img utf8"),
            "-p",
            "check",
        ])
        .output()
        .expect("run bad image");
    assert_eq!(out_bad.status.code(), Some(2));

    let _ = std::fs::remove_dir_all(&dir);
}
