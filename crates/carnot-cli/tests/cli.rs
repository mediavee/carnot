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
    assert!(audit.status.success(), "audit-prefix failed: {:?}", audit);
    let audit_val: serde_json::Value =
        serde_json::from_slice(&audit.stdout).expect("valid audit-prefix JSON");
    assert_eq!(audit_val["status"], "pass");
    assert_eq!(audit_val["tools_count"], 5);

    let doctor = carnot_bin().arg("doctor").output().expect("run doctor");
    assert!(doctor.status.success(), "doctor failed: {:?}", doctor);
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
    assert!(doctor.status.success(), "doctor failed: {:?}", doctor);
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
    assert!(run.status.success(), "prompt with --ext failed: {:?}", run);

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
