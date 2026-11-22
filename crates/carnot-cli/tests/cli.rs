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
