use carnot_protocol::host_ext::HostRegistration;
use serde_json::Value;

#[test]
fn host_contract_fixtures_match_protocol_schema() {
    let raw = include_str!("host_contract_fixtures.json");
    let root: Value = serde_json::from_str(raw).expect("valid contract fixture JSON");
    let hooks = root["hooks"].as_object().expect("hooks object");

    let reg: HostRegistration = serde_json::from_value(hooks["host/register"]["request"].clone())
        .expect("HostRegistration");
    assert_eq!(reg.tools.len(), 1);
    assert_eq!(reg.tools[0].name, "echo_tool");
    assert!(reg.hooks.iter().any(|h| h == "gate"));

    assert_eq!(hooks["host/gate"]["request"]["call_id"], "call-1");
    assert_eq!(hooks["host/gate"]["allow_response"]["decision"], "allow");
    assert_eq!(hooks["host/gate"]["deny_response"]["decision"], "deny");

    assert_eq!(hooks["host/tool_call"]["request"]["call_id"], "call-1");
    assert_eq!(hooks["host/tool_call"]["request"]["name"], "echo_tool");
    assert!(hooks["host/tool_call"]["ok_response"]["content"].is_array());

    assert!(hooks["host/before_prompt"]["response"]["injections"].is_array());
    assert_eq!(
        hooks["host/turn_end"]["continue_run_response"]["action"],
        "continue_run"
    );
    assert_eq!(
        hooks["host/before_compact"]["focus_response"]["action"],
        "add_focus"
    );
    assert_eq!(
        hooks["host/before_compact"]["cancel_response"]["action"],
        "cancel"
    );
}
