//! ACP wire payload builders, event-to-frame mappers, config option serializers, and confined
//! session path resolution.

use super::AcpModelInfo;
use carnot_core::{
    AgentEvent, CacheTtl, CarnotError, ContentBlock, EventEnvelope, ReasoningEffort, Result,
};
use carnot_session::read_session_header;
use serde_json::{Value, json};
use std::path::{Component, Path, PathBuf};

pub(super) fn effort_to_acp_str(e: ReasoningEffort) -> &'static str {
    match e {
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
}

pub(super) fn parse_acp_effort(s: &str) -> Option<ReasoningEffort> {
    match s.trim().to_ascii_lowercase().as_str() {
        "low" => Some(ReasoningEffort::Low),
        "medium" | "med" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" | "extra_high" | "extra-high" => Some(ReasoningEffort::Xhigh),
        "max" | "ultrathink" => Some(ReasoningEffort::Max),
        _ => None,
    }
}

pub(super) fn build_config_options(
    current_model: &str,
    current_effort: ReasoningEffort,
    current_cache_ttl: CacheTtl,
    available_models: &[AcpModelInfo],
) -> (Value, Value) {
    let mut model_choices: Vec<Value> = available_models
        .iter()
        .map(|m| {
            let mut obj = json!({
                "value": m.id,
                "name": m.name,
            });
            if let Some(ref desc) = m.description
                && !desc.is_empty()
                && let Some(map) = obj.as_object_mut()
            {
                map.insert("description".into(), json!(desc));
            }
            obj
        })
        .collect();
    if !available_models.iter().any(|m| m.id == current_model) {
        model_choices.insert(
            0,
            json!({
                "value": current_model,
                "name": current_model,
            }),
        );
    }

    let available_models_v1: Vec<Value> = model_choices
        .iter()
        .map(|c| {
            let mut obj = json!({
                "modelId": c["value"],
                "name": c["name"],
            });
            if let Some(desc) = c.get("description")
                && let Some(map) = obj.as_object_mut()
            {
                map.insert("description".into(), desc.clone());
            }
            obj
        })
        .collect();

    let config_options = json!([
        {
            "id": "model",
            "name": "Model",
            "description": "Active LLM model routed through Carnot's provider catalog.",
            "category": "model",
            "type": "select",
            "currentValue": current_model,
            "options": model_choices,
        },
        {
            "id": "thought_level",
            "name": "Reasoning effort",
            "description": "Thinking / reasoning effort level.",
            "category": "thought_level",
            "type": "select",
            "currentValue": effort_to_acp_str(current_effort),
            "options": [
                { "value": "low", "name": "Low" },
                { "value": "medium", "name": "Medium" },
                { "value": "high", "name": "High" },
                { "value": "xhigh", "name": "Extra High" },
                { "value": "max", "name": "Max" }
            ],
        },
        {
            "id": "cache_ttl",
            "name": "Cache retention",
            "description": "Provider-agnostic prompt-cache retention tier (short, hybrid, or long).",
            "category": "cache_ttl",
            "type": "select",
            "currentValue": current_cache_ttl.as_str(),
            "options": [
                { "value": "hybrid", "name": "Hybrid (static long · tail short)" },
                { "value": "short", "name": "Short (5m / default)" },
                { "value": "long", "name": "Long (1h / 24h extended)" }
            ],
        }
    ]);

    let models_state = json!({
        "currentModelId": current_model,
        "availableModels": available_models_v1,
    });

    (config_options, models_state)
}

pub(super) fn available_commands_frame(session_id: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": {
                "sessionUpdate": "available_commands_update",
                "availableCommands": [
                    {
                        "name": "compact",
                        "description": "Run Carnot cache-warm semantic compaction (preserves verbatim goal & file ledger).",
                        "input": { "hint": "[optional focus instructions]" }
                    }
                ]
            }
        }
    })
}

pub(super) fn map_acp_stop_reason(halted_reason: Option<&str>) -> &'static str {
    let Some(hr) = halted_reason else {
        return "end_turn";
    };
    if hr.contains("MaxTokens") {
        "max_tokens"
    } else if hr.contains("Refusal") {
        "refusal"
    } else if hr.contains("max_turns")
        || hr.contains("cost budget")
        || hr.contains("idle circuit breaker")
    {
        "max_turn_requests"
    } else {
        "end_turn"
    }
}

pub(super) fn list_recent_sessions(
    session_dir: &Path,
    fallback_cwd: &Path,
    limit: usize,
) -> Vec<Value> {
    let Ok(rd) = std::fs::read_dir(session_dir) else {
        return Vec::new();
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .map(|p| {
            let mtime = std::fs::metadata(&p)
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::UNIX_EPOCH);
            (mtime, p)
        })
        .collect();
    files.sort_by_key(|a| std::cmp::Reverse(a.0));
    let mut out = Vec::new();
    for (_, path) in files.into_iter().take(limit) {
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let header = read_session_header(&path);
        let cwd = header
            .as_ref()
            .and_then(|h| h.workspace_root.as_ref())
            .map_or_else(
                || fallback_cwd.display().to_string(),
                |p| p.display().to_string(),
            );
        let short_stem: String = stem.chars().take(8).collect();
        let title = match header {
            Some(ref h) => format!("Carnot ({}) {short_stem}", h.model),
            None => format!("Carnot session {stem}"),
        };
        out.push(json!({
            "sessionId": stem,
            "cwd": cwd,
            "title": title,
        }));
    }
    out
}

/// Strictly confines `session/load` to a single filename inside `session_dir` (blocks `..`, `/`
/// traversal, absolute paths, and symlink escapes).
pub(super) fn resolve_confined_session_path(session_dir: &Path, raw: &str) -> Result<PathBuf> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.contains('\0') {
        return Err(CarnotError::Session("sessionId must be non-empty".into()));
    }
    let p = Path::new(trimmed);
    let mut comps = p.components();
    let Some(Component::Normal(file_os)) = comps.next() else {
        return Err(CarnotError::Session(format!(
            "invalid sessionId `{trimmed}`: path traversal is forbidden"
        )));
    };
    if comps.next().is_some() {
        return Err(CarnotError::Session(format!(
            "invalid sessionId `{trimmed}`: subdirectories are forbidden"
        )));
    }
    let name = file_os.to_string_lossy();
    let filename = if name.ends_with(".jsonl") {
        name.to_string()
    } else {
        format!("{name}.jsonl")
    };
    let candidate = session_dir.join(&filename);
    let canon_dir =
        std::fs::canonicalize(session_dir).unwrap_or_else(|_| session_dir.to_path_buf());
    let canon_file = std::fs::canonicalize(&candidate).map_err(|_| {
        CarnotError::Session(format!("session not found in session_dir: {filename}"))
    })?;
    if !canon_file.starts_with(&canon_dir) {
        return Err(CarnotError::Session(
            "session symlink escapes session_dir".into(),
        ));
    }
    Ok(canon_file)
}

/// Extracts multimodal ACP prompt blocks (`text`, `resource`, `image`) into `Vec<ContentBlock>`.
pub(super) fn extract_acp_prompt_blocks(params: &Value) -> Vec<ContentBlock> {
    if let Some(s) = params.get("prompt").and_then(Value::as_str) {
        return vec![ContentBlock::Text {
            text: s.to_string(),
        }];
    }
    if let Some(arr) = params.get("prompt").and_then(Value::as_array) {
        let mut blocks = Vec::new();
        for b in arr {
            let kind = b.get("type").and_then(Value::as_str).unwrap_or("");
            if kind == "image" {
                let data = b.get("data").and_then(Value::as_str);
                let mime = b
                    .get("mimeType")
                    .or_else(|| b.get("mediaType"))
                    .and_then(Value::as_str)
                    .unwrap_or("image/png");
                if let Some(d) = data
                    && !d.is_empty()
                {
                    blocks.push(ContentBlock::Image {
                        media_type: mime.to_string(),
                        data_base64: d.to_string(),
                    });
                    continue;
                }
                if let Some(uri) = b.get("uri").and_then(Value::as_str)
                    && let Some(rest) = uri.strip_prefix("data:")
                    && let Some((meta, b64)) = rest.split_once(";base64,")
                    && !b64.is_empty()
                {
                    blocks.push(ContentBlock::Image {
                        media_type: if meta.is_empty() {
                            mime.to_string()
                        } else {
                            meta.to_string()
                        },
                        data_base64: b64.to_string(),
                    });
                    continue;
                }
            }
            if let Some(txt) = b.get("text").and_then(Value::as_str) {
                blocks.push(ContentBlock::Text {
                    text: txt.to_string(),
                });
            } else if let Some(res) = b.get("resource") {
                let uri = res.get("uri").and_then(Value::as_str).unwrap_or("resource");
                if let Some(txt) = res.get("text").and_then(Value::as_str) {
                    blocks.push(ContentBlock::Text {
                        text: format!("[embedded:{uri}]\n{txt}"),
                    });
                }
            }
        }
        return blocks;
    }
    if let Some(msg) = params.get("message").and_then(Value::as_str) {
        return vec![ContentBlock::Text {
            text: msg.to_string(),
        }];
    }
    Vec::new()
}

pub(super) fn prompt_blocks_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn acp_tool_kind(tool: &str) -> &'static str {
    match tool {
        "Read" => "read",
        "Edit" | "Write" => "edit",
        "Grep" | "ToolSearch" => "search",
        "Bash" => "execute",
        _ => "other",
    }
}

pub(super) fn format_tool_title(tool: &str, input: &Value) -> String {
    let worker_prefix = input
        .get("_carnot_worker")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(|w| format!("[worker:{w}] "))
        .unwrap_or_default();
    let base = match tool {
        "Read" | "Edit" | "Write" => {
            if let Some(fp) = input.get("file_path").and_then(Value::as_str) {
                format!("{tool} {fp}")
            } else {
                tool.to_string()
            }
        }
        "Bash" => {
            if let Some(cmd) = input.get("command").and_then(Value::as_str) {
                let first = cmd.lines().next().unwrap_or(cmd).trim();
                let clipped: String = first.chars().take(72).collect();
                format!("Bash: {clipped}")
            } else if let Some(job) = input.get("job").and_then(Value::as_str) {
                format!("Bash ({job})")
            } else {
                tool.to_string()
            }
        }
        "Grep" => {
            if let Some(pat) = input.get("pattern").and_then(Value::as_str) {
                let clipped: String = pat.chars().take(48).collect();
                format!("Grep `{clipped}`")
            } else {
                tool.to_string()
            }
        }
        _ => tool.to_string(),
    };
    format!("{worker_prefix}{base}")
}

pub(super) fn extract_tool_locations(input: &Value) -> Option<Vec<Value>> {
    let path = input
        .get("file_path")
        .or_else(|| input.get("path"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?;
    let mut loc = json!({ "path": path });
    if let Some(line) = input
        .get("start_line")
        .or_else(|| input.get("offset"))
        .and_then(Value::as_u64)
        && let Some(obj) = loc.as_object_mut()
    {
        obj.insert("line".into(), json!(line));
    }
    Some(vec![loc])
}

/// Maps `AgentEvent` to standard ACP `sessionUpdate` union (Zed / T3 Code) + `_meta.carnot`.
/// Returns `None` for internal lifecycle/telemetry events to avoid empty-chunk UI spam.
pub(super) fn acp_update_frame(session_id: &str, env: &EventEnvelope) -> Option<Value> {
    let update = match &env.event {
        AgentEvent::TextDelta { delta } if !delta.is_empty() => json!({
            "sessionUpdate": "agent_message_chunk",
            "content": { "type": "text", "text": delta },
        }),
        AgentEvent::ThinkingDelta { delta } if !delta.is_empty() => json!({
            "sessionUpdate": "agent_thought_chunk",
            "content": { "type": "text", "text": delta },
        }),
        AgentEvent::ToolStart {
            call_id,
            tool,
            eager,
            input,
        } => {
            let mut u = json!({
                "sessionUpdate": "tool_call",
                "toolCallId": call_id,
                "title": format_tool_title(tool, input),
                "kind": acp_tool_kind(tool),
                "status": "in_progress",
                "rawInput": input,
                "_meta": { "carnot": { "tool": tool, "eager": eager } },
            });
            if let Some(locs) = extract_tool_locations(input)
                && let Some(obj) = u.as_object_mut()
            {
                obj.insert("locations".into(), json!(locs));
            }
            u
        }
        AgentEvent::ToolUpdate { call_id, message } => json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": call_id,
            "status": "in_progress",
            "content": [{ "type": "content", "content": { "type": "text", "text": message } }],
        }),
        AgentEvent::ToolEnd {
            call_id,
            tool,
            is_error,
            duration_ms,
            output_preview,
        } => {
            let mut u = json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": call_id,
                "title": tool,
                "status": if *is_error { "failed" } else { "completed" },
                "_meta": { "carnot": { "durationMs": duration_ms } },
            });
            if !output_preview.is_empty()
                && let Some(obj) = u.as_object_mut()
            {
                obj.insert(
                    "content".into(),
                    json!([{ "type": "content", "content": { "type": "text", "text": output_preview } }]),
                );
                obj.insert("rawOutput".into(), json!(output_preview));
            }
            u
        }
        AgentEvent::ModuleAdvisory { signal, message } => json!({
            "sessionUpdate": "agent_thought_chunk",
            "content": { "type": "text", "text": format!("[carnot:{signal}] {message}") },
        }),
        _ => return None,
    };

    Some(json!({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "seq": env.seq,
            "update": update,
        }
    }))
}
