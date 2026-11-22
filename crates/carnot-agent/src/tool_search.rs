//! Cache-invariant universal `ToolSearch` + `CallDeferredTool` fallback for non-Anthropic providers
//! (Opus architectural point #1).
//!
//! When a provider does NOT support Anthropic's native `tool_search_tool_regex_20251119` (e.g. OpenAI
//! or local OpenAI-compatible models) and deferred tools (MCP, `WebFetch`, etc.) are registered:
//!   - The static `tools` array stays **100% FIXED** (`Bash`, `Read`, `Edit`, `Write`, `ToolSearch`,
//!     `CallDeferredTool`) so `prefix_hash` and `prompt_cache_key` NEVER mutate!
//!   - `ToolSearch(query)` matches deferred tools via regex/keyword scoring and returns their minified
//!     JSON schemas inside `tool_result` (append-only, cache-safe!).
//!   - `CallDeferredTool(name, arguments)` validates required fields and dispatches to the deferred `Tool`.

use carnot_core::{Result, ToolEffects, ToolOutput, ToolSpec};
use carnot_tools::{BoxFuture, Tool};
use regex::RegexBuilder;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Searches deferred tools (`defer_loading == true`) without mutating the static tool prefix.
#[derive(Debug)]
pub struct ToolSearchTool {
    deferred_specs: Vec<ToolSpec>,
}

impl ToolSearchTool {
    pub fn new(deferred_tools: &[Arc<dyn Tool>]) -> Self {
        let mut deferred_specs: Vec<ToolSpec> = deferred_tools.iter().map(|t| t.spec()).collect();
        deferred_specs.sort_by(|a, b| a.name.cmp(&b.name));
        Self { deferred_specs }
    }
}

impl Tool for ToolSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "ToolSearch".into(),
            description: "Search available deferred tools (MCP, web, extensions) by regex or keywords and return their schemas. Call matched tools via `CallDeferredTool`.".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["query"],
                "properties": {
                    "query": { "type": "string", "description": "Regex or space-separated keywords" },
                    "limit": { "type": "integer", "description": "Max tools to return (default 8)" }
                }
            }),
            defer_loading: false,
        }
    }

    fn effects(&self, _input: &Value) -> ToolEffects {
        ToolEffects::global_read()
    }

    fn execute(
        &self,
        input: Value,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let query = input
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let limit = input
                .get("limit")
                .and_then(Value::as_u64)
                .unwrap_or(8)
                .clamp(1, 32) as usize;

            if query.is_empty() {
                let names: Vec<&str> = self
                    .deferred_specs
                    .iter()
                    .take(limit)
                    .map(|s| s.name.as_str())
                    .collect();
                return Ok(ToolOutput::ok_text(format!(
                    "Available deferred tools ({} total): {}",
                    self.deferred_specs.len(),
                    names.join(", ")
                )));
            }

            let re = RegexBuilder::new(query).case_insensitive(true).build().ok();
            let tokens: Vec<String> = query
                .split_whitespace()
                .map(|t| t.to_ascii_lowercase())
                .collect();

            let mut scored: Vec<(usize, &ToolSpec)> = self
                .deferred_specs
                .iter()
                .filter_map(|spec| {
                    let hay = format!("{} {}", spec.name, spec.description);
                    let mut score = 0usize;
                    if let Some(ref r) = re
                        && r.is_match(&hay)
                    {
                        score += 10;
                    }
                    let low_name = spec.name.to_ascii_lowercase();
                    let low_desc = spec.description.to_ascii_lowercase();
                    for tok in &tokens {
                        if low_name.contains(tok) {
                            score += 5;
                        }
                        if low_desc.contains(tok) {
                            score += 2;
                        }
                    }
                    (score > 0).then_some((score, spec))
                })
                .collect();

            scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
            scored.truncate(limit);

            if scored.is_empty() {
                return Ok(ToolOutput::ok_text(format!(
                    "No deferred tools matched `{query}` ({} total deferred tools available).",
                    self.deferred_specs.len()
                )));
            }

            let matched: Vec<Value> = scored
                .into_iter()
                .map(|(_, s)| {
                    json!({
                        "name": s.name,
                        "description": s.description,
                        "input_schema": s.input_schema,
                    })
                })
                .collect();
            let rendered = serde_json::to_string(&matched).unwrap_or_default();
            Ok(ToolOutput::ok_text(format!(
                "Matched {} deferred tool(s). Invoke via CallDeferredTool{{\"name\":\"...\",\"arguments\":{{...}}}}:\n{rendered}",
                matched.len()
            )))
        })
    }
}

/// Dispatches calls to deferred tools discovered via `ToolSearch` without altering the wire tool list.
#[derive(Debug)]
pub struct CallDeferredTool {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl CallDeferredTool {
    pub fn new(deferred_tools: &[Arc<dyn Tool>]) -> Self {
        let tools = deferred_tools
            .iter()
            .map(|t| (t.spec().name, Arc::clone(t)))
            .collect();
        Self { tools }
    }
}

impl Tool for CallDeferredTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "CallDeferredTool".into(),
            description: "Execute a deferred tool discovered via `ToolSearch`.".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["name", "arguments"],
                "properties": {
                    "name": { "type": "string", "description": "Exact deferred tool name from ToolSearch" },
                    "arguments": { "type": "object", "description": "Tool arguments matching its input_schema" }
                }
            }),
            defer_loading: false,
        }
    }

    fn effects(&self, input: &Value) -> ToolEffects {
        if let Some(name) = input.get("name").and_then(Value::as_str)
            && let Some(t) = self.tools.get(name)
        {
            let args = input.get("arguments").cloned().unwrap_or_else(|| json!({}));
            return t.effects(&args);
        }
        ToolEffects::serial()
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let Some(name) = input.get("name").and_then(Value::as_str) else {
                return Ok(ToolOutput::err_text("CallDeferredTool requires `name`"));
            };
            let Some(tool) = self.tools.get(name) else {
                return Ok(ToolOutput::err_text(format!(
                    "Unknown deferred tool `{name}` — run `ToolSearch` first."
                )));
            };
            let args = input.get("arguments").cloned().unwrap_or_else(|| json!({}));

            // Validate required top-level schema keys locally before dispatch (Opus point #1!)
            let spec = tool.spec();
            if let Some(req_arr) = spec.input_schema.get("required").and_then(Value::as_array) {
                for r in req_arr {
                    if let Some(key) = r.as_str()
                        && args.get(key).is_none()
                    {
                        return Ok(ToolOutput::err_text(format!(
                            "Deferred tool `{name}` missing required argument `{key}`"
                        )));
                    }
                }
            }

            tool.execute(args, cancel).await
        })
    }
}
