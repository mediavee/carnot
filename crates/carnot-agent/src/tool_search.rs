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
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub const TOOL_SEARCH_NAME: &str = "ToolSearch";
pub const CALL_DEFERRED_TOOL_NAME: &str = "CallDeferredTool";

/// Filters tool specs for the currently active provider so live cross-provider model switches
/// (Anthropic native `defer_loading` <-> non-Anthropic `ToolSearch` + `CallDeferredTool`)
/// always expose the right deferred-tool interface without mutating `ToolRegistry`.
pub fn effective_tools_for_provider(
    all_specs: &[ToolSpec],
    supports_native_tool_search: bool,
) -> Vec<ToolSpec> {
    if supports_native_tool_search {
        all_specs
            .iter()
            .filter(|s| s.name != TOOL_SEARCH_NAME && s.name != CALL_DEFERRED_TOOL_NAME)
            .cloned()
            .collect()
    } else {
        all_specs.to_vec()
    }
}

/// Builds a compact Turn-1 user-context directory of deferred tools (hybrid Claude Code + Pi):
///   - Built-in deferred tools (`WebFetch`, `WebSearch`, `LspQuery`, ...) are listed by exact name.
///   - MCP servers are grouped as one compact line per server (`name`, tool count, and 1-line
///     description / handshake instructions or sample tool names).
///
/// Returns `None` when no deferred tools are registered (0 tokens).
pub fn format_deferred_tools_directory(
    deferred_tools: &[Arc<dyn Tool>],
    server_descriptions: &BTreeMap<String, String>,
) -> Option<String> {
    if deferred_tools.is_empty() {
        return None;
    }

    let mut builtin: Vec<String> = Vec::new();
    let mut mcp_groups: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for t in deferred_tools {
        let name = t.spec().name;
        if let Some(rest) = name.strip_prefix("mcp__")
            && let Some((server, tool_name)) = rest.split_once("__")
        {
            mcp_groups
                .entry(server.to_string())
                .or_default()
                .push(tool_name.to_string());
        } else {
            builtin.push(name);
        }
    }
    builtin.sort();
    for tools in mcp_groups.values_mut() {
        tools.sort();
    }

    let mut lines = vec![
        "[available-deferred-tools]".to_string(),
        "Load schemas on demand before calling (use `ToolSearch` with `select:Name1,Name2` or keywords + `CallDeferredTool`, or native tool search on Anthropic):".to_string(),
    ];
    if !builtin.is_empty() {
        lines.push(format!("- Built-in: {}", builtin.join(", ")));
    }
    for (server, tool_names) in &mcp_groups {
        let count = tool_names.len();
        let summary = server_descriptions
            .get(server)
            .cloned()
            .or_else(|| {
                // Handle sanitized server names (e.g. `t3-code` vs `t3_code`)
                server_descriptions
                    .iter()
                    .find(|(k, _)| {
                        k.replace(
                            |c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-',
                            "_",
                        ) == *server
                    })
                    .map(|(_, v)| v.clone())
            })
            .unwrap_or_else(|| {
                let sample: Vec<&str> = tool_names.iter().take(4).map(String::as_str).collect();
                if count > 4 {
                    format!("{}, ...", sample.join(", "))
                } else {
                    sample.join(", ")
                }
            });
        lines.push(format!(
            "- MCP `{server}` ({count} {}): {summary}",
            if count == 1 { "tool" } else { "tools" }
        ));
    }

    Some(lines.join("\n"))
}

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
            name: TOOL_SEARCH_NAME.into(),
            description: "Search available deferred tools (MCP, web, LSP, extensions) and return their schemas. Use 'select:ToolA,ToolB' for exact lookup, or regex/keywords/server name. Call matched tools via `CallDeferredTool`.".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["query"],
                "properties": {
                    "query": { "type": "string", "description": "Exact 'select:Name1,Name2' or regex/keywords/server name" },
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

            // Direct exact-selection syntax (`select:WebFetch,WebSearch`) — 1-shot lookup!
            if let Some(rest) = query
                .strip_prefix("select:")
                .or_else(|| query.strip_prefix("SELECT:"))
            {
                let wanted: Vec<String> = rest
                    .split(',')
                    .map(|s| s.trim().to_ascii_lowercase())
                    .filter(|s| !s.is_empty())
                    .collect();
                let mut matched_specs: Vec<&ToolSpec> = Vec::new();
                let mut missing: Vec<String> = Vec::new();
                for w in &wanted {
                    if let Some(spec) = self
                        .deferred_specs
                        .iter()
                        .find(|s| s.name.to_ascii_lowercase() == *w)
                    {
                        if !matched_specs.iter().any(|m| m.name == spec.name) {
                            matched_specs.push(spec);
                        }
                    } else {
                        missing.push(w.clone());
                    }
                }
                if matched_specs.is_empty() {
                    return Ok(ToolOutput::ok_text(format!(
                        "No deferred tools matched `{query}` ({} total deferred tools available).",
                        self.deferred_specs.len()
                    )));
                }
                let matched: Vec<Value> = matched_specs
                    .into_iter()
                    .map(|s| {
                        json!({
                            "name": s.name,
                            "description": s.description,
                            "input_schema": s.input_schema,
                        })
                    })
                    .collect();
                let rendered = serde_json::to_string(&matched).unwrap_or_default();
                let miss_note = if missing.is_empty() {
                    String::new()
                } else {
                    format!(" (not found: {})", missing.join(", "))
                };
                return Ok(ToolOutput::ok_text(format!(
                    "Matched {} deferred tool(s){miss_note}. Invoke via CallDeferredTool{{\"name\":\"...\",\"arguments\":{{...}}}}:\n{rendered}",
                    matched.len()
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
                        let mcp_prefix = format!("mcp__{tok}__");
                        let mcp_prefix_us = format!("mcp__{}__", tok.replace('-', "_"));
                        if low_name.starts_with(&mcp_prefix) || low_name.starts_with(&mcp_prefix_us)
                        {
                            score += 8;
                        }
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
            name: CALL_DEFERRED_TOOL_NAME.into(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct DummyDeferredTool {
        name: &'static str,
        desc: &'static str,
    }

    impl Tool for DummyDeferredTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: self.name.into(),
                description: self.desc.into(),
                input_schema: json!({"type": "object", "properties": {}}),
                defer_loading: true,
            }
        }

        fn execute(
            &self,
            _input: Value,
            _cancel: CancellationToken,
        ) -> BoxFuture<'_, Result<ToolOutput>> {
            Box::pin(async { Ok(ToolOutput::ok_text("ok")) })
        }
    }

    fn sample_deferred() -> Vec<Arc<dyn Tool>> {
        vec![
            Arc::new(DummyDeferredTool {
                name: "WebFetch",
                desc: "Fetch a URL",
            }),
            Arc::new(DummyDeferredTool {
                name: "WebSearch",
                desc: "Search the web",
            }),
            Arc::new(DummyDeferredTool {
                name: "mcp__t3-code__preview_open",
                desc: "Open a preview tab",
            }),
            Arc::new(DummyDeferredTool {
                name: "mcp__t3-code__preview_status",
                desc: "Check preview status",
            }),
        ]
    }

    #[tokio::test]
    async fn select_syntax_returns_exact_tools_in_one_shot() {
        let search = ToolSearchTool::new(&sample_deferred());
        let out = search
            .execute(
                json!({"query": "select:WebFetch,WebSearch"}),
                CancellationToken::new(),
            )
            .await
            .expect("select execute");
        let txt = out.text_only();
        assert!(txt.contains("Matched 2 deferred tool(s)"));
        assert!(txt.contains("\"WebFetch\""));
        assert!(txt.contains("\"WebSearch\""));
        assert!(!txt.contains("preview_open"));
    }

    #[test]
    fn deferred_directory_groups_builtin_and_mcp_servers_compactly() {
        let mut descs = BTreeMap::new();
        descs.insert("t3-code".into(), "T3 Code workspace & browser tools".into());
        let dir = format_deferred_tools_directory(&sample_deferred(), &descs)
            .expect("deferred directory");
        assert!(dir.contains("[available-deferred-tools]"));
        assert!(dir.contains("- Built-in: WebFetch, WebSearch"));
        assert!(dir.contains("- MCP `t3-code` (2 tools): T3 Code workspace & browser tools"));
    }

    #[test]
    fn effective_tools_switches_cleanly_between_native_and_fallback_providers() {
        let mut specs: Vec<ToolSpec> = sample_deferred().iter().map(|t| t.spec()).collect();
        specs.push(ToolSearchTool::new(&sample_deferred()).spec());
        specs.push(CallDeferredTool::new(&sample_deferred()).spec());

        let anthropic_specs = effective_tools_for_provider(&specs, true);
        assert!(!anthropic_specs.iter().any(|s| s.name == TOOL_SEARCH_NAME));
        assert!(
            !anthropic_specs
                .iter()
                .any(|s| s.name == CALL_DEFERRED_TOOL_NAME)
        );
        assert!(anthropic_specs.iter().any(|s| s.name == "WebFetch"));

        let openai_specs = effective_tools_for_provider(&specs, false);
        assert!(openai_specs.iter().any(|s| s.name == TOOL_SEARCH_NAME));
        assert!(
            openai_specs
                .iter()
                .any(|s| s.name == CALL_DEFERRED_TOOL_NAME)
        );
    }
}
