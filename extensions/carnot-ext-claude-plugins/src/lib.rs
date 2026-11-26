//! Official Carnot Rust Extension: Native Claude Code Plugin Loader (`ClaudePluginsModule`).
//!
//! Loads standard Claude Code plugin directories (both explicit `--claude-plugin <dir>` paths and
//! opt-in auto-discovered entries from `~/.claude/plugins/installed_plugins.json`), translating
//! every plugin component cleanly into Carnot's cache-invariant architecture:
//!   - **Manifest:** `.claude-plugin/plugin.json` (`name`, `version`, `description`, custom paths)
//!   - **Hooks:** `hooks/hooks.json` (or manifest `hooks`) executed via `carnot-ext-hooks` with
//!     `${CLAUDE_PLUGIN_ROOT}` expansion & `CLAUDE_PLUGIN_ROOT` env var
//!   - **Skills & Commands:** `skills/*/SKILL.md` + `commands/**/*.md` advertised in Turn-1 context
//!     ONLY (0 static prefix tokens!) with absolute paths for on-demand `Read`
//!   - **Agents:** `agents/*.md` frontmatter catalog advertised in Turn-1 context
//!   - **MCP Servers:** `.mcp.json` (or manifest `mcpServers`) with `${CLAUDE_PLUGIN_ROOT}` expanded,
//!     returned via [`ClaudePluginsModule::mcp_servers`] for `McpManager` (`defer_loading: true`)

use carnot_core::{CarnotConfig, CarnotError, Result, ToolOutput};
use carnot_ext::{
    BoundaryAction, BoxFuture, CompactionHookAction, CompactionHookContext, Module,
    ToolGateDecision, TurnBoundarySnapshot,
};
use carnot_ext_hooks::{
    CompiledHook, HooksModule, expand_hook_vars, parse_hooks_file, parse_hooks_value,
};
use carnot_mcp::{McpConfigFile, McpServerConfig};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Discovery configuration for loading Claude Code plugins.
#[derive(Debug, Clone, Default)]
pub struct ClaudePluginsConfig {
    /// Explicit Claude Code plugin root directories (equivalent to Claude Code `--plugin-dir`).
    pub plugin_dirs: Vec<PathBuf>,
    /// Also discover installed enabled plugins from `~/.claude/plugins/installed_plugins.json`
    /// respecting `~/.claude/settings.json` `enabledPlugins`.
    pub load_installed_plugins: bool,
    /// Forward `fail_closed` to plugin `PreToolUse` hooks.
    pub fail_closed_hooks: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PluginManifest {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    hooks: Option<Value>,
    #[serde(default)]
    mcp_servers: Option<Value>,
    #[serde(default)]
    skills: Option<Value>,
    #[serde(default)]
    commands: Option<Value>,
    #[serde(default)]
    agents: Option<Value>,
}

/// Summary of a discovered plugin skill, slash command, or agent prompt file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginAsset {
    pub plugin: String,
    pub kind: String,
    pub name: String,
    pub description: String,
    pub path: PathBuf,
}

/// Summary of one loaded Claude Code plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedClaudePlugin {
    pub name: String,
    pub version: Option<String>,
    pub description: Option<String>,
    pub root: PathBuf,
    pub hook_count: usize,
    pub mcp_server_names: Vec<String>,
    pub assets: Vec<PluginAsset>,
}

fn expand_json_vars(val: &mut Value, project_dir: &Path, plugin_root: &Path) {
    match val {
        Value::String(s) => {
            *s = expand_hook_vars(s, project_dir, Some(plugin_root));
        }
        Value::Array(arr) => {
            for item in arr {
                expand_json_vars(item, project_dir, plugin_root);
            }
        }
        Value::Object(map) => {
            for v in map.values_mut() {
                expand_json_vars(v, project_dir, plugin_root);
            }
        }
        _ => {}
    }
}

/// Parses optional YAML frontmatter (`--- ... ---`) extracting `name` and `description`.
fn parse_markdown_frontmatter(raw: &str) -> (Option<String>, Option<String>) {
    let trimmed = raw.trim_start();
    let Some(rest) = trimmed.strip_prefix("---") else {
        return (None, None);
    };
    let rest = rest
        .strip_prefix("\r\n")
        .or_else(|| rest.strip_prefix('\n'))
        .unwrap_or(rest);
    let Some(end) = rest.find("\n---") else {
        return (None, None);
    };
    let fm = &rest[..end];
    let mut name = None;
    let mut desc = None;
    for line in fm.lines() {
        let l = line.trim();
        if let Some(v) = l.strip_prefix("name:") {
            let clean = v.trim().trim_matches('"').trim_matches('\'').trim();
            if !clean.is_empty() {
                name = Some(clean.to_string());
            }
        } else if let Some(v) = l.strip_prefix("description:") {
            let clean = v.trim().trim_matches('"').trim_matches('\'').trim();
            if !clean.is_empty() {
                desc = Some(clean.to_string());
            }
        }
    }
    (name, desc)
}

fn first_nonempty_line(raw: &str) -> String {
    let body = if let Some(rest) = raw.trim_start().strip_prefix("---")
        && let Some(end) = rest.find("\n---")
    {
        &rest[end + 4..]
    } else {
        raw
    };
    body.lines()
        .map(|l| l.trim().trim_start_matches('#').trim())
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .chars()
        .take(120)
        .collect()
}

fn resolve_manifest_paths(root: &Path, val: Option<&Value>, default_rel: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    match val {
        Some(Value::String(s)) => {
            let p = root.join(s.trim_start_matches("./"));
            if p.exists() {
                out.push(p);
            }
        }
        Some(Value::Array(arr)) => {
            for item in arr {
                if let Some(s) = item.as_str() {
                    let p = root.join(s.trim_start_matches("./"));
                    if p.exists() {
                        out.push(p);
                    }
                }
            }
        }
        _ => {
            let p = root.join(default_rel);
            if p.exists() {
                out.push(p);
            }
        }
    }
    out
}

fn collect_markdown_files(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
    if depth > 4 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            collect_markdown_files(&p, out, depth + 1);
        } else if p.extension().and_then(|e| e.to_str()) == Some("md") {
            out.push(p);
        }
    }
}

fn discover_plugin_assets(
    plugin_name: &str,
    root: &Path,
    manifest: &PluginManifest,
) -> Vec<PluginAsset> {
    let mut assets = Vec::new();

    // 1. Skills: `skills/<name>/SKILL.md`
    for skills_dir in resolve_manifest_paths(root, manifest.skills.as_ref(), "skills") {
        if let Ok(rd) = std::fs::read_dir(&skills_dir) {
            let mut subdirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
            subdirs.sort();
            for sub in subdirs {
                let skill_file = if sub.is_dir() {
                    sub.join("SKILL.md")
                } else if sub.file_name().and_then(|f| f.to_str()) == Some("SKILL.md") {
                    sub
                } else {
                    continue;
                };
                if !skill_file.is_file() {
                    continue;
                }
                let Ok(raw) = std::fs::read_to_string(&skill_file) else {
                    continue;
                };
                let (fm_name, fm_desc) = parse_markdown_frontmatter(&raw);
                let fallback_name = skill_file
                    .parent()
                    .and_then(|p| p.file_name())
                    .and_then(|s| s.to_str())
                    .unwrap_or("skill")
                    .to_string();
                assets.push(PluginAsset {
                    plugin: plugin_name.to_string(),
                    kind: "skill".into(),
                    name: fm_name.unwrap_or(fallback_name),
                    description: fm_desc.unwrap_or_else(|| first_nonempty_line(&raw)),
                    path: skill_file,
                });
            }
        }
    }

    // 2. Commands: `commands/**/*.md`
    for cmd_dir in resolve_manifest_paths(root, manifest.commands.as_ref(), "commands") {
        let mut files = Vec::new();
        if cmd_dir.is_file() {
            files.push(cmd_dir);
        } else {
            collect_markdown_files(&cmd_dir, &mut files, 0);
        }
        for f in files {
            let Ok(raw) = std::fs::read_to_string(&f) else {
                continue;
            };
            let (fm_name, fm_desc) = parse_markdown_frontmatter(&raw);
            let stem = f
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("command")
                .to_string();
            assets.push(PluginAsset {
                plugin: plugin_name.to_string(),
                kind: "command".into(),
                name: fm_name.unwrap_or(stem),
                description: fm_desc.unwrap_or_else(|| first_nonempty_line(&raw)),
                path: f,
            });
        }
    }

    // 3. Agents: `agents/*.md`
    for agent_dir in resolve_manifest_paths(root, manifest.agents.as_ref(), "agents") {
        let mut files = Vec::new();
        if agent_dir.is_file() {
            files.push(agent_dir);
        } else {
            collect_markdown_files(&agent_dir, &mut files, 0);
        }
        for f in files {
            let Ok(raw) = std::fs::read_to_string(&f) else {
                continue;
            };
            let (fm_name, fm_desc) = parse_markdown_frontmatter(&raw);
            let stem = f
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("agent")
                .to_string();
            assets.push(PluginAsset {
                plugin: plugin_name.to_string(),
                kind: "agent".into(),
                name: fm_name.unwrap_or(stem),
                description: fm_desc.unwrap_or_else(|| first_nonempty_line(&raw)),
                path: f,
            });
        }
    }

    assets
}

fn load_plugin_hooks(
    root: &Path,
    project_dir: &Path,
    manifest: &PluginManifest,
) -> Result<Vec<CompiledHook>> {
    let mut hooks = Vec::new();
    let default_hooks = root.join("hooks").join("hooks.json");
    if default_hooks.is_file() {
        hooks.extend(parse_hooks_file(&default_hooks, project_dir, Some(root))?);
    }

    match manifest.hooks.as_ref() {
        Some(Value::String(rel)) => {
            let p = root.join(rel.trim_start_matches("./"));
            if p.is_file() && p != default_hooks {
                hooks.extend(parse_hooks_file(&p, project_dir, Some(root))?);
            }
        }
        Some(Value::Array(arr)) => {
            for item in arr {
                if let Some(rel) = item.as_str() {
                    let p = root.join(rel.trim_start_matches("./"));
                    if p.is_file() && p != default_hooks {
                        hooks.extend(parse_hooks_file(&p, project_dir, Some(root))?);
                    }
                }
            }
        }
        Some(val @ Value::Object(_)) => {
            hooks.extend(parse_hooks_value(
                val,
                project_dir,
                Some(root),
                &format!("{} (plugin.json)", root.display()),
            )?);
        }
        _ => {}
    }

    Ok(hooks)
}

fn parse_mcp_json_value(
    mut val: Value,
    project_dir: &Path,
    plugin_root: &Path,
    source: &str,
) -> Result<BTreeMap<String, McpServerConfig>> {
    expand_json_vars(&mut val, project_dir, plugin_root);
    // Support both `{ "mcpServers": { ... } }` and direct `{ "server-name": { "command": ... } }`
    let wrapped = if val.get("mcpServers").is_some() || val.get("mcp_servers").is_some() {
        val
    } else if val.is_object() {
        serde_json::json!({ "mcpServers": val })
    } else {
        return Ok(BTreeMap::new());
    };
    let parsed: McpConfigFile = serde_json::from_value(wrapped).map_err(|e| CarnotError::Tool {
        tool: "claude-plugins".into(),
        message: format!("invalid MCP config in {source}: {e}"),
    })?;
    Ok(parsed.mcp_servers)
}

fn load_plugin_mcp_servers(
    plugin_name: &str,
    root: &Path,
    project_dir: &Path,
    manifest: &PluginManifest,
) -> Result<BTreeMap<String, McpServerConfig>> {
    let mut raw_servers: BTreeMap<String, McpServerConfig> = BTreeMap::new();
    let default_mcp = root.join(".mcp.json");
    if default_mcp.is_file() {
        let txt = std::fs::read_to_string(&default_mcp).map_err(|e| CarnotError::Tool {
            tool: "claude-plugins".into(),
            message: format!("failed to read {}: {e}", default_mcp.display()),
        })?;
        let val: Value = serde_json::from_str(&txt).map_err(|e| CarnotError::Tool {
            tool: "claude-plugins".into(),
            message: format!("invalid JSON in {}: {e}", default_mcp.display()),
        })?;
        raw_servers.extend(parse_mcp_json_value(
            val,
            project_dir,
            root,
            &default_mcp.display().to_string(),
        )?);
    }

    match manifest.mcp_servers.as_ref() {
        Some(Value::String(rel)) => {
            let p = root.join(rel.trim_start_matches("./"));
            if p.is_file() && p != default_mcp {
                let txt = std::fs::read_to_string(&p).map_err(|e| CarnotError::Tool {
                    tool: "claude-plugins".into(),
                    message: format!("failed to read {}: {e}", p.display()),
                })?;
                let val: Value = serde_json::from_str(&txt).map_err(|e| CarnotError::Tool {
                    tool: "claude-plugins".into(),
                    message: format!("invalid JSON in {}: {e}", p.display()),
                })?;
                raw_servers.extend(parse_mcp_json_value(
                    val,
                    project_dir,
                    root,
                    &p.display().to_string(),
                )?);
            }
        }
        Some(val @ Value::Object(_)) => {
            raw_servers.extend(parse_mcp_json_value(
                val.clone(),
                project_dir,
                root,
                &format!("{} (plugin.json)", root.display()),
            )?);
        }
        _ => {}
    }

    let mut namespaced = BTreeMap::new();
    for (srv_name, cfg) in raw_servers {
        let key = if srv_name.starts_with(plugin_name) {
            srv_name
        } else {
            format!("{plugin_name}__{srv_name}")
        };
        namespaced.insert(key, cfg);
    }
    Ok(namespaced)
}

/// Discovers installed Claude Code plugin directories from `~/.claude/plugins/installed_plugins.json`
/// (supporting both v1 and v2 schemas), filtering out plugins disabled in `~/.claude/settings.json`.
pub fn discover_installed_claude_plugin_dirs() -> Vec<PathBuf> {
    let Some(home) = std::env::var_os("HOME") else {
        return Vec::new();
    };
    let claude_dir = PathBuf::from(home).join(".claude");

    // Read enabledPlugins map from ~/.claude/settings.json if present.
    let disabled: HashMap<String, bool> = std::fs::read_to_string(claude_dir.join("settings.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .and_then(|v| v.get("enabledPlugins").and_then(Value::as_object).cloned())
        .map(|obj| {
            obj.into_iter()
                .filter_map(|(k, v)| v.as_bool().map(|b| (k, b)))
                .collect()
        })
        .unwrap_or_default();

    let installed_path = claude_dir.join("plugins").join("installed_plugins.json");
    let Ok(raw) = std::fs::read_to_string(&installed_path) else {
        return Vec::new();
    };
    let Ok(val) = serde_json::from_str::<Value>(&raw) else {
        return Vec::new();
    };

    let plugins_obj = val
        .get("plugins")
        .and_then(Value::as_object)
        .or_else(|| val.as_object());
    let Some(map) = plugins_obj else {
        return Vec::new();
    };

    let mut dirs = Vec::new();
    for (plugin_key, entry) in map {
        if disabled.get(plugin_key) == Some(&false) {
            continue;
        }
        // v2: `{ "plugin@mkt": [ { "installPath": "..." } ] }`
        // v1: `{ "plugin@mkt": { "installPath": "..." } }`
        let candidate_path = if let Some(arr) = entry.as_array() {
            arr.first()
                .and_then(|e| e.get("installPath"))
                .and_then(Value::as_str)
        } else {
            entry.get("installPath").and_then(Value::as_str)
        };
        if let Some(p_str) = candidate_path {
            let p = PathBuf::from(p_str);
            if p.is_dir() && !dirs.contains(&p) {
                dirs.push(p);
            }
        }
    }
    dirs
}

/// Carnot `Module` executing hooks and advertising Turn-1 skills/commands/agents from loaded
/// Claude Code plugins (plus exporting their `.mcp.json` servers for `McpManager`).
#[derive(Debug)]
pub struct ClaudePluginsModule {
    plugins: Vec<LoadedClaudePlugin>,
    hooks: HooksModule,
    mcp_servers: BTreeMap<String, McpServerConfig>,
}

impl ClaudePluginsModule {
    /// Loads all configured Claude Code plugins relative to `workspace_root`.
    pub fn load(workspace_root: &Path, cfg: &ClaudePluginsConfig) -> Result<Self> {
        let mut dirs: Vec<PathBuf> = Vec::new();
        if cfg.load_installed_plugins {
            dirs.extend(discover_installed_claude_plugin_dirs());
        }
        for d in &cfg.plugin_dirs {
            let canon = std::fs::canonicalize(d).unwrap_or_else(|_| d.clone());
            if !dirs.contains(&canon) {
                dirs.push(canon);
            }
        }

        let mut plugins = Vec::new();
        let mut all_hooks = Vec::new();
        let mut all_mcp = BTreeMap::new();

        for dir in dirs {
            if !dir.is_dir() {
                return Err(CarnotError::Tool {
                    tool: "claude-plugins".into(),
                    message: format!("Claude Code plugin directory not found: {}", dir.display()),
                });
            }
            let manifest_path = dir.join(".claude-plugin").join("plugin.json");
            let manifest: PluginManifest = if manifest_path.is_file() {
                let raw =
                    std::fs::read_to_string(&manifest_path).map_err(|e| CarnotError::Tool {
                        tool: "claude-plugins".into(),
                        message: format!("failed to read {}: {e}", manifest_path.display()),
                    })?;
                serde_json::from_str(&raw).map_err(|e| CarnotError::Tool {
                    tool: "claude-plugins".into(),
                    message: format!("invalid JSON in {}: {e}", manifest_path.display()),
                })?
            } else {
                PluginManifest::default()
            };

            let name = manifest.name.clone().unwrap_or_else(|| {
                dir.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("plugin")
                    .to_string()
            });

            let hooks = load_plugin_hooks(&dir, workspace_root, &manifest)?;
            let hook_count = hooks.len();
            all_hooks.extend(hooks);

            let mcp_servers = load_plugin_mcp_servers(&name, &dir, workspace_root, &manifest)?;
            let mcp_server_names: Vec<String> = mcp_servers.keys().cloned().collect();
            all_mcp.extend(mcp_servers);

            let assets = discover_plugin_assets(&name, &dir, &manifest);

            plugins.push(LoadedClaudePlugin {
                name,
                version: manifest.version,
                description: manifest.description,
                root: dir,
                hook_count,
                mcp_server_names,
                assets,
            });
        }

        Ok(Self {
            plugins,
            hooks: HooksModule::from_compiled(
                workspace_root.to_path_buf(),
                all_hooks,
                cfg.fail_closed_hooks,
            ),
            mcp_servers: all_mcp,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    pub fn plugins(&self) -> &[LoadedClaudePlugin] {
        &self.plugins
    }

    /// Root directories of all loaded plugins so `ToolRegistry` can permit read-only access
    /// to advertised plugin `SKILL.md`, command, and agent files outside `workspace_root`.
    pub fn plugin_roots(&self) -> Vec<PathBuf> {
        self.plugins.iter().map(|p| p.root.clone()).collect()
    }

    /// MCP server configurations contributed by loaded Claude Code plugins (`${CLAUDE_PLUGIN_ROOT}`
    /// already expanded), ready to merge into `CarnotBuilder::extra_mcp_servers`.
    pub fn mcp_servers(&self) -> &BTreeMap<String, McpServerConfig> {
        &self.mcp_servers
    }

    fn format_turn1_catalog(&self) -> Option<String> {
        let mut lines = Vec::new();
        for p in &self.plugins {
            for a in &p.assets {
                let desc = if a.description.is_empty() {
                    String::new()
                } else {
                    format!(" — {}", a.description)
                };
                lines.push(format!(
                    "- [{}:{}] `{}`{} (Read `{}`)",
                    a.plugin,
                    a.kind,
                    a.name,
                    desc,
                    a.path.display()
                ));
            }
        }
        if lines.is_empty() {
            None
        } else {
            Some(format!(
                "[claude-plugins]\nLoaded Claude Code plugin skills, commands & agents (use `Read` on the path when relevant):\n{}",
                lines.join("\n")
            ))
        }
    }
}

impl Module for ClaudePluginsModule {
    fn name(&self) -> &'static str {
        "claude_plugins"
    }

    fn on_session_start(&self, config: &CarnotConfig) -> BoxFuture<'_, Result<Option<String>>> {
        let catalog = self.format_turn1_catalog();
        let hook_fut = self.hooks.on_session_start(config);
        Box::pin(async move {
            let hook_ctx = hook_fut.await?;
            match (catalog, hook_ctx) {
                (Some(c), Some(h)) => Ok(Some(format!("{c}\n\n{h}"))),
                (Some(c), None) => Ok(Some(c)),
                (None, Some(h)) => Ok(Some(h)),
                (None, None) => Ok(None),
            }
        })
    }

    fn before_tool_call(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> BoxFuture<'_, Result<ToolGateDecision>> {
        self.hooks.before_tool_call(tool_name, input)
    }

    fn after_tool_result<'a>(
        &'a self,
        tool_name: &'a str,
        input: &'a Value,
        output: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>> {
        self.hooks.after_tool_result(tool_name, input, output)
    }

    fn on_turn_boundary<'a>(
        &'a self,
        snapshot: &'a TurnBoundarySnapshot,
    ) -> BoxFuture<'a, Result<BoundaryAction>> {
        self.hooks.on_turn_boundary(snapshot)
    }

    fn before_compact<'a>(
        &'a self,
        ctx: &'a CompactionHookContext,
    ) -> BoxFuture<'a, Result<CompactionHookAction>> {
        self.hooks.before_compact(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn loads_full_claude_code_plugin_with_hooks_skills_commands_and_mcp() {
        let ws = std::env::temp_dir().join(format!("carnot-cc-ws-{}", uuid::Uuid::new_v4()));
        let plug = std::env::temp_dir().join(format!("carnot-cc-plug-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&ws).expect("ws");
        std::fs::create_dir_all(plug.join(".claude-plugin")).expect("manifest dir");
        std::fs::create_dir_all(plug.join("hooks")).expect("hooks dir");
        std::fs::create_dir_all(plug.join("skills/deploy-check")).expect("skills dir");
        std::fs::create_dir_all(plug.join("commands")).expect("cmds dir");
        std::fs::create_dir_all(plug.join("agents")).expect("agents dir");

        std::fs::write(
            plug.join(".claude-plugin/plugin.json"),
            json!({
                "name": "acme-guard",
                "version": "1.2.0",
                "description": "ACME security & workflow plugin"
            })
            .to_string(),
        )
        .expect("write plugin.json");

        std::fs::write(
            plug.join("hooks/hooks.json"),
            json!({
                "hooks": {
                    "PreToolUse": [{
                        "matcher": "Bash",
                        "hooks": [{
                            "type": "command",
                            "command": "python3 -c 'import os,sys; assert os.environ[\"CLAUDE_PLUGIN_ROOT\"]; sys.stderr.write(\"acme blocked\"); sys.exit(2)'"
                        }]
                    }]
                }
            })
            .to_string(),
        )
        .expect("write hooks.json");

        std::fs::write(
            plug.join("skills/deploy-check/SKILL.md"),
            "---\nname: deploy-check\ndescription: Verify release checklist before deploy\n---\n# Checklist\n",
        )
        .expect("write skill");

        std::fs::write(
            plug.join("commands/audit.md"),
            "---\ndescription: Run security audit\n---\nAudit instructions\n",
        )
        .expect("write cmd");

        std::fs::write(
            plug.join("agents/reviewer.md"),
            "---\nname: security-reviewer\ndescription: Reviews diffs for vulnerabilities\n---\n",
        )
        .expect("write agent");

        std::fs::write(
            plug.join(".mcp.json"),
            json!({
                "mcpServers": {
                    "scanner": {
                        "command": "${CLAUDE_PLUGIN_ROOT}/bin/scanner",
                        "args": ["--root", "${CLAUDE_PROJECT_DIR}"]
                    }
                }
            })
            .to_string(),
        )
        .expect("write .mcp.json");

        let module = ClaudePluginsModule::load(
            &ws,
            &ClaudePluginsConfig {
                plugin_dirs: vec![plug.clone()],
                load_installed_plugins: false,
                fail_closed_hooks: false,
            },
        )
        .expect("load plugin");

        assert_eq!(module.plugins().len(), 1);
        assert_eq!(module.plugins()[0].name, "acme-guard");
        assert_eq!(module.plugins()[0].hook_count, 1);
        assert_eq!(module.plugins()[0].assets.len(), 3);

        // MCP server has `${CLAUDE_PLUGIN_ROOT}` and `${CLAUDE_PROJECT_DIR}` expanded!
        let srv = module
            .mcp_servers()
            .get("acme-guard__scanner")
            .expect("namespaced mcp server");
        match srv {
            McpServerConfig::Stdio(s) => {
                assert!(s.command.ends_with("/bin/scanner"), "cmd={}", s.command);
                assert!(!s.command.contains("${CLAUDE_PLUGIN_ROOT}"));
                assert_eq!(s.args[1], ws.to_string_lossy());
            }
            McpServerConfig::Http(other) => panic!("expected stdio mcp config, got {other:?}"),
        }

        // Turn-1 catalog lists skill + command + agent (0 static tokens!)
        let cfg = CarnotConfig::new(ws.clone());
        let turn1 = module
            .on_session_start(&cfg)
            .await
            .expect("session start")
            .expect("turn1 catalog");
        assert!(turn1.contains("[acme-guard:skill] `deploy-check`"));
        assert!(turn1.contains("[acme-guard:command] `audit`"));
        assert!(turn1.contains("[acme-guard:agent] `security-reviewer`"));

        // Plugin PreToolUse hook receives CLAUDE_PLUGIN_ROOT and denies Bash with exit 2!
        let dec = module
            .before_tool_call("Bash", &json!({"command": "echo test"}))
            .await
            .expect("before tool");
        assert!(
            matches!(dec, ToolGateDecision::Deny { reason } if reason.contains("acme blocked"))
        );

        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&plug);
    }
}
