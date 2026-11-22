//! Declarative `.mcp.json` configuration loader with MCPoison RCE & secret-exfiltration guards.
//!
//! Security invariants (Opus architectural review):
//!   - Workspace-local `.mcp.json` is NEVER auto-executed unless explicitly approved (`trust_workspace_mcp`
//!     or hash-pinned in `~/.carnot/trusted-mcp.json`).
//!   - Rejects relative binary paths (`./node_modules/.bin/x`, `../foo`) on workspace configs.
//!   - Blocks dangerous dynamic-linker / runtime-hijack env vars (`LD_PRELOAD`, `DYLD_*`, `NODE_OPTIONS`, `PATH`, `PYTHONPATH`).
//!   - Restricts `${VAR}` interpolation in untrusted configs to a safe allowlist.

use carnot_core::{CarnotError, Result, canonicalize_json};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

const FORBIDDEN_ENV_KEYS: &[&str] = &[
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "NODE_OPTIONS",
    "PATH",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "RUBYOPT",
    "PERL5OPT",
    "BASH_ENV",
    "ENV",
    "ZDOTDIR",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpConfigFile {
    #[serde(default, alias = "mcp_servers")]
    pub mcp_servers: BTreeMap<String, McpServerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpServerConfig {
    Http(HttpServerConfig),
    Stdio(StdioServerConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StdioServerConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Only trusted servers may declare `annotations.readOnlyHint == true` for I2 parallel execution.
    #[serde(default)]
    pub trusted: bool,
    /// When true (default), server is only spawned on first `tools/call` if disk schema cache is warm.
    #[serde(default = "default_true")]
    pub lazy: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpServerConfig {
    pub url: String,
    /// Optional explicit transport override (`"sse"` for legacy HTTP+SSE MCP `2024-11-05`,
    /// `"http"` / `"streamable-http"` for Streamable HTTP). Auto-detects `"/sse"` suffix too.
    #[serde(default, alias = "type")]
    pub transport: Option<String>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub trusted: bool,
    #[serde(default = "default_true")]
    pub lazy: bool,
    /// Set automatically when loaded from a workspace `.mcp.json` (enforces SSRF per server — fixes Opus #5!).
    #[serde(skip)]
    pub from_workspace: bool,
}

impl HttpServerConfig {
    pub fn is_legacy_sse(&self) -> bool {
        if let Some(ref t) = self.transport {
            return t.eq_ignore_ascii_case("sse");
        }
        url::Url::parse(&self.url)
            .map(|u| u.path().trim_end_matches('/').ends_with("/sse"))
            .unwrap_or(false)
    }
}

fn default_timeout_ms() -> u64 {
    60_000
}

fn default_true() -> bool {
    true
}

impl McpServerConfig {
    pub fn is_enabled(&self) -> bool {
        match self {
            Self::Stdio(s) => s.enabled,
            Self::Http(h) => h.enabled,
        }
    }

    pub fn is_trusted(&self) -> bool {
        match self {
            Self::Stdio(s) => s.trusted,
            Self::Http(h) => h.trusted,
        }
    }

    pub fn is_lazy(&self) -> bool {
        match self {
            Self::Stdio(s) => s.lazy,
            Self::Http(h) => h.lazy,
        }
    }

    pub fn timeout_ms(&self) -> u64 {
        match self {
            Self::Stdio(s) => s.timeout_ms.clamp(1_000, 600_000),
            Self::Http(h) => h.timeout_ms.clamp(1_000, 600_000),
        }
    }

    /// Canonical BLAKE3 fingerprint of this server config (used for MCPoison trust-pinning & schema cache).
    pub fn fingerprint(&self) -> String {
        let val = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        let canon = canonicalize_json(&val);
        let bytes = serde_json::to_vec(&canon).unwrap_or_default();
        blake3::hash(&bytes).to_hex().to_string()
    }

    /// Hash of effective expanded auth material (used to isolate `cacheScope: "private"` entries
    /// across rotated tokens or different authorization contexts — Opus MCP 2026-07-28 rule!).
    pub fn auth_scope_hash(&self) -> String {
        match self {
            Self::Stdio(_) => "stdio-local".into(),
            Self::Http(h) => {
                let mut auth_pairs: Vec<(&str, &str)> = h
                    .headers
                    .iter()
                    .filter(|(k, _)| {
                        let lk = k.to_ascii_lowercase();
                        lk == "authorization"
                            || lk == "cookie"
                            || lk.contains("api-key")
                            || lk.contains("token")
                    })
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect();
                if auth_pairs.is_empty() {
                    return "public".into();
                }
                auth_pairs.sort_unstable();
                let mut hasher = blake3::Hasher::new();
                for (k, v) in auth_pairs {
                    hasher.update(k.to_ascii_lowercase().as_bytes());
                    hasher.update(b":");
                    hasher.update(v.as_bytes());
                    hasher.update(b"\0");
                }
                hasher.finalize().to_hex()[..16].to_string()
            }
        }
    }
}

/// Validates and sanitizes a server config against MCPoison attacks.
pub fn validate_server_config(
    name: &str,
    cfg: &mut McpServerConfig,
    from_workspace: bool,
) -> Result<()> {
    match cfg {
        McpServerConfig::Stdio(s) => {
            if s.command.trim().is_empty() {
                return Err(CarnotError::Tool {
                    tool: format!("mcp:{name}"),
                    message: "empty command".into(),
                });
            }
            if from_workspace {
                // Reject repo-relative executable paths (e.g. `./node_modules/.bin/x`, `../evil`)
                if s.command.starts_with('.') || s.command.contains("..") {
                    return Err(CarnotError::Sandbox(format!(
                        "MCPoison guard blocked relative executable `{}` in workspace MCP `{name}`",
                        s.command
                    )));
                }
            }
            for key in s.env.keys() {
                let up = key.to_ascii_uppercase();
                if up.starts_with("DYLD_") || FORBIDDEN_ENV_KEYS.contains(&up.as_str()) {
                    return Err(CarnotError::Sandbox(format!(
                        "MCPoison guard blocked forbidden env var `{key}` in MCP `{name}`"
                    )));
                }
            }
            // Expand `${VAR}` in env values ONLY for user-level configs (workspace configs cannot siphon host secrets!)
            if !from_workspace {
                for val in s.env.values_mut() {
                    *val = expand_env_vars(val);
                }
            }
            Ok(())
        }
        McpServerConfig::Http(h) => {
            h.from_workspace = from_workspace;
            // MCP 2026-07-28 SEP-2243: reject user headers starting with `Mcp-` so protocol headers
            // (`MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name`, `Mcp-Param-*`) can never collide!
            for key in h.headers.keys() {
                if key.to_ascii_lowercase().starts_with("mcp-") {
                    return Err(CarnotError::Sandbox(format!(
                        "MCP 2026-07-28 guard rejected reserved protocol header `{key}` in MCP `{name}`"
                    )));
                }
            }
            if from_workspace {
                // Project-supplied HTTP URLs MUST pass strict SSRF validation (cannot hit localhost/metadata!)
                crate::ssrf::validate_public_url(&h.url).map_err(|msg| {
                    CarnotError::Sandbox(format!(
                        "MCPoison SSRF guard rejected URL in workspace MCP `{name}`: {msg}"
                    ))
                })?;
            } else {
                for val in h.headers.values_mut() {
                    *val = expand_env_vars(val);
                }
            }
            Ok(())
        }
    }
}

fn expand_env_vars(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        if let Some(end) = after.find('}') {
            let var = &after[..end];
            if let Ok(val) = std::env::var(var) {
                out.push_str(&val);
            }
            rest = &after[end + 1..];
        } else {
            out.push_str("${");
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// Loads approved MCP server hashes from `~/.carnot/trusted-mcp.json` (never stored in repo!).
pub fn load_trusted_fingerprints() -> HashSet<String> {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return HashSet::new();
    };
    let path = home.join(".carnot").join("trusted-mcp.json");
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
        .map(|v| v.into_iter().collect())
        .unwrap_or_default()
}

/// Loads and validates MCP configs from explicit paths + optional user/workspace locations.
pub fn load_mcp_configs(
    workspace_root: &Path,
    explicit_paths: &[PathBuf],
    trust_workspace_mcp: bool,
) -> Result<BTreeMap<String, McpServerConfig>> {
    let mut merged: BTreeMap<String, McpServerConfig> = BTreeMap::new();
    let user_cfg =
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".carnot").join("mcp.json"));
    let has_user = user_cfg.as_deref().is_some_and(Path::is_file);
    let has_explicit = explicit_paths.iter().any(|p| p.is_file());
    let ws_paths: Vec<PathBuf> = [".carnot/mcp.json", ".mcp.json"]
        .into_iter()
        .map(|rel| workspace_root.join(rel))
        .filter(|p| p.is_file())
        .collect();

    // Fast path: zero MCP config files exist -> skip reading trusted-mcp.json completely.
    if !has_user && !has_explicit && ws_paths.is_empty() {
        return Ok(merged);
    }

    let trusted_hashes = if ws_paths.is_empty() {
        HashSet::new()
    } else {
        load_trusted_fingerprints()
    };

    // 1. User-level global `~/.carnot/mcp.json` (trusted user config)
    if let Some(ref uc) = user_cfg
        && uc.is_file()
    {
        merge_file(uc, false, true, &trusted_hashes, &mut merged)?;
    }

    // 2. Explicit CLI/SDK `--mcp-config` paths (user-approved)
    for p in explicit_paths {
        if p.is_file() {
            merge_file(p, false, true, &trusted_hashes, &mut merged)?;
        }
    }

    // 3. Workspace-local `.carnot/mcp.json` & `.mcp.json` (ONLY if explicitly trusted or hash-pinned!)
    for ws_cfg in ws_paths {
        merge_file(
            &ws_cfg,
            true,
            trust_workspace_mcp,
            &trusted_hashes,
            &mut merged,
        )?;
    }

    Ok(merged)
}

fn merge_file(
    path: &Path,
    from_workspace: bool,
    trust_workspace: bool,
    trusted_hashes: &HashSet<String>,
    out: &mut BTreeMap<String, McpServerConfig>,
) -> Result<()> {
    let txt = std::fs::read_to_string(path)?;
    let parsed: McpConfigFile = serde_json::from_str(&txt)?;
    for (name, mut cfg) in parsed.mcp_servers {
        if !cfg.is_enabled() {
            continue;
        }
        if from_workspace {
            let fp = cfg.fingerprint();
            if !trust_workspace && !trusted_hashes.contains(&fp) {
                tracing::warn!(
                    server = %name,
                    path = %path.display(),
                    fingerprint = %fp,
                    "skipping untrusted workspace MCP server (pass --trust-workspace-mcp to enable)"
                );
                continue;
            }
        }
        validate_server_config(&name, &mut cfg, from_workspace)?;
        out.insert(name, cfg);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcpoison_blocks_relative_binary_and_dyld_preload() {
        let mut rel = McpServerConfig::Stdio(StdioServerConfig {
            command: "./node_modules/.bin/evil".into(),
            args: vec![],
            env: BTreeMap::new(),
            cwd: None,
            timeout_ms: 5000,
            enabled: true,
            trusted: false,
            lazy: true,
        });
        assert!(validate_server_config("evil", &mut rel, true).is_err());

        let mut env = BTreeMap::new();
        env.insert("DYLD_INSERT_LIBRARIES".into(), "/tmp/pwn.dylib".into());
        let mut dyld = McpServerConfig::Stdio(StdioServerConfig {
            command: "npx".into(),
            args: vec![],
            env,
            cwd: None,
            timeout_ms: 5000,
            enabled: true,
            trusted: false,
            lazy: true,
        });
        assert!(validate_server_config("dyld", &mut dyld, false).is_err());
    }
}
