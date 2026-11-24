//! User-level persistent configuration (`~/.carnot/config.toml`).
//!
//! Only user-owned configuration is loaded (never workspace-local config files) so untrusted
//! repositories cannot disable the OS sandbox, enable outbound network, or redirect API traffic.
//! Explicit CLI flags always override values from `config.toml`.

use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Optional `[extensions]` section of `~/.carnot/config.toml`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct ExtensionsConfig {
    pub(crate) web: Option<bool>,
    pub(crate) lsp: Option<bool>,
    pub(crate) subagent: Option<bool>,
    pub(crate) memory: Option<bool>,
    pub(crate) mcp: Option<bool>,
    pub(crate) memory_dir: Option<PathBuf>,
}

/// Top-level schema for `~/.carnot/config.toml`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct UserConfig {
    pub(crate) provider: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) effort: Option<String>,
    #[serde(alias = "cache_retention")]
    pub(crate) cache_ttl: Option<String>,
    pub(crate) profile: Option<String>,
    pub(crate) plain: Option<bool>,
    pub(crate) sandbox: Option<bool>,
    pub(crate) allow_network: Option<bool>,
    pub(crate) persist_session: Option<bool>,
    pub(crate) session_dir: Option<PathBuf>,
    pub(crate) max_turns: Option<u64>,
    pub(crate) max_cost_usd: Option<f64>,
    pub(crate) compact_at: Option<u64>,
    pub(crate) timeout_secs: Option<u64>,
    pub(crate) exclude_tools: Option<Vec<String>>,
    pub(crate) extensions: ExtensionsConfig,
}

/// Resolves the user config path (`CARNOT_CONFIG` if set, otherwise `~/.carnot/config.toml`).
pub(crate) fn user_config_path() -> Option<(PathBuf, bool)> {
    if let Ok(explicit) = std::env::var("CARNOT_CONFIG") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            return Some((PathBuf::from(trimmed), true));
        }
    }
    std::env::var_os("HOME").map(|h| (PathBuf::from(h).join(".carnot").join("config.toml"), false))
}

/// Parses a `config.toml` file from disk.
pub(crate) fn load_config_from_path(path: &Path) -> Result<UserConfig, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    toml::from_str::<UserConfig>(&raw)
        .map_err(|e| format!("invalid TOML in {}: {e}", path.display()))
}

/// Loads `~/.carnot/config.toml` (or `CARNOT_CONFIG`) if present.
///
/// Missing `~/.carnot/config.toml` returns `Ok(None)`; a missing explicit `CARNOT_CONFIG` or a
/// syntax error returns `Err(...)`.
pub(crate) fn load_user_config() -> Result<Option<(PathBuf, UserConfig)>, String> {
    let Some((path, explicit)) = user_config_path() else {
        return Ok(None);
    };
    if !path.is_file() {
        if explicit {
            return Err(format!("CARNOT_CONFIG file not found: {}", path.display()));
        }
        return Ok(None);
    }
    load_config_from_path(&path).map(|cfg| Some((path, cfg)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_user_config_and_ignores_unknown_keys() {
        let raw = r#"
            provider = "openai"
            model = "gpt-5.4"
            effort = "medium"
            cache_ttl = "5m"
            profile = "standard"
            plain = true
            sandbox = false
            allow_network = true
            persist_session = false
            max_turns = 80
            max_cost_usd = 2.5
            compact_at = 120000
            timeout_secs = 90
            exclude_tools = ["Bash"]
            future_unknown_key = "ignored"

            [extensions]
            web = false
            lsp = true
            subagent = false
            memory = false
            mcp = false
        "#;
        let cfg: UserConfig = toml::from_str(raw).expect("valid config");
        assert_eq!(cfg.provider.as_deref(), Some("openai"));
        assert_eq!(cfg.model.as_deref(), Some("gpt-5.4"));
        assert_eq!(cfg.effort.as_deref(), Some("medium"));
        assert_eq!(cfg.cache_ttl.as_deref(), Some("5m"));
        assert_eq!(cfg.profile.as_deref(), Some("standard"));
        assert_eq!(cfg.plain, Some(true));
        assert_eq!(cfg.sandbox, Some(false));
        assert_eq!(cfg.allow_network, Some(true));
        assert_eq!(cfg.persist_session, Some(false));
        assert_eq!(cfg.max_turns, Some(80));
        assert_eq!(cfg.max_cost_usd, Some(2.5));
        assert_eq!(cfg.compact_at, Some(120_000));
        assert_eq!(cfg.timeout_secs, Some(90));
        assert_eq!(
            cfg.exclude_tools.as_deref(),
            Some(&["Bash".to_string()][..])
        );
        assert_eq!(cfg.extensions.web, Some(false));
        assert_eq!(cfg.extensions.lsp, Some(true));
        assert_eq!(cfg.extensions.subagent, Some(false));
        assert_eq!(cfg.extensions.memory, Some(false));
        assert_eq!(cfg.extensions.mcp, Some(false));
    }

    #[test]
    fn empty_toml_yields_all_none_defaults() {
        let cfg: UserConfig = toml::from_str("").expect("empty toml");
        assert!(cfg.provider.is_none());
        assert!(cfg.model.is_none());
        assert!(cfg.extensions.web.is_none());
    }
}
