//! Standards-compliant OAuth 2.0 engine for Carnot providers:
//! - RFC 7636 Authorization Code + PKCE (`S256`) with loopback HTTP callback & manual fallback
//! - RFC 8628 Device Authorization Grant (`device_code`) for headless / SSH environments
//! - Public PKCE JSON key-exchange flow (`pkce_key_exchange`, e.g. `callback_url` -> JSON `{"key": ...}`)
//! - Secure local token store (`~/.carnot/auth.json`, mode `0600` on Unix, atomic rename)
//! - Single-flight proactive & reactive (`401`) token refresh safe across concurrent fleet workers
//!
//! Strictly `#![forbid(unsafe_code)]`, zero hardcoded provider URLs (all endpoints come from `providers.json`).

use carnot_core::CarnotError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

mod flows;
mod pkce;
mod store;
#[cfg(test)]
mod tests;

pub use flows::{login_with_oauth, refresh_stored_token, save_manual_token};
pub use pkce::{
    extract_code_from_manual_input, generate_oauth_state, generate_pkce_pair, now_epoch_secs,
    pkce_s256_challenge, sha256, try_open_browser,
};
pub use store::{AuthStore, OAuthTokenManager};

pub(crate) fn oauth_err(provider: &str, message: String) -> CarnotError {
    CarnotError::Provider {
        provider: provider.to_string(),
        message,
        retryable: false,
        retry_after_ms: None,
    }
}

pub(crate) fn encode_form(pairs: &[(String, String)]) -> String {
    let mut ser = url::form_urlencoded::Serializer::new(String::new());
    for (k, v) in pairs {
        ser.append_pair(k, v);
    }
    ser.finish()
}
/// Proactive refresh skew before token expiry (2 minutes).
pub const DEFAULT_REFRESH_SKEW_SECS: u64 = 120;

/// Supported OAuth 2.0 authorization flows (all documented, standards-based, zero hardcoded URLs).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OAuthFlow {
    /// Standard RFC 7636 Authorization Code + PKCE (S256) with loopback redirect.
    #[default]
    Pkce,
    /// RFC 8628 Device Authorization Grant (headless / SSH friendly).
    DeviceCode,
    /// Public PKCE JSON key-exchange flow (`callback_url` + `S256` -> POST JSON `{code, code_verifier}`).
    #[serde(alias = "openrouter_pkce")]
    PkceKeyExchange,
}

/// Declarative per-provider OAuth 2.0 configuration in `providers.json` / `~/.carnot/providers.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OAuthConfig {
    #[serde(default)]
    pub flow: OAuthFlow,
    #[serde(default)]
    pub authorize_url: Option<String>,
    #[serde(default)]
    pub token_url: Option<String>,
    #[serde(default)]
    pub device_authorization_url: Option<String>,
    /// Optional explicit OAuth `client_id` (only for public documented flows or user-registered apps).
    #[serde(default)]
    pub client_id: Option<String>,
    /// Environment variable(s) supplying the user's OAuth `client_id`.
    #[serde(default)]
    pub client_id_env: Option<String>,
    /// Optional environment variable supplying the user's OAuth `client_secret` (confidential clients).
    #[serde(default)]
    pub client_secret_env: Option<String>,
    #[serde(default)]
    pub scopes: Vec<String>,
    #[serde(default)]
    pub redirect_port: Option<u16>,
    #[serde(default)]
    pub callback_path: Option<String>,
    #[serde(default)]
    pub extra_auth_params: BTreeMap<String, String>,
    #[serde(default)]
    pub extra_token_params: BTreeMap<String, String>,
    /// Exchange endpoint for `pkce_key_exchange` (configured in `providers.json`).
    #[serde(default)]
    pub exchange_url: Option<String>,
}

impl OAuthConfig {
    pub fn resolve_client_id(&self, cli_override: Option<&str>) -> Option<String> {
        if let Some(id) = cli_override
            && !id.trim().is_empty()
        {
            return Some(id.trim().to_string());
        }
        if let Some(ref env_name) = self.client_id_env
            && let Ok(val) = std::env::var(env_name)
            && !val.trim().is_empty()
        {
            return Some(val.trim().to_string());
        }
        self.client_id
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
    }

    pub fn resolve_client_secret(&self) -> Option<String> {
        let env_name = self.client_secret_env.as_deref()?;
        std::env::var(env_name)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    }

    pub fn callback_path(&self) -> &str {
        self.callback_path
            .as_deref()
            .filter(|p| p.starts_with('/'))
            .unwrap_or("/callback")
    }
}

/// Persisted OAuth/credential entry in `~/.carnot/auth.json`.
/// `Debug` redacts sensitive token values.
#[derive(Clone, Serialize, Deserialize)]
pub struct OAuthStoredToken {
    pub access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    #[serde(default = "default_token_type")]
    pub token_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at_epoch_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_url: Option<String>,
    #[serde(default)]
    pub updated_at_epoch_secs: u64,
}

fn default_token_type() -> String {
    "Bearer".into()
}

impl std::fmt::Debug for OAuthStoredToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthStoredToken")
            .field("access_token", &"<redacted>")
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "<redacted>"),
            )
            .field("token_type", &self.token_type)
            .field("expires_at_epoch_secs", &self.expires_at_epoch_secs)
            .field("scope", &self.scope)
            .field("token_url", &self.token_url)
            .field("updated_at_epoch_secs", &self.updated_at_epoch_secs)
            .finish_non_exhaustive()
    }
}

impl OAuthStoredToken {
    pub fn is_expired(&self) -> bool {
        self.is_expiring_within(0)
    }

    pub fn is_expiring_within(&self, skew_secs: u64) -> bool {
        let Some(exp) = self.expires_at_epoch_secs else {
            return false;
        };
        now_epoch_secs().saturating_add(skew_secs) >= exp
    }

    pub fn seconds_until_expiry(&self) -> Option<i64> {
        let exp = self.expires_at_epoch_secs?;
        let now = now_epoch_secs();
        if exp >= now {
            Some(i64::try_from(exp - now).unwrap_or(i64::MAX))
        } else {
            Some(-i64::try_from(now - exp).unwrap_or(i64::MAX))
        }
    }

    pub fn can_refresh(&self, oauth_cfg: Option<&OAuthConfig>) -> bool {
        if self
            .refresh_token
            .as_deref()
            .map(str::trim)
            .is_none_or(str::is_empty)
        {
            return false;
        }
        self.token_url
            .as_deref()
            .or_else(|| oauth_cfg.and_then(|c| c.token_url.as_deref()))
            .is_some()
    }
}

#[derive(Debug, Clone)]
pub struct OAuthLoginOptions {
    pub force_device_flow: bool,
    pub manual_callback: bool,
    pub open_browser: bool,
    pub port_override: Option<u16>,
    pub client_id_override: Option<String>,
    pub timeout: Duration,
}

impl Default for OAuthLoginOptions {
    fn default() -> Self {
        Self {
            force_device_flow: false,
            manual_callback: false,
            open_browser: true,
            port_override: None,
            client_id_override: None,
            timeout: Duration::from_secs(300),
        }
    }
}

/// Status prompt emitted during login so CLI/UI can render instructions cleanly.
#[derive(Debug, Clone)]
pub enum OAuthPrompt {
    OpenUrl {
        url: String,
        callback_url: String,
        manual_paste: bool,
    },
    DeviceCode {
        verification_uri: String,
        verification_uri_complete: Option<String>,
        user_code: String,
    },
}
