//! Standards-compliant OAuth 2.0 engine for Carnot providers:
//! - RFC 7636 Authorization Code + PKCE (`S256`) with loopback HTTP callback & manual fallback
//! - RFC 8628 Device Authorization Grant (`device_code`) for headless / SSH environments
//! - Public PKCE JSON key-exchange flow (`pkce_key_exchange`, e.g. `callback_url` -> JSON `{"key": ...}`)
//! - Secure local token store (`~/.carnot/auth.json`, mode `0600` on Unix, atomic rename)
//! - Single-flight proactive & reactive (`401`) token refresh safe across concurrent fleet workers
//!
//! Strictly `#![forbid(unsafe_code)]`, zero hardcoded provider URLs (all endpoints come from `providers.json`).

use crate::build_http_client_for_url;
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use carnot_core::{CarnotError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use url::Url;
use uuid::Uuid;

fn oauth_err(provider: &str, message: String) -> CarnotError {
    CarnotError::Provider {
        provider: provider.to_string(),
        message,
        retryable: false,
        retry_after_ms: None,
    }
}

fn encode_form(pairs: &[(String, String)]) -> String {
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
            .finish()
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
        Some((exp as i64) - (now_epoch_secs() as i64))
    }

    pub fn can_refresh(&self, oauth_cfg: Option<&OAuthConfig>) -> bool {
        if self
            .refresh_token
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .is_none()
        {
            return false;
        }
        self.token_url
            .as_deref()
            .or_else(|| oauth_cfg.and_then(|c| c.token_url.as_deref()))
            .is_some()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct AuthStoreFile {
    #[serde(default = "default_store_version")]
    schema_version: u32,
    #[serde(default)]
    profiles: BTreeMap<String, OAuthStoredToken>,
}

fn default_store_version() -> u32 {
    1
}

/// Secret-safe local OAuth token store (`~/.carnot/auth.json`, mode `0600` on Unix).
#[derive(Debug, Clone)]
pub struct AuthStore {
    path: PathBuf,
}

impl AuthStore {
    pub fn default_path() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("CARNOT_AUTH_FILE")
            && !p.trim().is_empty()
        {
            return Some(PathBuf::from(p.trim()));
        }
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
        Some(PathBuf::from(home).join(".carnot").join("auth.json"))
    }

    pub fn open_default() -> Option<Self> {
        Self::default_path().map(|path| Self { path })
    }

    pub fn open(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn read_file(&self) -> Result<AuthStoreFile> {
        if !self.path.exists() {
            return Ok(AuthStoreFile {
                schema_version: 1,
                profiles: BTreeMap::new(),
            });
        }
        let raw = std::fs::read_to_string(&self.path).map_err(|e| {
            oauth_err(
                "oauth",
                format!(
                    "failed to read OAuth token store {}: {e}",
                    self.path.display()
                ),
            )
        })?;
        if raw.trim().is_empty() {
            return Ok(AuthStoreFile {
                schema_version: 1,
                profiles: BTreeMap::new(),
            });
        }
        serde_json::from_str(&raw).map_err(|e| {
            oauth_err(
                "oauth",
                format!(
                    "invalid JSON in OAuth token store {}: {e}",
                    self.path.display()
                ),
            )
        })
    }

    fn write_file(&self, file: &AuthStoreFile) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                oauth_err(
                    "oauth",
                    format!("failed to create auth directory {}: {e}", parent.display()),
                )
            })?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
            }
        }
        let encoded = serde_json::to_string_pretty(file)
            .map_err(|e| oauth_err("oauth", format!("failed to serialize auth store: {e}")))?;

        let tmp = self.path.with_extension(format!("tmp.{}", Uuid::new_v4()));
        #[cfg(unix)]
        {
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;
            let mut f = std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&tmp)
                .map_err(|e| {
                    oauth_err(
                        "oauth",
                        format!("failed to create temp auth store {}: {e}", tmp.display()),
                    )
                })?;
            f.write_all(encoded.as_bytes()).map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                oauth_err("oauth", format!("failed to write temp auth store: {e}"))
            })?;
            f.sync_all().ok();
        }
        #[cfg(not(unix))]
        {
            std::fs::write(&tmp, encoded.as_bytes())
                .map_err(|e| oauth_err("oauth", format!("failed to write temp auth store: {e}")))?;
        }
        std::fs::rename(&tmp, &self.path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            oauth_err(
                "oauth",
                format!("failed to replace auth store {}: {e}", self.path.display()),
            )
        })?;
        Ok(())
    }

    fn lock_path(&self) -> PathBuf {
        let mut p = self.path.as_os_str().to_os_string();
        p.push(".lock");
        PathBuf::from(p)
    }

    fn with_exclusive_lock<T, F: FnOnce() -> Result<T>>(&self, f: F) -> Result<T> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                oauth_err(
                    "oauth",
                    format!("failed to create auth directory {}: {e}", parent.display()),
                )
            })?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
            }
        }
        let lock_path = self.lock_path();
        #[cfg(unix)]
        let lock_file = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .mode(0o600)
                .open(&lock_path)
        }
        .map_err(|e| {
            oauth_err(
                "oauth",
                format!("failed to open auth lockfile {}: {e}", lock_path.display()),
            )
        })?;
        #[cfg(not(unix))]
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|e| {
                oauth_err(
                    "oauth",
                    format!("failed to open auth lockfile {}: {e}", lock_path.display()),
                )
            })?;

        lock_file.lock().map_err(|e| {
            oauth_err(
                "oauth",
                format!("failed to lock auth store {}: {e}", lock_path.display()),
            )
        })?;
        let res = f();
        let _ = lock_file.unlock();
        res
    }

    pub fn get(&self, profile: &str) -> Result<Option<OAuthStoredToken>> {
        Ok(self.read_file()?.profiles.remove(profile))
    }

    pub fn list(&self) -> Result<BTreeMap<String, OAuthStoredToken>> {
        Ok(self.read_file()?.profiles)
    }

    pub fn save(&self, profile: &str, token: OAuthStoredToken) -> Result<()> {
        let res = self.with_exclusive_lock(|| {
            let mut data = self.read_file()?;
            data.profiles.insert(profile.to_string(), token);
            self.write_file(&data)
        });
        if res.is_ok() {
            OAuthTokenManager::evict(&self.path, profile);
        }
        res
    }

    /// Saves a refreshed token ONLY if `profile` is still present on disk (so a concurrent
    /// `carnot auth logout` is never undone by an in-flight refresh). Returns `true` if saved.
    pub fn update_if_present(&self, profile: &str, token: OAuthStoredToken) -> Result<bool> {
        self.with_exclusive_lock(|| {
            let mut data = self.read_file()?;
            if !data.profiles.contains_key(profile) {
                return Ok(false);
            }
            data.profiles.insert(profile.to_string(), token);
            self.write_file(&data)?;
            Ok(true)
        })
    }

    pub fn remove(&self, profile: &str) -> Result<bool> {
        let res = self.with_exclusive_lock(|| {
            let mut data = self.read_file()?;
            let removed = data.profiles.remove(profile).is_some();
            if removed {
                self.write_file(&data)?;
            }
            Ok(removed)
        });
        if res.is_ok() {
            OAuthTokenManager::evict(&self.path, profile);
        }
        res
    }
}

type ManagerRegistry = std::sync::Mutex<HashMap<(PathBuf, String), Arc<OAuthTokenManager>>>;

fn shared_manager_registry() -> &'static ManagerRegistry {
    static REGISTRY: OnceLock<ManagerRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Thread-safe, process-wide single-flight OAuth token manager attached to `ResolvedAuth::OAuth`.
pub struct OAuthTokenManager {
    profile: String,
    oauth_cfg: Option<OAuthConfig>,
    store: AuthStore,
    state: Mutex<OAuthStoredToken>,
    has_refresh_cap: AtomicBool,
}

impl std::fmt::Debug for OAuthTokenManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthTokenManager")
            .field("profile", &self.profile)
            .field("store", &self.store.path())
            .finish()
    }
}

impl OAuthTokenManager {
    pub fn new(
        profile: impl Into<String>,
        token: OAuthStoredToken,
        oauth_cfg: Option<OAuthConfig>,
        store: AuthStore,
    ) -> Self {
        let can_refresh = token.can_refresh(oauth_cfg.as_ref());
        Self {
            profile: profile.into(),
            oauth_cfg,
            store,
            state: Mutex::new(token),
            has_refresh_cap: AtomicBool::new(can_refresh),
        }
    }

    /// Returns the process-wide shared `Arc<OAuthTokenManager>` for `(store.path(), profile)` so
    /// parent sessions, subagents, and fleet workers serialize refreshes through one single-flight lock.
    pub fn shared(
        profile: impl Into<String>,
        token: OAuthStoredToken,
        oauth_cfg: Option<OAuthConfig>,
        store: AuthStore,
    ) -> Arc<Self> {
        let profile = profile.into();
        let key = (store.path().to_path_buf(), profile.clone());
        let mut reg = shared_manager_registry()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = reg.get(&key) {
            return Arc::clone(existing);
        }
        let mgr = Arc::new(Self::new(profile, token, oauth_cfg, store));
        reg.insert(key, Arc::clone(&mgr));
        mgr
    }

    /// Evicts any cached shared manager for `(store_path, profile)` (e.g. after login/logout).
    pub fn evict(store_path: &Path, profile: &str) {
        let mut reg = shared_manager_registry()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        reg.remove(&(store_path.to_path_buf(), profile.to_string()));
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn supports_refresh(&self) -> bool {
        self.has_refresh_cap.load(Ordering::Relaxed)
    }

    /// Returns a live access token, proactively refreshing when expiring within `DEFAULT_REFRESH_SKEW_SECS`.
    ///
    /// When `rejected_token` is `Some(bad)` (after an HTTP `401 Unauthorized`), performs compare-and-swap:
    /// if another worker already refreshed the token while waiting on the lock (`guard.access_token != bad`),
    /// returns the fresh token immediately without issuing a duplicate refresh request.
    pub async fn access_token(&self, rejected_token: Option<&str>) -> Result<String> {
        let mut guard = self.state.lock().await;

        if let Some(bad) = rejected_token {
            if guard.access_token != bad && !guard.is_expiring_within(DEFAULT_REFRESH_SKEW_SECS) {
                return Ok(guard.access_token.clone());
            }
        } else if !guard.is_expiring_within(DEFAULT_REFRESH_SKEW_SECS) {
            return Ok(guard.access_token.clone());
        }

        // Cross-process check via `spawn_blocking`: another Carnot process may have already refreshed.
        let store_clone = self.store.clone();
        let profile_clone = self.profile.clone();
        let disk_tok = tokio::task::spawn_blocking(move || store_clone.get(&profile_clone))
            .await
            .ok()
            .and_then(|r| r.ok())
            .flatten();

        if let Some(disk_tok) = disk_tok {
            let not_rejected = rejected_token != Some(disk_tok.access_token.as_str());
            let disk_is_fresher = disk_tok.updated_at_epoch_secs > guard.updated_at_epoch_secs
                || disk_tok.access_token != guard.access_token;
            if not_rejected
                && disk_is_fresher
                && !disk_tok.is_expiring_within(DEFAULT_REFRESH_SKEW_SECS)
            {
                let can_ref = disk_tok.can_refresh(self.oauth_cfg.as_ref());
                *guard = disk_tok;
                self.has_refresh_cap.store(can_ref, Ordering::Relaxed);
                return Ok(guard.access_token.clone());
            }
        }

        // If not handling a 401 and token isn't actually expired yet and cannot be refreshed, keep using it.
        if !guard.can_refresh(self.oauth_cfg.as_ref()) {
            self.has_refresh_cap.store(false, Ordering::Relaxed);
            if rejected_token.is_none() && !guard.is_expired() {
                return Ok(guard.access_token.clone());
            }
            return Err(CarnotError::Provider {
                provider: self.profile.clone(),
                message: format!(
                    "OAuth token for `{}` is expired or rejected and has no refresh_token; run `carnot auth login {}`",
                    self.profile, self.profile
                ),
                retryable: false,
                retry_after_ms: None,
            });
        }

        let refreshed =
            exchange_refresh_token(&self.profile, &guard, self.oauth_cfg.as_ref()).await?;
        let store_save = self.store.clone();
        let profile_save = self.profile.clone();
        let refreshed_save = refreshed.clone();
        let still_present = tokio::task::spawn_blocking(move || {
            store_save.update_if_present(&profile_save, refreshed_save)
        })
        .await
        .map_err(|e| oauth_err(&self.profile, format!("auth store join error: {e}")))??;

        let can_ref = refreshed.can_refresh(self.oauth_cfg.as_ref());
        *guard = refreshed;
        self.has_refresh_cap.store(can_ref, Ordering::Relaxed);
        if !still_present {
            tracing::debug!(
                provider = %self.profile,
                "OAuth profile was removed on disk during refresh; keeping in-memory token without resurrecting auth.json entry"
            );
        }
        Ok(guard.access_token.clone())
    }
}

/// Options controlling interactive `carnot auth login`.
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

/// Pure safe Rust FIPS 180-4 SHA-256 (zero unsafe, zero external crates).
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity((data.len() + 9).div_ceil(64) * 64);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, word) in chunk.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Computes RFC 7636 PKCE `S256` challenge: `BASE64URL-NO-PAD(SHA256(code_verifier))`.
pub fn pkce_s256_challenge(code_verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(sha256(code_verifier.as_bytes()))
}

/// Generates a CSPRNG PKCE `(code_verifier, code_challenge)` pair (64-char verifier).
pub fn generate_pkce_pair() -> (String, String) {
    let mut entropy = [0u8; 48];
    entropy[0..16].copy_from_slice(Uuid::new_v4().as_bytes());
    entropy[16..32].copy_from_slice(Uuid::new_v4().as_bytes());
    entropy[32..48].copy_from_slice(Uuid::new_v4().as_bytes());
    let verifier = URL_SAFE_NO_PAD.encode(entropy);
    let challenge = pkce_s256_challenge(&verifier);
    (verifier, challenge)
}

/// Generates a CSPRNG OAuth CSRF `state` token (256-bit entropy).
pub fn generate_oauth_state() -> String {
    let mut entropy = [0u8; 32];
    entropy[0..16].copy_from_slice(Uuid::new_v4().as_bytes());
    entropy[16..32].copy_from_slice(Uuid::new_v4().as_bytes());
    URL_SAFE_NO_PAD.encode(entropy)
}

pub fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Attempts to open a URL in the user's default browser (best-effort; never fatal).
pub fn try_open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("rundll32")
        .args(["url.dll,FileProtocolHandler", url])
        .spawn();
}

/// Runs a complete OAuth login flow for `profile` according to `cfg` and persists the resulting token.
pub async fn login_with_oauth<F, M>(
    profile: &str,
    cfg: &OAuthConfig,
    store: &AuthStore,
    opts: &OAuthLoginOptions,
    mut on_prompt: F,
    read_manual_input: M,
) -> Result<OAuthStoredToken>
where
    F: FnMut(OAuthPrompt),
    M: FnOnce() -> Result<String>,
{
    let flow = if opts.force_device_flow {
        OAuthFlow::DeviceCode
    } else {
        cfg.flow
    };

    let token = match flow {
        OAuthFlow::PkceKeyExchange => {
            run_pkce_key_exchange_flow(profile, cfg, opts, &mut on_prompt, read_manual_input)
                .await?
        }
        OAuthFlow::Pkce => {
            run_standard_pkce_flow(profile, cfg, opts, &mut on_prompt, read_manual_input).await?
        }
        OAuthFlow::DeviceCode => run_device_code_flow(profile, cfg, opts, &mut on_prompt).await?,
    };

    store.save(profile, token.clone())?;
    Ok(token)
}

/// Saves a manually supplied OAuth access token or API key into the `0600` auth store.
pub fn save_manual_token(
    profile: &str,
    access_token: &str,
    store: &AuthStore,
) -> Result<OAuthStoredToken> {
    let trimmed = access_token.trim();
    if trimmed.is_empty() {
        return Err(oauth_err("oauth", "empty token provided".into()));
    }
    let tok = OAuthStoredToken {
        access_token: trimmed.to_string(),
        refresh_token: None,
        token_type: "Bearer".into(),
        expires_at_epoch_secs: None,
        scope: None,
        client_id: None,
        token_url: None,
        updated_at_epoch_secs: now_epoch_secs(),
    };
    store.save(profile, tok.clone())?;
    Ok(tok)
}

/// Forces an immediate refresh of the stored token for `profile` and saves the updated credential.
pub async fn refresh_stored_token(
    profile: &str,
    oauth_cfg: Option<&OAuthConfig>,
    store: &AuthStore,
) -> Result<OAuthStoredToken> {
    let current = store.get(profile)?.ok_or_else(|| {
        oauth_err(
            "oauth",
            format!("no stored OAuth token for `{profile}`; run `carnot auth login {profile}`"),
        )
    })?;
    let refreshed = exchange_refresh_token(profile, &current, oauth_cfg).await?;
    store.save(profile, refreshed.clone())?;
    Ok(refreshed)
}

async fn run_pkce_key_exchange_flow<F, M>(
    profile: &str,
    cfg: &OAuthConfig,
    opts: &OAuthLoginOptions,
    on_prompt: &mut F,
    read_manual_input: M,
) -> Result<OAuthStoredToken>
where
    F: FnMut(OAuthPrompt),
    M: FnOnce() -> Result<String>,
{
    let authorize_base = cfg.authorize_url.as_deref().ok_or_else(|| {
        oauth_err(
            profile,
            format!("provider `{profile}` OAuth config is missing `authorize_url`"),
        )
    })?;
    let exchange_url = cfg.exchange_url.as_deref().ok_or_else(|| {
        oauth_err(
            profile,
            format!("provider `{profile}` OAuth config is missing `exchange_url`"),
        )
    })?;
    let (code_verifier, code_challenge) = generate_pkce_pair();
    let state = generate_oauth_state();
    let callback_path = cfg.callback_path();

    let (listener_opt, callback_url) = if opts.manual_callback {
        // Per OpenRouter PKCE docs, omitting `callback_url` in headless/manual mode makes the
        // authorization page display the code directly for manual paste.
        (None, None)
    } else {
        let bind_port = opts.port_override.or(cfg.redirect_port).unwrap_or(0);
        let listener = TcpListener::bind(("127.0.0.1", bind_port))
            .await
            .map_err(|e| {
                oauth_err(
                    "oauth",
                    format!(
                        "failed to bind local OAuth callback listener on 127.0.0.1:{bind_port}: {e}"
                    ),
                )
            })?;
        let port = listener
            .local_addr()
            .map_err(|e| oauth_err("oauth", format!("local_addr failed: {e}")))?
            .port();
        (
            Some(listener),
            Some(format!("http://localhost:{port}{callback_path}")),
        )
    };

    let mut auth_url = Url::parse(authorize_base).map_err(|e| {
        oauth_err(
            "oauth",
            format!("invalid OAuth authorize_url `{authorize_base}`: {e}"),
        )
    })?;
    {
        let mut qp = auth_url.query_pairs_mut();
        if let Some(ref cb) = callback_url {
            qp.append_pair("callback_url", cb);
        }
        qp.append_pair("code_challenge", &code_challenge);
        qp.append_pair("code_challenge_method", "S256");
        qp.append_pair("state", &state);
        for (k, v) in &cfg.extra_auth_params {
            qp.append_pair(k, v);
        }
    }

    let auth_url_str = auth_url.to_string();
    on_prompt(OAuthPrompt::OpenUrl {
        url: auth_url_str.clone(),
        callback_url: callback_url.unwrap_or_default(),
        manual_paste: opts.manual_callback,
    });
    if opts.open_browser && !opts.manual_callback {
        try_open_browser(&auth_url_str);
    }

    let code = if let Some(listener) = listener_opt {
        let (code, _state) =
            wait_for_loopback_callback(listener, callback_path, Some(&state), false, opts.timeout)
                .await?;
        code
    } else {
        let pasted = read_manual_input()?;
        extract_code_from_manual_input(&pasted, None)?
    };

    let client = build_http_client_for_url(exchange_url);
    let resp = client
        .post(exchange_url)
        .json(&serde_json::json!({
            "code": code,
            "code_verifier": code_verifier,
            "code_challenge_method": "S256",
        }))
        .send()
        .await
        .map_err(|e| CarnotError::Provider {
            provider: profile.into(),
            message: format!("OAuth key exchange request failed: {}", e.without_url()),
            retryable: true,
            retry_after_ms: None,
        })?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(CarnotError::Provider {
            provider: profile.into(),
            message: format!("PKCE key exchange HTTP {status}: {text}"),
            retryable: false,
            retry_after_ms: None,
        });
    }
    let body: Value = serde_json::from_str(&text).map_err(|e| CarnotError::Provider {
        provider: profile.into(),
        message: format!("invalid JSON from PKCE key exchange: {e}"),
        retryable: false,
        retry_after_ms: None,
    })?;
    let key = body
        .get("key")
        .or_else(|| body.get("access_token"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CarnotError::Provider {
            provider: profile.into(),
            message: "PKCE key-exchange response missing `key`".into(),
            retryable: false,
            retry_after_ms: None,
        })?;

    Ok(OAuthStoredToken {
        access_token: key.to_string(),
        refresh_token: None,
        token_type: "Bearer".into(),
        expires_at_epoch_secs: None,
        scope: None,
        client_id: None,
        token_url: None,
        updated_at_epoch_secs: now_epoch_secs(),
    })
}

async fn run_standard_pkce_flow<F, M>(
    profile: &str,
    cfg: &OAuthConfig,
    opts: &OAuthLoginOptions,
    on_prompt: &mut F,
    read_manual_input: M,
) -> Result<OAuthStoredToken>
where
    F: FnMut(OAuthPrompt),
    M: FnOnce() -> Result<String>,
{
    let authorize_base = cfg.authorize_url.as_deref().ok_or_else(|| {
        oauth_err(
            "oauth",
            format!("provider `{profile}` OAuth config is missing `authorize_url`"),
        )
    })?;
    let token_url = cfg.token_url.as_deref().ok_or_else(|| {
        oauth_err(
            "oauth",
            format!("provider `{profile}` OAuth config is missing `token_url`"),
        )
    })?;
    let client_id = cfg
        .resolve_client_id(opts.client_id_override.as_deref())
        .ok_or_else(|| {
            let hint = cfg
                .client_id_env
                .as_deref()
                .map(|e| format!(" (set `{e}` or pass `--client-id`)"))
                .unwrap_or_else(|| {
                    " (pass `--client-id` or set `oauth.client_id` in ~/.carnot/providers.json)"
                        .into()
                });
            oauth_err(
                "oauth",
                format!(
                    "provider `{profile}` OAuth login requires a registered OAuth `client_id`{hint}"
                ),
            )
        })?;
    let client_secret = cfg.resolve_client_secret();
    let (code_verifier, code_challenge) = generate_pkce_pair();
    let state = generate_oauth_state();
    let callback_path = cfg.callback_path();

    let (listener_opt, redirect_uri) =
        if opts.manual_callback {
            let port = opts.port_override.or(cfg.redirect_port).unwrap_or(8787);
            (None, format!("http://127.0.0.1:{port}{callback_path}"))
        } else {
            let bind_port = opts.port_override.or(cfg.redirect_port).unwrap_or(0);
            let listener = TcpListener::bind(("127.0.0.1", bind_port))
            .await
            .map_err(|e| {
                oauth_err("oauth", format!(
                    "failed to bind local OAuth callback listener on 127.0.0.1:{bind_port}: {e}"
                ))
            })?;
            let port = listener
                .local_addr()
                .map_err(|e| oauth_err("oauth", format!("local_addr failed: {e}")))?
                .port();
            (
                Some(listener),
                format!("http://127.0.0.1:{port}{callback_path}"),
            )
        };

    let mut auth_url = Url::parse(authorize_base).map_err(|e| {
        oauth_err(
            "oauth",
            format!("invalid OAuth authorize_url `{authorize_base}`: {e}"),
        )
    })?;
    {
        let mut qp = auth_url.query_pairs_mut();
        qp.append_pair("response_type", "code");
        qp.append_pair("client_id", &client_id);
        qp.append_pair("redirect_uri", &redirect_uri);
        qp.append_pair("code_challenge", &code_challenge);
        qp.append_pair("code_challenge_method", "S256");
        qp.append_pair("state", &state);
        if !cfg.scopes.is_empty() {
            qp.append_pair("scope", &cfg.scopes.join(" "));
        }
        for (k, v) in &cfg.extra_auth_params {
            qp.append_pair(k, v);
        }
    }

    let auth_url_str = auth_url.to_string();
    on_prompt(OAuthPrompt::OpenUrl {
        url: auth_url_str.clone(),
        callback_url: redirect_uri.clone(),
        manual_paste: opts.manual_callback,
    });
    if opts.open_browser && !opts.manual_callback {
        try_open_browser(&auth_url_str);
    }

    let code = if let Some(listener) = listener_opt {
        let (code, _) =
            wait_for_loopback_callback(listener, callback_path, Some(&state), true, opts.timeout)
                .await?;
        code
    } else {
        let pasted = read_manual_input()?;
        extract_code_from_manual_input(&pasted, Some(&state))?
    };

    let mut form: Vec<(String, String)> = vec![
        ("grant_type".into(), "authorization_code".into()),
        ("code".into(), code),
        ("redirect_uri".into(), redirect_uri),
        ("client_id".into(), client_id.clone()),
        ("code_verifier".into(), code_verifier),
    ];
    if let Some(secret) = client_secret {
        form.push(("client_secret".into(), secret));
    }
    for (k, v) in &cfg.extra_token_params {
        form.push((k.clone(), v.clone()));
    }

    let client = build_http_client_for_url(token_url);
    let resp = client
        .post(token_url)
        .header("accept", "application/json")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(encode_form(&form))
        .send()
        .await
        .map_err(|e| CarnotError::Provider {
            provider: profile.into(),
            message: format!("OAuth token exchange failed: {}", e.without_url()),
            retryable: true,
            retry_after_ms: None,
        })?;
    parse_token_response(profile, resp, Some(&client_id), Some(token_url), None).await
}

async fn run_device_code_flow<F>(
    profile: &str,
    cfg: &OAuthConfig,
    opts: &OAuthLoginOptions,
    on_prompt: &mut F,
) -> Result<OAuthStoredToken>
where
    F: FnMut(OAuthPrompt),
{
    let device_url = cfg.device_authorization_url.as_deref().ok_or_else(|| {
        oauth_err(
            "oauth",
            format!("provider `{profile}` OAuth config is missing `device_authorization_url`"),
        )
    })?;
    let token_url = cfg.token_url.as_deref().ok_or_else(|| {
        oauth_err(
            "oauth",
            format!("provider `{profile}` OAuth config is missing `token_url`"),
        )
    })?;
    let client_id = cfg
        .resolve_client_id(opts.client_id_override.as_deref())
        .ok_or_else(|| {
            oauth_err(
                "oauth",
                format!("provider `{profile}` device-code OAuth requires a `client_id`"),
            )
        })?;
    let client_secret = cfg.resolve_client_secret();

    let mut form: Vec<(String, String)> = vec![("client_id".into(), client_id.clone())];
    if !cfg.scopes.is_empty() {
        form.push(("scope".into(), cfg.scopes.join(" ")));
    }
    for (k, v) in &cfg.extra_auth_params {
        form.push((k.clone(), v.clone()));
    }

    let client = build_http_client_for_url(device_url);
    let resp = client
        .post(device_url)
        .header("accept", "application/json")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(encode_form(&form))
        .send()
        .await
        .map_err(|e| CarnotError::Provider {
            provider: profile.into(),
            message: format!("device authorization request failed: {}", e.without_url()),
            retryable: true,
            retry_after_ms: None,
        })?;
    let status = resp.status();
    let body_text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(CarnotError::Provider {
            provider: profile.into(),
            message: format!("device authorization HTTP {status}: {body_text}"),
            retryable: false,
            retry_after_ms: None,
        });
    }
    let body: Value = serde_json::from_str(&body_text).map_err(|e| CarnotError::Provider {
        provider: profile.into(),
        message: format!("invalid device authorization JSON: {e}"),
        retryable: false,
        retry_after_ms: None,
    })?;

    let device_code = body
        .get("device_code")
        .and_then(Value::as_str)
        .ok_or_else(|| CarnotError::Provider {
            provider: profile.into(),
            message: "device authorization response missing `device_code`".into(),
            retryable: false,
            retry_after_ms: None,
        })?
        .to_string();
    let user_code = body
        .get("user_code")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CarnotError::Provider {
            provider: profile.into(),
            message: "device authorization response missing `user_code`".into(),
            retryable: false,
            retry_after_ms: None,
        })?
        .to_string();
    let verification_uri = body
        .get("verification_uri")
        .or_else(|| body.get("verification_url"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CarnotError::Provider {
            provider: profile.into(),
            message: "device authorization response missing `verification_uri`".into(),
            retryable: false,
            retry_after_ms: None,
        })?
        .to_string();
    let verification_uri_complete = body
        .get("verification_uri_complete")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let expires_in = body
        .get("expires_in")
        .and_then(Value::as_u64)
        .unwrap_or(600);
    let mut interval_secs = body
        .get("interval")
        .and_then(Value::as_u64)
        .unwrap_or(5)
        .max(1);

    on_prompt(OAuthPrompt::DeviceCode {
        verification_uri: verification_uri.clone(),
        verification_uri_complete: verification_uri_complete.clone(),
        user_code,
    });
    if opts.open_browser {
        if let Some(ref full) = verification_uri_complete {
            try_open_browser(full);
        } else if !verification_uri.is_empty() {
            try_open_browser(&verification_uri);
        }
    }

    let deadline =
        tokio::time::Instant::now() + opts.timeout.min(Duration::from_secs(expires_in.max(10)));
    let token_client = build_http_client_for_url(token_url);
    let mut consecutive_net_errs: u32 = 0;

    loop {
        if tokio::time::Instant::now() >= deadline {
            return Err(CarnotError::Provider {
                provider: profile.into(),
                message: "device authorization timed out before approval".into(),
                retryable: false,
                retry_after_ms: None,
            });
        }
        tokio::time::sleep(Duration::from_secs(interval_secs)).await;

        let mut poll_form: Vec<(String, String)> = vec![
            (
                "grant_type".into(),
                "urn:ietf:params:oauth:grant-type:device_code".into(),
            ),
            ("device_code".into(), device_code.clone()),
            ("client_id".into(), client_id.clone()),
        ];
        if let Some(ref secret) = client_secret {
            poll_form.push(("client_secret".into(), secret.clone()));
        }
        for (k, v) in &cfg.extra_token_params {
            poll_form.push((k.clone(), v.clone()));
        }

        let r = match token_client
            .post(token_url)
            .header("accept", "application/json")
            .header("content-type", "application/x-www-form-urlencoded")
            .body(encode_form(&poll_form))
            .send()
            .await
        {
            Ok(resp) => {
                consecutive_net_errs = 0;
                resp
            }
            Err(e) => {
                consecutive_net_errs += 1;
                if consecutive_net_errs >= 5 {
                    return Err(CarnotError::Provider {
                        provider: profile.into(),
                        message: format!(
                            "device token poll failed after {consecutive_net_errs} consecutive network errors: {}",
                            e.without_url()
                        ),
                        retryable: true,
                        retry_after_ms: None,
                    });
                }
                tracing::debug!(
                    provider = %profile,
                    attempt = consecutive_net_errs,
                    "transient network error during device-code poll; retrying"
                );
                continue;
            }
        };

        let status = r.status();
        let text = r.text().await.unwrap_or_default();
        if status.is_success() {
            let v: Value = serde_json::from_str(&text).map_err(|e| CarnotError::Provider {
                provider: profile.into(),
                message: format!("invalid token JSON: {e}"),
                retryable: false,
                retry_after_ms: None,
            })?;
            if let Some(err_code) = v.get("error").and_then(Value::as_str) {
                match err_code {
                    "authorization_pending" => continue,
                    "slow_down" => {
                        interval_secs = interval_secs.saturating_add(5);
                        continue;
                    }
                    other => {
                        return Err(CarnotError::Provider {
                            provider: profile.into(),
                            message: format!("device authorization failed: {other}"),
                            retryable: false,
                            retry_after_ms: None,
                        });
                    }
                }
            }
            return parse_token_value(profile, &v, Some(&client_id), Some(token_url), None);
        }

        if let Ok(v) = serde_json::from_str::<Value>(&text)
            && let Some(err_code) = v.get("error").and_then(Value::as_str)
        {
            match err_code {
                "authorization_pending" => continue,
                "slow_down" => {
                    interval_secs = interval_secs.saturating_add(5);
                    continue;
                }
                "expired_token" => {
                    return Err(CarnotError::Provider {
                        provider: profile.into(),
                        message: "device code expired before authorization completed".into(),
                        retryable: false,
                        retry_after_ms: None,
                    });
                }
                "access_denied" => {
                    return Err(CarnotError::Provider {
                        provider: profile.into(),
                        message: "authorization was denied by user".into(),
                        retryable: false,
                        retry_after_ms: None,
                    });
                }
                other => {
                    return Err(CarnotError::Provider {
                        provider: profile.into(),
                        message: format!("device token error HTTP {status}: {other}"),
                        retryable: false,
                        retry_after_ms: None,
                    });
                }
            }
        }
        return Err(CarnotError::Provider {
            provider: profile.into(),
            message: format!("device token poll HTTP {status}: {text}"),
            retryable: false,
            retry_after_ms: None,
        });
    }
}

async fn exchange_refresh_token(
    profile: &str,
    current: &OAuthStoredToken,
    oauth_cfg: Option<&OAuthConfig>,
) -> Result<OAuthStoredToken> {
    let refresh_token = current
        .refresh_token
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CarnotError::Provider {
            provider: profile.into(),
            message: format!("stored token for `{profile}` has no refresh_token"),
            retryable: false,
            retry_after_ms: None,
        })?;
    let token_url = current
        .token_url
        .as_deref()
        .or_else(|| oauth_cfg.and_then(|c| c.token_url.as_deref()))
        .ok_or_else(|| CarnotError::Provider {
            provider: profile.into(),
            message: format!("no OAuth `token_url` configured to refresh `{profile}`"),
            retryable: false,
            retry_after_ms: None,
        })?;
    let client_id = current
        .client_id
        .clone()
        .or_else(|| oauth_cfg.and_then(|c| c.resolve_client_id(None)));
    let client_secret = oauth_cfg.and_then(|c| c.resolve_client_secret());

    let mut form: Vec<(String, String)> = vec![
        ("grant_type".into(), "refresh_token".into()),
        ("refresh_token".into(), refresh_token.to_string()),
    ];
    if let Some(ref cid) = client_id {
        form.push(("client_id".into(), cid.clone()));
    }
    if let Some(secret) = client_secret {
        form.push(("client_secret".into(), secret));
    }
    if let Some(cfg) = oauth_cfg {
        for (k, v) in &cfg.extra_token_params {
            form.push((k.clone(), v.clone()));
        }
    }

    let client = build_http_client_for_url(token_url);
    let resp = client
        .post(token_url)
        .header("accept", "application/json")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(encode_form(&form))
        .send()
        .await
        .map_err(|e| CarnotError::Provider {
            provider: profile.into(),
            message: format!("OAuth token refresh failed: {}", e.without_url()),
            retryable: true,
            retry_after_ms: None,
        })?;

    parse_token_response(
        profile,
        resp,
        client_id.as_deref(),
        Some(token_url),
        Some(refresh_token),
    )
    .await
}

async fn parse_token_response(
    profile: &str,
    resp: reqwest::Response,
    client_id: Option<&str>,
    token_url: Option<&str>,
    fallback_refresh_token: Option<&str>,
) -> Result<OAuthStoredToken> {
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(CarnotError::Provider {
            provider: profile.into(),
            message: format!("OAuth token endpoint HTTP {status}: {text}"),
            retryable: status.is_server_error(),
            retry_after_ms: None,
        });
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| CarnotError::Provider {
        provider: profile.into(),
        message: format!("invalid JSON from OAuth token endpoint: {e}"),
        retryable: false,
        retry_after_ms: None,
    })?;
    parse_token_value(profile, &v, client_id, token_url, fallback_refresh_token)
}

fn parse_token_value(
    profile: &str,
    v: &Value,
    client_id: Option<&str>,
    token_url: Option<&str>,
    fallback_refresh_token: Option<&str>,
) -> Result<OAuthStoredToken> {
    let access_token = v
        .get("access_token")
        .or_else(|| v.get("key"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| CarnotError::Provider {
            provider: profile.into(),
            message: "OAuth token response missing `access_token`".into(),
            retryable: false,
            retry_after_ms: None,
        })?
        .to_string();

    let refresh_token = v
        .get("refresh_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .or_else(|| fallback_refresh_token.map(ToString::to_string));

    let token_type = v
        .get("token_type")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("Bearer")
        .to_string();

    let now = now_epoch_secs();
    let expires_at_epoch_secs = v
        .get("expires_in")
        .and_then(Value::as_u64)
        .map(|secs| now.saturating_add(secs));

    let scope = v
        .get("scope")
        .and_then(Value::as_str)
        .map(ToString::to_string);

    Ok(OAuthStoredToken {
        access_token,
        refresh_token,
        token_type,
        expires_at_epoch_secs,
        scope,
        client_id: client_id.map(ToString::to_string),
        token_url: token_url.map(ToString::to_string),
        updated_at_epoch_secs: now,
    })
}

const CALLBACK_CONN_READ_TIMEOUT: Duration = Duration::from_secs(5);

async fn handle_loopback_connection(
    mut stream: tokio::net::TcpStream,
    expected_path: String,
    expected_state: Option<String>,
    require_state: bool,
    tx: tokio::sync::mpsc::Sender<Result<(String, Option<String>)>>,
) {
    let mut buf = [0u8; 8192];
    let n = match tokio::time::timeout(CALLBACK_CONN_READ_TIMEOUT, stream.read(&mut buf)).await {
        Ok(Ok(n)) if n > 0 => n,
        _ => return,
    };
    let req_str = String::from_utf8_lossy(&buf[..n]);
    let first_line = req_str.lines().next().unwrap_or("");
    let target = first_line.split_whitespace().nth(1).unwrap_or("/");

    let Ok(parsed) = Url::parse(&format!("http://127.0.0.1{target}")) else {
        let _ = write_http_response(&mut stream, "400 Bad Request", "Malformed request.").await;
        return;
    };

    // Exact path match (ignores `/favicon.ico`, `/callbackXYZ`, or stray probes without aborting).
    if parsed.path() != expected_path {
        let _ = write_http_response(&mut stream, "404 Not Found", "Not found").await;
        return;
    }

    let params: BTreeMap<String, String> = parsed.query_pairs().into_owned().collect();

    // Validate CSRF `state` BEFORE checking `error` or `code`, so untrusted local probes cannot
    // abort an active login with `?error=x` or inject a foreign authorization code.
    if let Some(ref exp) = expected_state {
        match params.get("state").map(String::as_str) {
            Some(got) if got == exp.as_str() => {}
            Some(_) => {
                let _ = write_http_response(
                    &mut stream,
                    "400 Bad Request",
                    "OAuth state mismatch. Please retry `carnot auth login`.",
                )
                .await;
                return;
            }
            None if require_state => {
                let _ = write_http_response(
                    &mut stream,
                    "400 Bad Request",
                    "Missing OAuth state parameter. Please retry `carnot auth login`.",
                )
                .await;
                return;
            }
            None => {}
        }
    }

    if let Some(err) = params.get("error") {
        let desc = params
            .get("error_description")
            .map(|d| format!(": {d}"))
            .unwrap_or_default();
        let _ = write_http_response(
            &mut stream,
            "400 Bad Request",
            "Authentication failed. You may close this tab and return to Carnot.",
        )
        .await;
        let _ = tx
            .send(Err(oauth_err(
                "oauth",
                format!("OAuth provider returned error `{err}`{desc}"),
            )))
            .await;
        return;
    }

    let Some(code) = params.get("code").filter(|s| !s.is_empty()) else {
        let _ = write_http_response(
            &mut stream,
            "400 Bad Request",
            "Missing authorization code in callback.",
        )
        .await;
        return;
    };

    let _ = write_http_response(
        &mut stream,
        "200 OK",
        "Carnot authentication succeeded. You may close this tab and return to your terminal.",
    )
    .await;
    let _ = tx
        .send(Ok((code.clone(), params.get("state").cloned())))
        .await;
}

async fn wait_for_loopback_callback(
    listener: TcpListener,
    expected_path: &str,
    expected_state: Option<&str>,
    require_state: bool,
    timeout: Duration,
) -> Result<(String, Option<String>)> {
    let deadline = tokio::time::Instant::now() + timeout;
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<(String, Option<String>)>>(8);

    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(oauth_err(
                "oauth",
                "OAuth login timed out waiting for browser callback".into(),
            ));
        }

        tokio::select! {
            res = rx.recv() => {
                if let Some(outcome) = res {
                    return outcome;
                }
            }
            accepted = tokio::time::timeout(remaining, listener.accept()) => {
                let (stream, _) = accepted
                    .map_err(|_| {
                        oauth_err(
                            "oauth",
                            "OAuth login timed out waiting for browser callback".into(),
                        )
                    })?
                    .map_err(|e| oauth_err("oauth", format!("OAuth callback accept failed: {e}")))?;

                let path = expected_path.to_string();
                let state = expected_state.map(ToString::to_string);
                let tx_clone = tx.clone();
                tokio::spawn(async move {
                    handle_loopback_connection(stream, path, state, require_state, tx_clone).await;
                });
            }
        }
    }
}

async fn write_http_response(
    stream: &mut tokio::net::TcpStream,
    status: &str,
    message: &str,
) -> std::io::Result<()> {
    let html = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Carnot OAuth</title></head>\
         <body style=\"font-family:system-ui,sans-serif;max-width:480px;margin:4rem auto;padding:1rem;\">\
         <h2>Carnot</h2><p>{message}</p></body></html>"
    );
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{html}",
        html.len()
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await
}

/// Extracts an authorization `code` from either a full pasted callback URL or a bare code string.
pub fn extract_code_from_manual_input(input: &str, expected_state: Option<&str>) -> Result<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(oauth_err(
            "oauth",
            "empty authorization code / callback URL".into(),
        ));
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        let parsed = Url::parse(trimmed)
            .map_err(|e| oauth_err("oauth", format!("invalid pasted callback URL: {e}")))?;
        let params: BTreeMap<String, String> = parsed.query_pairs().into_owned().collect();
        if let Some(exp) = expected_state
            && params.get("state").map(String::as_str) != Some(exp)
        {
            return Err(oauth_err(
                "oauth",
                "OAuth callback CSRF `state` mismatch".into(),
            ));
        }
        if let Some(err) = params.get("error") {
            return Err(oauth_err(
                "oauth",
                format!("OAuth callback contained error `{err}`"),
            ));
        }
        return params
            .get("code")
            .filter(|s| !s.is_empty())
            .cloned()
            .ok_or_else(|| {
                oauth_err(
                    "oauth",
                    "pasted callback URL is missing `code` query parameter".into(),
                )
            });
    }
    Ok(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn sha256_matches_nist_and_rfc7636_pkce_s256_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // RFC 7636 Appendix B official test vector:
        assert_eq!(
            pkce_s256_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn pkce_pair_and_state_meet_rfc7636_lengths() {
        let (v, c) = generate_pkce_pair();
        assert_eq!(v.len(), 64);
        assert_eq!(c, pkce_s256_challenge(&v));
        let st = generate_oauth_state();
        assert!(st.len() >= 40);
        assert_ne!(v, st);
    }

    #[test]
    fn auth_store_atomic_roundtrip_and_redacted_debug() {
        let dir = std::env::temp_dir().join(format!("carnot-oauth-test-{}", Uuid::new_v4()));
        let store = AuthStore::open(dir.join("auth.json"));

        let saved =
            save_manual_token("openrouter", "sk-or-secret-token-123", &store).expect("save token");
        let dbg = format!("{saved:?}");
        assert!(!dbg.contains("sk-or-secret-token-123"));
        assert!(dbg.contains("<redacted>"));

        let loaded = store.get("openrouter").expect("read").expect("present");
        assert_eq!(loaded.access_token, "sk-or-secret-token-123");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(store.path())
                .expect("meta")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        }

        assert!(store.remove("openrouter").expect("remove"));
        assert!(store.get("openrouter").expect("read").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn standard_pkce_login_and_refresh_end_to_end() {
        // Start a mock OAuth token endpoint on 127.0.0.1.
        let mock_listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind mock");
        let mock_addr = mock_listener.local_addr().expect("addr");
        let server_task = tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut s, _) = mock_listener.accept().await.expect("accept");
                let mut buf = [0u8; 4096];
                let n = s.read(&mut buf).await.expect("read");
                let req = String::from_utf8_lossy(&buf[..n]);
                let body = if attempt == 0 {
                    assert!(req.contains("grant_type=authorization_code"));
                    assert!(req.contains("code=test-auth-code"));
                    assert!(req.contains("code_verifier="));
                    r#"{"access_token":"tok-initial","refresh_token":"ref-1","expires_in":1,"token_type":"Bearer"}"#
                } else {
                    assert!(req.contains("grant_type=refresh_token"));
                    assert!(req.contains("refresh_token=ref-1"));
                    r#"{"access_token":"tok-refreshed","expires_in":3600,"token_type":"Bearer"}"#
                };
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                s.write_all(resp.as_bytes()).await.expect("write");
            }
        });

        let dir = std::env::temp_dir().join(format!("carnot-oauth-pkce-{}", Uuid::new_v4()));
        let store = AuthStore::open(dir.join("auth.json"));
        let cfg = OAuthConfig {
            flow: OAuthFlow::Pkce,
            authorize_url: Some("https://auth.example.test/authorize".into()),
            token_url: Some(format!("http://{mock_addr}/token")),
            client_id: Some("test-client".into()),
            scopes: vec!["read".into(), "write".into()],
            ..Default::default()
        };
        let opts = OAuthLoginOptions {
            open_browser: false,
            timeout: Duration::from_secs(10),
            ..Default::default()
        };

        let tok = login_with_oauth(
            "custom-oidc",
            &cfg,
            &store,
            &opts,
            |prompt| {
                if let OAuthPrompt::OpenUrl {
                    url, callback_url, ..
                } = prompt
                {
                    let parsed = Url::parse(&url).expect("valid auth url");
                    let params: BTreeMap<String, String> =
                        parsed.query_pairs().into_owned().collect();
                    let st = params.get("state").cloned().expect("state");
                    let hit_url = format!("{callback_url}?code=test-auth-code&state={st}");
                    tokio::spawn(async move {
                        let _ = reqwest::get(&hit_url).await;
                    });
                }
            },
            || unreachable!(),
        )
        .await
        .expect("pkce login");

        assert_eq!(tok.access_token, "tok-initial");
        assert_eq!(tok.refresh_token.as_deref(), Some("ref-1"));

        // Now verify automatic proactive refresh (expires_in=1 < skew 120s) preserves refresh_token!
        let mgr = OAuthTokenManager::new("custom-oidc", tok, Some(cfg), store.clone());
        let fresh = mgr.access_token(None).await.expect("refresh");
        assert_eq!(fresh, "tok-refreshed");

        let persisted = store.get("custom-oidc").expect("read").expect("present");
        assert_eq!(persisted.access_token, "tok-refreshed");
        assert_eq!(persisted.refresh_token.as_deref(), Some("ref-1"));

        server_task.await.expect("mock done");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn pkce_key_exchange_end_to_end() {
        let mock_listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
        let mock_addr = mock_listener.local_addr().expect("addr");
        let server_task = tokio::spawn(async move {
            let (mut s, _) = mock_listener.accept().await.expect("accept");
            let mut buf = [0u8; 4096];
            let n = s.read(&mut buf).await.expect("read");
            let req = String::from_utf8_lossy(&buf[..n]);
            assert!(req.contains("\"code\":\"or-code-99\""));
            assert!(req.contains("\"code_challenge_method\":\"S256\""));
            let body = r#"{"key":"sk-test-v1-live-key"}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            s.write_all(resp.as_bytes()).await.expect("write");
        });

        let dir = std::env::temp_dir().join(format!("carnot-oauth-kx-{}", Uuid::new_v4()));
        let store = AuthStore::open(dir.join("auth.json"));
        let cfg = OAuthConfig {
            flow: OAuthFlow::PkceKeyExchange,
            authorize_url: Some("https://auth.example.test/auth".into()),
            exchange_url: Some(format!("http://{mock_addr}/api/v1/auth/keys")),
            ..Default::default()
        };
        let opts = OAuthLoginOptions {
            open_browser: false,
            timeout: Duration::from_secs(10),
            ..Default::default()
        };

        let tok = login_with_oauth(
            "gateway-kx",
            &cfg,
            &store,
            &opts,
            |prompt| {
                if let OAuthPrompt::OpenUrl { callback_url, .. } = prompt {
                    let hit_url = format!("{callback_url}?code=or-code-99");
                    tokio::spawn(async move {
                        let _ = reqwest::get(&hit_url).await;
                    });
                }
            },
            || unreachable!(),
        )
        .await
        .expect("pkce key exchange");

        assert_eq!(tok.access_token, "sk-test-v1-live-key");
        server_task.await.expect("server");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn device_code_flow_handles_pending_and_succeeds() {
        let mock_listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
        let mock_addr = mock_listener.local_addr().expect("addr");
        let server_task = tokio::spawn(async move {
            for step in 0..3 {
                let (mut s, _) = mock_listener.accept().await.expect("accept");
                let mut buf = [0u8; 4096];
                let _ = s.read(&mut buf).await.expect("read");
                let (status, body) = match step {
                    0 => (
                        "200 OK",
                        r#"{"device_code":"dev-123","user_code":"ABCD-EFGH","verification_uri":"https://auth.example.test/device","expires_in":60,"interval":1}"#,
                    ),
                    1 => ("400 Bad Request", r#"{"error":"authorization_pending"}"#),
                    _ => (
                        "200 OK",
                        r#"{"access_token":"tok-device-ok","refresh_token":"ref-dev","expires_in":7200}"#,
                    ),
                };
                let resp = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                s.write_all(resp.as_bytes()).await.expect("write");
            }
        });

        let dir = std::env::temp_dir().join(format!("carnot-oauth-dev-{}", Uuid::new_v4()));
        let store = AuthStore::open(dir.join("auth.json"));
        let cfg = OAuthConfig {
            flow: OAuthFlow::DeviceCode,
            device_authorization_url: Some(format!("http://{mock_addr}/device")),
            token_url: Some(format!("http://{mock_addr}/token")),
            client_id: Some("dev-client".into()),
            ..Default::default()
        };
        let opts = OAuthLoginOptions {
            open_browser: false,
            timeout: Duration::from_secs(15),
            ..Default::default()
        };
        let mut saw_user_code = false;
        let tok = login_with_oauth(
            "device-prov",
            &cfg,
            &store,
            &opts,
            |prompt| {
                if let OAuthPrompt::DeviceCode { user_code, .. } = prompt {
                    assert_eq!(user_code, "ABCD-EFGH");
                    saw_user_code = true;
                }
            },
            || unreachable!(),
        )
        .await
        .expect("device flow");

        assert!(saw_user_code);
        assert_eq!(tok.access_token, "tok-device-ok");
        assert_eq!(tok.refresh_token.as_deref(), Some("ref-dev"));
        server_task.await.expect("server");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn loopback_callback_respects_timeout_even_with_idle_socket() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
        let addr = listener.local_addr().expect("addr");

        // Open an idle socket that sends nothing (simulates Chrome speculative preconnect / stall).
        let _idle_socket = tokio::net::TcpStream::connect(addr).await.expect("connect");

        let start = tokio::time::Instant::now();
        let res = wait_for_loopback_callback(
            listener,
            "/callback",
            Some("expected-state"),
            true,
            Duration::from_millis(350),
        )
        .await;
        assert!(res.is_err(), "idle socket must NOT defeat login timeout");
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "expected prompt timeout, elapsed {:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn loopback_callback_ignores_idle_socket_bad_path_and_spoofed_error_probe() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
        let addr = listener.local_addr().expect("addr");

        let client_task = tokio::spawn(async move {
            // 1. Keep an idle socket open throughout!
            let _idle = tokio::net::TcpStream::connect(addr).await.expect("idle");
            let client = reqwest::Client::new();
            // 2. Hit `/callbackXYZ` (prefix trick) -> must get 404 and not complete flow
            let r1 = client
                .get(format!("http://{addr}/callbackXYZ?code=evil&state=good-st"))
                .send()
                .await
                .expect("r1");
            assert_eq!(r1.status(), reqwest::StatusCode::NOT_FOUND);
            // 3. Hit `?error=access_denied` with WRONG state -> must get 400 and NOT abort flow
            let r2 = client
                .get(format!(
                    "http://{addr}/callback?error=access_denied&state=wrong-st"
                ))
                .send()
                .await
                .expect("r2");
            assert_eq!(r2.status(), reqwest::StatusCode::BAD_REQUEST);
            // 4. Send the real browser callback with valid state -> must succeed immediately!
            let r3 = client
                .get(format!(
                    "http://{addr}/callback?code=real-code&state=good-st"
                ))
                .send()
                .await
                .expect("r3");
            assert_eq!(r3.status(), reqwest::StatusCode::OK);
        });

        let (code, st) = wait_for_loopback_callback(
            listener,
            "/callback",
            Some("good-st"),
            true,
            Duration::from_secs(5),
        )
        .await
        .expect("real callback should succeed despite idle socket and spoofed probes");

        assert_eq!(code, "real-code");
        assert_eq!(st.as_deref(), Some("good-st"));
        client_task.await.expect("client task");
    }

    #[tokio::test]
    async fn shared_manager_deduplicates_concurrent_expiry_and_401_refreshes() {
        use std::sync::atomic::AtomicUsize;

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let refresh_count = Arc::new(AtomicUsize::new(0));
        let count_srv = Arc::clone(&refresh_count);

        let server_task = tokio::spawn(async move {
            // Serve up to 2 sequential refresh calls (1 for expiry wave, 1 for 401 wave).
            for wave in 0..2 {
                let (mut s, _) = listener.accept().await.expect("accept");
                let mut buf = [0u8; 4096];
                let n = s.read(&mut buf).await.expect("read");
                let req = String::from_utf8_lossy(&buf[..n]);
                assert!(req.contains("grant_type=refresh_token"));
                let c = count_srv.fetch_add(1, Ordering::SeqCst);
                // Small delay so the concurrent waiter definitely queues behind the mutex.
                tokio::time::sleep(Duration::from_millis(60)).await;
                let body = if wave == 0 {
                    assert!(req.contains("refresh_token=ref-0"));
                    format!(
                        r#"{{"access_token":"tok-wave1-{c}","refresh_token":"ref-1","expires_in":3600}}"#
                    )
                } else {
                    assert!(req.contains("refresh_token=ref-1"));
                    format!(
                        r#"{{"access_token":"tok-wave2-{c}","refresh_token":"ref-2","expires_in":3600}}"#
                    )
                };
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                s.write_all(resp.as_bytes()).await.expect("write");
            }
        });

        let dir = std::env::temp_dir().join(format!("carnot-oauth-sf-{}", Uuid::new_v4()));
        let store = AuthStore::open(dir.join("auth.json"));
        let expiring = OAuthStoredToken {
            access_token: "tok-0".into(),
            refresh_token: Some("ref-0".into()),
            token_type: "Bearer".into(),
            expires_at_epoch_secs: Some(now_epoch_secs() + 1),
            scope: None,
            client_id: Some("cid".into()),
            token_url: Some(format!("http://{addr}/token")),
            updated_at_epoch_secs: now_epoch_secs(),
        };
        store.save("sf-prof", expiring.clone()).expect("save");
        let cfg = OAuthConfig {
            flow: OAuthFlow::Pkce,
            token_url: Some(format!("http://{addr}/token")),
            client_id: Some("cid".into()),
            ..Default::default()
        };

        // Two callers resolving via `OAuthTokenManager::shared` share ONE single-flight manager!
        let m1 = OAuthTokenManager::shared(
            "sf-prof",
            expiring.clone(),
            Some(cfg.clone()),
            store.clone(),
        );
        let m2 = OAuthTokenManager::shared("sf-prof", expiring, Some(cfg), store.clone());
        assert!(Arc::ptr_eq(&m1, &m2));

        // Wave 1: two concurrent proactive expiry refreshes -> exactly ONE HTTP refresh!
        let (a, b) = tokio::join!(m1.access_token(None), m2.access_token(None));
        let tok1 = a.expect("a");
        let tok2 = b.expect("b");
        assert_eq!(tok1, tok2);
        assert_eq!(refresh_count.load(Ordering::SeqCst), 1);

        // Wave 2: two concurrent HTTP 401 retries reporting the SAME rejected token -> exactly ONE refresh!
        let (c, d) = tokio::join!(
            m1.access_token(Some(tok1.as_str())),
            m2.access_token(Some(tok1.as_str()))
        );
        let tok3 = c.expect("c");
        let tok4 = d.expect("d");
        assert_eq!(tok3, tok4);
        assert_ne!(tok3, tok1);
        assert_eq!(refresh_count.load(Ordering::SeqCst), 2);

        let on_disk = store.get("sf-prof").expect("disk").expect("present");
        assert_eq!(on_disk.access_token, tok3);
        assert_eq!(on_disk.refresh_token.as_deref(), Some("ref-2"));

        OAuthTokenManager::evict(store.path(), "sf-prof");
        server_task.await.expect("server");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn auth_store_concurrent_saves_and_logout_during_refresh_are_safe() {
        let dir = std::env::temp_dir().join(format!("carnot-oauth-lock-{}", Uuid::new_v4()));
        let store = AuthStore::open(dir.join("auth.json"));

        std::thread::scope(|scope| {
            for idx in 0..8 {
                let st = store.clone();
                scope.spawn(move || {
                    let prof = format!("prof-{idx}");
                    save_manual_token(&prof, &format!("tok-{idx}"), &st).expect("save");
                });
            }
        });

        let all = store.list().expect("list");
        assert_eq!(
            all.len(),
            8,
            "file lock must prevent lost writes across concurrent saves"
        );
        for idx in 0..8 {
            assert_eq!(
                all[&format!("prof-{idx}")].access_token,
                format!("tok-{idx}")
            );
        }

        // Removing a profile means `update_if_present` must NOT resurrect it!
        assert!(store.remove("prof-0").expect("remove"));
        let updated = store
            .update_if_present(
                "prof-0",
                OAuthStoredToken {
                    access_token: "resurrected".into(),
                    refresh_token: None,
                    token_type: "Bearer".into(),
                    expires_at_epoch_secs: None,
                    scope: None,
                    client_id: None,
                    token_url: None,
                    updated_at_epoch_secs: now_epoch_secs(),
                },
            )
            .expect("update_if_present");
        assert!(!updated);
        assert!(store.get("prof-0").expect("get").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
