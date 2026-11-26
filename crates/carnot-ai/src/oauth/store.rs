//! Secure local OAuth token store (`~/.carnot/auth.json`) and single-flight token manager.

use super::flows::exchange_refresh_token;
use super::{DEFAULT_REFRESH_SKEW_SECS, OAuthConfig, OAuthStoredToken, oauth_err};
use carnot_core::{CarnotError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use tokio::sync::Mutex;
use uuid::Uuid;

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
    static REGISTRY: LazyLock<ManagerRegistry> =
        LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));
    &REGISTRY
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
            .finish_non_exhaustive()
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
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
            .and_then(std::result::Result::ok)
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
