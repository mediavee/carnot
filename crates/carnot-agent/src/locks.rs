//! Deadlock-free effect-based async lock coordinator for concurrent tool execution.

use carnot_core::ToolEffects;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Deadlock-free effect-based async lock coordinator used by [`Agent::invoke_tool_gated`].
///
/// Enforces the same conflict rules as Carnot's wave scheduler for nested orchestration calls
/// (`CodeMode` `Promise.all`, future extensions):
/// - `ToolEffects::serial()` acquires the exclusive global write lock.
/// - Non-serial tools acquire the shared global read lock, then acquire per-path `RwLock`s
///   (write for `effects.writes`, read for `effects.reads`) in canonical sorted path order.
#[derive(Debug, Default)]
pub(crate) struct ToolEffectLocks {
    serial: Arc<tokio::sync::RwLock<()>>,
    /// Ensures `global_read` tools (e.g. `Grep`) never race with any write, while still allowing
    /// disjoint writes (`Edit(a)` + `Edit(b)`) to execute in parallel (`global_read` acquires
    /// write; writers acquire read).
    global_read_vs_write: Arc<tokio::sync::RwLock<()>>,
    paths: std::sync::Mutex<HashMap<std::path::PathBuf, Arc<tokio::sync::RwLock<()>>>>,
}

#[derive(Debug)]
pub(crate) struct ToolEffectGuard {
    _serial_write: Option<tokio::sync::OwnedRwLockWriteGuard<()>>,
    _serial_read: Option<tokio::sync::OwnedRwLockReadGuard<()>>,
    _global_rw_write: Option<tokio::sync::OwnedRwLockWriteGuard<()>>,
    _global_rw_read: Option<tokio::sync::OwnedRwLockReadGuard<()>>,
    _path_writes: Vec<tokio::sync::OwnedRwLockWriteGuard<()>>,
    _path_reads: Vec<tokio::sync::OwnedRwLockReadGuard<()>>,
}

impl ToolEffectLocks {
    fn normalize_key(path: &Path) -> std::path::PathBuf {
        #[cfg(any(target_os = "macos", target_os = "windows"))]
        {
            std::path::PathBuf::from(path.to_string_lossy().to_ascii_lowercase())
        }
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            path.to_path_buf()
        }
    }

    fn path_lock(&self, key: &Path) -> Arc<tokio::sync::RwLock<()>> {
        let mut map = self
            .paths
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Arc::clone(
            map.entry(key.to_path_buf())
                .or_insert_with(|| Arc::new(tokio::sync::RwLock::new(()))),
        )
    }

    pub(crate) async fn acquire(&self, effects: &ToolEffects) -> ToolEffectGuard {
        if effects.serial_barrier {
            let w = Arc::clone(&self.serial).write_owned().await;
            return ToolEffectGuard {
                _serial_write: Some(w),
                _serial_read: None,
                _global_rw_write: None,
                _global_rw_read: None,
                _path_writes: Vec::new(),
                _path_reads: Vec::new(),
            };
        }
        let sr = Arc::clone(&self.serial).read_owned().await;

        let (global_rw_write, global_rw_read) = if effects.global_read {
            (
                Some(Arc::clone(&self.global_read_vs_write).write_owned().await),
                None,
            )
        } else if !effects.writes.is_empty() {
            (
                None,
                Some(Arc::clone(&self.global_read_vs_write).read_owned().await),
            )
        } else {
            (None, None)
        };

        // Deduplicate & sort all touched paths in canonical order (write wins over read) -> deadlock-free.
        let mut targets: std::collections::BTreeMap<std::path::PathBuf, bool> =
            std::collections::BTreeMap::new();
        for p in &effects.reads {
            targets.entry(Self::normalize_key(p)).or_insert(false);
        }
        for p in &effects.writes {
            targets.insert(Self::normalize_key(p), true);
        }

        let mut path_writes = Vec::new();
        let mut path_reads = Vec::new();
        for (key, is_write) in targets {
            let lock = self.path_lock(&key);
            if is_write {
                path_writes.push(lock.write_owned().await);
            } else {
                path_reads.push(lock.read_owned().await);
            }
        }
        ToolEffectGuard {
            _serial_write: None,
            _serial_read: Some(sr),
            _global_rw_write: global_rw_write,
            _global_rw_read: global_rw_read,
            _path_writes: path_writes,
            _path_reads: path_reads,
        }
    }
}

