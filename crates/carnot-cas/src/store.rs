use carnot_core::{CarnotError, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime};

/// Metadata for an immutable text blob stored under `.carnot/h/<blake3>.log`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CasHandle {
    /// Full 64-char hex BLAKE3 digest (256-bit, collision-free).
    pub hash: String,
    /// Absolute path on disk (readable via `Read{offset,limit}` or `Bash rg`).
    pub path: PathBuf,
    /// Relative display path inside workspace (`.carnot/h/<hash>.log`).
    pub rel_path: String,
    pub bytes: usize,
    pub lines: usize,
}

#[derive(Debug, Clone)]
struct CasLocation {
    root: PathBuf,
    display_prefix: String,
}

/// Atomic, read-verified, 256-bit BLAKE3 file-handle store.
///
/// Initialization is lazy: `CasStore::open()` performs zero filesystem syscalls so clean
/// workspaces are never polluted with an empty `.carnot/` folder unless a spill actually occurs.
#[derive(Debug, Clone)]
pub struct CasStore {
    workspace_root: PathBuf,
    location: Arc<OnceLock<CasLocation>>,
}

impl CasStore {
    /// Deterministic temp fallback path for read-only / non-writable workspaces (zero syscalls).
    pub fn fallback_root(workspace_root: &Path) -> PathBuf {
        let ws_hash = blake3::hash(workspace_root.to_string_lossy().as_bytes())
            .to_hex()
            .to_string();
        std::env::temp_dir()
            .join(format!("carnot-cas-{}", &ws_hash[..12]))
            .join("h")
    }

    /// Creates a lazy `CasStore` handle bound to `workspace_root` (zero startup I/O).
    /// On first actual write or `root()` probe, initializes `.carnot/h/` (and `.git/info/exclude`)
    /// or falls back transparently to `$TMPDIR/carnot-cas-<hash>/h` if `workspace_root` is read-only.
    pub fn open(workspace_root: &Path) -> Result<Self> {
        Ok(Self {
            workspace_root: workspace_root.to_path_buf(),
            location: Arc::new(OnceLock::new()),
        })
    }

    fn resolve_location(&self) -> Result<&CasLocation> {
        if let Some(loc) = self.location.get() {
            return Ok(loc);
        }
        let local_root = self.workspace_root.join(".carnot").join("h");
        let probe = local_root.join(format!(".carnot-probe-{}", std::process::id()));
        let local_ok = fs::create_dir_all(&local_root).is_ok()
            && fs::File::create(&probe)
                .map(|_| {
                    let _ = fs::remove_file(&probe);
                })
                .is_ok();
        let computed = if local_ok {
            ensure_git_exclude(&self.workspace_root);
            CasLocation {
                root: local_root,
                display_prefix: ".carnot/h".to_string(),
            }
        } else {
            let fallback_root = Self::fallback_root(&self.workspace_root);
            fs::create_dir_all(&fallback_root)?;
            let display_prefix = fallback_root.to_string_lossy().to_string();
            CasLocation {
                root: fallback_root,
                display_prefix,
            }
        };
        let _ = self.location.set(computed);
        Ok(self.location.get().expect("cas location initialized"))
    }

    pub fn root(&self) -> &Path {
        self.resolve_location()
            .map(|l| l.root.as_path())
            .unwrap_or_else(|_| self.workspace_root.as_path())
    }

    /// Stores raw bytes atomically (`tmp + sync_all + rename`) under their full 256-bit BLAKE3 key.
    /// Supports both UTF-8 logs and arbitrary binary snapshots (used by atomic CAS Undo!).
    pub fn put_bytes(&self, bytes: &[u8]) -> Result<CasHandle> {
        let loc = self.resolve_location()?;
        let hash = blake3::hash(bytes).to_hex().to_string();
        let file_name = format!("{hash}.log");
        let path = loc.root.join(&file_name);
        let rel_path = format!("{}/{file_name}", loc.display_prefix);
        let lines = bytes.iter().filter(|&&b| b == b'\n').count() + usize::from(!bytes.is_empty());

        if path.exists()
            && let Ok(existing) = fs::read(&path)
            && blake3::hash(&existing).to_hex().as_str() == hash
        {
            return Ok(CasHandle {
                hash,
                path,
                rel_path,
                bytes: bytes.len(),
                lines,
            });
        }

        let tmp = loc.root.join(format!(
            ".{hash}.{}.{}.tmp",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(bytes)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, &path)?;

        Ok(CasHandle {
            hash,
            path,
            rel_path,
            bytes: bytes.len(),
            lines,
        })
    }

    /// Stores UTF-8 text atomically (`tmp + sync_all + rename`) under its full 256-bit BLAKE3 key.
    pub fn put_text(&self, text: &str) -> Result<CasHandle> {
        let mut h = self.put_bytes(text.as_bytes())?;
        h.lines = text.lines().count();
        Ok(h)
    }

    /// Reads back raw bytes and verifies their 256-bit BLAKE3 digest.
    pub fn get_bytes_verified(&self, hash: &str) -> Result<Vec<u8>> {
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(CarnotError::Session(format!(
                "invalid CAS hash key (expected 64 hex chars): {hash}"
            )));
        }
        let loc = self.resolve_location()?;
        let path = loc.root.join(format!("{hash}.log"));
        let raw = fs::read(&path)?;
        let actual = blake3::hash(&raw).to_hex().to_string();
        if actual != hash {
            return Err(CarnotError::Session(format!(
                "CAS integrity mismatch for {hash}: got {actual}"
            )));
        }
        Ok(raw)
    }

    /// Reads back a UTF-8 blob and verifies its 256-bit BLAKE3 digest.
    pub fn get_verified(&self, hash: &str) -> Result<String> {
        let raw = self.get_bytes_verified(hash)?;
        String::from_utf8(raw).map_err(|e| CarnotError::Session(format!("CAS non-UTF8 blob: {e}")))
    }

    /// Removes handle blobs older than `max_age` (hygiene for logs that may contain secrets).
    pub fn gc_older_than(&self, max_age: Duration) -> Result<usize> {
        let dir = match self.location.get() {
            Some(loc) => loc.root.clone(),
            None => {
                let local = self.workspace_root.join(".carnot").join("h");
                if local.is_dir() {
                    local
                } else {
                    return Ok(0);
                }
            }
        };
        let now = SystemTime::now();
        let mut removed = 0usize;
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("log") {
                continue;
            }
            if let Ok(meta) = entry.metadata()
                && let Ok(modified) = meta.modified()
                && let Ok(age) = now.duration_since(modified)
                && age > max_age
                && fs::remove_file(&path).is_ok()
            {
                removed += 1;
            }
        }
        Ok(removed)
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }
}

/// Appends `.carnot/` to the repo's exact git exclude file in pure Rust (0 subprocesses!)
/// (handles normal repos, linked worktrees via `gitdir:` + `commondir`, and submodules).
fn ensure_git_exclude(workspace_root: &Path) {
    let Some(exclude) = resolve_git_exclude_path(workspace_root) else {
        return;
    };
    let existing = fs::read_to_string(&exclude).unwrap_or_default();
    if existing.lines().any(|l| l.trim() == ".carnot/") {
        return;
    }
    if let Some(parent) = exclude.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&exclude) {
        let prefix = if existing.is_empty() || existing.ends_with('\n') {
            ""
        } else {
            "\n"
        };
        let _ = writeln!(f, "{prefix}.carnot/");
    }
}

fn resolve_git_exclude_path(start: &Path) -> Option<PathBuf> {
    let mut cur = start;
    loop {
        let dot_git = cur.join(".git");
        if dot_git.is_dir() {
            return Some(dot_git.join("info").join("exclude"));
        }
        if dot_git.is_file() {
            let content = fs::read_to_string(&dot_git).ok()?;
            let gitdir_raw = content.trim().strip_prefix("gitdir:")?.trim();
            let gitdir = {
                let p = PathBuf::from(gitdir_raw);
                if p.is_absolute() { p } else { cur.join(p) }
            };
            let commondir_file = gitdir.join("commondir");
            let common = if commondir_file.is_file() {
                let rel = fs::read_to_string(&commondir_file).ok()?;
                let rel = rel.trim();
                let cp = PathBuf::from(rel);
                if cp.is_absolute() {
                    cp
                } else {
                    gitdir.join(cp)
                }
            } else {
                gitdir
            };
            return Some(common.join("info").join("exclude"));
        }
        cur = cur.parent()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_put_and_verified_read() {
        let dir = std::env::temp_dir().join(format!("carnot-cas-test-{}", uuid::Uuid::new_v4()));
        let store = CasStore::open(&dir).expect("open cas");
        let h = store.put_text("line1\nline2\n").expect("put");
        assert_eq!(h.hash.len(), 64);
        assert_eq!(h.lines, 2);
        let read_back = store.get_verified(&h.hash).expect("get");
        assert_eq!(read_back, "line1\nline2\n");

        // Corrupt the blob and confirm verification catches it.
        fs::write(&h.path, "tampered").expect("corrupt");
        assert!(store.get_verified(&h.hash).is_err());

        // Re-putting the original text heals the corrupted blob.
        let h2 = store.put_text("line1\nline2\n").expect("heal");
        assert_eq!(h2.hash, h.hash);
        assert_eq!(
            store.get_verified(&h2.hash).expect("healed"),
            "line1\nline2\n"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_is_lazy_and_does_not_pollute_clean_workspace() {
        let dir = std::env::temp_dir().join(format!("carnot-cas-lazy-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("mkdir");
        let store = CasStore::open(&dir).expect("lazy open");
        assert!(
            !dir.join(".carnot").exists(),
            "CasStore::open must not create .carnot/ until first write"
        );
        let h = store.put_text("first spill\n").expect("put");
        assert!(dir.join(".carnot").join("h").is_dir());
        assert_eq!(store.get_verified(&h.hash).expect("read"), "first spill\n");
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn read_only_workspace_falls_back_to_temp_cas() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("carnot-cas-ro-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("mkdir");
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).expect("chmod 555");

        let store = CasStore::open(&dir).expect("fallback cas should open cleanly");
        assert!(!store.root().starts_with(&dir));
        let h = store.put_text("hello from ro workspace\n").expect("put");
        assert!(
            h.rel_path
                .starts_with(store.root().to_string_lossy().as_ref())
        );
        assert_eq!(
            store.get_verified(&h.hash).expect("read"),
            "hello from ro workspace\n"
        );

        let _ = fs::set_permissions(&dir, fs::Permissions::from_mode(0o755));
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(store.root().parent().unwrap_or(store.root()));
    }
}
