//! Workspace path confinement, magic-byte image detection, and atomic permission-preserving writes.

use carnot_core::Result;
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

/// Detects binary raster image formats by magic bytes (PNG, JPEG, GIF87a/89a, WebP).
/// SVG stays plain UTF-8 text so models can inspect and edit it directly.
pub(super) fn detect_image_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() >= 8 && &bytes[..8] == b"\x89PNG\r\n\x1a\n" {
        Some("image/png")
    } else if bytes.len() >= 3 && &bytes[..3] == b"\xFF\xD8\xFF" {
        Some("image/jpeg")
    } else if bytes.len() >= 6 && (&bytes[..6] == b"GIF87a" || &bytes[..6] == b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

/// Lexically normalizes `.` and `..` without hitting the filesystem.
fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Resolves a candidate path while following existing AND dangling symlinks up to 16 hops
/// so a dangling symlink (`workspace/escape -> /etc/nonexistent`) can never bypass confinement.
fn resolve_with_symlinks(candidate: &Path, depth: usize) -> std::result::Result<PathBuf, String> {
    if depth > 16 {
        return Err(format!(
            "too many levels of symbolic links resolving `{}`",
            candidate.display()
        ));
    }
    let mut existing = candidate;
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    while existing.symlink_metadata().is_err() {
        match existing.file_name() {
            Some(name) => {
                tail.push(name.to_os_string());
                match existing.parent() {
                    Some(parent) => existing = parent,
                    None => break,
                }
            }
            None => break,
        }
    }

    let mut resolved = if let Ok(canon) = fs::canonicalize(existing) {
        canon
    } else if existing
        .symlink_metadata()
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
    {
        let link_target = fs::read_link(existing)
            .map_err(|e| format!("cannot read symlink `{}`: {e}", existing.display()))?;
        let next = if link_target.is_absolute() {
            normalize_lexical(&link_target)
        } else {
            let parent = existing.parent().unwrap_or_else(|| Path::new("/"));
            normalize_lexical(&parent.join(link_target))
        };
        resolve_with_symlinks(&next, depth + 1)?
    } else {
        existing.to_path_buf()
    };

    for seg in tail.into_iter().rev() {
        resolved.push(seg);
    }
    Ok(normalize_lexical(&resolved))
}

/// Resolves `raw` relative to `workspace_root` and enforces confinement within
/// `workspace_root` or `extra_roots` (resolving symlinks, including dangling symlinks,
/// on the longest existing ancestor).
pub fn resolve_confined_path(
    workspace_root: &Path,
    extra_roots: &[PathBuf],
    raw: &str,
) -> std::result::Result<PathBuf, String> {
    if raw.is_empty() || raw.contains('\0') {
        return Err("invalid empty or NUL path".into());
    }
    let ws_canon =
        fs::canonicalize(workspace_root).unwrap_or_else(|_| normalize_lexical(workspace_root));
    let mut allowed_roots = vec![ws_canon.clone()];
    for r in extra_roots {
        allowed_roots.push(fs::canonicalize(r).unwrap_or_else(|_| normalize_lexical(r)));
    }

    let candidate = {
        let p = Path::new(raw);
        if p.is_absolute() {
            normalize_lexical(p)
        } else {
            normalize_lexical(&ws_canon.join(p))
        }
    };

    let resolved = resolve_with_symlinks(&candidate, 0)?;

    if allowed_roots.iter().any(|root| resolved.starts_with(root)) {
        Ok(resolved)
    } else {
        Err(format!(
            "path `{raw}` resolves to `{}` outside allowed workspace root `{}`",
            resolved.display(),
            ws_canon.display()
        ))
    }
}

pub(super) fn effect_key_path(
    workspace_root: &Path,
    extra_roots: &[PathBuf],
    input: &Value,
) -> Option<PathBuf> {
    let raw = input.get("file_path").and_then(Value::as_str)?;
    let resolved = resolve_confined_path(workspace_root, extra_roots, raw).ok()?;
    // On case-insensitive filesystems (macOS APFS / Windows), fold ASCII case so `Foo.rs` and `foo.rs` conflict.
    if cfg!(any(target_os = "macos", target_os = "windows")) {
        Some(PathBuf::from(
            resolved.to_string_lossy().to_ascii_lowercase(),
        ))
    } else {
        Some(resolved)
    }
}

/// Symlink-safe, permission-preserving atomic write with temp cleanup on failure.
pub(super) fn atomic_write_confined(
    workspace_root: &Path,
    extra_roots: &[PathBuf],
    path: &Path,
    content: &str,
) -> std::result::Result<(), String> {
    // If `path` already exists and is a symlink, resolve its final target and re-check confinement!
    let target = if path.symlink_metadata().is_ok() {
        let canon = fs::canonicalize(path)
            .map_err(|e| format!("cannot resolve {}: {e}", path.display()))?;
        resolve_confined_path(workspace_root, extra_roots, &canon.to_string_lossy())?
    } else {
        path.to_path_buf()
    };

    let existing_perms = fs::metadata(&target).ok().map(|m| m.permissions());

    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("create_dir_all {}: {e}", parent.display()))?;
    }

    let file_name = target
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file");
    let tmp = target.with_file_name(format!(".{file_name}.carnot-tmp-{}", uuid::Uuid::new_v4()));

    let write_res = (|| -> std::io::Result<()> {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(content.as_bytes())?;
        f.sync_all()?;
        if let Some(perms) = existing_perms {
            fs::set_permissions(&tmp, perms)?;
        }
        fs::rename(&tmp, &target)?;
        Ok(())
    })();

    if let Err(e) = write_res {
        let _ = fs::remove_file(&tmp);
        return Err(format!("atomic write {}: {e}", target.display()));
    }
    Ok(())
}

/// Atomic byte write (`tmp + sync_all + rename`) preserving permissions (used by CAS undo).
pub(crate) fn atomic_write_raw_bytes(target: &Path, bytes: &[u8]) -> Result<()> {
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    let existing_perms = fs::metadata(target).ok().map(|m| m.permissions());
    let tmp = parent.join(format!(".carnot-undo-{}.tmp", uuid::Uuid::new_v4()));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    if let Some(p) = existing_perms {
        let _ = fs::set_permissions(&tmp, p);
    }
    if let Err(e) = fs::rename(&tmp, target) {
        let _ = fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}
