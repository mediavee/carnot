//! Canonical `Write` tool: atomic confined file creation/replacement with CAS Undo and syntax guard.

use super::path::{
    atomic_write_confined, effect_key_path, reject_protected_git_write, resolve_confined_path,
    validate_regular_file_for_read,
};
use crate::syntax::syntax_regression_warning;
use crate::tracker::{PathLockPool, ReadTracker};
use crate::{BoxFuture, Tool, UndoEntry, UndoKind};
use carnot_cas::CasStore;
use carnot_core::{CarnotError, Result, ToolEffects, ToolOutput, ToolSpec};
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
pub struct WriteTool {
    workspace_root: PathBuf,
    extra_roots: Vec<PathBuf>,
    cas: Arc<CasStore>,
    undo_stack: Arc<Mutex<Vec<UndoEntry>>>,
    tracker: Arc<Mutex<ReadTracker>>,
    path_locks: Arc<PathLockPool>,
    syntax_check: bool,
}

impl WriteTool {
    pub fn new(
        workspace_root: PathBuf,
        extra_roots: Vec<PathBuf>,
        cas: Arc<CasStore>,
        undo_stack: Arc<Mutex<Vec<UndoEntry>>>,
        tracker: Arc<Mutex<ReadTracker>>,
        path_locks: Arc<PathLockPool>,
        syntax_check: bool,
    ) -> Self {
        Self {
            workspace_root,
            extra_roots,
            cas,
            undo_stack,
            tracker,
            path_locks,
            syntax_check,
        }
    }
}

impl Tool for WriteTool {
    fn spec(&self) -> ToolSpec {
        crate::canonical_tool_specs()
            .into_iter()
            .find(|s| s.name == "Write")
            .expect("Write spec")
    }

    fn effects(&self, input: &Value) -> ToolEffects {
        match effect_key_path(&self.workspace_root, &self.extra_roots, input) {
            Some(p) => ToolEffects::write_path(p),
            None => ToolEffects::serial(),
        }
    }

    fn execute(
        &self,
        input: Value,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        let ws = self.workspace_root.clone();
        let extra = self.extra_roots.clone();
        let syntax_check = self.syntax_check;
        let cas = Arc::clone(&self.cas);
        let undo_stack = Arc::clone(&self.undo_stack);
        let tracker = Arc::clone(&self.tracker);
        let path_locks = Arc::clone(&self.path_locks);

        Box::pin(async move {
            let Some(raw_path) = input
                .get("file_path")
                .and_then(Value::as_str)
                .map(str::to_string)
            else {
                return Ok(ToolOutput::err_text("Write requires `file_path`"));
            };
            let Some(content) = input
                .get("content")
                .and_then(Value::as_str)
                .map(str::to_string)
            else {
                return Ok(ToolOutput::err_text("Write requires `content`"));
            };

            let path_lock = effect_key_path(&ws, &extra, &input).map(|p| path_locks.lock_for(&p));
            let _path_guard = match path_lock {
                Some(ref m) => Some(m.lock().await),
                None => None,
            };

            tokio::task::spawn_blocking(move || -> Result<ToolOutput> {
                let path = match resolve_confined_path(&ws, &extra, &raw_path) {
                    Ok(p) => p,
                    Err(msg) => return Ok(ToolOutput::err_text(msg)),
                };
                if let Err(msg) = reject_protected_git_write(&ws, &path) {
                    return Ok(ToolOutput::err_text(msg));
                }
                let existed = path.exists();
                let before_bytes = if existed {
                    if let Err(msg) = validate_regular_file_for_read(&path) {
                        return Ok(ToolOutput::err_text(format!("Write aborted: {msg}")));
                    }
                    let bytes = match fs::read(&path) {
                        Ok(b) => b,
                        Err(e) => {
                            return Ok(ToolOutput::err_text(format!(
                                "Write aborted: cannot snapshot existing {raw_path}: {e}"
                            )));
                        }
                    };
                    // Refuse blind overwrites of existing unread or externally modified files.
                    let disk_digest = *blake3::hash(&bytes).as_bytes();
                    let tracked_digest = tracker
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .disk_hash(&path);
                    match tracked_digest {
                        None => {
                            return Ok(ToolOutput::err_text(format!(
                                "Write refused on existing file {raw_path}: file has not been read in this session (or tracker was cleared after compaction). Read {raw_path} first before overwriting it."
                            ))
                            .with_details(json!({
                                "file_path": raw_path,
                                "write_blocked": "unread",
                            })));
                        }
                        Some(seen) if seen != disk_digest => {
                            return Ok(ToolOutput::err_text(format!(
                                "Write refused on {raw_path}: file changed on disk since your last Read/Edit. Re-read {raw_path} before overwriting it so external changes are not clobbered."
                            ))
                            .with_details(json!({
                                "file_path": raw_path,
                                "write_blocked": "stale",
                            })));
                        }
                        Some(_) => {}
                    }
                    Some(bytes)
                } else {
                    None
                };
                let before_text = before_bytes
                    .as_deref()
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .map(str::to_string);
                let undo_hash = if let Some(ref b) = before_bytes {
                    match cas.put_bytes(b) {
                        Ok(h) => Some(h.hash),
                        Err(e) => {
                            return Ok(ToolOutput::err_text(format!(
                                "Write aborted: failed to snapshot {raw_path} into CAS: {e}"
                            )));
                        }
                    }
                } else {
                    None
                };
                if let Err(msg) = atomic_write_confined(&ws, &extra, &path, &content) {
                    return Ok(ToolOutput::err_text(msg));
                }
                tracker
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .record_write(&path, content.as_bytes());
                let post_hash = blake3::hash(content.as_bytes()).to_hex().to_string();
                let kind = match undo_hash.clone() {
                    Some(prev_hash) => UndoKind::Restored {
                        prev_hash,
                        post_hash,
                    },
                    None => UndoKind::Created { post_hash },
                };
                if let Ok(mut st) = undo_stack.lock() {
                    st.push(UndoEntry {
                        path: path.clone(),
                        kind,
                    });
                }

                let mut reply = format!("Wrote {} ({} bytes).", raw_path, content.len());
                if syntax_check
                    && let Some(warn) =
                        syntax_regression_warning(&path, before_text.as_deref(), &content)
                {
                    reply.push('\n');
                    reply.push_str(&warn);
                }
                Ok(ToolOutput::ok_text(reply).with_details(json!({
                    "file_path": raw_path,
                    "bytes": content.len(),
                    "undo_cas_hash": undo_hash,
                })))
            })
            .await
            .map_err(|e| CarnotError::Tool {
                tool: "Write".into(),
                message: format!("blocking write task failed: {e}"),
            })?
        })
    }
}
