//! Canonical `Read` tool: numbered slices, multimodal images, tree-sitter symbol/outline, and diff-on-reread.

use super::path::{detect_image_mime, effect_key_path, resolve_confined_path};
use crate::syntax::{find_symbol_ranges, structural_outline};
use crate::tracker::{ReadDelta, ReadTracker};
use crate::{BoxFuture, Tool};
use base64::Engine as _;
use carnot_core::{CarnotError, Result, ToolEffects, ToolOutput, ToolResultBlock, ToolSpec};
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

const AUTO_OUTLINE_MIN_LINES: usize = 800;
const MAX_READ_BYTES: u64 = 16 * 1024 * 1024;
const MAX_IMAGE_BYTES: u64 = 5 * 1024 * 1024;
const MAX_READ_LINES: usize = 2_000;
const MAX_RENDERED_LINE_CHARS: usize = 2_000;

#[derive(Debug)]
pub struct ReadTool {
    workspace_root: PathBuf,
    extra_roots: Vec<PathBuf>,
    tracker: Arc<Mutex<ReadTracker>>,
}

impl ReadTool {
    pub fn new(
        workspace_root: PathBuf,
        extra_roots: Vec<PathBuf>,
        tracker: Arc<Mutex<ReadTracker>>,
    ) -> Self {
        Self {
            workspace_root,
            extra_roots,
            tracker,
        }
    }
}

impl Tool for ReadTool {
    fn spec(&self) -> ToolSpec {
        crate::canonical_tool_specs()
            .into_iter()
            .find(|s| s.name == "Read")
            .expect("Read spec")
    }

    fn effects(&self, input: &Value) -> ToolEffects {
        match effect_key_path(&self.workspace_root, &self.extra_roots, input) {
            Some(p) => ToolEffects::read_path(p),
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
        let tracker = Arc::clone(&self.tracker);

        Box::pin(async move {
            let Some(raw_path) = input
                .get("file_path")
                .and_then(Value::as_str)
                .map(str::to_string)
            else {
                return Ok(ToolOutput::err_text("Read requires `file_path`"));
            };
            let offset = input
                .get("offset")
                .and_then(Value::as_u64)
                .filter(|&v| v > 0)
                .map(|v| usize::try_from(v).unwrap_or(usize::MAX));
            let limit = input
                .get("limit")
                .and_then(Value::as_u64)
                .filter(|&v| v > 0)
                .map(|v| usize::try_from(v).unwrap_or(MAX_READ_LINES));
            let symbol = input
                .get("symbol")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string);
            let want_outline = input
                .get("outline")
                .and_then(Value::as_bool)
                .unwrap_or(false);

            tokio::task::spawn_blocking(move || -> Result<ToolOutput> {
                let path = match resolve_confined_path(&ws, &extra, &raw_path) {
                    Ok(p) => p,
                    Err(msg) => return Ok(ToolOutput::err_text(msg)),
                };
                let meta = match fs::metadata(&path) {
                    Ok(m) => m,
                    Err(e) => return Ok(ToolOutput::err_text(format!("Cannot read {}: {e}", path.display()))),
                };
                // On-demand structural directory outline (zero Turn-1 RepoMap bloat).
                if meta.is_dir() {
                    let outline = build_directory_outline(&ws, &extra, &path, &raw_path);
                    return Ok(ToolOutput::ok_text(outline).with_details(json!({
                        "file_path": raw_path,
                        "mode": "dir_outline",
                    })));
                }
                // Refuse non-regular files (FIFOs, named pipes, sockets, device nodes) so `fs::read`
                // can NEVER hang `spawn_blocking` indefinitely.
                if !meta.is_file() {
                    return Ok(ToolOutput::err_text(format!(
                        "Refusing to Read non-regular file {raw_path} (FIFO, socket, or special device)."
                    )));
                }
                if meta.len() > MAX_READ_BYTES {
                    return Ok(ToolOutput::err_text(format!(
                        "File {} is {} bytes (exceeds 16 MiB Read cap); use Bash (`head`/`rg`/`sed`) to inspect slices.",
                        raw_path,
                        meta.len()
                    )));
                }

                // Multimodal image check via magic bytes BEFORE UTF-8 decoding (PNG/JPEG/GIF/WebP)
                let raw_bytes = match fs::read(&path) {
                    Ok(b) => b,
                    Err(e) => {
                        return Ok(ToolOutput::err_text(format!(
                            "Cannot read {}: {e}",
                            path.display()
                        )));
                    }
                };
                if let Some(mime) = detect_image_mime(&raw_bytes) {
                    if raw_bytes.len() as u64 > MAX_IMAGE_BYTES {
                        return Ok(ToolOutput::err_text(format!(
                            "Image {} is {} bytes (exceeds 5 MiB provider image cap); resize before reading.",
                            raw_path,
                            raw_bytes.len()
                        )));
                    }
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&raw_bytes);
                    tracker
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .record_disk_hash(&path, &raw_bytes);
                    return Ok(ToolOutput {
                        blocks: vec![
                            ToolResultBlock::text(format!(
                                "[image: {raw_path} ({mime}, {} bytes)]",
                                raw_bytes.len()
                            )),
                            ToolResultBlock::Image {
                                media_type: mime.to_string(),
                                data_base64: b64,
                            },
                        ],
                        is_error: false,
                        details: json!({
                            "file_path": raw_path,
                            "mode": "image",
                            "media_type": mime,
                            "bytes": raw_bytes.len(),
                        }),
                    });
                }

                // Detect binary files (NUL bytes in first 8 KiB) before UTF-8 decode
                let probe_len = raw_bytes.len().min(8_192);
                if raw_bytes[..probe_len].contains(&0) {
                    return Ok(ToolOutput::err_text(format!(
                        "File {} appears to be binary ({} bytes); use Bash (`file`/`hexdump`/`strings`) to inspect.",
                        raw_path,
                        raw_bytes.len()
                    )));
                }

                let content = match String::from_utf8(raw_bytes) {
                    Ok(s) => s,
                    Err(e) => {
                        return Ok(ToolOutput::err_text(format!(
                            "Cannot read UTF-8 {}: {e}",
                            path.display()
                        )));
                    }
                };

                let all_lines: Vec<&str> = content.lines().collect();
                let total_lines = all_lines.len();

                // Explicit outline or auto-outline large source files on a bare Read (no offset/limit/symbol)
                let do_outline = want_outline
                    || (symbol.is_none()
                        && offset.is_none()
                        && limit.is_none()
                        && total_lines > AUTO_OUTLINE_MIN_LINES);
                if do_outline
                    && let Some(outline) = structural_outline(&path, &content)
                {
                    tracker
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .record_disk_hash(&path, content.as_bytes());
                    return Ok(ToolOutput::ok_text(outline).with_details(json!({
                        "file_path": raw_path,
                        "mode": "outline",
                        "total_lines": total_lines,
                    })));
                }

                // Direct tree-sitter symbol lookup (`Read(file_path, symbol="Foo::bar")`)
                if let Some(ref sym) = symbol {
                    let spans = find_symbol_ranges(&path, &content, sym);
                    if spans.is_empty() {
                        return Ok(ToolOutput::err_text(format!(
                            "Symbol `{sym}` not found in {raw_path}; pass `outline: true` to list symbols."
                        )));
                    }
                    let mut rendered = String::new();
                    let mut viewed_spans: Vec<(usize, &[&str])> = Vec::with_capacity(4);
                    for (idx, (s_line, e_line)) in spans.iter().take(4).enumerate() {
                        if idx > 0 {
                            rendered.push_str("\n---\n");
                        }
                        let s = (*s_line).max(1).min(total_lines);
                        let e = (*e_line).max(s).min(total_lines);
                        let span_lines = &all_lines[s - 1..e];
                        viewed_spans.push((s, span_lines));
                        for (i, line) in span_lines.iter().enumerate() {
                            let _ = writeln!(rendered, "{}\t{line}", s + i);
                        }
                    }
                    tracker
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .record_viewed_spans(&path, content.as_bytes(), &viewed_spans);
                    return Ok(ToolOutput::ok_text(rendered.trim_end_matches('\n')).with_details(
                        json!({
                            "file_path": raw_path,
                            "mode": "symbol",
                            "symbol": sym,
                            "matches": spans.len(),
                        }),
                    ));
                }

                if total_lines == 0 {
                    tracker
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .record_disk_hash(&path, content.as_bytes());
                    return Ok(ToolOutput::ok_text(format!("[empty file: {raw_path}]"))
                        .with_details(json!({"file_path": raw_path, "mode": "full", "lines": [0, 0]})));
                }

                let start = offset.unwrap_or(1).max(1);
                if start > total_lines {
                    return Ok(ToolOutput::err_text(format!(
                        "offset {start} exceeds file length ({total_lines} lines) in {raw_path}"
                    )));
                }
                let max_lines = limit.unwrap_or(MAX_READ_LINES).clamp(1, MAX_READ_LINES);
                let end = start
                    .saturating_add(max_lines.saturating_sub(1))
                    .min(total_lines);

                let slice_lines = &all_lines[start.saturating_sub(1)..end];
                let raw_slice = slice_lines.join("\n");
                let is_full_file = start == 1 && end == total_lines;

                // Check disk-coherent ReadTracker on raw unnumbered lines.
                let delta = tracker
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .check_reread(&path, &raw_slice, (start, end), is_full_file);
                match delta {
                    ReadDelta::UnchangedStub(stub) => {
                        return Ok(ToolOutput::ok_text(stub)
                            .with_details(json!({"file_path": raw_path, "mode": "stub"})));
                    }
                    ReadDelta::UnifiedDiff(diff_txt) => {
                        tracker
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .record_slice(
                                &path,
                                content.as_bytes(),
                                &raw_slice,
                                (start, end),
                                is_full_file,
                            );
                        return Ok(ToolOutput::ok_text(diff_txt)
                            .with_details(json!({"file_path": raw_path, "mode": "diff"})));
                    }
                    ReadDelta::FullNeeded => {
                        tracker
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .record_slice(
                                &path,
                                content.as_bytes(),
                                &raw_slice,
                                (start, end),
                                is_full_file,
                            );
                    }
                }

                // Format compact `N\ttext` numbered lines (zero padding bloat!) with per-line cap so
                // 500 KB minified JS/JSON lines cannot blow the context window in one Read call.
                let mut rendered =
                    String::with_capacity(raw_slice.len().saturating_add(slice_lines.len() * 8));
                {
                    for (i, l) in slice_lines.iter().enumerate() {
                        if i > 0 {
                            rendered.push('\n');
                        }
                        if l.len() > MAX_RENDERED_LINE_CHARS {
                            let total_chars = l.chars().count();
                            if total_chars > MAX_RENDERED_LINE_CHARS {
                                let head_end = l
                                    .char_indices()
                                    .nth(MAX_RENDERED_LINE_CHARS)
                                    .map_or(l.len(), |(idx, _)| idx);
                                let _ = write!(
                                    rendered,
                                    "{}\t{} [... line truncated; {total_chars} total chars]",
                                    start + i,
                                    &l[..head_end]
                                );
                                continue;
                            }
                        }
                        let _ = write!(rendered, "{}\t{l}", start + i);
                    }
                }

                let truncated = end < total_lines;
                if truncated {
                    let _ = write!(
                        rendered,
                        "\n[truncated: showing lines {start}-{end} of {total_lines}; pass offset={} to read further]",
                        end + 1
                    );
                }

                Ok(ToolOutput::ok_text(rendered).with_details(json!({
                    "file_path": raw_path,
                    "mode": "full",
                    "lines": [start, end],
                    "total_lines": total_lines,
                    "truncated": truncated,
                })))
            })
            .await
            .map_err(|e| CarnotError::Tool {
                tool: "Read".into(),
                message: format!("blocking read task failed: {e}"),
            })?
        })
    }
}

/// On-demand structural directory outline using bundled `tree-sitter` (zero Turn-1 cost!).
/// Confines every entry via `symlink_metadata` + `resolve_confined_path`.
fn build_directory_outline(ws: &Path, extra: &[PathBuf], dir: &Path, raw_path: &str) -> String {
    let mut entries: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name();
            let ns = name.to_string_lossy();
            if ns.starts_with('.')
                || matches!(
                    ns.as_ref(),
                    "target" | "node_modules" | "dist" | "__pycache__"
                )
            {
                continue;
            }
            // Reject symlinks escaping workspace confinement!
            if fs::symlink_metadata(&p).is_ok_and(|m| m.file_type().is_symlink())
                && resolve_confined_path(ws, extra, &p.to_string_lossy()).is_err()
            {
                continue;
            }
            entries.push(p);
        }
    }
    entries.sort_unstable();
    let mut out = format!(
        "[directory outline: {raw_path} ({} entries)]\n",
        entries.len()
    );
    for p in entries.into_iter().take(32) {
        let fname = p
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if p.is_dir() {
            let _ = writeln!(out, "- {fname}/");
            continue;
        }
        let size = fs::metadata(&p).map_or(0, |m| m.len());
        if size <= 256 * 1024
            && let Ok(src) = fs::read_to_string(&p)
            && let Some(sym_outline) = structural_outline(&p, &src)
        {
            let compact: Vec<&str> = sym_outline.lines().skip(1).take(8).collect();
            if compact.is_empty() {
                let _ = writeln!(out, "- {fname} ({size} B)");
            } else {
                let _ = writeln!(out, "- {fname} ({size} B): {}", compact.join(" ; "));
            }
        } else {
            let _ = writeln!(out, "- {fname} ({size} B)");
        }
        if out.len() > 2_500 {
            out.push_str("[... directory outline truncated ...]\n");
            break;
        }
    }
    out.trim_end().to_string()
}
