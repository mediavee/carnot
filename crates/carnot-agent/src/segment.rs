//! Per-segment working-memory tracker for state-preserving compaction and live telemetry.

use carnot_core::{SessionMetrics, ToolOutput};
use serde_json::Value;

#[derive(Debug, Default)]
pub(crate) struct SegmentTracker {
    pub(crate) had_edit: bool,
    pub(crate) files_read: Vec<String>,
    pub(crate) files_edited: Vec<String>,
    pub(crate) handles_created: Vec<String>,
    pub(crate) failed_attempts: Vec<String>,
}

impl SegmentTracker {
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn push_unique(vec: &mut Vec<String>, item: String) {
        if !item.is_empty() && !vec.contains(&item) {
            vec.push(item);
        }
    }

    /// Records tool outcomes into both the segment state and live runtime metrics.
    pub(crate) fn record_tool_outcome(
        &mut self,
        tool_name: &str,
        input: &Value,
        out: &ToolOutput,
        metrics: &mut SessionMetrics,
    ) {
        if let Some(handle) = out.details.get("handle").and_then(|v| v.as_str()) {
            Self::push_unique(&mut self.handles_created, handle.to_string());
            metrics.cas_handles_created += 1;
        }
        if let Some(saved) = out.details.get("bytes_saved").and_then(|v| v.as_u64())
            && saved > 0
        {
            metrics.semantic_bytes_saved += saved;
        }
        if tool_name == "Edit" {
            metrics.edit_attempts += 1;
            if out.is_error {
                metrics.edit_failures += 1;
            } else if let Some(fc) = out.details.get("fuzzy_matches").and_then(|v| v.as_u64()) {
                metrics.edit_fuzzy_matches += fc;
            }
        }

        match tool_name {
            "Read" => {
                if let Some(p) = input.get("file_path").and_then(|v| v.as_str()) {
                    Self::push_unique(&mut self.files_read, p.to_string());
                }
            }
            "Edit" | "Write" if !out.is_error => {
                self.had_edit = true;
                if let Some(p) = input.get("file_path").and_then(|v| v.as_str()) {
                    Self::push_unique(&mut self.files_edited, p.to_string());
                }
            }
            "Bash" if out.is_error => {
                if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
                    let first_line = out
                        .text_only()
                        .lines()
                        .next()
                        .unwrap_or("failed")
                        .chars()
                        .take(120)
                        .collect::<String>();
                    Self::push_unique(
                        &mut self.failed_attempts,
                        format!("`{cmd}` -> {first_line}"),
                    );
                }
            }
            _ => {}
        }
    }
}
