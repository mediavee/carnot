//! Effect-based parallel tool scheduler: eager mid-stream read launches and conflict-free waves.

use carnot_core::{ToolEffects, ToolOutput};
use tokio::task::JoinHandle;

pub(crate) struct EagerLaunch {
    pub(crate) effects: ToolEffects,
    pub(crate) effective_input: serde_json::Value,
    pub(crate) denied: Option<String>,
    pub(crate) handle: Option<JoinHandle<(ToolOutput, u64)>>,
}

pub(crate) struct ActiveEager {
    pub(crate) orig_idx: usize,
    pub(crate) effects: ToolEffects,
    pub(crate) effective_input: serde_json::Value,
    pub(crate) handle: JoinHandle<(ToolOutput, u64)>,
}

pub(crate) struct ScheduledCall {
    pub(crate) orig_idx: usize,
    pub(crate) call_id: String,
    pub(crate) tool_name: String,
    pub(crate) effective_input: serde_json::Value,
    pub(crate) effects: ToolEffects,
}

/// Partitions tool calls into conflict-free parallel execution waves while preserving program order.
pub(crate) fn partition_conflict_free_waves(calls: Vec<ScheduledCall>) -> Vec<Vec<ScheduledCall>> {
    let mut waves: Vec<Vec<ScheduledCall>> = Vec::new();
    for call in calls {
        let can_join_last = waves
            .last()
            .map(|w| {
                w.iter()
                    .all(|prev| !prev.effects.conflicts_with(&call.effects))
            })
            .unwrap_or(false);
        if can_join_last && let Some(last) = waves.last_mut() {
            last.push(call);
        } else {
            waves.push(vec![call]);
        }
    }
    waves
}
