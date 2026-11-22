//! Deterministic Maxwell's Daemon (S1/S2/S3) — Advisory-First with Progress Decay
//! plus 3-Idle-Turn Circuit Breaker.

use crate::{BoundaryAction, BoxFuture, Module, TurnBoundarySnapshot};
use carnot_cas::WorktreeHasher;
use carnot_core::{Result, ToolOutput};
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

fn path_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"/[^\s:]+").expect("regex"))
}
fn hex_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"0x[0-9a-fA-F]+").expect("regex"))
}
fn num_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b\d+\b").expect("regex"))
}

pub fn normalize_error_signature(tool: &str, text: &str) -> String {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let key: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|l| {
            let low = l.to_ascii_lowercase();
            low.contains("error")
                || low.contains("failed")
                || low.contains("panic")
                || low.contains("exception")
                || low.contains("assert")
        })
        .take(3)
        .collect();
    let sample = if key.is_empty() {
        lines
            .iter()
            .rev()
            .take(2)
            .copied()
            .collect::<Vec<_>>()
            .join(" | ")
    } else {
        key.join(" | ")
    };
    let s1 = path_re().replace_all(&sample, "<P>");
    let s2 = hex_re().replace_all(&s1, "<X>");
    let s3 = num_re().replace_all(&s2, "<N>");
    let truncated: String = s3.chars().take(160).collect();
    format!("{tool}:{truncated}")
}

#[derive(Debug)]
struct DaemonState {
    hasher: WorktreeHasher,
    tree_history: Vec<[u8; 32]>,
    seen_trees: HashMap<[u8; 32], usize>,
    error_counts: HashMap<String, u32>,
    idle_turn_streak: u32,
    /// S4: counts consecutive passing shell-only turns after a green test without intervening edits.
    seen_verified_green: bool,
    consecutive_passing_bash_turns: u32,
    saw_bash_ok_this_turn: bool,
    saw_edit_this_turn: bool,
    pending_advisory: Option<(String, String)>,
    recent_failures: Vec<String>,
}

/// Deterministic stagnation detector operating at ~0.78 ms per turn (zero LLM calls).
#[derive(Debug)]
pub struct MaxwellDaemonModule {
    workspace_root: PathBuf,
    idle_limit: u32,
    state: Arc<Mutex<DaemonState>>,
}

impl MaxwellDaemonModule {
    pub fn new(workspace_root: PathBuf, idle_limit: u32) -> Self {
        Self {
            workspace_root,
            idle_limit,
            state: Arc::new(Mutex::new(DaemonState {
                hasher: WorktreeHasher::new(),
                tree_history: Vec::new(),
                seen_trees: HashMap::new(),
                error_counts: HashMap::new(),
                idle_turn_streak: 0,
                seen_verified_green: false,
                consecutive_passing_bash_turns: 0,
                saw_bash_ok_this_turn: false,
                saw_edit_this_turn: false,
                pending_advisory: None,
                recent_failures: Vec::new(),
            })),
        }
    }

    pub fn recent_failures(&self) -> Vec<String> {
        self.state
            .lock()
            .map(|s| s.recent_failures.clone())
            .unwrap_or_default()
    }
}

impl Module for MaxwellDaemonModule {
    fn name(&self) -> &str {
        "maxwell_daemon"
    }

    fn after_tool_result<'a>(
        &'a self,
        tool_name: &'a str,
        _input: &'a Value,
        output: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>> {
        let text = output.text_only();
        let is_err = output.is_error;
        let tname = tool_name.to_string();
        Box::pin(async move {
            let Ok(mut st) = self.state.lock() else {
                return Ok(());
            };
            if matches!(tname.as_str(), "Edit" | "Write") && !is_err {
                st.saw_edit_this_turn = true;
            } else if tname == "Bash" && !is_err {
                st.saw_bash_ok_this_turn = true;
            }
            if is_err {
                let sig = normalize_error_signature(&tname, &text);
                if sig.len() > 10 {
                    let count = {
                        let c = st.error_counts.entry(sig.clone()).or_insert(0);
                        *c += 1;
                        *c
                    };
                    if !st.recent_failures.contains(&sig) {
                        st.recent_failures.push(sig.clone());
                        if st.recent_failures.len() > 8 {
                            st.recent_failures.remove(0);
                        }
                    }
                    if count == 3 || count == 6 {
                        st.pending_advisory = Some((
                            "S3:error_cluster".into(),
                            format!(
                                "[maxwell-daemon] Same normalized error recurred {count} times ({sig}). Pause and reconsider root cause before retrying."
                            ),
                        ));
                    }
                }
            }
            Ok(())
        })
    }

    fn on_turn_boundary<'a>(
        &'a self,
        snap: &'a TurnBoundarySnapshot,
    ) -> BoxFuture<'a, Result<BoundaryAction>> {
        Box::pin(async move {
            let (edited, halt_reason, mut pending) = {
                let Ok(mut st) = self.state.lock() else {
                    return Ok(BoundaryAction::Continue);
                };

                // Progress decay: a verified green test resets transient error/plateau counters!
                if snap.verified_green_command.is_some() {
                    st.error_counts.clear();
                    st.recent_failures.clear();
                    st.pending_advisory = None;
                    st.idle_turn_streak = 0;
                    st.seen_verified_green = true;
                }

                // S4: detect genuinely runaway post-green shell-only loops (capped at 2 advisories)
                let edited = std::mem::take(&mut st.saw_edit_this_turn);
                let bash_ok = std::mem::take(&mut st.saw_bash_ok_this_turn);
                if edited {
                    st.seen_verified_green = false;
                    st.consecutive_passing_bash_turns = 0;
                } else if st.seen_verified_green && bash_ok {
                    st.consecutive_passing_bash_turns += 1;
                    if matches!(st.consecutive_passing_bash_turns, 8 | 16)
                        && st.pending_advisory.is_none()
                    {
                        st.pending_advisory = Some((
                            "S4:over_verification".into(),
                            format!(
                                "[maxwell-daemon] {} consecutive passing shell-only turns since the last edit. If you are still validating edge cases, consider batching remaining checks into a single command.",
                                st.consecutive_passing_bash_turns
                            ),
                        ));
                    }
                } else {
                    st.consecutive_passing_bash_turns = 0;
                }

                // 3-idle-turn circuit breaker (prevents runaway autonomous/goal loops)
                let halt_reason = if !snap.had_tool_call_in_turn {
                    st.idle_turn_streak += 1;
                    if self.idle_limit > 0 && st.idle_turn_streak >= self.idle_limit {
                        Some(format!(
                            "idle circuit breaker triggered ({} consecutive turns without tool calls)",
                            st.idle_turn_streak
                        ))
                    } else {
                        None
                    }
                } else {
                    st.idle_turn_streak = 0;
                    None
                };

                (edited, halt_reason, st.pending_advisory.take())
            };

            if let Some(reason) = halt_reason {
                return Ok(BoundaryAction::Halt { reason });
            }

            // S1: Incremental path-inclusive BLAKE3 worktree tree-hash cycle detection
            // Only runs on turns that ACTUALLY edited files, offloaded to spawn_blocking!
            if edited {
                let state_arc = Arc::clone(&self.state);
                let ws = self.workspace_root.clone();
                let cycle_msg = tokio::task::spawn_blocking(move || -> Option<String> {
                    let mut st = state_arc.lock().ok()?;
                    let wt = st.hasher.snapshot(&ws);
                    let idx = st.tree_history.len();
                    let cycle = st
                        .seen_trees
                        .get(&wt.tree_hash)
                        .copied()
                        .filter(|&prev| idx > prev + 1);
                    st.seen_trees.insert(wt.tree_hash, idx);
                    st.tree_history.push(wt.tree_hash);
                    cycle.map(|prev| {
                        format!(
                            "[maxwell-daemon] Worktree returned to an identical file-tree hash seen {} edits ago (A->B->A cycle). Verify whether you are oscillating between two broken states.",
                            idx - prev
                        )
                    })
                })
                .await
                .ok()
                .flatten();

                if let Some(message) = cycle_msg {
                    return Ok(BoundaryAction::AppendAdvisory {
                        signal: "S1:tree_cycle".into(),
                        message,
                    });
                }
            }

            if let Some((sig, msg)) = pending.take() {
                return Ok(BoundaryAction::AppendAdvisory {
                    signal: sig,
                    message: msg,
                });
            }

            Ok(BoundaryAction::Continue)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn snap(turn: u64, green: Option<&str>) -> TurnBoundarySnapshot {
        TurnBoundarySnapshot {
            turn_index: turn,
            tokens_since_checkpoint: 1_000,
            total_context_tokens: 1_000,
            had_edit_in_segment: false,
            had_tool_call_in_turn: true,
            verified_green_command: green.map(str::to_string),
            verified_green_summary: None,
            files_read_in_segment: Vec::new(),
            files_edited_in_segment: Vec::new(),
            handles_created_in_segment: Vec::new(),
            last_assistant_text: String::new(),
            failed_attempts_summary: Vec::new(),
        }
    }

    #[tokio::test]
    async fn s4_resets_on_edit_and_allows_normal_verification_checks() {
        let dir = std::env::temp_dir();
        let daemon = MaxwellDaemonModule::new(dir, 3);

        // Turn 1: verified green test
        let mut out = ToolOutput::ok_text("ok");
        daemon
            .after_tool_result("Bash", &json!({"command": "pytest"}), &mut out)
            .await
            .expect("after_tool_result");
        assert!(matches!(
            daemon
                .on_turn_boundary(&snap(1, Some("pytest")))
                .await
                .expect("boundary"),
            BoundaryAction::Continue
        ));

        // Turns 2..=6 (5 passing shell checks) must NOT nag the agent
        for t in 2..=6 {
            let mut o = ToolOutput::ok_text("ok");
            daemon
                .after_tool_result("Bash", &json!({"command": "curl localhost"}), &mut o)
                .await
                .expect("after_tool_result");
            assert!(matches!(
                daemon
                    .on_turn_boundary(&snap(t, None))
                    .await
                    .expect("boundary"),
                BoundaryAction::Continue
            ));
        }

        // Turn 7: Edit resets `seen_verified_green` so subsequent shell checks do NOT fire S4!
        let mut edit_out = ToolOutput::ok_text("edited");
        daemon
            .after_tool_result("Edit", &json!({"file_path": "a.py"}), &mut edit_out)
            .await
            .expect("after_tool_result");
        assert!(matches!(
            daemon
                .on_turn_boundary(&snap(7, None))
                .await
                .expect("boundary"),
            BoundaryAction::Continue
        ));

        for t in 8..=18 {
            let mut o = ToolOutput::ok_text("ok");
            daemon
                .after_tool_result("Bash", &json!({"command": "ls"}), &mut o)
                .await
                .expect("after_tool_result");
            assert!(matches!(
                daemon
                    .on_turn_boundary(&snap(t, None))
                    .await
                    .expect("boundary"),
                BoundaryAction::Continue
            ));
        }
    }
}
