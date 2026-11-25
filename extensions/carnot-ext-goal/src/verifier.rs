//! Deterministic sandboxed command verification, host-safe git evidence collection,
//! and independent virgin-context LLM verifier for `carnot-ext-goal`.

use carnot_ai::{Provider, ProviderRequest, StreamEvent};
use carnot_cas::{CasStore, shape_command_output_with_redaction};
use carnot_core::{CacheTtl, CarnotError, Message, NativeWebConfig, ReasoningEffort, Result};
use carnot_sandbox::{SandboxPolicy, build_sandboxed_shell, host_safe_git_command};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;

const MAX_CAPTURED_CMD_BYTES: usize = 256 * 1024;

const VERIFIER_SYSTEM_PROMPT: &str = "\
You are an independent, strict software verification judge evaluating whether a coding agent has \
satisfied a user's concrete goal criterion.\n\
Rules:\n\
- Judge ONLY from the provided evidence (deterministic command outputs, git status/diff, edited files, and worker summary). Never trust unverified claims.\n\
- If all requirements of the criterion are demonstrably satisfied, return status \"met\".\n\
- If any requirement is missing, broken, untested when testing was required, or incomplete, return status \"not_met\" with concise, actionable feedback naming the exact gap.\n\
- Return status \"unreachable\" ONLY when the goal is provably impossible in the current environment (e.g. missing external credentials, impossible requirement, unavailable tool/dependency that cannot be installed).\n\
- Respond with ONLY a single JSON object matching: {\"status\": \"met\" | \"not_met\" | \"unreachable\", \"feedback\": \"<concise explanation>\"}";

#[derive(Debug, Clone)]
pub(crate) struct CommandCheckResult {
    pub(crate) command: String,
    pub(crate) passed: bool,
    pub(crate) exit_code: i32,
    pub(crate) timed_out: bool,
    pub(crate) shaped_output: String,
    pub(crate) output_hash: [u8; 32],
}

#[derive(Debug, Clone)]
pub(crate) struct WorkspaceEvidence {
    pub(crate) status_short: String,
    pub(crate) diff_excerpt: String,
    pub(crate) state_hash: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifierDecision {
    Met { feedback: String },
    NotMet { feedback: String },
    Unreachable { feedback: String },
}

/// Runs one deterministic verification command inside Carnot's OS sandbox with bounded output and
/// timeout/cancellation enforcement.
#[expect(
    clippy::too_many_arguments,
    reason = "explicit sandbox and CAS policy parameters"
)]
pub(crate) async fn run_verify_command(
    command: &str,
    sandbox: &SandboxPolicy,
    extra_env: &BTreeMap<String, String>,
    cas: &CasStore,
    cas_handles: bool,
    spill_threshold: usize,
    preview_max: usize,
    redact_secrets: bool,
    timeout: Duration,
    cancel: CancellationToken,
) -> Result<CommandCheckResult> {
    let mut cmd =
        build_sandboxed_shell(sandbox, command, extra_env).map_err(|msg| CarnotError::Tool {
            tool: "Goal".into(),
            message: format!("sandbox refused verification command `{command}`: {msg}"),
        })?;
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = cmd.spawn().map_err(|e| CarnotError::Tool {
        tool: "Goal".into(),
        message: format!("failed to spawn verification command `{command}`: {e}"),
    })?;

    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();

    let out_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(ref mut s) = stdout {
            let mut limited = s.take(MAX_CAPTURED_CMD_BYTES as u64);
            let _ = limited.read_to_end(&mut buf).await;
        }
        buf
    });
    let err_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        if let Some(ref mut s) = stderr {
            let mut limited = s.take(MAX_CAPTURED_CMD_BYTES as u64);
            let _ = limited.read_to_end(&mut buf).await;
        }
        buf
    });

    let mut timed_out = false;
    let status_opt = tokio::select! {
        biased;
        () = cancel.cancelled() => {
            let _ = child.kill().await;
            return Err(CarnotError::Cancelled);
        }
        () = tokio::time::sleep(timeout) => {
            timed_out = true;
            let _ = child.kill().await;
            None
        }
        res = child.wait() => {
            res.ok()
        }
    };

    let out_bytes = out_task.await.unwrap_or_default();
    let err_bytes = err_task.await.unwrap_or_default();

    let mut combined = String::new();
    if !out_bytes.is_empty() {
        combined.push_str(&String::from_utf8_lossy(&out_bytes));
    }
    if !err_bytes.is_empty() {
        if !combined.is_empty() && !combined.ends_with('\n') {
            combined.push('\n');
        }
        combined.push_str(&String::from_utf8_lossy(&err_bytes));
    }

    let exit_code = if timed_out {
        124
    } else {
        status_opt.and_then(|s| s.code()).unwrap_or(-1)
    };
    let passed = !timed_out && exit_code == 0;
    let output_hash = *blake3::hash(combined.as_bytes()).as_bytes();

    let shaped_output = if combined.trim().is_empty() {
        if timed_out {
            format!("[timed out after {}s with no output]", timeout.as_secs())
        } else {
            "[no output]".to_string()
        }
    } else {
        shape_command_output_with_redaction(
            Some(command),
            &combined,
            cas_handles.then_some(cas),
            spill_threshold,
            preview_max,
            redact_secrets,
        )?
        .text
    };

    Ok(CommandCheckResult {
        command: command.to_string(),
        passed,
        exit_code,
        timed_out,
        shaped_output,
        output_hash,
    })
}

/// Collects host-safe `git status` and bounded `git diff` evidence for the verifier and computes a
/// deterministic workspace state digest for stagnation detection.
pub(crate) async fn collect_workspace_evidence(
    workspace_root: &Path,
    max_diff_bytes: usize,
) -> WorkspaceEvidence {
    let root = workspace_root.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let mut hasher = blake3::Hasher::new();

        let status_out = host_safe_git_command(&root)
            .args(["status", "--short", "--untracked-files=normal"])
            .output();
        let status_short = match status_out {
            Ok(o) if o.status.success() => {
                hasher.update(&o.stdout);
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                clip_utf8(&s, 2_048)
            }
            _ => String::new(),
        };

        let diff_out = host_safe_git_command(&root)
            .args(["diff", "HEAD"])
            .output()
            .or_else(|_| host_safe_git_command(&root).args(["diff"]).output());
        let diff_excerpt = match diff_out {
            Ok(o) if o.status.success() => {
                hasher.update(&o.stdout);
                let s = String::from_utf8_lossy(&o.stdout);
                clip_utf8(&s, max_diff_bytes)
            }
            _ => String::new(),
        };

        WorkspaceEvidence {
            status_short,
            diff_excerpt,
            state_hash: *hasher.finalize().as_bytes(),
        }
    })
    .await
    .unwrap_or_else(|_| WorkspaceEvidence {
        status_short: String::new(),
        diff_excerpt: String::new(),
        state_hash: [0u8; 32],
    })
}

/// Runs an independent, single-turn, virgin-context LLM evaluation (Worker ≠ Verifier).
/// Returns `(VerifierDecision, cost_usd)`.
#[expect(
    clippy::too_many_arguments,
    reason = "explicit verifier request inputs"
)]
pub(crate) async fn run_llm_verifier(
    provider: &Arc<dyn Provider>,
    model: &str,
    effort: ReasoningEffort,
    criterion: &str,
    check_results: &[CommandCheckResult],
    evidence: &WorkspaceEvidence,
    files_edited: &[String],
    last_assistant_text: &str,
    cancel: CancellationToken,
) -> Result<(VerifierDecision, f64)> {
    let mut prompt = String::new();
    let _ = writeln!(prompt, "## Goal criterion\n{criterion}\n");

    if !check_results.is_empty() {
        prompt.push_str("## Deterministic verification commands\n");
        for r in check_results {
            let tag = if r.passed { "PASS" } else { "FAIL" };
            let _ = writeln!(prompt, "- `{}` -> {tag} (exit {})", r.command, r.exit_code);
            if !r.shaped_output.is_empty() && r.shaped_output != "[no output]" {
                let _ = writeln!(prompt, "  Output:\n{}", clip_utf8(&r.shaped_output, 2_048));
            }
        }
        prompt.push('\n');
    }

    if !files_edited.is_empty() {
        let _ = writeln!(
            prompt,
            "## Files edited in segment\n{}\n",
            files_edited.join(", ")
        );
    }

    if !evidence.status_short.is_empty() {
        let _ = writeln!(
            prompt,
            "## Git status\n```text\n{}\n```\n",
            evidence.status_short
        );
    }

    if !evidence.diff_excerpt.is_empty() {
        let _ = writeln!(
            prompt,
            "## Git diff (bounded)\n```diff\n{}\n```\n",
            evidence.diff_excerpt
        );
    }

    let _ = writeln!(
        prompt,
        "## Worker's final response\n{}",
        clip_utf8(last_assistant_text, 4_096)
    );

    let prefix_hash = blake3::hash(VERIFIER_SYSTEM_PROMPT.as_bytes())
        .to_hex()
        .to_string();
    let req = ProviderRequest {
        model: model.to_string(),
        static_system_prompt: VERIFIER_SYSTEM_PROMPT.to_string(),
        tools: Vec::new(),
        prefix_hash,
        session_id: format!("goal-verifier-{}", uuid::Uuid::new_v4()),
        messages: vec![Arc::new(Message::user_text(prompt))],
        checkpoint_index: None,
        cache_ttl: CacheTtl::Short,
        effort,
        max_output_tokens: 1_024,
        skip_tail_cache_write: true,
        native_web: NativeWebConfig::default(),
        ephemeral_tail_notes: Vec::new(),
    };

    let mut rx = provider.stream(req, cancel.clone()).await?;
    let mut raw_text = String::new();
    let mut cost_usd = 0.0_f64;

    while let Some(ev) = rx.recv().await {
        match ev {
            StreamEvent::TextDelta(t) => raw_text.push_str(&t),
            StreamEvent::Error { message, retryable } => {
                return Err(CarnotError::Provider {
                    provider: "goal-verifier".to_string(),
                    message: format!("goal verifier error: {message}"),
                    retryable,
                    retry_after_ms: None,
                });
            }
            StreamEvent::Done { message, usage, .. } => {
                if raw_text.trim().is_empty() {
                    raw_text = message.text_concat();
                }
                if usage.cost_usd.is_finite() && usage.cost_usd > 0.0 {
                    cost_usd += usage.cost_usd;
                }
                break;
            }
            _ => {}
        }
    }

    Ok((parse_verifier_json(&raw_text), cost_usd))
}

/// Parses strict JSON verdict `{"status": "met"|"not_met"|"unreachable", "feedback": "..."}`
/// with resilient extraction from markdown code fences; fails closed as `NotMet` if unparseable.
pub(crate) fn parse_verifier_json(raw: &str) -> VerifierDecision {
    let trimmed = raw.trim();
    let candidate = extract_json_object(trimmed).unwrap_or(trimmed);
    let Ok(val) = serde_json::from_str::<serde_json::Value>(candidate) else {
        return VerifierDecision::NotMet {
            feedback: format!(
                "Verifier returned non-JSON response; failing closed. Raw output: {}",
                clip_utf8(trimmed, 400)
            ),
        };
    };
    let status = val
        .get("status")
        .and_then(serde_json::Value::as_str)
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_default();
    let feedback = val
        .get("feedback")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("No additional feedback provided.")
        .to_string();

    match status.as_str() {
        "met" | "pass" | "passed" | "satisfied" => VerifierDecision::Met { feedback },
        "unreachable" | "impossible" => VerifierDecision::Unreachable { feedback },
        _ => VerifierDecision::NotMet { feedback },
    }
}

fn extract_json_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let end = s.rfind('}')?;
    (end >= start).then_some(&s[start..=end])
}

pub(crate) fn clip_utf8(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n[... truncated ({} bytes total) ...]",
        &s[..end],
        s.len()
    )
}
