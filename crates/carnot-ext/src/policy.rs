//! Enterprise Governance, Secret Redaction & Structured Audit Module (`PolicyGuardModule`).
//!
//! Plugs directly into `Module::before_tool_call` and `Module::after_tool_result` with ZERO
//! prompt-cache token overhead:
//!   - Configurable command deny regexes (e.g. `rm -rf /`, `curl ... | sh`, `git push --force`)
//!   - Protected path prefixes (e.g. `.env`, `secrets/`, `prod.tfvars`)
//!   - Automatic secret/token redaction on tool outputs (`sk-ant-...`, `sk-...`, `ghp_...`, `AKIA...`,
//!     `-----BEGIN ... PRIVATE KEY-----`) before tokens reach the model or session JSONL!
//!   - Optional append-only JSONL audit trail

use crate::{BoxFuture, Module, ToolGateDecision};
use carnot_core::{Result, ToolOutput, ToolResultBlock};
use regex::Regex;
use serde_json::{Value, json};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

fn secret_redaction_patterns() -> &'static [(Regex, &'static str)] {
    static PATS: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    PATS.get_or_init(|| {
        vec![
            (
                Regex::new(r"sk-ant-[A-Za-z0-9_-]{20,}").expect("re"),
                "[REDACTED:ANTHROPIC_KEY]",
            ),
            (
                Regex::new(r"sk-or-v1-[A-Za-z0-9]{24,}").expect("re"),
                "[REDACTED:OPENROUTER_KEY]",
            ),
            (
                Regex::new(r"sk-(?:proj-)?[A-Za-z0-9_-]{24,}").expect("re"),
                "[REDACTED:API_KEY]",
            ),
            (
                Regex::new(r"AIza[0-9A-Za-z_-]{35}").expect("re"),
                "[REDACTED:GOOGLE_KEY]",
            ),
            (
                Regex::new(r"xai-[A-Za-z0-9_-]{24,}").expect("re"),
                "[REDACTED:XAI_KEY]",
            ),
            (
                Regex::new(r"gh[pousr]_[A-Za-z0-9]{20,}").expect("re"),
                "[REDACTED:GITHUB_TOKEN]",
            ),
            (
                Regex::new(r"xox[baprs]-[A-Za-z0-9-]{20,}").expect("re"),
                "[REDACTED:SLACK_TOKEN]",
            ),
            (
                Regex::new(r"AKIA[0-9A-Z]{16}").expect("re"),
                "[REDACTED:AWS_ACCESS_KEY]",
            ),
            (
                Regex::new(r"(?i)authorization:\s*bearer\s+[A-Za-z0-9._-]{20,}").expect("re"),
                "Authorization: Bearer [REDACTED:BEARER_TOKEN]",
            ),
            (
                Regex::new(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----")
                    .expect("re"),
                "[REDACTED:PRIVATE_KEY]",
            ),
        ]
    })
}

/// Redacts well-known API keys, tokens, and PEM private keys from text.
pub fn redact_secrets(input: &str) -> String {
    let mut out = input.to_string();
    for (re, repl) in secret_redaction_patterns() {
        if re.is_match(&out) {
            out = re.replace_all(&out, *repl).into_owned();
        }
    }
    out
}

#[derive(Debug)]
pub struct PolicyGuardModule {
    denied_commands: Vec<Regex>,
    protected_path_fragments: Vec<String>,
    redact_tool_outputs: bool,
    audit_log: Option<Mutex<PathBuf>>,
}

impl Default for PolicyGuardModule {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyGuardModule {
    pub fn new() -> Self {
        Self {
            denied_commands: vec![
                Regex::new(r"(?i)\brm\s+-[a-z]*(r[a-z]*f|f[a-z]*r)[a-z]*\s+/(\*|\s|$)")
                    .expect("re"),
                Regex::new(r"(?i)\bgit\s+push\s+.*(--force|-f\b|--force-with-lease)").expect("re"),
                Regex::new(r"(?i)(curl|wget)\s+.*\|\s*(sh|bash|zsh)\b").expect("re"),
            ],
            protected_path_fragments: vec![".env".into(), "id_rsa".into(), "id_ed25519".into()],
            redact_tool_outputs: true,
            audit_log: None,
        }
    }

    pub fn with_denied_command_pattern(
        mut self,
        pattern: &str,
    ) -> std::result::Result<Self, regex::Error> {
        self.denied_commands.push(Regex::new(pattern)?);
        Ok(self)
    }

    pub fn with_protected_path(mut self, fragment: impl Into<String>) -> Self {
        self.protected_path_fragments.push(fragment.into());
        self
    }

    pub fn with_audit_log(mut self, path: PathBuf) -> Self {
        self.audit_log = Some(Mutex::new(path));
        self
    }

    fn write_audit(&self, event: Value) {
        let Some(ref lock) = self.audit_log else {
            return;
        };
        if let Ok(path) = lock.lock()
            && let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&*path)
            && let Ok(line) = serde_json::to_string(&event)
        {
            let _ = writeln!(f, "{line}");
        }
    }
}

impl Module for PolicyGuardModule {
    fn name(&self) -> &str {
        "policy_guard"
    }

    fn before_tool_call(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> BoxFuture<'_, Result<ToolGateDecision>> {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let mut deny_reason: Option<String> = None;

        if tool_name == "Bash"
            && let Some(cmd) = input.get("command").and_then(Value::as_str)
        {
            for re in &self.denied_commands {
                if re.is_match(cmd) {
                    deny_reason = Some(format!(
                        "command blocked by PolicyGuard pattern `{}`",
                        re.as_str()
                    ));
                    break;
                }
            }
        }

        if deny_reason.is_none()
            && matches!(tool_name, "Read" | "Edit" | "Write" | "Grep")
            && let Some(fp) = input
                .get("file_path")
                .or_else(|| input.get("path"))
                .and_then(Value::as_str)
        {
            let norm = fp.replace('\\', "/").to_ascii_lowercase();
            let norm_trimmed = norm.trim_start_matches("./").trim_start_matches('/');
            let file_name = std::path::Path::new(fp)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(fp)
                .to_ascii_lowercase();
            for frag in &self.protected_path_fragments {
                let fl = frag.replace('\\', "/").to_ascii_lowercase();
                let fl_dir = fl.trim_end_matches('/');
                let is_dir_match = fl.ends_with('/')
                    && (norm_trimmed == fl_dir
                        || norm_trimmed.starts_with(&format!("{fl_dir}/"))
                        || norm_trimmed.contains(&format!("/{fl_dir}/")));
                let is_file_match = file_name == fl || file_name.starts_with(&format!("{fl}."));
                if is_dir_match || is_file_match {
                    deny_reason = Some(format!(
                        "path `{fp}` is protected by PolicyGuard (`{frag}`)"
                    ));
                    break;
                }
            }
        }

        self.write_audit(json!({
            "ts_ms": ts,
            "phase": "before_tool",
            "tool": tool_name,
            "decision": if deny_reason.is_some() { "deny" } else { "allow" },
            "reason": deny_reason,
        }));

        Box::pin(async move {
            match deny_reason {
                Some(reason) => Ok(ToolGateDecision::Deny { reason }),
                None => Ok(ToolGateDecision::Allow),
            }
        })
    }

    fn after_tool_result<'a>(
        &'a self,
        _tool_name: &'a str,
        _input: &'a Value,
        output: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>> {
        if self.redact_tool_outputs {
            for b in &mut output.blocks {
                if let ToolResultBlock::Text { text } = b {
                    let clean = redact_secrets(text);
                    if clean.len() != text.len() || clean != *text {
                        *text = clean;
                    }
                }
            }
        }
        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_api_keys_github_tokens_and_aws_keys() {
        let raw = "key=sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456 tok=ghp_1234567890abcdefghij123456 aws=AKIAIOSFODNN7EXAMPLE";
        let clean = redact_secrets(raw);
        assert!(!clean.contains("sk-ant-"));
        assert!(!clean.contains("ghp_"));
        assert!(!clean.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(clean.contains("[REDACTED:ANTHROPIC_KEY]"));
        assert!(clean.contains("[REDACTED:GITHUB_TOKEN]"));
        assert!(clean.contains("[REDACTED:AWS_ACCESS_KEY]"));
    }
}
