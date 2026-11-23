//! Zero-allocation-fast-path credential redaction shared by CAS disk spilling and `PolicyGuard`.

use regex::Regex;
use std::borrow::Cow;
use std::sync::OnceLock;

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
                Regex::new(
                    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
                )
                .expect("re"),
                "[REDACTED:PRIVATE_KEY]",
            ),
        ]
    })
}

/// Redacts well-known API keys, bearer tokens, and PEM private keys from `input`.
/// Returns `Cow::Borrowed` with zero allocation when no pattern matches.
pub fn redact_secrets_cow(input: &str) -> Cow<'_, str> {
    let mut current: Cow<'_, str> = Cow::Borrowed(input);
    for (re, repl) in secret_redaction_patterns() {
        if re.is_match(current.as_ref()) {
            let replaced = re.replace_all(current.as_ref(), *repl).into_owned();
            current = Cow::Owned(replaced);
        }
    }
    current
}

/// Owned convenience wrapper returning a redacted `String`.
pub fn redact_secrets(input: &str) -> String {
    redact_secrets_cow(input).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_keys_and_borrows_clean_text_without_allocating() {
        let clean = "normal compiler output with zero secrets";
        assert!(matches!(redact_secrets_cow(clean), Cow::Borrowed(_)));

        let raw = "key=sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456 tok=ghp_1234567890abcdefghij123456 aws=AKIAIOSFODNN7EXAMPLE";
        let redacted = redact_secrets(raw);
        assert!(!redacted.contains("sk-ant-"));
        assert!(!redacted.contains("ghp_"));
        assert!(!redacted.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(redacted.contains("[REDACTED:ANTHROPIC_KEY]"));
        assert!(redacted.contains("[REDACTED:GITHUB_TOKEN]"));
        assert!(redacted.contains("[REDACTED:AWS_ACCESS_KEY]"));
    }
}
