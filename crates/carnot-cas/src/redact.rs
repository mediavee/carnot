//! Zero-allocation-fast-path credential redaction shared by CAS disk spilling and `PolicyGuard`.

use regex::{Regex, RegexSet};
use std::borrow::Cow;
use std::sync::LazyLock;

const SECRET_SPECS: &[(&str, &str)] = &[
    (r"sk-ant-[A-Za-z0-9_-]{20,}", "[REDACTED:ANTHROPIC_KEY]"),
    (r"sk-or-v1-[A-Za-z0-9]{24,}", "[REDACTED:OPENROUTER_KEY]"),
    (r"sk-(?:proj-)?[A-Za-z0-9_-]{24,}", "[REDACTED:API_KEY]"),
    (r"AIza[0-9A-Za-z_-]{35}", "[REDACTED:GOOGLE_KEY]"),
    (r"xai-[A-Za-z0-9_-]{24,}", "[REDACTED:XAI_KEY]"),
    (r"gh[pousr]_[A-Za-z0-9]{20,}", "[REDACTED:GITHUB_TOKEN]"),
    (r"xox[baprs]-[A-Za-z0-9-]{20,}", "[REDACTED:SLACK_TOKEN]"),
    (r"AKIA[0-9A-Z]{16}", "[REDACTED:AWS_ACCESS_KEY]"),
    (
        r"(?i)authorization:\s*bearer\s+[A-Za-z0-9._-]{20,}",
        "Authorization: Bearer [REDACTED:BEARER_TOKEN]",
    ),
    (
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
        "[REDACTED:PRIVATE_KEY]",
    ),
];

struct SecretRedactor {
    set: RegexSet,
    patterns: Vec<(Regex, &'static str)>,
}

static REDACTOR: LazyLock<SecretRedactor> = LazyLock::new(|| {
    let set = RegexSet::new(SECRET_SPECS.iter().map(|(pat, _)| *pat))
        .expect("static secret redaction RegexSet must compile");
    let patterns = SECRET_SPECS
        .iter()
        .map(|(pat, repl)| {
            (
                Regex::new(pat).expect("static secret redaction Regex must compile"),
                *repl,
            )
        })
        .collect();
    SecretRedactor { set, patterns }
});

/// Redacts well-known API keys, bearer tokens, and PEM private keys from `input`.
/// Uses a single-pass `RegexSet` pre-check so clean text returns `Cow::Borrowed`
/// with zero allocations and without running individual replacement regexes.
pub fn redact_secrets_cow(input: &str) -> Cow<'_, str> {
    let matches = REDACTOR.set.matches(input);
    if !matches.matched_any() {
        return Cow::Borrowed(input);
    }
    let mut current: Cow<'_, str> = Cow::Borrowed(input);
    for idx in matches {
        let (ref re, repl) = REDACTOR.patterns[idx];
        let replaced = re.replace_all(current.as_ref(), repl).into_owned();
        current = Cow::Owned(replaced);
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
