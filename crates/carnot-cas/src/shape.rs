use crate::store::{CasHandle, CasStore};
use carnot_core::Result;
use regex::Regex;
use std::sync::OnceLock;

/// Result of shaping a text tool output (never called on image blocks).
#[derive(Debug, Clone)]
pub struct ShapedOutput {
    pub text: String,
    pub spilled: Option<CasHandle>,
    pub raw_bytes: usize,
    pub shaped_bytes: usize,
    pub compressor: Option<&'static str>,
}

fn ansi_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\x1B\[[0-9;?]*[ -/]*[@-~]|\x1B\][^\x07\x1B]*(?:\x07|\x1B\\)")
            .expect("valid ANSI regex")
    })
}

/// Cleans terminal noise (ANSI escapes, carriage-return progress overwrites, consecutive duplicate
/// lines) while preserving every diagnostic/failure line intact.
pub fn clean_terminal_noise(raw: &str) -> String {
    let stripped = ansi_regex().replace_all(raw, "");
    let mut cleaned_lines: Vec<String> = Vec::new();
    let mut prev_line: Option<String> = None;
    let mut repeat_count: usize = 0;

    for raw_line in stripped.split('\n') {
        let resolved = raw_line
            .trim_end_matches('\r')
            .rsplit('\r')
            .next()
            .unwrap_or("");
        if let Some(ref p) = prev_line {
            if p == resolved && !resolved.trim().is_empty() {
                repeat_count += 1;
                continue;
            } else if repeat_count > 0 {
                cleaned_lines.push(format!("  [... repeated {repeat_count} more times]"));
                repeat_count = 0;
            }
        }
        prev_line = Some(resolved.to_string());
        cleaned_lines.push(resolved.to_string());
    }
    if repeat_count > 0 {
        cleaned_lines.push(format!("  [... repeated {repeat_count} more times]"));
    }
    cleaned_lines.join("\n")
}

fn clip_line(line: &str, max_bytes: usize) -> &str {
    if line.len() <= max_bytes {
        return line;
    }
    let mut end = max_bytes;
    while end > 0 && !line.is_char_boundary(end) {
        end -= 1;
    }
    &line[..end]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandKind {
    Cargo,
    Pytest,
    JestVitest,
    Tsc,
    Eslint,
    GitDiffOrLog,
    Generic,
}

/// Splits shell chain segments (`&&`, `;`, `|`) only outside single/double quotes.
fn last_unquoted_command_segment(command: &str) -> &str {
    let bytes = command.as_bytes();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let mut seg_start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        if b == b'\\' && !in_single {
            escaped = true;
            i += 1;
            continue;
        }
        if b == b'\'' && !in_double {
            in_single = !in_single;
            i += 1;
            continue;
        }
        if b == b'"' && !in_single {
            in_double = !in_double;
            i += 1;
            continue;
        }
        if !in_single && !in_double {
            if b == b';' {
                seg_start = i + 1;
            } else if b == b'&' && i + 1 < bytes.len() && bytes[i + 1] == b'&' {
                seg_start = i + 2;
                i += 2;
                continue;
            }
        }
        i += 1;
    }
    let tail = &command[seg_start..];
    // Take before unquoted pipe `|`
    let tb = tail.as_bytes();
    in_single = false;
    in_double = false;
    escaped = false;
    for (idx, &b) in tb.iter().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if b == b'\\' && !in_single {
            escaped = true;
            continue;
        }
        if b == b'\'' && !in_double {
            in_single = !in_single;
            continue;
        }
        if b == b'"' && !in_single {
            in_double = !in_double;
            continue;
        }
        if !in_single && !in_double && b == b'|' {
            return &tail[..idx];
        }
    }
    tail
}

/// Dispatches purely on `argv[0]` + subcommand (handles `cargo +nightly test`
/// and never blindly assumes `npm test` is Jest/Vitest!).
fn classify_command(command: &str) -> CommandKind {
    let segment = last_unquoted_command_segment(command);
    let tokens: Vec<&str> = segment.split_whitespace().collect();
    let mut idx = 0usize;
    while idx < tokens.len() {
        let t = tokens[idx];
        if t.contains('=') && !t.starts_with('-') {
            idx += 1;
            continue;
        }
        if matches!(t, "env" | "npx" | "pnpx" | "bunx" | "time" | "nice") {
            idx += 1;
            continue;
        }
        if matches!(t, "pnpm" | "yarn" | "npm" | "bun" | "uv" | "poetry")
            && tokens.get(idx + 1).copied() == Some("run")
        {
            idx += 2;
            continue;
        }
        if matches!(t, "python" | "python3") && tokens.get(idx + 1).copied() == Some("-m") {
            idx += 2;
            continue;
        }
        if matches!(t, "pnpm" | "yarn" | "npm" | "bun")
            && tokens
                .get(idx + 1)
                .is_some_and(|sub| matches!(*sub, "exec" | "dlx"))
        {
            idx += 2;
            continue;
        }
        break;
    }

    let bin = tokens
        .get(idx)
        .copied()
        .unwrap_or("")
        .rsplit('/')
        .next()
        .unwrap_or("");
    let mut sub_idx = idx + 1;
    // Skip rustup toolchain selector e.g. `cargo +nightly test`.
    if bin == "cargo" && tokens.get(sub_idx).is_some_and(|t| t.starts_with('+')) {
        sub_idx += 1;
    }
    let sub = tokens.get(sub_idx).copied().unwrap_or("");

    match bin {
        "cargo" if matches!(sub, "test" | "check" | "build" | "clippy" | "nextest") => {
            CommandKind::Cargo
        }
        "pytest" | "py.test" => CommandKind::Pytest,
        "jest" | "vitest" => CommandKind::JestVitest,
        "tsc" => CommandKind::Tsc,
        "eslint" => CommandKind::Eslint,
        "git" if matches!(sub, "diff" | "log" | "show") => CommandKind::GitDiffOrLog,
        _ => CommandKind::Generic,
    }
}

/// Accurate diagnostic matcher that avoids false positives like `"0 errors"` or `"error_handling.rs"`.
fn is_genuine_diagnostic_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    let low = trimmed.to_ascii_lowercase();
    if low.contains("0 errors") || low.contains("0 failed") || low.contains("0 warnings") {
        return false;
    }
    low.starts_with("error:")
        || low.starts_with("error[")
        || low.contains(": error:")
        || low.contains(": error ts")
        || low.starts_with("failed ")
        || low.starts_with("fail ")
        || low.starts_with("panicked at")
        || low.contains("thread '") && low.contains("panicked")
        || low.starts_with("traceback (most recent call last)")
        || low.starts_with("assertion")
        || (low.starts_with("test ") && low.ends_with("... failed"))
}

/// Deterministic, lossless-diagnostic compressor for `cargo test|check|build|clippy`.
/// Prioritizes errors, `failures:`, and `test result:` BEFORE warnings so truncation NEVER drops
/// real errors behind noisy warnings.
fn compress_cargo(cleaned: &str) -> Option<(String, &'static str)> {
    let mut compiled_count = 0usize;
    let mut passed_tests = 0usize;
    let mut kept_priority: Vec<&str> = Vec::new();
    let mut kept_warnings: Vec<&str> = Vec::new();
    let mut diag_is_warning = false;
    let mut in_diag_block = false;
    let mut in_failures_section = false;

    for line in cleaned.lines() {
        let trim = line.trim_start();
        if trim.starts_with("Compiling ")
            || trim.starts_with("Checking ")
            || trim.starts_with("Downloading ")
            || trim.starts_with("Downloaded ")
            || trim.starts_with("Fresh ")
            || trim.starts_with("Locking ")
            || trim.starts_with("Adding ")
        {
            compiled_count += 1;
            in_diag_block = false;
            continue;
        }
        if ((trim.starts_with("test ") && trim.ends_with("... ok")) || trim.starts_with("PASS ["))
            && !in_failures_section
        {
            passed_tests += 1;
            in_diag_block = false;
            continue;
        }
        if trim == "failures:" {
            in_failures_section = true;
            in_diag_block = true;
            diag_is_warning = false;
            kept_priority.push(line);
            continue;
        }
        if trim.starts_with("test result:") || trim.starts_with("Summary [") {
            in_failures_section = false;
            in_diag_block = false;
            kept_priority.push(line);
            continue;
        }
        if in_failures_section {
            kept_priority.push(line);
            continue;
        }
        if trim.starts_with("error")
            || trim.starts_with("FAILED")
            || trim.starts_with("FAIL [")
            || (trim.starts_with("test ") && trim.contains("FAILED"))
        {
            in_diag_block = true;
            diag_is_warning = false;
            kept_priority.push(line);
            continue;
        }
        if trim.starts_with("warning:") || trim.starts_with("warning[") {
            in_diag_block = true;
            diag_is_warning = true;
            kept_warnings.push(line);
            continue;
        }
        if in_diag_block {
            if trim.starts_with("-->")
                || trim.starts_with('|')
                || trim.starts_with('=')
                || trim.chars().next().is_some_and(|c| c.is_ascii_digit())
                || trim.starts_with("help:")
                || trim.starts_with("note:")
                || trim.starts_with("For more information")
                || trim.is_empty()
            {
                if diag_is_warning {
                    kept_warnings.push(line);
                } else {
                    kept_priority.push(line);
                }
                if trim.is_empty() {
                    in_diag_block = false;
                }
                continue;
            }
            in_diag_block = false;
        }
        if trim.starts_with("Finished ")
            || trim.starts_with("Running ")
            || trim.starts_with("Doc-tests ")
            || trim.starts_with("running ")
        {
            kept_priority.push(line);
        }
    }

    if compiled_count + passed_tests < 8 {
        return None;
    }
    let mut out = format!(
        "[cargo summary: {compiled_count} crate build steps collapsed, {passed_tests} passing tests collapsed]\n"
    );
    // Errors, failures, and `test result:` ALWAYS come first!
    if !kept_priority.is_empty() {
        out.push_str(&kept_priority.join("\n"));
    }
    if !kept_warnings.is_empty() {
        out.push_str("\n[warnings]\n");
        out.push_str(&kept_warnings.join("\n"));
    }
    Some((out, "cargo"))
}

/// Deterministic, lossless-diagnostic compressor for `pytest`.
fn compress_pytest(cleaned: &str) -> Option<(String, &'static str)> {
    let mut passed_lines = 0usize;
    let mut kept: Vec<&str> = Vec::new();
    let mut in_failure_section = false;

    for line in cleaned.lines() {
        let trim = line.trim();
        if trim.contains("FAILURES") && trim.starts_with('=')
            || trim.contains("ERRORS") && trim.starts_with('=')
            || trim.contains("short test summary info")
        {
            in_failure_section = true;
            kept.push(line);
            continue;
        }
        if !in_failure_section
            && (trim.ends_with(" PASSED")
                || (trim.contains("::") && trim.contains(" PASSED "))
                || (trim.len() > 20 && trim.chars().all(|c| c == '.' || c == 's' || c == 'x')))
        {
            passed_lines += 1;
            continue;
        }
        if in_failure_section
            || trim.starts_with("FAILED ")
            || trim.starts_with("ERROR ")
            || trim.starts_with("E   ")
            || (trim.starts_with('=') && trim.ends_with('='))
            || trim.starts_with("collected ")
        {
            kept.push(line);
        }
    }

    if passed_lines < 6 {
        return None;
    }
    let out = format!(
        "[pytest summary: {passed_lines} passing test lines collapsed]\n{}",
        kept.join("\n")
    );
    Some((out, "pytest"))
}

/// Deterministic, lossless-diagnostic compressor for `jest` / `vitest`.
fn compress_jest_vitest(cleaned: &str) -> Option<(String, &'static str)> {
    let mut passed_lines = 0usize;
    let mut kept: Vec<&str> = Vec::new();
    let mut in_fail_block = false;

    for line in cleaned.lines() {
        let trim = line.trim_start();
        if trim.starts_with("PASS ")
            || trim.starts_with("✓ ")
            || trim.starts_with("✔ ")
            || trim.starts_with("√ ")
        {
            passed_lines += 1;
            in_fail_block = false;
            continue;
        }
        if trim.starts_with("FAIL ")
            || trim.starts_with("● ")
            || trim.starts_with("× ")
            || trim.starts_with("✕ ")
            || trim.starts_with("FAILURES")
        {
            in_fail_block = true;
            kept.push(line);
            continue;
        }
        if trim.starts_with("Test Suites:")
            || trim.starts_with("Test Files")
            || trim.starts_with("Tests:")
            || trim.starts_with("Snapshots:")
            || trim.starts_with("Time:")
            || trim.starts_with("Duration")
        {
            in_fail_block = false;
            kept.push(line);
            continue;
        }
        if in_fail_block {
            kept.push(line);
        }
    }

    if passed_lines < 6 {
        return None;
    }
    let out = format!(
        "[jest/vitest summary: {passed_lines} passing suites/tests collapsed]\n{}",
        kept.join("\n")
    );
    Some((out, "jest_vitest"))
}

/// Deterministic, lossless-diagnostic compressor for `tsc` (standard & `--pretty`!) and `eslint`.
fn compress_tsc_or_eslint(cleaned: &str, kind: CommandKind) -> Option<(String, &'static str)> {
    let mut skipped = 0usize;
    let mut kept: Vec<&str> = Vec::new();
    for line in cleaned.lines() {
        let trim = line.trim();
        let is_keep = match kind {
            CommandKind::Tsc => {
                // Supports both standard `file(l,c): error TS...` and `tsc --pretty` `file:l:c - error TS...`.
                trim.contains("): error TS")
                    || trim.contains(": error TS")
                    || trim.contains(" - error TS")
                    || trim.starts_with("Found ")
                    || line.starts_with("  ")
            }
            CommandKind::Eslint => {
                // Real eslint file headers are bare paths without spaces (never "Linting file ...")!
                let is_bare_file_header = !trim.is_empty()
                    && !line.starts_with(' ')
                    && !trim.contains(' ')
                    && (trim.ends_with(".ts")
                        || trim.ends_with(".tsx")
                        || trim.ends_with(".js")
                        || trim.ends_with(".jsx")
                        || trim.ends_with(".mjs")
                        || trim.ends_with(".cjs"));
                trim.contains(" error ")
                    || trim.contains(" warning ")
                    || trim.starts_with('✖')
                    || trim.ends_with(" problems")
                    || trim.contains(" errors, ")
                    || is_bare_file_header
            }
            _ => true,
        };
        if is_keep {
            kept.push(line);
        } else {
            skipped += 1;
        }
    }
    if skipped < 6 || kept.is_empty() {
        return None;
    }
    let label = if kind == CommandKind::Tsc {
        "tsc"
    } else {
        "eslint"
    };
    let out = format!(
        "[{label} summary: {skipped} non-diagnostic lines collapsed]\n{}",
        kept.join("\n")
    );
    Some((out, label))
}

/// Shapes a text output using generic head/diagnostics/tail spilling (no command context).
pub fn shape_text_output(
    raw: &str,
    store: Option<&CasStore>,
    spill_threshold_bytes: usize,
    preview_max_bytes: usize,
) -> Result<ShapedOutput> {
    shape_command_output(None, raw, store, spill_threshold_bytes, preview_max_bytes)
}

/// Command-aware semantic shaper + lossless CAS spillover.
/// Calls `shape_command_output_with_redaction(..., false)` (use `redact_secrets = true`
/// for shell/MCP outputs so credentials are scrubbed before writing `.carnot/h/<hash>.log`).
pub fn shape_command_output(
    command: Option<&str>,
    raw: &str,
    store: Option<&CasStore>,
    spill_threshold_bytes: usize,
    preview_max_bytes: usize,
) -> Result<ShapedOutput> {
    shape_command_output_with_redaction(
        command,
        raw,
        store,
        spill_threshold_bytes,
        preview_max_bytes,
        false,
    )
}

/// Command-aware semantic shaper + lossless CAS spillover with optional pre-spill secret redaction.
/// When `redact_secrets` is `true`, well-known credentials/keys are redacted BEFORE spilling bytes
/// to `.carnot/h/<hash>.log` on disk AND before constructing the inline preview.
pub fn shape_command_output_with_redaction(
    command: Option<&str>,
    raw: &str,
    store: Option<&CasStore>,
    spill_threshold_bytes: usize,
    preview_max_bytes: usize,
    redact_secrets: bool,
) -> Result<ShapedOutput> {
    let safe_raw = if redact_secrets {
        crate::redact::redact_secrets_cow(raw)
    } else {
        std::borrow::Cow::Borrowed(raw)
    };
    let raw = safe_raw.as_ref();
    let raw_bytes = raw.len();
    let cleaned = clean_terminal_noise(raw);

    // Without a CAS backing store, keep head + tail (8 KiB + 24 KiB) so failure diagnostics at
    // the end of long build/test logs are preserved cleanly.
    let Some(cas) = store else {
        const NO_CAS_WIRE_CAP: usize = 32 * 1024;
        const NO_CAS_HEAD: usize = 8 * 1024;
        const NO_CAS_TAIL: usize = 24 * 1024;
        let out = if cleaned.len() > NO_CAS_WIRE_CAP {
            let mut h = NO_CAS_HEAD.min(cleaned.len());
            while h > 0 && !cleaned.is_char_boundary(h) {
                h -= 1;
            }
            let mut t = cleaned.len().saturating_sub(NO_CAS_TAIL);
            while t < cleaned.len() && !cleaned.is_char_boundary(t) {
                t += 1;
            }
            let omitted = t.saturating_sub(h);
            format!(
                "{}\n[... {omitted} bytes omitted (CAS disabled; keeping head + tail) ...]\n{}",
                &cleaned[..h],
                &cleaned[t..]
            )
        } else {
            cleaned
        };
        let shaped_bytes = out.len();
        return Ok(ShapedOutput {
            text: out,
            spilled: None,
            raw_bytes,
            shaped_bytes,
            compressor: None,
        });
    };

    // 1. Try specialized deterministic semantic compressor when command is known & output is sizable
    let kind = command
        .map(classify_command)
        .unwrap_or(CommandKind::Generic);
    if cleaned.len() > 1_200 {
        let compressed_opt = match kind {
            CommandKind::Cargo => compress_cargo(&cleaned),
            CommandKind::Pytest => compress_pytest(&cleaned),
            CommandKind::JestVitest => compress_jest_vitest(&cleaned),
            CommandKind::Tsc => compress_tsc_or_eslint(&cleaned, CommandKind::Tsc),
            CommandKind::Eslint => compress_tsc_or_eslint(&cleaned, CommandKind::Eslint),
            CommandKind::GitDiffOrLog | CommandKind::Generic => None,
        };
        if let Some((mut comp_text, comp_name)) = compressed_opt
            && comp_text.len() * 100 <= cleaned.len() * 80
        {
            // Store exact raw bytes in CAS for lossless retrieval.
            let handle = cas.put_bytes(raw.as_bytes())?;
            if comp_text.len() > preview_max_bytes.saturating_mul(2).max(2_400) {
                let cap = preview_max_bytes.saturating_mul(2).max(2_400);
                let mut end = cap;
                while end > 0 && !comp_text.is_char_boundary(end) {
                    end -= 1;
                }
                comp_text.truncate(end);
                comp_text.push_str(
                    "\n[... compressed diagnostics truncated; inspect full CAS handle ...]",
                );
            }
            comp_text.push_str(&format!(
                "\n[full raw output: {} ({} lines, {} bytes, compressor={comp_name})]",
                handle.rel_path, handle.lines, handle.bytes
            ));
            let shaped_bytes = comp_text.len();
            return Ok(ShapedOutput {
                text: comp_text,
                spilled: Some(handle),
                raw_bytes,
                shaped_bytes,
                compressor: Some(comp_name),
            });
        }
    }

    if cleaned.len() <= spill_threshold_bytes {
        let shaped_bytes = cleaned.len();
        return Ok(ShapedOutput {
            text: cleaned,
            spilled: None,
            raw_bytes,
            shaped_bytes,
            compressor: None,
        });
    }

    // Store exact raw bytes in CAS for lossless retrieval.
    let handle = Some(cas.put_bytes(raw.as_bytes())?);

    let lines: Vec<&str> = cleaned.lines().collect();
    let total_lines = lines.len();
    let line_cap = (preview_max_bytes / 4).max(160);

    let mut diag_lines: Vec<(usize, &str)> = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if is_genuine_diagnostic_line(line) {
            diag_lines.push((idx + 1, clip_line(line, line_cap)));
        }
    }

    let mut preview = String::with_capacity(preview_max_bytes + 256);
    let head_budget = preview_max_bytes * 35 / 100;
    let mut head_end = 0usize;
    for (i, l) in lines.iter().enumerate() {
        let clipped = clip_line(l, head_budget.min(line_cap));
        if preview.len() + clipped.len() + 1 > head_budget && i > 0 {
            break;
        }
        if !preview.is_empty() {
            preview.push('\n');
        }
        preview.push_str(clipped);
        head_end = i + 1;
    }

    let diag_budget = preview_max_bytes * 25 / 100;
    let mut diag_buf = String::new();
    let tail_start_est = total_lines.saturating_sub(12);
    for (ln, txt) in diag_lines {
        if ln <= head_end || ln > tail_start_est {
            continue;
        }
        let entry = format!("L{ln}: {txt}\n");
        if diag_buf.len() + entry.len() > diag_budget {
            break;
        }
        diag_buf.push_str(&entry);
    }

    let tail_budget = preview_max_bytes * 40 / 100;
    let mut tail_rev: Vec<&str> = Vec::new();
    let mut tail_len = 0usize;
    let mut tail_start = total_lines;
    for i in (head_end..total_lines).rev() {
        let clipped = clip_line(lines[i], tail_budget.min(line_cap));
        if tail_len + clipped.len() + 1 > tail_budget && !tail_rev.is_empty() {
            break;
        }
        tail_rev.push(clipped);
        tail_len += clipped.len() + 1;
        tail_start = i;
    }
    tail_rev.reverse();

    let omitted = tail_start.saturating_sub(head_end);
    if omitted > 0 {
        preview.push_str(&format!("\n... [{omitted} lines omitted] ..."));
    }
    if !diag_buf.is_empty() {
        preview.push_str("\n[key diagnostics]\n");
        preview.push_str(diag_buf.trim_end());
    }
    if !tail_rev.is_empty() {
        preview.push('\n');
        preview.push_str(&tail_rev.join("\n"));
    }

    if let Some(ref h) = handle {
        preview.push_str(&format!(
            "\n[full output: {} ({} lines, {} bytes) — use Read(offset, limit) or Grep]",
            h.rel_path, h.lines, h.bytes
        ));
    }

    let shaped_bytes = preview.len();
    Ok(ShapedOutput {
        text: preview,
        spilled: handle,
        raw_bytes,
        shaped_bytes,
        compressor: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleans_ansi_and_progress_cr() {
        let raw = "\x1b[32mCompiling\x1b[0m\n10%\r50%\r100% done\nwarn\nwarn\nwarn\n";
        let out = clean_terminal_noise(raw);
        assert!(out.contains("Compiling"));
        assert!(out.contains("100% done"));
        assert!(!out.contains("10%"));
        assert!(out.contains("repeated 2 more times"));
    }

    #[test]
    fn spills_large_text_with_diagnostics_preserved() {
        let dir = std::env::temp_dir().join(format!("carnot-shape-test-{}", uuid::Uuid::new_v4()));
        let store = CasStore::open(&dir).expect("cas");
        let mut big = String::new();
        for i in 0..300 {
            if i == 150 {
                big.push_str("error: src/auth.rs:42 invalid token signature\n");
            } else {
                big.push_str(&format!("info line {i}: normal log output padding text\n"));
            }
        }
        let shaped = shape_text_output(&big, Some(&store), 4096, 1400).expect("shape");
        assert!(shaped.spilled.is_some());
        assert!(shaped.text.contains("error: src/auth.rs:42"));
        assert!(shaped.text.contains(".carnot/h/"));
        assert!(shaped.shaped_bytes < 2400);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cargo_compressor_preserves_100pct_diagnostics_and_is_deterministic() {
        let dir = std::env::temp_dir().join(format!("carnot-cargo-comp-{}", uuid::Uuid::new_v4()));
        let store = CasStore::open(&dir).expect("cas");
        let mut log = String::new();
        for i in 0..60 {
            log.push_str(&format!("   Compiling crate_{i} v0.1.{i}\n"));
        }
        for i in 0..120 {
            log.push_str(&format!("test module::case_{i} ... ok\n"));
        }
        log.push_str("test module::auth_check ... FAILED\n\nfailures:\n\n---- module::auth_check stdout ----\nthread 'module::auth_check' panicked at src/auth.rs:88:9:\nassertion `left == right` failed\n  left: 401\n right: 200\n\nfailures:\n    module::auth_check\n\ntest result: FAILED. 120 passed; 1 failed; 0 ignored\n");

        let s1 = shape_command_output(
            Some("cargo test --workspace"),
            &log,
            Some(&store),
            4096,
            1400,
        )
        .expect("s1");
        let s2 = shape_command_output(
            Some("cargo test --workspace"),
            &log,
            Some(&store),
            4096,
            1400,
        )
        .expect("s2");
        assert_eq!(s1.compressor, Some("cargo"));
        assert_eq!(s1.text, s2.text, "compressor MUST be byte-deterministic!");
        assert!(
            s1.shaped_bytes * 4 < s1.raw_bytes,
            "expected >75% reduction"
        );
        // 100% diagnostic recall
        assert!(s1.text.contains("src/auth.rs:88:9"));
        assert!(s1.text.contains("left: 401"));
        assert!(
            s1.text
                .contains("test result: FAILED. 120 passed; 1 failed")
        );
        // Exact CAS integrity
        let h = s1.spilled.expect("cas handle");
        assert_eq!(store.get_verified(&h.hash).expect("read"), log);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
