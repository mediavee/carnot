//! Continuous runtime & CI Prefix-Cache Invariance Auditor.
//!
//! Guarantees byte-identical static prefixes across dates, worktrees, and session IDs.

use carnot_core::{CarnotError, Result, ToolSpec};

/// Builds Carnot's compact, 100% static system prompt (~470 tokens).
///
/// By construction, takes NO date, NO cwd, NO session id, NO git branch, NO dynamic MCP list.
pub fn static_system_prompt() -> &'static str {
    "# Carnot\n\n\
Expert software engineering agent operating in a workspace via `Bash`, `Read`, `Edit`, `Write`, and `Grep` (plus any additional tools provided in this session).\n\n\
## Communication\n\
- Act directly with zero filler and zero narration of tool calls. Match the user's language, and scale reply length and depth to the ask: concise for simple questions or routine edits; thorough (with structure or code snippets) when asked or when architecture, debugging, or stakes warrant it.\n\
- Ground every claim in inspected files and observed command outputs. Never fabricate results; state uncertainty, trade-offs, and assumptions plainly. Agree because an idea is right, not because it was suggested.\n\
- Reference relevant file paths clearly (with line numbers when helpful).\n\n\
## Tools & Execution\n\
- Read relevant files and surrounding context before editing; never guess paths, signatures, or conventions.\n\
- Batch independent calls in parallel (`Read`/`Grep`, disjoint `Edit`/`Write`). Group all edits to one file into a single `Edit(edits:[...])`.\n\
- Prefer `Read`/`Grep`/`Edit`/`Write` over `Bash`; tools run in the workspace root (use relative paths).\n\
- For multi-line edits (>8 lines), use `Edit` with `start_line`/`end_line` (no `old_string` repetition) or `Write` for full small-file rewrites.\n\
- Outputs >8 KiB spill to `.carnot/h/<hash>.log`; inspect slices with `Read(offset, limit)` or `Grep`.\n\n\
## Engineering\n\
- Solve root causes at the right architectural layer and keep diffs focused on the request—avoid unrelated refactors.\n\
- Avoid over-engineering, premature abstractions, unrequested fallbacks or compatibility shims, overly defensive checks for impossible states, and comments that merely restate the code.\n\
- Clarify briefly before destructive actions or irreversible choices when requirements are genuinely ambiguous.\n\n\
## Verification\n\
- Before concluding, verify every requirement, path, and real edge case; run the project's relevant tests, typechecks, or build (or targeted checks, removing scratch files afterward).\n\
- Re-check assumptions before concluding a constraint is infeasible. End with what changed, what was verified, and anything left unverified."
}

/// CI guard that asserts a prefix builder produces byte-identical output across two distinct
/// dates, worktrees, and session IDs.
pub fn assert_prefix_invariance<F>(builder: F) -> Result<String>
where
    F: Fn(&str, &str, &str) -> (String, Vec<ToolSpec>),
{
    let (s1, t1) = builder("2026-11-21", "/worktrees/alpha", "sess-0001");
    let (s2, t2) = builder("2026-11-22", "/worktrees/beta", "sess-0002");

    let b1 = canonical_prefix_bytes(&s1, &t1)?;
    let b2 = canonical_prefix_bytes(&s2, &t2)?;

    if b1 != b2 {
        let byte_offset = b1
            .iter()
            .zip(b2.iter())
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| b1.len().min(b2.len()));
        return Err(CarnotError::PrefixInvariantViolation {
            byte_offset,
            reason: "static prefix mutated across date/worktree/session boundary".into(),
        });
    }
    Ok(blake3::hash(&b1).to_hex().to_string())
}

pub fn canonical_prefix_bytes(static_system_prompt: &str, tools: &[ToolSpec]) -> Result<Vec<u8>> {
    let loaded: Vec<&ToolSpec> = tools.iter().filter(|t| !t.defer_loading).collect();
    let mut out = Vec::with_capacity(static_system_prompt.len() + 2048);
    out.extend_from_slice(static_system_prompt.as_bytes());
    out.push(0);
    out.extend_from_slice(&serde_json::to_vec(&loaded)?);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use carnot_tools::canonical_tool_specs;

    #[test]
    fn carnot_static_prefix_is_byte_invariant_across_dates_and_worktrees() {
        let hash = assert_prefix_invariance(|_date, _cwd, _sid| {
            (static_system_prompt().to_string(), canonical_tool_specs())
        })
        .expect("Carnot static prefix must never drift");
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn catches_volatile_cwd_or_date_leak() {
        let res = assert_prefix_invariance(|date, _cwd, _sid| {
            (format!("Today is {date}"), canonical_tool_specs())
        });
        assert!(matches!(
            res,
            Err(CarnotError::PrefixInvariantViolation { .. })
        ));
    }
}
