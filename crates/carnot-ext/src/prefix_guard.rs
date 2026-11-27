//! Continuous runtime & CI Prefix-Cache Invariance Auditor.
//!
//! Guarantees byte-identical static prefixes across dates, worktrees, and session IDs.

use carnot_core::{CarnotError, Result, ToolSpec};

/// Builds Carnot's compact, 100% static system prompt (~250 tokens).
///
/// By construction, takes NO date, NO cwd, NO session id, NO git branch, NO dynamic MCP list.
pub fn static_system_prompt() -> &'static str {
    "# Carnot\n\
Coding agent in a workspace with `Bash`, `Read`, `Edit`, `Write`, `Grep` (plus any session tools).\n\n\
- Read files before editing; never guess paths or signatures.\n\
- Run independent calls in parallel (`Read`/`Grep`, disjoint `Edit`/`Write`). Batch same-file hunks in `Edit(edits:[...])`.\n\
- Prefer `Read`/`Grep`/`Edit`/`Write` over `Bash`; paths are workspace-relative.\n\
- For >8-line edits, prefer `Edit(start_line, end_line)` over repeating `old_string`, or `Write` for small full rewrites.\n\
- Outputs >8 KiB spill to `.carnot/h/<hash>.log`; read slices with `Read(offset, limit)` or `Grep`.\n\
- Be direct and concise (no filler/narration), cite file paths, ground claims in tool output, keep scope tight, and verify the final state after the last edit."
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
