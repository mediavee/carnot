//! Continuous runtime & CI Prefix-Cache Invariance Auditor.
//!
//! Guarantees byte-identical static prefixes across dates, worktrees, and session IDs.

use carnot_core::{CarnotError, Result, ToolSpec};

/// Builds Carnot's compact, 100% static system prompt (~250 tokens).
///
/// By construction, takes NO date, NO cwd, NO session id, NO git branch, NO dynamic MCP list.
pub fn static_system_prompt() -> &'static str {
    "# Carnot\n\n\
Software engineering agent operating in a workspace via `Bash`, `Read`, `Edit`, `Write`, and `Grep` (plus any additional session tools).\n\n\
## Tools\n\
- Read relevant files before editing; never guess paths or signatures.\n\
- Batch independent calls in parallel (`Read`/`Grep`, disjoint `Edit`/`Write`). Group edits to one file into one `Edit(edits:[...])`.\n\
- Prefer `Read`/`Grep`/`Edit`/`Write` over `Bash`; tools run in the workspace root (use relative paths).\n\
- For multi-line edits (>8 lines), use `Edit` with `start_line`/`end_line` (avoid repeating `old_string`) or `Write` for full small-file rewrites.\n\
- Outputs >8 KiB spill to `.carnot/h/<hash>.log`; inspect slices with `Read(offset, limit)` or `Grep`.\n\n\
## Rules\n\
- Act directly with no filler or tool-call narration; be concise, ground claims in observed output, and cite file paths.\n\
- Keep changes focused on the request; verify the final state end-to-end (including exact formats/bounds) after your last edit or cleanup."
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
