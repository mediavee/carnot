//! Canonical `Read`, `Edit`, and `Write` tools:
//!   - Strict path confinement to `workspace_root` + `extra_writable_roots` (normalizes `..`
//!     and resolves canonical symlink targets before reading or writing)
//!   - `Read` never loops through CAS spill; caps at 16 MiB & 2,000 lines with explicit offset hint
//!   - `ReadTracker` stores only unnumbered raw lines actually returned
//!   - `Edit` rejects empty `old_string` and preserves CRLF line endings
//!   - `atomic_write` follows symlink targets (re-checked for confinement), preserves permissions,
//!     and removes temp files on error
//!   - Saturating range math & `tokio::task::spawn_blocking` off the async executor
//!   - Path-level `ToolEffects` for parallel disjoint `Edit`/`Write` execution

mod edit;
mod path;
mod read;
mod write;

pub use edit::EditTool;
pub(crate) use path::atomic_write_raw_bytes;
pub use path::resolve_confined_path;
pub use read::ReadTool;
pub use write::WriteTool;

#[cfg(test)]
mod tests {
    use super::edit::{HunkMatchMode, apply_resilient_hunk};
    use super::*;
    use carnot_cas::CasStore;
    use serde_json::json;
    use std::fs;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    #[test]
    fn confinement_blocks_parent_traversal_and_abs_escape() {
        let tmp = std::env::temp_dir().join(format!("carnot-s2-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).expect("mkdir");
        assert!(resolve_confined_path(&tmp, &[], "../etc/passwd").is_err());
        assert!(resolve_confined_path(&tmp, &[], "/etc/passwd").is_err());
        assert!(resolve_confined_path(&tmp, &[], "src/main.rs").is_ok());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn resilient_edit_reindents_unique_match_rejects_ambiguous_and_diagnoses_miss() {
        let src = "fn demo() {\n    let x = 1;\n    let y = x + 2;\n    println!(\"{y}\");\n}\n";
        // 1. Indentation-adjusted unique match (2 spaces instead of 4) succeeds & preserves 4-space indent!
        let (after, cnt, mode) = apply_resilient_hunk(
            src,
            "  let x = 1;\n  let y = x + 2;",
            "  let x = 10;\n  let y = x + 20;",
            false,
            "demo.rs",
        )
        .expect("unique indent match");
        assert_eq!(cnt, 1);
        assert!(matches!(mode, HunkMatchMode::IndentAdjustedUnique(2, 3)));
        assert!(after.contains("    let x = 10;\n    let y = x + 20;"));

        // 2. Ambiguous fuzzy match MUST be rejected (never silently applied!)
        let dup = "fn a() {\n    let k = 1;\n    return k;\n}\nfn b() {\n        let k = 1;\n        return k;\n}\n";
        let err_amb = apply_resilient_hunk(
            dup,
            "let k = 1;\nreturn k;",
            "let k = 2;\nreturn k;",
            false,
            "dup.rs",
        )
        .expect_err("expected ambiguity error");
        assert!(err_amb.contains("ambiguous"), "got: {err_amb}");

        // 3. Miss returns numbered closest window + hint!
        let err_miss = apply_resilient_hunk(
            src,
            "let x = 999;\nlet y = x + 2;",
            "let x = 0;",
            false,
            "demo.rs",
        )
        .expect_err("expected miss error");
        assert!(
            err_miss.contains("Closest matching region"),
            "got: {err_miss}"
        );
        assert!(err_miss.contains("let x = 1;"), "got: {err_miss}");
    }

    #[tokio::test]
    async fn atomic_batch_edit_dir_outline_and_cas_undo() {
        let tmp = std::env::temp_dir().join(format!("carnot-undo-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).expect("mkdir");
        let file = tmp.join("lib.rs");
        let orig = "pub fn alpha() -> i32 { 1 }\npub fn beta() -> i32 { 2 }\n";
        fs::write(&file, orig).expect("write");

        let cfg = carnot_core::CarnotConfig::new(tmp.clone());
        let cas = Arc::new(CasStore::open(&tmp).expect("cas"));
        let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);

        // Read directory returns structural tree-sitter outline!
        let dir_out = reg
            .get("Read")
            .expect("Read")
            .execute(json!({"file_path": "."}), CancellationToken::new())
            .await
            .expect("dir read");
        assert!(!dir_out.is_error);
        assert!(dir_out.text_only().contains("directory outline"));
        assert!(dir_out.text_only().contains("lib.rs"));

        // Atomic batch Edit applies 2 hunks cleanly
        let edit_out = reg
            .get("Edit")
            .expect("Edit")
            .execute(
                json!({
                    "file_path": "lib.rs",
                    "edits": [
                        {"old_string": "alpha() -> i32 { 1 }", "new_string": "alpha() -> i32 { 10 }"},
                        {"old_string": "beta() -> i32 { 2 }", "new_string": "beta() -> i32 { 20 }"}
                    ]
                }),
                CancellationToken::new(),
            )
            .await
            .expect("edit");
        assert!(!edit_out.is_error);
        assert!(fs::read_to_string(&file).expect("read").contains("{ 20 }"));

        // Instant CAS Undo restores exact original file!
        let undone = reg.undo_last_edit().expect("undo");
        assert!(undone.is_some());
        assert_eq!(fs::read_to_string(&file).expect("read"), orig);

        let _ = fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn line_range_edit_batch_bottom_up_overlap_delete_and_crlf() {
        let tmp = std::env::temp_dir().join(format!("carnot-lr-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).expect("mkdir");
        let file = tmp.join("mod.rs");
        let orig = "line1\nline2\nline3\nline4\nline5\n";
        fs::write(&file, orig).expect("write");

        let cfg = carnot_core::CarnotConfig::new(tmp.clone());
        let cas = Arc::new(CasStore::open(&tmp).expect("cas"));
        let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let edit = reg.get("Edit").expect("Edit");

        // 1. Multi-range batch applied bottom-up against ORIGINAL line numbers (expanding L2 does not shift L4-L5!)
        let res = edit
            .execute(
                json!({
                    "file_path": "mod.rs",
                    "edits": [
                        {"start_line": 2, "end_line": 2, "new_string": "L2a\nL2b\nL2c"},
                        {"start_line": 4, "end_line": 5, "new_string": "L45"}
                    ]
                }),
                CancellationToken::new(),
            )
            .await
            .expect("batch");
        assert!(!res.is_error, "got: {}", res.text_only());
        assert_eq!(
            fs::read_to_string(&file).expect("read"),
            "line1\nL2a\nL2b\nL2c\nline3\nL45\n"
        );

        // 2. Overlapping line ranges are rejected atomically without modifying the file
        let err_ov = edit
            .execute(
                json!({
                    "file_path": "mod.rs",
                    "edits": [
                        {"start_line": 1, "end_line": 3, "new_string": "x"},
                        {"start_line": 3, "end_line": 4, "new_string": "y"}
                    ]
                }),
                CancellationToken::new(),
            )
            .await
            .expect("overlap");
        assert!(err_ov.is_error);
        assert!(err_ov.text_only().contains("overlapping"));

        // 3. Empty `new_string` deletes the line range cleanly
        let del = edit
            .execute(
                json!({"file_path": "mod.rs", "start_line": 2, "end_line": 4, "new_string": ""}),
                CancellationToken::new(),
            )
            .await
            .expect("delete");
        assert!(!del.is_error);
        assert_eq!(
            fs::read_to_string(&file).expect("read"),
            "line1\nline3\nL45\n"
        );

        // 4. CRLF line endings preserved
        let crlf_file = tmp.join("win.txt");
        fs::write(&crlf_file, "a\r\nb\r\nc\r\n").expect("write crlf");
        let crlf_res = edit
            .execute(
                json!({"file_path": "win.txt", "start_line": 2, "end_line": 2, "new_string": "B1\nB2"}),
                CancellationToken::new(),
            )
            .await
            .expect("crlf");
        assert!(!crlf_res.is_error);
        assert_eq!(
            fs::read_to_string(&crlf_file).expect("read"),
            "a\r\nB1\r\nB2\r\nc\r\n"
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[cfg(unix)]
    #[test]
    fn confinement_blocks_dangling_symlink_escape() {
        use std::os::unix::fs::symlink;
        let tmp = std::env::temp_dir().join(format!("carnot-dsym-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).expect("mkdir");
        let outside_missing = std::env::temp_dir()
            .join(format!("carnot-outside-missing-{}", uuid::Uuid::new_v4()))
            .join("secret.txt");
        let evil_link = tmp.join("escape");
        symlink(&outside_missing, &evil_link).expect("symlink outside");
        assert!(
            resolve_confined_path(&tmp, &[], "escape").is_err(),
            "dangling symlink pointing outside workspace MUST be rejected"
        );

        let inside_missing = tmp.join("sub/valid.txt");
        let ok_link = tmp.join("local_link");
        symlink(&inside_missing, &ok_link).expect("symlink inside");
        assert!(
            resolve_confined_path(&tmp, &[], "local_link").is_ok(),
            "dangling symlink pointing inside workspace should be allowed"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn edit_rejects_mixed_line_range_and_text_hunks() {
        let tmp = std::env::temp_dir().join(format!("carnot-mix-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).expect("mkdir");
        let file = tmp.join("mix.rs");
        let orig = "a\nb\nc\n";
        fs::write(&file, orig).expect("write");

        let cfg = carnot_core::CarnotConfig::new(tmp.clone());
        let cas = Arc::new(CasStore::open(&tmp).expect("cas"));
        let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let edit = reg.get("Edit").expect("Edit");

        let res = edit
            .execute(
                json!({
                    "file_path": "mix.rs",
                    "edits": [
                        {"old_string": "a", "new_string": "a1\na2"},
                        {"start_line": 2, "end_line": 2, "new_string": "B"}
                    ]
                }),
                CancellationToken::new(),
            )
            .await
            .expect("edit exec");
        assert!(res.is_error);
        assert!(res.text_only().contains("cannot mix"));
        assert_eq!(fs::read_to_string(&file).expect("read"), orig);
        let _ = fs::remove_dir_all(&tmp);
    }
}
