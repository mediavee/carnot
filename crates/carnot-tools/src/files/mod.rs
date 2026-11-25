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
        let read = reg.get("Read").expect("Read");
        let edit = reg.get("Edit").expect("Edit");

        // 0. Blind line-range edit before Read is rejected!
        let blind = edit
            .execute(
                json!({"file_path": "mod.rs", "start_line": 2, "end_line": 2, "new_string": "X"}),
                CancellationToken::new(),
            )
            .await
            .expect("blind");
        assert!(blind.is_error, "blind line-range edit must be rejected");
        assert!(
            blind.text_only().contains("not previously read"),
            "got: {}",
            blind.text_only()
        );

        // Read populates the harness-side line view.
        let r0 = read
            .execute(json!({"file_path": "mod.rs"}), CancellationToken::new())
            .await
            .expect("read");
        assert!(!r0.is_error, "read failed: {}", r0.text_only());

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

        // 3. Re-read then empty `new_string` deletes the line range cleanly
        let r1 = read
            .execute(json!({"file_path": "mod.rs"}), CancellationToken::new())
            .await
            .expect("reread");
        assert!(!r1.is_error, "reread failed: {}", r1.text_only());
        let del = edit
            .execute(
                json!({"file_path": "mod.rs", "start_line": 2, "end_line": 4, "new_string": ""}),
                CancellationToken::new(),
            )
            .await
            .expect("delete");
        assert!(!del.is_error, "delete failed: {}", del.text_only());
        assert_eq!(
            fs::read_to_string(&file).expect("read"),
            "line1\nline3\nL45\n"
        );

        // 4. CRLF line endings preserved
        let crlf_file = tmp.join("win.txt");
        fs::write(&crlf_file, "a\r\nb\r\nc\r\n").expect("write crlf");
        let r_crlf = read
            .execute(json!({"file_path": "win.txt"}), CancellationToken::new())
            .await
            .expect("read crlf");
        assert!(!r_crlf.is_error, "read crlf failed: {}", r_crlf.text_only());
        let crlf_res = edit
            .execute(
                json!({"file_path": "win.txt", "start_line": 2, "end_line": 2, "new_string": "B1\nB2"}),
                CancellationToken::new(),
            )
            .await
            .expect("crlf");
        assert!(
            !crlf_res.is_error,
            "crlf edit failed: {}",
            crlf_res.text_only()
        );
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

    #[tokio::test]
    async fn line_range_edit_relocates_shifted_second_edit_and_rejects_stale_zone() {
        let tmp = std::env::temp_dir().join(format!("carnot-anchor-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).expect("mkdir");
        let file = tmp.join("app.rs");
        let mut orig = String::new();
        for i in 1..=30 {
            use std::fmt::Write as _;
            let _ = writeln!(orig, "pub fn step_{i:02}() -> usize {{ {i} }}");
        }
        fs::write(&file, &orig).expect("write");

        let cfg = carnot_core::CarnotConfig::new(tmp.clone());
        let cas = Arc::new(CasStore::open(&tmp).expect("cas"));
        let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let read = reg.get("Read").expect("Read");
        let edit = reg.get("Edit").expect("Edit");

        // 1. Model reads lines 1..=30
        let r1 = read
            .execute(json!({"file_path": "app.rs"}), CancellationToken::new())
            .await
            .expect("read");
        assert!(!r1.is_error, "initial read failed: {}", r1.text_only());

        // 2. First Edit expands L3..=L4 by +3 lines -> rebases `view` AND emits `[lines after L4 shifted by +3]`
        let e1 = edit
            .execute(
                json!({
                    "file_path": "app.rs",
                    "start_line": 3,
                    "end_line": 4,
                    "new_string": "pub fn step_03_a() -> usize { 301 }\npub fn step_03_b() -> usize { 302 }\npub fn step_03_c() -> usize { 303 }\npub fn step_04_a() -> usize { 401 }\npub fn step_04_b() -> usize { 402 }"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("edit 1");
        assert!(!e1.is_error, "edit 1 failed: {}", e1.text_only());
        assert!(
            e1.text_only().contains("[lines after L4 shifted by +3]"),
            "Edit must report line shift so model uses current numbering, got: {}",
            e1.text_only()
        );

        // 3. Second Edit uses CURRENT line numbers L18..=L19 (where step_15..step_16 now live!) ->
        //    must hit directly WITHOUT mis-relocating to L21..L22!
        let e2 = edit
            .execute(
                json!({
                    "file_path": "app.rs",
                    "start_line": 18,
                    "end_line": 19,
                    "new_string": "pub fn step_15_updated() -> usize { 1500 }\npub fn step_16_updated() -> usize { 1600 }"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("edit 2");
        assert!(!e2.is_error, "edit 2 failed: {}", e2.text_only());
        assert!(
            !e2.text_only().contains("relocated"),
            "current line numbers after rebased Edit must apply directly, got: {}",
            e2.text_only()
        );
        let after_e2 = fs::read_to_string(&file).expect("read after e2");
        assert!(
            after_e2.contains("pub fn step_12() -> usize { 12 }\npub fn step_13() -> usize { 13 }\npub fn step_14() -> usize { 14 }\npub fn step_15_updated()"),
            "surrounding lines corrupted:\n{after_e2}"
        );
        assert!(
            !after_e2.contains("pub fn step_15() -> usize { 15 }"),
            "old step_15 should be replaced"
        );

        // 4. External file shift (e.g. Bash prepending 2 lines behind the model's back):
        //    line-range Edit on unchanged block at old L25..=L26 relocates cleanly to L27..=L28,
        //    while editing externally-modified lines L1..=L2 is rejected with diagnostic window!
        let ext_shifted = format!("// external header 1\n// external header 2\n{after_e2}");
        fs::write(&file, &ext_shifted).expect("external prepend");

        let e3_reloc = edit
            .execute(
                json!({
                    "file_path": "app.rs",
                    "start_line": 25,
                    "end_line": 26,
                    "new_string": "pub fn step_22_relocated() -> usize { 2200 }\npub fn step_23_relocated() -> usize { 2300 }"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("edit 3 reloc");
        assert!(
            !e3_reloc.is_error,
            "external shift relocation failed: {}",
            e3_reloc.text_only()
        );
        assert!(
            e3_reloc
                .text_only()
                .contains("relocated L25-L26 -> L27-L28"),
            "expected relocation note on external shift, got: {}",
            e3_reloc.text_only()
        );

        // External rewrite of step_01 (viewed L1..=L2 no longer exists on disk) MUST be rejected!
        let cur_disk = fs::read_to_string(&file).expect("read cur");
        fs::write(
            &file,
            cur_disk.replace("step_01() -> usize { 1 }", "step_01_ext() -> usize { 999 }"),
        )
        .expect("external rewrite");
        let e4_stale = edit
            .execute(
                json!({
                    "file_path": "app.rs",
                    "start_line": 1,
                    "end_line": 2,
                    "new_string": "corrupted"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("edit 4 stale");
        assert!(
            e4_stale.is_error
                && e4_stale
                    .text_only()
                    .contains("Stale line-range Edit rejected"),
            "stale edit on externally modified zone must be rejected, got: {}",
            e4_stale.text_only()
        );

        // 5. Re-read refreshes the model view with all surviving edits intact
        let r2 = read
            .execute(json!({"file_path": "app.rs"}), CancellationToken::new())
            .await
            .expect("reread");
        assert!(!r2.is_error, "reread failed: {}", r2.text_only());
        let reread_txt = r2.text_only();
        assert!(
            reread_txt.contains("step_15_updated")
                && reread_txt.contains("step_22_relocated")
                && reread_txt.contains("step_01_ext"),
            "expected all surviving edits on reread, got:\n{reread_txt}"
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn write_after_trailing_newline_external_change_does_not_loop_with_read_stub() {
        let tmp = std::env::temp_dir().join(format!("carnot-nl-loop-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).expect("mkdir");
        let file = tmp.join("cfg.txt");
        fs::write(&file, "a=1\nb=2\n").expect("write");

        let cfg = carnot_core::CarnotConfig::new(tmp.clone());
        let cas = Arc::new(CasStore::open(&tmp).expect("cas"));
        let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let read = reg.get("Read").expect("Read");
        let write = reg.get("Write").expect("Write");

        // 1. Model reads cfg.txt
        let r1 = read
            .execute(json!({"file_path": "cfg.txt"}), CancellationToken::new())
            .await
            .expect("r1");
        assert!(!r1.is_error);

        // 2. External formatter strips trailing newline (same lines(), different raw bytes!)
        fs::write(&file, "a=1\nb=2").expect("strip trailing nl");

        // 3. Write detects stale raw bytes and asks model to re-read
        let w_stale = write
            .execute(
                json!({"file_path": "cfg.txt", "content": "a=10\nb=20\n"}),
                CancellationToken::new(),
            )
            .await
            .expect("w_stale");
        assert!(w_stale.is_error, "stale write must refuse");

        // 4. Re-read MUST refresh disk_hash (not return stale UnchangedStub!) so next Write succeeds!
        let r2 = read
            .execute(json!({"file_path": "cfg.txt"}), CancellationToken::new())
            .await
            .expect("r2");
        assert!(!r2.is_error);
        assert!(
            !r2.text_only().contains("[unchanged:"),
            "re-read after byte-level disk change must not return unchanged stub: {}",
            r2.text_only()
        );
        let w_ok = write
            .execute(
                json!({"file_path": "cfg.txt", "content": "a=10\nb=20\n"}),
                CancellationToken::new(),
            )
            .await
            .expect("w_ok");
        assert!(
            !w_ok.is_error,
            "write after re-read must succeed: {}",
            w_ok.text_only()
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_edits_on_same_file_serialize_without_losing_updates() {
        let tmp = std::env::temp_dir().join(format!("carnot-conc-edit-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).expect("mkdir");
        let file = tmp.join("items.txt");
        let mut init = String::new();
        for i in 0..8 {
            use std::fmt::Write as _;
            let _ = writeln!(init, "item_{i}=old");
        }
        fs::write(&file, &init).expect("write");

        let cfg = carnot_core::CarnotConfig::new(tmp.clone());
        let cas = Arc::new(CasStore::open(&tmp).expect("cas"));
        let reg = Arc::new(crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas));

        let mut handles = Vec::new();
        for i in 0..8 {
            let edit = reg.get("Edit").expect("Edit");
            handles.push(tokio::spawn(async move {
                edit.execute(
                    json!({
                        "file_path": "items.txt",
                        "old_string": format!("item_{i}=old"),
                        "new_string": format!("item_{i}=new")
                    }),
                    CancellationToken::new(),
                )
                .await
                .expect("edit exec")
            }));
        }
        for h in handles {
            let out = h.await.expect("join");
            assert!(!out.is_error, "concurrent edit failed: {}", out.text_only());
        }

        let final_txt = fs::read_to_string(&file).expect("read final");
        for i in 0..8 {
            assert!(
                final_txt.contains(&format!("item_{i}=new")),
                "lost concurrent edit for item_{i}! disk:\n{final_txt}"
            );
        }

        let _ = fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn read_symbol_anchors_line_range_edit_and_outline_only_anchors_write() {
        let tmp = std::env::temp_dir().join(format!("carnot-sym-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).expect("mkdir");
        let file = tmp.join("lib.rs");
        fs::write(
            &file,
            "pub fn alpha() -> i32 {\n    1\n}\n\npub fn beta() -> i32 {\n    2\n}\n",
        )
        .expect("write");

        let cfg = carnot_core::CarnotConfig::new(tmp.clone());
        let cas = Arc::new(CasStore::open(&tmp).expect("cas"));
        let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let read = reg.get("Read").expect("Read");
        let edit = reg.get("Edit").expect("Edit");
        let write = reg.get("Write").expect("Write");

        // 1. Outline records disk_hash (unblocks Write) but NOT line view (blocks blind line-range Edit).
        let out = read
            .execute(
                json!({"file_path": "lib.rs", "outline": true}),
                CancellationToken::new(),
            )
            .await
            .expect("outline");
        assert!(!out.is_error, "outline failed: {}", out.text_only());
        let blind_after_outline = edit
            .execute(
                json!({"file_path": "lib.rs", "start_line": 5, "end_line": 7, "new_string": "pub fn beta() -> i32 { 20 }"}),
                CancellationToken::new(),
            )
            .await
            .expect("blind after outline");
        assert!(
            blind_after_outline.is_error,
            "outline must not authorize line-range Edit"
        );

        // 2. Symbol read populates view for `beta` only -> line-range Edit on `beta` succeeds, on `alpha` still fails!
        let sym = read
            .execute(
                json!({"file_path": "lib.rs", "symbol": "beta"}),
                CancellationToken::new(),
            )
            .await
            .expect("symbol read");
        assert!(!sym.is_error, "symbol read failed: {}", sym.text_only());
        let alpha_err = edit
            .execute(
                json!({"file_path": "lib.rs", "start_line": 1, "end_line": 3, "new_string": "pub fn alpha() -> i32 { 10 }"}),
                CancellationToken::new(),
            )
            .await
            .expect("alpha edit");
        assert!(
            alpha_err.is_error,
            "unread symbol `alpha` must still reject line-range Edit"
        );
        let beta_ok = edit
            .execute(
                json!({"file_path": "lib.rs", "start_line": 5, "end_line": 7, "new_string": "pub fn beta() -> i32 {\n    20\n}"}),
                CancellationToken::new(),
            )
            .await
            .expect("beta edit");
        assert!(
            !beta_ok.is_error,
            "beta edit failed: {}",
            beta_ok.text_only()
        );

        // 3. Write after Edit succeeds because Edit updated disk_hash.
        let w_ok = write
            .execute(
                json!({"file_path": "lib.rs", "content": "pub fn replaced() {}\n"}),
                CancellationToken::new(),
            )
            .await
            .expect("write after edit");
        assert!(
            !w_ok.is_error,
            "write after edit failed: {}",
            w_ok.text_only()
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn write_guard_blocks_unread_and_stale_external_overwrites() {
        let tmp = std::env::temp_dir().join(format!("carnot-wguard-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).expect("mkdir");
        let existing = tmp.join("existing.txt");
        fs::write(&existing, "v1\n").expect("write");
        let empty = tmp.join("empty.txt");
        fs::write(&empty, "").expect("write empty");

        let cfg = carnot_core::CarnotConfig::new(tmp.clone());
        let cas = Arc::new(CasStore::open(&tmp).expect("cas"));
        let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let read = reg.get("Read").expect("Read");
        let write = reg.get("Write").expect("Write");

        // 1. Creating a brand-new file succeeds without Read.
        let w_new = write
            .execute(
                json!({"file_path": "brand_new.txt", "content": "hello\n"}),
                CancellationToken::new(),
            )
            .await
            .expect("write new");
        assert!(
            !w_new.is_error,
            "new file write failed: {}",
            w_new.text_only()
        );

        // 2. Overwriting an existing unread file is refused.
        let w_unread = write
            .execute(
                json!({"file_path": "existing.txt", "content": "clobber\n"}),
                CancellationToken::new(),
            )
            .await
            .expect("write unread");
        assert!(w_unread.is_error, "unread overwrite must be rejected");
        assert!(
            w_unread.text_only().contains("has not been read"),
            "got: {}",
            w_unread.text_only()
        );
        assert_eq!(fs::read_to_string(&existing).expect("read"), "v1\n");

        // 3. Read -> Write -> Write succeeds.
        let r = read
            .execute(
                json!({"file_path": "existing.txt"}),
                CancellationToken::new(),
            )
            .await
            .expect("read");
        assert!(!r.is_error, "read failed: {}", r.text_only());
        let w1 = write
            .execute(
                json!({"file_path": "existing.txt", "content": "v2\n"}),
                CancellationToken::new(),
            )
            .await
            .expect("write 1");
        assert!(!w1.is_error, "write after read failed: {}", w1.text_only());
        let w2 = write
            .execute(
                json!({"file_path": "existing.txt", "content": "v3\n"}),
                CancellationToken::new(),
            )
            .await
            .expect("write 2");
        assert!(!w2.is_error, "second write failed: {}", w2.text_only());

        // 4. External disk mutation (simulating Bash / git checkout) causes Write to refuse.
        fs::write(&existing, "external_change\n").expect("external write");
        let w_stale = write
            .execute(
                json!({"file_path": "existing.txt", "content": "clobber_external\n"}),
                CancellationToken::new(),
            )
            .await
            .expect("write stale");
        assert!(w_stale.is_error, "stale overwrite must be rejected");
        assert!(
            w_stale.text_only().contains("changed on disk"),
            "got: {}",
            w_stale.text_only()
        );
        assert_eq!(
            fs::read_to_string(&existing).expect("read"),
            "external_change\n"
        );

        // 5. Reading an existing empty file records disk_hash and unblocks Write.
        let r_empty = read
            .execute(json!({"file_path": "empty.txt"}), CancellationToken::new())
            .await
            .expect("read empty");
        assert!(
            !r_empty.is_error,
            "read empty failed: {}",
            r_empty.text_only()
        );
        let w_empty = write
            .execute(
                json!({"file_path": "empty.txt", "content": "filled\n"}),
                CancellationToken::new(),
            )
            .await
            .expect("write empty");
        assert!(
            !w_empty.is_error,
            "write empty failed: {}",
            w_empty.text_only()
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[cfg(feature = "syntax")]
    #[tokio::test]
    async fn symbol_edit_supports_rust_python_batch_autoindent_ambiguity_and_guards() {
        let tmp = std::env::temp_dir().join(format!("carnot-sym-edit-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).expect("mkdir");

        let rs_file = tmp.join("calc.rs");
        fs::write(
            &rs_file,
            "pub struct Calc;\n\nimpl Calc {\n    pub fn add(&self, a: i32, b: i32) -> i32 {\n        a + b\n    }\n\n    pub fn sub(&self, a: i32, b: i32) -> i32 {\n        a - b\n    }\n}\n\npub struct Other;\n\nimpl Other {\n    pub fn add(&self, a: i32, b: i32) -> i32 {\n        a + b + 1\n    }\n}\n",
        )
        .expect("write calc.rs");

        let py_file = tmp.join("service.py");
        fs::write(
            &py_file,
            "class Service:\n    def greet(self, name: str) -> str:\n        return f\"hi {name}\"\n\n    def obsolete(self) -> None:\n        pass\n",
        )
        .expect("write service.py");

        let cfg = carnot_core::CarnotConfig::new(tmp.clone());
        let cas = Arc::new(CasStore::open(&tmp).expect("cas"));
        let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let read = reg.get("Read").expect("Read");
        let edit = reg.get("Edit").expect("Edit");

        // 1. Unread guard refuses blind symbol edit before Read.
        let unread = edit
            .execute(
                json!({
                    "file_path": "calc.rs",
                    "symbol": "Calc::add",
                    "new_string": "pub fn add(&self, a: i32, b: i32) -> i32 { a * b }"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("unread edit");
        assert!(unread.is_error, "unread symbol edit must be rejected");
        assert!(
            unread.text_only().contains("not been read"),
            "got: {}",
            unread.text_only()
        );

        // 2. Read(outline: true) anchors disk_hash and enables AST symbol edits!
        let outline = read
            .execute(
                json!({"file_path": "calc.rs", "outline": true}),
                CancellationToken::new(),
            )
            .await
            .expect("outline");
        assert!(!outline.is_error, "outline failed: {}", outline.text_only());

        // 3. Ambiguous unqualified symbol `add` is rejected with candidate line ranges.
        let ambig = edit
            .execute(
                json!({
                    "file_path": "calc.rs",
                    "symbol": "add",
                    "new_string": "pub fn add(&self, a: i32, b: i32) -> i32 { 0 }"
                }),
                CancellationToken::new(),
            )
            .await
            .expect("ambig edit");
        assert!(ambig.is_error, "ambiguous symbol must be rejected");
        assert!(
            ambig.text_only().contains("ambiguous"),
            "got: {}",
            ambig.text_only()
        );

        // 4. Atomic batch of 2 qualified Rust symbols (`Calc::add` + `Calc::sub`) with auto-indent!
        let batch = edit
            .execute(
                json!({
                    "file_path": "calc.rs",
                    "edits": [
                        {
                            "symbol": "Calc::add",
                            "new_string": "pub fn add(&self, a: i32, b: i32) -> i32 {\n    a.saturating_add(b)\n}"
                        },
                        {
                            "symbol": "Calc::sub",
                            "new_string": "pub fn sub(&self, a: i32, b: i32) -> i32 {\n    a.saturating_sub(b)\n}"
                        }
                    ]
                }),
                CancellationToken::new(),
            )
            .await
            .expect("batch edit");
        assert!(!batch.is_error, "batch symbol edit failed: {}", batch.text_only());
        assert!(batch.text_only().contains("[symbol `Calc::add`"));
        assert!(batch.text_only().contains("[symbol `Calc::sub`"));
        let rs_after = fs::read_to_string(&rs_file).expect("read calc.rs");
        assert!(
            rs_after.contains("    pub fn add(&self, a: i32, b: i32) -> i32 {\n        a.saturating_add(b)\n    }"),
            "expected preserved 4-space indentation:\n{rs_after}"
        );
        assert!(
            rs_after.contains("    pub fn sub(&self, a: i32, b: i32) -> i32 {\n        a.saturating_sub(b)\n    }"),
            "expected preserved 4-space indentation for sub:\n{rs_after}"
        );
        assert!(
            rs_after.contains("a + b + 1"),
            "Other::add must remain untouched:\n{rs_after}"
        );

        // 5. Python `Service.greet` replacement + `Service.obsolete` deletion (`new_string: ""`).
        let _ = read
            .execute(
                json!({"file_path": "service.py", "outline": true}),
                CancellationToken::new(),
            )
            .await
            .expect("py outline");
        let py_edit = edit
            .execute(
                json!({
                    "file_path": "service.py",
                    "edits": [
                        {
                            "symbol": "Service.greet",
                            "new_string": "def greet(self, name: str) -> str:\n    return f\"hello {name}\""
                        },
                        {
                            "symbol": "Service.obsolete",
                            "new_string": ""
                        }
                    ]
                }),
                CancellationToken::new(),
            )
            .await
            .expect("py edit");
        assert!(!py_edit.is_error, "py symbol edit failed: {}", py_edit.text_only());
        let py_after = fs::read_to_string(&py_file).expect("read py");
        assert!(
            py_after.contains("    def greet(self, name: str) -> str:\n        return f\"hello {name}\""),
            "expected preserved Python class indentation:\n{py_after}"
        );
        assert!(
            !py_after.contains("obsolete"),
            "deleted symbol must be gone:\n{py_after}"
        );

        let _ = fs::remove_dir_all(&tmp);
    }
}
