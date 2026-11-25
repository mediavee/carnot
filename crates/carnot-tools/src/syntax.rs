//! Tree-sitter Tier-0 silent syntax delta & structural code outliner
//! (Rust, Python, TypeScript/JS, Go).
//!
//! Controlled by the `syntax` Cargo feature (enabled by default). When compiled with
//! `--no-default-features`, falls back to zero-cost no-op stubs without linking C grammars.

#[cfg(feature = "syntax")]
mod imp {
    use std::path::Path;
    use tree_sitter::{Language, Node, Parser};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SourceLang {
        Rust,
        Python,
        TypeScript,
        Tsx,
        Go,
    }

    impl SourceLang {
        pub fn from_path(path: &Path) -> Option<Self> {
            let ext = path.extension()?.to_str()?.to_ascii_lowercase();
            match ext.as_str() {
                "rs" => Some(Self::Rust),
                "py" => Some(Self::Python),
                "ts" | "js" | "mjs" | "cjs" => Some(Self::TypeScript),
                "tsx" | "jsx" => Some(Self::Tsx),
                "go" => Some(Self::Go),
                _ => None,
            }
        }

        fn language(self) -> Language {
            match self {
                Self::Rust => tree_sitter_rust::LANGUAGE.into(),
                Self::Python => tree_sitter_python::LANGUAGE.into(),
                Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
                Self::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
                Self::Go => tree_sitter_go::LANGUAGE.into(),
            }
        }
    }

    /// Counts `ERROR` and `MISSING` syntax nodes and collects their 1-based line numbers.
    pub fn parse_error_lines(path: &Path, src: &str) -> Option<Vec<usize>> {
        let lang = SourceLang::from_path(path)?;
        let mut parser = Parser::new();
        parser.set_language(&lang.language()).ok()?;
        let tree = parser.parse(src, None)?;
        let mut lines = Vec::new();
        collect_errors(tree.root_node(), &mut lines);
        lines.sort_unstable();
        lines.dedup();
        Some(lines)
    }

    fn collect_errors(node: Node<'_>, out: &mut Vec<usize>) {
        if node.is_error() || node.is_missing() {
            out.push(node.start_position().row + 1);
        }
        if node.has_error() {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                collect_errors(child, out);
            }
        }
    }

    /// Tier-0 silent syntax delta: returns `Some(warning)` ONLY when `after` introduces MORE
    /// syntax error nodes than `before`. Includes up to 2 compact offending source lines so the
    /// agent can repair immediately without an extra `Read` round-trip. Returns `None` (0 bytes!)
    /// on clean edits.
    pub(super) fn tree_sitter_regression_warning(
        path: &Path,
        before: Option<&str>,
        after: &str,
    ) -> Option<String> {
        let after_errs = parse_error_lines(path, after)?;
        let before_count = before
            .and_then(|b| parse_error_lines(path, b))
            .map_or(0, |v| v.len());
        if after_errs.len() > before_count {
            let sample: Vec<String> = after_errs.iter().take(6).map(usize::to_string).collect();
            let lines: Vec<&str> = after.lines().collect();
            let mut snippets = String::new();
            for &ln in after_errs.iter().take(2) {
                if let Some(src_line) = lines.get(ln.saturating_sub(1)) {
                    let trimmed = src_line.trim();
                    let clipped: String = if trimmed.chars().count() > 110 {
                        let s: String = trimmed.chars().take(110).collect();
                        format!("{s}…")
                    } else {
                        trimmed.to_string()
                    };
                    if !clipped.is_empty() {
                        use std::fmt::Write as _;
                        let _ = write!(snippets, "\n  L{ln}: {clipped}");
                    }
                }
            }
            Some(format!(
                "[syntax-warning] {} now has {} parse error(s) (was {}) near line(s): {}{snippets}",
                path.display(),
                after_errs.len(),
                before_count,
                sample.join(", ")
            ))
        } else {
            None
        }
    }

    /// Generates a compact symbol+line-range outline for large source files (>300 lines),
    /// saving 70–85% input tokens on initial orientation.
    pub fn structural_outline(path: &Path, src: &str) -> Option<String> {
        let lang = SourceLang::from_path(path)?;
        let mut parser = Parser::new();
        parser.set_language(&lang.language()).ok()?;
        let tree = parser.parse(src, None)?;
        let lines: Vec<&str> = src.lines().collect();
        let mut items = Vec::new();
        collect_outline_items(tree.root_node(), lang, &lines, 0, &mut items);
        if items.is_empty() {
            return None;
        }
        let mut out = format!(
            "[outline: {} ({} lines) — pass offset & limit to Read exact ranges]\n",
            path.display(),
            lines.len()
        );
        for item in items {
            out.push_str(&item);
            out.push('\n');
        }
        Some(out)
    }

    fn collect_outline_items(
        node: Node<'_>,
        lang: SourceLang,
        lines: &[&str],
        depth: usize,
        out: &mut Vec<String>,
    ) {
        if depth > 3 || out.len() >= 250 {
            return;
        }
        let kind = node.kind();
        let is_sym = match lang {
            SourceLang::Rust => matches!(
                kind,
                "function_item"
                    | "struct_item"
                    | "enum_item"
                    | "trait_item"
                    | "impl_item"
                    | "mod_item"
            ),
            SourceLang::Python => matches!(kind, "function_definition" | "class_definition"),
            SourceLang::TypeScript | SourceLang::Tsx => matches!(
                kind,
                "function_declaration"
                    | "class_declaration"
                    | "interface_declaration"
                    | "type_alias_declaration"
                    | "method_definition"
            ),
            SourceLang::Go => matches!(
                kind,
                "function_declaration" | "method_declaration" | "type_declaration"
            ),
        };

        if is_sym {
            let start = node.start_position().row + 1;
            let end = node.end_position().row + 1;
            let sig = lines
                .get(start.saturating_sub(1))
                .map_or(kind, |s| s.trim());
            let indent = "  ".repeat(depth);
            out.push(format!("{indent}L{start}-{end}: {sig}"));
        }

        let next_depth = if is_sym { depth + 1 } else { depth };
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            collect_outline_items(child, lang, lines, next_depth, out);
        }
    }

    /// Looks up one or more 1-based `(start_line, end_line)` spans for a symbol query
    /// (`name`, `Type::method`, or `Class.method`) using tree-sitter.
    pub fn find_symbol_ranges(path: &Path, src: &str, query: &str) -> Vec<(usize, usize)> {
        let Some(lang) = SourceLang::from_path(path) else {
            return Vec::new();
        };
        let mut parser = Parser::new();
        if parser.set_language(&lang.language()).is_err() {
            return Vec::new();
        }
        let Some(tree) = parser.parse(src, None) else {
            return Vec::new();
        };
        let (want_parent, want_leaf) = if let Some((p, m)) = query.rsplit_once("::") {
            let parent = p.rsplit("::").next().unwrap_or(p).trim();
            (Some(parent), m.trim())
        } else if let Some((p, m)) = query.rsplit_once('.') {
            let parent = p.rsplit('.').next().unwrap_or(p).trim();
            (Some(parent), m.trim())
        } else {
            (None, query.trim())
        };
        if want_leaf.is_empty() {
            return Vec::new();
        }
        let mut matches = Vec::new();
        collect_symbol_matches(
            tree.root_node(),
            lang,
            src.as_bytes(),
            None,
            want_parent,
            want_leaf,
            &mut matches,
        );
        matches
    }

    fn node_symbol_name<'a>(node: Node<'a>, src: &'a [u8]) -> Option<&'a str> {
        if let Some(n) = node.child_by_field_name("name") {
            return n.utf8_text(src).ok();
        }
        if node.kind() == "impl_item"
            && let Some(t) = node.child_by_field_name("type")
        {
            return t.utf8_text(src).ok();
        }
        None
    }

    fn collect_symbol_matches(
        node: Node<'_>,
        lang: SourceLang,
        src: &[u8],
        parent_name: Option<&str>,
        want_parent: Option<&str>,
        want_leaf: &str,
        out: &mut Vec<(usize, usize)>,
    ) {
        let kind = node.kind();
        let is_sym = match lang {
            SourceLang::Rust => matches!(
                kind,
                "function_item"
                    | "struct_item"
                    | "enum_item"
                    | "trait_item"
                    | "impl_item"
                    | "mod_item"
            ),
            SourceLang::Python => matches!(kind, "function_definition" | "class_definition"),
            SourceLang::TypeScript | SourceLang::Tsx => matches!(
                kind,
                "function_declaration"
                    | "class_declaration"
                    | "interface_declaration"
                    | "type_alias_declaration"
                    | "method_definition"
            ),
            SourceLang::Go => matches!(
                kind,
                "function_declaration" | "method_declaration" | "type_declaration"
            ),
        };

        let own_name = if is_sym {
            node_symbol_name(node, src)
        } else {
            None
        };

        if let Some(name) = own_name {
            let leaf_ok = name == want_leaf;
            let parent_ok = match want_parent {
                Some(wp) => {
                    parent_name.is_some_and(|pn| pn == wp || pn.starts_with(&format!("{wp}<")))
                }
                None => true,
            };
            if leaf_ok && parent_ok {
                out.push((node.start_position().row + 1, node.end_position().row + 1));
            }
        }

        let next_parent = if is_sym && own_name.is_some() {
            own_name
        } else {
            parent_name
        };
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            collect_symbol_matches(child, lang, src, next_parent, want_parent, want_leaf, out);
        }
    }
}

#[cfg(feature = "syntax")]
pub use imp::{SourceLang, find_symbol_ranges, parse_error_lines, structural_outline};

/// Tier-0 silent structural validation for `.json` and `.toml` config files (zero C deps).
/// Returns `Some(warning)` only when `after` fails to parse and `before` was valid (or new file).
fn structured_data_regression_warning(
    path: &std::path::Path,
    before: Option<&str>,
    after: &str,
) -> Option<String> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "json" => {
            if let Err(err) = serde_json::from_str::<serde_json::Value>(after) {
                let was_broken = before.is_some_and(|b| {
                    serde_json::from_str::<serde_json::Value>(b).is_err()
                });
                if !was_broken {
                    let line = err.line();
                    let snippet = after
                        .lines()
                        .nth(line.saturating_sub(1))
                        .map(|l| format!("\n  L{line}: {}", l.trim()))
                        .unwrap_or_default();
                    return Some(format!(
                        "[syntax-warning] {} invalid JSON at line {}, col {}: {err}{snippet}",
                        path.display(),
                        line,
                        err.column()
                    ));
                }
            }
            None
        }
        "toml" => {
            if let Err(err) = toml::from_str::<toml::Value>(after) {
                let was_broken = before.is_some_and(|b| toml::from_str::<toml::Value>(b).is_err());
                if !was_broken {
                    return Some(format!(
                        "[syntax-warning] {} invalid TOML: {}",
                        path.display(),
                        err.message()
                    ));
                }
            }
            None
        }
        _ => None,
    }
}

/// Tier-0 silent syntax delta across Rust/Python/TypeScript/Go (via tree-sitter when `syntax` is
/// enabled) AND `.json`/`.toml` config files. Returns `None` (0 bytes!) on clean edits.
pub fn syntax_regression_warning(
    path: &std::path::Path,
    before: Option<&str>,
    after: &str,
) -> Option<String> {
    if let Some(w) = structured_data_regression_warning(path, before, after) {
        return Some(w);
    }
    #[cfg(feature = "syntax")]
    {
        imp::tree_sitter_regression_warning(path, before, after)
    }
    #[cfg(not(feature = "syntax"))]
    {
        None
    }
}

#[cfg(not(feature = "syntax"))]
pub fn parse_error_lines(_path: &std::path::Path, _src: &str) -> Option<Vec<usize>> {
    None
}

#[cfg(not(feature = "syntax"))]
pub fn structural_outline(_path: &std::path::Path, _src: &str) -> Option<String> {
    None
}

#[cfg(not(feature = "syntax"))]
pub fn find_symbol_ranges(
    _path: &std::path::Path,
    _src: &str,
    _symbol: &str,
) -> Vec<(usize, usize)> {
    Vec::new()
}

#[cfg(all(test, feature = "syntax"))]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn tier0_silent_on_valid_edit_warns_on_broken_rust_with_snippet() {
        let p = Path::new("lib.rs");
        let ok_before = "pub fn add(a: i32, b: i32) -> i32 { a + b }\n";
        let ok_after = "pub fn add(a: i32, b: i32) -> i32 { a + b + 1 }\n";
        assert!(
            syntax_regression_warning(p, Some(ok_before), ok_after).is_none(),
            "valid Rust must produce zero warning"
        );

        let broken = "pub fn add(a: i32, b: i32) -> i32 { a + \n";
        let warn = syntax_regression_warning(p, Some(ok_before), broken).expect("warning");
        assert!(warn.contains("L1:"), "expected offending line snippet: {warn}");
        assert!(warn.contains("pub fn add"), "expected source snippet: {warn}");
    }

    #[test]
    fn tier0_validates_json_and_toml_regressions_silently() {
        // JSON
        let jp = Path::new("package.json");
        assert!(
            syntax_regression_warning(jp, Some("{\"a\":1}"), "{\"a\":2}").is_none(),
            "valid JSON must stay silent"
        );
        let jwarn = syntax_regression_warning(jp, Some("{\"a\":1}"), "{\n  \"a\": 1,\n}")
            .expect("broken JSON warning");
        assert!(jwarn.contains("invalid JSON"), "got: {jwarn}");
        assert!(jwarn.contains("line 3"), "expected exact line: {jwarn}");

        // TOML
        let tp = Path::new("Cargo.toml");
        assert!(
            syntax_regression_warning(tp, Some("[pkg]\nname = \"a\"\n"), "[pkg]\nname = \"b\"\n")
                .is_none(),
            "valid TOML must stay silent"
        );
        let twarn =
            syntax_regression_warning(tp, Some("[pkg]\nname = \"a\"\n"), "[pkg\nname = \"b\"\n")
                .expect("broken TOML warning");
        assert!(twarn.contains("invalid TOML"), "got: {twarn}");
    }

    #[test]
    fn generates_rust_structural_outline() {
        let src = "pub struct Engine {}\nimpl Engine {\n    pub fn run(&self) {}\n}\n";
        let outline = structural_outline(Path::new("engine.rs"), src).expect("outline");
        assert!(outline.contains("pub struct Engine"), "got: {outline}");
        assert!(outline.contains("pub fn run"), "got: {outline}");

        let ranges = find_symbol_ranges(Path::new("engine.rs"), src, "Engine::run");
        assert_eq!(ranges, vec![(3, 3)], "symbol range");
    }
}
