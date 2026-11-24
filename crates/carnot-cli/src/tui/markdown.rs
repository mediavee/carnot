//! Fast, zero-dependency streaming Markdown renderer -> `Vec<ratatui::text::Line<'static>>`.
//!
//! Handles:
//!   - Headings (`#`..`####`), horizontal rules (`---`)
//!   - Inline spans: `**bold**`, `*italic*`, `` `code` ``, `~~strike~~`, `[text](url)`
//!   - Bullet & numbered lists with proper hanging-indent wrapping
//!   - Blockquotes (`> `)
//!   - Fenced code blocks (` ```lang `) including unclosed streaming blocks + lightweight syntax coloring
//!   - Pipe tables (`| a | b |`) with auto-aligned box borders
//!   - Unicode display-width aware word wrapping

use super::theme::Palette;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthChar;

/// Render Markdown text into wrapped `Line<'static>`s fitting `width` columns.
pub(crate) fn render_markdown(input: &str, width: usize, p: &Palette) -> Vec<Line<'static>> {
    let width = width.max(16);
    let mut out: Vec<Line<'static>> = Vec::new();
    let raw_lines: Vec<&str> = input.lines().collect();
    let mut i = 0;

    while i < raw_lines.len() {
        let line = raw_lines[i];
        let trimmed = line.trim();

        // Blank line -> collapse consecutive empty lines
        if trimmed.is_empty() {
            if !out.is_empty() && !out.last().is_some_and(|l| l.spans.is_empty()) {
                out.push(Line::default());
            }
            i += 1;
            continue;
        }

        // Fenced code block (``` or ~~~) — works even if still streaming (unclosed)
        if let Some((fence_char, fence_len, lang)) = parse_fence_open(trimmed) {
            i += 1;
            let mut code_lines: Vec<&str> = Vec::new();
            let mut closed = false;
            while i < raw_lines.len() {
                let candidate = raw_lines[i].trim();
                if is_fence_close(candidate, fence_char, fence_len) {
                    closed = true;
                    i += 1;
                    break;
                }
                code_lines.push(raw_lines[i]);
                i += 1;
            }
            render_code_block(&code_lines, lang, closed, width, p, &mut out);
            continue;
        }

        // Horizontal rule
        if is_horizontal_rule(trimmed) {
            let rule_w = width.saturating_sub(2).min(64);
            out.push(Line::from(Span::styled("─".repeat(rule_w), p.border())));
            i += 1;
            continue;
        }

        // ATX Heading (# .. ####)
        if let Some((level, heading_text)) = parse_heading(trimmed) {
            if !out.is_empty() && !out.last().is_some_and(|l| l.spans.is_empty()) {
                out.push(Line::default());
            }
            let base = match level {
                1 => p.brand_bold().add_modifier(Modifier::UNDERLINED),
                2 => p.brand_bold(),
                3 => p.accent_bold(),
                _ => p.text().add_modifier(Modifier::BOLD),
            };
            let spans = parse_inline(heading_text, base, p);
            wrap_spans(&spans, width, vec![], vec![], &mut out);
            i += 1;
            continue;
        }

        // Markdown pipe table (header row + separator row)
        if trimmed.contains('|')
            && i + 1 < raw_lines.len()
            && is_table_separator(raw_lines[i + 1].trim())
        {
            let mut table_rows: Vec<Vec<String>> = Vec::new();
            table_rows.push(split_table_row(trimmed));
            i += 2; // skip header + separator
            while i < raw_lines.len() {
                let r = raw_lines[i].trim();
                if r.is_empty() || !r.contains('|') {
                    break;
                }
                table_rows.push(split_table_row(r));
                i += 1;
            }
            render_table(&table_rows, width, p, &mut out);
            continue;
        }

        // Blockquote (`> `)
        if let Some(rest) = trimmed.strip_prefix('>') {
            let content = rest.strip_prefix(' ').unwrap_or(rest);
            let spans = parse_inline(content, p.muted().add_modifier(Modifier::ITALIC), p);
            let prefix = vec![Span::styled("│ ", p.border())];
            wrap_spans(&spans, width, prefix.clone(), prefix, &mut out);
            i += 1;
            continue;
        }

        // Unordered list (`- `, `* `, `+ `)
        if let Some((indent_cols, body)) = parse_unordered_item(line) {
            let pad = " ".repeat(indent_cols.min(8));
            let bullet = if indent_cols == 0 { "• " } else { "◦ " };
            let first_prefix = vec![Span::raw(pad.clone()), Span::styled(bullet, p.brand_bold())];
            let cont_prefix = vec![Span::raw(format!("{pad}  "))];
            let spans = parse_inline(body, p.text(), p);
            wrap_spans(&spans, width, first_prefix, cont_prefix, &mut out);
            i += 1;
            continue;
        }

        // Ordered list (`1. `, `2) `)
        if let Some((indent_cols, num, body)) = parse_ordered_item(line) {
            let pad = " ".repeat(indent_cols.min(8));
            let marker = format!("{num}. ");
            let cont_spaces = " ".repeat(marker.len());
            let first_prefix = vec![Span::raw(pad.clone()), Span::styled(marker, p.brand_bold())];
            let cont_prefix = vec![Span::raw(format!("{pad}{cont_spaces}"))];
            let spans = parse_inline(body, p.text(), p);
            wrap_spans(&spans, width, first_prefix, cont_prefix, &mut out);
            i += 1;
            continue;
        }

        // Regular paragraph line
        let spans = parse_inline(line, p.text(), p);
        wrap_spans(&spans, width, vec![], vec![], &mut out);
        i += 1;
    }

    // Trim trailing empty line
    while out.last().is_some_and(|l| l.spans.is_empty()) {
        out.pop();
    }
    out
}

fn parse_fence_open(trimmed: &str) -> Option<(char, usize, &str)> {
    let ch = trimmed.chars().next()?;
    if ch != '`' && ch != '~' {
        return None;
    }
    let count = trimmed.chars().take_while(|&c| c == ch).count();
    if count < 3 {
        return None;
    }
    let rest = trimmed[count..].trim();
    if ch == '`' && rest.contains('`') {
        return None;
    }
    let lang = rest.split_whitespace().next().unwrap_or("");
    Some((ch, count, lang))
}

fn is_fence_close(trimmed: &str, ch: char, min_len: usize) -> bool {
    let count = trimmed.chars().take_while(|&c| c == ch).count();
    count >= min_len && trimmed[count..].trim().is_empty()
}

fn is_horizontal_rule(trimmed: &str) -> bool {
    if trimmed.len() < 3 {
        return false;
    }
    let chars: Vec<char> = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
    chars.len() >= 3 && matches!(chars[0], '-' | '*' | '_') && chars.iter().all(|&c| c == chars[0])
}

fn parse_heading(trimmed: &str) -> Option<(usize, &str)> {
    let hashes = trimmed.bytes().take_while(|&b| b == b'#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = trimmed[hashes..].strip_prefix(' ')?;
    Some((hashes, rest.trim()))
}

fn parse_unordered_item(line: &str) -> Option<(usize, &str)> {
    let leading = line.bytes().take_while(|&b| b == b' ').count();
    let rest = &line[leading..];
    let after = rest
        .strip_prefix("- ")
        .or_else(|| rest.strip_prefix("* "))
        .or_else(|| rest.strip_prefix("+ "))?;
    Some((leading, after))
}

fn parse_ordered_item(line: &str) -> Option<(usize, &str, &str)> {
    let leading = line.bytes().take_while(|&b| b == b' ').count();
    let rest = &line[leading..];
    let digits = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
    if digits == 0 || digits > 4 {
        return None;
    }
    let after_digits = &rest[digits..];
    let body = after_digits
        .strip_prefix(". ")
        .or_else(|| after_digits.strip_prefix(") "))?;
    Some((leading, &rest[..digits], body))
}

fn is_table_separator(trimmed: &str) -> bool {
    if !trimmed.contains('-') || !trimmed.contains('|') {
        return false;
    }
    trimmed
        .chars()
        .all(|c| matches!(c, '|' | '-' | ':' | ' ' | '\t'))
}

fn split_table_row(trimmed: &str) -> Vec<String> {
    let stripped = trimmed
        .strip_prefix('|')
        .unwrap_or(trimmed)
        .strip_suffix('|')
        .unwrap_or(trimmed.strip_prefix('|').unwrap_or(trimmed));
    stripped
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect()
}

fn render_table(rows: &[Vec<String>], width: usize, p: &Palette, out: &mut Vec<Line<'static>>) {
    let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    if cols == 0 {
        return;
    }
    let mut col_w = vec![1usize; cols];
    for row in rows {
        for (idx, cell) in row.iter().enumerate() {
            let w = display_width(cell);
            if w > col_w[idx] {
                col_w[idx] = w;
            }
        }
    }
    let overhead = cols * 3 + 1;
    let avail = width.saturating_sub(overhead).max(cols * 3);
    let total: usize = col_w.iter().sum();
    if total > avail {
        for w in &mut col_w {
            *w = ((*w * avail) / total.max(1)).max(3);
        }
    }

    let make_border = |left: &str, mid: &str, right: &str| -> Line<'static> {
        let mut s = String::from(left);
        for (idx, &w) in col_w.iter().enumerate() {
            if idx > 0 {
                s.push_str(mid);
            }
            s.push_str(&"─".repeat(w + 2));
        }
        s.push_str(right);
        Line::from(Span::styled(s, p.border()))
    };

    out.push(make_border("┌", "┬", "┐"));
    for (r_idx, row) in rows.iter().enumerate() {
        let mut spans: Vec<Span<'static>> = Vec::with_capacity(cols * 2 + 1);
        spans.push(Span::styled("│", p.border()));
        for (c_idx, &w) in col_w.iter().enumerate() {
            let raw = row.get(c_idx).map(String::as_str).unwrap_or("");
            let clipped = truncate_display(raw, w);
            let pad = w.saturating_sub(display_width(&clipped));
            let cell_style = if r_idx == 0 { p.brand_bold() } else { p.text() };
            spans.push(Span::styled(
                format!(" {clipped}{} ", " ".repeat(pad)),
                cell_style,
            ));
            spans.push(Span::styled("│", p.border()));
        }
        out.push(Line::from(spans));
        if r_idx == 0 && rows.len() > 1 {
            out.push(make_border("├", "┼", "┤"));
        }
    }
    out.push(make_border("└", "┴", "┘"));
}

fn render_code_block(
    code_lines: &[&str],
    lang: &str,
    closed: bool,
    width: usize,
    p: &Palette,
    out: &mut Vec<Line<'static>>,
) {
    let box_w = width.saturating_sub(2).max(16);
    let header_label = if lang.is_empty() {
        if closed {
            " code ".to_string()
        } else {
            " code … ".to_string()
        }
    } else if closed {
        format!(" {lang} ")
    } else {
        format!(" {lang} … ")
    };
    let fill = box_w.saturating_sub(display_width(&header_label) + 2);
    out.push(Line::from(vec![
        Span::styled("╭─", p.border()),
        Span::styled(header_label, p.muted()),
        Span::styled("─".repeat(fill), p.border()),
    ]));

    let inner_w = width.saturating_sub(4).max(12);
    let bg_style = if p.color {
        Style::default().bg(p.surface)
    } else {
        Style::default()
    };

    for &raw in code_lines {
        let expanded = raw.replace('\t', "    ");
        let highlighted = highlight_code_line(&expanded, lang, p, bg_style);
        wrap_code_spans(&highlighted, inner_w, p, bg_style, out);
    }

    if closed {
        out.push(Line::from(Span::styled(
            format!("╰{}", "─".repeat(box_w.saturating_sub(1))),
            p.border(),
        )));
    }
}

fn wrap_code_spans(
    spans: &[Span<'static>],
    inner_w: usize,
    p: &Palette,
    bg_style: Style,
    out: &mut Vec<Line<'static>>,
) {
    if spans.is_empty() {
        out.push(Line::from(vec![
            Span::styled("│ ", p.border()),
            Span::styled(" ".repeat(inner_w), bg_style),
        ]));
        return;
    }

    let mut cur: Vec<Span<'static>> = vec![Span::styled("│ ", p.border())];
    let mut cur_w = 0usize;

    for sp in spans {
        let mut buf = String::new();
        for ch in sp.content.chars() {
            let cw = ch.width().unwrap_or(1).max(1);
            if cur_w + cw > inner_w && cur_w > 0 {
                if !buf.is_empty() {
                    cur.push(Span::styled(std::mem::take(&mut buf), sp.style));
                }
                if cur_w < inner_w {
                    cur.push(Span::styled(" ".repeat(inner_w - cur_w), bg_style));
                }
                out.push(Line::from(cur));
                cur = vec![Span::styled("│ ", p.border())];
                cur_w = 0;
            }
            buf.push(ch);
            cur_w += cw;
        }
        if !buf.is_empty() {
            cur.push(Span::styled(buf, sp.style));
        }
    }

    if cur_w < inner_w {
        cur.push(Span::styled(" ".repeat(inner_w - cur_w), bg_style));
    }
    out.push(Line::from(cur));
}

/// Lightweight, fast tokenizer-based syntax highlighting (zero external crates).
fn highlight_code_line(line: &str, lang: &str, p: &Palette, bg: Style) -> Vec<Span<'static>> {
    if !p.color {
        return vec![Span::styled(line.to_string(), bg)];
    }

    let lang_lc = lang.to_ascii_lowercase();
    let is_rust = matches!(lang_lc.as_str(), "rs" | "rust");
    let is_hash_comment = matches!(
        lang_lc.as_str(),
        "py" | "python" | "sh" | "bash" | "zsh" | "shell" | "toml" | "yaml" | "yml" | "ruby" | "rb"
    );
    let is_slash_comment = matches!(
        lang_lc.as_str(),
        "rs" | "rust"
            | "ts"
            | "tsx"
            | "js"
            | "jsx"
            | "go"
            | "c"
            | "cpp"
            | "java"
            | "swift"
            | "kt"
            | "jsonc"
            | ""
    );
    let is_diff = matches!(lang_lc.as_str(), "diff" | "patch");

    if is_diff {
        let style = if line.starts_with('+') && !line.starts_with("+++") {
            bg.fg(p.success)
        } else if line.starts_with('-') && !line.starts_with("---") {
            bg.fg(p.danger)
        } else if line.starts_with("@@") {
            bg.fg(p.accent)
        } else {
            bg.fg(p.fg)
        };
        return vec![Span::styled(line.to_string(), style)];
    }

    let mut spans: Vec<Span<'static>> = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        // Line comments
        if (is_slash_comment && c == '/' && i + 1 < chars.len() && chars[i + 1] == '/')
            || (is_hash_comment && c == '#')
        {
            let rest: String = chars[i..].iter().collect();
            spans.push(Span::styled(
                rest,
                bg.fg(p.syn_comment).add_modifier(Modifier::ITALIC),
            ));
            break;
        }

        // Rust lifetimes (`'a`, `'static`) vs char literals (`'x'`, `'\n'`)
        if is_rust && c == '\'' {
            let is_char_lit = (i + 2 < chars.len() && chars[i + 2] == '\'')
                || (i + 3 < chars.len() && chars[i + 1] == '\\' && chars[i + 3] == '\'');
            if !is_char_lit {
                let start = i;
                i += 1;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                spans.push(Span::styled(s, bg.fg(p.syn_type)));
                continue;
            }
        }

        // Strings ("..." or '...' or `...`)
        if c == '"' || c == '\'' || c == '`' {
            let quote = c;
            let start = i;
            i += 1;
            let mut escaped = false;
            while i < chars.len() {
                let ch = chars[i];
                i += 1;
                if escaped {
                    escaped = false;
                    continue;
                }
                if ch == '\\' {
                    escaped = true;
                    continue;
                }
                if ch == quote {
                    break;
                }
            }
            let s: String = chars[start..i].iter().collect();
            spans.push(Span::styled(s, bg.fg(p.syn_string)));
            continue;
        }

        // Numbers
        if c.is_ascii_digit()
            && (i == 0 || !chars[i - 1].is_ascii_alphanumeric() && chars[i - 1] != '_')
        {
            let start = i;
            while i < chars.len()
                && (chars[i].is_ascii_alphanumeric() || chars[i] == '.' || chars[i] == '_')
            {
                i += 1;
            }
            let s: String = chars[start..i].iter().collect();
            spans.push(Span::styled(s, bg.fg(p.syn_number)));
            continue;
        }

        // Identifiers / keywords / PascalCase types
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let style = if is_keyword(&word) {
                bg.fg(p.syn_keyword).add_modifier(Modifier::BOLD)
            } else if word
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_uppercase())
                && word.chars().any(|ch| ch.is_ascii_lowercase())
            {
                bg.fg(p.syn_type)
            } else {
                bg.fg(p.fg)
            };
            spans.push(Span::styled(word, style));
            continue;
        }

        // Punctuation / whitespace batch
        let start = i;
        i += 1;
        while i < chars.len() {
            let ch = chars[i];
            if ch.is_ascii_alphanumeric()
                || ch == '_'
                || ch == '"'
                || ch == '\''
                || ch == '`'
                || (is_hash_comment && ch == '#')
                || (is_slash_comment && ch == '/' && i + 1 < chars.len() && chars[i + 1] == '/')
            {
                break;
            }
            i += 1;
        }
        let s: String = chars[start..i].iter().collect();
        spans.push(Span::styled(s, bg.fg(p.fg)));
    }

    spans
}

fn is_keyword(word: &str) -> bool {
    matches!(
        word,
        "fn" | "pub"
            | "let"
            | "mut"
            | "const"
            | "static"
            | "struct"
            | "enum"
            | "trait"
            | "impl"
            | "for"
            | "while"
            | "loop"
            | "if"
            | "else"
            | "match"
            | "return"
            | "async"
            | "await"
            | "use"
            | "mod"
            | "crate"
            | "self"
            | "Self"
            | "where"
            | "type"
            | "true"
            | "false"
            | "None"
            | "Some"
            | "Ok"
            | "Err"
            | "def"
            | "class"
            | "import"
            | "from"
            | "export"
            | "default"
            | "function"
            | "interface"
            | "extends"
            | "new"
            | "try"
            | "catch"
            | "throw"
            | "switch"
            | "case"
            | "break"
            | "continue"
            | "var"
            | "func"
            | "package"
            | "defer"
            | "go"
            | "chan"
            | "select"
            | "null"
            | "nil"
            | "in"
            | "as"
            | "is"
            | "not"
            | "and"
            | "or"
            | "with"
            | "yield"
            | "lambda"
    )
}

/// Parse inline Markdown (`**bold**`, `*italic*`, `` `code` ``, `~~strike~~`, `[label](url)`).
fn parse_inline(text: &str, base: Style, p: &Palette) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let bytes = text.as_bytes();
    let mut i = 0;

    let flush = |buf: &mut String, spans: &mut Vec<Span<'static>>, style: Style| {
        if !buf.is_empty() {
            spans.push(Span::styled(std::mem::take(buf), style));
        }
    };

    while i < bytes.len() {
        // Inline code: `...`
        if bytes[i] == b'`'
            && let Some(end_rel) = text[i + 1..].find('`')
        {
            flush(&mut buf, &mut spans, base);
            let inner = &text[i + 1..i + 1 + end_rel];
            spans.push(Span::styled(format!(" {inner} "), p.code_inline()));
            i += end_rel + 2;
            continue;
        }

        // Bold: **...** or __...__
        if (text[i..].starts_with("**") || text[i..].starts_with("__")) && i + 2 < bytes.len() {
            let delim = &text[i..i + 2];
            if let Some(end_rel) = text[i + 2..].find(delim)
                && end_rel > 0
            {
                flush(&mut buf, &mut spans, base);
                let inner = &text[i + 2..i + 2 + end_rel];
                spans.extend(parse_inline(inner, base.add_modifier(Modifier::BOLD), p));
                i += end_rel + 4;
                continue;
            }
        }

        // Strikethrough: ~~...~~
        if text[i..].starts_with("~~")
            && i + 2 < bytes.len()
            && let Some(end_rel) = text[i + 2..].find("~~")
            && end_rel > 0
        {
            flush(&mut buf, &mut spans, base);
            let inner = &text[i + 2..i + 2 + end_rel];
            spans.push(Span::styled(
                inner.to_string(),
                base.add_modifier(Modifier::CROSSED_OUT),
            ));
            i += end_rel + 4;
            continue;
        }

        // Link: [label](url)
        if bytes[i] == b'['
            && let Some(close_bracket) = text[i + 1..].find(']')
        {
            let after_bracket = i + 1 + close_bracket + 1;
            if text[after_bracket..].starts_with('(')
                && let Some(close_paren) = text[after_bracket + 1..].find(')')
            {
                flush(&mut buf, &mut spans, base);
                let label = &text[i + 1..i + 1 + close_bracket];
                let url = &text[after_bracket + 1..after_bracket + 1 + close_paren];
                spans.push(Span::styled(
                    label.to_string(),
                    p.accent().add_modifier(Modifier::UNDERLINED),
                ));
                if !url.is_empty() && url != label {
                    spans.push(Span::styled(format!(" ({url})"), p.muted()));
                }
                i = after_bracket + 1 + close_paren + 1;
                continue;
            }
        }

        // Italic: *...*
        if bytes[i] == b'*'
            && !text[i..].starts_with("**")
            && i + 1 < bytes.len()
            && !text[i + 1..].starts_with(' ')
            && let Some(end_rel) = text[i + 1..].find('*')
            && end_rel > 0
        {
            flush(&mut buf, &mut spans, base);
            let inner = &text[i + 1..i + 1 + end_rel];
            spans.push(Span::styled(
                inner.to_string(),
                base.add_modifier(Modifier::ITALIC),
            ));
            i += end_rel + 2;
            continue;
        }

        // Regular UTF-8 char
        if let Some(ch) = text[i..].chars().next() {
            buf.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }

    flush(&mut buf, &mut spans, base);
    spans
}

/// Word-wrap styled spans into `out`, respecting Unicode display widths and hanging prefixes.
pub(crate) fn wrap_spans(
    spans: &[Span<'static>],
    width: usize,
    first_prefix: Vec<Span<'static>>,
    cont_prefix: Vec<Span<'static>>,
    out: &mut Vec<Line<'static>>,
) {
    let width = width.max(8);
    let first_pw: usize = first_prefix.iter().map(|s| display_width(&s.content)).sum();
    let cont_pw: usize = cont_prefix.iter().map(|s| display_width(&s.content)).sum();

    let mut cur_spans = first_prefix;
    let mut cur_w = first_pw;
    let mut has_content = false;

    for span in spans {
        for tok in split_tokens(&span.content) {
            let tok_w = display_width(tok);
            let is_ws = tok.chars().all(char::is_whitespace);

            if is_ws && !has_content {
                continue;
            }

            if cur_w + tok_w <= width {
                push_merged(&mut cur_spans, tok, span.style);
                cur_w += tok_w;
                if !is_ws {
                    has_content = true;
                }
                continue;
            }

            if has_content {
                out.push(Line::from(cur_spans));
                cur_spans = cont_prefix.clone();
                cur_w = cont_pw;
                has_content = false;
                if is_ws {
                    continue;
                }
            }

            if cur_w + tok_w > width {
                let mut piece = String::new();
                for ch in tok.chars() {
                    let cw = ch.width().unwrap_or(1).max(1);
                    if cur_w + cw > width && (cur_w > cont_pw || !piece.is_empty()) {
                        if !piece.is_empty() {
                            push_merged(&mut cur_spans, &piece, span.style);
                            piece.clear();
                        }
                        out.push(Line::from(cur_spans));
                        cur_spans = cont_prefix.clone();
                        cur_w = cont_pw;
                    }
                    piece.push(ch);
                    cur_w += cw;
                }
                if !piece.is_empty() {
                    push_merged(&mut cur_spans, &piece, span.style);
                    has_content = true;
                }
            } else {
                push_merged(&mut cur_spans, tok, span.style);
                cur_w += tok_w;
                has_content = true;
            }
        }
    }

    if has_content || cur_spans.len() > cont_prefix.len() || out.is_empty() {
        out.push(Line::from(cur_spans));
    }
}

fn push_merged(target: &mut Vec<Span<'static>>, text: &str, style: Style) {
    if let Some(last) = target.last_mut()
        && last.style == style
    {
        last.content.to_mut().push_str(text);
        return;
    }
    target.push(Span::styled(text.to_string(), style));
}

fn split_tokens(s: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut start = 0;
    let mut in_ws = None;
    for (idx, ch) in s.char_indices() {
        let ws = ch.is_whitespace();
        match in_ws {
            None => in_ws = Some(ws),
            Some(prev) if prev != ws => {
                tokens.push(&s[start..idx]);
                start = idx;
                in_ws = Some(ws);
            }
            _ => {}
        }
    }
    if start < s.len() {
        tokens.push(&s[start..]);
    }
    tokens
}

pub(crate) fn display_width(s: &str) -> usize {
    s.chars().map(|c| c.width().unwrap_or(0)).sum()
}

pub(crate) fn truncate_display(s: &str, max_w: usize) -> String {
    if display_width(s) <= max_w {
        return s.to_string();
    }
    let mut out = String::new();
    let mut w = 0;
    let limit = max_w.saturating_sub(1);
    for ch in s.chars() {
        let cw = ch.width().unwrap_or(0);
        if w + cw > limit {
            break;
        }
        out.push(ch);
        w += cw;
    }
    if max_w > 0 {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_streaming_unclosed_fence_and_table_and_inline() {
        let p = Palette::detect();
        let md = "# Title\n\nHello **world** with `code` and [docs](https://example.com).\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n```rust\nfn main() {\n    let x = 42;\n";
        let lines = render_markdown(md, 60, &p);
        assert!(lines.len() >= 10);
        let joined: String = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("Title"));
        assert!(joined.contains("rust …"));
        assert!(joined.contains("fn main()"));
        assert!(joined.contains("┌"));
    }

    #[test]
    fn wraps_unicode_without_panic() {
        let p = Palette::detect();
        let md = "- Déploiement 🚀 très long sur plusieurs colonnes avec des accents éàùç et du **gras**";
        let lines = render_markdown(md, 24, &p);
        assert!(lines.len() >= 3);
    }

    #[test]
    fn rust_lifetimes_do_not_swallow_rest_of_line() {
        let p = Palette::detect();
        let lines = render_markdown(
            "```rust\nfn foo<'a>(s: &'static str) -> char { 'x' }\n```",
            80,
            &p,
        );
        let joined: String = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("fn foo<'a>(s: &'static str) -> char { 'x' }"));
    }
}
