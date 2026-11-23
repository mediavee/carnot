//! Fast, zero-dependency HTML -> Markdown extractor for `WebFetch`.
//!
//! Focuses on `<main>` / `<article>` when substantial, strips navigation/boilerplate/script/style/svg,
//! preserves headings, fenced code blocks, inline code, lists, and base-resolved `[label](url)` links.

use url::Url;

const SKIP_TAGS: &[&str] = &[
    "script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "template", "iframe",
];

/// Converts an HTML document into compact, readable Markdown with links resolved against `base_url`.
pub fn html_to_markdown(html: &str, base_url: Option<&Url>) -> String {
    let title = extract_title(html);
    let body = extract_primary_region(html);
    let rendered = render_html_fragment(body, base_url);

    let mut out = String::new();
    if let Some(t) = title
        && !t.is_empty()
        && !rendered.starts_with(&format!("# {t}"))
    {
        out.push_str("# ");
        out.push_str(&t);
        out.push_str("\n\n");
    }
    out.push_str(&rendered);
    normalize_markdown(&out)
}

fn extract_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let after_open = html[start..].find('>')? + start + 1;
    let end = lower[after_open..].find("</title>")? + after_open;
    let decoded = decode_entities(&html[after_open..end]);
    let collapsed = collapse_ws(&decoded);
    (!collapsed.is_empty()).then_some(collapsed)
}

fn extract_primary_region(html: &str) -> &str {
    let lower = html.to_ascii_lowercase();
    for tag in ["main", "article"] {
        if let Some(open_pos) = find_open_tag(&lower, tag)
            && let Some(gt_rel) = html[open_pos..].find('>')
        {
            let inner_start = open_pos + gt_rel + 1;
            let close_pat = format!("</{tag}>");
            if let Some(close_rel) = lower[inner_start..].rfind(&close_pat) {
                let candidate = &html[inner_start..inner_start + close_rel];
                if candidate.len() >= 200 {
                    return candidate;
                }
            }
        }
    }
    if let Some(open_pos) = find_open_tag(&lower, "body")
        && let Some(gt_rel) = html[open_pos..].find('>')
    {
        let inner_start = open_pos + gt_rel + 1;
        let end = lower[inner_start..]
            .rfind("</body>")
            .map(|r| inner_start + r)
            .unwrap_or(html.len());
        return &html[inner_start..end];
    }
    html
}

fn find_open_tag(lower: &str, tag: &str) -> Option<usize> {
    let needle = format!("<{tag}");
    let mut offset = 0usize;
    while let Some(rel) = lower[offset..].find(&needle) {
        let pos = offset + rel;
        let next_b = lower.as_bytes().get(pos + needle.len()).copied();
        if matches!(next_b, Some(b'>' | b' ' | b'\t' | b'\n' | b'\r' | b'/')) {
            return Some(pos);
        }
        offset = pos + needle.len();
    }
    None
}

fn render_html_fragment(html: &str, base_url: Option<&Url>) -> String {
    let bytes = html.as_bytes();
    let mut out = String::with_capacity(html.len() / 2);
    let mut i = 0usize;
    let mut skip_depth: usize = 0;
    let mut skip_tag: Option<String> = None;
    let mut in_pre = false;
    let mut in_code = false;
    let mut active_link: Option<(String, String)> = None; // (href, accumulated text)

    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Skip HTML comments `<!-- ... -->`
            if html[i..].starts_with("<!--") {
                if let Some(end) = html[i + 4..].find("-->") {
                    i = i + 4 + end + 3;
                    continue;
                }
                break;
            }
            let Some(close_rel) = html[i..].find('>') else {
                break;
            };
            let raw_tag = &html[i + 1..i + close_rel];
            i = i + close_rel + 1;

            let trimmed = raw_tag.trim();
            if trimmed.is_empty() || trimmed.starts_with('!') || trimmed.starts_with('?') {
                continue;
            }
            let is_closing = trimmed.starts_with('/');
            let self_closing = trimmed.ends_with('/');
            let body = trimmed.trim_start_matches('/').trim_end_matches('/').trim();
            let tag_name = body
                .split(|c: char| c.is_whitespace() || c == '/')
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();

            if skip_depth > 0 {
                if let Some(ref st) = skip_tag
                    && tag_name == *st
                {
                    if is_closing {
                        skip_depth -= 1;
                        if skip_depth == 0 {
                            skip_tag = None;
                        }
                    } else if !self_closing {
                        skip_depth += 1;
                    }
                }
                continue;
            }

            if !is_closing && !self_closing && SKIP_TAGS.contains(&tag_name.as_str()) {
                skip_depth = 1;
                skip_tag = Some(tag_name);
                continue;
            }

            match tag_name.as_str() {
                "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                    if !is_closing {
                        let lvl = tag_name.as_bytes()[1].saturating_sub(b'0').clamp(1, 6) as usize;
                        ensure_blank_line(&mut out);
                        for _ in 0..lvl {
                            out.push('#');
                        }
                        out.push(' ');
                    } else {
                        ensure_blank_line(&mut out);
                    }
                }
                "p" | "div" | "section" | "article" | "main" | "blockquote" | "tr" => {
                    if !is_closing && tag_name == "blockquote" {
                        ensure_blank_line(&mut out);
                        out.push_str("> ");
                    } else {
                        ensure_newline(&mut out);
                        if matches!(tag_name.as_str(), "p" | "blockquote") {
                            ensure_blank_line(&mut out);
                        }
                    }
                }
                "br" => out.push('\n'),
                "hr" => {
                    ensure_blank_line(&mut out);
                    out.push_str("---\n\n");
                }
                "li" => {
                    if !is_closing {
                        ensure_newline(&mut out);
                        out.push_str("- ");
                    } else {
                        ensure_newline(&mut out);
                    }
                }
                "td" | "th" => {
                    if is_closing {
                        out.push_str(" | ");
                    }
                }
                "pre" => {
                    if !is_closing {
                        in_pre = true;
                        ensure_blank_line(&mut out);
                        out.push_str("```\n");
                    } else {
                        in_pre = false;
                        if !out.ends_with('\n') {
                            out.push('\n');
                        }
                        out.push_str("```\n\n");
                    }
                }
                "code" => {
                    if !in_pre {
                        if !is_closing && !in_code {
                            in_code = true;
                            push_target(&mut out, &mut active_link, "`");
                        } else if is_closing && in_code {
                            in_code = false;
                            push_target(&mut out, &mut active_link, "`");
                        }
                    }
                }
                "a" => {
                    if !is_closing {
                        if let Some(href) = extract_attr(body, "href") {
                            let resolved = resolve_href(&href, base_url);
                            if !resolved.is_empty() && !resolved.starts_with("javascript:") {
                                active_link = Some((resolved, String::new()));
                            }
                        }
                    } else if let Some((href, label)) = active_link.take() {
                        let clean_lbl = collapse_ws(&label);
                        if !clean_lbl.is_empty() {
                            if clean_lbl == href {
                                out.push_str(&href);
                            } else {
                                out.push_str(&format!("[{clean_lbl}]({href})"));
                            }
                        }
                    }
                }
                _ => {}
            }
        } else {
            let next_lt = html[i..].find('<').map(|r| i + r).unwrap_or(html.len());
            if skip_depth == 0 {
                let raw_text = &html[i..next_lt];
                let decoded = decode_entities(raw_text);
                if in_pre {
                    out.push_str(&decoded);
                } else {
                    let collapsed = collapse_ws_preserve_edge(&decoded);
                    if !collapsed.is_empty() {
                        push_target(&mut out, &mut active_link, &collapsed);
                    }
                }
            }
            i = next_lt;
        }
    }

    if let Some((href, label)) = active_link.take() {
        let clean_lbl = collapse_ws(&label);
        if !clean_lbl.is_empty() {
            out.push_str(&format!("[{clean_lbl}]({href})"));
        }
    }
    out
}

fn push_target(out: &mut String, active_link: &mut Option<(String, String)>, text: &str) {
    if let Some((_, buf)) = active_link.as_mut() {
        buf.push_str(text);
    } else {
        // Avoid double spaces at line boundaries
        if text == " " && (out.is_empty() || out.ends_with(['\n', ' '])) {
            return;
        }
        out.push_str(text);
    }
}

fn ensure_newline(out: &mut String) {
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
}

fn ensure_blank_line(out: &mut String) {
    if out.is_empty() {
        return;
    }
    if out.ends_with("\n\n") {
        return;
    }
    if out.ends_with('\n') {
        out.push('\n');
    } else {
        out.push_str("\n\n");
    }
}

fn extract_attr(tag_body: &str, attr: &str) -> Option<String> {
    let lower = tag_body.to_ascii_lowercase();
    let needle = format!("{attr}=");
    let pos = lower.find(&needle)?;
    let rest = tag_body[pos + needle.len()..].trim_start();
    if let Some(after_q) = rest.strip_prefix('"') {
        let end = after_q.find('"')?;
        return Some(decode_entities(&after_q[..end]));
    }
    if let Some(after_q) = rest.strip_prefix('\'') {
        let end = after_q.find('\'')?;
        return Some(decode_entities(&after_q[..end]));
    }
    let val = rest.split_whitespace().next()?;
    Some(decode_entities(val))
}

fn resolve_href(href: &str, base_url: Option<&Url>) -> String {
    let trimmed = href.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return String::new();
    }
    if let Some(base) = base_url
        && let Ok(joined) = base.join(trimmed)
    {
        return joined.to_string();
    }
    trimmed.to_string()
}

pub fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp + 1..];
        if let Some(semi) = after.find(';').filter(|&p| p <= 12) {
            let ent = &after[..semi];
            let decoded: Option<char> = match ent {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" | "#39" | "#x27" => Some('\''),
                "nbsp" => Some(' '),
                "mdash" => Some('—'),
                "ndash" => Some('–'),
                "hellip" => Some('…'),
                _ if ent.starts_with("#x") || ent.starts_with("#X") => {
                    u32::from_str_radix(&ent[2..], 16)
                        .ok()
                        .and_then(char::from_u32)
                }
                _ if ent.starts_with('#') => ent[1..].parse::<u32>().ok().and_then(char::from_u32),
                _ => None,
            };
            if let Some(ch) = decoded {
                out.push(ch);
                rest = &after[semi + 1..];
                continue;
            }
        }
        out.push('&');
        rest = after;
    }
    out.push_str(rest);
    out
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn collapse_ws_preserve_edge(s: &str) -> String {
    let lead = s.chars().next().is_some_and(char::is_whitespace);
    let trail = s.chars().last().is_some_and(char::is_whitespace);
    let core = collapse_ws(s);
    if core.is_empty() {
        return if lead || trail {
            " ".to_string()
        } else {
            String::new()
        };
    }
    let mut out = String::with_capacity(core.len() + 2);
    if lead {
        out.push(' ');
    }
    out.push_str(&core);
    if trail {
        out.push(' ');
    }
    out
}

fn normalize_markdown(s: &str) -> String {
    let mut out = Vec::new();
    let mut blank_run = 0usize;
    let mut in_fence = false;
    for raw_line in s.lines() {
        let trimmed = raw_line.trim();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            blank_run = 0;
            out.push(raw_line.trim_end().to_string());
            continue;
        }
        if in_fence {
            out.push(raw_line.to_string());
            continue;
        }
        if trimmed.is_empty() {
            blank_run += 1;
            if blank_run <= 1 && !out.is_empty() {
                out.push(String::new());
            }
        } else {
            blank_run = 0;
            out.push(trimmed.to_string());
        }
    }
    while out.last().is_some_and(|l| l.is_empty()) {
        out.pop();
    }
    out.join("\n")
}

/// Converts HTML to clean plain text (no Markdown formatting punctuation, boilerplate stripped).
pub fn html_to_text(html: &str) -> String {
    let md = html_to_markdown(html, None);
    let mut out = String::with_capacity(md.len());
    for line in md.lines() {
        let trimmed = line.trim_start_matches('#').trim_start_matches('>').trim();
        let trimmed = trimmed.strip_prefix("- ").unwrap_or(trimmed);
        if trimmed == "---" || trimmed.starts_with("```") {
            continue;
        }
        out.push_str(&trimmed.replace('`', ""));
        out.push('\n');
    }
    normalize_markdown(&out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_to_markdown_strips_boilerplate_preserves_code_and_resolves_links() {
        let html = r#"
        <html>
          <head><title>Carnot &amp; Rust</title><style>body{color:red}</style></head>
          <body>
            <nav><a href="/home">Home</a></nav>
            <main>
              <h1>Overview</h1>
              <p>Read the <a href="/docs/guide">Guide</a> &amp; run <code>cargo test</code>.</p>
              <pre><code>fn main() {
    println!("hi &#x21;");
}</code></pre>
              <ul><li>Fast</li><li>Safe</li></ul>
            </main>
            <footer>Copyright 2026</footer>
          </body>
        </html>"#;
        let base = Url::parse("https://example.com/start/").expect("url");
        let md = html_to_markdown(html, Some(&base));
        assert!(md.contains("# Carnot & Rust"), "md={md}");
        assert!(md.contains("# Overview"), "md={md}");
        assert!(
            md.contains("[Guide](https://example.com/docs/guide)"),
            "md={md}"
        );
        assert!(md.contains("`cargo test`"), "md={md}");
        assert!(md.contains("println!(\"hi !\");"), "md={md}");
        assert!(md.contains("- Fast"), "md={md}");
        assert!(
            !md.contains("Copyright 2026"),
            "footer must be stripped: {md}"
        );
        assert!(!md.contains("Home"), "nav must be stripped: {md}");
    }
}
