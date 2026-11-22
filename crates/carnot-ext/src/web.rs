//! Opt-in SSRF-safe `WebFetch` + `WebSearch` module (behind Cargo feature `web`).
//!
//! Hardened per Opus architectural checklist:
//!   - Custom `reqwest::dns::Resolve` validating EVERY resolved IP at TCP connect time (zero DNS rebinding!)
//!   - Blocks 0/8, 10/8, 100.64/10, 127/8, 169.254/16, 172.16/12, 192.168/16, multicast/broadcast,
//!     `::1`, `fc00::/7`, `fe80::/10`, `::ffff:a.b.c.d`, `64:ff9b::/96`, `2002::/16`, `localhost`, `*.local`
//!   - `.no_proxy()` + `redirect::Policy::none()` with manual redirect re-validation (max 5 hops)
//!   - Decompressed body cap (2 MiB — blocks gzip bombs!) + `webpki-root-certs` TLS fallback
//!   - Boilerplate-stripping HTML -> Markdown converter (`<main>`/`<article>` focus, strips
//!     `<nav>`/`<header>`/`<footer>`/`<aside>`/`<script>`/`<style>`/`<svg>`, preserves headings,
//!     fenced code blocks, lists, tables, and resolved `[label](url)` links!)
//!   - Deferred `WebSearch` tool (`CARNOT_SEARCH_URL` SearXNG / `BRAVE_API_KEY` / `TAVILY_API_KEY`
//!     / zero-key DuckDuckGo HTML fallback) — both `defer_loading: true` => 0 tokens in static prefix!
//!   - `ToolEffects::global_read()` so multiple `WebFetch`/`WebSearch` calls run concurrently in I2 waves!

use crate::Module;
use carnot_cas::{CasStore, shape_text_output};
use carnot_core::{CarnotError, Result, ToolEffects, ToolOutput, ToolSpec};
use carnot_tools::{BoxFuture, Tool};
use futures::StreamExt;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use serde_json::{Value, json};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use url::{Host, Url};

const MAX_REDIRECTS: usize = 5;
const MAX_DECOMPRESSED_BYTES: usize = 2 * 1024 * 1024;
const MAX_SEARCH_BODY_BYTES: usize = 1024 * 1024;

fn is_blocked_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_multicast()
        || ip.is_unspecified()
        || o[0] == 0
        || (o[0] == 100 && (o[1] & 0b1100_0000) == 64)
        || (o[0] == 192 && o[1] == 0 && (o[2] == 0 || o[2] == 2))
        || (o[0] == 198 && (o[1] == 18 || o[1] == 19))
        || (o[0] == 198 && o[1] == 51 && o[2] == 100)
        || (o[0] == 203 && o[1] == 0 && o[2] == 113)
        || o[0] >= 240
}

fn is_blocked_v6(ip: Ipv6Addr) -> bool {
    if ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() {
        return true;
    }
    let s = ip.segments();
    if (s[0] & 0xfe00) == 0xfc00 || (s[0] & 0xffc0) == 0xfe80 {
        return true;
    }
    if let Some(v4) = ip.to_ipv4_mapped() {
        return is_blocked_v4(v4);
    }
    if s[0] == 0x0064 && s[1] == 0xff9b {
        return is_blocked_v4(Ipv4Addr::new(
            (s[6] >> 8) as u8,
            (s[6] & 0xff) as u8,
            (s[7] >> 8) as u8,
            (s[7] & 0xff) as u8,
        ));
    }
    if s[0] == 0x2002 {
        return is_blocked_v4(Ipv4Addr::new(
            (s[1] >> 8) as u8,
            (s[1] & 0xff) as u8,
            (s[2] >> 8) as u8,
            (s[2] & 0xff) as u8,
        ));
    }
    false
}

fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_blocked_v4(v4),
        IpAddr::V6(v6) => is_blocked_v6(v6),
    }
}

fn validate_url(raw: &str) -> std::result::Result<Url, String> {
    let u = Url::parse(raw).map_err(|e| format!("invalid URL: {e}"))?;
    if !matches!(u.scheme(), "http" | "https") {
        return Err(format!("disallowed scheme `{}`", u.scheme()));
    }
    match u.host() {
        Some(Host::Ipv4(v4)) if is_blocked_v4(v4) => Err(format!("SSRF blocked `{v4}`")),
        Some(Host::Ipv6(v6)) if is_blocked_v6(v6) => Err(format!("SSRF blocked `{v6}`")),
        Some(Host::Domain(d)) => {
            let low = d.to_ascii_lowercase();
            if low == "localhost" || low.ends_with(".localhost") || low.ends_with(".local") {
                Err(format!("SSRF blocked local domain `{d}`"))
            } else {
                Ok(u)
            }
        }
        Some(_) => Ok(u),
        None => Err("URL missing host".into()),
    }
}

#[derive(Debug, Default)]
struct WebSsrfResolver;

impl Resolve for WebSsrfResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_string();
        Box::pin(async move {
            let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host.as_str(), 0))
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?
                .collect();
            let mut safe = Vec::with_capacity(addrs.len());
            for a in addrs {
                if is_blocked_ip(a.ip()) {
                    let err: Box<dyn std::error::Error + Send + Sync> =
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            format!("SSRF blocked resolved IP {} for `{host}`", a.ip()),
                        ));
                    return Err(err);
                }
                safe.push(a);
            }
            if safe.is_empty() {
                let err: Box<dyn std::error::Error + Send + Sync> = Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("DNS returned no addresses for `{host}`"),
                ));
                return Err(err);
            }
            let iter: Addrs = Box::new(safe.into_iter());
            Ok(iter)
        })
    }
}

fn build_web_client(enforce_ssrf: bool, timeout_secs: u64) -> Result<reqwest::Client> {
    let make_builder = || {
        let mut b = reqwest::Client::builder()
            .user_agent(concat!("carnot/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(timeout_secs));
        if enforce_ssrf {
            b = b
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .dns_resolver(Arc::new(WebSsrfResolver));
        }
        b
    };
    if let Ok(c) = make_builder().build() {
        return Ok(c);
    }
    let certs: Vec<reqwest::Certificate> = webpki_root_certs::TLS_SERVER_ROOT_CERTS
        .iter()
        .filter_map(|der| reqwest::Certificate::from_der(der.as_ref()).ok())
        .collect();
    make_builder()
        .tls_certs_only(certs)
        .build()
        .map_err(|e| CarnotError::Tool {
            tool: "Web".into(),
            message: e.to_string(),
        })
}

/// Decodes common HTML named & numeric entities in a single pass (avoids double-unescaping `&amp;lt;`).
fn decode_html_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let Some(semi) = rest[1..].find(';').filter(|&pos| pos <= 12) else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let entity = &rest[1..1 + semi];
        let decoded: Option<char> = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "nbsp" => Some(' '),
            "mdash" => Some('—'),
            "ndash" => Some('–'),
            "hellip" => Some('…'),
            "copy" => Some('©'),
            _ if entity.starts_with("#x") || entity.starts_with("#X") => {
                u32::from_str_radix(&entity[2..], 16)
                    .ok()
                    .and_then(char::from_u32)
            }
            _ if entity.starts_with('#') => {
                entity[1..].parse::<u32>().ok().and_then(char::from_u32)
            }
            _ => None,
        };
        if let Some(ch) = decoded {
            out.push(ch);
            rest = &rest[2 + semi..];
        } else {
            out.push('&');
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    out
}

/// Extracts an HTML attribute value (case-insensitive name) from a raw tag interior (e.g. `a href="https://..."`).
fn extract_html_attr(tag_inner: &str, attr_name: &str) -> Option<String> {
    let low = tag_inner.to_ascii_lowercase();
    let mut search_from = 0usize;
    while let Some(rel) = low[search_from..].find(attr_name) {
        let pos = search_from + rel;
        let boundary_before = pos == 0
            || tag_inner.as_bytes()[pos - 1].is_ascii_whitespace()
            || tag_inner.as_bytes()[pos - 1] == b'/';
        let after = pos + attr_name.len();
        if boundary_before {
            let tail = tag_inner[after..].trim_start();
            if let Some(after_eq) = tail.strip_prefix('=') {
                let val_start = after_eq.trim_start();
                if let Some(q) = val_start.chars().next()
                    && (q == '"' || q == '\'')
                {
                    let rest = &val_start[1..];
                    let end = rest.find(q).unwrap_or(rest.len());
                    return Some(decode_html_entities(&rest[..end]));
                }
                let end = val_start
                    .find(|c: char| c.is_ascii_whitespace() || c == '>' || c == '/')
                    .unwrap_or(val_start.len());
                return Some(decode_html_entities(&val_start[..end]));
            }
        }
        search_from = after;
    }
    None
}

/// Extracts the inner HTML of the first `<tag_name ...>...</tag_name>` block (respecting nesting).
fn extract_primary_container<'a>(html: &'a str, target_tag: &str) -> Option<&'a str> {
    let bytes = html.as_bytes();
    let mut i = 0usize;
    let mut start_inner: Option<usize> = None;
    let mut depth = 0usize;

    while i < bytes.len() {
        if bytes[i] == b'<' {
            if html[i..].starts_with("<!--") {
                i = html[i + 4..]
                    .find("-->")
                    .map(|r| i + 4 + r + 3)
                    .unwrap_or(bytes.len());
                continue;
            }
            let close = html[i..].find('>').map(|r| i + r).unwrap_or(bytes.len());
            let inner = html[i + 1..close].trim();
            let self_closing = inner.ends_with('/');
            let (is_closing, rest) = match inner.strip_prefix('/') {
                Some(r) => (true, r.trim_start()),
                None => (false, inner),
            };
            let name = rest
                .split(|c: char| c.is_ascii_whitespace() || c == '/' || c == '>')
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();
            if name == target_tag {
                if !is_closing && !self_closing {
                    if start_inner.is_none() {
                        start_inner = Some((close + 1).min(bytes.len()));
                    }
                    depth += 1;
                } else if is_closing && depth > 0 {
                    depth -= 1;
                    if depth == 0
                        && let Some(s) = start_inner
                    {
                        return Some(&html[s..i]);
                    }
                }
            }
            i = (close + 1).min(bytes.len());
        } else {
            i += 1;
        }
    }
    None
}

fn is_skip_subtree_tag(tag: &str) -> bool {
    matches!(
        tag,
        "script"
            | "style"
            | "noscript"
            | "svg"
            | "template"
            | "nav"
            | "header"
            | "footer"
            | "aside"
    )
}

fn render_html_slice_to_markdown(html: &str, base_url: Option<&Url>) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let bytes = html.as_bytes();
    let mut i = 0usize;
    let mut skip_stack: Vec<String> = Vec::new();
    let mut in_pre = false;
    let mut in_inline_code = false;
    let mut active_link: Option<(usize, String)> = None;
    let mut in_tr_first_cell = false;
    let mut last_ws = true;

    let push_newlines = |buf: &mut String, count: usize, last_ws: &mut bool| {
        if buf.is_empty() {
            return;
        }
        let existing = buf.bytes().rev().take_while(|&b| b == b'\n').count();
        for _ in existing..count {
            buf.push('\n');
        }
        *last_ws = true;
    };

    while i < bytes.len() {
        if bytes[i] == b'<' {
            if html[i..].starts_with("<!--") {
                i = html[i + 4..]
                    .find("-->")
                    .map(|r| i + 4 + r + 3)
                    .unwrap_or(bytes.len());
                continue;
            }
            let close = html[i..].find('>').map(|r| i + r).unwrap_or(bytes.len());
            let raw_inner = html[i + 1..close].trim();
            let self_closing = raw_inner.ends_with('/');
            let (is_closing, rest) = match raw_inner.strip_prefix('/') {
                Some(r) => (true, r.trim_start()),
                None => (false, raw_inner),
            };
            let tag_name = rest
                .split(|c: char| c.is_ascii_whitespace() || c == '/' || c == '>')
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();

            if !skip_stack.is_empty() {
                if !is_closing && !self_closing && is_skip_subtree_tag(&tag_name) {
                    skip_stack.push(tag_name);
                } else if is_closing && skip_stack.last().map(String::as_str) == Some(&tag_name) {
                    skip_stack.pop();
                }
                i = (close + 1).min(bytes.len());
                continue;
            }

            if !is_closing && !self_closing && is_skip_subtree_tag(&tag_name) {
                skip_stack.push(tag_name);
                i = (close + 1).min(bytes.len());
                continue;
            }

            match (is_closing, tag_name.as_str()) {
                (false, "h1") => {
                    push_newlines(&mut out, 2, &mut last_ws);
                    out.push_str("# ");
                    last_ws = true;
                }
                (false, "h2") => {
                    push_newlines(&mut out, 2, &mut last_ws);
                    out.push_str("## ");
                    last_ws = true;
                }
                (false, "h3") => {
                    push_newlines(&mut out, 2, &mut last_ws);
                    out.push_str("### ");
                    last_ws = true;
                }
                (false, "h4") => {
                    push_newlines(&mut out, 2, &mut last_ws);
                    out.push_str("#### ");
                    last_ws = true;
                }
                (false, "h5") => {
                    push_newlines(&mut out, 2, &mut last_ws);
                    out.push_str("##### ");
                    last_ws = true;
                }
                (false, "h6") => {
                    push_newlines(&mut out, 2, &mut last_ws);
                    out.push_str("###### ");
                    last_ws = true;
                }
                (true, "h1" | "h2" | "h3" | "h4" | "h5" | "h6") => {
                    push_newlines(&mut out, 2, &mut last_ws);
                }
                (false, "pre") => {
                    push_newlines(&mut out, 2, &mut last_ws);
                    out.push_str("```\n");
                    in_pre = true;
                    last_ws = true;
                }
                (true, "pre") => {
                    if in_pre {
                        if !out.ends_with('\n') {
                            out.push('\n');
                        }
                        out.push_str("```\n\n");
                        in_pre = false;
                        last_ws = true;
                    }
                }
                (false, "code") if !in_pre => {
                    out.push('`');
                    in_inline_code = true;
                    last_ws = false;
                }
                (true, "code") if !in_pre && in_inline_code => {
                    out.push('`');
                    in_inline_code = false;
                    last_ws = false;
                }
                (false, "a") if !in_pre && active_link.is_none() => {
                    if let Some(href) = extract_html_attr(rest, "href") {
                        let href_t = href.trim();
                        if !href_t.is_empty()
                            && !href_t.starts_with('#')
                            && !href_t.to_ascii_lowercase().starts_with("javascript:")
                        {
                            let resolved = match base_url {
                                Some(base) => base
                                    .join(href_t)
                                    .map(|u| u.to_string())
                                    .unwrap_or_else(|_| href_t.to_string()),
                                None => href_t.to_string(),
                            };
                            if !last_ws && !out.ends_with(['(', '[', '`']) {
                                out.push(' ');
                            }
                            out.push('[');
                            active_link = Some((out.len(), resolved));
                            last_ws = true;
                        }
                    }
                }
                (true, "a") => {
                    if let Some((start_pos, resolved)) = active_link.take() {
                        let label = out[start_pos..].trim().to_string();
                        if label.is_empty() {
                            // Remove dangling `[` if link had no visible text (e.g. icon-only link)
                            out.truncate(start_pos.saturating_sub(1));
                        } else if label == resolved {
                            out.truncate(start_pos.saturating_sub(1));
                            out.push_str(&resolved);
                        } else {
                            let trimmed_len = out.trim_end().len();
                            out.truncate(trimmed_len);
                            out.push_str("](");
                            out.push_str(&resolved);
                            out.push(')');
                        }
                        last_ws = false;
                    }
                }
                (false, "li") => {
                    push_newlines(&mut out, 1, &mut last_ws);
                    out.push_str("- ");
                    last_ws = true;
                }
                (true, "li") => {
                    push_newlines(&mut out, 1, &mut last_ws);
                }
                (false, "blockquote") => {
                    push_newlines(&mut out, 2, &mut last_ws);
                    out.push_str("> ");
                    last_ws = true;
                }
                (true, "blockquote") => {
                    push_newlines(&mut out, 2, &mut last_ws);
                }
                (false, "tr") => {
                    push_newlines(&mut out, 1, &mut last_ws);
                    out.push_str("| ");
                    in_tr_first_cell = true;
                    last_ws = true;
                }
                (false, "td" | "th") => {
                    if !in_tr_first_cell {
                        if !out.ends_with("| ") {
                            out.push_str(" | ");
                        }
                        last_ws = true;
                    }
                    in_tr_first_cell = false;
                }
                (true, "tr") => {
                    let tlen = out.trim_end_matches(' ').len();
                    out.truncate(tlen);
                    if !out.ends_with('|') {
                        out.push_str(" |");
                    }
                    push_newlines(&mut out, 1, &mut last_ws);
                }
                (false, "hr") => {
                    push_newlines(&mut out, 2, &mut last_ws);
                    out.push_str("---\n\n");
                    last_ws = true;
                }
                (false, "br") => {
                    out.push('\n');
                    last_ws = true;
                }
                (false, "p") | (true, "p") => {
                    push_newlines(&mut out, 2, &mut last_ws);
                }
                (false, "div" | "section" | "article" | "main" | "ul" | "ol" | "table")
                | (true, "div" | "section" | "article" | "main" | "ul" | "ol" | "table") => {
                    push_newlines(&mut out, 1, &mut last_ws);
                }
                _ => {}
            }

            i = (close + 1).min(bytes.len());
            continue;
        }

        if !skip_stack.is_empty() {
            i += 1;
            continue;
        }

        // Read text run up to next `<` and decode entities once cleanly
        let next_lt = html[i..].find('<').map(|r| i + r).unwrap_or(bytes.len());
        let decoded = decode_html_entities(&html[i..next_lt]);
        if in_pre {
            out.push_str(&decoded);
            last_ws = decoded.ends_with('\n');
        } else {
            for ch in decoded.chars() {
                if ch.is_whitespace() {
                    if !last_ws {
                        out.push(' ');
                        last_ws = true;
                    }
                } else {
                    out.push(ch);
                    last_ws = false;
                }
            }
        }
        i = next_lt;
    }

    // Close any unclosed link cleanly
    if let Some((start_pos, resolved)) = active_link {
        let label = out[start_pos..].trim().to_string();
        if label.is_empty() {
            out.truncate(start_pos.saturating_sub(1));
        } else {
            let trimmed_len = out.trim_end().len();
            out.truncate(trimmed_len);
            out.push_str("](");
            out.push_str(&resolved);
            out.push(')');
        }
    }

    // Collapse trailing spaces per line and excess blank lines (>2)
    let mut cleaned = String::with_capacity(out.len());
    let mut blank_run = 0usize;
    for line in out.lines() {
        let t = line.trim_end();
        if t.is_empty() {
            blank_run += 1;
            if blank_run <= 1 && !cleaned.is_empty() {
                cleaned.push('\n');
            }
        } else {
            blank_run = 0;
            if !cleaned.is_empty() {
                cleaned.push('\n');
            }
            cleaned.push_str(t);
        }
    }
    cleaned.trim().to_string()
}

/// High-signal, zero-dependency HTML-to-Markdown extractor:
///   - Prefers `<main>` or `<article>` when substantial (>=180 chars) to drop site chrome
///   - Strips `<script>`, `<style>`, `<noscript>`, `<svg>`, `<nav>`, `<header>`, `<footer>`, `<aside>`
///   - Preserves headings (`#`), fenced code blocks, lists, tables, and resolved `[label](url)` links
pub fn html_to_markdown(html: &str, base_url: Option<&Url>) -> String {
    for primary in ["main", "article"] {
        if let Some(inner) = extract_primary_container(html, primary) {
            let md = render_html_slice_to_markdown(inner, base_url);
            if md.len() >= 180 {
                return md;
            }
        }
    }
    render_html_slice_to_markdown(html, base_url)
}

/// Backwards-compatible alias converting HTML to clean Markdown text.
pub fn html_to_clean_text(html: &str) -> String {
    html_to_markdown(html, None)
}

#[derive(Debug)]
pub struct WebFetchTool {
    client: reqwest::Client,
    cas: Arc<CasStore>,
    cas_enabled: bool,
    spill_threshold: usize,
    preview_max: usize,
}

impl WebFetchTool {
    pub fn new(
        cas: Arc<CasStore>,
        cas_enabled: bool,
        spill_threshold: usize,
        preview_max: usize,
    ) -> Result<Self> {
        let client = build_web_client(true, 30)?;
        Ok(Self {
            client,
            cas,
            cas_enabled,
            spill_threshold,
            preview_max,
        })
    }
}

impl Tool for WebFetchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "WebFetch".into(),
            description: "Fetch a public HTTP/HTTPS URL safely (SSRF-hardened, HTML converted to clean Markdown preserving links/code/tables while stripping nav/footer noise; large pages spilled to .carnot/h/).".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["url"],
                "properties": {
                    "url": { "type": "string", "description": "Public http:// or https:// URL" },
                    "raw": { "type": "boolean", "description": "Return raw text without HTML-to-Markdown extraction" }
                }
            }),
            defer_loading: true,
        }
    }

    fn effects(&self, _input: &Value) -> ToolEffects {
        ToolEffects::global_read()
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let Some(raw_url) = input.get("url").and_then(Value::as_str) else {
                return Ok(ToolOutput::err_text("WebFetch requires `url`"));
            };
            let raw_mode = input.get("raw").and_then(Value::as_bool).unwrap_or(false);
            let mut current = match validate_url(raw_url) {
                Ok(u) => u,
                Err(msg) => return Ok(ToolOutput::err_text(msg)),
            };

            let mut resp_opt = None;
            for _hop in 0..=MAX_REDIRECTS {
                let send_fut = self.client.get(current.as_str()).send();
                let resp = tokio::select! {
                    _ = cancel.cancelled() => return Err(CarnotError::Cancelled),
                    r = send_fut => match r {
                        Ok(r) => r,
                        Err(e) => return Ok(ToolOutput::err_text(format!("WebFetch error: {e}"))),
                    }
                };
                if resp.status().is_redirection() {
                    let Some(loc) = resp.headers().get("location").and_then(|v| v.to_str().ok())
                    else {
                        return Ok(ToolOutput::err_text("redirect missing Location header"));
                    };
                    let next = match current.join(loc) {
                        Ok(u) => u,
                        Err(e) => {
                            return Ok(ToolOutput::err_text(format!("invalid redirect URL: {e}")));
                        }
                    };
                    current = match validate_url(next.as_str()) {
                        Ok(u) => u,
                        Err(msg) => return Ok(ToolOutput::err_text(msg)),
                    };
                    continue;
                }
                resp_opt = Some(resp);
                break;
            }

            let Some(resp) = resp_opt else {
                return Ok(ToolOutput::err_text(
                    "WebFetch exceeded maximum 5 redirects",
                ));
            };
            let status = resp.status();
            let content_type = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_ascii_lowercase();

            // Read bounded bytes with cancellation (anti-gzip-bomb 2 MiB cap)
            let mut body = Vec::new();
            let mut truncated_body = false;
            let mut stream = resp.bytes_stream();
            loop {
                let next_chunk = tokio::select! {
                    _ = cancel.cancelled() => return Err(CarnotError::Cancelled),
                    c = stream.next() => c,
                };
                let Some(chunk) = next_chunk else { break };
                let bytes = match chunk {
                    Ok(b) => b,
                    Err(e) => {
                        return Ok(ToolOutput::err_text(format!("WebFetch stream error: {e}")));
                    }
                };
                if body.len() + bytes.len() > MAX_DECOMPRESSED_BYTES {
                    let take = MAX_DECOMPRESSED_BYTES.saturating_sub(body.len());
                    body.extend_from_slice(&bytes[..take]);
                    truncated_body = true;
                    break;
                }
                body.extend_from_slice(&bytes);
            }

            let raw_text = String::from_utf8_lossy(&body);
            let clean = if !raw_mode
                && (content_type.contains("text/html")
                    || content_type.contains("application/xhtml"))
            {
                html_to_markdown(&raw_text, Some(&current))
            } else {
                raw_text.to_string()
            };

            let framed = format!(
                "[external-untrusted-content url={} status={}{}]\n{}",
                current,
                status.as_u16(),
                if truncated_body {
                    " truncated=2MiB"
                } else {
                    ""
                },
                clean
            );

            let shaped = shape_text_output(
                &framed,
                self.cas_enabled.then_some(self.cas.as_ref()),
                self.spill_threshold,
                self.preview_max,
            )?;
            let handle = shaped.spilled.map(|h| h.rel_path);
            Ok(ToolOutput {
                blocks: vec![carnot_core::ToolResultBlock::text(shaped.text)],
                is_error: !status.is_success(),
                details: json!({
                    "url": current.as_str(),
                    "status": status.as_u16(),
                    "handle": handle,
                }),
            })
        })
    }
}

/// Structured search result item returned by `WebSearch`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

fn format_search_hits(engine: &str, query: &str, hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        return format!(
            "[external-untrusted-content search_engine={engine} query={query:?}]\nNo search results found."
        );
    }
    let mut out = format!(
        "[external-untrusted-content search_engine={engine} query={query:?} results={}]\n",
        hits.len()
    );
    for (i, h) in hits.iter().enumerate() {
        out.push_str(&format!("[{}] {} — {}\n", i + 1, h.title, h.url));
        if !h.snippet.is_empty() {
            out.push_str(&format!("    {}\n", h.snippet));
        }
    }
    out.trim_end().to_string()
}

/// Parses SearXNG JSON (`{"results":[{"title","url","content"}]}`).
pub fn parse_searxng_results(body: &Value, max_results: usize) -> Vec<SearchHit> {
    body.get("results")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let url = item.get("url").and_then(Value::as_str)?.trim().to_string();
                    if url.is_empty() {
                        return None;
                    }
                    let title = item
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or(&url)
                        .trim()
                        .to_string();
                    let snippet = item
                        .get("content")
                        .or_else(|| item.get("snippet"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    Some(SearchHit {
                        title,
                        url,
                        snippet,
                    })
                })
                .take(max_results)
                .collect()
        })
        .unwrap_or_default()
}

/// Parses Brave Search API JSON (`{"web":{"results":[{"title","url","description"}]}}`).
pub fn parse_brave_results(body: &Value, max_results: usize) -> Vec<SearchHit> {
    body.pointer("/web/results")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let url = item.get("url").and_then(Value::as_str)?.trim().to_string();
                    if url.is_empty() {
                        return None;
                    }
                    let title = item
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or(&url)
                        .trim()
                        .to_string();
                    let snippet = item
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    Some(SearchHit {
                        title,
                        url,
                        snippet,
                    })
                })
                .take(max_results)
                .collect()
        })
        .unwrap_or_default()
}

/// Parses Tavily Search API JSON (`{"results":[{"title","url","content"}]}`).
pub fn parse_tavily_results(body: &Value, max_results: usize) -> Vec<SearchHit> {
    parse_searxng_results(body, max_results)
}

fn unwrap_ddg_url(raw_href: &str) -> String {
    let trimmed = raw_href.trim();
    let full = if trimmed.starts_with("//") {
        format!("https:{trimmed}")
    } else if trimmed.starts_with('/') {
        format!("https://duckduckgo.com{trimmed}")
    } else {
        trimmed.to_string()
    };
    if let Ok(parsed) = Url::parse(&full) {
        for (k, v) in parsed.query_pairs() {
            if k == "uddg" && !v.is_empty() {
                return v.into_owned();
            }
        }
    }
    full
}

/// Parses zero-key DuckDuckGo HTML lite results (`result__a` + `result__snippet`).
pub fn parse_ddg_html_results(html: &str, max_results: usize) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    let mut cursor = 0usize;
    while hits.len() < max_results {
        let Some(rel_a) = html[cursor..].find("result__a") else {
            break;
        };
        let marker = cursor + rel_a;
        let Some(tag_open) = html[..marker].rfind("<a") else {
            cursor = marker + 9;
            continue;
        };
        let Some(tag_close_rel) = html[tag_open..].find('>') else {
            break;
        };
        let tag_close = tag_open + tag_close_rel;
        let tag_inner = &html[tag_open + 1..tag_close];
        let Some(raw_href) = extract_html_attr(tag_inner, "href") else {
            cursor = tag_close + 1;
            continue;
        };
        let Some(end_a_rel) = html[tag_close + 1..].find("</a>") else {
            break;
        };
        let title_raw = &html[tag_close + 1..tag_close + 1 + end_a_rel];
        let title = render_html_slice_to_markdown(title_raw, None)
            .replace('\n', " ")
            .trim()
            .to_string();
        let url = unwrap_ddg_url(&raw_href);

        let after_a = tag_close + 1 + end_a_rel + 4;
        let next_result = html[after_a..]
            .find("result__a")
            .map(|r| after_a + r)
            .unwrap_or(html.len());
        let region = &html[after_a..next_result];
        let snippet = if let Some(sn_pos) = region.find("result__snippet")
            && let Some(gt_rel) = region[sn_pos..].find('>')
        {
            let content_start = sn_pos + gt_rel + 1;
            let content_end = region[content_start..]
                .find("</a>")
                .or_else(|| region[content_start..].find("</div>"))
                .map(|r| content_start + r)
                .unwrap_or(region.len());
            render_html_slice_to_markdown(&region[content_start..content_end], None)
                .replace('\n', " ")
                .trim()
                .to_string()
        } else {
            String::new()
        };

        if !url.is_empty() && !title.is_empty() && !url.contains("duckduckgo.com/y.js") {
            hits.push(SearchHit {
                title,
                url,
                snippet,
            });
        }
        cursor = after_a;
    }
    hits
}

async fn read_bounded_bytes(
    resp: reqwest::Response,
    max_bytes: usize,
    cancel: &CancellationToken,
) -> std::result::Result<Vec<u8>, String> {
    let mut body = Vec::new();
    let mut stream = resp.bytes_stream();
    loop {
        let next_chunk = tokio::select! {
            _ = cancel.cancelled() => return Err("cancelled".into()),
            c = stream.next() => c,
        };
        let Some(chunk) = next_chunk else { break };
        let bytes = chunk.map_err(|e| format!("stream error: {e}"))?;
        if body.len() + bytes.len() > max_bytes {
            let take = max_bytes.saturating_sub(body.len());
            body.extend_from_slice(&bytes[..take]);
            break;
        }
        body.extend_from_slice(&bytes);
    }
    Ok(body)
}

/// Deferred SSRF-hardened `WebSearch` tool (0 tokens in static prefix).
/// Automatically selects the best configured backend:
///   1. `CARNOT_SEARCH_URL` (self-hosted SearXNG JSON endpoint)
///   2. `BRAVE_API_KEY` (Brave Search API)
///   3. `TAVILY_API_KEY` (Tavily Search API)
///   4. Zero-key DuckDuckGo HTML fallback
#[derive(Debug)]
pub struct WebSearchTool {
    ssrf_client: reqwest::Client,
    local_client: reqwest::Client,
    cas: Arc<CasStore>,
    cas_enabled: bool,
    spill_threshold: usize,
    preview_max: usize,
}

impl WebSearchTool {
    pub fn new(
        cas: Arc<CasStore>,
        cas_enabled: bool,
        spill_threshold: usize,
        preview_max: usize,
    ) -> Result<Self> {
        Ok(Self {
            ssrf_client: build_web_client(true, 20)?,
            local_client: build_web_client(false, 20)?,
            cas,
            cas_enabled,
            spill_threshold,
            preview_max,
        })
    }
}

impl Tool for WebSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "WebSearch".into(),
            description: "Search the web for documentation, issues, or current facts. Uses CARNOT_SEARCH_URL (SearXNG), BRAVE_API_KEY, TAVILY_API_KEY, or zero-key DuckDuckGo fallback.".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["query"],
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "max_results": { "type": "integer", "description": "Max results to return (default 5, max 10)" },
                    "site": { "type": "string", "description": "Optional domain filter (e.g. docs.rs)" }
                }
            }),
            defer_loading: true,
        }
    }

    fn effects(&self, _input: &Value) -> ToolEffects {
        ToolEffects::global_read()
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let Some(raw_query) = input
                .get("query")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
            else {
                return Ok(ToolOutput::err_text(
                    "WebSearch requires a non-empty `query`",
                ));
            };
            let max_results = input
                .get("max_results")
                .and_then(Value::as_u64)
                .unwrap_or(5)
                .clamp(1, 10) as usize;
            let effective_query = match input
                .get("site")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                Some(site) => format!("site:{site} {raw_query}"),
                None => raw_query.to_string(),
            };

            let (engine, hits): (&str, Vec<SearchHit>) = if let Ok(custom_url) =
                std::env::var("CARNOT_SEARCH_URL")
                && !custom_url.trim().is_empty()
            {
                let mut u = match Url::parse(custom_url.trim()) {
                    Ok(u) => u,
                    Err(e) => {
                        return Ok(ToolOutput::err_text(format!(
                            "Invalid CARNOT_SEARCH_URL: {e}"
                        )));
                    }
                };
                u.query_pairs_mut()
                    .append_pair("q", &effective_query)
                    .append_pair("format", "json");
                let resp = tokio::select! {
                    _ = cancel.cancelled() => return Err(CarnotError::Cancelled),
                    r = self.local_client.get(u).send() => match r {
                        Ok(r) => r,
                        Err(e) => return Ok(ToolOutput::err_text(format!("WebSearch (SearXNG) error: {e}"))),
                    }
                };
                if !resp.status().is_success() {
                    return Ok(ToolOutput::err_text(format!(
                        "WebSearch (SearXNG) HTTP {}",
                        resp.status()
                    )));
                }
                let bytes = match read_bounded_bytes(resp, MAX_SEARCH_BODY_BYTES, &cancel).await {
                    Ok(b) => b,
                    Err(e) if e == "cancelled" => return Err(CarnotError::Cancelled),
                    Err(e) => {
                        return Ok(ToolOutput::err_text(format!("WebSearch read error: {e}")));
                    }
                };
                let val: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
                ("searxng", parse_searxng_results(&val, max_results))
            } else if let Ok(key) = std::env::var("BRAVE_API_KEY")
                && !key.trim().is_empty()
            {
                let mut u = Url::parse("https://api.search.brave.com/res/v1/web/search")
                    .expect("static brave url");
                u.query_pairs_mut()
                    .append_pair("q", &effective_query)
                    .append_pair("count", &max_results.to_string());
                let resp = tokio::select! {
                    _ = cancel.cancelled() => return Err(CarnotError::Cancelled),
                    r = self.ssrf_client.get(u)
                        .header("Accept", "application/json")
                        .header("X-Subscription-Token", key.trim())
                        .send() => match r {
                        Ok(r) => r,
                        Err(e) => return Ok(ToolOutput::err_text(format!("WebSearch (Brave) error: {e}"))),
                    }
                };
                if !resp.status().is_success() {
                    return Ok(ToolOutput::err_text(format!(
                        "WebSearch (Brave) HTTP {}",
                        resp.status()
                    )));
                }
                let bytes = match read_bounded_bytes(resp, MAX_SEARCH_BODY_BYTES, &cancel).await {
                    Ok(b) => b,
                    Err(e) if e == "cancelled" => return Err(CarnotError::Cancelled),
                    Err(e) => {
                        return Ok(ToolOutput::err_text(format!("WebSearch read error: {e}")));
                    }
                };
                let val: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
                ("brave", parse_brave_results(&val, max_results))
            } else if let Ok(key) = std::env::var("TAVILY_API_KEY")
                && !key.trim().is_empty()
            {
                let body = json!({
                    "api_key": key.trim(),
                    "query": effective_query,
                    "max_results": max_results,
                });
                let resp = tokio::select! {
                    _ = cancel.cancelled() => return Err(CarnotError::Cancelled),
                    r = self.ssrf_client.post("https://api.tavily.com/search")
                        .json(&body)
                        .send() => match r {
                        Ok(r) => r,
                        Err(e) => return Ok(ToolOutput::err_text(format!("WebSearch (Tavily) error: {e}"))),
                    }
                };
                if !resp.status().is_success() {
                    return Ok(ToolOutput::err_text(format!(
                        "WebSearch (Tavily) HTTP {}",
                        resp.status()
                    )));
                }
                let bytes = match read_bounded_bytes(resp, MAX_SEARCH_BODY_BYTES, &cancel).await {
                    Ok(b) => b,
                    Err(e) if e == "cancelled" => return Err(CarnotError::Cancelled),
                    Err(e) => {
                        return Ok(ToolOutput::err_text(format!("WebSearch read error: {e}")));
                    }
                };
                let val: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
                ("tavily", parse_tavily_results(&val, max_results))
            } else {
                let mut u =
                    Url::parse("https://html.duckduckgo.com/html/").expect("static ddg url");
                u.query_pairs_mut().append_pair("q", &effective_query);
                let resp = tokio::select! {
                    _ = cancel.cancelled() => return Err(CarnotError::Cancelled),
                    r = self.ssrf_client.get(u)
                        .header("Accept", "text/html")
                        .send() => match r {
                        Ok(r) => r,
                        Err(e) => return Ok(ToolOutput::err_text(format!("WebSearch (DuckDuckGo) error: {e}"))),
                    }
                };
                if !resp.status().is_success() {
                    return Ok(ToolOutput::err_text(format!(
                        "WebSearch (DuckDuckGo) HTTP {}",
                        resp.status()
                    )));
                }
                let bytes = match read_bounded_bytes(resp, MAX_SEARCH_BODY_BYTES, &cancel).await {
                    Ok(b) => b,
                    Err(e) if e == "cancelled" => return Err(CarnotError::Cancelled),
                    Err(e) => {
                        return Ok(ToolOutput::err_text(format!("WebSearch read error: {e}")));
                    }
                };
                let html = String::from_utf8_lossy(&bytes);
                ("duckduckgo", parse_ddg_html_results(&html, max_results))
            };

            let framed = format_search_hits(engine, &effective_query, &hits);
            let shaped = shape_text_output(
                &framed,
                self.cas_enabled.then_some(self.cas.as_ref()),
                self.spill_threshold,
                self.preview_max,
            )?;
            let handle = shaped.spilled.map(|h| h.rel_path);
            Ok(ToolOutput::ok_text(shaped.text).with_details(json!({
                "engine": engine,
                "query": effective_query,
                "results_count": hits.len(),
                "handle": handle,
            })))
        })
    }
}

/// Opt-in `Module` registering `WebFetch` and `WebSearch` (both `defer_loading: true` => 0 tokens in static prefix!).
#[derive(Debug)]
pub struct WebFetchModule {
    fetch_tool: Arc<dyn Tool>,
    search_tool: Arc<dyn Tool>,
}

impl WebFetchModule {
    pub fn new(
        cas: Arc<CasStore>,
        cas_enabled: bool,
        spill_threshold: usize,
        preview_max: usize,
    ) -> Result<Self> {
        let fetch_tool = Arc::new(WebFetchTool::new(
            Arc::clone(&cas),
            cas_enabled,
            spill_threshold,
            preview_max,
        )?);
        let search_tool = Arc::new(WebSearchTool::new(
            cas,
            cas_enabled,
            spill_threshold,
            preview_max,
        )?);
        Ok(Self {
            fetch_tool,
            search_tool,
        })
    }
}

impl Module for WebFetchModule {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![self.fetch_tool.clone(), self.search_tool.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_to_markdown_strips_boilerplate_and_preserves_links_code_and_tables() {
        let base = Url::parse("https://docs.example.com/guide/intro.html").expect("url");
        let html = r#"
        <html>
          <head><style>body{color:red}</style><script>alert(1)</script></head>
          <body>
            <header><nav><a href="/home">NavHome</a><a href="/pricing">NavPricing</a></nav></header>
            <!-- hidden > comment -->
            <main>
              <h1>Carnot Guide</h1>
              <p>Read the <a href="/api/v2">API &amp; SDK</a> docs&nbsp;today &mdash; fast &apos;n&apos; safe.</p>
              <pre><code>fn main() {
    println!("hi");
}</code></pre>
              <ul><li>Item one</li><li>Item two</li></ul>
              <table><tr><th>Key</th><th>Value</th></tr><tr><td>TTL</td><td>5m</td></tr></table>
              <p>Padding paragraph so main container passes the 180-character primary content threshold cleanly for realistic documentation extraction.</p>
            </main>
            <footer>Copyright 2026 Noise Footer</footer>
          </body>
        </html>"#;
        let md = html_to_markdown(html, Some(&base));
        assert!(md.contains("# Carnot Guide"), "got:\n{md}");
        assert!(
            md.contains("[API & SDK](https://docs.example.com/api/v2)"),
            "got:\n{md}"
        );
        assert!(
            md.contains("```\nfn main() {\n    println!(\"hi\");\n}\n```"),
            "got:\n{md}"
        );
        assert!(md.contains("- Item one"), "got:\n{md}");
        assert!(md.contains("| Key | Value |"), "got:\n{md}");
        assert!(!md.contains("NavHome"), "navbar should be stripped: {md}");
        assert!(
            !md.contains("Noise Footer"),
            "footer should be stripped: {md}"
        );
        assert!(!md.contains("alert(1)"), "script should be stripped: {md}");
        assert!(!md.contains("hidden"), "comment should be stripped: {md}");
    }

    #[test]
    fn parses_ddg_brave_and_searxng_search_results() {
        let ddg_html = r##"
        <div class="result">
          <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdoc.rust-lang.org%2Fbook%2F&amp;rut=1">The Rust Book</a>
          <a class="result__snippet" href="#">Learn <b>Rust</b> programming safely.</a>
        </div>"##;
        let ddg = parse_ddg_html_results(ddg_html, 5);
        assert_eq!(ddg.len(), 1);
        assert_eq!(ddg[0].title, "The Rust Book");
        assert_eq!(ddg[0].url, "https://doc.rust-lang.org/book/");
        assert!(ddg[0].snippet.contains("Learn Rust programming safely."));

        let brave_json = json!({
            "web": { "results": [
                {"title": "Brave Hit", "url": "https://example.org", "description": "Fast search"}
            ]}
        });
        let brave = parse_brave_results(&brave_json, 5);
        assert_eq!(brave.len(), 1);
        assert_eq!(brave[0].title, "Brave Hit");

        let searx_json = json!({
            "results": [
                {"title": "SearX Hit", "url": "https://searx.local/doc", "content": "Private"}
            ]
        });
        let searx = parse_searxng_results(&searx_json, 5);
        assert_eq!(searx.len(), 1);
        assert_eq!(searx[0].snippet, "Private");
    }

    #[test]
    fn ssrf_validator_rejects_loopback_and_decimal_ip() {
        assert!(validate_url("http://127.0.0.1:7744/v1/prompt").is_err());
        assert!(validate_url("http://169.254.169.254/latest/meta-data/").is_err());
        assert!(validate_url("http://2130706433/").is_err());
        assert!(validate_url("http://[::ffff:127.0.0.1]/").is_err());
        assert!(validate_url("https://example.com/").is_ok());
    }
}
