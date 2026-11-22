//! Opt-in SSRF-safe `WebFetch` module (behind Cargo feature `web`).
//!
//! Hardened per Opus architectural checklist:
//!   - Custom `reqwest::dns::Resolve` validating EVERY resolved IP at TCP connect time (zero DNS rebinding!)
//!   - Blocks 0/8, 10/8, 100.64/10, 127/8, 169.254/16, 172.16/12, 192.168/16, multicast/broadcast,
//!     `::1`, `fc00::/7`, `fe80::/10`, `::ffff:a.b.c.d`, `64:ff9b::/96`, `2002::/16`, `localhost`, `*.local`
//!   - `.no_proxy()` + `redirect::Policy::none()` with manual redirect re-validation (max 5 hops)
//!   - Decompressed body cap (2 MiB — blocks gzip bombs!)
//!   - Fast HTML->clean text extraction + CAS BLAKE3 spill (>4 KiB) + `[external-untrusted-content]` framing
//!   - `ToolEffects::global_read()` so multiple `WebFetch` calls run concurrently in I2 waves!

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

/// Fast zero-dependency HTML-to-clean-text extractor (strips `<!-- comments -->`, `<script>`/`<style>`,
/// HTML tags with exact tag-name boundary matching, decodes common entities, and collapses whitespace).
pub fn html_to_clean_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let bytes = html.as_bytes();
    let mut i = 0;
    let mut in_script_or_style = false;
    let mut last_ws = true;

    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Skip HTML comments (`<!-- ... -->`) even when they contain `>` characters.
            if html[i..].starts_with("<!--") {
                i = html[i + 4..]
                    .find("-->")
                    .map(|r| i + 4 + r + 3)
                    .unwrap_or(bytes.len());
                continue;
            }
            let close = html[i..].find('>').map(|r| i + r).unwrap_or(bytes.len());
            let inner = html[i + 1..close].trim_start();
            let (is_closing, rest) = match inner.strip_prefix('/') {
                Some(r) => (true, r.trim_start()),
                None => (false, inner),
            };
            let tag_name = rest
                .split(|c: char| c.is_ascii_whitespace() || c == '/' || c == '>')
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();

            if !is_closing && matches!(tag_name.as_str(), "script" | "style") {
                in_script_or_style = true;
            } else if is_closing && matches!(tag_name.as_str(), "script" | "style") {
                in_script_or_style = false;
            } else if !in_script_or_style
                && matches!(
                    tag_name.as_str(),
                    "p" | "br"
                        | "div"
                        | "li"
                        | "h1"
                        | "h2"
                        | "h3"
                        | "h4"
                        | "h5"
                        | "h6"
                        | "tr"
                        | "pre"
                        | "blockquote"
                        | "section"
                        | "article"
                )
                && !out.ends_with('\n')
            {
                out.push('\n');
                last_ws = true;
            }
            i = (close + 1).min(bytes.len());
            continue;
        }
        if !in_script_or_style {
            let ch = html[i..].chars().next().unwrap_or(' ');
            if ch.is_whitespace() {
                if !last_ws {
                    out.push(if ch == '\n' { '\n' } else { ' ' });
                    last_ws = true;
                }
            } else {
                out.push(ch);
                last_ws = false;
            }
            i += ch.len_utf8();
        } else {
            i += 1;
        }
    }
    out.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .trim()
        .to_string()
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
        let make_builder = || {
            reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .dns_resolver(Arc::new(WebSsrfResolver))
                .user_agent(concat!("carnot/", env!("CARGO_PKG_VERSION")))
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(30))
        };
        let client = match make_builder().build() {
            Ok(c) => c,
            Err(_) => {
                // Minimal containers (Alpine / distroless) may lack OS CA bundles; fall back to bundled Mozilla roots.
                let certs: Vec<reqwest::Certificate> = webpki_root_certs::TLS_SERVER_ROOT_CERTS
                    .iter()
                    .filter_map(|der| reqwest::Certificate::from_der(der.as_ref()).ok())
                    .collect();
                make_builder()
                    .tls_certs_only(certs)
                    .build()
                    .map_err(|e| CarnotError::Tool {
                        tool: "WebFetch".into(),
                        message: e.to_string(),
                    })?
            }
        };
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
            description: "Fetch a public HTTP/HTTPS URL safely (SSRF-hardened, HTML converted to clean text, large pages spilled to .carnot/h/).".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["url"],
                "properties": {
                    "url": { "type": "string", "description": "Public http:// or https:// URL" }
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
            let clean = if content_type.contains("text/html") {
                html_to_clean_text(&raw_text)
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

/// Opt-in `Module` registering `WebFetch` (with `defer_loading: true` => 0 tokens in static prefix!).
#[derive(Debug)]
pub struct WebFetchModule {
    tool: Arc<dyn Tool>,
}

impl WebFetchModule {
    pub fn new(
        cas: Arc<CasStore>,
        cas_enabled: bool,
        spill_threshold: usize,
        preview_max: usize,
    ) -> Result<Self> {
        let tool = Arc::new(WebFetchTool::new(
            cas,
            cas_enabled,
            spill_threshold,
            preview_max,
        )?);
        Ok(Self { tool })
    }
}

impl Module for WebFetchModule {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![self.tool.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_extractor_strips_scripts_comments_and_respects_tag_boundaries() {
        let html = "<html><head><style>body{color:red}</style><script>alert(1)</script></head><body><!-- hidden > comment --><h1>Hello</h1><p>Carnot&nbsp;&amp;&nbsp;Rust&apos;s <svg><path d=\"M0\"/>inline</svg> <track kind=\"sub\"/>ok</p></body></html>";
        let txt = html_to_clean_text(html);
        assert!(txt.contains("Hello"));
        assert!(txt.contains("Carnot & Rust's inline ok"), "got: {txt:?}");
        assert!(!txt.contains("alert(1)"));
        assert!(!txt.contains("color:red"));
        assert!(!txt.contains("hidden"));
        assert!(!txt.contains("comment"));
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
