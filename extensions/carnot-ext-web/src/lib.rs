//! Official Carnot Rust Extension: hybrid Native + Local Web Search & Fetch (`WebModule`).
//!
//! Supports three cleanly separated modes:
//!   - **`WebMode::Auto` (default):** negotiates with the active provider's `ProviderWebCapabilities`
//!     (`providers.json`). Uses zero-key native server tools (`web_search` / `web_fetch` on Anthropic,
//!     `web_search` on OpenAI Responses, `googleSearch` / `urlContext` on Gemini 3) whenever supported,
//!     and seamlessly falls back to local SSRF-hardened `WebFetch` and API-backed `WebSearch` for any
//!     capability not supported natively — with zero duplicate tools!
//!   - **`WebMode::Native`:** requires provider-native server tools (fails fast if unsupported).
//!   - **`WebMode::Local`:** always uses in-process Rust `WebFetch` (SSRF-safe via `carnot_mcp::ssrf`,
//!     `Accept: text/markdown` negotiation, HTML→Markdown/Text, 2 MiB cap, pre-spill CAS redaction)
//!     and `WebSearch` (SearXNG/Exa/Tavily/Brave), with `defer_loading: true` (0 static prefix tokens)
//!     and `ToolEffects::global_read()` (parallel read-only waves).

pub mod html;
pub mod search;

use carnot_cas::{CasStore, shape_command_output_with_redaction};
use carnot_core::{
    CarnotError, NativeWebConfig, ProviderWebCapabilities, Result, ToolEffects, ToolOutput,
    ToolSpec,
};
use carnot_ext::Module;
use carnot_mcp::ssrf::{build_ssrf_safe_client, validate_public_url};
use carnot_tools::{BoxFuture, Tool};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

const MAX_REDIRECTS: usize = 5;
const MAX_DECOMPRESSED_BYTES: usize = 2 * 1024 * 1024;
const FETCH_TIMEOUT_SECS: u64 = 30;

/// Execution mode for `WebModule` (native provider server tools vs local Rust tools).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WebMode {
    /// Automatically prefer native provider server tools when supported; fall back to local tools.
    #[default]
    Auto,
    /// Require native provider server tools only (error if provider supports neither search nor fetch).
    Native,
    /// Always use local in-process Rust `WebFetch` + `WebSearch` (never send native server tools).
    Local,
}

impl fmt::Display for WebMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Native => f.write_str("native"),
            Self::Local => f.write_str("local"),
        }
    }
}

impl FromStr for WebMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "native" => Ok(Self::Native),
            "local" => Ok(Self::Local),
            other => Err(format!(
                "invalid web mode `{other}` (expected: auto, native, local)"
            )),
        }
    }
}

/// Configuration for `WebModule`, uniformly enforced across native server tools and local tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebConfig {
    #[serde(default)]
    pub mode: WebMode,
    #[serde(default = "default_true")]
    pub search: bool,
    #[serde(default = "default_true")]
    pub fetch: bool,
    #[serde(default)]
    pub max_uses: Option<u32>,
    #[serde(default)]
    pub max_content_tokens: Option<u32>,
    #[serde(default)]
    pub allowed_domains: Vec<String>,
    #[serde(default)]
    pub blocked_domains: Vec<String>,
    #[serde(default)]
    pub search_context_size: Option<String>,
}

fn default_true() -> bool {
    true
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            mode: WebMode::Auto,
            search: true,
            fetch: true,
            max_uses: Some(5),
            max_content_tokens: Some(12_000),
            allowed_domains: Vec::new(),
            blocked_domains: Vec::new(),
            search_context_size: None,
        }
    }
}

/// Checks whether a hostname satisfies `allowed_domains` / `blocked_domains` rules.
///
/// Matches exact hostnames or dot-bounded subdomains (so `docs.rs` matches `api.docs.rs`
/// but NEVER `evildocs.rs`). If `allowed_domains` is non-empty, only listed domains pass.
pub fn domain_allowed(host: &str, allowed: &[String], blocked: &[String]) -> bool {
    let h = host.trim().trim_end_matches('.').to_ascii_lowercase();
    if h.is_empty() {
        return false;
    }
    let matches_domain = |rule: &str| {
        let r = rule
            .trim()
            .trim_start_matches("*.")
            .trim_end_matches('.')
            .to_ascii_lowercase();
        !r.is_empty() && (h == r || h.ends_with(&format!(".{r}")))
    };
    if !allowed.is_empty() {
        return allowed.iter().any(|r| matches_domain(r));
    }
    !blocked.iter().any(|r| matches_domain(r))
}

/// Resolved native + local web wiring returned by [`WebModule::resolve`].
#[derive(Debug)]
pub struct ResolvedWebExtension {
    pub native_web: NativeWebConfig,
    pub module: WebModule,
}

/// In-process Rust `Module` contributing deferred local `WebFetch` and/or `WebSearch` tools
/// whenever not covered natively by the active provider.
#[derive(Debug)]
pub struct WebModule {
    fetch_tool: Option<Arc<WebFetchTool>>,
    search_tool: Option<Arc<search::WebSearchTool>>,
}

impl WebModule {
    /// Creates a purely local `WebModule` with both `WebFetch` and `WebSearch` enabled.
    pub fn new(
        cas: Arc<CasStore>,
        cas_enabled: bool,
        spill_threshold: usize,
        preview_max: usize,
    ) -> Result<Self> {
        let resolved = Self::resolve(
            &WebConfig {
                mode: WebMode::Local,
                ..WebConfig::default()
            },
            ProviderWebCapabilities::default(),
            cas,
            cas_enabled,
            spill_threshold,
            preview_max,
        )?;
        Ok(resolved.module)
    }

    /// Resolves `WebConfig` against the active provider's native capabilities without duplicating tools.
    pub fn resolve(
        cfg: &WebConfig,
        caps: ProviderWebCapabilities,
        cas: Arc<CasStore>,
        cas_enabled: bool,
        spill_threshold: usize,
        preview_max: usize,
    ) -> Result<ResolvedWebExtension> {
        let (use_native_search, use_native_fetch, need_local_search, need_local_fetch) = match cfg
            .mode
        {
            WebMode::Local => (false, false, cfg.search, cfg.fetch),
            WebMode::Native => {
                let ns = cfg.search && caps.native_search;
                let nf = cfg.fetch && caps.native_fetch;
                if (cfg.search || cfg.fetch) && !ns && !nf {
                    return Err(CarnotError::Tool {
                        tool: "web".into(),
                        message: "web mode `native` requested, but the active provider/model does not declare native web search or fetch support (use `--web-mode auto` or `--web-mode local`, or enable `supports_native_web_search`/`supports_native_web_fetch` in providers.json)".into(),
                    });
                }
                (ns, nf, false, false)
            }
            WebMode::Auto => {
                let ns = cfg.search && caps.native_search;
                let nf = cfg.fetch && caps.native_fetch;
                (ns, nf, cfg.search && !ns, cfg.fetch && !nf)
            }
        };

        let native_web = NativeWebConfig {
            search: use_native_search,
            fetch: use_native_fetch,
            max_uses: cfg.max_uses,
            max_content_tokens: cfg.max_content_tokens,
            allowed_domains: cfg.allowed_domains.clone(),
            blocked_domains: cfg.blocked_domains.clone(),
            search_context_size: cfg.search_context_size.clone(),
        };

        let fetch_tool = if need_local_fetch {
            Some(Arc::new(WebFetchTool::with_domain_filters(
                Arc::clone(&cas),
                cas_enabled,
                spill_threshold,
                preview_max,
                cfg.allowed_domains.clone(),
                cfg.blocked_domains.clone(),
            )?))
        } else {
            None
        };

        let search_tool = if need_local_search {
            Some(Arc::new(search::WebSearchTool::with_domain_filters(
                cas,
                cas_enabled,
                spill_threshold,
                preview_max,
                cfg.allowed_domains.clone(),
                cfg.blocked_domains.clone(),
            )?))
        } else {
            None
        };

        Ok(ResolvedWebExtension {
            native_web,
            module: Self {
                fetch_tool,
                search_tool,
            },
        })
    }
}

impl Module for WebModule {
    fn name(&self) -> &str {
        "web"
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        let mut out: Vec<Arc<dyn Tool>> = Vec::new();
        if let Some(ref f) = self.fetch_tool {
            out.push(Arc::clone(f) as Arc<dyn Tool>);
        }
        if let Some(ref s) = self.search_tool {
            out.push(Arc::clone(s) as Arc<dyn Tool>);
        }
        out
    }
}

#[derive(Debug)]
pub struct WebFetchTool {
    client: reqwest::Client,
    cas: Arc<CasStore>,
    cas_enabled: bool,
    spill_threshold: usize,
    preview_max: usize,
    allowed_domains: Vec<String>,
    blocked_domains: Vec<String>,
}

impl WebFetchTool {
    pub fn new(
        cas: Arc<CasStore>,
        cas_enabled: bool,
        spill_threshold: usize,
        preview_max: usize,
    ) -> Result<Self> {
        Self::with_domain_filters(
            cas,
            cas_enabled,
            spill_threshold,
            preview_max,
            Vec::new(),
            Vec::new(),
        )
    }

    pub fn with_domain_filters(
        cas: Arc<CasStore>,
        cas_enabled: bool,
        spill_threshold: usize,
        preview_max: usize,
        allowed_domains: Vec<String>,
        blocked_domains: Vec<String>,
    ) -> Result<Self> {
        let client = build_ssrf_safe_client(FETCH_TIMEOUT_SECS).map_err(|e| CarnotError::Tool {
            tool: "WebFetch".into(),
            message: e,
        })?;
        Ok(Self {
            client,
            cas,
            cas_enabled,
            spill_threshold,
            preview_max,
            allowed_domains,
            blocked_domains,
        })
    }

    fn check_domain(&self, url: &url::Url) -> std::result::Result<(), String> {
        if self.allowed_domains.is_empty() && self.blocked_domains.is_empty() {
            return Ok(());
        }
        let host = url.host_str().unwrap_or("");
        if domain_allowed(host, &self.allowed_domains, &self.blocked_domains) {
            Ok(())
        } else {
            Err(format!(
                "domain `{host}` is not permitted by web domain policy"
            ))
        }
    }
}

impl Tool for WebFetchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "WebFetch".into(),
            description: "Fetch a public HTTP(S) URL (SSRF-safe, negotiates text/markdown, converts HTML to clean Markdown or plain text, large pages spilled to CAS).".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["url"],
                "properties": {
                    "url": { "type": "string", "description": "Public http:// or https:// URL" },
                    "format": { "type": "string", "enum": ["markdown", "text", "raw"], "description": "Output format (default: markdown)" },
                    "raw": { "type": "boolean", "description": "Shorthand for format='raw'" },
                    "max_chars": { "type": "integer", "minimum": 256, "maximum": 100000, "description": "Optional character cap on returned text" }
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
            let Some(raw_url) = input.get("url").and_then(Value::as_str).map(str::trim) else {
                return Ok(ToolOutput::err_text("WebFetch requires `url`"));
            };
            let format = if input.get("raw").and_then(Value::as_bool) == Some(true) {
                "raw"
            } else {
                input
                    .get("format")
                    .and_then(Value::as_str)
                    .unwrap_or("markdown")
            };
            let max_chars = input
                .get("max_chars")
                .and_then(Value::as_u64)
                .map(|n| n.clamp(256, 100_000) as usize);

            let mut current = match validate_public_url(raw_url) {
                Ok(u) => u,
                Err(msg) => return Ok(ToolOutput::err_text(msg)),
            };
            if let Err(msg) = self.check_domain(&current) {
                return Ok(ToolOutput::err_text(msg));
            }

            let accept_header = match format {
                "raw" => "text/html, application/json;q=0.9, text/plain;q=0.8, */*;q=0.5",
                "text" => {
                    "text/plain, text/markdown;q=0.9, text/html;q=0.8, application/json;q=0.7"
                }
                _ => "text/markdown, text/html;q=0.9, application/json;q=0.8, text/plain;q=0.7",
            };

            let fetch_fut = async {
                let mut resp = None;
                for hop in 0..=MAX_REDIRECTS {
                    let r = self
                        .client
                        .get(current.clone())
                        .header(
                            "User-Agent",
                            format!("carnot/{} (coding-agent)", env!("CARGO_PKG_VERSION")),
                        )
                        .header("Accept", accept_header)
                        .send()
                        .await
                        .map_err(|e| format!("WebFetch request failed: {e}"))?;

                    if r.status().is_redirection() {
                        if hop == MAX_REDIRECTS {
                            return Err(format!("too many redirects (>{MAX_REDIRECTS})"));
                        }
                        let Some(loc) = r
                            .headers()
                            .get(reqwest::header::LOCATION)
                            .and_then(|v| v.to_str().ok())
                        else {
                            return Err(format!("redirect {} missing Location header", r.status()));
                        };
                        let next_url = current
                            .join(loc)
                            .map_err(|e| format!("invalid redirect Location `{loc}`: {e}"))?;
                        current = validate_public_url(next_url.as_str())?;
                        self.check_domain(&current)?;
                        continue;
                    }
                    resp = Some(r);
                    break;
                }

                let r = resp.ok_or_else(|| "no HTTP response received".to_string())?;
                let status = r.status();
                let content_type = r
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_ascii_lowercase();

                if !status.is_success() {
                    return Err(format!("HTTP {status} fetching {current}"));
                }

                let mut buf = Vec::new();
                let mut truncated_cap = false;
                let mut stream = r.bytes_stream();
                while let Some(chunk) = stream.next().await {
                    let bytes = chunk.map_err(|e| format!("stream read failed: {e}"))?;
                    if buf.len() + bytes.len() > MAX_DECOMPRESSED_BYTES {
                        let take = MAX_DECOMPRESSED_BYTES.saturating_sub(buf.len());
                        buf.extend_from_slice(&bytes[..take]);
                        truncated_cap = true;
                        break;
                    }
                    buf.extend_from_slice(&bytes);
                }

                if buf[..buf.len().min(4096)].contains(&0) {
                    return Err(format!(
                        "refusing binary response ({content_type}, {} bytes) from {current}",
                        buf.len()
                    ));
                }

                let text = String::from_utf8_lossy(&buf).into_owned();
                let is_html = content_type.contains("text/html")
                    || content_type.contains("application/xhtml")
                    || text.trim_start().starts_with("<!DOCTYPE html")
                    || text.trim_start().starts_with("<html");
                let body = if format == "raw" || !is_html {
                    text
                } else if format == "text" {
                    html::html_to_text(&text)
                } else {
                    html::html_to_markdown(&text, Some(&current))
                };
                Ok::<_, String>((current.to_string(), content_type, body, truncated_cap))
            };

            let (final_url, content_type, mut body, truncated_cap) = tokio::select! {
                _ = cancel.cancelled() => return Err(CarnotError::Cancelled),
                res = fetch_fut => match res {
                    Ok(v) => v,
                    Err(msg) => return Ok(ToolOutput::err_text(msg)),
                }
            };

            if let Some(limit) = max_chars
                && body.chars().count() > limit
            {
                body = body.chars().take(limit).collect();
                body.push_str("\n\n[... truncated to requested max_chars ...]");
            }
            if truncated_cap {
                body.push_str("\n\n[... response truncated at 2 MiB safety cap ...]");
            }
            let header = format!("[WebFetch {final_url}]\n\n");
            let full = format!("{header}{body}");
            let shaped = shape_command_output_with_redaction(
                None,
                &full,
                self.cas_enabled.then_some(self.cas.as_ref()),
                self.spill_threshold,
                self.preview_max,
                true,
            )?;

            Ok(ToolOutput::ok_text(shaped.text).with_details(json!({
                "url": final_url,
                "content_type": content_type,
                "format": format,
                "raw_bytes": shaped.raw_bytes,
                "handle": shaped.spilled.map(|h| h.rel_path),
            })))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn web_fetch_blocks_loopback_private_and_file_schemes() {
        let dir = std::env::temp_dir().join(format!("carnot-web-ssrf-{}", uuid::Uuid::new_v4()));
        let cas = Arc::new(CasStore::open(&dir).expect("cas"));
        let tool = WebFetchTool::new(cas, true, 4096, 1024).expect("web fetch tool");

        for bad in [
            "http://127.0.0.1:8080/secret",
            "http://169.254.169.254/latest/meta-data/",
            "http://localhost/admin",
            "file:///etc/passwd",
        ] {
            let out = tool
                .execute(json!({"url": bad}), CancellationToken::new())
                .await
                .expect("execute");
            assert!(out.is_error, "expected SSRF error for {bad}, got: {out:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn domain_policy_enforces_dot_bounded_allow_and_block_rules() {
        let allow = vec!["docs.rs".to_string(), "*.rust-lang.org".to_string()];
        assert!(domain_allowed("docs.rs", &allow, &[]));
        assert!(domain_allowed("api.docs.rs", &allow, &[]));
        assert!(domain_allowed("doc.rust-lang.org", &allow, &[]));
        assert!(!domain_allowed("evildocs.rs", &allow, &[]));
        assert!(!domain_allowed("example.com", &allow, &[]));

        let block = vec!["internal.corp".to_string()];
        assert!(domain_allowed("docs.rs", &[], &block));
        assert!(!domain_allowed("internal.corp", &[], &block));
        assert!(!domain_allowed("wiki.internal.corp", &[], &block));
    }

    #[test]
    fn web_module_resolve_auto_hybrid_native_and_local_modes() {
        let dir = std::env::temp_dir().join(format!("carnot-web-res-{}", uuid::Uuid::new_v4()));
        let cas = Arc::new(CasStore::open(&dir).expect("cas"));

        // 1. Auto with full native caps (e.g. Anthropic / OpenAI Responses / Gemini 3):
        //    0 local fallback tools registered, both native flags enabled!
        let full_native = WebModule::resolve(
            &WebConfig::default(),
            ProviderWebCapabilities {
                native_search: true,
                native_fetch: true,
            },
            Arc::clone(&cas),
            true,
            4096,
            1024,
        )
        .expect("auto full native");
        assert!(full_native.native_web.search);
        assert!(full_native.native_web.fetch);
        assert!(full_native.module.tools().is_empty());

        // 2. Auto with partial native caps (e.g. search-only provider):
        //    native search + local WebFetch fallback (zero duplicate!)
        let partial = WebModule::resolve(
            &WebConfig::default(),
            ProviderWebCapabilities {
                native_search: true,
                native_fetch: false,
            },
            Arc::clone(&cas),
            true,
            4096,
            1024,
        )
        .expect("auto partial");
        assert!(partial.native_web.search);
        assert!(!partial.native_web.fetch);
        let names: Vec<String> = partial
            .module
            .tools()
            .into_iter()
            .map(|t| t.spec().name)
            .collect();
        assert_eq!(names, vec!["WebFetch"]);

        // 3. Native mode on unsupported provider fails fast with clear Config error
        let err = WebModule::resolve(
            &WebConfig {
                mode: WebMode::Native,
                ..WebConfig::default()
            },
            ProviderWebCapabilities::default(),
            Arc::clone(&cas),
            true,
            4096,
            1024,
        )
        .expect_err("native mode must fail when unsupported");
        assert!(err.to_string().contains("native"));

        // 4. Local mode forces local tools even when provider supports native
        let local = WebModule::resolve(
            &WebConfig {
                mode: WebMode::Local,
                ..WebConfig::default()
            },
            ProviderWebCapabilities {
                native_search: true,
                native_fetch: true,
            },
            cas,
            true,
            4096,
            1024,
        )
        .expect("local mode");
        assert!(!local.native_web.is_enabled());
        assert_eq!(local.module.tools().len(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
