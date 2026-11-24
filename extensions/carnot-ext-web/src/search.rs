//! Structured `WebSearch` tool backed by real search APIs (SearXNG, Exa, Tavily, Brave).
//!
//! Zero brittle HTML scraping; declares `defer_loading: true` (0 static prefix tokens) and
//! `ToolEffects::global_read()` (parallelizable in read-only waves).

use carnot_cas::{CasStore, shape_command_output_with_redaction};
use carnot_core::{CarnotError, Result, ToolEffects, ToolOutput, ToolSpec};
use carnot_tools::{BoxFuture, Tool};
use futures::StreamExt;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

const MAX_SEARCH_BODY_BYTES: usize = 1024 * 1024;
const SEARCH_TIMEOUT_SECS: u64 = 25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchBackend {
    Searxng,
    Exa,
    Tavily,
    Brave,
}

impl SearchBackend {
    fn from_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "searxng" => Some(Self::Searxng),
            "exa" => Some(Self::Exa),
            "tavily" => Some(Self::Tavily),
            "brave" => Some(Self::Brave),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Searxng => "searxng",
            Self::Exa => "exa",
            Self::Tavily => "tavily",
            Self::Brave => "brave",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

fn build_api_client() -> Result<reqwest::Client> {
    let make = || {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(SEARCH_TIMEOUT_SECS))
            .user_agent(format!("carnot/{}", env!("CARGO_PKG_VERSION")))
    };
    if let Ok(c) = make().build() {
        return Ok(c);
    }
    let certs: Vec<reqwest::Certificate> = webpki_root_certs::TLS_SERVER_ROOT_CERTS
        .iter()
        .filter_map(|der| reqwest::Certificate::from_der(der.as_ref()).ok())
        .collect();
    make()
        .tls_certs_only(certs)
        .build()
        .map_err(|e| CarnotError::Tool {
            tool: "WebSearch".into(),
            message: format!("failed to build HTTP client: {e}"),
        })
}

async fn read_bounded_json(resp: reqwest::Response) -> std::result::Result<Value, String> {
    let mut stream = resp.bytes_stream();
    let mut buf = Vec::new();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(|e| format!("stream error: {e}"))?;
        if buf.len() + bytes.len() > MAX_SEARCH_BODY_BYTES {
            return Err("search response exceeded 1 MiB safety cap".into());
        }
        buf.extend_from_slice(&bytes);
    }
    serde_json::from_slice(&buf).map_err(|e| format!("invalid JSON from search API: {e}"))
}

pub fn parse_searxng_hits(v: &Value, max_results: usize) -> Vec<SearchHit> {
    v.get("results")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .take(max_results)
                .filter_map(|r| {
                    let url = r.get("url").and_then(Value::as_str)?.trim().to_string();
                    let title = r
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or(&url)
                        .trim()
                        .to_string();
                    let snippet = r
                        .get("content")
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
                .collect()
        })
        .unwrap_or_default()
}

pub fn parse_exa_hits(v: &Value, max_results: usize) -> Vec<SearchHit> {
    v.get("results")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .take(max_results)
                .filter_map(|r| {
                    let url = r.get("url").and_then(Value::as_str)?.trim().to_string();
                    let title = r
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or(&url)
                        .trim()
                        .to_string();
                    let snippet = r
                        .get("highlights")
                        .and_then(Value::as_array)
                        .map(|hl| {
                            hl.iter()
                                .filter_map(Value::as_str)
                                .collect::<Vec<_>>()
                                .join(" … ")
                        })
                        .filter(|s| !s.trim().is_empty())
                        .or_else(|| {
                            r.get("text")
                                .and_then(Value::as_str)
                                .map(|t| t.chars().take(360).collect())
                        })
                        .unwrap_or_default();
                    Some(SearchHit {
                        title,
                        url,
                        snippet,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn parse_tavily_hits(v: &Value, max_results: usize) -> Vec<SearchHit> {
    v.get("results")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .take(max_results)
                .filter_map(|r| {
                    let url = r.get("url").and_then(Value::as_str)?.trim().to_string();
                    let title = r
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or(&url)
                        .trim()
                        .to_string();
                    let snippet = r
                        .get("content")
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
                .collect()
        })
        .unwrap_or_default()
}

pub fn parse_brave_hits(v: &Value, max_results: usize) -> Vec<SearchHit> {
    v.pointer("/web/results")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .take(max_results)
                .filter_map(|r| {
                    let url = r.get("url").and_then(Value::as_str)?.trim().to_string();
                    let title = r
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or(&url)
                        .trim()
                        .to_string();
                    let snippet = r
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
                .collect()
        })
        .unwrap_or_default()
}

fn format_hits_markdown(query: &str, backend: SearchBackend, hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        return format!(
            "[WebSearch ({}) returned 0 results for `{query}`]",
            backend.as_str()
        );
    }
    let mut out = format!(
        "[WebSearch ({}) — {} result(s) for `{query}`]\n",
        backend.as_str(),
        hits.len()
    );
    for (idx, h) in hits.iter().enumerate() {
        out.push_str(&format!("\n{}. [{}]({})\n", idx + 1, h.title, h.url));
        if !h.snippet.is_empty() {
            let clipped: String = h.snippet.chars().take(400).collect();
            out.push_str(&format!("   {clipped}\n"));
        }
    }
    out
}

#[derive(Debug)]
pub struct WebSearchTool {
    client: reqwest::Client,
    cas: Arc<CasStore>,
    cas_enabled: bool,
    spill_threshold: usize,
    preview_max: usize,
    allowed_domains: Vec<String>,
    blocked_domains: Vec<String>,
}

impl WebSearchTool {
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
        Ok(Self {
            client: build_api_client()?,
            cas,
            cas_enabled,
            spill_threshold,
            preview_max,
            allowed_domains,
            blocked_domains,
        })
    }

    fn resolve_backend(
        requested: Option<&str>,
    ) -> std::result::Result<(SearchBackend, String), String> {
        let env_nonempty = |k: &str| {
            std::env::var(k)
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        };
        if let Some(req) = requested.filter(|s| !s.trim().is_empty()) {
            let b = SearchBackend::from_name(req).ok_or_else(|| {
                format!("unsupported WebSearch provider `{req}` (supported: searxng, exa, tavily, brave)")
            })?;
            let val = match b {
                SearchBackend::Searxng => env_nonempty("CARNOT_SEARCH_URL")
                    .ok_or("CARNOT_SEARCH_URL is not set for provider `searxng`")?,
                SearchBackend::Exa => env_nonempty("EXA_API_KEY")
                    .ok_or("EXA_API_KEY is not set for provider `exa`")?,
                SearchBackend::Tavily => env_nonempty("TAVILY_API_KEY")
                    .ok_or("TAVILY_API_KEY is not set for provider `tavily`")?,
                SearchBackend::Brave => env_nonempty("BRAVE_API_KEY")
                    .ok_or("BRAVE_API_KEY is not set for provider `brave`")?,
            };
            return Ok((b, val));
        }

        if let Some(u) = env_nonempty("CARNOT_SEARCH_URL") {
            return Ok((SearchBackend::Searxng, u));
        }
        if let Some(k) = env_nonempty("EXA_API_KEY") {
            return Ok((SearchBackend::Exa, k));
        }
        if let Some(k) = env_nonempty("TAVILY_API_KEY") {
            return Ok((SearchBackend::Tavily, k));
        }
        if let Some(k) = env_nonempty("BRAVE_API_KEY") {
            return Ok((SearchBackend::Brave, k));
        }
        Err(
            "WebSearch requires one of CARNOT_SEARCH_URL (SearXNG), EXA_API_KEY, TAVILY_API_KEY, or BRAVE_API_KEY"
                .into(),
        )
    }
}

impl Tool for WebSearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "WebSearch".into(),
            description: "Search the web via configured API (SearXNG, Exa, Tavily, or Brave) and return concise ranked links + snippets.".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["query"],
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "max_results": { "type": "integer", "description": "Max results (1..10, default 5)" },
                    "provider": { "type": "string", "enum": ["searxng", "exa", "tavily", "brave"] }
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
            let Some(query) = input
                .get("query")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
            else {
                return Ok(ToolOutput::err_text("WebSearch requires non-empty `query`"));
            };
            let max_results = input
                .get("max_results")
                .and_then(Value::as_u64)
                .unwrap_or(5)
                .clamp(1, 10) as usize;
            let requested = input.get("provider").and_then(Value::as_str);

            let (backend, cred) = match Self::resolve_backend(requested) {
                Ok(pair) => pair,
                Err(msg) => return Ok(ToolOutput::err_text(msg)),
            };

            let req_fut = async {
                let resp = match backend {
                    SearchBackend::Searxng => {
                        let base = cred.trim_end_matches('/');
                        let raw_url = if base.ends_with("/search") {
                            base.to_string()
                        } else {
                            format!("{base}/search")
                        };
                        let url = url::Url::parse_with_params(
                            &raw_url,
                            &[("q", query), ("format", "json")],
                        )
                        .map_err(|e| format!("Invalid SearXNG URL: {e}"))?;
                        self.client
                            .get(url)
                            .send()
                            .await
                            .map_err(|e| format!("SearXNG request failed: {e}"))?
                    }
                    SearchBackend::Exa => self
                        .client
                        .post("https://api.exa.ai/search")
                        .header("x-api-key", &cred)
                        .json(&json!({
                            "query": query,
                            "numResults": max_results,
                            "contents": { "highlights": { "numSentences": 3 } }
                        }))
                        .send()
                        .await
                        .map_err(|e| format!("Exa request failed: {e}"))?,
                    SearchBackend::Tavily => self
                        .client
                        .post("https://api.tavily.com/search")
                        .json(&json!({
                            "api_key": cred,
                            "query": query,
                            "max_results": max_results,
                            "search_depth": "basic"
                        }))
                        .send()
                        .await
                        .map_err(|e| format!("Tavily request failed: {e}"))?,
                    SearchBackend::Brave => {
                        let count = max_results.to_string();
                        let url = url::Url::parse_with_params(
                            "https://api.search.brave.com/res/v1/web/search",
                            &[("q", query), ("count", count.as_str())],
                        )
                        .map_err(|e| format!("Invalid Brave URL: {e}"))?;
                        self.client
                            .get(url)
                            .header("X-Subscription-Token", &cred)
                            .header("Accept", "application/json")
                            .send()
                            .await
                            .map_err(|e| format!("Brave request failed: {e}"))?
                    }
                };

                if !resp.status().is_success() {
                    return Err(format!(
                        "{} returned HTTP {}",
                        backend.as_str(),
                        resp.status()
                    ));
                }
                let body = read_bounded_json(resp).await?;
                let hits = match backend {
                    SearchBackend::Searxng => parse_searxng_hits(&body, max_results),
                    SearchBackend::Exa => parse_exa_hits(&body, max_results),
                    SearchBackend::Tavily => parse_tavily_hits(&body, max_results),
                    SearchBackend::Brave => parse_brave_hits(&body, max_results),
                };
                Ok::<_, String>(hits)
            };

            let mut hits = tokio::select! {
                _ = cancel.cancelled() => return Err(CarnotError::Cancelled),
                res = req_fut => match res {
                    Ok(h) => h,
                    Err(msg) => return Ok(ToolOutput::err_text(msg)),
                }
            };

            if !self.allowed_domains.is_empty() || !self.blocked_domains.is_empty() {
                hits.retain(|h| {
                    url::Url::parse(&h.url)
                        .ok()
                        .and_then(|u| u.host_str().map(str::to_string))
                        .is_some_and(|host| {
                            crate::domain_allowed(
                                &host,
                                &self.allowed_domains,
                                &self.blocked_domains,
                            )
                        })
                });
            }

            let rendered = format_hits_markdown(query, backend, &hits);
            let shaped = shape_command_output_with_redaction(
                None,
                &rendered,
                self.cas_enabled.then_some(self.cas.as_ref()),
                self.spill_threshold,
                self.preview_max,
                true,
            )?;
            Ok(ToolOutput::ok_text(shaped.text).with_details(json!({
                "provider": backend.as_str(),
                "count": hits.len(),
                "handle": shaped.spilled.map(|h| h.rel_path),
            })))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_exa_tavily_brave_and_searxng_payloads() {
        let exa = json!({"results": [{"title": "Rust", "url": "https://rust-lang.org", "highlights": ["Fast and safe"]}]});
        assert_eq!(parse_exa_hits(&exa, 5).len(), 1);
        assert_eq!(parse_exa_hits(&exa, 5)[0].snippet, "Fast and safe");

        let tav = json!({"results": [{"title": "Doc", "url": "https://docs.rs", "content": "Crates docs"}]});
        assert_eq!(parse_tavily_hits(&tav, 5)[0].url, "https://docs.rs");

        let brave = json!({"web": {"results": [{"title": "B", "url": "https://b.dev", "description": "desc"}]}});
        assert_eq!(parse_brave_hits(&brave, 5)[0].snippet, "desc");

        let sx = json!({"results": [{"title": "S", "url": "https://s.dev", "content": "sx"}]});
        assert_eq!(parse_searxng_hits(&sx, 5)[0].title, "S");
    }
}
