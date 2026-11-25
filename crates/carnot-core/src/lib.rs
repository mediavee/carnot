//! Leaf vocabulary crate for Carnot.
//!
//! Zero workspace dependencies. Every other crate speaks only these wire and domain types.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use thiserror::Error;

/// Unified error type across the Carnot workspace.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum CarnotError {
    #[error("provider error ({provider}): {message}")]
    Provider {
        provider: String,
        message: String,
        retryable: bool,
        retry_after_ms: Option<u64>,
    },
    #[error("context overflow: {0}")]
    ContextOverflow(String),
    #[error("tool error ({tool}): {message}")]
    Tool { tool: String, message: String },
    #[error("sandbox error: {0}")]
    Sandbox(String),
    #[error("session error: {0}")]
    Session(String),
    #[error("extension error ({extension}): {message}")]
    Extension { extension: String, message: String },
    #[error("prefix cache invariant violated at byte {byte_offset}: {reason}")]
    PrefixInvariantViolation { byte_offset: usize, reason: String },
    #[error("cancelled")]
    Cancelled,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, CarnotError>;

/// Conversation message roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Wire origin metadata tagging provider-bound blocks (`Thinking`, `ToolUse`, `Opaque`)
/// so cross-provider history replay never sends foreign signatures or opaque payloads (400 fix).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentOrigin {
    pub api: String,
    pub provider: String,
    pub model: String,
}

impl ContentOrigin {
    pub fn new(
        api: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            api: api.into(),
            provider: provider.into(),
            model: model.into(),
        }
    }

    pub fn matches(&self, api: &str, provider: &str, model: &str) -> bool {
        self.api == api && self.provider == provider && self.model == model
    }
}

/// Typed content blocks.
///
/// Images are modeled as an explicit variant so CAS text-spill logic NEVER converts
/// image payloads into file handles. `Opaque` preserves provider-specific blocks such as
/// `redacted_thinking`, `OpenAI` encrypted reasoning items, or server tool results verbatim
/// when replaying to the same `(api, provider, model)` origin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    /// Signed thinking block bound to its exact preceding prefix (Anthropic / Gemini / `DeepSeek`).
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        origin: Option<ContentOrigin>,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        /// Set when streaming produced malformed JSON; the agent returns an error `tool_result`
        /// instead of silently executing `{}`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        invalid_json: Option<String>,
        /// Opaque provider metadata bound to this call (e.g. Gemini/Antigravity `thoughtSignature`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_meta: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        origin: Option<ContentOrigin>,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<ToolResultBlock>,
        #[serde(default)]
        is_error: bool,
    },
    Image {
        media_type: String,
        data_base64: String,
    },
    /// Provider-native opaque block (e.g. `redacted_thinking`, `reasoning`, `server_tool_use`)
    /// replayed verbatim ONLY to matching `(api, provider, model)` origin.
    Opaque {
        raw: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        origin: Option<ContentOrigin>,
    },
}

impl ContentBlock {
    pub fn is_thinking_like(&self) -> bool {
        match self {
            Self::Thinking { .. } => true,
            Self::Opaque { raw, .. } => raw
                .get("type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|t| t == "redacted_thinking" || t == "thinking" || t == "reasoning"),
            _ => false,
        }
    }
}

/// Content inside a `ToolResult` — keeps text and images strictly disjoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    Text {
        text: String,
    },
    Image {
        media_type: String,
        data_base64: String,
    },
}

impl ToolResultBlock {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }
}

/// A single conversation message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

/// Content-aware, single-pass token estimator (~2 GB/s, zero allocation).
///
/// Distinguishes ASCII prose (~4.0 B/tok), dense code/JSON/digits/punctuation (~2.4 B/tok),
/// and multi-byte UTF-8 / CJK / accented text (~1.8 B/tok) so compaction and context budgets
/// stay accurate on JSON logs, source code, and non-ASCII prompts before provider calibration.
pub fn estimate_text_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    let mut prose_bytes: u64 = 0;
    let mut dense_bytes: u64 = 0;
    let mut non_ascii_bytes: u64 = 0;
    for &b in text.as_bytes() {
        if b >= 0x80 {
            non_ascii_bytes += 1;
        } else if b.is_ascii_alphabetic() || b == b' ' {
            prose_bytes += 1;
        } else {
            dense_bytes += 1;
        }
    }
    let weighted = prose_bytes
        .saturating_mul(25)
        .saturating_add(dense_bytes.saturating_mul(42))
        .saturating_add(non_ascii_bytes.saturating_mul(55));
    weighted.div_ceil(100).max(1)
}

impl Message {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn system_note(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// Concatenated text content of this message (ignores images & thinking).
    pub fn text_concat(&self) -> String {
        let mut out = String::new();
        for b in &self.content {
            if let ContentBlock::Text { text } = b {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(text);
            }
        }
        out
    }

    pub fn has_text(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, ContentBlock::Text { text } if !text.trim().is_empty()))
    }

    pub fn has_tool_use(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
    }

    pub fn has_tool_result(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
    }

    pub fn has_signed_thinking(&self) -> bool {
        self.content.iter().any(ContentBlock::is_thinking_like)
    }

    /// Returns a clone with all `Thinking`, `redacted_thinking`, and heavy `Image` blocks stripped
    /// (used on compaction so old screenshots aren't re-billed every turn and signatures stay valid).
    #[must_use]
    pub fn without_thinking(&self) -> Self {
        Self {
            role: self.role,
            content: self
                .content
                .iter()
                .filter_map(|b| match b {
                    b if b.is_thinking_like() => None,
                    ContentBlock::Image { media_type, .. } => Some(ContentBlock::Text {
                        text: format!("[compacted image ({media_type})]"),
                    }),
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        let cleaned = content
                            .iter()
                            .map(|tb| match tb {
                                ToolResultBlock::Text { text } => {
                                    ToolResultBlock::Text { text: text.clone() }
                                }
                                ToolResultBlock::Image { media_type, .. } => {
                                    ToolResultBlock::Text {
                                        text: format!("[compacted image ({media_type})]"),
                                    }
                                }
                            })
                            .collect();
                        Some(ContentBlock::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content: cleaned,
                            is_error: *is_error,
                        })
                    }
                    other => Some(other.clone()),
                })
                .collect(),
        }
    }

    /// Rough character-based token estimate (text only; images billed at ~1,500 tok each).
    pub fn estimated_tokens(&self) -> u64 {
        let mut tok = 8u64;
        for b in &self.content {
            match b {
                ContentBlock::Text { text } => tok += estimate_text_tokens(text) + 1,
                ContentBlock::Thinking { thinking, .. } => tok += estimate_text_tokens(thinking) + 1,
                ContentBlock::ToolUse { name, input, .. } => {
                    tok += estimate_text_tokens(name)
                        + estimate_text_tokens(&input.to_string())
                        + 4;
                }
                ContentBlock::ToolResult { content, .. } => {
                    for tb in content {
                        match tb {
                            ToolResultBlock::Text { text } => {
                                tok += estimate_text_tokens(text) + 1;
                            }
                            ToolResultBlock::Image { .. } => tok += 1_500,
                        }
                    }
                }
                ContentBlock::Image { .. } => tok += 1_500,
                ContentBlock::Opaque { raw, .. } => {
                    tok += estimate_text_tokens(&raw.to_string()) + 1;
                }
            }
        }
        tok
    }
}

/// Wire tool declaration sent to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    /// When true, declared with Anthropic `defer_loading: true` alongside server tool search.
    #[serde(default)]
    pub defer_loading: bool,
}

/// Recursively canonicalizes a `serde_json::Value` by sorting all object keys lexicographically.
/// Critical for byte-identical prefix-cache hashing when `serde_json`'s `preserve_order` is active.
pub fn canonicalize_json(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&str> = map.keys().map(String::as_str).collect();
            keys.sort_unstable();
            let mut out = serde_json::Map::with_capacity(map.len());
            for k in keys {
                out.insert(k.to_string(), canonicalize_json(&map[k]));
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(canonicalize_json).collect())
        }
        other => other.clone(),
    }
}

/// Effect declaration used by the effect-based parallel scheduler.
///
/// Enables parallel `Edit`/`Write` on disjoint files, concurrent eager `Read`s & trusted read-only
/// MCP/WebFetch calls (`global_read`), and strict serialization against overlapping writes & `Bash`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolEffects {
    pub reads: Vec<PathBuf>,
    pub writes: Vec<PathBuf>,
    /// True for read-only tools whose scope is unknown/global (e.g. trusted MCP `readOnlyHint`, `WebFetch`).
    /// Runs concurrently with other reads, but conflicts with ANY write or `serial_barrier`.
    #[serde(default)]
    pub global_read: bool,
    /// When true, the tool may run in parallel read-only waves AFTER stream completion, but MUST NOT
    /// launch speculatively mid-stream (e.g. paid `Subagent` calls).
    #[serde(default)]
    pub no_eager: bool,
    /// True for tools with global side effects (`Bash`, untrusted/mutating MCP/extensions).
    pub serial_barrier: bool,
}

impl ToolEffects {
    pub fn read_path(p: PathBuf) -> Self {
        Self {
            reads: vec![p],
            writes: Vec::new(),
            global_read: false,
            no_eager: false,
            serial_barrier: false,
        }
    }

    pub fn global_read() -> Self {
        Self {
            reads: Vec::new(),
            writes: Vec::new(),
            global_read: true,
            no_eager: false,
            serial_barrier: false,
        }
    }

    pub fn global_read_no_eager() -> Self {
        Self {
            reads: Vec::new(),
            writes: Vec::new(),
            global_read: true,
            no_eager: true,
            serial_barrier: false,
        }
    }

    pub fn write_path(p: PathBuf) -> Self {
        Self {
            reads: Vec::new(),
            writes: vec![p],
            global_read: false,
            no_eager: true,
            serial_barrier: false,
        }
    }

    pub fn serial() -> Self {
        Self {
            reads: Vec::new(),
            writes: Vec::new(),
            global_read: false,
            no_eager: true,
            serial_barrier: true,
        }
    }

    /// True if `self` cannot safely run concurrently with `other`.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        if self.serial_barrier || other.serial_barrier {
            return true;
        }
        if (self.global_read && !other.writes.is_empty())
            || (other.global_read && !self.writes.is_empty())
        {
            return true;
        }
        for w in &self.writes {
            if other.writes.contains(w) || other.reads.contains(w) {
                return true;
            }
        }
        for r in &self.reads {
            if other.writes.contains(r) {
                return true;
            }
        }
        false
    }

    pub fn is_read_only(&self) -> bool {
        !self.serial_barrier && self.writes.is_empty()
    }

    pub fn is_eager_safe(&self) -> bool {
        self.is_read_only() && !self.no_eager
    }
}

/// Normalized output returned by any tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub blocks: Vec<ToolResultBlock>,
    pub is_error: bool,
    #[serde(default)]
    pub details: serde_json::Value,
}

impl ToolOutput {
    pub fn ok_text(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![ToolResultBlock::text(text)],
            is_error: false,
            details: serde_json::Value::Null,
        }
    }

    pub fn err_text(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![ToolResultBlock::text(text)],
            is_error: true,
            details: serde_json::Value::Null,
        }
    }

    #[must_use]
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = details;
        self
    }

    pub fn text_only(&self) -> String {
        let mut out = String::new();
        for b in &self.blocks {
            if let ToolResultBlock::Text { text } = b {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(text);
            }
        }
        out
    }

    pub fn has_image(&self) -> bool {
        self.blocks
            .iter()
            .any(|b| matches!(b, ToolResultBlock::Image { .. }))
    }
}

/// Provider-agnostic prompt-cache retention policy.
///
/// Each provider maps these tiers to its native caching primitives:
/// - **Anthropic / `OpenRouter` Claude:** `Short` = `5m` everywhere (1.25× write), `Hybrid` = `1h`
///   on static prefix + Turn-1 and `5m` on rolling conversation turns, `Long` = `1h` everywhere
///   (2.0× write, ideal for interactive sessions with >5 min human pauses).
/// - **`OpenAI` / `OpenAI` Responses:** `Short` = default in-memory retention, `Hybrid` / `Long` =
///   extended `24h` retention (`prompt_cache_retention: "24h"`) with stable `prompt_cache_key`.
/// - **Gemini / `DeepSeek` / Grok / local:** automatic server-side prefix caching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CacheTtl {
    #[serde(alias = "5m", alias = "short5m")]
    Short,
    #[serde(alias = "1h", alias = "24h", alias = "long1h")]
    Long,
    #[default]
    Hybrid,
}

impl CacheTtl {
    /// Parses a provider-agnostic retention name (`short`, `hybrid`, `long`) or duration alias
    /// (`5m`, `1h`, `24h`), case-insensitively.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "short" | "5m" | "short5m" => Some(Self::Short),
            "hybrid" => Some(Self::Hybrid),
            "long" | "1h" | "24h" | "long1h" => Some(Self::Long),
            _ => None,
        }
    }

    /// Canonical wire/display name (`"short"`, `"hybrid"`, `"long"`).
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Short => "short",
            Self::Hybrid => "hybrid",
            Self::Long => "long",
        }
    }

    /// Retention tier applied to static prefix breakpoints (system prompt, tools, Turn-1 anchor).
    #[inline]
    #[must_use]
    pub fn static_prefix_tier(self) -> Self {
        match self {
            Self::Short => Self::Short,
            Self::Long | Self::Hybrid => Self::Long,
        }
    }

    /// Retention tier applied to rolling conversation breakpoints.
    #[inline]
    #[must_use]
    pub fn conversation_tier(self) -> Self {
        match self {
            Self::Short | Self::Hybrid => Self::Short,
            Self::Long => Self::Long,
        }
    }
}

/// Reasoning effort setting passed via Anthropic `output_config.effort` / `OpenAI` `reasoning_effort`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    #[default]
    High,
    Xhigh,
    Max,
}

/// Exact per-model pricing table ($/`MTok`) with cache-efficiency helpers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelPricing {
    pub input_per_mtok: f64,
    pub output_per_mtok: f64,
    pub cache_read_mult: f64,
    pub cache_write_5m_mult: f64,
    pub cache_write_1h_mult: f64,
}

impl ModelPricing {
    pub const OPUS_5_5: Self = Self {
        input_per_mtok: 4.0,
        output_per_mtok: 20.0,
        cache_read_mult: 0.05,
        cache_write_5m_mult: 1.25,
        cache_write_1h_mult: 2.0,
    };

    /// Exact pricing for `claude-opus-5`, `claude-opus-4-8`, `claude-opus-4-7`, `claude-opus-4-6` ($5/$25).
    pub const OPUS_5_0: Self = Self {
        input_per_mtok: 5.0,
        output_per_mtok: 25.0,
        cache_read_mult: 0.10,
        cache_write_5m_mult: 1.25,
        cache_write_1h_mult: 2.0,
    };

    pub const SONNET_5_5: Self = Self {
        input_per_mtok: 2.0,
        output_per_mtok: 10.0,
        cache_read_mult: 0.05,
        cache_write_5m_mult: 1.25,
        cache_write_1h_mult: 2.0,
    };

    pub const HAIKU_5_5: Self = Self {
        input_per_mtok: 0.10,
        output_per_mtok: 0.50,
        cache_read_mult: 0.10,
        cache_write_5m_mult: 1.25,
        cache_write_1h_mult: 2.0,
    };

    /// Long-context (>100K input tokens) pricing tier for `claude-haiku-5-5` ($0.50/$2.50).
    pub const HAIKU_5_5_LONG: Self = Self {
        input_per_mtok: 0.50,
        output_per_mtok: 2.50,
        cache_read_mult: 0.10,
        cache_write_5m_mult: 1.25,
        cache_write_1h_mult: 2.0,
    };

    /// Pricing for `claude-haiku-4-5` ($1/$5 per `MTok`).
    pub const HAIKU_4_5: Self = Self {
        input_per_mtok: 1.0,
        output_per_mtok: 5.0,
        cache_read_mult: 0.10,
        cache_write_5m_mult: 1.25,
        cache_write_1h_mult: 2.0,
    };

    pub const FABLE_5_2: Self = Self {
        input_per_mtok: 8.0,
        output_per_mtok: 40.0,
        cache_read_mult: 0.025,
        cache_write_5m_mult: 1.25,
        cache_write_1h_mult: 2.0,
    };

    /// Pricing for `claude-fable-5` / `claude-fable-5-1` ($10/$50 per `MTok`).
    pub const FABLE_5_0: Self = Self {
        input_per_mtok: 10.0,
        output_per_mtok: 50.0,
        cache_read_mult: 0.025,
        cache_write_5m_mult: 1.25,
        cache_write_1h_mult: 2.0,
    };

    /// Pricing for `claude-mythos-*` ($8/$40 per `MTok`).
    pub const MYTHOS: Self = Self {
        input_per_mtok: 8.0,
        output_per_mtok: 40.0,
        cache_read_mult: 0.05,
        cache_write_5m_mult: 1.25,
        cache_write_1h_mult: 2.0,
    };

    pub const GPT_5_4: Self = Self {
        input_per_mtok: 2.5,
        output_per_mtok: 10.0,
        cache_read_mult: 0.10,
        cache_write_5m_mult: 1.0,
        cache_write_1h_mult: 1.0,
    };

    pub const GEMINI_3_PRO: Self = Self {
        input_per_mtok: 2.0,
        output_per_mtok: 12.0,
        cache_read_mult: 0.10,
        cache_write_5m_mult: 1.0,
        cache_write_1h_mult: 1.0,
    };

    pub const SONNET_4_X: Self = Self {
        input_per_mtok: 3.0,
        output_per_mtok: 15.0,
        cache_read_mult: 0.10,
        cache_write_5m_mult: 1.25,
        cache_write_1h_mult: 2.0,
    };

    /// Exact per-model pricing lookup (fallback when no `providers.json` override matches;
    /// unknown models default to conservative $5/$25 so `subagent_max_cost_usd` never under-counts).
    pub fn for_model(model: &str) -> Self {
        let m = model.to_ascii_lowercase();
        if m.contains("fable-5-2") || m.contains("fable-5.2") {
            Self::FABLE_5_2
        } else if m.contains("fable") {
            Self::FABLE_5_0
        } else if m.contains("mythos") {
            Self::MYTHOS
        } else if m.contains("opus-5-5") || m.contains("opus-5.5") {
            Self::OPUS_5_5
        } else if m.contains("opus") {
            Self::OPUS_5_0
        } else if m.contains("haiku-5-5") || m.contains("haiku-5.5") {
            Self::HAIKU_5_5
        } else if m.contains("haiku") {
            Self::HAIKU_4_5
        } else if m.contains("sonnet-4-5")
            || m.contains("sonnet-4.5")
            || m.contains("sonnet-4-6")
            || m.contains("sonnet-4.6")
        {
            Self::SONNET_4_X
        } else if m.contains("sonnet") {
            Self::SONNET_5_5
        } else if m.contains("gpt-5") || m.contains("gpt-6") || m.contains("o3") || m.contains("o4")
        {
            Self::GPT_5_4
        } else if m.contains("deepseek") {
            Self {
                input_per_mtok: 0.55,
                output_per_mtok: 2.19,
                cache_read_mult: 0.25,
                cache_write_5m_mult: 1.0,
                cache_write_1h_mult: 1.0,
            }
        } else if m.contains("grok") {
            Self {
                input_per_mtok: 2.0,
                output_per_mtok: 6.0,
                cache_read_mult: 0.25,
                cache_write_5m_mult: 1.0,
                cache_write_1h_mult: 1.0,
            }
        } else if m.contains("muse") {
            Self {
                input_per_mtok: 1.25,
                output_per_mtok: 4.25,
                cache_read_mult: 0.12,
                cache_write_5m_mult: 1.0,
                cache_write_1h_mult: 1.0,
            }
        } else if m.contains("mistral")
            || m.contains("devstral")
            || m.contains("magistral")
            || m.contains("codestral")
        {
            Self {
                input_per_mtok: 0.70,
                output_per_mtok: 2.10,
                cache_read_mult: 0.10,
                cache_write_5m_mult: 1.0,
                cache_write_1h_mult: 1.0,
            }
        } else if m.contains("llama") {
            Self {
                input_per_mtok: 0.25,
                output_per_mtok: 0.75,
                cache_read_mult: 0.25,
                cache_write_5m_mult: 1.0,
                cache_write_1h_mult: 1.0,
            }
        } else if m.contains("gemini") {
            Self::GEMINI_3_PRO
        } else {
            Self::OPUS_5_0
        }
    }

    /// Resolves effective pricing tier accounting for Haiku 5.5 >100K long-context tier ($0.50/$2.50).
    #[must_use]
    pub fn for_context_tokens(self, total_context_tokens: u64) -> Self {
        if (self.input_per_mtok - Self::HAIKU_5_5.input_per_mtok).abs() < 1e-9
            && total_context_tokens > 100_000
        {
            Self::HAIKU_5_5_LONG
        } else {
            self
        }
    }

    #[inline]
    pub fn read_usd(&self, tokens: u64) -> f64 {
        tokens as f64 * self.input_per_mtok * self.cache_read_mult / 1_000_000.0
    }

    #[inline]
    pub fn write_usd(&self, tokens: u64, ttl: CacheTtl) -> f64 {
        let mult = match ttl.conversation_tier() {
            CacheTtl::Short | CacheTtl::Hybrid => self.cache_write_5m_mult,
            CacheTtl::Long => self.cache_write_1h_mult,
        };
        tokens as f64 * self.input_per_mtok * mult / 1_000_000.0
    }

    #[inline]
    pub fn uncached_input_usd(&self, tokens: u64) -> f64 {
        tokens as f64 * self.input_per_mtok / 1_000_000.0
    }

    #[inline]
    pub fn output_usd(&self, tokens: u64) -> f64 {
        tokens as f64 * self.output_per_mtok / 1_000_000.0
    }

    #[inline]
    pub fn breakeven_turns(&self, ttl: CacheTtl) -> f64 {
        let mult = match ttl.conversation_tier() {
            CacheTtl::Short | CacheTtl::Hybrid => self.cache_write_5m_mult,
            CacheTtl::Long => self.cache_write_1h_mult,
        };
        mult / self.cache_read_mult
    }

    #[inline]
    pub fn cache_lifecycle_usd(&self, tokens: u64, remaining_turns: f64, ttl: CacheTtl) -> f64 {
        self.write_usd(tokens, ttl) + self.read_usd(tokens) * remaining_turns
    }
}

/// Estimated token breakdown of the active conversation context across high-level categories.
/// Contains sizes only (never prompt text or tool names).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextBreakdown {
    /// Static system prompt tokens (excluding memory and context files when split).
    pub system: u64,
    /// Declared active tool schema tokens (deferred tools cost 0 until loaded).
    pub tools: u64,
    /// Memory blocks and injected context files / skills.
    pub memory: u64,
    /// User, assistant, thinking, compaction summaries, attachments, and images.
    pub messages: u64,
    /// Tool calls (`ToolUse`) and tool/shell outputs (`ToolResult`).
    pub tool_results: u64,
    /// True while category counts are estimated (~4 chars/token) rather than exact provider counts.
    pub estimated: bool,
}

impl ContextBreakdown {
    pub fn total(&self) -> u64 {
        self.system
            .saturating_add(self.tools)
            .saturating_add(self.memory)
            .saturating_add(self.messages)
            .saturating_add(self.tool_results)
    }

    /// Scales the five category counts proportionally so their sum equals `target_total` exactly
    /// using the largest-remainder method.
    #[must_use]
    pub fn scaled_to(&self, target_total: u64) -> Self {
        let raw = [
            self.system,
            self.tools,
            self.memory,
            self.messages,
            self.tool_results,
        ];
        let raw_sum: u128 = raw.iter().map(|&v| u128::from(v)).sum();
        if target_total == 0 {
            return Self {
                estimated: self.estimated,
                ..Self::default()
            };
        }
        if raw_sum == 0 {
            return Self {
                messages: target_total,
                estimated: self.estimated,
                ..Self::default()
            };
        }

        let target = u128::from(target_total);
        let mut floors = [0u64; 5];
        let mut remainders = [(0u128, 0usize); 5];
        let mut assigned = 0u64;
        for i in 0..5 {
            let num = u128::from(raw[i]) * target;
            let q = (num / raw_sum) as u64;
            let r = num % raw_sum;
            floors[i] = q;
            remainders[i] = (r, i);
            assigned = assigned.saturating_add(q);
        }

        let mut leftover = target_total.saturating_sub(assigned);
        remainders.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        for &(_, idx) in &remainders {
            if leftover == 0 {
                break;
            }
            floors[idx] = floors[idx].saturating_add(1);
            leftover -= 1;
        }

        Self {
            system: floors[0],
            tools: floors[1],
            memory: floors[2],
            messages: floors[3],
            tool_results: floors[4],
            estimated: self.estimated,
        }
    }
}

/// Normalized provider rate-limit window parsed from HTTP response headers
/// (e.g. Anthropic `anthropic-ratelimit-unified-{5h,7d}-{utilization,reset}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RateLimitWindow {
    pub id: String,
    /// Fraction of window consumed in `[0.0, 1.0]` (or percentage when > 1.0).
    pub utilization: f64,
    /// Unix epoch seconds when this window resets, if reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resets_at: Option<f64>,
    /// Window duration in seconds (e.g. `18000.0` for 5h, `604800.0` for 7d).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_seconds: Option<f64>,
}

/// Token usage and exact billed cost for a single model request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_5m_tokens: u64,
    pub cache_write_1h_tokens: u64,
    pub cost_usd: f64,
}

impl Usage {
    pub fn total_context_tokens(&self) -> u64 {
        self.input_tokens
            + self.cache_read_tokens
            + self.cache_write_5m_tokens
            + self.cache_write_1h_tokens
    }

    pub fn compute_cost(&mut self, pricing: ModelPricing) {
        let eff = pricing.for_context_tokens(self.total_context_tokens());
        self.cost_usd = eff.uncached_input_usd(self.input_tokens)
            + eff.output_usd(self.output_tokens)
            + eff.read_usd(self.cache_read_tokens)
            + eff.write_usd(self.cache_write_5m_tokens, CacheTtl::Short)
            + eff.write_usd(self.cache_write_1h_tokens, CacheTtl::Long);
    }

    pub fn add_assign(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
        self.cache_read_tokens += other.cache_read_tokens;
        self.cache_write_5m_tokens += other.cache_write_5m_tokens;
        self.cache_write_1h_tokens += other.cache_write_1h_tokens;
        self.cost_usd += other.cost_usd;
    }
}

/// Live runtime telemetry for the running session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMetrics {
    pub turns: u64,
    pub requests: u64,
    pub context_auc_tokens: u64,
    pub total_usage: Usage,
    pub compactions_applied: u32,
    pub compacted_tokens_saved: u64,
    pub cas_handles_created: u64,
    pub cas_bytes_spilled: u64,
    pub prefix_cache_hits: u64,
    pub prefix_cache_misses: u64,
    #[serde(default)]
    pub cache_breaks: u64,
    #[serde(default)]
    pub cacheable_prefix_tokens: u64,
    #[serde(default)]
    pub retained_prefix_tokens: u64,
    pub eager_tools_started: u64,
    pub parallel_edits_executed: u64,
    pub edit_attempts: u64,
    pub edit_fuzzy_matches: u64,
    pub edit_failures: u64,
    pub semantic_bytes_saved: u64,
}

impl SessionMetrics {
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.total_usage.total_context_tokens();
        if total == 0 {
            0.0
        } else {
            self.total_usage.cache_read_tokens as f64 / total as f64
        }
    }

    /// Fraction of reusable prefix tokens from prior turns actually served from provider cache.
    ///
    /// Unlike raw `cache_hit_rate()`, this metric is independent of static prompt size and turn
    /// count: it excludes brand-new per-turn deltas (which cannot be cached before first sight)
    /// and intentional compaction/rewind resets, while penalizing cache breaks and TTL expirations.
    pub fn prefix_retention_rate(&self) -> f64 {
        if self.cacheable_prefix_tokens == 0 {
            0.0
        } else {
            (self.retained_prefix_tokens as f64 / self.cacheable_prefix_tokens as f64)
                .clamp(0.0, 1.0)
        }
    }

    pub fn record_prefix_retention(&mut self, prev_context_tokens: u64, cache_read_tokens: u64) {
        if prev_context_tokens > 0 {
            self.cacheable_prefix_tokens += prev_context_tokens;
            self.retained_prefix_tokens += cache_read_tokens.min(prev_context_tokens);
        }
    }

    pub fn cache_write_ratio(&self) -> f64 {
        let total = self.total_usage.total_context_tokens();
        if total == 0 {
            0.0
        } else {
            (self.total_usage.cache_write_5m_tokens + self.total_usage.cache_write_1h_tokens) as f64
                / total as f64
        }
    }
}

/// Sequenced event envelope stored in the lossless ring-buffered `EventLog`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub seq: u64,
    #[serde(flatten)]
    pub event: AgentEvent,
}

/// Structured event stream emitted by `carnot-agent`.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum AgentEvent {
    SessionStarted {
        session_id: String,
        model: String,
        prefix_hash: String,
    },
    TurnStart {
        turn: u64,
        context_tokens_est: u64,
    },
    TextDelta {
        delta: String,
    },
    ThinkingDelta {
        delta: String,
    },
    ToolStart {
        call_id: String,
        tool: String,
        eager: bool,
        #[serde(default)]
        input: serde_json::Value,
    },
    ToolUpdate {
        call_id: String,
        message: String,
    },
    ToolEnd {
        call_id: String,
        tool: String,
        is_error: bool,
        duration_ms: u64,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        output_preview: String,
    },
    CompactionApplied {
        mode: String,
        tokens_before: u64,
        tokens_after: u64,
        semantic_llm: bool,
    },
    ModuleAdvisory {
        signal: String,
        message: String,
    },
    PrefixAudit {
        prefix_hash: String,
        stable: bool,
        note: Option<String>,
    },
    /// Emitted when the live runtime prefix-freeze detector observes an unexpected drift in the
    /// static prefix (`system` + `tools` + `native_web`) or reused conversation message prefix.
    CacheBreakDetected {
        turn: u64,
        reason: String,
        expected_hash: String,
        actual_hash: String,
    },
    /// Live fleet worker lifecycle/activity telemetry pushed to the coordinator's `EventLog`
    /// (feeds TUI status bar, `/fleet`, ACP, HTTP SSE, and RPC without polling).
    WorkerUpdated {
        worker: String,
        state: String,
        model: String,
        turns: u64,
        tools_called: u64,
        cost_usd: f64,
        activity: String,
    },
    UsageRecorded {
        turn: u64,
        usage: Usage,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        rate_limits: Vec<RateLimitWindow>,
    },
    TurnEnd {
        turn: u64,
        stop_reason: StopReason,
    },
    RunSettled {
        final_text: String,
        metrics: SessionMetrics,
    },
    /// Emitted when a subscriber's `after_seq` fell behind the 4,096-event ring buffer.
    EventsTruncated {
        dropped_through_seq: u64,
    },
    /// Interactive tool permission requested by an attached `PermissionBroker` (RPC/HTTP/UI).
    PermissionRequested {
        request_id: String,
        call_id: String,
        tool: String,
        input: serde_json::Value,
    },
    PermissionResolved {
        request_id: String,
        allowed: bool,
    },
    /// Structured clarifying question(s) raised by the agent via `AskUser`.
    UserQuestionRequested {
        request_id: String,
        questions: Vec<UserQuestion>,
    },
    UserQuestionAnswered {
        request_id: String,
    },
    Error {
        message: String,
    },
}

/// Option for a multiple-choice interactive user question (`AskUser`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserQuestionOption {
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Structured interactive question raised by the agent via `AskUser`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserQuestion {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<String>,
    pub question: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<UserQuestionOption>,
    #[serde(default)]
    pub multi_select: bool,
}

/// Answer returned by the user/host for one `UserQuestion`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserQuestionAnswer {
    pub id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selected: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Complete stop reason taxonomy (Anthropic Opus/Sonnet 5.5 + `OpenAI`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    #[default]
    EndTurn,
    ToolUse,
    MaxTokens,
    Refusal,
    PauseTurn,
    ContextWindowExceeded,
    Aborted,
    Error,
}

/// Presets of Carnot modules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    Minimal,
    Standard,
    #[default]
    Carnot,
}

/// Toggleable switches for every core Carnot module.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleSwitches {
    pub cas_handles: bool,
    pub adaptive_yield: bool,
    pub tx_edit_syntax: bool,
    pub exec_microskills: bool,
    pub prefix_guard: bool,
    pub eager_read: bool,
    pub sandbox: bool,
    pub sandbox_network: bool,
    pub compaction: bool,
    #[serde(default)]
    pub subagent: bool,
}

impl ModuleSwitches {
    pub fn from_profile(profile: Profile) -> Self {
        match profile {
            Profile::Minimal => Self {
                cas_handles: false,
                adaptive_yield: false,
                tx_edit_syntax: false,
                exec_microskills: false,
                prefix_guard: true,
                eager_read: false,
                sandbox: false,
                sandbox_network: false,
                compaction: true,
                subagent: false,
            },
            Profile::Standard | Profile::Carnot => Self {
                cas_handles: true,
                adaptive_yield: true,
                tx_edit_syntax: true,
                exec_microskills: true,
                prefix_guard: true,
                eager_read: true,
                sandbox: true,
                sandbox_network: false,
                compaction: true,
                subagent: false,
            },
        }
    }
}

impl Default for ModuleSwitches {
    fn default() -> Self {
        Self::from_profile(Profile::Carnot)
    }
}

/// Provider-native server-side web tool capabilities (populated declaratively from `providers.json`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderWebCapabilities {
    /// Provider supports native server-side web search (Anthropic `web_search`, `OpenAI` Responses
    /// `web_search`, Gemini `googleSearch`).
    pub native_search: bool,
    /// Provider supports native server-side URL fetch (Anthropic `web_fetch`, `OpenAI` Responses
    /// `web_search`, Gemini `urlContext`).
    pub native_fetch: bool,
}

/// Configuration for provider-native server-side web search & fetch tools.
///
/// Pure data in `carnot-core`; populated by `carnot-ext-web` (or directly via `CarnotConfig`)
/// and serialized cleanly by each protocol adapter in `carnot-ai`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeWebConfig {
    #[serde(default)]
    pub search: bool,
    #[serde(default)]
    pub fetch: bool,
    /// Maximum server-side web tool invocations per turn (e.g. Anthropic `max_uses`; default 5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,
    /// Token cap for server-side fetched pages (e.g. Anthropic `max_content_tokens`; default 12 000).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_content_tokens: Option<u32>,
    /// Optional hostname allowlist (no scheme, e.g. `docs.rs`). Takes precedence over `blocked_domains`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_domains: Vec<String>,
    /// Optional hostname blocklist (no scheme, e.g. `example.com`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked_domains: Vec<String>,
    /// Optional search context size hint (`"low"`, `"medium"`, `"high"` — `OpenAI` Responses).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<String>,
}

impl NativeWebConfig {
    pub fn is_enabled(&self) -> bool {
        self.search || self.fetch
    }
}

/// Top-level runtime configuration for a Carnot session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarnotConfig {
    pub workspace_root: PathBuf,
    pub extra_writable_roots: Vec<PathBuf>,
    pub session_dir: PathBuf,
    pub model: String,
    pub provider: String,
    pub effort: ReasoningEffort,
    pub cache_ttl: CacheTtl,
    pub max_turns: u64,
    /// Optional hard USD spend cap for root sessions (circuit-breaker for CI/production).
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
    pub max_output_tokens: u32,
    /// Max consecutive automatic continuations when a turn hits `StopReason::MaxTokens`.
    #[serde(default = "default_max_tokens_continuations")]
    pub max_tokens_continuations: u8,
    pub context_window: u64,
    pub compaction_reserve_tokens: u64,
    /// Optional soft compaction threshold decoupled from physical `context_window`
    /// (prevents attention rot & quadratic cost on 200+ turn sessions; `None` = hard-limit only).
    #[serde(default = "default_soft_compaction_tokens")]
    pub soft_compaction_tokens: Option<u64>,
    /// Minimum turns between proactive soft compactions so prompt cache stays warm (>96%).
    #[serde(default = "default_compaction_cooldown_turns")]
    pub compaction_cooldown_turns: u64,
    /// Token budget for verbatim recent messages preserved after compaction.
    #[serde(default = "default_compaction_tail_budget_tokens")]
    pub compaction_tail_budget_tokens: u64,
    pub cas_spill_threshold_bytes: usize,
    pub cas_preview_max_bytes: usize,
    /// Redact well-known credentials/tokens BEFORE spilling tool outputs to `.carnot/h/` on disk.
    #[serde(default = "default_true")]
    pub redact_cas_secrets: bool,
    pub bash_yield_ms: u64,
    pub bash_max_timeout_ms: u64,
    pub idle_turn_circuit_breaker: u32,
    pub autonomous_gates: Vec<String>,
    pub modules: ModuleSwitches,
    pub extra_env: BTreeMap<String, String>,
    /// Sensitive env var names (populated from provider catalog) to scrub from unsandboxed child shells.
    #[serde(default)]
    pub scrubbed_env_vars: Vec<String>,
    /// Explicit `.mcp.json` config file paths (user-level or explicitly approved).
    #[serde(default)]
    pub mcp_config_paths: Vec<PathBuf>,
    /// Explicit Agent Skills directories containing `<skill>/SKILL.md` (repeatable).
    #[serde(default)]
    pub skills_dirs: Vec<PathBuf>,
    /// Explicit provider catalog JSON override paths (`providers.json`).
    #[serde(default)]
    pub provider_config_paths: Vec<PathBuf>,
    /// Must be explicitly opted-in to execute workspace-local `.mcp.json` (blocks `MCPoison` RCE).
    #[serde(default)]
    pub trust_workspace_mcp: bool,
    /// Explicit drop-in file extension paths (`--ext <file>`) to load via JSON-RPC stdio (`carnot-ext/1`).
    #[serde(default)]
    pub extension_paths: Vec<PathBuf>,
    /// Must be explicitly opted-in to execute workspace-local `.carnot/extensions/` (blocks repo RCE).
    #[serde(default)]
    pub trust_workspace_extensions: bool,
    /// Provider-native server-side web search / fetch configuration (wired by `carnot-ext-web` or SDK).
    #[serde(default)]
    pub native_web: NativeWebConfig,
    /// Register the deferred interactive `AskUser` tool (`defer_loading: true`, 0 static tokens).
    /// Enabled automatically on interactive surfaces (ACP, RPC, HTTP, REPL); false in headless `-p`.
    #[serde(default)]
    pub enable_ask_user: bool,
    /// Maximum subagent recursion depth (0 = root agent only; default 1 = root -> child).
    #[serde(default)]
    pub subagent_depth: u8,
    pub subagent_max_depth: u8,
    pub subagent_max_cost_usd: f64,
    /// Optional replacement for the default static system prompt (frozen before Turn 1).
    #[serde(default)]
    pub static_system_prompt_override: Option<String>,
    /// Optional static section appended to the system prompt (frozen before Turn 1).
    #[serde(default)]
    pub static_system_prompt_append: Option<String>,
    /// Tool names (case-insensitive) excluded from `ToolRegistry` before Turn 1.
    #[serde(default)]
    pub excluded_tools: Vec<String>,
    /// Load workspace/global `AGENTS.md`/`CLAUDE.md` in Turn-1 micro-skills (default `true`).
    #[serde(default = "default_true")]
    pub load_context_files: bool,
    /// Load workspace/extra `SKILL.md` directories in Turn-1 micro-skills (default `true`).
    #[serde(default = "default_true")]
    pub load_skills: bool,
}

fn default_true() -> bool {
    true
}

fn default_max_tokens_continuations() -> u8 {
    4
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default attribute requires fn() -> Option<u64>"
)]
fn default_soft_compaction_tokens() -> Option<u64> {
    Some(220_000)
}

fn default_compaction_cooldown_turns() -> u64 {
    30
}

fn default_compaction_tail_budget_tokens() -> u64 {
    8_000
}

impl CarnotConfig {
    pub fn new(workspace_root: PathBuf) -> Self {
        let session_dir = std::env::var_os("HOME").map_or_else(
            || workspace_root.join(".carnot").join("sessions"),
            |h| PathBuf::from(h).join(".carnot").join("sessions"),
        );
        Self {
            workspace_root,
            extra_writable_roots: Vec::new(),
            session_dir,
            model: "claude-sonnet-5-5".to_string(),
            provider: "anthropic".to_string(),
            effort: ReasoningEffort::High,
            cache_ttl: CacheTtl::Hybrid,
            max_turns: 500,
            max_cost_usd: None,
            max_output_tokens: 32_768,
            max_tokens_continuations: default_max_tokens_continuations(),
            context_window: 200_000,
            compaction_reserve_tokens: 16_384,
            soft_compaction_tokens: default_soft_compaction_tokens(),
            compaction_cooldown_turns: default_compaction_cooldown_turns(),
            compaction_tail_budget_tokens: default_compaction_tail_budget_tokens(),
            cas_spill_threshold_bytes: 8_192,
            cas_preview_max_bytes: 3_200,
            redact_cas_secrets: true,
            bash_yield_ms: 60_000,
            bash_max_timeout_ms: 3_600_000,
            idle_turn_circuit_breaker: 3,
            autonomous_gates: Vec::new(),
            modules: ModuleSwitches::default(),
            extra_env: BTreeMap::new(),
            scrubbed_env_vars: Vec::new(),
            mcp_config_paths: Vec::new(),
            skills_dirs: Vec::new(),
            provider_config_paths: Vec::new(),
            trust_workspace_mcp: false,
            extension_paths: Vec::new(),
            trust_workspace_extensions: false,
            native_web: NativeWebConfig::default(),
            enable_ask_user: false,
            subagent_depth: 0,
            subagent_max_depth: 1,
            subagent_max_cost_usd: 2.0,
            static_system_prompt_override: None,
            static_system_prompt_append: None,
            excluded_tools: Vec::new(),
            load_context_files: true,
            load_skills: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pricing_ratios_match_anthropic_cache_breakeven_table() {
        assert!((ModelPricing::OPUS_5_5.breakeven_turns(CacheTtl::Short) - 25.0).abs() < 1e-6);
        assert!((ModelPricing::OPUS_5_5.breakeven_turns(CacheTtl::Long) - 40.0).abs() < 1e-6);
        assert!((ModelPricing::FABLE_5_2.breakeven_turns(CacheTtl::Short) - 50.0).abs() < 1e-6);
    }

    #[test]
    fn usage_cost_computation_is_exact() {
        let mut u = Usage {
            input_tokens: 1_000,
            output_tokens: 1_000,
            reasoning_tokens: 200,
            cache_read_tokens: 100_000,
            cache_write_5m_tokens: 10_000,
            cache_write_1h_tokens: 10_000,
            cost_usd: 0.0,
        };
        u.compute_cost(ModelPricing::OPUS_5_5);
        assert!((u.cost_usd - 0.174).abs() < 1e-9);
    }

    #[test]
    fn tool_effects_allow_parallel_disjoint_writes_and_block_overlaps() {
        let wa = ToolEffects::write_path(PathBuf::from("/ws/a.rs"));
        let wb = ToolEffects::write_path(PathBuf::from("/ws/b.rs"));
        let ra = ToolEffects::read_path(PathBuf::from("/ws/a.rs"));
        assert!(!wa.conflicts_with(&wb));
        assert!(wa.conflicts_with(&ra));
        assert!(!ra.conflicts_with(&wb));
        assert!(wa.conflicts_with(&ToolEffects::serial()));
    }

    #[test]
    fn context_breakdown_scales_to_exact_total() {
        let raw = ContextBreakdown {
            system: 100,
            tools: 250,
            memory: 75,
            messages: 400,
            tool_results: 175,
            estimated: true,
        };
        for target in [0u64, 1, 7, 999, 10_000, 123_457] {
            let scaled = raw.scaled_to(target);
            assert_eq!(scaled.total(), target);
            assert!(scaled.estimated);
        }
        let zero = ContextBreakdown {
            estimated: true,
            ..ContextBreakdown::default()
        };
        assert_eq!(zero.scaled_to(500).total(), 500);
        assert_eq!(zero.scaled_to(500).messages, 500);
    }

    #[test]
    fn cache_ttl_parses_provider_agnostic_names_and_duration_aliases() {
        assert_eq!(CacheTtl::parse("short"), Some(CacheTtl::Short));
        assert_eq!(CacheTtl::parse("5m"), Some(CacheTtl::Short));
        assert_eq!(CacheTtl::parse("HYBRID"), Some(CacheTtl::Hybrid));
        assert_eq!(CacheTtl::parse("long"), Some(CacheTtl::Long));
        assert_eq!(CacheTtl::parse("1h"), Some(CacheTtl::Long));
        assert_eq!(CacheTtl::parse("24h"), Some(CacheTtl::Long));
        assert_eq!(CacheTtl::parse("invalid"), None);
        assert_eq!(
            serde_json::from_str::<CacheTtl>("\"5m\"").expect("alias 5m"),
            CacheTtl::Short
        );
        assert_eq!(
            serde_json::from_str::<CacheTtl>("\"24h\"").expect("alias 24h"),
            CacheTtl::Long
        );
    }

    #[test]
    fn prefix_retention_tracks_reusable_prefix_independently_of_new_deltas() {
        let mut m = SessionMetrics::default();
        assert_eq!(m.prefix_retention_rate(), 0.0);
        // Turn 2: 10k previous tokens, 10k read from cache (100% retention even if 2k new tokens were written)
        m.record_prefix_retention(10_000, 10_000);
        assert!((m.prefix_retention_rate() - 1.0).abs() < 1e-9);
        // Turn 3: 12k previous tokens, 9k read from cache (partial expiry)
        m.record_prefix_retention(12_000, 9_000);
        // 19k / 22k
        assert!((m.prefix_retention_rate() - (19_000.0 / 22_000.0)).abs() < 1e-9);
    }
}
