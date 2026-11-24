//! Shared wire-protocol compatibility helpers (RFC 7396 JSON merge patch, Retry-After parsing,
//! declarative UsageMap extraction, cross-provider origin matching, and Gemini schema sanitizer).
//!
//! ZERO provider names or URLs live in this module.

use crate::catalog::UsageMap;
use carnot_core::{ContentOrigin, RateLimitWindow, Usage};
use serde_json::{Map, Value};
use std::collections::HashSet;

/// RFC 7396 JSON Merge Patch applied deterministically to a request body.
pub fn json_merge_patch(target: &mut Value, patch: &Value) {
    match (target, patch) {
        (Value::Object(t_map), Value::Object(p_map)) => {
            for (k, v) in p_map {
                if v.is_null() {
                    t_map.remove(k);
                } else if let Some(existing) = t_map.get_mut(k) {
                    json_merge_patch(existing, v);
                } else {
                    t_map.insert(k.clone(), v.clone());
                }
            }
        }
        (t, p) => {
            *t = p.clone();
        }
    }
}

/// Parses `retry-after-ms` or standard HTTP `retry-after` (integer/float seconds) into milliseconds.
pub fn parse_retry_after_ms(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    if let Some(ms) = headers
        .get("retry-after-ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        return Some(ms.ceil() as u64);
    }
    headers
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .map(|secs| (secs * 1000.0).ceil() as u64)
}

/// Parses unified subscription rate-limit headers (e.g. Anthropic's
/// `anthropic-ratelimit-unified-{5h,7d}-{utilization,reset}`) into normalized windows.
pub fn parse_rate_limit_windows(headers: &reqwest::header::HeaderMap) -> Vec<RateLimitWindow> {
    const SPECS: &[(&str, f64)] = &[("5h", 5.0 * 3600.0), ("7d", 7.0 * 86_400.0)];
    let mut windows = Vec::new();
    for &(id, window_seconds) in SPECS {
        let util_key = format!("anthropic-ratelimit-unified-{id}-utilization");
        let Some(utilization) = headers
            .get(&util_key)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
        else {
            continue;
        };
        let reset_key = format!("anthropic-ratelimit-unified-{id}-reset");
        let resets_at = headers
            .get(&reset_key)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0);
        windows.push(RateLimitWindow {
            id: id.to_string(),
            utilization,
            resets_at,
            window_seconds: Some(window_seconds),
        });
    }
    windows
}

/// True if an HTTP status code is transient and safe to retry with backoff.
#[inline]
pub fn is_retryable_status(status: u16) -> bool {
    matches!(status, 408 | 429 | 500 | 502 | 503 | 504 | 529)
}

/// Exhaustive multi-provider context-window overflow classifier (covers HTTP 413, combined
/// input+output ceilings, and 20+ gateway/provider messages while excluding TPM/rate-limit errors).
pub fn is_context_overflow(status: Option<u16>, body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    // Never misclassify rate-limit / TPM throttling as context overflow (would trigger destructive compaction!)
    const NON_OVERFLOW: &[&str] = &[
        "rate limit",
        "rate_limit",
        "throttl",
        "too many requests",
        "tokens per minute",
        "requests per minute",
        "please wait",
        "quota exceeded",
    ];
    if NON_OVERFLOW.iter().any(|p| lower.contains(p)) {
        return false;
    }
    if status == Some(413) {
        return true;
    }
    const PATTERNS: &[&str] = &[
        "context_length_exceeded",
        "context_window_exceeded",
        "prompt is too long",
        "maximum context length",
        "exceeds the context window",
        "exceeds context window",
        "input tokens exceed",
        "input token count",
        "token count exceeds",
        "request payload size exceeds",
        "request_too_large",
        "payload too large",
        "too many tokens",
        "too many input tokens",
        "reduce the input length",
        "reduce the length of the messages",
        "accepts at most",
        "exceeds the maximum number of tokens",
        "max_prompt_tokens",
        "input is too long",
        "prompt exceeds",
        "context length",
        "sequence length",
    ];
    PATTERNS.iter().any(|p| lower.contains(p))
}

/// Attempts strict JSON parsing first, then repairs common LLM tool-argument glitches
/// (raw newlines/tabs inside strings, invalid regex escapes like `\s`/`\d`, trailing commas).
pub fn repair_json(raw: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(raw) {
        return Some(v);
    }
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Some(serde_json::json!({}));
    }

    // Pass 1: escape raw control chars & invalid backslash escapes inside JSON strings
    let mut fixed = String::with_capacity(trimmed.len() + 16);
    let chars: Vec<char> = trimmed.chars().collect();
    let mut i = 0usize;
    let mut in_str = false;
    while i < chars.len() {
        let ch = chars[i];
        if !in_str {
            if ch == '"' {
                in_str = true;
            }
            fixed.push(ch);
            i += 1;
        } else if ch == '\\' {
            if let Some(&next) = chars.get(i + 1) {
                match next {
                    '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {
                        fixed.push('\\');
                        fixed.push(next);
                        i += 2;
                    }
                    'u' if i + 5 < chars.len()
                        && chars[i + 2..i + 6].iter().all(|c| c.is_ascii_hexdigit()) =>
                    {
                        for &c in &chars[i..i + 6] {
                            fixed.push(c);
                        }
                        i += 6;
                    }
                    _ => {
                        // Invalid JSON escape (e.g. `\s`, `\d`, `\w`, `\.`) -> double backslash
                        fixed.push('\\');
                        fixed.push('\\');
                        fixed.push(next);
                        i += 2;
                    }
                }
            } else {
                fixed.push('\\');
                fixed.push('\\');
                i += 1;
            }
        } else if ch == '"' {
            in_str = false;
            fixed.push(ch);
            i += 1;
        } else if ch == '\n' {
            fixed.push_str("\\n");
            i += 1;
        } else if ch == '\r' {
            fixed.push_str("\\r");
            i += 1;
        } else if ch == '\t' {
            fixed.push_str("\\t");
            i += 1;
        } else {
            fixed.push(ch);
            i += 1;
        }
    }

    if let Ok(v) = serde_json::from_str::<Value>(&fixed) {
        return Some(v);
    }

    // Pass 2: strip trailing commas before `}` or `]` outside strings
    let mut no_trailing = String::with_capacity(fixed.len());
    let fchars: Vec<char> = fixed.chars().collect();
    let mut j = 0usize;
    let mut in_s = false;
    let mut esc = false;
    while j < fchars.len() {
        let ch = fchars[j];
        if in_s {
            no_trailing.push(ch);
            if esc {
                esc = false;
            } else if ch == '\\' {
                esc = true;
            } else if ch == '"' {
                in_s = false;
            }
            j += 1;
            continue;
        }
        if ch == '"' {
            in_s = true;
            no_trailing.push(ch);
            j += 1;
            continue;
        }
        if ch == ',' {
            let mut k = j + 1;
            while k < fchars.len() && fchars[k].is_whitespace() {
                k += 1;
            }
            if matches!(fchars.get(k), Some('}' | ']')) {
                j += 1;
                continue;
            }
        }
        no_trailing.push(ch);
        j += 1;
    }
    serde_json::from_str::<Value>(&no_trailing).ok()
}

const BASE62_ALPHABET: &[u8; 62] =
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

fn fnv1a_u128(parts: &[&[u8]]) -> u128 {
    let mut h1: u64 = 0xcbf2_9ce4_8422_2325;
    let mut h2: u64 = 0xaf63_dc4c_8601_ec8c;
    for part in parts {
        for &b in *part {
            h1 ^= u64::from(b);
            h1 = h1.wrapping_mul(0x0000_0100_0000_01b3);
            h2 ^= u64::from(b).wrapping_add(17);
            h2 = h2.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h1 ^= 0xff;
        h1 = h1.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (u128::from(h1) << 64) | u128::from(h2)
}

fn fnv1a_hex(parts: &[&[u8]]) -> String {
    format!("{:032x}", fnv1a_u128(parts))
}

/// Sanitizes foreign tool call IDs to `[a-zA-Z0-9_-]{1,64}` deterministically for cross-provider replay.
pub fn normalize_tool_call_id(raw: &str) -> String {
    if !raw.is_empty()
        && raw.len() <= 64
        && raw
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return raw.to_string();
    }
    let digest = fnv1a_hex(&[raw.as_bytes()]);
    let clean: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .take(40)
        .collect();
    if clean.is_empty() {
        format!("tc_{}", &digest[..24])
    } else {
        format!("{clean}_{}", &digest[..16])
    }
}

/// Formats a tool call ID to satisfy a declarative wire-protocol format constraint.
///
/// - `None`: returns `raw` unchanged.
/// - `Some("alnum9")`: preserves IDs already matching `^[A-Za-z0-9]{9}$`, otherwise deterministically
///   maps any foreign ID into a collision-resistant 9-char base62 identifier (`62^9` space).
/// - `Some("sanitized")`: applies [`normalize_tool_call_id`].
pub fn format_tool_call_id(raw: &str, format: Option<&str>) -> String {
    match format {
        Some("alnum9") => {
            if raw.len() == 9 && raw.bytes().all(|b| b.is_ascii_alphanumeric()) {
                return raw.to_string();
            }
            let mut n = fnv1a_u128(&[b"alnum9:", raw.as_bytes()]);
            let mut out = [0u8; 9];
            for slot in &mut out {
                *slot = BASE62_ALPHABET[(n % 62) as usize];
                n /= 62;
            }
            out.into_iter().map(char::from).collect()
        }
        Some("sanitized") => normalize_tool_call_id(raw),
        _ => raw.to_string(),
    }
}

/// Derives a collision-free per-session routing/cache key combining static `prefix_hash` + `session_id`.
pub fn session_cache_key(prefix_hash: &str, session_id: &str) -> String {
    if session_id.is_empty() {
        let end = prefix_hash.len().min(24);
        return format!("carnot-{}", &prefix_hash[..end]);
    }
    let hex = fnv1a_hex(&[prefix_hash.as_bytes(), session_id.as_bytes()]);
    format!("carnot-{}", &hex[..24])
}

/// True if a `ContentOrigin` belongs to the exact current `(api, provider, model)` tuple.
#[inline]
pub fn origin_matches(
    origin: Option<&ContentOrigin>,
    api: &str,
    provider: &str,
    model: &str,
) -> bool {
    match origin {
        Some(o) => o.matches(api, provider, model),
        None => false,
    }
}

/// True if an origin-tagged block may be replayed to the current `(api, provider, model)`.
/// Legacy untagged (`origin: None`) blocks are only accepted if `allow_untagged` is true.
#[inline]
pub fn can_replay_origin(
    origin: Option<&ContentOrigin>,
    api: &str,
    provider: &str,
    model: &str,
    allow_untagged: bool,
) -> bool {
    match origin {
        Some(o) => o.matches(api, provider, model),
        None => allow_untagged,
    }
}

/// Extracts usage numbers from any JSON object using declarative JSON pointers in `UsageMap`.
pub fn apply_usage_map(usage_json: &Value, map: &UsageMap, out: &mut Usage) {
    let raw_in = usage_json
        .pointer(&map.input_ptr)
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = map
        .cache_read_ptr
        .as_deref()
        .and_then(|ptr| usage_json.pointer(ptr))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_write = map
        .cache_write_ptr
        .as_deref()
        .and_then(|ptr| usage_json.pointer(ptr))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let raw_out = usage_json
        .pointer(&map.output_ptr)
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reasoning = map
        .reasoning_ptr
        .as_deref()
        .and_then(|ptr| usage_json.pointer(ptr))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    out.cache_read_tokens = cached;
    out.cache_write_5m_tokens = cache_write;
    out.input_tokens = if map.cached_included_in_input {
        raw_in.saturating_sub(cached.saturating_add(cache_write))
    } else {
        raw_in
    };
    out.output_tokens = if map.reasoning_included_in_output {
        raw_out
    } else {
        raw_out.saturating_add(reasoning)
    };
    out.reasoning_tokens = reasoning;
}

/// Sanitizes a JSON Schema object into the stricter Gemini / Cloud Code OpenAPI-subset schema:
/// - Uppercases `"type"` strings (`"object"` -> `"OBJECT"`, `"string"` -> `"STRING"`, etc.)
/// - Strips unsupported keywords (`$schema`, `$ref`, `$defs`, `definitions`, `additionalProperties`,
///   `const`, `default`, `examples`, `patternProperties`, `minLength`, `maxLength`, `exclusiveMinimum`, `exclusiveMaximum`)
/// - Converts single-element `type: ["string", "null"]` unions to nullable `"STRING"`
/// - Ensures every `"ARRAY"` schema has an `"items"` schema
/// - Filters `"required"` to keys actually present in `"properties"`
pub fn sanitize_gemini_schema(schema: &Value) -> Value {
    match schema {
        Value::Object(obj) => {
            let mut out = Map::new();
            let mut nullable = false;

            if let Some(const_val) = obj.get("const") {
                if let Some(s) = const_val.as_str() {
                    out.insert("type".into(), Value::String("STRING".into()));
                    out.insert("enum".into(), Value::Array(vec![Value::String(s.into())]));
                } else {
                    out.insert("type".into(), Value::String("STRING".into()));
                }
            }

            if let Some(ty) = obj.get("type") {
                match ty {
                    Value::String(s) => {
                        out.insert("type".into(), Value::String(gemini_type_name(s).into()));
                    }
                    Value::Array(arr) => {
                        let mut chosen = "STRING";
                        for item in arr {
                            if let Some(s) = item.as_str() {
                                if s.eq_ignore_ascii_case("null") {
                                    nullable = true;
                                } else {
                                    chosen = gemini_type_name(s);
                                }
                            }
                        }
                        out.insert("type".into(), Value::String(chosen.into()));
                    }
                    _ => {}
                }
            }

            if nullable {
                out.insert("nullable".into(), Value::Bool(true));
            }
            if let Some(desc) = obj.get("description").and_then(Value::as_str) {
                out.insert("description".into(), Value::String(desc.to_string()));
            }
            if let Some(en) = obj.get("enum").and_then(Value::as_array) {
                let strings: Vec<Value> = en
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| Value::String(s.to_string())))
                    .collect();
                if !strings.is_empty() {
                    out.insert("enum".into(), Value::Array(strings));
                }
            }

            let mut prop_keys: HashSet<String> = HashSet::new();
            if let Some(props) = obj.get("properties").and_then(Value::as_object) {
                let mut new_props = Map::new();
                // Deterministic key order for byte-identical prefix cache!
                let mut keys: Vec<&String> = props.keys().collect();
                keys.sort();
                for k in keys {
                    prop_keys.insert(k.clone());
                    new_props.insert(k.clone(), sanitize_gemini_schema(&props[k]));
                }
                out.insert("properties".into(), Value::Object(new_props));
                out.entry("type".to_string())
                    .or_insert_with(|| Value::String("OBJECT".into()));
            }

            if let Some(req) = obj.get("required").and_then(Value::as_array) {
                let filtered: Vec<Value> = req
                    .iter()
                    .filter_map(|v| v.as_str())
                    .filter(|k| prop_keys.contains(*k))
                    .map(|k| Value::String(k.to_string()))
                    .collect();
                if !filtered.is_empty() {
                    out.insert("required".into(), Value::Array(filtered));
                }
            }

            if let Some(items) = obj.get("items") {
                out.insert("items".into(), sanitize_gemini_schema(items));
            } else if out.get("type").and_then(Value::as_str) == Some("ARRAY") {
                let mut fallback_item = Map::new();
                fallback_item.insert("type".into(), Value::String("STRING".into()));
                out.insert("items".into(), Value::Object(fallback_item));
            }

            if !out.contains_key("type") {
                out.insert("type".into(), Value::String("OBJECT".into()));
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

fn gemini_type_name(raw: &str) -> &'static str {
    match raw.to_ascii_lowercase().as_str() {
        "object" => "OBJECT",
        "array" => "ARRAY",
        "integer" => "INTEGER",
        "number" => "NUMBER",
        "boolean" => "BOOLEAN",
        _ => "STRING",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_overflow_detects_413_and_excludes_rate_limits() {
        assert!(is_context_overflow(Some(413), "payload too large"));
        assert!(is_context_overflow(
            Some(400),
            "This model accepts at most 200000 combined input and output tokens"
        ));
        assert!(is_context_overflow(
            Some(400),
            "input token count exceeds the maximum number of tokens allowed"
        ));
        // TPM / rate-limit errors must NEVER trigger destructive emergency compaction
        assert!(!is_context_overflow(
            Some(429),
            "Throttling error: Too many tokens per minute, please wait"
        ));
        assert!(!is_context_overflow(
            Some(429),
            "rate_limit_error: tokens per minute exceeded"
        ));
    }

    #[test]
    fn repair_json_fixes_regex_escapes_raw_newlines_and_trailing_commas() {
        let broken = "{\n  \"pattern\": \"foo\\s+\\d+\",\n  \"text\": \"line1\nline2\",\n}";
        let v = repair_json(broken).expect("repaired");
        assert_eq!(v["pattern"], r"foo\s+\d+");
        assert_eq!(v["text"], "line1\nline2");
    }

    #[test]
    fn normalize_tool_call_id_sanitizes_and_bounds_foreign_ids() {
        assert_eq!(normalize_tool_call_id("toolu_01Ab-9"), "toolu_01Ab-9");
        let weird = normalize_tool_call_id("call:foo/bar baz!@#");
        assert!(weird.len() <= 64);
        assert!(
            weird
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        );
    }

    #[test]
    fn format_tool_call_id_alnum9_preserves_native_and_maps_foreign_deterministically() {
        assert_eq!(
            format_tool_call_id("aBc123XyZ", Some("alnum9")),
            "aBc123XyZ"
        );
        let mapped1 = format_tool_call_id("toolu_01AbCdEfGhIjKlMnOp", Some("alnum9"));
        let mapped2 = format_tool_call_id("toolu_01AbCdEfGhIjKlMnOp", Some("alnum9"));
        let mapped3 = format_tool_call_id("call_other_id_99", Some("alnum9"));
        assert_eq!(mapped1, mapped2);
        assert_ne!(mapped1, mapped3);
        assert_eq!(mapped1.len(), 9);
        assert!(mapped1.bytes().all(|b| b.is_ascii_alphanumeric()));
        assert_eq!(format_tool_call_id("toolu_01Ab", None), "toolu_01Ab");
    }

    #[test]
    fn session_cache_key_isolates_concurrent_sessions() {
        let a = session_cache_key("same_prefix_hash", "sess-a");
        let b = session_cache_key("same_prefix_hash", "sess-b");
        assert_ne!(a, b);
    }

    #[test]
    fn parse_rate_limit_windows_extracts_anthropic_unified_headers() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "anthropic-ratelimit-unified-5h-utilization",
            "0.42".parse().expect("hdr"),
        );
        headers.insert(
            "anthropic-ratelimit-unified-5h-reset",
            "1790000000".parse().expect("hdr"),
        );
        headers.insert(
            "anthropic-ratelimit-unified-7d-utilization",
            "0.15".parse().expect("hdr"),
        );

        let windows = parse_rate_limit_windows(&headers);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].id, "5h");
        assert!((windows[0].utilization - 0.42).abs() < 1e-9);
        assert_eq!(windows[0].resets_at, Some(1_790_000_000.0));
        assert_eq!(windows[0].window_seconds, Some(18_000.0));
        assert_eq!(windows[1].id, "7d");
        assert!((windows[1].utilization - 0.15).abs() < 1e-9);
        assert_eq!(windows[1].resets_at, None);
        assert_eq!(windows[1].window_seconds, Some(604_800.0));
    }
}
