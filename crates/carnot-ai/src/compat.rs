//! Shared wire-protocol compatibility helpers (RFC 7396 JSON merge patch, Retry-After parsing,
//! declarative UsageMap extraction, cross-provider origin matching, and Gemini schema sanitizer).
//!
//! ZERO provider names or URLs live in this module.

use crate::catalog::UsageMap;
use carnot_core::{ContentOrigin, Usage};
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

/// Parses standard HTTP `retry-after` header (integer seconds) into milliseconds.
pub fn parse_retry_after_ms(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|secs| secs.saturating_mul(1000))
}

/// True if an HTTP status code is transient and safe to retry with backoff.
#[inline]
pub fn is_retryable_status(status: u16) -> bool {
    matches!(status, 408 | 429 | 500 | 502 | 503 | 504 | 529)
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
    out.input_tokens = if map.cached_included_in_input {
        raw_in.saturating_sub(cached)
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
