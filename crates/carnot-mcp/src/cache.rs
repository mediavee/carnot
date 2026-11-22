//! Zero-cold-start user-level disk schema cache & MCP `2026-07-28` JSON Schema minifier/validator.
//!
//! Incorporates MCP `2026-07-28` (`SEP-2549`, `SEP-2243`, `SEP-2106`) & Opus Phase 4 audit fixes:
//!   - Persists negotiated `ProtocolEra` (`Modern20260728` | `Legacy20251125`) + `protocol_version`
//!     so warm starts achieve **0 negotiation round-trips**!
//!   - Honors `CacheableResult` (`ttlMs` + `cacheScope: "public" | "private"`) and binds cache keys
//!     to `cfg.auth_scope_hash()` (effective expanded auth material) with `0600` permissions on Unix!
//!   - Full `SEP-2243` `x-mcp-header`: traverses nested `properties` chains, validates RFC 9110
//!     tchar tokens, enforces case-insensitive uniqueness, forbids `x-mcp-header` under `items`/`oneOf`/
//!     `anyOf`/`allOf`/`$defs`, stores JSON path, and strips `x-mcp-header` from the model schema!
//!   - Enforces `SEP-2106` anti-DoS bounds: rejects external (non-`#`) `$ref`, caps schema at 512
//!     nodes and 16 KiB minified bytes without inline `$ref` expansion!

use crate::config::McpServerConfig;
use carnot_core::canonicalize_json;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Latest stable official MCP specification revision (released 2026-07-28).
pub const MCP_PROTOCOL_VERSION: &str = "2026-07-28";
/// Supported legacy stateful revision for seamless dual-era backward compatibility.
pub const MCP_LEGACY_PROTOCOL_VERSION: &str = "2025-11-25";

const DEFAULT_CACHE_TTL_MS: u64 = 24 * 3600 * 1000;
const MAX_SCHEMA_NODES: usize = 512;
const MAX_SCHEMA_BYTES: usize = 16 * 1024;
const CACHE_SCHEMA_FORMAT_VERSION: u32 = 3;

/// Negotiated MCP protocol era persisted in the schema cache (0 probe RTTs on warm start!).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolEra {
    #[default]
    Modern20260728,
    Legacy20251125,
}

/// One validated `SEP-2243` `x-mcp-header` mapping (JSON path inside tool arguments -> HTTP header).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeaderParamSpec {
    /// Sequence of property keys navigating nested `properties` inside tool `arguments`.
    pub json_path: Vec<String>,
    /// Exact HTTP header name: `Mcp-Param-{token}`.
    pub header_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedToolEntry {
    pub wire_name: String,
    pub remote_name: String,
    pub description: String,
    pub input_schema: Value,
    #[serde(default)]
    pub output_schema: Option<Value>,
    #[serde(default)]
    pub read_only_hint: bool,
    /// `SEP-2243` `x-mcp-header` mappings (supports nested `properties` paths!).
    #[serde(default)]
    pub header_params: Vec<HeaderParamSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedServerSchema {
    #[serde(default = "default_format_version")]
    pub format_version: u32,
    pub protocol_version: String,
    #[serde(default)]
    pub era: ProtocolEra,
    pub server_name: String,
    #[serde(default = "default_cache_scope")]
    pub cache_scope: String,
    #[serde(default)]
    pub ttl_ms: Option<u64>,
    #[serde(default)]
    pub cached_at_ms: u64,
    /// Optional one-line server description / first line of `initialize.instructions`.
    #[serde(default)]
    pub description: Option<String>,
    pub tools: Vec<CachedToolEntry>,
}

fn default_format_version() -> u32 {
    CACHE_SCHEMA_FORMAT_VERSION
}

fn default_cache_scope() -> String {
    "private".into()
}

/// RFC 9110 `tchar` check for HTTP header name tokens (`SEP-2243`).
fn is_rfc9110_token(s: &str) -> bool {
    !s.is_empty()
        && s.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(
                    b,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

/// Validates (`SEP-2106` bounded `$ref` + `SEP-2243` `x-mcp-header`), strips metadata, and
/// recursively sorts all JSON keys for 100% prompt-cache prefix invariance.
pub fn extract_and_validate_tool_schema(
    raw_schema: &Value,
) -> std::result::Result<(Value, Vec<HeaderParamSpec>), String> {
    let mut node_count = 0usize;
    validate_schema_bounds(raw_schema, &mut node_count)?;
    let mut header_params = Vec::new();
    let mut seen_lower = HashSet::new();
    let mut path = Vec::new();
    walk_x_mcp_headers(
        raw_schema,
        true,
        &mut path,
        &mut header_params,
        &mut seen_lower,
    )?;
    header_params.sort_by(|a, b| a.header_name.cmp(&b.header_name));

    let minified = minify_and_canonicalize_schema(raw_schema);
    let serialized_len = serde_json::to_vec(&minified).map(|b| b.len()).unwrap_or(0);
    if serialized_len > MAX_SCHEMA_BYTES {
        return Err(format!(
            "minified schema exceeds {} KiB limit ({} bytes)",
            MAX_SCHEMA_BYTES / 1024,
            serialized_len
        ));
    }
    Ok((minified, header_params))
}

fn validate_schema_bounds(v: &Value, nodes: &mut usize) -> std::result::Result<(), String> {
    *nodes += 1;
    if *nodes > MAX_SCHEMA_NODES {
        return Err(format!(
            "schema exceeds maximum node count ({MAX_SCHEMA_NODES})"
        ));
    }
    match v {
        Value::Object(map) => {
            if let Some(ref_val) = map.get("$ref").and_then(Value::as_str)
                && !ref_val.starts_with('#')
            {
                return Err(format!(
                    "external JSON Schema $ref `{ref_val}` is forbidden (SEP-2106)"
                ));
            }
            for val in map.values() {
                validate_schema_bounds(val, nodes)?;
            }
        }
        Value::Array(arr) => {
            for item in arr {
                validate_schema_bounds(item, nodes)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Recursively walks a JSON Schema validating `x-mcp-header` per `SEP-2243`:
///   - Allowed ONLY on primitive (`string` | `boolean` | `integer`) leaf properties reached purely
///     through `properties` chains (NEVER under `items`, `oneOf`, `anyOf`, `allOf`, `$defs`, `$ref`).
///   - Annotation value MUST be a valid RFC 9110 token `String` (never boolean!).
///   - Header name is ALWAYS `Mcp-Param-{token}` and MUST be unique case-insensitively!
fn walk_x_mcp_headers(
    v: &Value,
    in_property_chain: bool,
    path: &mut Vec<String>,
    out: &mut Vec<HeaderParamSpec>,
    seen_lower: &mut HashSet<String>,
) -> std::result::Result<(), String> {
    match v {
        Value::Object(map) => {
            if let Some(annot) = map.get("x-mcp-header") {
                if !in_property_chain || path.is_empty() {
                    return Err(
                        "`x-mcp-header` is only valid on properties reached via `properties` chains (SEP-2243)"
                            .into(),
                    );
                }
                if map.contains_key("$ref") {
                    return Err("`x-mcp-header` cannot be combined with `$ref` (SEP-2243)".into());
                }
                let Some(token) = annot.as_str() else {
                    return Err(format!(
                        "invalid `x-mcp-header` on `{}`: value must be an RFC 9110 token string",
                        path.join(".")
                    ));
                };
                if !is_rfc9110_token(token) {
                    return Err(format!(
                        "invalid `x-mcp-header` token `{token}` on `{}`: not a valid RFC 9110 token",
                        path.join(".")
                    ));
                }
                let ty = map.get("type").and_then(Value::as_str);
                if !matches!(ty, Some("string" | "boolean" | "integer")) {
                    return Err(format!(
                        "invalid `x-mcp-header` on `{}`: type `{}` must be string, boolean, or integer",
                        path.join("."),
                        ty.unwrap_or("unspecified")
                    ));
                }
                let header_name = format!("Mcp-Param-{token}");
                let lower = header_name.to_ascii_lowercase();
                if !seen_lower.insert(lower) {
                    return Err(format!(
                        "duplicate case-insensitive `x-mcp-header` `{header_name}` in tool schema"
                    ));
                }
                out.push(HeaderParamSpec {
                    json_path: path.clone(),
                    header_name,
                });
            }

            for (k, val) in map {
                if k == "x-mcp-header" {
                    continue;
                }
                if k == "properties" && in_property_chain {
                    if let Some(props) = val.as_object() {
                        for (prop_name, prop_schema) in props {
                            path.push(prop_name.clone());
                            walk_x_mcp_headers(prop_schema, true, path, out, seen_lower)?;
                            path.pop();
                        }
                    }
                } else {
                    // Anything outside direct `properties` (items, oneOf, anyOf, allOf, $defs, etc.)
                    // is walked with `in_property_chain = false` so misplaced `x-mcp-header` is REJECTED!
                    walk_x_mcp_headers(val, false, path, out, seen_lower)?;
                }
            }
        }
        Value::Array(arr) => {
            for item in arr {
                walk_x_mcp_headers(item, false, path, out, seen_lower)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Strips redundant JSON Schema metadata (`$schema`, `$id`, `title`, `examples`, `$comment`,
/// `x-mcp-header`) and sorts all keys. Preserves parameter names inside `properties` /
/// `patternProperties` / `$defs` / `definitions` so `create_issue(title)` is NEVER stripped!
pub fn minify_and_canonicalize_schema(schema: &Value) -> Value {
    let pruned = prune_schema(schema, false);
    canonicalize_json(&pruned)
}

/// Truncates a top-level tool description cleanly at UTF-8 char boundary.
pub fn truncate_description(desc: &str, max_bytes: usize) -> String {
    if desc.len() <= max_bytes {
        return desc.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !desc.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &desc[..end])
}

fn prune_schema(v: &Value, inside_name_map: bool) -> Value {
    match v {
        Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, val) in map {
                if !inside_name_map
                    && matches!(
                        k.as_str(),
                        "$schema" | "$id" | "title" | "examples" | "$comment" | "x-mcp-header"
                    )
                {
                    continue;
                }
                if !inside_name_map
                    && k == "description"
                    && let Some(s) = val.as_str()
                    && s.len() > 280
                {
                    out.insert(k.clone(), Value::String(truncate_description(s, 280)));
                    continue;
                }
                let child_is_name_map = !inside_name_map
                    && matches!(
                        k.as_str(),
                        "properties" | "patternProperties" | "$defs" | "definitions"
                    );
                out.insert(k.clone(), prune_schema(val, child_is_name_map));
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(|x| prune_schema(x, false)).collect()),
        other => other.clone(),
    }
}

/// Computes a deterministic cache key incorporating config + effective auth scope hash + resolved
/// binary `(path, size, mtime)` so `cacheScope: "private"` NEVER leaks across token rotations!
pub fn server_cache_key(server_name: &str, cfg: &McpServerConfig) -> String {
    let mut h = blake3::Hasher::new();
    h.update(&CACHE_SCHEMA_FORMAT_VERSION.to_le_bytes());
    h.update(b"\0");
    h.update(MCP_PROTOCOL_VERSION.as_bytes());
    h.update(b"\0");
    h.update(server_name.as_bytes());
    h.update(b"\0");
    h.update(cfg.fingerprint().as_bytes());
    h.update(b"\0auth:");
    h.update(cfg.auth_scope_hash().as_bytes());
    if let McpServerConfig::Stdio(s) = cfg
        && let Some(bin_path) = resolve_on_path(&s.command)
        && let Ok(meta) = fs::metadata(&bin_path)
    {
        h.update(b"\0bin:");
        h.update(bin_path.to_string_lossy().as_bytes());
        h.update(&meta.len().to_le_bytes());
        if let Ok(m) = meta.modified()
            && let Ok(d) = m.duration_since(UNIX_EPOCH)
        {
            h.update(&d.as_nanos().to_le_bytes());
        }
    }
    h.finalize().to_hex().to_string()
}

fn resolve_on_path(cmd: &str) -> Option<PathBuf> {
    let p = Path::new(cmd);
    if p.is_absolute() && p.is_file() {
        return Some(p.to_path_buf());
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let cand = dir.join(cmd);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

fn cache_dir() -> Option<PathBuf> {
    if let Some(custom) = std::env::var_os("CARNOT_MCP_CACHE_DIR")
        && !custom.is_empty()
    {
        return Some(PathBuf::from(custom));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".carnot").join("mcp-cache"))
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Loads a cached schema entry and returns `(entry, is_stale)` honoring `SEP-2549` `ttlMs`.
pub fn load_cached_schema(cache_key: &str) -> Option<(CachedServerSchema, bool)> {
    let dir = cache_dir()?;
    let path = dir.join(format!("{cache_key}.json"));
    let txt = fs::read_to_string(path).ok()?;
    let entry: CachedServerSchema = serde_json::from_str(&txt).ok()?;
    if entry.format_version != CACHE_SCHEMA_FORMAT_VERSION {
        return None;
    }
    if entry.ttl_ms == Some(0) {
        return None;
    }
    let effective_ttl = entry.ttl_ms.unwrap_or(DEFAULT_CACHE_TTL_MS);
    let age_ms = now_ms().saturating_sub(entry.cached_at_ms);
    let is_stale = age_ms > effective_ttl;
    if age_ms > 7 * DEFAULT_CACHE_TTL_MS {
        return None;
    }
    Some((entry, is_stale))
}

/// Saves a cached server schema atomically with owner-only `0600` permissions on Unix (Opus #23).
pub fn save_cached_schema(cache_key: &str, mut schema: CachedServerSchema) {
    if schema.ttl_ms == Some(0) {
        return;
    }
    let Some(dir) = cache_dir() else {
        return;
    };
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    schema.format_version = CACHE_SCHEMA_FORMAT_VERSION;
    schema.tools.sort_by(|a, b| a.wire_name.cmp(&b.wire_name));
    schema.cached_at_ms = now_ms();
    let target = dir.join(format!("{cache_key}.json"));
    let tmp = dir.join(format!(".{cache_key}.tmp-{}", uuid::Uuid::new_v4()));

    let open_res = {
        let mut opts = fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        opts.open(&tmp)
    };
    if let Ok(bytes) = serde_json::to_vec(&schema)
        && let Ok(mut f) = open_res
        && f.write_all(&bytes).is_ok()
        && f.sync_data().is_ok()
    {
        let _ = fs::rename(&tmp, &target);
    } else {
        let _ = fs::remove_file(&tmp);
    }
}

pub fn invalidate_cached_schema(cache_key: &str) {
    if let Some(dir) = cache_dir() {
        let _ = fs::remove_file(dir.join(format!("{cache_key}.json")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn minify_and_canonicalize_is_key_order_invariant() {
        let a = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "CreateIssueInput",
            "type": "object",
            "required": ["title"],
            "properties": {
                "title": { "type": "string", "title": "Issue Title", "examples": ["Bug"] },
                "b": { "type": "string" },
                "a": { "type": "integer", "examples": [1, 2] }
            }
        });
        let b = json!({
            "properties": {
                "a": { "type": "integer" },
                "b": { "type": "string" },
                "title": { "type": "string" }
            },
            "required": ["title"],
            "type": "object"
        });
        let ma = minify_and_canonicalize_schema(&a);
        assert!(ma.pointer("/properties/title").is_some());
        assert_eq!(
            serde_json::to_string(&ma).expect("ser a"),
            serde_json::to_string(&minify_and_canonicalize_schema(&b)).expect("ser b"),
        );
    }

    #[test]
    fn validates_nested_x_mcp_header_uniqueness_tokens_and_blocks_external_ref() {
        // Valid top-level + nested `properties` chain with RFC 9110 string tokens!
        let valid = json!({
            "type": "object",
            "properties": {
                "tenant_id": { "type": "string", "x-mcp-header": "Tenant-Id" },
                "meta": {
                    "type": "object",
                    "properties": {
                        "region": { "type": "string", "x-mcp-header": "Region" }
                    }
                }
            }
        });
        let (clean, headers) = extract_and_validate_tool_schema(&valid).expect("valid");
        assert_eq!(headers.len(), 2);
        assert!(headers.iter().any(|h| h.header_name == "Mcp-Param-Tenant-Id" && h.json_path == vec!["tenant_id"]));
        assert!(
            headers
                .iter()
                .any(|h| h.header_name == "Mcp-Param-Region"
                    && h.json_path == vec!["meta", "region"])
        );
        assert!(
            clean
                .pointer("/properties/tenant_id/x-mcp-header")
                .is_none()
        );
        assert!(
            clean
                .pointer("/properties/meta/properties/region/x-mcp-header")
                .is_none()
        );

        // Boolean `true` is rejected per SEP-2243 (must be string token!)
        let bad_bool = json!({
            "type": "object",
            "properties": { "a": { "type": "string", "x-mcp-header": true } }
        });
        assert!(extract_and_validate_tool_schema(&bad_bool).is_err());

        // Invalid RFC 9110 token (contains space or colon) MUST be rejected!
        let bad_tok = json!({
            "type": "object",
            "properties": { "a": { "type": "string", "x-mcp-header": "bad: token" } }
        });
        assert!(extract_and_validate_tool_schema(&bad_tok).is_err());

        // Case-insensitive collision MUST be rejected!
        let dup_hdr = json!({
            "type": "object",
            "properties": {
                "a": { "type": "string", "x-mcp-header": "Tenant" },
                "b": { "type": "string", "x-mcp-header": "tenant" }
            }
        });
        assert!(extract_and_validate_tool_schema(&dup_hdr).is_err());

        // Under `items` or `oneOf` MUST be rejected!
        let under_items = json!({
            "type": "object",
            "properties": {
                "list": { "type": "array", "items": { "type": "string", "x-mcp-header": "Item" } }
            }
        });
        assert!(extract_and_validate_tool_schema(&under_items).is_err());

        // Float `number` is forbidden
        let bad_float = json!({
            "type": "object",
            "properties": { "ratio": { "type": "number", "x-mcp-header": "Ratio" } }
        });
        assert!(extract_and_validate_tool_schema(&bad_float).is_err());

        // External $ref is forbidden per SEP-2106
        let bad_ref = json!({
            "type": "object",
            "properties": { "x": { "$ref": "https://evil.example/schema.json" } }
        });
        assert!(extract_and_validate_tool_schema(&bad_ref).is_err());
    }
}
