//! Official Carnot Rust Extension: Two-Tier Cache-Safe Persistent Memory (`MemoryModule`).
//!
//! Designed for high signal, zero bloat, and sub-millisecond recall across sessions:
//!   - **Two-tier architecture (0 static prefix tokens):**
//!     - Tier 1 (`on_session_start`, Turn-1 ONLY): curated bounded digest (default 1 800 chars)
//!       of pinned & highest-priority project/global memories, plus entry count.
//!     - Tier 2 (deferred tools `defer_loading: true`): `MemorySearch` (pure-Rust BM25 + tag/title
//!       weighting + recency boost, `ToolEffects::global_read()`) and `MemorySave` (idempotent
//!       slug upsert/delete).
//!   - **Anti-rot & security invariants:**
//!     - Automatic secret redaction via `carnot_cas::redact_secrets` BEFORE writing to disk.
//!     - Bounded entry size (2 000 chars) and bounded store size (200 entries/scope with LRU
//!       eviction of unpinned entries).
//!     - Crash-safe atomic persistence (`tmp` + `rename`) + human-readable synced `MEMORY.md`.

use carnot_cas::redact_secrets;
use carnot_core::{CarnotConfig, CarnotError, Result, ToolEffects, ToolOutput, ToolSpec};
use carnot_ext::{BoxFuture, CompactionHookAction, CompactionHookContext, Module};
use carnot_tools::{BoxFuture as ToolFuture, Tool};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_util::sync::CancellationToken;

const MAX_ENTRY_CONTENT_CHARS: usize = 2_000;
const MAX_ENTRIES_PER_SCOPE: usize = 200;
const DEFAULT_TURN1_MAX_CHARS: usize = 1_800;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    Project,
    Global,
}

impl MemoryScope {
    fn from_str_opt(s: Option<&str>) -> Self {
        match s.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            Some("global") => Self::Global,
            _ => Self::Project,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::Global => "global",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Preference,
    Pitfall,
    Decision,
    ProjectFact,
    Workflow,
}

impl MemoryKind {
    fn from_str_opt(s: Option<&str>) -> Self {
        match s.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            Some("preference" | "user" | "feedback") => Self::Preference,
            Some("pitfall" | "gotcha" | "warning" | "bug") => Self::Pitfall,
            Some("decision" | "architecture") => Self::Decision,
            Some("workflow" | "command" | "reference") => Self::Workflow,
            _ => Self::ProjectFact,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Preference => "preference",
            Self::Pitfall => "pitfall",
            Self::Decision => "decision",
            Self::ProjectFact => "project_fact",
            Self::Workflow => "workflow",
        }
    }

    /// Lower number = higher priority in the curated Turn-1 digest.
    fn priority_rank(self) -> u8 {
        match self {
            Self::Preference => 0,
            Self::Pitfall => 1,
            Self::Decision => 2,
            Self::Workflow => 3,
            Self::ProjectFact => 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub scope: MemoryScope,
    pub kind: MemoryKind,
    pub title: String,
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub pinned: bool,
    pub created_at_unix: u64,
    pub updated_at_unix: u64,
    #[serde(default)]
    pub access_count: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ScopeStoreFile {
    #[serde(default = "default_schema_version")]
    schema_version: u32,
    #[serde(default)]
    entries: Vec<MemoryEntry>,
}

fn default_schema_version() -> u32 {
    1
}

/// Configuration for [`MemoryModule`].
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Root directory storing `global/` and `projects/<slug>/` (defaults to `~/.carnot/memory`).
    pub root_dir: PathBuf,
    /// Maximum character budget for the curated Turn-1 memory digest (0 disables Turn-1 injection).
    pub max_turn1_chars: usize,
    /// Maximum entries retained per scope before LRU eviction of unpinned entries.
    pub max_entries_per_scope: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        let root_dir = std::env::var_os("HOME").map_or_else(
            || PathBuf::from(".carnot").join("memory"),
            |h| PathBuf::from(h).join(".carnot").join("memory"),
        );
        Self {
            root_dir,
            max_turn1_chars: DEFAULT_TURN1_MAX_CHARS,
            max_entries_per_scope: MAX_ENTRIES_PER_SCOPE,
        }
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn slugify(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len().min(64));
    let mut prev_dash = false;
    for ch in raw.chars() {
        let low = ch.to_ascii_lowercase();
        if low.is_ascii_alphanumeric() {
            out.push(low);
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
        if out.len() >= 64 {
            break;
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        format!("mem-{}", now_unix())
    } else {
        trimmed
    }
}

fn project_dir_slug(workspace_root: &Path) -> String {
    let canon =
        std::fs::canonicalize(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf());
    let base = canon
        .file_name()
        .and_then(|s| s.to_str())
        .map_or_else(|| "workspace".to_string(), slugify);
    let hash = blake3::hash(canon.to_string_lossy().as_bytes()).to_hex();
    format!("{}-{}", &base[..base.len().min(28)], &hash[..10])
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            if cur.len() >= 2 {
                tokens.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if cur.len() >= 2 {
        tokens.push(cur);
    }
    tokens
}

fn atomic_write_file(path: &Path, content: &[u8]) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp.{}.{seq}", std::process::id()));
    std::fs::write(&tmp, content)?;
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}

fn render_markdown_index(scope_label: &str, entries: &[MemoryEntry]) -> String {
    let mut sorted: Vec<&MemoryEntry> = entries.iter().collect();
    sorted.sort_by(|a, b| {
        b.pinned
            .cmp(&a.pinned)
            .then_with(|| a.kind.priority_rank().cmp(&b.kind.priority_rank()))
            .then_with(|| b.updated_at_unix.cmp(&a.updated_at_unix))
    });

    let mut out = format!("# Carnot Memory ({scope_label})\n\n");
    if sorted.is_empty() {
        out.push_str("_No memories recorded yet._\n");
        return out;
    }
    for e in sorted {
        let pin = if e.pinned { " [pinned]" } else { "" };
        let tags = if e.tags.is_empty() {
            String::new()
        } else {
            format!(" — tags: {}", e.tags.join(", "))
        };
        let _ = write!(
            out,
            "## `{}` ({}{pin}): {}{tags}\n\n{}\n\n",
            e.id,
            e.kind.as_str(),
            e.title,
            e.content.trim()
        );
    }
    out
}

#[derive(Debug)]
struct MemoryState {
    project_dir: PathBuf,
    global_dir: PathBuf,
    project_entries: Vec<MemoryEntry>,
    global_entries: Vec<MemoryEntry>,
    max_entries_per_scope: usize,
}

impl MemoryState {
    fn open(workspace_root: &Path, cfg: &MemoryConfig) -> Self {
        let project_dir = cfg
            .root_dir
            .join("projects")
            .join(project_dir_slug(workspace_root));
        let global_dir = cfg.root_dir.join("global");
        let project_entries = Self::load_scope(&project_dir);
        let global_entries = Self::load_scope(&global_dir);
        Self {
            project_dir,
            global_dir,
            project_entries,
            global_entries,
            max_entries_per_scope: cfg.max_entries_per_scope.max(16),
        }
    }

    fn load_scope(dir: &Path) -> Vec<MemoryEntry> {
        let path = dir.join("store.json");
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Vec::new();
        };
        serde_json::from_str::<ScopeStoreFile>(&raw)
            .map(|f| f.entries)
            .unwrap_or_default()
    }

    fn persist_scope(&self, scope: MemoryScope) -> Result<()> {
        let (dir, entries, label) = match scope {
            MemoryScope::Project => (&self.project_dir, &self.project_entries, "project"),
            MemoryScope::Global => (&self.global_dir, &self.global_entries, "global"),
        };
        let file = ScopeStoreFile {
            schema_version: 1,
            entries: entries.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&file).map_err(|e| CarnotError::Tool {
            tool: "MemorySave".into(),
            message: format!("serialize memory store failed: {e}"),
        })?;
        atomic_write_file(&dir.join("store.json"), &bytes)?;
        let md = render_markdown_index(label, entries);
        atomic_write_file(&dir.join("MEMORY.md"), md.as_bytes())?;
        Ok(())
    }

    fn upsert(&mut self, mut entry: MemoryEntry) -> Result<(MemoryEntry, bool)> {
        // Redact secrets BEFORE touching disk!
        entry.title = redact_secrets(&entry.title).chars().take(160).collect();
        entry.content = redact_secrets(&entry.content)
            .chars()
            .take(MAX_ENTRY_CONTENT_CHARS)
            .collect();
        entry.tags = entry
            .tags
            .into_iter()
            .map(|t| slugify(&t))
            .filter(|t| !t.is_empty())
            .take(12)
            .collect();
        entry.tags.sort();
        entry.tags.dedup();

        let scope = entry.scope;
        let cap = self.max_entries_per_scope;
        let list = match scope {
            MemoryScope::Project => &mut self.project_entries,
            MemoryScope::Global => &mut self.global_entries,
        };

        let mut updated_existing = false;
        if let Some(existing) = list.iter_mut().find(|e| e.id == entry.id) {
            entry.created_at_unix = existing.created_at_unix;
            entry.access_count = existing.access_count;
            *existing = entry.clone();
            updated_existing = true;
        } else {
            list.push(entry.clone());
            // Bounded anti-rot eviction: drop least-used, oldest unpinned entries above cap
            while list.len() > cap {
                let victim_idx = list
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| !e.pinned)
                    .min_by_key(|(_, e)| (e.access_count, e.updated_at_unix))
                    .map(|(i, _)| i);
                if let Some(idx) = victim_idx {
                    list.remove(idx);
                } else {
                    break;
                }
            }
        }
        self.persist_scope(scope)?;
        Ok((entry, updated_existing))
    }

    fn delete(&mut self, scope: MemoryScope, id: &str) -> Result<bool> {
        let list = match scope {
            MemoryScope::Project => &mut self.project_entries,
            MemoryScope::Global => &mut self.global_entries,
        };
        let before = list.len();
        list.retain(|e| e.id != id);
        let removed = list.len() != before;
        if removed {
            self.persist_scope(scope)?;
        }
        Ok(removed)
    }

    fn entry_at(&self, sc: MemoryScope, idx: usize) -> &MemoryEntry {
        match sc {
            MemoryScope::Project => &self.project_entries[idx],
            MemoryScope::Global => &self.global_entries[idx],
        }
    }

    /// Pure-Rust BM25 + tag/title boost + pinned/recency scoring across requested scopes.
    fn search(
        &mut self,
        query: &str,
        scope_filter: Option<MemoryScope>,
        kind_filter: Option<MemoryKind>,
        limit: usize,
    ) -> Vec<(f64, MemoryEntry)> {
        let q_tokens = tokenize(query);
        let q_set: HashSet<&str> = q_tokens.iter().map(String::as_str).collect();
        let q_lower = query.trim().to_ascii_lowercase();

        let mut candidates: Vec<(MemoryScope, usize)> = Vec::new();
        if matches!(scope_filter, None | Some(MemoryScope::Project)) {
            for (i, e) in self.project_entries.iter().enumerate() {
                if kind_filter.is_none_or(|k| e.kind == k) {
                    candidates.push((MemoryScope::Project, i));
                }
            }
        }
        if matches!(scope_filter, None | Some(MemoryScope::Global)) {
            for (i, e) in self.global_entries.iter().enumerate() {
                if kind_filter.is_none_or(|k| e.kind == k) {
                    candidates.push((MemoryScope::Global, i));
                }
            }
        }
        if candidates.is_empty() {
            return Vec::new();
        }

        // Empty query -> return top pinned/recent memories cleanly
        if q_tokens.is_empty() {
            let mut ranked: Vec<(f64, MemoryScope, usize)> = candidates
                .into_iter()
                .map(|(sc, idx)| {
                    let e = self.entry_at(sc, idx);
                    let score = if e.pinned { 10.0 } else { 1.0 }
                        + (5.0 - f64::from(e.kind.priority_rank()))
                        + (e.updated_at_unix as f64 / 1e12);
                    (score, sc, idx)
                })
                .collect();
            ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
            ranked.truncate(limit);
            return ranked
                .into_iter()
                .map(|(score, sc, idx)| (score, self.entry_at(sc, idx).clone()))
                .collect();
        }

        // Compute document frequencies & avg doc length for BM25
        let n_docs = candidates.len() as f64;
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut doc_toks: Vec<Vec<String>> = Vec::with_capacity(candidates.len());
        let mut total_len = 0usize;

        for &(sc, idx) in &candidates {
            let e = self.entry_at(sc, idx);
            let mut toks = tokenize(&e.content);
            // Weight title & tags naturally by repeating them in the indexed term bag
            let title_t = tokenize(&e.title);
            toks.extend(title_t.iter().cloned());
            toks.extend(title_t);
            for tag in &e.tags {
                let tt = tokenize(tag);
                toks.extend(tt.iter().cloned());
                toks.extend(tt.iter().cloned());
                toks.extend(tt);
            }
            total_len += toks.len().max(1);
            let uniq: HashSet<&str> = toks.iter().map(String::as_str).collect();
            for term in uniq {
                if q_set.contains(term) {
                    *df.entry(term.to_string()).or_insert(0) += 1;
                }
            }
            doc_toks.push(toks);
        }

        let avgdl = (total_len as f64 / n_docs).max(1.0);
        let k1 = 1.5;
        let b = 0.75;
        let now = now_unix();

        let mut scored: Vec<(f64, MemoryScope, usize)> = Vec::new();
        for (doc_i, &(sc, idx)) in candidates.iter().enumerate() {
            let e = self.entry_at(sc, idx);
            let toks = &doc_toks[doc_i];
            let dl = toks.len().max(1) as f64;
            let mut tf_map: HashMap<&str, usize> = HashMap::new();
            for t in toks {
                if q_set.contains(t.as_str()) {
                    *tf_map.entry(t.as_str()).or_insert(0) += 1;
                }
            }

            let mut score = 0.0;
            for q in &q_set {
                let tf = *tf_map.get(q).unwrap_or(&0) as f64;
                if tf > 0.0 {
                    let df_q = *df.get(*q).unwrap_or(&0) as f64;
                    let idf = ((n_docs - df_q + 0.5) / (df_q + 0.5) + 1.0).ln();
                    score += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl));
                }
            }
            // Exact ID / substring title / tag bonus
            if !q_lower.is_empty()
                && (e.id.contains(&q_lower) || e.title.to_ascii_lowercase().contains(&q_lower))
            {
                score += 2.5;
            }
            if score <= 0.0 {
                continue;
            }
            if e.pinned {
                score *= 1.25;
            }
            if sc == MemoryScope::Project {
                score *= 1.15;
            }
            // Mild recency boost (decays over ~30 days)
            let age_days = now.saturating_sub(e.updated_at_unix) as f64 / 86_400.0;
            score *= 1.0 + 0.15 / (1.0 + age_days / 30.0);

            scored.push((score, sc, idx));
        }

        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(limit);

        // Update access counts for returned hits (best-effort in-memory; persisted on next save)
        let mut results = Vec::with_capacity(scored.len());
        for (score, sc, idx) in scored {
            let entry = match sc {
                MemoryScope::Project => {
                    let e = &mut self.project_entries[idx];
                    e.access_count = e.access_count.saturating_add(1);
                    e.clone()
                }
                MemoryScope::Global => {
                    let e = &mut self.global_entries[idx];
                    e.access_count = e.access_count.saturating_add(1);
                    e.clone()
                }
            };
            results.push((score, entry));
        }
        results
    }

    /// Formats a bounded, high-signal Turn-1 digest (0 tokens in static prefix!).
    fn format_turn1_digest(&self, max_chars: usize) -> Option<String> {
        if max_chars == 0 {
            return None;
        }
        let total = self.project_entries.len() + self.global_entries.len();
        if total == 0 {
            return None;
        }

        let mut all: Vec<&MemoryEntry> = self
            .project_entries
            .iter()
            .chain(self.global_entries.iter())
            .collect();
        all.sort_by(|a, b| {
            b.pinned
                .cmp(&a.pinned)
                .then_with(|| {
                    (b.scope == MemoryScope::Project).cmp(&(a.scope == MemoryScope::Project))
                })
                .then_with(|| a.kind.priority_rank().cmp(&b.kind.priority_rank()))
                .then_with(|| b.updated_at_unix.cmp(&a.updated_at_unix))
        });

        let mut out = format!(
            "[carnot-memory — {total} stored entries (project={}, global={}); use `MemorySearch`/`MemorySave` for full recall/updates]\n",
            self.project_entries.len(),
            self.global_entries.len(),
        );
        let mut shown = 0usize;
        for e in &all {
            let snippet: String = e
                .content
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            let clipped: String = snippet.chars().take(180).collect();
            let pin = if e.pinned { "*" } else { "" };
            let line = format!(
                "- [{}:{}{pin}] `{}` — {}: {}\n",
                e.scope.as_str(),
                e.kind.as_str(),
                e.id,
                e.title,
                clipped
            );
            if out.len() + line.len() > max_chars && shown > 0 {
                break;
            }
            out.push_str(&line);
            shown += 1;
        }
        if shown < total {
            let _ = writeln!(
                out,
                "(+{} more entries searchable via `MemorySearch`)",
                total - shown
            );
        }
        Some(out.trim_end().to_string())
    }
}

/// Official Carnot persistent memory `Module` (Turn-1 curated index + deferred `MemorySearch` & `MemorySave`).
#[derive(Debug)]
pub struct MemoryModule {
    state: Arc<Mutex<MemoryState>>,
    max_turn1_chars: usize,
    search_tool: Arc<MemorySearchTool>,
    save_tool: Arc<MemorySaveTool>,
}

impl MemoryModule {
    pub fn open(workspace_root: &Path, cfg: &MemoryConfig) -> Self {
        let state = Arc::new(Mutex::new(MemoryState::open(workspace_root, cfg)));
        let search_tool = Arc::new(MemorySearchTool {
            state: Arc::clone(&state),
        });
        let save_tool = Arc::new(MemorySaveTool {
            state: Arc::clone(&state),
        });
        Self {
            state,
            max_turn1_chars: cfg.max_turn1_chars,
            search_tool,
            save_tool,
        }
    }

    pub fn entry_counts(&self) -> (usize, usize) {
        let g = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (g.project_entries.len(), g.global_entries.len())
    }
}

impl Module for MemoryModule {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![
            Arc::clone(&self.search_tool) as Arc<dyn Tool>,
            Arc::clone(&self.save_tool) as Arc<dyn Tool>,
        ]
    }

    fn on_session_start(&self, _config: &CarnotConfig) -> BoxFuture<'_, Result<Option<String>>> {
        let digest = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .format_turn1_digest(self.max_turn1_chars);
        Box::pin(async move { Ok(digest) })
    }

    fn before_compact<'a>(
        &'a self,
        _ctx: &'a CompactionHookContext,
    ) -> BoxFuture<'a, Result<CompactionHookAction>> {
        Box::pin(async {
            Ok(CompactionHookAction::AddFocus(
                "Preserve durable user preferences, verified repo pitfalls, and architectural decisions so they stay consistent with cross-session memory.".into(),
            ))
        })
    }
}

#[derive(Debug)]
struct MemorySearchTool {
    state: Arc<Mutex<MemoryState>>,
}

impl Tool for MemorySearchTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "MemorySearch".into(),
            description: "Search persistent cross-session memories (project & global) using fast BM25 + tag/title scoring. Omit `query` to list top pinned/recent memories.".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "query": { "type": "string", "description": "Keywords, file path, symbol, or topic to search (optional)" },
                    "scope": { "type": "string", "enum": ["all", "project", "global"], "description": "Memory scope filter (default: all)" },
                    "kind": { "type": "string", "enum": ["preference", "pitfall", "decision", "project_fact", "workflow"], "description": "Optional kind filter" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 20, "description": "Max results (default: 6)" }
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
        _cancel: CancellationToken,
    ) -> ToolFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let query = input.get("query").and_then(Value::as_str).unwrap_or("");
            let scope_filter = match input.get("scope").and_then(Value::as_str) {
                Some("project") => Some(MemoryScope::Project),
                Some("global") => Some(MemoryScope::Global),
                _ => None,
            };
            let kind_filter = input
                .get("kind")
                .and_then(Value::as_str)
                .map(|k| MemoryKind::from_str_opt(Some(k)));
            let limit = input
                .get("limit")
                .and_then(Value::as_u64)
                .unwrap_or(6)
                .clamp(1, 20) as usize;

            let hits = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .search(query, scope_filter, kind_filter, limit);

            if hits.is_empty() {
                return Ok(ToolOutput::ok_text(format!(
                    "No memories matched `{query}`."
                )));
            }

            let mut out = format!("[MemorySearch — {} hit(s)]\n", hits.len());
            for (score, e) in &hits {
                let pin = if e.pinned { " [pinned]" } else { "" };
                let tags = if e.tags.is_empty() {
                    String::new()
                } else {
                    format!(" | tags: {}", e.tags.join(", "))
                };
                let _ = writeln!(
                    out,
                    "\n- `{}` ({}/{}{pin}, score={score:.2}{tags}): **{}**\n  {}",
                    e.id,
                    e.scope.as_str(),
                    e.kind.as_str(),
                    e.title,
                    e.content.trim().replace('\n', "\n  ")
                );
            }
            Ok(ToolOutput::ok_text(out).with_details(json!({
                "count": hits.len(),
            })))
        })
    }
}

#[derive(Debug)]
struct MemorySaveTool {
    state: Arc<Mutex<MemoryState>>,
}

impl Tool for MemorySaveTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "MemorySave".into(),
            description: "Persist, update (idempotent by `id`), or delete a durable cross-session memory entry (secrets are redacted automatically; avoid ephemeral task scratchpad notes).".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "action": { "type": "string", "enum": ["upsert", "delete"], "description": "Operation (default: upsert)" },
                    "id": { "type": "string", "description": "Stable slug identifier (derived from `title` if omitted; required for `delete`)" },
                    "scope": { "type": "string", "enum": ["project", "global"], "description": "Target scope (default: project)" },
                    "kind": { "type": "string", "enum": ["preference", "pitfall", "decision", "project_fact", "workflow"], "description": "Memory category (default: project_fact)" },
                    "title": { "type": "string", "description": "Short descriptive title (required for upsert)" },
                    "content": { "type": "string", "description": "Concise durable fact/preference/pitfall (max 2000 chars, required for upsert)" },
                    "tags": { "type": "array", "items": { "type": "string" }, "description": "Optional search tags" },
                    "pinned": { "type": "boolean", "description": "Pin in Turn-1 digest (use sparingly for critical rules/preferences)" }
                }
            }),
            defer_loading: true,
        }
    }

    fn effects(&self, _input: &Value) -> ToolEffects {
        ToolEffects::serial()
    }

    fn execute(
        &self,
        input: Value,
        _cancel: CancellationToken,
    ) -> ToolFuture<'_, Result<ToolOutput>> {
        Box::pin(async move {
            let action = input
                .get("action")
                .and_then(Value::as_str)
                .unwrap_or("upsert");
            let scope = MemoryScope::from_str_opt(input.get("scope").and_then(Value::as_str));

            if action == "delete" {
                let Some(id) = input
                    .get("id")
                    .and_then(Value::as_str)
                    .map(slugify)
                    .filter(|s| !s.is_empty())
                else {
                    return Ok(ToolOutput::err_text("MemorySave delete requires `id`"));
                };
                let removed = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .delete(scope, &id)?;
                return Ok(ToolOutput::ok_text(if removed {
                    format!("Deleted memory `{id}` from {} scope.", scope.as_str())
                } else {
                    format!("Memory `{id}` not found in {} scope.", scope.as_str())
                }));
            }

            let Some(title) = input
                .get("title")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
            else {
                return Ok(ToolOutput::err_text("MemorySave upsert requires `title`"));
            };
            let Some(content) = input
                .get("content")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
            else {
                return Ok(ToolOutput::err_text("MemorySave upsert requires `content`"));
            };

            let id = input
                .get("id")
                .and_then(Value::as_str)
                .map(slugify)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| slugify(title));
            let kind = MemoryKind::from_str_opt(input.get("kind").and_then(Value::as_str));
            let tags: Vec<String> = input
                .get("tags")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            let pinned = input
                .get("pinned")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let ts = now_unix();

            let entry = MemoryEntry {
                id,
                scope,
                kind,
                title: title.to_string(),
                content: content.to_string(),
                tags,
                pinned,
                created_at_unix: ts,
                updated_at_unix: ts,
                access_count: 0,
            };

            let (saved, updated) = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .upsert(entry)?;

            let verb = if updated { "Updated" } else { "Saved" };
            Ok(ToolOutput::ok_text(format!(
                "{verb} memory `{}` ({}/{}): {}",
                saved.id,
                saved.scope.as_str(),
                saved.kind.as_str(),
                saved.title
            ))
            .with_details(json!({
                "id": saved.id,
                "scope": saved.scope.as_str(),
                "updated": updated,
            })))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn memory_persists_redacts_secrets_upserts_idempotently_and_ranks_with_bm25() {
        let ws = std::env::temp_dir().join(format!("carnot-mem-ws-{}", uuid::Uuid::new_v4()));
        let mem_root =
            std::env::temp_dir().join(format!("carnot-mem-root-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&ws).expect("ws");

        let cfg = MemoryConfig {
            root_dir: mem_root.clone(),
            max_turn1_chars: 1800,
            max_entries_per_scope: 20,
        };
        let m1 = MemoryModule::open(&ws, &cfg);
        let tools = m1.tools();
        let search = tools
            .iter()
            .find(|t| t.spec().name == "MemorySearch")
            .expect("search");
        let save = tools
            .iter()
            .find(|t| t.spec().name == "MemorySave")
            .expect("save");

        // All memory tools MUST be deferred (0 static tokens)
        assert!(search.spec().defer_loading);
        assert!(save.spec().defer_loading);

        // Save a pitfall containing an API secret -> must redact before persisting!
        let r1 = save
            .execute(
                json!({
                    "id": "cargo-build-pitfall",
                    "kind": "pitfall",
                    "title": "Use musl target for static Linux release",
                    "content": "Never leak sk-ant-api03-1234567890abcdefghijklmnopqrstuvwxyz123456; build with cargo build --release --target x86_64-unknown-linux-musl.",
                    "tags": ["rust", "musl", "release"],
                    "pinned": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("save 1");
        assert!(!r1.is_error);

        // Upsert the same ID -> updates in place (count stays 1!)
        let r2 = save
            .execute(
                json!({
                    "id": "cargo-build-pitfall",
                    "kind": "pitfall",
                    "title": "Use musl target for static Linux release",
                    "content": "Updated: build with musl and never store sk-ant-api03-1234567890abcdefghijklmnopqrstuvwxyz123456.",
                    "tags": ["rust", "musl"],
                    "pinned": true
                }),
                CancellationToken::new(),
            )
            .await
            .expect("save 2");
        assert!(r2.text_only().contains("Updated"));
        assert_eq!(m1.entry_counts(), (1, 0));

        // Save a global preference
        save.execute(
            json!({
                "scope": "global",
                "kind": "preference",
                "title": "Concise French responses",
                "content": "User prefers concise direct French answers without filler.",
                "tags": ["style", "language"]
            }),
            CancellationToken::new(),
        )
        .await
        .expect("save global");

        // Reload in a brand-new session instance -> persists across sessions + secret redacted on disk!
        let m2 = MemoryModule::open(&ws, &cfg);
        assert_eq!(m2.entry_counts(), (1, 1));
        let start_ctx = m2
            .on_session_start(&CarnotConfig::new(ws.clone()))
            .await
            .expect("start")
            .expect("digest");
        assert!(start_ctx.contains("cargo-build-pitfall"));
        assert!(start_ctx.contains("Concise French responses"));
        assert!(
            !start_ctx.contains("sk-ant-api03-1234567890"),
            "secret must be redacted: {start_ctx}"
        );

        // BM25 search ranks musl pitfall #1 for "musl linux build"
        let s2 = m2
            .tools()
            .into_iter()
            .find(|t| t.spec().name == "MemorySearch")
            .expect("search 2");
        let res = s2
            .execute(
                json!({"query": "musl linux build", "limit": 3}),
                CancellationToken::new(),
            )
            .await
            .expect("search exec");
        assert!(res.text_only().contains("cargo-build-pitfall"));
        assert!(!res.text_only().contains("sk-ant-api03-1234567890"));

        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&mem_root);
    }
}
