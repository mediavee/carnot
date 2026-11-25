//! Append-only, branching JSONL v2 session engine.
//!
//! Safety invariants enforced by construction:
//!   - Disk-first durability: serializes & writes to persistent `BufWriter` + `sync_data()` BEFORE
//!     mutating in-memory state (never diverges on I/O errors).
//!   - Crash recovery: `load()` tolerates a single truncated final JSONL line from a mid-write crash.
//!   - Zero-copy message sharing: `Vec<Arc<Message>>` avoids cloning base64 images on every turn.
//!   - Preserved-Thinking-safe Rewind-Receipt: trailing-only truncation back to Checkpoint C,
//!     refusing to cross any message containing user text (even mixed with `tool_result`).
//!   - Compaction safety: never strands an orphan `tool_result` (S-1) and strips `Thinking` /
//!     `redacted_thinking` blocks from the kept tail (S-2) so Anthropic signatures stay valid.

use carnot_core::{CarnotError, Message, Result, Role, ToolSpec, Usage};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Durable append-only JSONL entry stored on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "entry_type", rename_all = "snake_case")]
pub enum SessionEntry {
    Header {
        version: u32,
        session_id: String,
        /// Stable provider cache-routing key inherited across `--fork`, `/fork`, and forked
        /// subagents so `OpenAI` `prompt_cache_key` shards stay warm on forked branches.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_key_id: Option<String>,
        created_unix_ms: u64,
        model: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        workspace_root: Option<PathBuf>,
        static_system_prompt: String,
        tools: Vec<ToolSpec>,
        prefix_hash: String,
    },
    Message {
        id: String,
        parent_id: Option<String>,
        timestamp_unix_ms: u64,
        message: Message,
    },
    Checkpoint {
        id: String,
        message_index: usize,
        reason: String,
        timestamp_unix_ms: u64,
    },
    Rewind {
        id: String,
        checkpoint_message_index: usize,
        shed_tokens: u64,
        expected_net_usd: f64,
        receipt_message: Message,
        timestamp_unix_ms: u64,
    },
    Usage {
        turn: u64,
        usage: Usage,
        timestamp_unix_ms: u64,
    },
    Compaction {
        id: String,
        summary_message: Message,
        kept_tail_count: usize,
        strip_thinking_from_tail: bool,
        timestamp_unix_ms: u64,
    },
    /// Zero-LLM-cost cold-cache pruning (applied ONLY when the rolling provider cache has already
    /// expired): replaces oversized stale `ToolResult` text blocks outside `keep_tail_count` with
    /// compact pointers while preserving every `.carnot/h/` handle and the full JSONL audit trail.
    ColdPrune {
        id: String,
        keep_tail_count: usize,
        min_bytes: usize,
        timestamp_unix_ms: u64,
    },
    /// Host/embedder custom metadata entry persisted append-only in the session JSONL.
    Custom {
        id: String,
        kind: String,
        data: serde_json::Value,
        timestamp_unix_ms: u64,
    },
}

/// Active view projected for the next provider request (`Arc<Message>` — zero image cloning).
#[derive(Debug, Clone)]
pub struct ProjectedContext {
    pub static_system_prompt: String,
    pub tools: Vec<ToolSpec>,
    pub prefix_hash: String,
    pub cache_key_id: String,
    pub messages: Vec<Arc<Message>>,
    pub checkpoint_index: Option<usize>,
}

impl ProjectedContext {
    pub fn estimated_tokens(&self) -> u64 {
        let sys_tok = (self.static_system_prompt.len() as u64) / 4;
        let tool_tok: u64 = self
            .tools
            .iter()
            .filter(|t| !t.defer_loading)
            .map(|t| {
                (t.name.len() + t.description.len() + t.input_schema.to_string().len()) as u64 / 4
            })
            .sum();
        let msg_tok: u64 = self.messages.iter().map(|m| m.estimated_tokens()).sum();
        sys_tok + tool_tok + msg_tok
    }
}

/// Computes the deterministic BLAKE3 digest of the static prefix (`system_prompt` + canonicalized `tools`).
/// Sorts tools by name and recursively sorts JSON schema keys so `preserve_order` NEVER busts the cache!
pub fn compute_static_prefix_hash(
    static_system_prompt: &str,
    tools: &[ToolSpec],
) -> Result<String> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(static_system_prompt.as_bytes());
    hasher.update(&[0u8]);
    let mut canonical_tools: Vec<ToolSpec> = tools
        .iter()
        .map(|t| ToolSpec {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: carnot_core::canonicalize_json(&t.input_schema),
            defer_loading: t.defer_loading,
        })
        .collect();
    canonical_tools.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    let tools_json = serde_json::to_vec(&canonical_tools)?;
    hasher.update(&tools_json);
    Ok(hasher.finalize().to_hex().to_string())
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Lightweight metadata read from line 1 of a persisted session JSONL (used by ACP `session/list`).
#[derive(Debug, Clone)]
pub struct SessionHeaderSummary {
    pub session_id: String,
    pub model: String,
    pub workspace_root: Option<PathBuf>,
    pub created_unix_ms: u64,
}

/// Reads only the first line (`Header`) of a session JSONL file without loading the full trace.
pub fn read_session_header(path: &Path) -> Option<SessionHeaderSummary> {
    use std::io::{BufRead, BufReader};
    let f = File::open(path).ok()?;
    let mut reader = BufReader::new(f);
    let mut first_line = String::new();
    let n = reader.read_line(&mut first_line).ok()?;
    if n == 0 {
        return None;
    }
    match serde_json::from_str::<SessionEntry>(first_line.trim()).ok()? {
        SessionEntry::Header {
            session_id,
            model,
            workspace_root,
            created_unix_ms,
            ..
        } => Some(SessionHeaderSummary {
            session_id,
            model,
            workspace_root,
            created_unix_ms,
        }),
        _ => None,
    }
}

/// Live session backed by a persistent buffered JSONL writer.
///
/// Only exposes append, cold-cache pruning, and safe compaction (`apply_compaction`) —
/// making mid-history mutation impossible by type construction.
#[derive(Debug)]
pub struct SessionStore {
    pub session_id: String,
    /// Stable cache-routing identifier (equals `session_id` on root sessions; inherited from
    /// parent on forked sessions so provider cache keys remain warm).
    pub cache_key_id: String,
    pub path: Option<PathBuf>,
    writer: Option<File>,
    pub model: String,
    pub workspace_root: Option<PathBuf>,
    pub static_system_prompt: String,
    pub tools: Vec<ToolSpec>,
    pub prefix_hash: String,
    entries: Vec<SessionEntry>,
    active_messages: Vec<Arc<Message>>,
    checkpoint_index: Option<usize>,
    last_message_id: Option<String>,
    /// Last valid provider-reported input token count anchored at `(active_messages.len(), tokens)`.
    /// Invalidated automatically by any compaction or cold prune.
    last_usage_anchor: Option<(usize, u64)>,
    /// Unix timestamp (ms) of the most recent recorded session activity (used to detect cold provider caches).
    last_activity_unix_ms: Option<u64>,
}

impl SessionStore {
    pub fn create(
        session_dir: Option<&Path>,
        model: String,
        static_system_prompt: String,
        tools: Vec<ToolSpec>,
    ) -> Result<Self> {
        Self::create_with_workspace(session_dir, None, model, static_system_prompt, tools)
    }

    pub fn create_with_workspace(
        session_dir: Option<&Path>,
        workspace_root: Option<PathBuf>,
        model: String,
        static_system_prompt: String,
        tools: Vec<ToolSpec>,
    ) -> Result<Self> {
        Self::create_with_workspace_and_cache_key(
            session_dir,
            workspace_root,
            model,
            static_system_prompt,
            tools,
            None,
        )
    }

    fn create_with_workspace_and_cache_key(
        session_dir: Option<&Path>,
        workspace_root: Option<PathBuf>,
        model: String,
        static_system_prompt: String,
        tools: Vec<ToolSpec>,
        inherited_cache_key_id: Option<String>,
    ) -> Result<Self> {
        let session_id = uuid::Uuid::now_v7().to_string();
        let cache_key_id = inherited_cache_key_id.unwrap_or_else(|| session_id.clone());
        let prefix_hash = compute_static_prefix_hash(&static_system_prompt, &tools)?;

        let (path, writer) = if let Some(dir) = session_dir {
            fs::create_dir_all(dir)?;
            let p = dir.join(format!("{session_id}.jsonl"));
            let f = OpenOptions::new().create(true).append(true).open(&p)?;
            (Some(p), Some(f))
        } else {
            (None, None)
        };

        let mut store = Self {
            session_id: session_id.clone(),
            cache_key_id: cache_key_id.clone(),
            path,
            writer,
            model: model.clone(),
            workspace_root: workspace_root.clone(),
            static_system_prompt: static_system_prompt.clone(),
            tools: tools.clone(),
            prefix_hash: prefix_hash.clone(),
            entries: Vec::new(),
            active_messages: Vec::new(),
            checkpoint_index: None,
            last_message_id: None,
            last_usage_anchor: None,
            last_activity_unix_ms: None,
        };

        let header_cache_key = (cache_key_id != session_id).then_some(cache_key_id);
        store.append_entry(
            SessionEntry::Header {
                version: 2,
                session_id,
                cache_key_id: header_cache_key,
                created_unix_ms: now_unix_ms(),
                model,
                workspace_root,
                static_system_prompt,
                tools,
                prefix_hash,
            },
            true,
        )?;
        Ok(store)
    }

    /// Replays a JSONL session from disk, truncating any crash-truncated final line before
    /// reopening for append.
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_internal(path, false)
    }

    /// Replays a JSONL session from disk in read-only mode (used by `--fork`): ignores any
    /// crash-truncated trailing line in memory without modifying or opening the file for writing.
    pub fn load_readonly(path: &Path) -> Result<Self> {
        Self::load_internal(path, true)
    }

    fn load_internal(path: &Path, readonly: bool) -> Result<Self> {
        let raw = fs::read(path)?;
        let mut entries = Vec::new();
        let mut offset: usize = 0;
        let mut valid_end: usize = 0;
        let mut truncated_tail = false;

        while offset < raw.len() {
            let line_end = raw[offset..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(raw.len(), |rel| offset + rel + 1);
            let is_last = line_end == raw.len();
            let slice = &raw[offset..line_end];
            let trimmed = std::str::from_utf8(slice).map(str::trim);

            match trimmed {
                Ok("") => {
                    valid_end = line_end;
                }
                Ok(line_str) => match serde_json::from_str::<SessionEntry>(line_str) {
                    Ok(entry) => {
                        entries.push(entry);
                        valid_end = line_end;
                    }
                    Err(_) if is_last => {
                        truncated_tail = true;
                        break;
                    }
                    Err(e) => return Err(CarnotError::Json(e)),
                },
                Err(_) if is_last => {
                    truncated_tail = true;
                    break;
                }
                Err(e) => {
                    return Err(CarnotError::Session(format!(
                        "invalid UTF-8 in session {}: {e}",
                        path.display()
                    )));
                }
            }
            offset = line_end;
        }

        if truncated_tail && !readonly {
            let f = OpenOptions::new().write(true).open(path)?;
            f.set_len(valid_end as u64)?;
            f.sync_data()?;
        }

        let Some(SessionEntry::Header {
            session_id,
            cache_key_id,
            model,
            workspace_root,
            static_system_prompt,
            tools,
            prefix_hash,
            ..
        }) = entries.first().cloned()
        else {
            return Err(CarnotError::Session(format!(
                "missing header in session {}",
                path.display()
            )));
        };

        let (stored_path, writer) = if readonly {
            (None, None)
        } else {
            let append_file = OpenOptions::new().create(true).append(true).open(path)?;
            (Some(path.to_path_buf()), Some(append_file))
        };
        let resolved_cache_key = cache_key_id.unwrap_or_else(|| session_id.clone());
        let mut store = Self {
            session_id,
            cache_key_id: resolved_cache_key,
            path: stored_path,
            writer,
            model,
            workspace_root,
            static_system_prompt,
            tools,
            prefix_hash,
            entries: Vec::new(),
            active_messages: Vec::new(),
            checkpoint_index: None,
            last_message_id: None,
            last_usage_anchor: None,
            last_activity_unix_ms: None,
        };

        for entry in entries {
            store.apply_in_memory(&entry);
            store.entries.push(entry);
        }
        Ok(store)
    }

    /// Strips any trailing incomplete assistant `ToolUse` message(s) so a forked session or
    /// forked subagent never starts with an orphan `tool_use` block (which providers reject with
    /// HTTP 400). Preserves zero-copy `Arc<Message>` sharing for every earlier turn.
    pub fn sanitize_fork_messages(messages: &[Arc<Message>]) -> Vec<Arc<Message>> {
        let mut end = messages.len();
        while end > 0
            && messages[end - 1].role == Role::Assistant
            && messages[end - 1].has_tool_use()
        {
            end -= 1;
        }
        messages[..end].to_vec()
    }

    /// Forks this live session into a brand-new `SessionStore` (fresh UUID v7 `session_id`)
    /// with identical static prefix (`static_system_prompt`, `tools`, `prefix_hash`) and a
    /// sanitized zero-copy clone of `active_messages`.
    ///
    /// The original session file on disk is NEVER modified.
    pub fn fork(&self, session_dir: Option<&Path>) -> Result<Self> {
        Self::from_forked_snapshot_with_cache_key(
            session_dir,
            self.workspace_root.clone(),
            self.model.clone(),
            self.static_system_prompt.clone(),
            self.tools.clone(),
            &self.active_messages,
            self.checkpoint_index,
            Some(self.cache_key_id.clone()),
        )
    }

    /// Builds a new `SessionStore` seeded from a parent conversation snapshot (used by
    /// `SessionStore::fork` and prompt-cache-warm `Subagent` forks).
    pub fn from_forked_snapshot(
        session_dir: Option<&Path>,
        workspace_root: Option<PathBuf>,
        model: String,
        static_system_prompt: String,
        tools: Vec<ToolSpec>,
        snapshot_messages: &[Arc<Message>],
        checkpoint_index: Option<usize>,
    ) -> Result<Self> {
        Self::from_forked_snapshot_with_cache_key(
            session_dir,
            workspace_root,
            model,
            static_system_prompt,
            tools,
            snapshot_messages,
            checkpoint_index,
            None,
        )
    }

    /// Builds a new `SessionStore` seeded from a parent conversation snapshot while optionally
    /// inheriting the parent's `cache_key_id` so provider prompt-cache shards (e.g. `OpenAI`
    /// `prompt_cache_key`) stay hot on the forked branch.
    #[expect(
        clippy::too_many_arguments,
        reason = "forked session constructor forwards all explicit snapshot fields without bundling a throwaway struct"
    )]
    pub fn from_forked_snapshot_with_cache_key(
        session_dir: Option<&Path>,
        workspace_root: Option<PathBuf>,
        model: String,
        static_system_prompt: String,
        tools: Vec<ToolSpec>,
        snapshot_messages: &[Arc<Message>],
        checkpoint_index: Option<usize>,
        inherited_cache_key_id: Option<String>,
    ) -> Result<Self> {
        let mut store = Self::create_with_workspace_and_cache_key(
            session_dir,
            workspace_root,
            model,
            static_system_prompt,
            tools,
            inherited_cache_key_id,
        )?;
        let sanitized = Self::sanitize_fork_messages(snapshot_messages);
        let persist = store.writer.is_some();
        for msg in &sanitized {
            if persist {
                let id = uuid::Uuid::now_v7().to_string();
                let entry = SessionEntry::Message {
                    id: id.clone(),
                    parent_id: store.last_message_id.clone(),
                    timestamp_unix_ms: now_unix_ms(),
                    message: (**msg).clone(),
                };
                // Buffer writes without per-message flush; flush once below for instant fork speed.
                store.append_entry(entry, false)?;
                store.last_message_id = Some(id);
            } else {
                // In-memory fork (e.g. disposable forked subagent): true zero-copy `Arc::clone`!
                store.active_messages.push(Arc::clone(msg));
            }
        }
        if let Some(cp) = checkpoint_index
            && cp < store.active_messages.len()
            && !store.active_messages[cp].has_tool_use()
        {
            store.checkpoint_index = Some(cp);
            if persist {
                let entry = SessionEntry::Checkpoint {
                    id: uuid::Uuid::now_v7().to_string(),
                    message_index: cp,
                    reason: "forked-parent-checkpoint".into(),
                    timestamp_unix_ms: now_unix_ms(),
                };
                store.append_entry(entry, false)?;
            }
        }
        if let Some(ref mut w) = store.writer {
            w.flush()?;
        }
        Ok(store)
    }

    pub fn push_message(&mut self, message: Message) -> Result<usize> {
        let id = uuid::Uuid::now_v7().to_string();
        let entry = SessionEntry::Message {
            id: id.clone(),
            parent_id: self.last_message_id.clone(),
            timestamp_unix_ms: now_unix_ms(),
            message,
        };
        self.append_entry(entry, true)?;
        self.last_message_id = Some(id);
        Ok(self.active_messages.len().saturating_sub(1))
    }

    pub fn mark_checkpoint(&mut self, reason: impl Into<String>) -> Result<Option<usize>> {
        let Some(last_idx) = self.active_messages.len().checked_sub(1) else {
            return Ok(None);
        };
        if self.active_messages[last_idx].has_tool_use() {
            return Ok(self.checkpoint_index);
        }
        let entry = SessionEntry::Checkpoint {
            id: uuid::Uuid::now_v7().to_string(),
            message_index: last_idx,
            reason: reason.into(),
            timestamp_unix_ms: now_unix_ms(),
        };
        self.append_entry(entry, false)?;
        Ok(Some(last_idx))
    }

    /// Preserved-Thinking-safe trailing truncation back to `checkpoint_index`.
    /// Refuses to cross any user message carrying text (even mixed with `tool_result`, S-5).
    pub fn apply_rewind(
        &mut self,
        checkpoint_index: usize,
        shed_tokens: u64,
        expected_net_usd: f64,
        receipt_text: String,
    ) -> Result<()> {
        if checkpoint_index >= self.active_messages.len() {
            return Err(CarnotError::Session(format!(
                "rewind checkpoint {checkpoint_index} out of bounds ({})",
                self.active_messages.len()
            )));
        }
        if self.active_messages[checkpoint_index].has_tool_use() {
            return Err(CarnotError::Session(
                "cannot rewind to an open assistant tool_use message".into(),
            ));
        }
        for msg in &self.active_messages[(checkpoint_index + 1)..] {
            if msg.role == Role::User && msg.has_text() {
                return Err(CarnotError::Session(
                    "refusing to rewind across a user text message".into(),
                ));
            }
        }

        let receipt_message = Message::system_note(receipt_text);
        let entry = SessionEntry::Rewind {
            id: uuid::Uuid::now_v7().to_string(),
            checkpoint_message_index: checkpoint_index,
            shed_tokens,
            expected_net_usd,
            receipt_message,
            timestamp_unix_ms: now_unix_ms(),
        };
        self.append_entry(entry, true)
    }

    pub fn record_usage(&mut self, turn: u64, usage: Usage) -> Result<()> {
        self.append_entry(
            SessionEntry::Usage {
                turn,
                usage,
                timestamp_unix_ms: now_unix_ms(),
            },
            false,
        )
    }

    /// Appends a durable custom metadata entry to the session JSONL (does not alter active messages).
    pub fn append_custom(
        &mut self,
        kind: impl Into<String>,
        data: serde_json::Value,
    ) -> Result<String> {
        let id = uuid::Uuid::now_v7().to_string();
        self.append_entry(
            SessionEntry::Custom {
                id: id.clone(),
                kind: kind.into(),
                data,
                timestamp_unix_ms: now_unix_ms(),
            },
            true,
        )?;
        Ok(id)
    }

    /// Repairs a crash/SIGTERM-interrupted session whose trailing active message is an
    /// `Assistant` turn containing unpaired `ToolUse` blocks by persisting paired error
    /// `ToolResult` blocks so subsequent provider calls never fail with HTTP 400.
    pub fn repair_dangling_tool_uses(&mut self) -> Result<usize> {
        let Some(last) = self.active_messages.last() else {
            return Ok(0);
        };
        if last.role != Role::Assistant {
            return Ok(0);
        }
        let recovery: Vec<carnot_core::ContentBlock> = last
            .content
            .iter()
            .filter_map(|b| match b {
                carnot_core::ContentBlock::ToolUse { id, name, .. } => {
                    Some(carnot_core::ContentBlock::ToolResult {
                        tool_use_id: id.clone(),
                        content: vec![carnot_core::ToolResultBlock::text(format!(
                            "Tool `{name}` was interrupted before completion; verify any side effects before retrying."
                        ))],
                        is_error: true,
                    })
                }
                _ => None,
            })
            .collect();
        let count = recovery.len();
        if count > 0 {
            self.push_message(Message {
                role: Role::User,
                content: recovery,
            })?;
        }
        Ok(count)
    }

    /// Safe compaction: adjusts `tail_start` backward so no orphan `tool_result` is kept (S-1)
    /// and strips `Thinking` / `redacted_thinking` from the kept tail so Preserved Thinking stays valid (S-2).
    pub fn apply_compaction(&mut self, summary: String, requested_keep_tail: usize) -> Result<()> {
        let n = self.active_messages.len();
        let mut tail_start = n.saturating_sub(requested_keep_tail);
        // S-1: Never start kept tail on a `tool_result` without its preceding `assistant(tool_use)`!
        while tail_start > 0 && tail_start < n && self.active_messages[tail_start].has_tool_result()
        {
            tail_start -= 1;
        }
        let effective_keep_tail = n.saturating_sub(tail_start);
        // Emit as Role::User so `summary -> kept_assistant_tail` obeys strict role alternation
        // across ALL providers (Anthropic, OpenAI, DeepSeek, Grok, Gemini).
        let entry = SessionEntry::Compaction {
            id: uuid::Uuid::now_v7().to_string(),
            summary_message: Message::user_text(summary),
            kept_tail_count: effective_keep_tail,
            strip_thinking_from_tail: true,
            timestamp_unix_ms: now_unix_ms(),
        };
        self.append_entry(entry, true)
    }

    /// Token-budgeted compaction: preserves as many recent verbatim messages as fit within
    /// `tail_token_budget` (clamped to `min_tail_messages..=max_tail_messages`), strictly aligned
    /// on `tool_use`/`tool_result` boundaries (S-1), returning `(tokens_before, tokens_after)`.
    pub fn apply_compaction_budgeted(
        &mut self,
        summary: String,
        tail_token_budget: u64,
        min_tail_messages: usize,
        max_tail_messages: usize,
    ) -> Result<(u64, u64)> {
        let tokens_before = self.effective_context_tokens();
        let n = self.active_messages.len();
        let min_keep = min_tail_messages.min(n);
        let max_keep = max_tail_messages.max(min_keep).min(n);

        let mut acc_tokens = 0u64;
        let mut keep = 0usize;
        for msg in self.active_messages.iter().rev() {
            let mt = msg.without_thinking().estimated_tokens();
            if keep >= max_keep {
                break;
            }
            if keep >= min_keep && acc_tokens.saturating_add(mt) > tail_token_budget {
                break;
            }
            acc_tokens = acc_tokens.saturating_add(mt);
            keep += 1;
        }

        // Never compact to keep 100% of active messages when there are older turns to fold
        let capped_keep = if n > 2 { keep.min(n - 1) } else { keep };
        self.apply_compaction(summary, capped_keep)?;
        let tokens_after = self.project_context().estimated_tokens();
        Ok((tokens_before, tokens_after))
    }

    pub fn checkpoint_index(&self) -> Option<usize> {
        self.checkpoint_index
    }

    pub fn active_messages(&self) -> &[Arc<Message>] {
        &self.active_messages
    }

    pub fn entries(&self) -> &[SessionEntry] {
        &self.entries
    }

    pub fn last_activity_unix_ms(&self) -> Option<u64> {
        self.last_activity_unix_ms
    }

    /// Opportunistically prunes oversized stale `ToolResult` text blocks outside the last
    /// `keep_tail_count` messages when the caller has confirmed the provider's rolling prompt cache
    /// is already cold. Zero LLM calls, zero cache breaks, lossless JSONL durability.
    /// Returns the total byte reduction (`0` if reclaimable bytes `< min_reclaimable_bytes`).
    pub fn prune_cold_tool_results(
        &mut self,
        keep_tail_count: usize,
        min_bytes: usize,
        min_reclaimable_bytes: usize,
    ) -> Result<usize> {
        let cutoff = self.active_messages.len().saturating_sub(keep_tail_count);
        if cutoff == 0 {
            return Ok(0);
        }
        let mut reclaimable = 0usize;
        for msg in &self.active_messages[..cutoff] {
            for block in &msg.content {
                if let carnot_core::ContentBlock::ToolResult { content, .. } = block {
                    for tb in content {
                        if let carnot_core::ToolResultBlock::Text { text } = tb
                            && text.len() >= min_bytes
                            && !text.starts_with("[pruned cold tool_result:")
                        {
                            let stub = build_cold_prune_stub(text);
                            reclaimable += text.len().saturating_sub(stub.len());
                        }
                    }
                }
            }
        }
        if reclaimable < min_reclaimable_bytes {
            return Ok(0);
        }

        let entry = SessionEntry::ColdPrune {
            id: uuid::Uuid::now_v7().to_string(),
            keep_tail_count,
            min_bytes,
            timestamp_unix_ms: now_unix_ms(),
        };
        self.append_entry(entry, true)?;
        Ok(reclaimable)
    }

    pub fn tokens_since_checkpoint(&self) -> u64 {
        let start = self.checkpoint_index.map_or(0, |i| i + 1);
        self.active_messages
            .get(start..)
            .unwrap_or(&[])
            .iter()
            .map(|m| m.estimated_tokens())
            .sum()
    }

    /// True when the session holds a provider-reported input token measurement for the current
    /// message sequence (reset to `false` on compaction, rewind, or cold prune until the next turn).
    pub fn has_provider_usage_anchor(&self) -> bool {
        self.last_usage_anchor.is_some()
    }

    /// Provider-grounded context size estimate: uses the exact last provider-billed input token
    /// count plus `estimated_tokens()` for any trailing messages added since that call.
    /// Automatically falls back to full `project_context().estimated_tokens()` after compaction
    /// or rewind.
    pub fn effective_context_tokens(&self) -> u64 {
        if let Some((anchor_len, input_tokens)) = self.last_usage_anchor
            && anchor_len <= self.active_messages.len()
        {
            let trailing: u64 = self.active_messages[anchor_len..]
                .iter()
                .map(|m| m.estimated_tokens())
                .sum();
            return input_tokens.saturating_add(trailing);
        }
        self.project_context().estimated_tokens()
    }

    pub fn project_context(&self) -> ProjectedContext {
        ProjectedContext {
            static_system_prompt: self.static_system_prompt.clone(),
            tools: self.tools.clone(),
            prefix_hash: self.prefix_hash.clone(),
            cache_key_id: self.cache_key_id.clone(),
            messages: self.active_messages.clone(),
            checkpoint_index: self.checkpoint_index,
        }
    }

    /// Disk-first unbuffered append: write complete line (+ optional `sync_data`) BEFORE mutating
    /// memory so no stale bytes are retained in a buffer after a failed flush.
    fn append_entry(&mut self, entry: SessionEntry, sync: bool) -> Result<()> {
        if let Some(ref mut f) = self.writer {
            let mut buf = serde_json::to_vec(&entry)?;
            buf.push(b'\n');
            f.write_all(&buf)?;
            if sync {
                f.sync_data()?;
            }
        }
        self.apply_in_memory(&entry);
        self.entries.push(entry);
        Ok(())
    }

    /// Infallible in-memory state transition (replayed identically during `load()`).
    fn apply_in_memory(&mut self, entry: &SessionEntry) {
        match entry {
            SessionEntry::Header {
                created_unix_ms, ..
            } => {
                self.last_activity_unix_ms = Some(*created_unix_ms);
            }
            SessionEntry::Usage {
                usage,
                timestamp_unix_ms,
                ..
            } => {
                self.last_activity_unix_ms = Some(*timestamp_unix_ms);
                let billed = usage.total_context_tokens();
                if billed > 0 {
                    self.last_usage_anchor = Some((self.active_messages.len(), billed));
                }
            }
            SessionEntry::Message {
                id,
                message,
                timestamp_unix_ms,
                ..
            } => {
                self.last_activity_unix_ms = Some(*timestamp_unix_ms);
                self.active_messages.push(Arc::new(message.clone()));
                self.last_message_id = Some(id.clone());
            }
            SessionEntry::Checkpoint {
                message_index,
                timestamp_unix_ms,
                ..
            } => {
                self.last_activity_unix_ms = Some(*timestamp_unix_ms);
                if *message_index < self.active_messages.len() {
                    self.checkpoint_index = Some(*message_index);
                }
            }
            SessionEntry::Rewind {
                checkpoint_message_index,
                receipt_message,
                timestamp_unix_ms,
                ..
            } => {
                self.last_activity_unix_ms = Some(*timestamp_unix_ms);
                let keep_len = (*checkpoint_message_index + 1).min(self.active_messages.len());
                self.active_messages.truncate(keep_len);
                self.active_messages.push(Arc::new(receipt_message.clone()));
                self.checkpoint_index = Some(self.active_messages.len() - 1);
                self.last_usage_anchor = None;
            }
            SessionEntry::Compaction {
                summary_message,
                kept_tail_count,
                strip_thinking_from_tail,
                timestamp_unix_ms,
                ..
            } => {
                self.last_activity_unix_ms = Some(*timestamp_unix_ms);
                let n = self.active_messages.len();
                let tail_start = n.saturating_sub(*kept_tail_count);
                let tail: Vec<Arc<Message>> = self.active_messages[tail_start..]
                    .iter()
                    .filter_map(|m| {
                        let base = if *strip_thinking_from_tail {
                            m.without_thinking()
                        } else {
                            (**m).clone()
                        };
                        if base.content.is_empty() {
                            return None;
                        }
                        Some(Arc::new(trim_oversized_tail_blocks(base)))
                    })
                    .collect();
                self.active_messages.clear();
                self.active_messages.push(Arc::new(summary_message.clone()));
                self.active_messages.extend(tail);
                self.checkpoint_index = Some(0);
                self.last_usage_anchor = None;
            }
            SessionEntry::ColdPrune {
                keep_tail_count,
                min_bytes,
                timestamp_unix_ms,
                ..
            } => {
                self.last_activity_unix_ms = Some(*timestamp_unix_ms);
                let cutoff = self.active_messages.len().saturating_sub(*keep_tail_count);
                for slot in &mut self.active_messages[..cutoff] {
                    let needs_prune = slot.content.iter().any(|block| {
                        matches!(
                            block,
                            carnot_core::ContentBlock::ToolResult { content, .. }
                                if content.iter().any(|tb| matches!(
                                    tb,
                                    carnot_core::ToolResultBlock::Text { text }
                                        if text.len() >= *min_bytes
                                            && !text.starts_with("[pruned cold tool_result:")
                                ))
                        )
                    });
                    if needs_prune {
                        let mut cloned = (**slot).clone();
                        for block in &mut cloned.content {
                            if let carnot_core::ContentBlock::ToolResult { content, .. } = block {
                                for tb in content.iter_mut() {
                                    if let carnot_core::ToolResultBlock::Text { text } = tb
                                        && text.len() >= *min_bytes
                                        && !text.starts_with("[pruned cold tool_result:")
                                    {
                                        *text = build_cold_prune_stub(text);
                                    }
                                }
                            }
                        }
                        *slot = Arc::new(cloned);
                    }
                }
                self.last_usage_anchor = None;
            }
            SessionEntry::Custom {
                timestamp_unix_ms, ..
            } => {
                self.last_activity_unix_ms = Some(*timestamp_unix_ms);
            }
        }
    }
}

fn extract_cas_handle(text: &str) -> Option<&str> {
    let idx = text.find(".carnot/h/")?;
    let rest = &text[idx..];
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '-' | '_')))
        .unwrap_or(rest.len());
    let candidate = &rest[..end];
    if candidate.len() > ".carnot/h/".len() {
        Some(candidate)
    } else {
        None
    }
}

fn build_cold_prune_stub(text: &str) -> String {
    if let Some(handle) = extract_cas_handle(text) {
        format!(
            "[pruned cold tool_result: {} B — inspect archived handle {handle} or re-read if needed]",
            text.len()
        )
    } else {
        format!(
            "[pruned cold tool_result: {} B (full output preserved in session JSONL; re-run or re-read if needed)]",
            text.len()
        )
    }
}

/// Deterministic safety cap for gigantic individual `ToolResult` blocks preserved in a compacted
/// tail (>32 KiB -> 12 KiB head + 12 KiB tail) so emergency compaction can never re-overflow.
fn trim_oversized_tail_blocks(mut msg: Message) -> Message {
    const MAX_TAIL_RESULT_BYTES: usize = 32_768;
    const KEEP_EDGE_BYTES: usize = 12_288;
    for block in &mut msg.content {
        if let carnot_core::ContentBlock::ToolResult { content, .. } = block {
            for tb in content.iter_mut() {
                if let carnot_core::ToolResultBlock::Text { text } = tb
                    && text.len() > MAX_TAIL_RESULT_BYTES
                {
                    let mut h = KEEP_EDGE_BYTES.min(text.len());
                    while h > 0 && !text.is_char_boundary(h) {
                        h -= 1;
                    }
                    let mut t = text.len().saturating_sub(KEEP_EDGE_BYTES);
                    while t < text.len() && !text.is_char_boundary(t) {
                        t += 1;
                    }
                    let omitted = t.saturating_sub(h);
                    *text = format!(
                        "{}\n[... {omitted} bytes trimmed during compaction; full trace preserved in session JSONL ...]\n{}",
                        &text[..h],
                        &text[t..]
                    );
                }
            }
        }
    }
    msg
}

#[cfg(test)]
mod tests {
    use super::*;
    use carnot_core::{ContentBlock, ToolResultBlock};

    #[test]
    fn compaction_never_leaves_orphan_tool_result_and_strips_thinking() {
        let mut s = SessionStore::create(None, "claude-opus-5-5".into(), "STATIC".into(), vec![])
            .expect("create");
        s.push_message(Message::user_text("task")).expect("u0");
        s.push_message(Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "secret".into(),
                    signature: Some("sig123".into()),
                    origin: None,
                },
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "Read".into(),
                    input: serde_json::json!({"file_path": "a.rs"}),
                    invalid_json: None,
                    provider_meta: None,
                    origin: None,
                },
            ],
        })
        .expect("a1");
        s.push_message(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".into(),
                content: vec![ToolResultBlock::text("ok")],
                is_error: false,
            }],
        })
        .expect("u1");

        // Request keeping only 1 tail message (would strand tool_result!); S-1 pulls in assistant(tool_use) too!
        s.apply_compaction("summary".into(), 1).expect("compact");
        let msgs = s.active_messages();
        assert_eq!(msgs.len(), 3);
        assert!(msgs[1].has_tool_use());
        assert!(
            !msgs[1].has_signed_thinking(),
            "S-2: thinking must be stripped on compaction"
        );
        assert!(msgs[2].has_tool_result());
    }

    #[test]
    fn load_recovers_from_crash_truncated_last_line() {
        let dir = std::env::temp_dir().join(format!("carnot-crash-{}", uuid::Uuid::new_v4()));
        let mut s = SessionStore::create(
            Some(&dir),
            "claude-opus-5-5".into(),
            "STATIC".into(),
            vec![],
        )
        .expect("create");
        s.push_message(Message::user_text("hello")).expect("u0");
        let path = s.path.clone().expect("path");
        drop(s);

        // Simulate a mid-write crash appending partial JSON
        let mut f = OpenOptions::new().append(true).open(&path).expect("open");
        write!(f, "{{\"entry_type\":\"message\",\"broken").expect("corrupt");

        let mut recovered = SessionStore::load(&path).expect("recover");
        assert_eq!(recovered.active_messages().len(), 1);
        assert_eq!(recovered.active_messages()[0].text_concat(), "hello");

        // Appending after crash recovery and reloading a second time must succeed.
        recovered
            .push_message(Message::assistant_text("after-crash"))
            .expect("append post-crash");
        let reloaded = SessionStore::load(&path).expect("reload after post-crash append");
        assert_eq!(reloaded.active_messages().len(), 2);
        assert_eq!(reloaded.active_messages()[1].text_concat(), "after-crash");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rewind_refuses_to_drop_steering_text_mixed_with_tool_result() {
        let mut s = SessionStore::create(None, "claude-opus-5-5".into(), "STATIC".into(), vec![])
            .expect("create");
        s.push_message(Message::user_text("start")).expect("u0");
        s.mark_checkpoint("cp0").expect("cp");
        s.push_message(Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "c1".into(),
                    content: vec![ToolResultBlock::text("ok")],
                    is_error: false,
                },
                ContentBlock::Text {
                    text: "also change X".into(),
                },
            ],
        })
        .expect("steer");
        assert!(s.apply_rewind(0, 20_000, 0.2, "receipt".into()).is_err());
    }

    #[test]
    fn budgeted_compaction_caps_oversized_tail_and_preserves_pairs() {
        let mut s = SessionStore::create(None, "claude-opus-5-5".into(), "STATIC".into(), vec![])
            .expect("create");
        s.push_message(Message::user_text("initial")).expect("u0");
        for i in 0..5 {
            s.push_message(Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: format!("c{i}"),
                    name: "Bash".into(),
                    input: serde_json::json!({"command": "echo ok"}),
                    invalid_json: None,
                    provider_meta: None,
                    origin: None,
                }],
            })
            .expect("a");
            let big = if i == 4 {
                "X".repeat(60_000)
            } else {
                format!("small-{i}")
            };
            s.push_message(Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: format!("c{i}"),
                    content: vec![ToolResultBlock::text(big)],
                    is_error: false,
                }],
            })
            .expect("u");
        }

        let (before, after) = s
            .apply_compaction_budgeted("HANDOFF".into(), 2_000, 2, 6)
            .expect("budgeted");
        assert!(after < before, "expected token drop: {before} -> {after}");
        let msgs = s.active_messages();
        assert_eq!(msgs[0].text_concat(), "HANDOFF");
        assert!(
            msgs[1].has_tool_use(),
            "must never start tail on tool_result"
        );
        // Verify oversized 60 KiB tail block was trimmed safely
        let last_json = serde_json::to_string(msgs.last().expect("last").as_ref()).expect("ser");
        assert!(last_json.contains("bytes trimmed during compaction"));
        assert!(last_json.len() < 32_000);
    }

    #[test]
    fn compaction_with_zero_tail_never_panics() {
        let mut s = SessionStore::create(None, "claude-sonnet-5-5".into(), "STATIC".into(), vec![])
            .expect("create");
        s.push_message(Message::user_text("initial")).expect("u0");
        s.push_message(Message::assistant_text("reply"))
            .expect("a0");
        s.push_message(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "c1".into(),
                content: vec![ToolResultBlock::text("X".repeat(50_000))],
                is_error: false,
            }],
        })
        .expect("tr");

        let (before, after) = s
            .apply_compaction_budgeted("DENSE HANDOFF".into(), 4_000, 0, 6)
            .expect("zero-tail compaction must not panic");
        assert!(after < before, "expected drop: {before} -> {after}");
        assert_eq!(s.active_messages().len(), 1);
        assert_eq!(s.active_messages()[0].text_concat(), "DENSE HANDOFF");
    }

    #[test]
    fn effective_context_tokens_uses_provider_usage_and_resets_on_compaction() {
        let mut s = SessionStore::create(None, "claude-sonnet-5-5".into(), "STATIC".into(), vec![])
            .expect("create");
        s.push_message(Message::user_text("short prompt"))
            .expect("u0");
        // Provider reports 50_000 actual billed input tokens (e.g. dense JSON/CJK/PDF)
        s.record_usage(
            1,
            Usage {
                input_tokens: 100,
                cache_read_tokens: 49_900,
                ..Default::default()
            },
        )
        .expect("usage");
        assert_eq!(s.effective_context_tokens(), 50_000);

        // Appending a trailing assistant reply adds its estimate on top of 50_000
        let reply = Message::assistant_text("A".repeat(400));
        let reply_est = reply.estimated_tokens();
        s.push_message(reply).expect("a0");
        assert_eq!(s.effective_context_tokens(), 50_000 + reply_est);

        // Compaction invalidates the stale pre-compaction Usage anchor immediately
        let (before, after) = s
            .apply_compaction_budgeted("COMPACT".into(), 4_000, 0, 4)
            .expect("compact");
        assert_eq!(before, 50_000 + reply_est);
        assert!(
            after < 500,
            "post-compaction must not reuse stale 50k usage: {after}"
        );
    }

    #[test]
    fn cold_prune_shrinks_stale_tool_results_preserves_handles_and_replays_from_jsonl() {
        let dir = std::env::temp_dir().join(format!("carnot-coldprune-{}", uuid::Uuid::new_v4()));
        let mut s = SessionStore::create(
            Some(&dir),
            "claude-sonnet-5-5".into(),
            "STATIC".into(),
            vec![],
        )
        .expect("create");
        let path = s.path.clone().expect("path");

        s.push_message(Message::user_text("goal")).expect("u0");
        // Old bulky tool result WITH a CAS handle pointer
        s.push_message(Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "old1".into(),
                content: vec![ToolResultBlock::text(format!(
                    "{}\n[spilled to .carnot/h/abcdef123456.txt]",
                    "L".repeat(30_000)
                ))],
                is_error: false,
            }],
        })
        .expect("old tr");
        // Recent tail (4 messages) that must remain verbatim
        for i in 0..4 {
            s.push_message(Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: format!("recent{i}"),
                    content: vec![ToolResultBlock::text("R".repeat(5_000))],
                    is_error: false,
                }],
            })
            .expect("recent");
        }

        let reclaimed = s.prune_cold_tool_results(4, 2_000, 10_000).expect("prune");
        assert!(reclaimed > 25_000, "reclaimed={reclaimed}");

        // Second call immediately after must be a no-op (already pruned!)
        let second = s.prune_cold_tool_results(4, 2_000, 1_000).expect("prune2");
        assert_eq!(second, 0);

        // Verify persisted JSONL replays identically on reload
        let reloaded = SessionStore::load(&path).expect("reload");
        let old_json = serde_json::to_string(reloaded.active_messages()[1].as_ref()).expect("ser");
        assert!(old_json.contains("[pruned cold tool_result:"));
        assert!(old_json.contains(".carnot/h/abcdef123456.txt"));
        let recent_json =
            serde_json::to_string(reloaded.active_messages()[5].as_ref()).expect("ser");
        assert!(recent_json.len() > 5_000, "recent tail must remain intact");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn repair_dangling_tool_uses_pairs_uncompleted_calls_and_is_idempotent() {
        let dir = std::env::temp_dir().join(format!("carnot-repair-{}", uuid::Uuid::new_v4()));
        let mut s = SessionStore::create(
            Some(&dir),
            "claude-sonnet-5-5".into(),
            "STATIC".into(),
            vec![],
        )
        .expect("create");
        let path = s.path.clone().expect("path");

        s.push_message(Message::user_text("run tool")).expect("u0");
        s.push_message(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call-crash-1".into(),
                name: "Bash".into(),
                input: serde_json::json!({"command": "echo hi"}),
                invalid_json: None,
                provider_meta: None,
                origin: None,
            }],
        })
        .expect("a0");

        let mut reloaded = SessionStore::load(&path).expect("reload");
        assert_eq!(reloaded.repair_dangling_tool_uses().expect("repair"), 1);
        assert_eq!(reloaded.repair_dangling_tool_uses().expect("idempotent"), 0);
        let last = reloaded.active_messages().last().expect("last");
        assert_eq!(last.role, Role::User);
        assert!(last.has_tool_result());

        let _ = fs::remove_dir_all(&dir);
    }
}
