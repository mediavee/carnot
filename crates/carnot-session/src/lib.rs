//! Append-only, branching JSONL v2 session engine (Opus I1 + S-1..S-7 hardened).
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
        created_unix_ms: u64,
        model: String,
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
}

/// Active view projected for the next provider request (`Arc<Message>` — zero image cloning).
#[derive(Debug, Clone)]
pub struct ProjectedContext {
    pub static_system_prompt: String,
    pub tools: Vec<ToolSpec>,
    pub prefix_hash: String,
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
    canonical_tools.sort_by(|a, b| a.name.cmp(&b.name));
    let tools_json = serde_json::to_vec(&canonical_tools)?;
    hasher.update(&tools_json);
    Ok(hasher.finalize().to_hex().to_string())
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Live session backed by a persistent buffered JSONL writer.
///
/// Only exposes append, trailing-only rewind (`apply_rewind`), and safe compaction
/// (`apply_compaction`) — making mid-history mutation impossible by type construction.
#[derive(Debug)]
pub struct SessionStore {
    pub session_id: String,
    pub path: Option<PathBuf>,
    writer: Option<File>,
    pub model: String,
    pub static_system_prompt: String,
    pub tools: Vec<ToolSpec>,
    pub prefix_hash: String,
    entries: Vec<SessionEntry>,
    active_messages: Vec<Arc<Message>>,
    checkpoint_index: Option<usize>,
    last_message_id: Option<String>,
}

impl SessionStore {
    pub fn create(
        session_dir: Option<&Path>,
        model: String,
        static_system_prompt: String,
        tools: Vec<ToolSpec>,
    ) -> Result<Self> {
        let session_id = uuid::Uuid::now_v7().to_string();
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
            path,
            writer,
            model: model.clone(),
            static_system_prompt: static_system_prompt.clone(),
            tools: tools.clone(),
            prefix_hash: prefix_hash.clone(),
            entries: Vec::new(),
            active_messages: Vec::new(),
            checkpoint_index: None,
            last_message_id: None,
        };

        store.append_entry(
            SessionEntry::Header {
                version: 2,
                session_id,
                created_unix_ms: now_unix_ms(),
                model,
                static_system_prompt,
                tools,
                prefix_hash,
            },
            true,
        )?;
        Ok(store)
    }

    /// Replays a JSONL session from disk, truncating any crash-truncated final line before
    /// reopening for append (fixes Opus S-4 & N1!).
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read(path)?;
        let mut entries = Vec::new();
        let mut offset: usize = 0;
        let mut valid_end: usize = 0;
        let mut truncated_tail = false;

        while offset < raw.len() {
            let line_end = raw[offset..]
                .iter()
                .position(|&b| b == b'\n')
                .map(|rel| offset + rel + 1)
                .unwrap_or(raw.len());
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

        if truncated_tail {
            let f = OpenOptions::new().write(true).open(path)?;
            f.set_len(valid_end as u64)?;
            f.sync_data()?;
        }

        let Some(SessionEntry::Header {
            session_id,
            model,
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

        let append_file = OpenOptions::new().create(true).append(true).open(path)?;
        let mut store = Self {
            session_id,
            path: Some(path.to_path_buf()),
            writer: Some(append_file),
            model,
            static_system_prompt,
            tools,
            prefix_hash,
            entries: Vec::new(),
            active_messages: Vec::new(),
            checkpoint_index: None,
            last_message_id: None,
        };

        for entry in entries {
            store.apply_in_memory(&entry);
            store.entries.push(entry);
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

    /// Safe compaction: adjusts `tail_start` backward so no orphan `tool_result` is kept (S-1)
    /// and strips `Thinking` / `redacted_thinking` from the kept tail so Preserved Thinking stays valid (S-2).
    pub fn apply_compaction(&mut self, summary: String, requested_keep_tail: usize) -> Result<()> {
        let n = self.active_messages.len();
        let mut tail_start = n.saturating_sub(requested_keep_tail);
        // S-1: Never start kept tail on a `tool_result` without its preceding `assistant(tool_use)`!
        while tail_start > 0 && self.active_messages[tail_start].has_tool_result() {
            tail_start -= 1;
        }
        let effective_keep_tail = n.saturating_sub(tail_start);
        let entry = SessionEntry::Compaction {
            id: uuid::Uuid::now_v7().to_string(),
            summary_message: Message::system_note(summary),
            kept_tail_count: effective_keep_tail,
            strip_thinking_from_tail: true,
            timestamp_unix_ms: now_unix_ms(),
        };
        self.append_entry(entry, true)
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

    pub fn tokens_since_checkpoint(&self) -> u64 {
        let start = self.checkpoint_index.map(|i| i + 1).unwrap_or(0);
        self.active_messages
            .get(start..)
            .unwrap_or(&[])
            .iter()
            .map(|m| m.estimated_tokens())
            .sum()
    }

    pub fn project_context(&self) -> ProjectedContext {
        ProjectedContext {
            static_system_prompt: self.static_system_prompt.clone(),
            tools: self.tools.clone(),
            prefix_hash: self.prefix_hash.clone(),
            messages: self.active_messages.clone(),
            checkpoint_index: self.checkpoint_index,
        }
    }

    /// Disk-first unbuffered append: write complete line (+ optional `sync_data`) BEFORE mutating
    /// memory (fixes Opus S-3: no stale bytes retained in a BufWriter after a failed flush!).
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
            SessionEntry::Header { .. } | SessionEntry::Usage { .. } => {}
            SessionEntry::Message { message, .. } => {
                self.active_messages.push(Arc::new(message.clone()));
            }
            SessionEntry::Checkpoint { message_index, .. } => {
                if *message_index < self.active_messages.len() {
                    self.checkpoint_index = Some(*message_index);
                }
            }
            SessionEntry::Rewind {
                checkpoint_message_index,
                receipt_message,
                ..
            } => {
                let keep_len = (*checkpoint_message_index + 1).min(self.active_messages.len());
                self.active_messages.truncate(keep_len);
                self.active_messages.push(Arc::new(receipt_message.clone()));
                self.checkpoint_index = Some(self.active_messages.len() - 1);
            }
            SessionEntry::Compaction {
                summary_message,
                kept_tail_count,
                strip_thinking_from_tail,
                ..
            } => {
                let n = self.active_messages.len();
                let tail_start = n.saturating_sub(*kept_tail_count);
                let tail: Vec<Arc<Message>> = self.active_messages[tail_start..]
                    .iter()
                    .map(|m| {
                        if *strip_thinking_from_tail {
                            Arc::new(m.without_thinking())
                        } else {
                            m.clone()
                        }
                    })
                    .collect();
                self.active_messages.clear();
                self.active_messages.push(Arc::new(summary_message.clone()));
                self.active_messages.extend(tail);
                self.checkpoint_index = Some(0);
            }
        }
    }
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

        // N1 regression: appending after recovery and reloading a second time MUST succeed!
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
}
