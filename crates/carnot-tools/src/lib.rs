//! The 5 Canonical Core Tools (`Bash`, `Read`, `Edit`, `Write`, `Grep`),
//! tree-sitter Tier-0 silent syntax delta & structural outliner, disk-coherent read tracker,
//! and adaptive-yield sandboxed shell.

pub mod bash;
pub mod files;
pub mod grep;
pub mod syntax;
pub mod tracker;

use carnot_cas::CasStore;
use carnot_core::{CarnotConfig, Result, ToolEffects, ToolOutput, ToolSpec};
use carnot_sandbox::SandboxPolicy;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Extensible runtime tool trait (implemented by built-ins, Rust modules, MCP, and extensions).
pub trait Tool: Send + Sync + std::fmt::Debug {
    fn spec(&self) -> ToolSpec;

    /// Declares read/write path effects for the effect-based parallel scheduler.
    /// Defaults to a global serial barrier for safety.
    fn effects(&self, _input: &Value) -> ToolEffects {
        ToolEffects::serial()
    }

    fn execute(&self, input: Value, cancel: CancellationToken)
    -> BoxFuture<'_, Result<ToolOutput>>;

    /// Executes the tool with the provider's correlated `call_id` (used by host-RPC / MCP bridges
    /// that key approvals and UI visuals on the call id). Defaults to [`Tool::execute`].
    fn execute_with_id<'a>(
        &'a self,
        _call_id: &'a str,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'a, Result<ToolOutput>> {
        self.execute(input, cancel)
    }
}

/// Explicit pre-edit snapshot kind (never conflates newly created files with snapshot failures).
#[derive(Debug, Clone)]
pub enum UndoKind {
    Created {
        post_hash: String,
    },
    Restored {
        prev_hash: String,
        post_hash: String,
    },
}

/// One atomic pre-edit snapshot entry stored in CAS for instant rollback (`undo_last_edit`).
#[derive(Debug, Clone)]
pub struct UndoEntry {
    pub path: std::path::PathBuf,
    pub kind: UndoKind,
}

/// Registry holding built-in + extension + deferred MCP tools.
#[derive(Debug)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    order: Vec<String>,
    pub read_tracker: Arc<Mutex<tracker::ReadTracker>>,
    pub seen_cas_hashes: Arc<Mutex<HashSet<String>>>,
    pub job_registry: Arc<bash::JobRegistry>,
    pub undo_stack: Arc<Mutex<Vec<UndoEntry>>>,
    cas: Option<Arc<CasStore>>,
}

impl ToolRegistry {
    pub fn with_canonical_lid_tools(config: &CarnotConfig, cas: Arc<CasStore>) -> Self {
        let read_tracker = Arc::new(Mutex::new(tracker::ReadTracker::new()));
        let seen_cas_hashes = Arc::new(Mutex::new(HashSet::new()));
        let job_registry = Arc::new(bash::JobRegistry::new());
        let undo_stack = Arc::new(Mutex::new(Vec::new()));
        let sandbox = SandboxPolicy::for_workspace(
            config.workspace_root.clone(),
            config.modules.sandbox,
            config.modules.sandbox_network,
        )
        .with_extra_writable_roots(config.extra_writable_roots.clone())
        .with_scrubbed_env_vars(config.scrubbed_env_vars.clone());

        let mut reg = Self {
            tools: HashMap::new(),
            order: Vec::new(),
            read_tracker: Arc::clone(&read_tracker),
            seen_cas_hashes: Arc::clone(&seen_cas_hashes),
            job_registry: Arc::clone(&job_registry),
            undo_stack: Arc::clone(&undo_stack),
            cas: Some(Arc::clone(&cas)),
        };

        reg.register(Arc::new(
            bash::BashTool::new(
                config.workspace_root.clone(),
                sandbox,
                Arc::clone(&cas),
                job_registry,
                config.modules.cas_handles,
                config.cas_spill_threshold_bytes,
                config.cas_preview_max_bytes,
                if config.modules.adaptive_yield {
                    config.bash_yield_ms
                } else {
                    config.bash_max_timeout_ms
                },
                config.bash_max_timeout_ms,
                config.extra_env.clone(),
            )
            .with_redact_cas_secrets(config.redact_cas_secrets)
            .with_seen_cas_hashes(seen_cas_hashes),
        ));

        // Allow read-only access to configured `--skills-dir` and global `~/.carnot/skills`
        // without granting write access to Edit/Write/Bash sandbox.
        let mut extra_read_roots = config.extra_writable_roots.clone();
        for dir in &config.skills_dirs {
            if !extra_read_roots.iter().any(|r| r == dir) {
                extra_read_roots.push(dir.clone());
            }
        }
        if let Some(home) = std::env::var_os("HOME") {
            let global_skills = std::path::PathBuf::from(home)
                .join(".carnot")
                .join("skills");
            if global_skills.is_dir() && !extra_read_roots.iter().any(|r| r == &global_skills) {
                extra_read_roots.push(global_skills);
            }
        }

        reg.register(Arc::new(files::ReadTool::new(
            config.workspace_root.clone(),
            extra_read_roots.clone(),
            Arc::clone(&read_tracker),
        )));

        reg.register(Arc::new(files::EditTool::new(
            config.workspace_root.clone(),
            config.extra_writable_roots.clone(),
            Arc::clone(&cas),
            Arc::clone(&undo_stack),
            Arc::clone(&read_tracker),
            config.modules.tx_edit_syntax,
        )));

        reg.register(Arc::new(files::WriteTool::new(
            config.workspace_root.clone(),
            config.extra_writable_roots.clone(),
            cas,
            undo_stack,
            read_tracker,
            config.modules.tx_edit_syntax,
        )));

        reg.register(Arc::new(grep::GrepTool::new(
            config.workspace_root.clone(),
            extra_read_roots,
        )));

        reg
    }

    /// Atomically rolls back the most recent `Edit`/`Write` using its verified BLAKE3 CAS snapshot.
    /// Refuses to clobber external modifications made after the edit.
    pub fn undo_last_edit(&self) -> Result<Option<std::path::PathBuf>> {
        let entry = self.undo_stack.lock().ok().and_then(|mut st| st.pop());
        let Some(entry) = entry else {
            return Ok(None);
        };
        let expected_post = match &entry.kind {
            UndoKind::Created { post_hash } | UndoKind::Restored { post_hash, .. } => post_hash,
        };
        let current_bytes = std::fs::read(&entry.path)?;
        let current_hash = blake3::hash(&current_bytes).to_hex().to_string();
        if current_hash != *expected_post {
            // Put entry back and refuse to silently overwrite concurrent external edits!
            if let Ok(mut st) = self.undo_stack.lock() {
                st.push(entry);
            }
            return Err(carnot_core::CarnotError::Tool {
                tool: "Undo".into(),
                message: "file was modified externally since last Carnot edit; refusing to clobber"
                    .into(),
            });
        }
        match entry.kind {
            UndoKind::Restored { prev_hash, .. } => {
                let Some(ref cas) = self.cas else {
                    return Ok(None);
                };
                let prev_bytes = cas.get_bytes_verified(&prev_hash)?;
                files::atomic_write_raw_bytes(&entry.path, &prev_bytes)?;
            }
            UndoKind::Created { .. } => {
                std::fs::remove_file(&entry.path)?;
            }
        }
        self.invalidate_read_path(&entry.path);
        Ok(Some(entry.path))
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.spec().name;
        if !self.tools.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Removes a registered tool by exact name, falling back to a deterministic case-insensitive
    /// match (first in registration order) before Turn-1 freeze.
    pub fn remove(&mut self, name: &str) -> bool {
        let key = if self.tools.contains_key(name) {
            Some(name.to_string())
        } else {
            self.order
                .iter()
                .find(|k| k.eq_ignore_ascii_case(name))
                .cloned()
        };
        if let Some(k) = key {
            self.tools.remove(&k);
            self.order.retain(|n| n != &k);
            true
        } else {
            false
        }
    }

    /// Returns all registered tool specifications in deterministic canonical order
    /// (`(defer_loading, name)`) with recursively sorted JSON schema keys so provider wire
    /// prefixes stay byte-identical regardless of module or async MCP registration order.
    pub fn specs(&self) -> Vec<ToolSpec> {
        let mut out: Vec<ToolSpec> = self
            .order
            .iter()
            .filter_map(|n| {
                self.tools.get(n).map(|t| {
                    let mut spec = t.spec();
                    spec.input_schema = carnot_core::canonicalize_json(&spec.input_schema);
                    spec
                })
            })
            .collect();
        out.sort_by(|a, b| {
            a.defer_loading
                .cmp(&b.defer_loading)
                .then_with(|| a.name.cmp(&b.name))
        });
        out
    }

    /// Returns all tools marked `defer_loading: true` in deterministic sorted order.
    pub fn deferred_tools(&self) -> Vec<Arc<dyn Tool>> {
        let mut out: Vec<Arc<dyn Tool>> = self
            .order
            .iter()
            .filter_map(|n| self.tools.get(n).cloned())
            .filter(|t| t.spec().defer_loading)
            .collect();
        out.sort_by_key(|t| t.spec().name);
        out
    }

    /// Resets disk-read tracking and seen CAS dedup hashes (called automatically after
    /// `Rewind-Receipt`, compaction, cold-pruning, or `load_session` so the model is never
    /// handed an "unchanged" or "identical output" stub for forgotten turns).
    pub fn reset_read_tracker(&self) {
        if let Ok(mut guard) = self.read_tracker.lock() {
            guard.clear();
        }
        if let Ok(mut seen) = self.seen_cas_hashes.lock() {
            seen.clear();
        }
    }

    /// Invalidates a single tracked path (called when a speculative eager `Read` is discarded).
    pub fn invalidate_read_path(&self, path: &std::path::Path) {
        if let Ok(mut guard) = self.read_tracker.lock() {
            guard.invalidate(path);
        }
    }
}

/// Returns the canonical 5-tool schema set (~650 tokens total, ultra-compact).
pub fn canonical_tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "Bash".into(),
            description: "Run shell command in workspace sandbox (auto-yields long jobs).".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "command": { "type": "string" },
                    "cwd": { "type": "string", "description": "Subdir in workspace" },
                    "job": { "type": "string", "description": "Collect yielded job ID" },
                    "kill": { "type": "boolean" },
                    "timeout": { "type": "integer", "description": "Seconds" }
                }
            }),
            defer_loading: false,
        },
        ToolSpec {
            name: "Read".into(),
            description: "Read file (`N\\ttext`), image, or directory outline. Supports `symbol` (`Type::method`/`Class.method`) and `outline: true`. Unchanged re-reads return a stub; modified re-reads return a diff.".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["file_path"],
                "properties": {
                    "file_path": { "type": "string" },
                    "offset": { "type": "integer", "description": "1-based start line" },
                    "limit": { "type": "integer" },
                    "symbol": { "type": "string" },
                    "outline": { "type": "boolean" }
                }
            }),
            defer_loading: false,
        },
        ToolSpec {
            name: "Edit".into(),
            description: "Atomically edit a file via `start_line`+`end_line` (1-based inclusive, avoids repeating old text) OR `old_string`, with `new_string` (empty deletes). Batch all hunks to a file in `edits` (non-overlapping ranges use original line numbers).".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["file_path"],
                "properties": {
                    "file_path": { "type": "string" },
                    "start_line": { "type": "integer" },
                    "end_line": { "type": "integer" },
                    "old_string": { "type": "string" },
                    "new_string": { "type": "string" },
                    "replace_all": { "type": "boolean" },
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["new_string"],
                            "properties": {
                                "start_line": { "type": "integer" },
                                "end_line": { "type": "integer" },
                                "old_string": { "type": "string" },
                                "new_string": { "type": "string" },
                                "replace_all": { "type": "boolean" }
                            }
                        }
                    }
                }
            }),
            defer_loading: false,
        },
        ToolSpec {
            name: "Write".into(),
            description: "Atomically write a new or full file.".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": { "type": "string" },
                    "content": { "type": "string" }
                }
            }),
            defer_loading: false,
        },
        ToolSpec {
            name: "Grep".into(),
            description: "Regex search across workspace files (skips .git/target/node_modules/binaries).".into(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["pattern"],
                "properties": {
                    "pattern": { "type": "string" },
                    "path": { "type": "string" },
                    "ext": { "type": "string" },
                    "glob": { "type": "string" },
                    "case_insensitive": { "type": "boolean" },
                    "limit": { "type": "integer" }
                }
            }),
            defer_loading: false,
        },
    ]
}
