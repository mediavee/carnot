//! Clean, ultra-fast interactive TUI for Carnot (mini Claude Code / OpenCode style).
//!
//! Architecture:
//!   - Full-screen alternate screen with RAII + panic-hook terminal restoration
//!   - 0% idle CPU: adaptive 12 FPS tick only while streaming / running tools
//!   - Streaming Markdown renderer with syntax-highlighted code blocks, tables & wrapping
//!   - Live in-place tool cards (`call_id`-tracked) with compact `Edit` diffs (`similar`)
//!   - Interactive overlays: Slash command autocomplete, Model Picker (`Ctrl+P`),
//!     `AskUser` selector modal, and single-key Tool Approval modal (`/approve`)
//!   - Multi-line composer with bracketed paste, cursor editing, and prompt history

mod editor;
mod markdown;
mod theme;

use self::editor::PromptEditor;
use self::markdown::{display_width, render_markdown, truncate_display, wrap_spans};
use self::theme::Palette;
use crate::repl::{clip_one_line, fmt_tokens, parse_effort, shorten_path, summarize_tool_call};
use carnot::{
    Agent, AgentEvent, PermissionBroker, ProviderRegistry, ReasoningEffort, RunOutcome,
    ToolGateDecision, UserQuestion, UserQuestionAnswer, UserQuestionBroker,
};
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};
use ratatui::{Frame, Terminal};
use serde_json::Value;
use similar::{ChangeTag, TextDiff};
use std::collections::{HashSet, VecDeque};
use std::io::{self, Stdout};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, mpsc, oneshot};

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[derive(Debug, Clone, Copy)]
struct SlashSpec {
    name: &'static str,
    usage: &'static str,
    desc: &'static str,
}

const SLASH_COMMANDS: &[SlashSpec] = &[
    SlashSpec {
        name: "model",
        usage: "/model [id]",
        desc: "Open model picker or switch model live",
    },
    SlashSpec {
        name: "effort",
        usage: "/effort [lvl]",
        desc: "Set reasoning effort (low|medium|high|xhigh|max)",
    },
    SlashSpec {
        name: "thinking",
        usage: "/thinking",
        desc: "Toggle reasoning display (^T)",
    },
    SlashSpec {
        name: "approve",
        usage: "/approve [on|off]",
        desc: "Toggle interactive tool approvals",
    },
    SlashSpec {
        name: "tools",
        usage: "/tools",
        desc: "List active & deferred tools",
    },
    SlashSpec {
        name: "compact",
        usage: "/compact [focus]",
        desc: "Run cache-warm semantic compaction",
    },
    SlashSpec {
        name: "fork",
        usage: "/fork",
        desc: "Fork session into a new branch JSONL",
    },
    SlashSpec {
        name: "undo",
        usage: "/undo",
        desc: "Revert last Edit/Write via CAS",
    },
    SlashSpec {
        name: "cost",
        usage: "/cost",
        desc: "Show turns, context, cache hit & USD cost",
    },
    SlashSpec {
        name: "cache",
        usage: "/cache",
        desc: "Show prompt-cache audit, prefix hash & TTL tiers",
    },
    SlashSpec {
        name: "fleet",
        usage: "/fleet",
        desc: "Show live status of background subagent workers",
    },
    SlashSpec {
        name: "session",
        usage: "/session",
        desc: "Show session ID, prefix hash & workspace",
    },
    SlashSpec {
        name: "clear",
        usage: "/clear",
        desc: "Clear screen transcript (^L)",
    },
    SlashSpec {
        name: "help",
        usage: "/help",
        desc: "Show keybindings & commands",
    },
    SlashSpec {
        name: "exit",
        usage: "/exit",
        desc: "Quit Carnot",
    },
];

/// Startup metadata shown in the TUI header & welcome card.
#[derive(Debug, Clone)]
pub struct TuiLaunchInfo {
    pub default_provider: String,
    pub provider_configs: Vec<PathBuf>,
    pub active_extensions: Vec<String>,
}

/// RAII terminal guard that guarantees raw mode / alt screen / mouse / paste cleanup.
struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalGuard {
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(
            stdout,
            EnterAlternateScreen,
            EnableMouseCapture,
            EnableBracketedPaste
        )?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        // Ensure panics also restore the terminal cleanly
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = disable_raw_mode();
            let mut out = io::stdout();
            let _ = execute!(
                out,
                DisableBracketedPaste,
                DisableMouseCapture,
                LeaveAlternateScreen
            );
            prev_hook(info);
        }));

        Ok(Self { terminal })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            DisableBracketedPaste,
            DisableMouseCapture,
            LeaveAlternateScreen
        );
        let _ = self.terminal.show_cursor();
    }
}

enum UiMsg {
    Term(Event),
    Agent(AgentEvent),
    TurnFinished(carnot::Result<RunOutcome>),
    CompactionFinished(carnot::Result<(u64, u64, bool)>),
    PermissionReq {
        effective_tool: String,
        summary: String,
        reply: oneshot::Sender<PermissionReply>,
    },
    QuestionReq {
        questions: Vec<UserQuestion>,
        reply: oneshot::Sender<Vec<UserQuestionAnswer>>,
    },
}

#[derive(Debug, Clone, Copy)]
enum PermissionReply {
    AllowOnce,
    AllowAlways,
    Deny,
}

#[derive(Debug)]
struct TuiQuestionBroker {
    tx: mpsc::UnboundedSender<UiMsg>,
}

impl UserQuestionBroker for TuiQuestionBroker {
    fn ask<'a>(
        &'a self,
        _request_id: &'a str,
        questions: &'a [UserQuestion],
    ) -> carnot_ext::BoxFuture<'a, carnot::Result<Vec<UserQuestionAnswer>>> {
        let qs = questions.to_vec();
        let tx = self.tx.clone();
        Box::pin(async move {
            let (reply_tx, reply_rx) = oneshot::channel();
            if tx
                .send(UiMsg::QuestionReq {
                    questions: qs.clone(),
                    reply: reply_tx,
                })
                .is_err()
            {
                return Ok(Vec::new());
            }
            Ok(reply_rx.await.unwrap_or_default())
        })
    }
}

#[derive(Debug)]
struct TuiPermissionBroker {
    tx: mpsc::UnboundedSender<UiMsg>,
    allowed_always: Mutex<HashSet<String>>,
}

impl PermissionBroker for TuiPermissionBroker {
    fn request_permission(
        &self,
        _call_id: &str,
        tool_name: &str,
        input: &Value,
    ) -> carnot_ext::BoxFuture<'_, carnot::Result<ToolGateDecision>> {
        let tool_name = tool_name.to_string();
        let input = input.clone();
        Box::pin(async move {
            let effective_tool = if tool_name == "CallDeferredTool" {
                input
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or(&tool_name)
                    .to_string()
            } else {
                tool_name.clone()
            };
            if matches!(
                effective_tool.as_str(),
                "Read" | "Grep" | "ToolSearch" | "AskUser" | "MemorySearch" | "LspQuery"
            ) || self.allowed_always.lock().await.contains(&effective_tool)
            {
                return Ok(ToolGateDecision::Allow);
            }

            let summary = summarize_tool_call(&tool_name, &input);
            let (reply_tx, reply_rx) = oneshot::channel();
            if self
                .tx
                .send(UiMsg::PermissionReq {
                    effective_tool: effective_tool.clone(),
                    summary,
                    reply: reply_tx,
                })
                .is_err()
            {
                return Ok(ToolGateDecision::Deny {
                    reason: "TUI channel closed".into(),
                });
            }

            match reply_rx.await.unwrap_or(PermissionReply::Deny) {
                PermissionReply::AllowOnce => Ok(ToolGateDecision::Allow),
                PermissionReply::AllowAlways => {
                    self.allowed_always.lock().await.insert(effective_tool);
                    Ok(ToolGateDecision::Allow)
                }
                PermissionReply::Deny => Ok(ToolGateDecision::Deny {
                    reason: format!("denied by user in TUI for `{effective_tool}`"),
                }),
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolStatus {
    Running,
    Success,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NoticeKind {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone)]
enum TranscriptItem {
    User {
        text: String,
    },
    Assistant {
        text: String,
    },
    Thinking {
        text: String,
        started_at: Instant,
        duration_ms: Option<u64>,
    },
    Tool {
        call_id: String,
        name: String,
        summary: String,
        status: ToolStatus,
        started_at: Instant,
        duration_ms: Option<u64>,
        progress: Option<String>,
        output_preview: String,
        edit_diff: Option<(String, String)>,
    },
    Notice {
        kind: NoticeKind,
        title: String,
        body: Option<String>,
    },
}

struct PendingPermission {
    effective_tool: String,
    summary: String,
    reply: oneshot::Sender<PermissionReply>,
}

struct PendingQuestion {
    questions: Vec<UserQuestion>,
    current_idx: usize,
    selected_row: usize,
    /// For multi-select questions: toggled option indices.
    checked: HashSet<usize>,
    /// Free-text input buffer when typing custom answer (or open-ended question).
    custom_input: String,
    typing_custom: bool,
    answers: Vec<UserQuestionAnswer>,
    reply: oneshot::Sender<Vec<UserQuestionAnswer>>,
}

impl PendingQuestion {
    fn current(&self) -> Option<&UserQuestion> {
        self.questions.get(self.current_idx)
    }

    fn reset_step(&mut self) {
        self.selected_row = 0;
        self.checked.clear();
        self.custom_input.clear();
        self.typing_custom = self.current().is_some_and(|q| q.options.is_empty());
    }

    fn total_rows(&self) -> usize {
        self.current()
            .map(|q| {
                if q.options.is_empty() {
                    1
                } else {
                    q.options.len() + 1
                }
            })
            .unwrap_or(1)
    }

    fn is_custom_row(&self) -> bool {
        self.current()
            .is_some_and(|q| q.options.is_empty() || self.selected_row >= q.options.len())
    }
}

#[derive(Debug, Clone)]
struct ModelChoice {
    id: String,
    label: String,
    provider: String,
}

struct ModelPicker {
    all: Vec<ModelChoice>,
    filter: String,
    selected: usize,
}

impl ModelPicker {
    fn filtered(&self) -> Vec<&ModelChoice> {
        let q = self.filter.trim().to_ascii_lowercase();
        if q.is_empty() {
            return self.all.iter().collect();
        }
        self.all
            .iter()
            .filter(|m| {
                m.id.to_ascii_lowercase().contains(&q)
                    || m.label.to_ascii_lowercase().contains(&q)
                    || m.provider.to_ascii_lowercase().contains(&q)
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct WorkerSummary {
    state: String,
    model: String,
    turns: u64,
    tools_called: u64,
    cost_usd: f64,
    activity: String,
}

struct TuiApp {
    agent: Arc<Agent>,
    palette: Palette,
    launch: TuiLaunchInfo,
    ui_tx: mpsc::UnboundedSender<UiMsg>,

    // Header / status metadata
    session_id: String,
    model: String,
    effort: ReasoningEffort,
    sandbox_badge: &'static str,
    workspace_short: String,
    git_branch: Option<String>,

    // Live metrics
    ctx_tokens: u64,
    ctx_window: u64,
    cost_usd: f64,
    cache_hit_pct: f64,
    cache_breaks: u64,
    total_turns: u64,
    workers: std::collections::BTreeMap<String, WorkerSummary>,

    // Transcript + cached lines
    items: Vec<TranscriptItem>,
    cached_lines: Vec<Line<'static>>,
    cache_width: usize,
    cache_dirty: bool,
    scroll_offset: usize,
    auto_follow: bool,
    last_viewport_height: usize,

    // Interactive toggles & state
    editor: PromptEditor,
    show_thinking: bool,
    expand_tools: bool,
    approve_enabled: bool,
    running: bool,
    compacting: bool,
    turn_started_at: Option<Instant>,
    spinner_frame: usize,
    last_ctrl_c: Option<Instant>,
    slash_selected: usize,

    // Modals / overlays
    perm_queue: VecDeque<PendingPermission>,
    active_perm: Option<PendingPermission>,
    question_queue: VecDeque<PendingQuestion>,
    active_question: Option<PendingQuestion>,
    model_picker: Option<ModelPicker>,
}

impl TuiApp {
    async fn new(
        agent: Arc<Agent>,
        launch: TuiLaunchInfo,
        ui_tx: mpsc::UnboundedSender<UiMsg>,
    ) -> Self {
        let session_id = agent.session_id().await;
        let model = agent.current_model().await;
        let effort = agent.current_effort().await;
        let ctx_tokens = agent.current_context_tokens().await;
        let ctx_window = agent.current_context_window().await.max(1);
        let m = agent.metrics().await;

        let sandbox_badge = if !agent.config.modules.sandbox {
            "sandbox:off"
        } else if agent.config.modules.sandbox_network {
            "sandbox:net"
        } else {
            "sandbox:offline"
        };
        let workspace_short = shorten_path(&agent.config.workspace_root);
        let git_branch = detect_git_branch(&agent.config.workspace_root);

        Self {
            agent,
            palette: Palette::detect(),
            launch,
            ui_tx,
            session_id,
            model,
            effort,
            sandbox_badge,
            workspace_short,
            git_branch,
            ctx_tokens,
            ctx_window,
            cost_usd: m.total_usage.cost_usd,
            cache_hit_pct: m.cache_hit_rate() * 100.0,
            cache_breaks: m.cache_breaks,
            total_turns: m.turns,
            workers: std::collections::BTreeMap::new(),
            items: Vec::new(),
            cached_lines: Vec::new(),
            cache_width: 0,
            cache_dirty: true,
            scroll_offset: 0,
            auto_follow: true,
            last_viewport_height: 20,
            editor: PromptEditor::default(),
            show_thinking: false,
            expand_tools: false,
            approve_enabled: false,
            running: false,
            compacting: false,
            turn_started_at: None,
            spinner_frame: 0,
            last_ctrl_c: None,
            slash_selected: 0,
            perm_queue: VecDeque::new(),
            active_perm: None,
            question_queue: VecDeque::new(),
            active_question: None,
            model_picker: None,
        }
    }

    fn is_animating(&self) -> bool {
        self.running || self.compacting || self.workers.values().any(|w| w.state == "running")
    }

    fn push_notice(&mut self, kind: NoticeKind, title: impl Into<String>, body: Option<String>) {
        self.items.push(TranscriptItem::Notice {
            kind,
            title: title.into(),
            body,
        });
        self.cache_dirty = true;
    }

    fn finalize_open_thinking(&mut self) {
        if let Some(TranscriptItem::Thinking {
            started_at,
            duration_ms,
            ..
        }) = self.items.last_mut()
            && duration_ms.is_none()
        {
            *duration_ms = Some(started_at.elapsed().as_millis() as u64);
            self.cache_dirty = true;
        }
    }

    fn handle_agent_event(&mut self, ev: AgentEvent) {
        match ev {
            AgentEvent::TurnStart {
                context_tokens_est, ..
            } => {
                self.ctx_tokens = context_tokens_est;
            }
            AgentEvent::ThinkingDelta { delta } => {
                if let Some(TranscriptItem::Thinking {
                    text, duration_ms, ..
                }) = self.items.last_mut()
                    && duration_ms.is_none()
                {
                    text.push_str(&delta);
                } else {
                    self.items.push(TranscriptItem::Thinking {
                        text: delta,
                        started_at: Instant::now(),
                        duration_ms: None,
                    });
                }
                self.cache_dirty = true;
            }
            AgentEvent::TextDelta { delta } => {
                self.finalize_open_thinking();
                if let Some(TranscriptItem::Assistant { text }) = self.items.last_mut() {
                    text.push_str(&delta);
                } else {
                    self.items.push(TranscriptItem::Assistant { text: delta });
                }
                self.cache_dirty = true;
            }
            AgentEvent::ToolStart {
                call_id,
                tool,
                input,
                ..
            } => {
                self.finalize_open_thinking();
                let summary = summarize_tool_call(&tool, &input);
                let edit_diff = extract_edit_diff(&tool, &input);
                self.items.push(TranscriptItem::Tool {
                    call_id,
                    name: tool,
                    summary,
                    status: ToolStatus::Running,
                    started_at: Instant::now(),
                    duration_ms: None,
                    progress: None,
                    output_preview: String::new(),
                    edit_diff,
                });
                self.cache_dirty = true;
            }
            AgentEvent::ToolUpdate { call_id, message } => {
                if let Some(TranscriptItem::Tool { progress, .. }) = self
                    .items
                    .iter_mut()
                    .rev()
                    .find(|it| matches!(it, TranscriptItem::Tool { call_id: id, .. } if id == &call_id))
                {
                    *progress = Some(clip_one_line(&message, 80));
                    self.cache_dirty = true;
                }
            }
            AgentEvent::ToolEnd {
                call_id,
                tool,
                is_error,
                duration_ms,
                output_preview,
            } => {
                if let Some(TranscriptItem::Tool {
                    status,
                    duration_ms: dur,
                    output_preview: prev,
                    progress,
                    ..
                }) = self.items.iter_mut().rev().find(
                    |it| matches!(it, TranscriptItem::Tool { call_id: id, .. } if id == &call_id),
                ) {
                    *status = if is_error {
                        ToolStatus::Error
                    } else {
                        ToolStatus::Success
                    };
                    *dur = Some(duration_ms);
                    *prev = output_preview;
                    *progress = None;
                } else {
                    self.items.push(TranscriptItem::Tool {
                        call_id,
                        name: tool.clone(),
                        summary: tool,
                        status: if is_error {
                            ToolStatus::Error
                        } else {
                            ToolStatus::Success
                        },
                        started_at: Instant::now(),
                        duration_ms: Some(duration_ms),
                        progress: None,
                        output_preview,
                        edit_diff: None,
                    });
                }
                self.cache_dirty = true;
            }
            AgentEvent::CompactionApplied {
                mode,
                tokens_before,
                tokens_after,
                semantic_llm,
            } => {
                let kind = if semantic_llm { "semantic" } else { "det" };
                self.push_notice(
                    NoticeKind::Warning,
                    format!(
                        "Compacted ({mode}/{kind}): ~{} → ~{} tokens",
                        fmt_tokens(tokens_before),
                        fmt_tokens(tokens_after)
                    ),
                    None,
                );
            }
            AgentEvent::ModuleAdvisory { signal, message } => {
                self.push_notice(
                    NoticeKind::Warning,
                    format!("[{signal}] {}", clip_one_line(&message, 100)),
                    None,
                );
            }
            AgentEvent::CacheBreakDetected { turn, reason, .. } => {
                self.cache_breaks += 1;
                self.push_notice(
                    NoticeKind::Warning,
                    format!("Cache break at turn {turn}: {reason}"),
                    None,
                );
            }
            AgentEvent::WorkerUpdated {
                worker,
                state,
                model,
                turns,
                tools_called,
                cost_usd,
                activity,
            } => {
                if state == "closed" {
                    self.workers.remove(&worker);
                } else {
                    self.workers.insert(
                        worker,
                        WorkerSummary {
                            state,
                            model,
                            turns,
                            tools_called,
                            cost_usd,
                            activity,
                        },
                    );
                }
            }
            AgentEvent::UsageRecorded { usage, .. } => {
                self.cost_usd += usage.cost_usd;
                let denom = usage.input_tokens
                    + usage.cache_read_tokens
                    + usage.cache_write_5m_tokens
                    + usage.cache_write_1h_tokens;
                if denom > 0 {
                    self.cache_hit_pct = (usage.cache_read_tokens as f64 / denom as f64) * 100.0;
                }
            }
            AgentEvent::RunSettled { metrics, .. } => {
                self.finalize_open_thinking();
                self.cost_usd = metrics.total_usage.cost_usd;
                self.cache_hit_pct = metrics.cache_hit_rate() * 100.0;
                self.cache_breaks = metrics.cache_breaks;
                self.total_turns = metrics.turns;
                self.cache_dirty = true;
            }
            AgentEvent::Error { message } => {
                self.finalize_open_thinking();
                self.push_notice(NoticeKind::Error, message, None);
            }
            _ => {}
        }
    }

    async fn refresh_agent_snapshot(&mut self) {
        self.model = self.agent.current_model().await;
        self.effort = self.agent.current_effort().await;
        self.ctx_tokens = self.agent.current_context_tokens().await;
        self.ctx_window = self.agent.current_context_window().await.max(1);
        let m = self.agent.metrics().await;
        self.cost_usd = m.total_usage.cost_usd;
        self.cache_hit_pct = m.cache_hit_rate() * 100.0;
        self.total_turns = m.turns;
    }

    fn slash_matches(&self) -> Vec<&'static SlashSpec> {
        if self.active_perm.is_some()
            || self.active_question.is_some()
            || self.model_picker.is_some()
        {
            return Vec::new();
        }
        let text = self.editor.text();
        if text.contains('\n') {
            return Vec::new();
        }
        let Some(rest) = text.strip_prefix('/') else {
            return Vec::new();
        };
        // Hide autocomplete once user starts typing arguments
        if rest.contains(' ') {
            return Vec::new();
        }
        let q = rest.to_ascii_lowercase();
        SLASH_COMMANDS
            .iter()
            .filter(|s| q.is_empty() || s.name.starts_with(&q) || s.name.contains(&q))
            .collect()
    }

    fn open_model_picker(&mut self) {
        let mut all = Vec::new();
        if let Ok(reg) = ProviderRegistry::load_default(&self.launch.provider_configs) {
            all = reg
                .available_models(false)
                .into_iter()
                .map(|m| ModelChoice {
                    id: m.id,
                    label: m.label,
                    provider: if m.configured {
                        format!("{} ✓", m.provider)
                    } else {
                        m.provider
                    },
                })
                .collect();
        }
        if all.is_empty() {
            self.push_notice(
                NoticeKind::Warning,
                format!(
                    "Current model: {} (context: {})",
                    self.model,
                    fmt_tokens(self.ctx_window)
                ),
                None,
            );
            return;
        }
        let selected = all.iter().position(|m| m.id == self.model).unwrap_or(0);
        self.model_picker = Some(ModelPicker {
            all,
            filter: String::new(),
            selected,
        });
    }

    async fn switch_model(&mut self, model_id: &str) {
        match ProviderRegistry::load_default(&self.launch.provider_configs) {
            Ok(reg) => {
                let prov_id = reg
                    .catalog()
                    .provider_for_model(model_id)
                    .unwrap_or(&self.launch.default_provider)
                    .to_string();
                match reg.build_provider(&prov_id, Some(model_id)) {
                    Ok(provider) => {
                        let cw = reg.context_window_for_model(model_id);
                        let max_out = reg.max_output_tokens_for_model(model_id);
                        match self
                            .agent
                            .set_model(model_id.to_string(), provider, cw, max_out)
                            .await
                        {
                            Ok(()) => {
                                self.refresh_agent_snapshot().await;
                                self.push_notice(
                                    NoticeKind::Success,
                                    format!("Switched model → {model_id} ({prov_id})"),
                                    None,
                                );
                            }
                            Err(e) => {
                                self.push_notice(
                                    NoticeKind::Error,
                                    format!("Model switch failed: {e}"),
                                    None,
                                );
                            }
                        }
                    }
                    Err(e) => {
                        self.push_notice(
                            NoticeKind::Error,
                            format!("Provider build failed: {e}"),
                            None,
                        );
                    }
                }
            }
            Err(e) => {
                self.push_notice(NoticeKind::Error, format!("Catalog load failed: {e}"), None);
            }
        }
    }

    /// Returns `true` if the application should exit.
    async fn handle_slash_command(&mut self, cmd: &str) -> bool {
        let mut parts = cmd.split_whitespace();
        let verb = parts.next().unwrap_or("");
        let rest = cmd.strip_prefix(verb).map(str::trim).unwrap_or("");

        match verb {
            "exit" | "quit" | "q" => return true,
            "clear" => {
                self.items.clear();
                self.scroll_offset = 0;
                self.auto_follow = true;
                self.cache_dirty = true;
            }
            "help" | "h" => {
                let body = [
                    "  /model [id]         Open model picker (^P) or switch live",
                    "  /effort [level]     low | medium | high | xhigh | max",
                    "  /thinking           Toggle reasoning stream (^T)",
                    "  /approve [on|off]   Toggle interactive tool approvals",
                    "  /tools              List active & deferred tools",
                    "  /compact [focus]    Run cache-warm semantic compaction",
                    "  /fork               Fork session into a new branch JSONL",
                    "  /undo               Revert last Edit/Write via CAS",
                    "  /cost               Show turns, context %, cache & USD",
                    "  /session            Show session ID, prefix hash & workspace",
                    "  /clear              Clear visual transcript (^L)",
                    "  /exit               Quit Carnot",
                    "",
                    "Shortcuts: Enter send · Shift+Enter / ^J newline · Esc abort turn",
                    "           ^P models · ^T thinking · ^O expand tools · PgUp/PgDn scroll",
                ]
                .join("\n");
                self.push_notice(NoticeKind::Info, "Carnot Commands & Shortcuts", Some(body));
            }
            "model" => {
                if rest.is_empty() {
                    self.open_model_picker();
                } else {
                    self.switch_model(rest).await;
                }
            }
            "effort" => {
                if rest.is_empty() {
                    self.push_notice(
                        NoticeKind::Info,
                        format!("Current effort: {:?}", self.effort),
                        None,
                    );
                } else if let Some(eff) = parse_effort(rest) {
                    self.agent.set_effort(eff).await;
                    self.effort = eff;
                    self.push_notice(NoticeKind::Success, format!("Effort → {eff:?}"), None);
                } else {
                    self.push_notice(
                        NoticeKind::Warning,
                        "Usage: /effort low|medium|high|xhigh|max",
                        None,
                    );
                }
            }
            "thinking" => {
                self.show_thinking = !self.show_thinking;
                self.cache_dirty = true;
                self.push_notice(
                    NoticeKind::Info,
                    format!(
                        "Thinking display: {}",
                        if self.show_thinking { "on" } else { "off" }
                    ),
                    None,
                );
            }
            "approve" => {
                let target = match rest.to_ascii_lowercase().as_str() {
                    "on" | "true" | "1" => true,
                    "off" | "false" | "0" => false,
                    "" => !self.approve_enabled,
                    _ => {
                        self.push_notice(NoticeKind::Warning, "Usage: /approve [on|off]", None);
                        return false;
                    }
                };
                self.approve_enabled = target;
                if self.approve_enabled {
                    self.agent
                        .set_permission_broker(Arc::new(TuiPermissionBroker {
                            tx: self.ui_tx.clone(),
                            allowed_always: Mutex::new(HashSet::new()),
                        }))
                        .await;
                    self.push_notice(
                        NoticeKind::Success,
                        "Interactive tool approvals enabled",
                        None,
                    );
                } else {
                    self.agent.clear_permission_broker().await;
                    self.push_notice(
                        NoticeKind::Info,
                        "Interactive tool approvals disabled",
                        None,
                    );
                }
            }
            "tools" => {
                let (eager, deferred) = self.agent.tool_names();
                let body = format!(
                    "  Active ({}):   {}\n  Deferred ({}): {}",
                    eager.len(),
                    eager.join(", "),
                    deferred.len(),
                    if deferred.is_empty() {
                        "none".to_string()
                    } else {
                        deferred.join(", ")
                    }
                );
                self.push_notice(NoticeKind::Info, "Registered Tools", Some(body));
            }
            "cost" | "stats" => {
                self.refresh_agent_snapshot().await;
                let m = self.agent.metrics().await;
                let u = &m.total_usage;
                let pct = (self.ctx_tokens as f64 / self.ctx_window.max(1) as f64) * 100.0;
                let body = format!(
                    "  Turns: {} · Requests: {} · Edits: {}/{}\n  Context: {}/{} ({pct:.1}%) · Compactions: {}\n  Cache hit: {:.1}% (in={} cr={} cw5={} cw1={} out={})\n  Total cost: ${:.4}",
                    m.turns,
                    m.requests,
                    m.edit_attempts.saturating_sub(m.edit_failures),
                    m.edit_attempts,
                    fmt_tokens(self.ctx_tokens),
                    fmt_tokens(self.ctx_window),
                    m.compactions_applied,
                    m.cache_hit_rate() * 100.0,
                    u.input_tokens,
                    u.cache_read_tokens,
                    u.cache_write_5m_tokens,
                    u.cache_write_1h_tokens,
                    u.output_tokens,
                    u.cost_usd
                );
                self.push_notice(NoticeKind::Info, "Session Telemetry", Some(body));
            }
            "cache" => {
                let m = self.agent.metrics().await;
                let u = &m.total_usage;
                let ph = self.agent.prefix_hash().await;
                let body = format!(
                    "  Prefix hash:  {}\n  TTL policy:   {:?} (Anthropic 4-slot: system 1h · Turn-1 1h · checkpoint 5m · tail 5m)\n  Hit rate:     {:.1}% (hits={} misses={} unexpected_breaks={})\n  Tokens:       read={} · write_5m={} · write_1h={} · uncached_in={}",
                    ph,
                    self.agent.config.cache_ttl,
                    m.cache_hit_rate() * 100.0,
                    m.prefix_cache_hits,
                    m.prefix_cache_misses,
                    m.cache_breaks,
                    fmt_tokens(u.cache_read_tokens),
                    fmt_tokens(u.cache_write_5m_tokens),
                    fmt_tokens(u.cache_write_1h_tokens),
                    fmt_tokens(u.input_tokens),
                );
                self.push_notice(NoticeKind::Info, "Prompt Cache Audit", Some(body));
            }
            "fleet" => {
                if self.workers.is_empty() {
                    self.push_notice(
                        NoticeKind::Info,
                        "Fleet Status",
                        Some("  No active background workers.".into()),
                    );
                } else {
                    let mut rows = Vec::new();
                    let mut total_w_cost = 0.0;
                    for (name, w) in &self.workers {
                        total_w_cost += w.cost_usd;
                        let icon = match w.state.as_str() {
                            "running" => "⚡",
                            "idle" => "⏸",
                            "failed" => "✗",
                            _ => "·",
                        };
                        rows.push(format!(
                            "  {icon} {name:<14} {:<8} {} · turns={} tools={} ${:.4} · {}",
                            w.state, w.model, w.turns, w.tools_called, w.cost_usd, w.activity
                        ));
                    }
                    rows.push(format!(
                        "  Total tracked worker cost: ${total_w_cost:.4} (session total: ${:.4})",
                        self.cost_usd
                    ));
                    self.push_notice(
                        NoticeKind::Info,
                        format!("Fleet Status ({} worker(s))", self.workers.len()),
                        Some(rows.join("\n")),
                    );
                }
            }
            "session" => {
                let ph = self.agent.prefix_hash().await;
                let ext = if self.launch.active_extensions.is_empty() {
                    "none".to_string()
                } else {
                    self.launch.active_extensions.join(", ")
                };
                let body = format!(
                    "  Session:    {}\n  Model:      {} (effort: {:?})\n  Sandbox:    {}\n  Workspace:  {}\n  Extensions: {}\n  Prefix:     {}",
                    self.session_id,
                    self.model,
                    self.effort,
                    self.sandbox_badge,
                    self.agent.config.workspace_root.display(),
                    ext,
                    ph
                );
                self.push_notice(NoticeKind::Info, "Session Info", Some(body));
            }
            "fork" => {
                if self.running || self.compacting {
                    self.push_notice(
                        NoticeKind::Warning,
                        "Wait for the current turn to finish before forking",
                        None,
                    );
                    return false;
                }
                match self.agent.fork_session().await {
                    Ok((new_id, path)) => {
                        self.session_id = new_id.clone();
                        let detail = path
                            .as_deref()
                            .map(|p| format!("  Saved branch: {}", shorten_path(p)));
                        self.push_notice(
                            NoticeKind::Success,
                            format!("Forked session → {}", &new_id[..8.min(new_id.len())]),
                            detail,
                        );
                    }
                    Err(e) => {
                        self.push_notice(NoticeKind::Error, format!("Fork failed: {e}"), None);
                    }
                }
            }
            "undo" => match self.agent.undo_last_edit() {
                Ok(Some(p)) => self.push_notice(
                    NoticeKind::Success,
                    format!("Reverted {}", shorten_path(&p)),
                    None,
                ),
                Ok(None) => self.push_notice(NoticeKind::Info, "Nothing to undo", None),
                Err(e) => {
                    self.push_notice(NoticeKind::Error, format!("Undo failed: {e}"), None);
                }
            },
            "compact" => {
                if self.running || self.compacting {
                    self.push_notice(
                        NoticeKind::Warning,
                        "Wait for the current turn to finish before compacting",
                        None,
                    );
                    return false;
                }
                self.compacting = true;
                self.turn_started_at = Some(Instant::now());
                let focus = (!rest.is_empty()).then(|| rest.to_string());
                let agent = Arc::clone(&self.agent);
                let tx = self.ui_tx.clone();
                tokio::spawn(async move {
                    let res = agent.compact_now_with_focus(focus.as_deref()).await;
                    let _ = tx.send(UiMsg::CompactionFinished(res));
                });
            }
            other => {
                self.push_notice(
                    NoticeKind::Warning,
                    format!("Unknown command `/{other}` — try /help"),
                    None,
                );
            }
        }
        false
    }

    fn start_prompt(&mut self, prompt: String) {
        self.items.push(TranscriptItem::User {
            text: prompt.clone(),
        });
        self.cache_dirty = true;
        self.auto_follow = true;
        self.running = true;
        self.turn_started_at = Some(Instant::now());
        self.spinner_frame = 0;

        let agent = Arc::clone(&self.agent);
        let tx = self.ui_tx.clone();
        tokio::spawn(async move {
            let res = agent.prompt(prompt).await;
            let _ = tx.send(UiMsg::TurnFinished(res));
        });
    }

    fn scroll_up(&mut self, lines: usize) {
        let max_scroll = self
            .cached_lines
            .len()
            .saturating_sub(self.last_viewport_height);
        if self.auto_follow {
            self.scroll_offset = max_scroll;
            self.auto_follow = false;
        }
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
    }

    fn scroll_down(&mut self, lines: usize) {
        let max_scroll = self
            .cached_lines
            .len()
            .saturating_sub(self.last_viewport_height);
        self.scroll_offset = (self.scroll_offset + lines).min(max_scroll);
        if self.scroll_offset >= max_scroll {
            self.auto_follow = true;
        }
    }

    fn jump_bottom(&mut self) {
        self.auto_follow = true;
        let max_scroll = self
            .cached_lines
            .len()
            .saturating_sub(self.last_viewport_height);
        self.scroll_offset = max_scroll;
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollUp => self.scroll_up(3),
            MouseEventKind::ScrollDown => self.scroll_down(3),
            _ => {}
        }
    }

    /// Returns `true` if the application should exit.
    async fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }

        // Priority 1: Tool Approval Modal
        if let Some(perm) = self.active_perm.take() {
            match key.code {
                KeyCode::Char('y' | 'Y') => {
                    let _ = perm.reply.send(PermissionReply::AllowOnce);
                    self.active_perm = self.perm_queue.pop_front();
                }
                KeyCode::Char('a' | 'A') => {
                    let _ = perm.reply.send(PermissionReply::AllowAlways);
                    self.active_perm = self.perm_queue.pop_front();
                }
                KeyCode::Char('n' | 'N') | KeyCode::Esc => {
                    let _ = perm.reply.send(PermissionReply::Deny);
                    self.active_perm = self.perm_queue.pop_front();
                }
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    let _ = perm.reply.send(PermissionReply::Deny);
                    self.active_perm = self.perm_queue.pop_front();
                    self.agent.abort().await;
                }
                _ => {
                    self.active_perm = Some(perm);
                }
            }
            return false;
        }

        // Priority 2: AskUser Interactive Modal
        if self.active_question.is_some() {
            self.handle_question_key(key);
            return false;
        }

        // Priority 3: Interactive Model Picker
        if self.model_picker.is_some() {
            self.handle_model_picker_key(key).await;
            return false;
        }

        // Global Ctrl shortcuts
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') => {
                    if self.running {
                        self.agent.abort().await;
                        self.push_notice(NoticeKind::Warning, "Aborting current turn…", None);
                        return false;
                    }
                    if !self.editor.is_empty() {
                        self.editor.clear();
                        self.last_ctrl_c = Some(Instant::now());
                        return false;
                    }
                    if let Some(prev) = self.last_ctrl_c
                        && prev.elapsed() < Duration::from_millis(1600)
                    {
                        return true;
                    }
                    self.last_ctrl_c = Some(Instant::now());
                    return false;
                }
                KeyCode::Char('d') if self.editor.is_empty() && !self.running => {
                    return true;
                }
                KeyCode::Char('t') => {
                    self.show_thinking = !self.show_thinking;
                    self.cache_dirty = true;
                    return false;
                }
                KeyCode::Char('o') => {
                    self.expand_tools = !self.expand_tools;
                    self.cache_dirty = true;
                    return false;
                }
                KeyCode::Char('p') if !self.running => {
                    self.open_model_picker();
                    return false;
                }
                KeyCode::Char('l') => {
                    self.items.clear();
                    self.scroll_offset = 0;
                    self.auto_follow = true;
                    self.cache_dirty = true;
                    return false;
                }
                KeyCode::Char('j') => {
                    self.editor.insert_char('\n');
                    return false;
                }
                KeyCode::Char('a') => {
                    self.editor.move_home();
                    return false;
                }
                KeyCode::Char('e') => {
                    self.editor.move_end();
                    return false;
                }
                KeyCode::Char('w') => {
                    self.editor.delete_word_back();
                    return false;
                }
                KeyCode::Char('u') => {
                    if self.editor.is_empty() {
                        self.scroll_up(self.last_viewport_height / 2);
                    } else {
                        self.editor.kill_to_line_start();
                    }
                    return false;
                }
                KeyCode::Char('k') => {
                    self.editor.kill_to_line_end();
                    return false;
                }
                KeyCode::Char('d') if !self.editor.is_empty() => {
                    self.editor.delete();
                    return false;
                }
                _ => {}
            }
        }

        // Scroll shortcuts
        match key.code {
            KeyCode::PageUp => {
                self.scroll_up(self.last_viewport_height.saturating_sub(2).max(4));
                return false;
            }
            KeyCode::PageDown => {
                self.scroll_down(self.last_viewport_height.saturating_sub(2).max(4));
                return false;
            }
            KeyCode::End if self.editor.is_empty() => {
                self.jump_bottom();
                return false;
            }
            KeyCode::Esc => {
                if self.running {
                    self.agent.abort().await;
                    self.push_notice(NoticeKind::Warning, "Aborting current turn…", None);
                } else if !self.editor.is_empty() {
                    self.editor.clear();
                }
                return false;
            }
            _ => {}
        }

        // Slash autocomplete navigation when active
        let slash = self.slash_matches();
        if !slash.is_empty() {
            match key.code {
                KeyCode::Up => {
                    self.slash_selected = self
                        .slash_selected
                        .checked_sub(1)
                        .unwrap_or(slash.len() - 1);
                    return false;
                }
                KeyCode::Down => {
                    self.slash_selected = (self.slash_selected + 1) % slash.len();
                    return false;
                }
                KeyCode::Tab => {
                    let chosen = slash[self.slash_selected.min(slash.len() - 1)];
                    let needs_arg =
                        matches!(chosen.name, "model" | "effort" | "approve" | "compact");
                    if needs_arg {
                        self.editor.set_text(&format!("/{} ", chosen.name));
                    } else {
                        self.editor.set_text(&format!("/{}", chosen.name));
                    }
                    self.slash_selected = 0;
                    return false;
                }
                KeyCode::Enter
                    if !key
                        .modifiers
                        .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
                {
                    let chosen = slash[self.slash_selected.min(slash.len() - 1)];
                    self.editor.clear();
                    self.slash_selected = 0;
                    return self.handle_slash_command(chosen.name).await;
                }
                _ => {}
            }
        }

        // Standard composer editing
        match key.code {
            KeyCode::Enter => {
                if key
                    .modifiers
                    .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT)
                {
                    self.editor.insert_char('\n');
                } else {
                    let raw = self.editor.take_and_record();
                    let trimmed = raw.trim();
                    if trimmed.is_empty() {
                        return false;
                    }
                    if let Some(cmd) = trimmed.strip_prefix('/') {
                        return self.handle_slash_command(cmd).await;
                    }
                    if self.running || self.compacting {
                        self.push_notice(
                            NoticeKind::Warning,
                            "A turn is already running — press Esc to abort first",
                            None,
                        );
                        self.editor.set_text(&raw);
                    } else {
                        self.start_prompt(trimmed.to_string());
                    }
                }
            }
            KeyCode::Backspace => {
                if key.modifiers.contains(KeyModifiers::ALT) {
                    self.editor.delete_word_back();
                } else {
                    self.editor.backspace();
                }
                self.slash_selected = 0;
            }
            KeyCode::Delete => self.editor.delete(),
            KeyCode::Left => {
                if key.modifiers.contains(KeyModifiers::ALT) {
                    self.editor.move_word_left();
                } else {
                    self.editor.move_left();
                }
            }
            KeyCode::Right => {
                if key.modifiers.contains(KeyModifiers::ALT) {
                    self.editor.move_word_right();
                } else {
                    self.editor.move_right();
                }
            }
            KeyCode::Home => self.editor.move_home(),
            KeyCode::End => self.editor.move_end(),
            KeyCode::Up => self.editor.move_up_or_history(),
            KeyCode::Down => self.editor.move_down_or_history(),
            KeyCode::Char(c) => {
                self.editor.insert_char(c);
                self.slash_selected = 0;
            }
            _ => {}
        }
        false
    }

    fn handle_question_key(&mut self, key: KeyEvent) {
        let Some(mut pq) = self.active_question.take() else {
            return;
        };
        let total_rows = pq.total_rows();
        let is_multi = pq.current().is_some_and(|q| q.multi_select);

        match key.code {
            KeyCode::Up
                if !pq.typing_custom || pq.current().is_some_and(|q| !q.options.is_empty()) =>
            {
                pq.selected_row = pq.selected_row.checked_sub(1).unwrap_or(total_rows - 1);
                pq.typing_custom = pq.is_custom_row();
                self.active_question = Some(pq);
            }
            KeyCode::Down
                if !pq.typing_custom || pq.current().is_some_and(|q| !q.options.is_empty()) =>
            {
                pq.selected_row = (pq.selected_row + 1) % total_rows;
                pq.typing_custom = pq.is_custom_row();
                self.active_question = Some(pq);
            }
            KeyCode::Char(' ') if is_multi && !pq.is_custom_row() => {
                if !pq.checked.remove(&pq.selected_row) {
                    pq.checked.insert(pq.selected_row);
                }
                self.active_question = Some(pq);
            }
            KeyCode::Backspace if pq.typing_custom => {
                pq.custom_input.pop();
                self.active_question = Some(pq);
            }
            KeyCode::Char(c) if pq.typing_custom => {
                pq.custom_input.push(c);
                self.active_question = Some(pq);
            }
            KeyCode::Char(c) if !pq.typing_custom && c.is_ascii_digit() => {
                if let Some(d) = c.to_digit(10)
                    && d >= 1
                    && let Some(q) = pq.current()
                    && (d as usize) <= q.options.len()
                {
                    pq.selected_row = (d as usize) - 1;
                    if is_multi && !pq.checked.remove(&pq.selected_row) {
                        pq.checked.insert(pq.selected_row);
                    }
                }
                self.active_question = Some(pq);
            }
            KeyCode::Enter => {
                if let Some(q) = pq.current().cloned() {
                    let mut selected = Vec::new();
                    let mut text = None;
                    if pq.is_custom_row() {
                        let t = pq.custom_input.trim().to_string();
                        if !t.is_empty() {
                            text = Some(t);
                        }
                    } else if q.multi_select {
                        for (idx, opt) in q.options.iter().enumerate() {
                            if pq.checked.contains(&idx) {
                                selected.push(opt.label.clone());
                            }
                        }
                        if selected.is_empty()
                            && let Some(opt) = q.options.get(pq.selected_row)
                        {
                            selected.push(opt.label.clone());
                        }
                    } else if let Some(opt) = q.options.get(pq.selected_row) {
                        selected.push(opt.label.clone());
                    }
                    pq.answers.push(UserQuestionAnswer {
                        id: q.id,
                        selected,
                        text,
                    });
                }

                pq.current_idx += 1;
                if pq.current_idx >= pq.questions.len() {
                    let _ = pq.reply.send(pq.answers);
                    self.active_question = self.question_queue.pop_front();
                } else {
                    pq.reset_step();
                    self.active_question = Some(pq);
                }
            }
            KeyCode::Esc => {
                // Return answers collected so far (empty for remaining)
                for q in pq.questions.iter().skip(pq.current_idx) {
                    pq.answers.push(UserQuestionAnswer {
                        id: q.id.clone(),
                        selected: Vec::new(),
                        text: None,
                    });
                }
                let _ = pq.reply.send(pq.answers);
                self.active_question = self.question_queue.pop_front();
            }
            _ => {
                self.active_question = Some(pq);
            }
        }
    }

    async fn handle_model_picker_key(&mut self, key: KeyEvent) {
        let Some(mut picker) = self.model_picker.take() else {
            return;
        };
        let count = picker.filtered().len();

        match key.code {
            KeyCode::Esc => {
                // Close without switching
            }
            KeyCode::Up if count > 0 => {
                picker.selected = picker.selected.checked_sub(1).unwrap_or(count - 1);
                self.model_picker = Some(picker);
            }
            KeyCode::Down if count > 0 => {
                picker.selected = (picker.selected + 1) % count;
                self.model_picker = Some(picker);
            }
            KeyCode::Enter => {
                let chosen_id = picker
                    .filtered()
                    .get(picker.selected.min(count.saturating_sub(1)))
                    .map(|m| m.id.clone());
                if let Some(id) = chosen_id {
                    self.switch_model(&id).await;
                }
            }
            KeyCode::Backspace => {
                picker.filter.pop();
                picker.selected = 0;
                self.model_picker = Some(picker);
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                picker.filter.push(c);
                picker.selected = 0;
                self.model_picker = Some(picker);
            }
            _ => {
                self.model_picker = Some(picker);
            }
        }
    }

    fn rebuild_transcript_cache(&mut self, width: usize) {
        let content_w = width.saturating_sub(2).max(20);
        let p = &self.palette;
        let spin = SPINNER[self.spinner_frame % SPINNER.len()];
        let mut lines: Vec<Line<'static>> = Vec::new();

        for item in &self.items {
            if !lines.is_empty() {
                lines.push(Line::default());
            }
            match item {
                TranscriptItem::User { text } => {
                    let bar = Span::styled("┃ ", p.accent_bold());
                    let msg_style = if p.color {
                        Style::default().fg(p.fg).bg(p.surface)
                    } else {
                        Style::default().add_modifier(Modifier::BOLD)
                    };
                    for raw_line in text.lines() {
                        let spans = vec![Span::styled(raw_line.to_string(), msg_style)];
                        wrap_spans(
                            &spans,
                            content_w,
                            vec![bar.clone()],
                            vec![bar.clone()],
                            &mut lines,
                        );
                    }
                }
                TranscriptItem::Assistant { text } => {
                    let md_lines = render_markdown(text, content_w, p);
                    for l in md_lines {
                        let mut prefixed = vec![Span::raw("  ")];
                        prefixed.extend(l.spans);
                        lines.push(Line::from(prefixed));
                    }
                }
                TranscriptItem::Thinking {
                    text,
                    started_at,
                    duration_ms,
                } => {
                    let ms = duration_ms.unwrap_or_else(|| started_at.elapsed().as_millis() as u64);
                    let secs = format!("{:.1}s", ms as f64 / 1000.0);
                    let active = duration_ms.is_none();
                    if self.show_thinking {
                        let icon = if active { spin } else { "◇" };
                        lines.push(Line::from(vec![
                            Span::styled(format!("  {icon} Thinking "), p.thinking()),
                            Span::styled(format!("({secs})"), p.muted()),
                        ]));
                        for raw_line in text.lines() {
                            let spans = vec![Span::styled(raw_line.to_string(), p.thinking())];
                            let pref = vec![Span::styled("  │ ", p.border())];
                            wrap_spans(&spans, content_w, pref.clone(), pref, &mut lines);
                        }
                    } else if active {
                        lines.push(Line::from(vec![
                            Span::styled(format!("  {spin} Thinking… "), p.thinking()),
                            Span::styled(format!("({secs})  ^T to expand"), p.muted()),
                        ]));
                    } else {
                        lines.push(Line::from(vec![
                            Span::styled("  ◇ ", p.thinking()),
                            Span::styled(format!("Thought for {secs}  (^T to show)"), p.muted()),
                        ]));
                    }
                }
                TranscriptItem::Tool {
                    name,
                    summary,
                    status,
                    started_at,
                    duration_ms,
                    progress,
                    output_preview,
                    edit_diff,
                    ..
                } => {
                    let (icon, icon_style, dur_text) = match status {
                        ToolStatus::Running => {
                            let ms = started_at.elapsed().as_millis();
                            (spin, p.warning(), format!("{:.1}s", ms as f64 / 1000.0))
                        }
                        ToolStatus::Success => (
                            "✓",
                            p.success(),
                            duration_ms.map(fmt_duration_ms).unwrap_or_default(),
                        ),
                        ToolStatus::Error => (
                            "✗",
                            p.danger(),
                            duration_ms.map(fmt_duration_ms).unwrap_or_default(),
                        ),
                    };

                    let mut header_spans = vec![
                        Span::raw("  "),
                        Span::styled(icon.to_string(), icon_style),
                        Span::raw(" "),
                        Span::styled(
                            truncate_display(summary, content_w.saturating_sub(24)),
                            if *status == ToolStatus::Error {
                                p.danger()
                            } else {
                                p.text()
                            },
                        ),
                    ];
                    if let Some((old_s, new_s)) = edit_diff {
                        let (adds, dels) = count_diff_stats(old_s, new_s);
                        if adds > 0 || dels > 0 {
                            header_spans.push(Span::raw("  "));
                            if adds > 0 {
                                header_spans.push(Span::styled(format!("+{adds}"), p.success()));
                            }
                            if adds > 0 && dels > 0 {
                                header_spans.push(Span::raw(" "));
                            }
                            if dels > 0 {
                                header_spans.push(Span::styled(format!("-{dels}"), p.danger()));
                            }
                        }
                    }
                    if !dur_text.is_empty() {
                        header_spans.push(Span::styled(format!("  {dur_text}"), p.muted()));
                    }
                    lines.push(Line::from(header_spans));

                    if let Some(prog) = progress {
                        lines.push(Line::from(vec![
                            Span::styled("    ↳ ", p.border()),
                            Span::styled(prog.clone(), p.muted()),
                        ]));
                    }

                    if *status == ToolStatus::Error && !output_preview.is_empty() {
                        let err_line = clip_one_line(output_preview, content_w.saturating_sub(8));
                        lines.push(Line::from(vec![
                            Span::styled("    ↳ ", p.danger()),
                            Span::styled(err_line, p.danger()),
                        ]));
                    }

                    // Compact colored diff for Edit tool
                    if let Some((old_s, new_s)) = edit_diff {
                        let max_diff_lines = if self.expand_tools { 40 } else { 6 };
                        render_diff_preview(old_s, new_s, max_diff_lines, content_w, p, &mut lines);
                    } else if self.expand_tools && !output_preview.is_empty() && name != "AskUser" {
                        let max_prev = 14;
                        let preview_lines: Vec<&str> = output_preview.lines().collect();
                        for pl in preview_lines.iter().take(max_prev) {
                            lines.push(Line::from(vec![
                                Span::styled("    │ ", p.border()),
                                Span::styled(
                                    truncate_display(pl, content_w.saturating_sub(8)),
                                    p.muted(),
                                ),
                            ]));
                        }
                        if preview_lines.len() > max_prev {
                            lines.push(Line::from(vec![
                                Span::styled("    │ ", p.border()),
                                Span::styled(
                                    format!("… ({} more lines)", preview_lines.len() - max_prev),
                                    p.muted(),
                                ),
                            ]));
                        }
                    }
                }
                TranscriptItem::Notice { kind, title, body } => {
                    let (icon, style) = match kind {
                        NoticeKind::Info => ("●", p.accent()),
                        NoticeKind::Success => ("✓", p.success()),
                        NoticeKind::Warning => ("▲", p.warning()),
                        NoticeKind::Error => ("✗", p.danger()),
                    };
                    lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(icon, style),
                        Span::raw(" "),
                        Span::styled(title.clone(), style.add_modifier(Modifier::BOLD)),
                    ]));
                    if let Some(b) = body {
                        for bl in b.lines() {
                            lines.push(Line::from(vec![
                                Span::raw("  "),
                                Span::styled(
                                    truncate_display(bl, content_w.saturating_sub(4)),
                                    p.muted(),
                                ),
                            ]));
                        }
                    }
                }
            }
        }

        self.cached_lines = lines;
        self.cache_width = width;
        self.cache_dirty = false;
    }

    fn draw(&mut self, frame: &mut Frame<'_>) {
        let area = frame.area();
        if area.width < 20 || area.height < 8 {
            let msg = Paragraph::new("Terminal too small (min 20x8)").style(self.palette.warning());
            frame.render_widget(msg, area);
            return;
        }

        // Compute dynamic overlay & composer heights
        let slash_matches = self.slash_matches();
        let overlay_h: u16 = if self.active_perm.is_some() {
            5
        } else if let Some(ref pq) = self.active_question {
            (pq.total_rows() as u16 + 4).min(12)
        } else if let Some(ref mp) = self.model_picker {
            (mp.filtered().len().clamp(1, 8) as u16 + 4).min(13)
        } else if !slash_matches.is_empty() {
            (slash_matches.len().min(7) as u16) + 2
        } else {
            0
        };

        let composer_inner_w = area.width.saturating_sub(4) as usize;
        let max_composer_rows = 6usize;
        let preview_render = self.editor.render(composer_inner_w, max_composer_rows);
        let composer_h = (preview_render.total_rows.clamp(1, max_composer_rows) as u16) + 2;

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),          // Header
                Constraint::Min(4),             // Transcript
                Constraint::Length(overlay_h),  // Optional overlay
                Constraint::Length(composer_h), // Input box
                Constraint::Length(1),          // Status bar
            ])
            .split(area);

        self.draw_header(frame, chunks[0]);
        self.draw_transcript(frame, chunks[1]);
        if overlay_h > 0 {
            self.draw_overlay(frame, chunks[2], &slash_matches);
        }
        self.draw_composer(frame, chunks[3], &preview_render);
        self.draw_status_bar(frame, chunks[4]);
    }

    fn draw_header(&self, frame: &mut Frame<'_>, area: Rect) {
        let p = &self.palette;
        let sb_style = if self.sandbox_badge == "sandbox:off" {
            p.danger()
        } else if self.sandbox_badge == "sandbox:net" {
            p.warning()
        } else {
            p.success()
        };

        let mut left = vec![
            Span::styled(" ◆ carnot ", p.brand_bold()),
            Span::styled(format!("v{}", env!("CARGO_PKG_VERSION")), p.muted()),
            Span::styled(" · ", p.border()),
            Span::styled(self.model.clone(), p.accent()),
            Span::styled(" · ", p.border()),
            Span::styled(format!("{:?}", self.effort).to_lowercase(), p.muted()),
            Span::styled(" · ", p.border()),
            Span::styled(self.sandbox_badge, sb_style),
        ];

        let mut right_parts = Vec::new();
        if let Some(ref br) = self.git_branch {
            right_parts.push(format!("⎇ {br}"));
        }
        right_parts.push(self.workspace_short.clone());
        right_parts.push(self.session_id[..8.min(self.session_id.len())].to_string());
        let right_str = format!("{} ", right_parts.join(" · "));

        let left_w: usize = left.iter().map(|s| display_width(&s.content)).sum();
        let right_w = display_width(&right_str);
        let total_w = area.width as usize;

        if left_w + right_w + 2 <= total_w {
            left.push(Span::raw(" ".repeat(total_w - left_w - right_w)));
            left.push(Span::styled(right_str, p.muted()));
        } else if left_w + 1 < total_w {
            let rem = total_w - left_w - 1;
            if rem > 6 {
                left.push(Span::raw(" "));
                left.push(Span::styled(truncate_display(&right_str, rem), p.muted()));
            }
        }

        frame.render_widget(Paragraph::new(Line::from(left)).style(p.bar_bg()), area);
    }

    fn draw_transcript(&mut self, frame: &mut Frame<'_>, area: Rect) {
        self.last_viewport_height = area.height as usize;

        if self.items.is_empty() {
            self.draw_welcome(frame, area);
            return;
        }

        let width = area.width as usize;
        let has_active_spinner = self.items.iter().any(|it| {
            matches!(
                it,
                TranscriptItem::Tool {
                    status: ToolStatus::Running,
                    ..
                } | TranscriptItem::Thinking {
                    duration_ms: None,
                    ..
                }
            )
        });
        if self.cache_dirty || self.cache_width != width || has_active_spinner {
            self.rebuild_transcript_cache(width);
        }

        let total = self.cached_lines.len();
        let view_h = area.height as usize;
        let max_scroll = total.saturating_sub(view_h);
        if self.auto_follow || self.scroll_offset > max_scroll {
            self.scroll_offset = max_scroll;
        }

        let end = (self.scroll_offset + view_h).min(total);
        let visible = self.cached_lines[self.scroll_offset..end].to_vec();
        frame.render_widget(Paragraph::new(visible), area);

        if total > view_h {
            let mut sb_state = ScrollbarState::new(max_scroll).position(self.scroll_offset);
            let sb = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .track_symbol(Some("│"))
                .thumb_symbol("┃")
                .track_style(self.palette.border())
                .thumb_style(self.palette.muted());
            frame.render_stateful_widget(sb, area, &mut sb_state);
        }

        // Floating "jump to bottom" indicator when user scrolled up
        if !self.auto_follow && max_scroll > self.scroll_offset {
            let pill = " ↓ latest (End) ";
            let pw = display_width(pill) as u16;
            if area.width > pw + 2 && area.height >= 2 {
                let pill_area = Rect {
                    x: area.x + area.width - pw - 2,
                    y: area.y + area.height - 1,
                    width: pw,
                    height: 1,
                };
                frame.render_widget(Clear, pill_area);
                frame.render_widget(
                    Paragraph::new(Span::styled(pill, self.palette.selected_row())),
                    pill_area,
                );
            }
        }
    }

    fn draw_welcome(&self, frame: &mut Frame<'_>, area: Rect) {
        let p = &self.palette;
        let ext_line = if self.launch.active_extensions.is_empty() {
            "core only (pass --web --memory --subagent --lsp to enable)".to_string()
        } else {
            self.launch.active_extensions.join(" · ")
        };

        let mut lines = vec![
            Line::default(),
            Line::from(vec![
                Span::styled("  ◆ ", p.brand_bold()),
                Span::styled("Carnot", p.brand_bold()),
                Span::styled(
                    format!(
                        " v{}  —  Thermodynamic Coding Agent",
                        env!("CARGO_PKG_VERSION")
                    ),
                    p.muted(),
                ),
            ]),
            Line::default(),
            Line::from(vec![
                Span::styled("    Model       ", p.muted()),
                Span::styled(self.model.clone(), p.accent()),
                Span::styled(format!("  (effort: {:?})", self.effort), p.muted()),
            ]),
            Line::from(vec![
                Span::styled("    Workspace   ", p.muted()),
                Span::styled(self.workspace_short.clone(), p.text()),
                Span::styled(
                    self.git_branch
                        .as_deref()
                        .map(|b| format!("  ⎇ {b}"))
                        .unwrap_or_default(),
                    p.success(),
                ),
            ]),
            Line::from(vec![
                Span::styled("    Extensions  ", p.muted()),
                Span::styled(ext_line, p.text()),
            ]),
            Line::default(),
            Line::from(vec![
                Span::styled("    Tips  ", p.brand_bold()),
                Span::styled("/model", p.accent()),
                Span::styled(" or ", p.muted()),
                Span::styled("^P", p.accent()),
                Span::styled(" switch model · ", p.muted()),
                Span::styled("/approve", p.accent()),
                Span::styled(" guard tools · ", p.muted()),
                Span::styled("^T", p.accent()),
                Span::styled(" thinking · ", p.muted()),
                Span::styled("^O", p.accent()),
                Span::styled(" expand diffs", p.muted()),
            ]),
        ];

        // Vertically center lightly if plenty of space
        let top_pad = area.height.saturating_sub(lines.len() as u16) / 3;
        if top_pad > 0 {
            let mut padded = vec![Line::default(); top_pad as usize];
            padded.append(&mut lines);
            lines = padded;
        }
        frame.render_widget(Paragraph::new(lines), area);
    }

    fn draw_overlay(&self, frame: &mut Frame<'_>, area: Rect, slash_matches: &[&SlashSpec]) {
        let p = &self.palette;
        frame.render_widget(Clear, area);

        if let Some(ref perm) = self.active_perm {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(p.warning())
                .title(Span::styled(
                    format!(" ▲ Approve tool: {} ", perm.effective_tool),
                    p.warning().add_modifier(Modifier::BOLD),
                ));
            let inner = block.inner(area);
            frame.render_widget(block, area);
            let lines = vec![
                Line::from(vec![
                    Span::raw(" "),
                    Span::styled(
                        truncate_display(&perm.summary, inner.width.saturating_sub(2) as usize),
                        p.text(),
                    ),
                ]),
                Line::default(),
                Line::from(vec![
                    Span::raw(" "),
                    Span::styled("[y]", p.success().add_modifier(Modifier::BOLD)),
                    Span::styled(" Allow once   ", p.muted()),
                    Span::styled("[a]", p.accent_bold()),
                    Span::styled(" Allow always   ", p.muted()),
                    Span::styled("[n / Esc]", p.danger().add_modifier(Modifier::BOLD)),
                    Span::styled(" Deny", p.muted()),
                ]),
            ];
            frame.render_widget(Paragraph::new(lines), inner);
            return;
        }

        if let Some(ref pq) = self.active_question {
            let header_tag = pq
                .current()
                .and_then(|q| q.header.as_deref())
                .map(|h| format!(" · {h}"))
                .unwrap_or_default();
            let title = format!(
                " ? Question {}/{}{header_tag} ",
                pq.current_idx + 1,
                pq.questions.len()
            );
            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(p.warning())
                .title(Span::styled(
                    title,
                    p.warning().add_modifier(Modifier::BOLD),
                ));
            let inner = block.inner(area);
            frame.render_widget(block, area);

            if let Some(q) = pq.current() {
                let mut lines = vec![Line::from(vec![
                    Span::raw(" "),
                    Span::styled(
                        truncate_display(&q.question, inner.width.saturating_sub(2) as usize),
                        p.text().add_modifier(Modifier::BOLD),
                    ),
                ])];
                for (idx, opt) in q.options.iter().enumerate() {
                    let is_sel = idx == pq.selected_row;
                    let marker = if q.multi_select {
                        if pq.checked.contains(&idx) {
                            "[x]"
                        } else {
                            "[ ]"
                        }
                    } else if is_sel {
                        "●"
                    } else {
                        "○"
                    };
                    let desc = opt
                        .description
                        .as_deref()
                        .map(|d| format!(" — {d}"))
                        .unwrap_or_default();
                    let row_text = format!(
                        " {} {marker} {}{desc}",
                        if is_sel { "▸" } else { " " },
                        opt.label
                    );
                    lines.push(Line::from(Span::styled(
                        truncate_display(&row_text, inner.width as usize),
                        if is_sel { p.selected_row() } else { p.text() },
                    )));
                }
                // Custom answer row
                let custom_sel = pq.is_custom_row();
                let custom_label = if pq.custom_input.is_empty() {
                    if custom_sel {
                        " ▸ Custom answer: ▏".to_string()
                    } else {
                        "   Custom answer…".to_string()
                    }
                } else {
                    format!(" ▸ Custom: {}▏", pq.custom_input)
                };
                lines.push(Line::from(Span::styled(
                    truncate_display(&custom_label, inner.width as usize),
                    if custom_sel {
                        p.selected_row()
                    } else {
                        p.muted()
                    },
                )));
                lines.push(Line::from(Span::styled(
                    if q.multi_select {
                        " ↑↓ move · Space toggle · Enter submit · Esc skip"
                    } else {
                        " ↑↓ move · 1-9 pick · Enter submit · Esc skip"
                    },
                    p.muted(),
                )));
                frame.render_widget(Paragraph::new(lines), inner);
            }
            return;
        }

        if let Some(ref mp) = self.model_picker {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(p.accent())
                .title(Span::styled(
                    " ◆ Select Model (type to filter · Enter switch · Esc close) ",
                    p.accent_bold(),
                ));
            let inner = block.inner(area);
            frame.render_widget(block, area);

            let mut lines = vec![Line::from(vec![
                Span::styled(" Filter: ", p.muted()),
                Span::styled(
                    if mp.filter.is_empty() {
                        "type model or provider…".to_string()
                    } else {
                        format!("{}▏", mp.filter)
                    },
                    if mp.filter.is_empty() {
                        p.muted()
                    } else {
                        p.text()
                    },
                ),
            ])];

            let filtered = mp.filtered();
            if filtered.is_empty() {
                lines.push(Line::from(Span::styled(
                    "  (no matching models)",
                    p.muted(),
                )));
            } else {
                let max_show = inner.height.saturating_sub(1) as usize;
                let sel = mp.selected.min(filtered.len() - 1);
                let start = sel.saturating_sub(max_show.saturating_sub(1));
                for (offset, m) in filtered.iter().skip(start).take(max_show).enumerate() {
                    let idx = start + offset;
                    let is_sel = idx == sel;
                    let active_dot = if m.id == self.model { "●" } else { " " };
                    let text = format!(" {active_dot} {:<28}  [{}]", m.id, m.provider);
                    lines.push(Line::from(Span::styled(
                        truncate_display(&text, inner.width as usize),
                        if is_sel {
                            p.selected_row()
                        } else if m.id == self.model {
                            p.success()
                        } else {
                            p.text()
                        },
                    )));
                }
            }
            frame.render_widget(Paragraph::new(lines), inner);
            return;
        }

        if !slash_matches.is_empty() {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(p.border())
                .title(Span::styled(
                    " Commands (Tab complete · Enter run) ",
                    p.muted(),
                ));
            let inner = block.inner(area);
            frame.render_widget(block, area);

            let sel = self.slash_selected.min(slash_matches.len() - 1);
            let lines: Vec<Line<'static>> = slash_matches
                .iter()
                .take(7)
                .enumerate()
                .map(|(idx, s)| {
                    let row = format!(" {:<18} {}", s.usage, s.desc);
                    Line::from(Span::styled(
                        truncate_display(&row, inner.width as usize),
                        if idx == sel {
                            p.selected_row()
                        } else {
                            p.text()
                        },
                    ))
                })
                .collect();
            frame.render_widget(Paragraph::new(lines), inner);
        }
    }

    fn draw_composer(&self, frame: &mut Frame<'_>, area: Rect, rendered: &editor::RenderedEditor) {
        let p = &self.palette;
        let has_modal = self.active_perm.is_some()
            || self.active_question.is_some()
            || self.model_picker.is_some();

        let (border_style, title_span) = if self.running {
            let spin = SPINNER[self.spinner_frame % SPINNER.len()];
            let secs = self
                .turn_started_at
                .map(|t| format!("{:.1}s", t.elapsed().as_secs_f64()))
                .unwrap_or_default();
            (
                p.warning(),
                Span::styled(
                    format!(" {spin} Working… ({secs}) · Esc to abort "),
                    p.warning().add_modifier(Modifier::BOLD),
                ),
            )
        } else if self.compacting {
            let spin = SPINNER[self.spinner_frame % SPINNER.len()];
            (
                p.warning(),
                Span::styled(format!(" {spin} Compacting context… "), p.warning()),
            )
        } else if has_modal {
            (p.border(), Span::styled(" › Prompt ", p.muted()))
        } else {
            (p.brand_bold(), Span::styled(" › Prompt ", p.brand_bold()))
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(border_style)
            .title(title_span);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.editor.is_empty() {
            let placeholder = if self.running {
                "Agent is running… press Esc or Ctrl+C to abort turn"
            } else {
                "Ask anything… (/ commands · Shift+Enter newline · ^P model)"
            };
            frame.render_widget(
                Paragraph::new(Span::styled(
                    truncate_display(placeholder, inner.width as usize),
                    p.muted(),
                )),
                inner,
            );
        } else {
            let lines: Vec<Line<'static>> = rendered
                .lines
                .iter()
                .map(|s| Line::from(Span::styled(s.clone(), p.text())))
                .collect();
            frame.render_widget(Paragraph::new(lines), inner);
        }

        if !has_modal && inner.width > 0 && inner.height > 0 {
            let cx = inner.x + rendered.cursor_col.min(inner.width.saturating_sub(1));
            let cy = inner.y + rendered.cursor_row.min(inner.height.saturating_sub(1));
            frame.set_cursor_position((cx, cy));
        }
    }

    fn draw_status_bar(&self, frame: &mut Frame<'_>, area: Rect) {
        let p = &self.palette;
        let mut left = Vec::new();

        if let Some(prev) = self.last_ctrl_c
            && prev.elapsed() < Duration::from_millis(1600)
            && !self.running
        {
            left.push(Span::styled(" Press Ctrl+C again to quit ", p.warning()));
        } else if self.running {
            let spin = SPINNER[self.spinner_frame % SPINNER.len()];
            left.push(Span::styled(format!(" {spin} running "), p.warning()));
        } else if self.compacting {
            let spin = SPINNER[self.spinner_frame % SPINNER.len()];
            left.push(Span::styled(format!(" {spin} compacting "), p.warning()));
        } else {
            left.push(Span::styled(" ● ready ", p.success()));
        }

        if self.approve_enabled {
            left.push(Span::styled(" approve:on ", p.warning()));
        }
        if self.show_thinking {
            left.push(Span::styled(" think:on ", p.thinking()));
        }
        if self.expand_tools {
            left.push(Span::styled(" expand:on ", p.accent()));
        }
        if !self.workers.is_empty() {
            let w_run = self
                .workers
                .values()
                .filter(|w| w.state == "running")
                .count();
            let w_idle = self.workers.values().filter(|w| w.state == "idle").count();
            let chip = if w_run > 0 && w_idle > 0 {
                format!(" fleet:{w_run}⚡/{w_idle}⏸ ")
            } else if w_run > 0 {
                format!(" fleet:{w_run}⚡ ")
            } else {
                format!(" fleet:{w_idle}⏸ ")
            };
            let style = if w_run > 0 { p.warning() } else { p.accent() };
            left.push(Span::styled(chip, style));
        }
        if self.cache_breaks > 0 {
            left.push(Span::styled(
                format!(" cache-breaks:{} ", self.cache_breaks),
                p.danger(),
            ));
        }
        let ext_span = if !self.launch.active_extensions.is_empty() {
            let ext_chip = self
                .launch
                .active_extensions
                .iter()
                .map(|e| format!("+{e}"))
                .collect::<Vec<_>>()
                .join(" ");
            Some(Span::styled(format!(" {ext_chip} "), p.muted()))
        } else {
            None
        };

        // Context meter + cache + cost on the right
        let pct = ((self.ctx_tokens as f64 / self.ctx_window.max(1) as f64) * 100.0).min(100.0);
        let filled = ((pct / 100.0) * 6.0).round() as usize;
        let meter = format!(
            "{}{}",
            "█".repeat(filled.min(6)),
            "░".repeat(6usize.saturating_sub(filled))
        );
        let ctx_style = if pct >= 85.0 {
            p.danger()
        } else if pct >= 60.0 {
            p.warning()
        } else {
            p.success()
        };

        let right_spans = vec![
            Span::styled(
                format!(
                    "ctx {}/{} ",
                    fmt_tokens(self.ctx_tokens),
                    fmt_tokens(self.ctx_window)
                ),
                p.muted(),
            ),
            Span::styled(format!("[{meter}] {pct:.0}%"), ctx_style),
            Span::styled(
                format!(
                    " · cache {:.0}% · ${:.4} ",
                    self.cache_hit_pct, self.cost_usd
                ),
                p.muted(),
            ),
        ];

        let total_w = area.width as usize;
        let right_w: usize = right_spans.iter().map(|s| display_width(&s.content)).sum();

        // Include the `+ext` chip only when it fits comfortably alongside left & right status
        if let Some(ext) = ext_span {
            let lw: usize = left.iter().map(|s| display_width(&s.content)).sum();
            let ew = display_width(&ext.content);
            if lw + ew + right_w < total_w {
                left.push(ext);
            }
        }

        let left_w: usize = left.iter().map(|s| display_width(&s.content)).sum();
        if left_w + right_w < total_w {
            left.push(Span::raw(" ".repeat(total_w - left_w - right_w)));
            left.extend(right_spans);
        } else if left_w + 18 < total_w {
            // Compact right fallback for narrow split panes (~60-80 cols)
            let compact_right = vec![
                Span::styled(format!("{pct:.0}% "), ctx_style),
                Span::styled(format!("${:.4} ", self.cost_usd), p.muted()),
            ];
            let cw: usize = compact_right
                .iter()
                .map(|s| display_width(&s.content))
                .sum();
            if left_w + cw < total_w {
                left.push(Span::raw(" ".repeat(total_w - left_w - cw)));
                left.extend(compact_right);
            }
        } else if right_w < total_w {
            left = vec![Span::raw(" ".repeat(total_w - right_w))];
            left.extend(right_spans);
        }

        frame.render_widget(Paragraph::new(Line::from(left)).style(p.bar_bg()), area);
    }
}

fn fmt_duration_ms(ms: u64) -> String {
    if ms >= 1000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        format!("{ms}ms")
    }
}

fn extract_edit_diff(tool: &str, input: &Value) -> Option<(String, String)> {
    if tool != "Edit" {
        return None;
    }
    if let (Some(old_s), Some(new_s)) = (
        input.get("old_string").and_then(Value::as_str),
        input.get("new_string").and_then(Value::as_str),
    ) && old_s != new_s
    {
        return Some((old_s.to_string(), new_s.to_string()));
    }
    if let Some(arr) = input.get("edits").and_then(Value::as_array) {
        let mut old_parts = Vec::new();
        let mut new_parts = Vec::new();
        for item in arr {
            let o = item.get("old_string").and_then(Value::as_str).unwrap_or("");
            let n = item.get("new_string").and_then(Value::as_str).unwrap_or("");
            if o != n {
                old_parts.push(o);
                new_parts.push(n);
            }
        }
        if !old_parts.is_empty() {
            return Some((old_parts.join("\n"), new_parts.join("\n")));
        }
    }
    None
}

fn count_diff_stats(old_s: &str, new_s: &str) -> (usize, usize) {
    let diff = TextDiff::from_lines(old_s, new_s);
    let mut adds = 0usize;
    let mut dels = 0usize;
    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Insert => adds += 1,
            ChangeTag::Delete => dels += 1,
            ChangeTag::Equal => {}
        }
    }
    (adds, dels)
}

fn render_diff_preview(
    old_s: &str,
    new_s: &str,
    max_lines: usize,
    content_w: usize,
    p: &Palette,
    out: &mut Vec<Line<'static>>,
) {
    let diff = TextDiff::from_lines(old_s, new_s);
    let mut changed: Vec<(char, String, Style)> = Vec::new();
    for change in diff.iter_all_changes() {
        let line_text = change.value().trim_end_matches(['\r', '\n']).to_string();
        match change.tag() {
            ChangeTag::Delete => changed.push(('-', line_text, p.danger())),
            ChangeTag::Insert => changed.push(('+', line_text, p.success())),
            ChangeTag::Equal => {}
        }
    }
    let total = changed.len();
    for (sign, text, style) in changed.into_iter().take(max_lines) {
        out.push(Line::from(vec![
            Span::styled("    │ ", p.border()),
            Span::styled(
                format!(
                    "{sign} {}",
                    truncate_display(&text, content_w.saturating_sub(10))
                ),
                style,
            ),
        ]));
    }
    if total > max_lines {
        out.push(Line::from(vec![
            Span::styled("    │ ", p.border()),
            Span::styled(
                format!("… ({} more diff lines — ^O to expand)", total - max_lines),
                p.muted(),
            ),
        ]));
    }
}

fn detect_git_branch(workspace: &Path) -> Option<String> {
    let mut dir = workspace.to_path_buf();
    loop {
        let git_path = dir.join(".git");
        if git_path.is_dir() {
            let head = std::fs::read_to_string(git_path.join("HEAD")).ok()?;
            return parse_git_head(&head);
        }
        if git_path.is_file() {
            let content = std::fs::read_to_string(&git_path).ok()?;
            let rel = content.trim().strip_prefix("gitdir:")?.trim();
            let resolved = if Path::new(rel).is_absolute() {
                PathBuf::from(rel)
            } else {
                dir.join(rel)
            };
            let head = std::fs::read_to_string(resolved.join("HEAD")).ok()?;
            return parse_git_head(&head);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn parse_git_head(head: &str) -> Option<String> {
    let trimmed = head.trim();
    if let Some(r) = trimmed.strip_prefix("ref: refs/heads/") {
        Some(r.to_string())
    } else if trimmed.len() >= 7 {
        Some(trimmed[..7].to_string())
    } else {
        None
    }
}

/// Entry point for the interactive Carnot TUI.
pub async fn run_tui(agent: Arc<Agent>, launch: TuiLaunchInfo) -> i32 {
    let mut guard = match TerminalGuard::enter() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("failed to initialize TUI (try --plain): {e}");
            return 1;
        }
    };

    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<UiMsg>();

    // Wire AskUser interactive modal broker immediately
    agent
        .set_question_broker(Arc::new(TuiQuestionBroker { tx: ui_tx.clone() }))
        .await;

    // Background agent event forwarder
    let mut ev_stream = agent.subscribe_from(agent.latest_event_seq());
    let ev_tx = ui_tx.clone();
    let ev_task = tokio::spawn(async move {
        loop {
            let env = ev_stream.next().await;
            if ev_tx.send(UiMsg::Agent(env.event)).is_err() {
                break;
            }
        }
    });

    // Dedicated terminal input reader thread (exits cleanly in <=50ms on shutdown)
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_reader = Arc::clone(&stop_flag);
    let term_tx = ui_tx.clone();
    let reader_handle = std::thread::spawn(move || {
        while !stop_reader.load(Ordering::Relaxed) {
            match event::poll(Duration::from_millis(50)) {
                Ok(true) => match event::read() {
                    Ok(ev) => {
                        if term_tx.send(UiMsg::Term(ev)).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                },
                Ok(false) => {}
                Err(_) => break,
            }
        }
    });

    let mut app = TuiApp::new(Arc::clone(&agent), launch, ui_tx).await;
    let mut tick = tokio::time::interval(Duration::from_millis(80));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let mut should_quit = false;

    loop {
        if guard.terminal.draw(|f| app.draw(f)).is_err() {
            break;
        }
        if should_quit {
            break;
        }

        // Wait reactively: only poll the animation timer while running/compacting (0% CPU when idle!)
        let msg = if app.is_animating() {
            tokio::select! {
                m = ui_rx.recv() => m,
                _ = tick.tick() => {
                    app.spinner_frame = app.spinner_frame.wrapping_add(1);
                    continue;
                }
            }
        } else {
            ui_rx.recv().await
        };

        let Some(first_msg) = msg else {
            break;
        };

        // Drain any immediately queued messages before next redraw (keeps streaming buttery-smooth)
        let mut batch = vec![first_msg];
        while let Ok(more) = ui_rx.try_recv() {
            batch.push(more);
            if batch.len() >= 64 {
                break;
            }
        }

        for m in batch {
            match m {
                UiMsg::Term(Event::Key(key)) => {
                    if app.handle_key(key).await {
                        should_quit = true;
                        break;
                    }
                }
                UiMsg::Term(Event::Paste(text)) => {
                    if let Some(ref mut pq) = app.active_question {
                        if pq.typing_custom {
                            pq.custom_input.push_str(&text.replace(['\r', '\n'], " "));
                        }
                    } else if let Some(ref mut mp) = app.model_picker {
                        mp.filter.push_str(text.trim());
                        mp.selected = 0;
                    } else if app.active_perm.is_none() {
                        app.editor.insert_str(&text);
                    }
                }
                UiMsg::Term(Event::Mouse(mouse)) => {
                    app.handle_mouse(mouse);
                }
                UiMsg::Term(Event::Resize(_, _)) => {
                    app.cache_dirty = true;
                }
                UiMsg::Term(_) => {}
                UiMsg::Agent(ev) => {
                    app.handle_agent_event(ev);
                }
                UiMsg::TurnFinished(res) => {
                    app.running = false;
                    app.turn_started_at = None;
                    app.finalize_open_thinking();
                    app.refresh_agent_snapshot().await;
                    match res {
                        Ok(outcome) => {
                            if let Some(reason) = outcome.halted_reason {
                                app.push_notice(
                                    NoticeKind::Warning,
                                    format!("Halted: {reason}"),
                                    None,
                                );
                            }
                        }
                        Err(carnot::CarnotError::Cancelled) => {
                            app.push_notice(
                                NoticeKind::Warning,
                                "Turn aborted (session preserved)",
                                None,
                            );
                        }
                        Err(e) => {
                            app.push_notice(NoticeKind::Error, format!("{e}"), None);
                        }
                    }
                }
                UiMsg::CompactionFinished(res) => {
                    app.compacting = false;
                    app.turn_started_at = None;
                    app.refresh_agent_snapshot().await;
                    match res {
                        Ok((before, after, semantic)) => {
                            let mode = if semantic {
                                "semantic-llm"
                            } else {
                                "deterministic"
                            };
                            app.push_notice(
                                NoticeKind::Success,
                                format!(
                                    "Compacted ({mode}): ~{} → ~{} est. tokens",
                                    fmt_tokens(before),
                                    fmt_tokens(after)
                                ),
                                None,
                            );
                        }
                        Err(e) => {
                            app.push_notice(
                                NoticeKind::Error,
                                format!("Compaction failed: {e}"),
                                None,
                            );
                        }
                    }
                }
                UiMsg::PermissionReq {
                    effective_tool,
                    summary,
                    reply,
                    ..
                } => {
                    let req = PendingPermission {
                        effective_tool,
                        summary,
                        reply,
                    };
                    if app.active_perm.is_none() {
                        app.active_perm = Some(req);
                    } else {
                        app.perm_queue.push_back(req);
                    }
                }
                UiMsg::QuestionReq { questions, reply } => {
                    let mut pq = PendingQuestion {
                        questions,
                        current_idx: 0,
                        selected_row: 0,
                        checked: HashSet::new(),
                        custom_input: String::new(),
                        typing_custom: false,
                        answers: Vec::new(),
                        reply,
                    };
                    pq.reset_step();
                    if app.active_question.is_none() {
                        app.active_question = Some(pq);
                    } else {
                        app.question_queue.push_back(pq);
                    }
                }
            }
        }
    }

    stop_flag.store(true, Ordering::Relaxed);
    ev_task.abort();
    if app.running {
        agent.abort().await;
    }
    drop(guard);
    let _ = reader_handle.join();
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use carnot::{CarnotBuilder, UserQuestionOption};
    use ratatui::backend::TestBackend;

    #[tokio::test]
    async fn tui_renders_all_views_and_modals_without_panic() {
        let dir = std::env::temp_dir().join(format!("carnot-tui-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let agent = CarnotBuilder::new(dir.clone())
            .provider_name("faux")
            .persist_session(false)
            .sandbox(false)
            .build_async()
            .await
            .expect("build faux agent")
            .0;

        let (tx, _rx) = mpsc::unbounded_channel();
        let mut app = TuiApp::new(
            agent,
            TuiLaunchInfo {
                default_provider: "faux".into(),
                provider_configs: Vec::new(),
                active_extensions: vec!["web".into(), "memory".into()],
            },
            tx,
        )
        .await;

        let mut terminal = Terminal::new(TestBackend::new(100, 30)).expect("test backend");

        // 1. Welcome state
        terminal.draw(|f| app.draw(f)).expect("draw welcome");

        // 2. Rich transcript (user + thinking + tool with Edit diff + markdown assistant)
        app.items.push(TranscriptItem::User {
            text: "Refactor `main.rs` and add tests".into(),
        });
        app.items.push(TranscriptItem::Thinking {
            text: "Checking the diff first.".into(),
            started_at: Instant::now(),
            duration_ms: Some(420),
        });
        app.items.push(TranscriptItem::Tool {
            call_id: "c1".into(),
            name: "Edit".into(),
            summary: "Edit src/main.rs".into(),
            status: ToolStatus::Success,
            started_at: Instant::now(),
            duration_ms: Some(14),
            progress: None,
            output_preview: "ok".into(),
            edit_diff: Some(("let a = 1;\n".into(), "let a = 2;\nlet b = 3;\n".into())),
        });
        app.items.push(TranscriptItem::Assistant {
            text: "## Done\n\nUpdated `src/main.rs`:\n\n```rust\nfn main() { println!(\"ok\"); }\n```\n\n| File | Status |\n|---|---|\n| main.rs | OK |".into(),
        });
        app.cache_dirty = true;
        terminal.draw(|f| app.draw(f)).expect("draw transcript");

        // 3. Slash autocomplete popup
        app.editor.set_text("/mo");
        terminal.draw(|f| app.draw(f)).expect("draw slash popup");

        // 4. AskUser modal
        let (qtx, _qrx) = oneshot::channel();
        let mut pq = PendingQuestion {
            questions: vec![UserQuestion {
                id: "q1".into(),
                header: Some("Mode".into()),
                question: "Pick a mode?".into(),
                options: vec![
                    UserQuestionOption {
                        label: "Fast".into(),
                        description: Some("Minimal overhead".into()),
                    },
                    UserQuestionOption {
                        label: "Strict".into(),
                        description: None,
                    },
                ],
                multi_select: false,
            }],
            current_idx: 0,
            selected_row: 0,
            checked: HashSet::new(),
            custom_input: String::new(),
            typing_custom: false,
            answers: Vec::new(),
            reply: qtx,
        };
        pq.reset_step();
        app.active_question = Some(pq);
        terminal.draw(|f| app.draw(f)).expect("draw question modal");
        app.active_question = None;

        // 5. Permission approval modal
        let (ptx, _prx) = oneshot::channel();
        app.active_perm = Some(PendingPermission {
            effective_tool: "Bash".into(),
            summary: "Bash cargo test".into(),
            reply: ptx,
        });
        terminal.draw(|f| app.draw(f)).expect("draw approval modal");
        app.active_perm = None;

        // 6. Tiny viewport safety
        let mut tiny = Terminal::new(TestBackend::new(24, 10)).expect("tiny backend");
        tiny.draw(|f| app.draw(f)).expect("draw tiny");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
