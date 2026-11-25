//! Core TUI application state, slash-command catalog, and agent event reducer.

use crate::TuiLaunchInfo;
use crate::editor::PromptEditor;
use crate::format::{
    clip_one_line, detect_git_branch, extract_edit_diff, fmt_tokens, shorten_path,
    summarize_tool_call,
};
use crate::theme::Palette;
use carnot::{Agent, AgentEvent, ReasoningEffort, RunOutcome, UserQuestion, UserQuestionAnswer};
use crossterm::event::Event;
use ratatui::text::Line;
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};

pub(crate) const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[derive(Debug, Clone, Copy)]
pub(crate) struct SlashSpec {
    pub(crate) name: &'static str,
    pub(crate) usage: &'static str,
    pub(crate) desc: &'static str,
}

pub(crate) const SLASH_COMMANDS: &[SlashSpec] = &[
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
        name: "context",
        usage: "/context",
        desc: "Show context window breakdown by category",
    },
    SlashSpec {
        name: "cache",
        usage: "/cache",
        desc: "Show prompt-cache audit, prefix hash & TTL tiers",
    },
    SlashSpec {
        name: "goal",
        usage: "/goal [criterion | --cmd <sh> | clear]",
        desc: "Configure, inspect, or clear autonomous goal verification",
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

pub(crate) enum UiMsg {
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
pub(crate) enum PermissionReply {
    AllowOnce,
    AllowAlways,
    Deny,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolStatus {
    Running,
    Success,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NoticeKind {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub(crate) enum TranscriptItem {
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

pub(crate) struct PendingPermission {
    pub(crate) effective_tool: String,
    pub(crate) summary: String,
    pub(crate) reply: oneshot::Sender<PermissionReply>,
}

pub(crate) struct PendingQuestion {
    pub(crate) questions: Vec<UserQuestion>,
    pub(crate) current_idx: usize,
    pub(crate) selected_row: usize,
    /// For multi-select questions: toggled option indices.
    pub(crate) checked: HashSet<usize>,
    /// Free-text input buffer when typing custom answer (or open-ended question).
    pub(crate) custom_input: String,
    pub(crate) typing_custom: bool,
    pub(crate) answers: Vec<UserQuestionAnswer>,
    pub(crate) reply: oneshot::Sender<Vec<UserQuestionAnswer>>,
}

impl PendingQuestion {
    pub(crate) fn current(&self) -> Option<&UserQuestion> {
        self.questions.get(self.current_idx)
    }

    pub(crate) fn reset_step(&mut self) {
        self.selected_row = 0;
        self.checked.clear();
        self.custom_input.clear();
        self.typing_custom = self.current().is_some_and(|q| q.options.is_empty());
    }

    pub(crate) fn total_rows(&self) -> usize {
        self.current().map_or(1, |q| {
            if q.options.is_empty() {
                1
            } else {
                q.options.len() + 1
            }
        })
    }

    pub(crate) fn is_custom_row(&self) -> bool {
        self.current()
            .is_some_and(|q| q.options.is_empty() || self.selected_row >= q.options.len())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ModelChoice {
    pub(crate) id: String,
    pub(crate) label: String,
    pub(crate) provider: String,
}

pub(crate) struct ModelPicker {
    pub(crate) all: Vec<ModelChoice>,
    pub(crate) filter: String,
    pub(crate) selected: usize,
}

impl ModelPicker {
    pub(crate) fn filtered(&self) -> Vec<&ModelChoice> {
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
pub(crate) struct WorkerSummary {
    pub(crate) state: String,
    pub(crate) model: String,
    pub(crate) turns: u64,
    pub(crate) tools_called: u64,
    pub(crate) cost_usd: f64,
    pub(crate) activity: String,
}

pub(crate) struct TuiApp {
    pub(crate) agent: Arc<Agent>,
    pub(crate) palette: Palette,
    pub(crate) launch: TuiLaunchInfo,
    pub(crate) ui_tx: mpsc::UnboundedSender<UiMsg>,

    // Header / status metadata
    pub(crate) session_id: String,
    pub(crate) model: String,
    pub(crate) effort: ReasoningEffort,
    pub(crate) sandbox_badge: &'static str,
    pub(crate) workspace_short: String,
    pub(crate) git_branch: Option<String>,

    // Live metrics
    pub(crate) ctx_tokens: u64,
    pub(crate) ctx_window: u64,
    pub(crate) cost_usd: f64,
    pub(crate) cache_hit_pct: f64,
    pub(crate) cache_breaks: u64,
    pub(crate) total_turns: u64,
    pub(crate) workers: std::collections::BTreeMap<String, WorkerSummary>,

    // Transcript + cached lines
    pub(crate) items: Vec<TranscriptItem>,
    pub(crate) cached_lines: Vec<Line<'static>>,
    pub(crate) cache_width: usize,
    pub(crate) cache_dirty: bool,
    pub(crate) scroll_offset: usize,
    pub(crate) auto_follow: bool,
    pub(crate) last_viewport_height: usize,

    // Interactive toggles & state
    pub(crate) editor: PromptEditor,
    pub(crate) show_thinking: bool,
    pub(crate) expand_tools: bool,
    pub(crate) approve_enabled: bool,
    pub(crate) running: bool,
    pub(crate) compacting: bool,
    pub(crate) turn_started_at: Option<Instant>,
    pub(crate) spinner_frame: usize,
    pub(crate) last_ctrl_c: Option<Instant>,
    pub(crate) slash_selected: usize,

    // Modals / overlays
    pub(crate) perm_queue: VecDeque<PendingPermission>,
    pub(crate) active_perm: Option<PendingPermission>,
    pub(crate) question_queue: VecDeque<PendingQuestion>,
    pub(crate) active_question: Option<PendingQuestion>,
    pub(crate) model_picker: Option<ModelPicker>,
}

impl TuiApp {
    pub(crate) async fn new(
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

    pub(crate) fn is_animating(&self) -> bool {
        self.running || self.compacting || self.workers.values().any(|w| w.state == "running")
    }

    pub(crate) fn push_notice(
        &mut self,
        kind: NoticeKind,
        title: impl Into<String>,
        body: Option<String>,
    ) {
        self.items.push(TranscriptItem::Notice {
            kind,
            title: title.into(),
            body,
        });
        self.cache_dirty = true;
    }

    pub(crate) fn finalize_open_thinking(&mut self) {
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

    pub(crate) fn handle_agent_event(&mut self, ev: AgentEvent) {
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

    pub(crate) async fn refresh_agent_snapshot(&mut self) {
        self.model = self.agent.current_model().await;
        self.effort = self.agent.current_effort().await;
        self.ctx_tokens = self.agent.current_context_tokens().await;
        self.ctx_window = self.agent.current_context_window().await.max(1);
        let m = self.agent.metrics().await;
        self.cost_usd = m.total_usage.cost_usd;
        self.cache_hit_pct = m.cache_hit_rate() * 100.0;
        self.total_turns = m.turns;
    }
}
