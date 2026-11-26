//! Optional interactive full-screen terminal UI (TUI) crate for Carnot.
//!
//! Keeps `ratatui`, `crossterm`, `unicode-width`, and `similar` completely isolated from the
//! headless Carnot core (`carnot`, `carnot-agent`, `carnot-protocol`) so headless CLI, ACP, RPC,
//! HTTP/SSE, and embedded Rust hosts pay zero compile-time or binary-size cost for the TUI.

mod commands;
mod editor;
mod format;
mod input;
mod markdown;
mod render;
mod state;
mod theme;

use crate::format::{fmt_tokens, summarize_tool_call};
use crate::state::{
    NoticeKind, PendingPermission, PendingQuestion, PermissionReply, TuiApp, UiMsg,
};
use carnot::{
    Agent, PermissionBroker, ToolGateDecision, UserQuestion, UserQuestionAnswer, UserQuestionBroker,
};
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use serde_json::Value;
use std::collections::HashSet;
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, mpsc, oneshot};

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
        if let Err(e) = execute!(
            stdout,
            EnterAlternateScreen,
            EnableMouseCapture,
            EnableBracketedPaste
        ) {
            let _ = disable_raw_mode();
            return Err(e);
        }
        let backend = CrosstermBackend::new(stdout);
        let terminal = match Terminal::new(backend) {
            Ok(t) => t,
            Err(e) => {
                let _ = disable_raw_mode();
                let mut out = io::stdout();
                let _ = execute!(
                    out,
                    DisableBracketedPaste,
                    DisableMouseCapture,
                    LeaveAlternateScreen
                );
                return Err(e);
            }
        };

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
pub(crate) struct TuiPermissionBroker {
    pub(crate) tx: mpsc::UnboundedSender<UiMsg>,
    pub(crate) allowed_always: Mutex<HashSet<String>>,
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
    use crate::state::{ToolStatus, TranscriptItem};
    use carnot::{CarnotBuilder, UserQuestionOption};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use ratatui::backend::TestBackend;
    use std::time::Instant;

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

        // 7. `/context` command (before and after a turn) and live steer / follow-up while running
        assert!(!app.handle_slash_command("context").await);
        assert!(app.items.iter().any(|it| matches!(
            it,
            TranscriptItem::Notice { title, body: Some(d), .. }
                if title == "Context Breakdown" && d.contains("Total:")
        )));
        let _ = app.agent.prompt("hello").await.expect("faux turn");
        assert!(!app.handle_slash_command("context").await);
        assert!(app.items.iter().any(|it| matches!(
            it,
            TranscriptItem::Notice { title, body: Some(d), .. }
                if title == "Context Breakdown" && d.contains("system") && d.contains("tools")
        )));

        app.running = true;
        app.editor.set_text("change direction");
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .await;
        assert!(app.items.iter().any(|it| matches!(
            it,
            TranscriptItem::Notice { title, .. } if title.contains("Steer queued")
        )));

        app.editor.set_text("run tests next");
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT))
            .await;
        assert!(app.items.iter().any(|it| matches!(
            it,
            TranscriptItem::Notice { title, .. } if title.contains("Follow-up queued")
        )));
        app.running = false;

        let _ = std::fs::remove_dir_all(&dir);
    }
}
