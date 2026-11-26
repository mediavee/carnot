//! Slash command dispatch, autocomplete matching, model picker population, and prompt launch.

use crate::TuiPermissionBroker;
use crate::format::{fmt_tokens, parse_effort, shorten_path};
use crate::state::{
    ModelChoice, ModelPicker, NoticeKind, SLASH_COMMANDS, SlashSpec, TranscriptItem, TuiApp, UiMsg,
};
use carnot::ProviderRegistry;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

impl TuiApp {
    pub(crate) fn slash_matches(&self) -> Vec<&'static SlashSpec> {
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

    pub(crate) fn open_model_picker(&mut self) {
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

    pub(crate) async fn switch_model(&mut self, model_spec: &str, provider_override: Option<&str>) {
        match ProviderRegistry::load_default(&self.launch.provider_configs) {
            Ok(reg) => {
                let (prov_id, model_id) = reg
                    .resolve_model_spec(model_spec, provider_override)
                    .unwrap_or_else(|| {
                        (
                            provider_override
                                .unwrap_or(&self.launch.default_provider)
                                .to_string(),
                            model_spec.trim().to_string(),
                        )
                    });
                match reg.build_provider(&prov_id, Some(&model_id)) {
                    Ok(provider) => {
                        let cw = reg.context_window_for_model(&model_id);
                        let max_out = reg.max_output_tokens_for_model(&model_id);
                        match self
                            .agent
                            .set_model(model_id.clone(), provider, cw, max_out)
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
    pub(crate) async fn handle_slash_command(&mut self, cmd: &str) -> bool {
        let mut parts = cmd.split_whitespace();
        let verb = parts.next().unwrap_or("");
        let rest = cmd.strip_prefix(verb).map_or("", str::trim);

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
                    "  /model [id]         Open model picker (^P) or switch live (e.g. opencode/gpt-5.4)",
                    "  /effort [level]     low | medium | high | xhigh | max",
                    "  /cache [tier]       Inspect prefix retention or set short | hybrid | long",
                    "  /fleet              Show active & settled background subagent workers",
                    "  /thinking           Toggle reasoning stream (^T)",
                    "  /approve [on|off]   Toggle interactive tool approvals",
                    "  /tools              List active & deferred tools",
                    "  /compact [focus]    Run cache-warm semantic compaction",
                    "  /fork               Fork session into a new branch JSONL",
                    "  /undo               Revert last Edit/Write via CAS",
                    "  /cost               Show turns, context %, cache & USD",
                    "  /context            Show context breakdown by category",
                    "  /session            Show session ID, prefix hash & workspace",
                    "  /clear              Clear visual transcript (^L)",
                    "  /exit               Quit Carnot",
                    "",
                    "Shortcuts: Enter send/steer · Alt+Enter follow-up (while running)",
                    "           Shift+Enter / ^J newline · Esc abort turn · ^P models",
                    "           ^T thinking · ^O expand tools · PgUp/PgDn scroll",
                ]
                .join("\n");
                self.push_notice(NoticeKind::Info, "Carnot Commands & Shortcuts", Some(body));
            }
            "model" => {
                if rest.is_empty() {
                    self.open_model_picker();
                } else {
                    self.switch_model(rest, None).await;
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
            "context" | "ctx" => {
                self.refresh_agent_snapshot().await;
                let (tokens, breakdown) = self.agent.current_context_with_breakdown().await;
                let window = self.agent.current_context_window().await.max(1);
                let pct = (tokens as f64 / window as f64) * 100.0;
                let body = match breakdown {
                    Some(b) => {
                        let row = |label: &str, val: u64| {
                            let share = if tokens > 0 {
                                (val as f64 / tokens as f64) * 100.0
                            } else {
                                0.0
                            };
                            let filled = ((val as f64 / window as f64) * 20.0).round() as usize;
                            let bar = format!(
                                "{}{}",
                                "█".repeat(filled.min(20)),
                                "░".repeat(20usize.saturating_sub(filled.min(20)))
                            );
                            format!(
                                "  {label:<13} [{bar}] {:>7} ({share:>4.1}%)",
                                fmt_tokens(val)
                            )
                        };
                        let est = if b.estimated { " (estimated)" } else { "" };
                        format!(
                            "  Total: {} / {} ({pct:.1}%){est}\n{}\n{}\n{}\n{}\n{}",
                            fmt_tokens(tokens),
                            fmt_tokens(window),
                            row("system", b.system),
                            row("tools", b.tools),
                            row("memory", b.memory),
                            row("messages", b.messages),
                            row("tool_results", b.tool_results),
                        )
                    }
                    None => format!(
                        "  Total: {} / {} ({pct:.1}%)\n  Breakdown pending next provider turn (reset after compaction).",
                        fmt_tokens(tokens),
                        fmt_tokens(window),
                    ),
                };
                self.push_notice(NoticeKind::Info, "Context Breakdown", Some(body));
            }
            "cache" => {
                if !rest.is_empty() {
                    let arg = rest.strip_prefix("ttl").map_or(rest, str::trim);
                    if let Some(ttl) = carnot::CacheTtl::parse(arg) {
                        self.agent.set_cache_ttl(ttl).await;
                        self.push_notice(
                            NoticeKind::Success,
                            format!("Cache retention → {}", ttl.as_str()),
                            None,
                        );
                    } else {
                        self.push_notice(
                            NoticeKind::Warning,
                            "Usage: /cache [short|hybrid|long]",
                            None,
                        );
                    }
                    return false;
                }
                let m = self.agent.metrics().await;
                let u = &m.total_usage;
                let ph = self.agent.prefix_hash().await;
                let ttl = self.agent.current_cache_ttl().await;
                let retention_str = if m.cacheable_prefix_tokens > 0 {
                    format!("{:.1}%", m.prefix_retention_rate() * 100.0)
                } else {
                    "n/a".to_string()
                };
                let body = format!(
                    "  Prefix hash:       {}\n  Retention tier:    {} (/cache short|hybrid|long)\n  Cache hit:         {:.1}% (hits={} misses={} breaks={})\n  Prefix retention:  {} ({}/{} reusable tokens)\n  Tokens:            read={} · write_5m={} · write_1h={} · uncached_in={}",
                    ph,
                    ttl.as_str(),
                    m.cache_hit_rate() * 100.0,
                    m.prefix_cache_hits,
                    m.prefix_cache_misses,
                    m.cache_breaks,
                    retention_str,
                    fmt_tokens(m.retained_prefix_tokens),
                    fmt_tokens(m.cacheable_prefix_tokens),
                    fmt_tokens(u.cache_read_tokens),
                    fmt_tokens(u.cache_write_5m_tokens),
                    fmt_tokens(u.cache_write_1h_tokens),
                    fmt_tokens(u.input_tokens),
                );
                self.push_notice(NoticeKind::Info, "Prompt Cache Audit", Some(body));
            }
            "goal" => {
                let Some(ctrl) = self.agent.goal_controller() else {
                    self.push_notice(
                        NoticeKind::Warning,
                        "Goal verification extension is not enabled in this build.",
                        None,
                    );
                    return false;
                };
                let arg = rest.trim();
                if arg.is_empty() || arg.eq_ignore_ascii_case("status") {
                    let st = ctrl.status_json();
                    let status = st
                        .get("status")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("idle");
                    let iter = st
                        .get("iterations")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0);
                    let max_iter = st
                        .get("max_iterations")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(8);
                    let criterion = st
                        .get("criterion")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("(none)");
                    let cmds: Vec<&str> = st
                        .get("verify_commands")
                        .and_then(serde_json::Value::as_array)
                        .map(|a| a.iter().filter_map(serde_json::Value::as_str).collect())
                        .unwrap_or_default();
                    let fb = st
                        .get("last_feedback")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("-");
                    let cmds_str = if cmds.is_empty() {
                        "(none)".to_string()
                    } else {
                        cmds.join(" && ")
                    };
                    let body = format!(
                        "  Status:     {status} ({iter}/{max_iter} iterations)\n  Criterion:  {criterion}\n  Commands:   {cmds_str}\n  Feedback:   {fb}\n  Usage:      /goal <criterion> | /goal --cmd <sh> | /goal clear"
                    );
                    self.push_notice(NoticeKind::Info, "Goal Verification", Some(body));
                } else if arg.eq_ignore_ascii_case("clear") || arg.eq_ignore_ascii_case("off") {
                    let _ = ctrl.clear_goal();
                    self.push_notice(NoticeKind::Success, "Active goal cleared.", None);
                } else if let Some(cmd_body) = arg
                    .strip_prefix("--cmd ")
                    .or_else(|| arg.strip_prefix("cmd:"))
                {
                    let cmd_clean = cmd_body.trim();
                    if cmd_clean.is_empty() {
                        self.push_notice(
                            NoticeKind::Warning,
                            "Usage: /goal --cmd <shell command>",
                            None,
                        );
                    } else {
                        let _ = ctrl.set_goal(None, vec![cmd_clean.to_string()], None, None);
                        self.push_notice(
                            NoticeKind::Success,
                            format!("Goal check set → `{cmd_clean}`"),
                            None,
                        );
                    }
                } else {
                    let _ = ctrl.set_goal(Some(arg.to_string()), Vec::new(), None, None);
                    self.push_notice(
                        NoticeKind::Success,
                        format!("Goal criterion set → {arg}"),
                        None,
                    );
                }
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
                        self.session_id.clone_from(&new_id);
                        let detail = path
                            .as_deref()
                            .map(|p| format!("  Saved branch: {}", shorten_path(p)));
                        let short_id: String = new_id.chars().take(8).collect();
                        self.push_notice(
                            NoticeKind::Success,
                            format!("Forked session → {short_id}"),
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

    pub(crate) fn start_prompt(&mut self, prompt: String) {
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
}
