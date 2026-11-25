//! Minimalist, zero-dependency interactive terminal REPL for daily-driver Carnot sessions.
//!
//! Features:
//!   - Single dedicated stdin reader channel (zero thread leaks or races between prompts, `AskUser`,
//!     and interactive `/approve` confirmations)
//!   - Live streaming output, thinking indicator (toggleable full stream via `/thinking`), and
//!     non-overlapping parallel tool badges with concise input/error summaries
//!   - Multi-line input via trailing `\` or fenced `"""` blocks
//!   - Clean Ctrl+C turn cancellation (never kills the session or corrupts tool pairing)
//!   - Live slash commands: `/help`, `/model`, `/effort`, `/thinking`, `/approve`, `/tools`,
//!     `/compact`, `/undo`, `/cost`, `/session`, `/exit`

use carnot::{
    Agent, AgentEvent, PermissionBroker, ProviderRegistry, ReasoningEffort, ToolGateDecision,
    UserQuestion, UserQuestionAnswer, UserQuestionBroker,
};
use serde_json::Value;
use std::collections::HashSet;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};

type SharedStdinRx = Arc<Mutex<mpsc::Receiver<Option<String>>>>;

#[derive(Debug, Clone, Copy)]
struct Theme {
    color: bool,
}

impl Theme {
    fn detect() -> Self {
        let no_color = std::env::var_os("NO_COLOR").is_some();
        let is_tty = io::stderr().is_terminal();
        Self {
            color: !no_color && is_tty,
        }
    }

    fn paint(self, code: &str, text: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    }
}

fn spawn_stdin_reader() -> SharedStdinRx {
    let (tx, rx) = mpsc::channel::<Option<String>>(16);
    std::thread::spawn(move || {
        let stdin = io::stdin();
        let mut lock = stdin.lock();
        loop {
            let mut buf = String::new();
            match lock.read_line(&mut buf) {
                Ok(0) | Err(_) => {
                    let _ = tx.blocking_send(None);
                    break;
                }
                Ok(_) => {
                    if tx.blocking_send(Some(buf)).is_err() {
                        break;
                    }
                }
            }
        }
    });
    Arc::new(Mutex::new(rx))
}

pub(crate) fn shorten_path(p: &Path) -> String {
    let full = p.display().to_string();
    if let Some(home) = std::env::var_os("HOME") {
        let h = PathBuf::from(home).display().to_string();
        if let Some(rest) = full.strip_prefix(&h) {
            return format!("~{rest}");
        }
    }
    full
}

pub(crate) fn clip_one_line(s: &str, max_chars: usize) -> String {
    let first = s
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    let count = first.chars().count();
    if count <= max_chars {
        first.to_string()
    } else {
        let head: String = first.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{head}…")
    }
}

pub(crate) fn summarize_tool_call(tool: &str, input: &Value) -> String {
    let (effective_tool, effective_input) = if tool == "CallDeferredTool" {
        let inner_name = input
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("CallDeferredTool");
        let inner_args = input.get("arguments").unwrap_or(input);
        (inner_name, inner_args)
    } else {
        (tool, input)
    };

    let detail = match effective_tool {
        "Read" | "Edit" | "Write" => effective_input
            .get("file_path")
            .or_else(|| effective_input.get("path"))
            .and_then(Value::as_str)
            .map(|s| clip_one_line(s, 72)),
        "Bash" => effective_input
            .get("command")
            .and_then(Value::as_str)
            .map(|s| clip_one_line(s, 72)),
        "CodeMode" => effective_input
            .get("code")
            .and_then(Value::as_str)
            .map(|s| {
                let first = s
                    .lines()
                    .map(str::trim)
                    .find(|l| !l.is_empty() && !l.starts_with("```"))
                    .unwrap_or("js");
                clip_one_line(first, 64)
            }),
        "Grep" => effective_input
            .get("pattern")
            .and_then(Value::as_str)
            .map(|s| format!("/{}/", clip_one_line(s, 56))),
        "WebFetch" => effective_input
            .get("url")
            .and_then(Value::as_str)
            .map(|s| clip_one_line(s, 72)),
        "WebSearch" | "ToolSearch" | "MemorySearch" => effective_input
            .get("query")
            .and_then(Value::as_str)
            .map(|s| clip_one_line(s, 64)),
        "MemorySave" => effective_input
            .get("title")
            .or_else(|| effective_input.get("id"))
            .and_then(Value::as_str)
            .map(|s| clip_one_line(s, 64)),
        "LspQuery" => {
            let op = effective_input
                .get("action")
                .and_then(Value::as_str)
                .unwrap_or("query");
            let fp = effective_input
                .get("file_path")
                .and_then(Value::as_str)
                .unwrap_or("");
            Some(clip_one_line(&format!("{op} {fp}"), 72))
        }
        "Subagent" => {
            let mut tags = Vec::new();
            if let Some(n) = effective_input
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                tags.push(n.to_string());
            }
            if effective_input
                .get("land")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                tags.push("land".to_string());
            }
            if effective_input
                .get("fork")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                tags.push("fork".to_string());
            }
            if let Some(m) = effective_input
                .get("model")
                .or_else(|| effective_input.get("provider"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                tags.push(m.to_string());
            }
            let body = effective_input
                .get("task")
                .or_else(|| effective_input.get("job"))
                .and_then(Value::as_str)
                .map(|s| clip_one_line(s, 56))
                .unwrap_or_default();
            if tags.is_empty() {
                (!body.is_empty()).then_some(body)
            } else if body.is_empty() {
                Some(format!("[{}]", tags.join(" · ")))
            } else {
                Some(format!("[{}] {body}", tags.join(" · ")))
            }
        }
        "SendMessage" => {
            let to = effective_input
                .get("to")
                .and_then(Value::as_str)
                .unwrap_or("?");
            let msg = effective_input
                .get("message")
                .and_then(Value::as_str)
                .map(|s| clip_one_line(s, 52))
                .unwrap_or_default();
            Some(format!("→ {to}: {msg}"))
        }
        _ => None,
    };

    let worker_prefix = input
        .get("_carnot_worker")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(|w| format!("[worker:{w}] "))
        .unwrap_or_default();

    match detail {
        Some(d) if !d.is_empty() => format!("{worker_prefix}{effective_tool} {d}"),
        _ => format!("{worker_prefix}{effective_tool}"),
    }
}

pub(crate) fn fmt_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

#[derive(Debug)]
struct ReplQuestionBroker {
    stdin_rx: SharedStdinRx,
    theme: Theme,
}

impl UserQuestionBroker for ReplQuestionBroker {
    fn ask<'a>(
        &'a self,
        _request_id: &'a str,
        questions: &'a [UserQuestion],
    ) -> carnot_ext::BoxFuture<'a, carnot::Result<Vec<UserQuestionAnswer>>> {
        let qs = questions.to_vec();
        let theme = self.theme;
        Box::pin(async move {
            let mut rx = self.stdin_rx.lock().await;
            let mut answers = Vec::with_capacity(qs.len());
            for q in &qs {
                eprintln!("\n{} {}", theme.paint("1;33", "[Question]"), q.question);
                for (idx, opt) in q.options.iter().enumerate() {
                    if let Some(ref desc) = opt.description {
                        eprintln!("  {}) {} — {}", idx + 1, opt.label, theme.paint("2", desc));
                    } else {
                        eprintln!("  {}) {}", idx + 1, opt.label);
                    }
                }
                let hint = if q.options.is_empty() {
                    "Answer> "
                } else if q.multi_select {
                    "Answer (numbers comma-separated or text)> "
                } else {
                    "Answer (number or text)> "
                };
                eprint!("{}", theme.paint("1;33", hint));
                let _ = io::stderr().flush();

                let raw_line = rx.recv().await.flatten().unwrap_or_default();
                let raw = raw_line.trim();
                let mut selected = Vec::new();
                let mut free_text = None;
                if !q.options.is_empty() {
                    for tok in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                        if let Ok(n) = tok.parse::<usize>()
                            && n >= 1
                            && n <= q.options.len()
                        {
                            selected.push(q.options[n - 1].label.clone());
                        }
                    }
                }
                if selected.is_empty() && !raw.is_empty() {
                    free_text = Some(raw.to_string());
                }
                answers.push(UserQuestionAnswer {
                    id: q.id.clone(),
                    selected,
                    text: free_text,
                });
            }
            Ok(answers)
        })
    }
}

#[derive(Debug)]
struct ReplPermissionBroker {
    stdin_rx: SharedStdinRx,
    allowed_always: Mutex<HashSet<String>>,
    theme: Theme,
}

impl PermissionBroker for ReplPermissionBroker {
    fn request_permission(
        &self,
        _call_id: &str,
        tool_name: &str,
        input: &Value,
    ) -> carnot_ext::BoxFuture<'_, carnot::Result<ToolGateDecision>> {
        let tool_name = tool_name.to_string();
        let input = input.clone();
        let theme = self.theme;
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
            let mut rx = self.stdin_rx.lock().await;
            eprintln!("\n{} {}", theme.paint("1;35", "[Approve?]"), summary);
            eprint!(
                "{}",
                theme.paint("1;35", "[y] allow once / [a] always / [n] deny> ")
            );
            let _ = io::stderr().flush();

            let raw = rx.recv().await.flatten().unwrap_or_default();
            match raw.trim().to_ascii_lowercase().as_str() {
                "y" | "yes" => Ok(ToolGateDecision::Allow),
                "a" | "always" => {
                    self.allowed_always.lock().await.insert(effective_tool);
                    Ok(ToolGateDecision::Allow)
                }
                _ => Ok(ToolGateDecision::Deny {
                    reason: format!("denied by user in REPL for `{effective_tool}`"),
                }),
            }
        })
    }
}

async fn read_prompt_line(stdin_rx: &SharedStdinRx, theme: Theme) -> Option<String> {
    let mut rx = stdin_rx.lock().await;
    let first = rx.recv().await??;
    let first_trimmed = first.trim_end_matches(['\r', '\n']);

    // Multi-line fenced mode: start with `"""` and end with `"""`
    if first_trimmed.trim() == "\"\"\"" {
        let mut lines = Vec::new();
        loop {
            eprint!("{}", theme.paint("2", "… "));
            let _ = io::stderr().flush();
            let Some(next) = rx.recv().await.flatten() else {
                break;
            };
            let t = next.trim_end_matches(['\r', '\n']);
            if t.trim() == "\"\"\"" {
                break;
            }
            lines.push(t.to_string());
        }
        return Some(lines.join("\n"));
    }

    // Backslash line continuation
    let mut buf = first_trimmed.to_string();
    while buf.ends_with('\\') {
        buf.pop();
        buf.push('\n');
        eprint!("{}", theme.paint("2", "… "));
        let _ = io::stderr().flush();
        let Some(next) = rx.recv().await.flatten() else {
            break;
        };
        buf.push_str(next.trim_end_matches(['\r', '\n']));
    }
    Some(buf)
}

pub(crate) fn parse_effort(s: &str) -> Option<ReasoningEffort> {
    match s.trim().to_ascii_lowercase().as_str() {
        "low" => Some(ReasoningEffort::Low),
        "medium" | "med" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::Xhigh),
        "max" => Some(ReasoningEffort::Max),
        _ => None,
    }
}

pub(crate) async fn run_repl(
    agent: Arc<Agent>,
    default_provider: String,
    provider_configs: Vec<PathBuf>,
) -> i32 {
    let theme = Theme::detect();
    let stdin_rx = spawn_stdin_reader();

    agent
        .set_question_broker(Arc::new(ReplQuestionBroker {
            stdin_rx: Arc::clone(&stdin_rx),
            theme,
        }))
        .await;

    let mut show_thinking = false;
    let mut approve_enabled = false;

    let sid = agent.session_id().await;
    let model = agent.current_model().await;
    let effort = agent.current_effort().await;
    let sandbox_badge = if !agent.config.modules.sandbox {
        "sandbox:off"
    } else if agent.config.modules.sandbox_network {
        "sandbox:net"
    } else {
        "sandbox:offline"
    };

    eprintln!(
        "{} {}",
        theme.paint(
            "1;38;5;208",
            &format!("carnot v{}", env!("CARGO_PKG_VERSION"))
        ),
        theme.paint(
            "2",
            &format!(
                "({model} · effort:{effort:?} · {sandbox_badge} · session:{})",
                sid.chars().take(8).collect::<String>()
            )
        )
    );
    eprintln!(
        "{}",
        theme.paint(
            "2",
            &format!(
                "cwd: {} · /help for commands · \\ or \"\"\" for multiline · Ctrl+C aborts turn",
                shorten_path(&agent.config.workspace_root)
            )
        )
    );

    loop {
        eprint!("\n{} ", theme.paint("1;36", ">"));
        let _ = io::stderr().flush();

        let prompt_opt = tokio::select! {
            line = read_prompt_line(&stdin_rx, theme) => line,
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\n{}", theme.paint("2", "(interrupted — type /exit or press Ctrl+D to quit)"));
                continue;
            }
        };

        let Some(input) = prompt_opt else {
            eprintln!();
            return 0;
        };

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(cmd) = trimmed.strip_prefix('/') {
            let mut parts = cmd.split_whitespace();
            let verb = parts.next().unwrap_or("");
            let rest = cmd.strip_prefix(verb).map_or("", str::trim);
            match verb {
                "exit" | "quit" | "q" => return 0,
                "help" | "h" => {
                    eprintln!(
                        "Commands:\n\
                         \x20 /model [id]         Show or switch model/provider live\n\
                         \x20 /effort [level]     Show or set effort (low|medium|high|xhigh|max)\n\
                         \x20 /thinking           Toggle streaming reasoning text (currently {})\n\
                         \x20 /approve [on|off]   Toggle interactive tool approvals (currently {})\n\
                         \x20 /tools              List active & deferred tools\n\
                         \x20 /compact [focus]    Run cache-warm semantic compaction\n\
                         \x20 /fork               Fork this session into a new branch JSONL\n\
                         \x20 /undo               Atomically revert last Edit/Write via CAS\n\
                         \x20 /cost               Show turns, context %, cache hit rate & USD cost\n\
                         \x20 /session            Show session ID, prefix hash, sandbox & workspace\n\
                         \x20 /exit               Quit interactive session\n\
                         \nTips:\n\
                         \x20 - Pass `--allow-network` at startup to allow outbound network inside sandboxed Bash.\n\
                         \x20 - End a line with `\\` or wrap with `\"\"\"` for multi-line prompts.",
                        if show_thinking { "on" } else { "off" },
                        if approve_enabled { "on" } else { "off" },
                    );
                }
                "model" => {
                    if rest.is_empty() {
                        let cur = agent.current_model().await;
                        let cw = agent.current_context_window().await;
                        eprintln!("model={cur} context_window={}", fmt_tokens(cw));
                    } else {
                        match ProviderRegistry::load_default(&provider_configs) {
                            Ok(reg) => {
                                let (prov_id, model_id) =
                                    reg.resolve_model_spec(rest, None).unwrap_or_else(|| {
                                        (default_provider.clone(), rest.trim().to_string())
                                    });
                                match reg.build_provider(&prov_id, Some(&model_id)) {
                                    Ok(provider) => {
                                        let cw = reg.context_window_for_model(&model_id);
                                        let max_out = reg.max_output_tokens_for_model(&model_id);
                                        match agent
                                            .set_model(model_id.clone(), provider, cw, max_out)
                                            .await
                                        {
                                            Ok(()) => eprintln!(
                                                "{} {model_id} (provider={prov_id})",
                                                theme.paint("32", "switched model ->")
                                            ),
                                            Err(e) => eprintln!(
                                                "{} {e}",
                                                theme.paint("31", "model switch failed:")
                                            ),
                                        }
                                    }
                                    Err(e) => eprintln!(
                                        "{} {e}",
                                        theme.paint("31", "provider build failed:")
                                    ),
                                }
                            }
                            Err(e) => eprintln!(
                                "{} {e}",
                                theme.paint("31", "provider catalog load failed:")
                            ),
                        }
                    }
                }
                "effort" => {
                    if rest.is_empty() {
                        eprintln!("effort={:?}", agent.current_effort().await);
                    } else if let Some(eff) = parse_effort(rest) {
                        agent.set_effort(eff).await;
                        eprintln!("{} {eff:?}", theme.paint("32", "effort ->"));
                    } else {
                        eprintln!("usage: /effort low|medium|high|xhigh|max");
                    }
                }
                "thinking" => {
                    show_thinking = !show_thinking;
                    eprintln!(
                        "thinking display: {}",
                        if show_thinking { "on" } else { "off" }
                    );
                }
                "approve" => {
                    let target = match rest.to_ascii_lowercase().as_str() {
                        "on" | "true" | "1" => true,
                        "off" | "false" | "0" => false,
                        "" => !approve_enabled,
                        _ => {
                            eprintln!("usage: /approve [on|off]");
                            continue;
                        }
                    };
                    approve_enabled = target;
                    if approve_enabled {
                        agent
                            .set_permission_broker(Arc::new(ReplPermissionBroker {
                                stdin_rx: Arc::clone(&stdin_rx),
                                allowed_always: Mutex::new(HashSet::new()),
                                theme,
                            }))
                            .await;
                        eprintln!(
                            "{}",
                            theme.paint("32", "interactive tool approvals enabled")
                        );
                    } else {
                        agent.clear_permission_broker().await;
                        eprintln!(
                            "{}",
                            theme.paint("2", "interactive tool approvals disabled")
                        );
                    }
                }
                "tools" => {
                    let (eager, deferred) = agent.tool_names();
                    eprintln!("active ({}):   {}", eager.len(), eager.join(", "));
                    if deferred.is_empty() {
                        eprintln!("deferred (0): none");
                    } else {
                        eprintln!("deferred ({}): {}", deferred.len(), deferred.join(", "));
                    }
                }
                "cost" | "stats" => {
                    let m = agent.metrics().await;
                    let u = &m.total_usage;
                    let ctx_now = agent.current_context_tokens().await;
                    let ctx_win = agent.current_context_window().await.max(1);
                    let pct = (ctx_now as f64 / ctx_win as f64) * 100.0;
                    eprintln!(
                        "turns={} requests={} ctx={}/{} ({pct:.1}%) cost=${:.4} cache_hit={:.1}% (in={} cr={} cw5={} cw1={} out={}) edits={}/{} compactions={}",
                        m.turns,
                        m.requests,
                        fmt_tokens(ctx_now),
                        fmt_tokens(ctx_win),
                        u.cost_usd,
                        m.cache_hit_rate() * 100.0,
                        u.input_tokens,
                        u.cache_read_tokens,
                        u.cache_write_5m_tokens,
                        u.cache_write_1h_tokens,
                        u.output_tokens,
                        m.edit_attempts.saturating_sub(m.edit_failures),
                        m.edit_attempts,
                        m.compactions_applied
                    );
                }
                "session" => {
                    let id = agent.session_id().await;
                    let ph = agent.prefix_hash().await;
                    let cur_model = agent.current_model().await;
                    let cur_effort = agent.current_effort().await;
                    eprintln!(
                        "session_id={id}\nmodel={cur_model}\neffort={cur_effort:?}\nsandbox={sandbox_badge}\nworkspace={}\nprefix_blake3={ph}",
                        agent.config.workspace_root.display()
                    );
                }
                "fork" => match agent.fork_session().await {
                    Ok((new_id, path)) => {
                        let p_note = path
                            .as_deref()
                            .map(|p| format!(" ({})", shorten_path(p)))
                            .unwrap_or_default();
                        eprintln!(
                            "{} {new_id}{p_note}",
                            theme.paint("32", "forked session ->")
                        );
                    }
                    Err(e) => eprintln!("{} {e}", theme.paint("31", "fork failed:")),
                },
                "undo" => match agent.undo_last_edit() {
                    Ok(Some(p)) => eprintln!("{} {}", theme.paint("32", "reverted"), p.display()),
                    Ok(None) => eprintln!("nothing to undo"),
                    Err(e) => eprintln!("{} {e}", theme.paint("31", "undo failed:")),
                },
                "compact" => {
                    let focus = (!rest.is_empty()).then_some(rest);
                    eprintln!(
                        "{}",
                        theme.paint("2", "compacting context (cache-warm semantic handoff)...")
                    );
                    match agent.compact_now_with_focus(focus).await {
                        Ok((before, after, semantic)) => {
                            let mode = if semantic {
                                "semantic-llm"
                            } else {
                                "deterministic"
                            };
                            eprintln!(
                                "{} ~{before} -> ~{after} est. tokens",
                                theme.paint("32", &format!("compacted ({mode}):"))
                            );
                        }
                        Err(e) => eprintln!("{} {e}", theme.paint("31", "compaction failed:")),
                    }
                }
                other => eprintln!("unknown command `/{other}` (try /help)"),
            }
            continue;
        }

        // Stream live output, thinking indicator, and parallel-safe tool lines; Ctrl+C aborts ONLY this turn!
        let mut ev_stream = agent.subscribe_from(agent.latest_event_seq());
        let ev_agent = Arc::clone(&agent);
        let ev_task = tokio::spawn(async move {
            let mut printed_text = false;
            let mut in_thinking = false;
            let mut showed_thinking_hint = false;

            loop {
                let env = ev_stream.next().await;
                match &env.event {
                    AgentEvent::ThinkingDelta { delta } => {
                        if show_thinking {
                            if !in_thinking {
                                if printed_text {
                                    println!();
                                    printed_text = false;
                                }
                                eprint!("{}", theme.paint("2;3", "  [thinking] "));
                                in_thinking = true;
                            }
                            eprint!("{}", theme.paint("2", delta));
                            let _ = io::stderr().flush();
                        } else if !showed_thinking_hint {
                            if printed_text {
                                println!();
                                printed_text = false;
                            }
                            eprintln!("{}", theme.paint("2", "  … thinking"));
                            showed_thinking_hint = true;
                        }
                    }
                    AgentEvent::TextDelta { delta } => {
                        if in_thinking {
                            eprintln!();
                            in_thinking = false;
                        }
                        showed_thinking_hint = false;
                        print!("{delta}");
                        let _ = io::stdout().flush();
                        printed_text = true;
                    }
                    AgentEvent::ToolStart { tool, input, .. } => {
                        if in_thinking {
                            eprintln!();
                            in_thinking = false;
                        }
                        showed_thinking_hint = false;
                        if printed_text {
                            println!();
                            printed_text = false;
                        }
                        let summary = summarize_tool_call(tool, input);
                        eprintln!("{}", theme.paint("2", &format!("  ▸ {summary}")));
                    }
                    AgentEvent::ToolUpdate { message, .. } => {
                        if let Some(last) = message.lines().rev().find(|l| !l.trim().is_empty()) {
                            let preview = clip_one_line(last, 96);
                            eprintln!("{}", theme.paint("2", &format!("    ↳ {preview}")));
                        }
                    }
                    AgentEvent::ToolEnd {
                        tool,
                        is_error,
                        duration_ms,
                        output_preview,
                        ..
                    } => {
                        if *is_error {
                            let err_hint = clip_one_line(output_preview, 100);
                            if err_hint.is_empty() {
                                eprintln!(
                                    "  {} {} {}",
                                    theme.paint("31", "✗"),
                                    theme.paint("2", tool),
                                    theme.paint("2", &format!("({duration_ms}ms)"))
                                );
                            } else {
                                eprintln!(
                                    "  {} {} {} — {}",
                                    theme.paint("31", "✗"),
                                    theme.paint("2", tool),
                                    theme.paint("2", &format!("({duration_ms}ms)")),
                                    theme.paint("31", &err_hint)
                                );
                            }
                        } else {
                            eprintln!(
                                "  {} {} {}",
                                theme.paint("32", "✓"),
                                theme.paint("2", tool),
                                theme.paint("2", &format!("({duration_ms}ms)"))
                            );
                        }
                    }
                    AgentEvent::CompactionApplied {
                        mode,
                        tokens_before,
                        tokens_after,
                        semantic_llm,
                    } => {
                        let kind = if *semantic_llm { "semantic" } else { "det" };
                        eprintln!(
                            "  {} ~{} -> ~{} tok",
                            theme.paint("33", &format!("[compact:{mode}/{kind}]")),
                            fmt_tokens(*tokens_before),
                            fmt_tokens(*tokens_after)
                        );
                    }
                    AgentEvent::ModuleAdvisory { signal, message } => {
                        eprintln!(
                            "  {} {}",
                            theme.paint("33", &format!("[{signal}]")),
                            clip_one_line(message, 120)
                        );
                    }
                    AgentEvent::RunSettled { metrics, .. } => {
                        if in_thinking {
                            eprintln!();
                        }
                        if printed_text {
                            println!();
                        }
                        let ctx_now = ev_agent.current_context_tokens().await;
                        let ctx_win = ev_agent.current_context_window().await.max(1);
                        let pct = (ctx_now as f64 / ctx_win as f64) * 100.0;
                        eprintln!(
                            "{}",
                            theme.paint(
                                "2",
                                &format!(
                                    "  [done · {} turns · ctx {}/{} ({pct:.0}%) · ${:.4} · {:.1}% cache]",
                                    metrics.turns,
                                    fmt_tokens(ctx_now),
                                    fmt_tokens(ctx_win),
                                    metrics.total_usage.cost_usd,
                                    metrics.cache_hit_rate() * 100.0
                                )
                            )
                        );
                        break;
                    }
                    AgentEvent::Error { .. } => {
                        if in_thinking {
                            eprintln!();
                        }
                        if printed_text {
                            println!();
                        }
                        break;
                    }
                    _ => {}
                }
            }
        });

        let agent_ctrlc = Arc::clone(&agent);
        let ctrlc_task = tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                eprintln!("\n{}", theme.paint("33", "^C aborting turn..."));
                agent_ctrlc.abort().await;
            }
        });

        let res = agent.prompt(trimmed).await;
        ctrlc_task.abort();
        let _ = tokio::time::timeout(std::time::Duration::from_millis(250), ev_task).await;

        match res {
            Ok(outcome) => {
                if let Some(ref reason) = outcome.halted_reason {
                    eprintln!("{} {reason}", theme.paint("33", "halted:"));
                }
            }
            Err(carnot::CarnotError::Cancelled) => {
                eprintln!("{}", theme.paint("33", "turn aborted (session preserved)."));
            }
            Err(e) => {
                eprintln!("{} {e}", theme.paint("31", "error:"));
            }
        }
    }
}
