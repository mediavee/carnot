//! Clean zero-dependency interactive terminal REPL for daily-driver Carnot sessions.
//!
//! Supports live streaming output, tool status badges, Ctrl+C turn cancellation (without exiting
//! the session), and slash commands: `/help`, `/compact`, `/undo`, `/cost`, `/session`, `/exit`.

use carnot::{Agent, AgentEvent};
use std::io::{self, BufRead, Write};
use std::sync::Arc;

pub async fn run_repl(agent: Arc<Agent>) -> i32 {
    let sid = agent.session_id().await;
    eprintln!(
        "\x1b[1;38;5;208mcarnot\x1b[0m \x1b[2m({} · effort {:?} · session {})\x1b[0m",
        agent.config.model,
        agent.config.effort,
        &sid[..8.min(sid.len())]
    );
    eprintln!("\x1b[2mType a prompt or /help, /compact, /undo, /cost, /exit\x1b[0m");

    loop {
        eprint!("\n\x1b[1;36m>\x1b[0m ");
        let _ = io::stderr().flush();

        let read_res = tokio::task::spawn_blocking(move || {
            let mut buf = String::new();
            let stdin = io::stdin();
            let mut lock = stdin.lock();
            match lock.read_line(&mut buf) {
                Ok(0) => Ok(None),
                Ok(_) => Ok(Some(buf)),
                Err(e) => Err(e),
            }
        })
        .await;

        let input = match read_res {
            Ok(Ok(Some(s))) => s,
            Ok(Ok(None)) => {
                eprintln!();
                return 0;
            }
            Ok(Err(e)) => {
                eprintln!("stdin error: {e}");
                return 1;
            }
            Err(_) => return 1,
        };

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(cmd) = trimmed.strip_prefix('/') {
            match cmd.split_whitespace().next().unwrap_or("") {
                "exit" | "quit" | "q" => return 0,
                "help" | "h" => {
                    eprintln!(
                        "Commands:\n  /compact [focus]  Run cache-warm semantic context compaction (optional focus instructions)\n  /undo             Atomically revert last Edit/Write via CAS snapshot\n  /cost             Show live turns, tokens, cache hit rate, and cost\n  /session          Show current session ID and prefix BLAKE3 hash\n  /exit             Quit interactive session"
                    );
                }
                "cost" | "stats" => {
                    let m = agent.metrics().await;
                    let u = &m.total_usage;
                    eprintln!(
                        "turns={} requests={} cost=${:.4} cache_hit={:.1}% (in={} cr={} cw5={} cw1={} out={}) edits={}/{} rewinds={}",
                        m.turns,
                        m.requests,
                        u.cost_usd,
                        m.cache_hit_rate() * 100.0,
                        u.input_tokens,
                        u.cache_read_tokens,
                        u.cache_write_5m_tokens,
                        u.cache_write_1h_tokens,
                        u.output_tokens,
                        m.edit_attempts.saturating_sub(m.edit_failures),
                        m.edit_attempts,
                        m.rewinds_applied
                    );
                }
                "session" => {
                    let id = agent.session_id().await;
                    let ph = agent.prefix_hash().await;
                    eprintln!("session_id={id}\nprefix_blake3={ph}");
                }
                "undo" => match agent.undo_last_edit() {
                    Ok(Some(p)) => eprintln!("\x1b[32mreverted\x1b[0m {}", p.display()),
                    Ok(None) => eprintln!("nothing to undo"),
                    Err(e) => eprintln!("\x1b[31mundo failed:\x1b[0m {e}"),
                },
                "compact" => {
                    let focus = cmd
                        .strip_prefix("compact")
                        .map(str::trim)
                        .filter(|s| !s.is_empty());
                    eprintln!("\x1b[2mcompacting context (cache-warm semantic handoff)...\x1b[0m");
                    match agent.compact_now_with_focus(focus).await {
                        Ok((before, after, semantic)) => {
                            let mode = if semantic {
                                "semantic-llm"
                            } else {
                                "deterministic"
                            };
                            eprintln!(
                                "\x1b[32mcompacted ({mode}):\x1b[0m ~{before} -> ~{after} est. tokens"
                            );
                        }
                        Err(e) => eprintln!("\x1b[31mcompaction failed:\x1b[0m {e}"),
                    }
                }
                other => eprintln!("unknown command `/{other}` (try /help)"),
            }
            continue;
        }

        // Stream live output & tool badges; Ctrl+C aborts ONLY the current turn!
        let mut ev_stream = agent.subscribe_from(agent.latest_event_seq());
        let ev_task = tokio::spawn(async move {
            let mut printed_text = false;
            loop {
                let env = ev_stream.next().await;
                match &env.event {
                    AgentEvent::TextDelta { delta } => {
                        print!("{delta}");
                        let _ = io::stdout().flush();
                        printed_text = true;
                    }
                    AgentEvent::ToolStart { tool, .. } => {
                        if printed_text {
                            println!();
                            printed_text = false;
                        }
                        eprint!("\x1b[2m  [{tool}] ...\x1b[0m");
                        let _ = io::stderr().flush();
                    }
                    AgentEvent::ToolEnd {
                        tool,
                        is_error,
                        duration_ms,
                        ..
                    } => {
                        let status = if *is_error {
                            "\x1b[31merr\x1b[0m"
                        } else {
                            "\x1b[32mok\x1b[0m"
                        };
                        eprintln!(
                            "\r\x1b[2m  [{tool}]\x1b[0m {status} \x1b[2m({duration_ms}ms)\x1b[0m"
                        );
                    }
                    AgentEvent::CompactionApplied {
                        mode,
                        tokens_before,
                        tokens_after,
                        semantic_llm,
                    } => {
                        let kind = if *semantic_llm { "semantic" } else { "det" };
                        eprintln!(
                            "\x1b[33m  [compact:{mode}/{kind}]\x1b[0m ~{tokens_before} -> ~{tokens_after} tok"
                        );
                    }
                    AgentEvent::RunSettled { metrics, .. } => {
                        if printed_text {
                            println!();
                        }
                        eprintln!(
                            "\x1b[2m  [done · {} turns · ${:.4} · {:.1}% cache]\x1b[0m",
                            metrics.turns,
                            metrics.total_usage.cost_usd,
                            metrics.cache_hit_rate() * 100.0
                        );
                        break;
                    }
                    AgentEvent::Error { .. } => break,
                    _ => {}
                }
            }
        });

        let agent_ctrlc = Arc::clone(&agent);
        let ctrlc_task = tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                agent_ctrlc.abort().await;
            }
        });

        let res = agent.prompt(trimmed).await;
        ctrlc_task.abort();
        let _ = tokio::time::timeout(std::time::Duration::from_millis(250), ev_task).await;

        if let Err(e) = res {
            eprintln!("\x1b[31merror:\x1b[0m {e}");
        }
    }
}
