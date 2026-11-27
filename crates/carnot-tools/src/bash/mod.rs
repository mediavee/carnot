//! Adaptive-Yield Sandboxed Shell (`Bash`) + Interactive PTY + Background `JobRegistry`:
//!   - `watch::channel<Option<i32>>` eliminates lost-wakeup races on fast commands
//!   - Live streaming chunked stdout/stderr into bounded 8 MiB ring/head+tail buffer + throttled
//!     live [`ToolProgressSink`] emission (`AgentEvent::ToolUpdate`)
//!   - Optional OS-sandboxed interactive PTY mode (`pty: true` + `stdin`) via `portable-pty`
//!   - 500 ms post-exit pipe drain grace so backgrounded grandchildren (`sleep 8 &`) never hang
//!   - Entire process-group SIGTERM/SIGKILL cleanup via safe `rustix::process::kill_process_group`
//!     syscall (zero `unsafe`, zero `/bin/kill` subprocess dependency)
//!   - Signal exit codes reported as `128 + sig`, saturating timeout arithmetic
//!   - Finished jobs pruned on collection and bounded to 64 retained entries

use crate::files::resolve_confined_path;
use crate::{BoxFuture, Tool, ToolProgressSink};
use carnot_cas::{CasStore, shape_command_output_with_redaction};
use carnot_core::{CarnotError, Result, ToolEffects, ToolOutput, ToolSpec};
use carnot_sandbox::{
    SandboxPolicy, annotate_sandbox_denial, build_sandboxed_shell_in_dir,
    build_sandboxed_shell_spec_in_dir,
};
use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};
use std::io::Read as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

mod jobs;
#[cfg(test)]
mod tests;

pub use jobs::JobRegistry;

#[cfg(unix)]
use self::jobs::signal_process_group;
use self::jobs::{
    BackgroundJob, BoundedOutput, DEFAULT_PTY_COLS, DEFAULT_PTY_ROWS, ProgressSlot, PtyHandle,
    kill_process_group, pty_exit_status_to_code, status_to_exit_code,
};

struct FormattedBashOutput {
    text: String,
    handle: Option<String>,
    bytes_saved: usize,
    compressor: Option<&'static str>,
    deduplicated: bool,
}

#[derive(Debug)]
pub struct BashTool {
    workspace_root: PathBuf,
    sandbox: SandboxPolicy,
    cas: Arc<CasStore>,
    jobs: Arc<JobRegistry>,
    seen_cas_hashes: Arc<Mutex<HashSet<String>>>,
    cas_enabled: bool,
    redact_cas_secrets: bool,
    spill_threshold: usize,
    preview_max: usize,
    yield_ms: u64,
    max_timeout_ms: u64,
    extra_env: BTreeMap<String, String>,
}

impl BashTool {
    #[expect(
        clippy::too_many_arguments,
        reason = "BashTool constructor wires explicit sandbox, CAS, and timeout settings from CarnotConfig"
    )]
    pub fn new(
        workspace_root: PathBuf,
        sandbox: SandboxPolicy,
        cas: Arc<CasStore>,
        jobs: Arc<JobRegistry>,
        cas_enabled: bool,
        spill_threshold: usize,
        preview_max: usize,
        yield_ms: u64,
        max_timeout_ms: u64,
        extra_env: BTreeMap<String, String>,
    ) -> Self {
        Self {
            workspace_root,
            sandbox,
            cas,
            jobs,
            seen_cas_hashes: Arc::new(Mutex::new(HashSet::new())),
            cas_enabled,
            redact_cas_secrets: true,
            spill_threshold,
            preview_max,
            yield_ms,
            max_timeout_ms,
            extra_env,
        }
    }

    #[must_use]
    pub fn with_redact_cas_secrets(mut self, redact: bool) -> Self {
        self.redact_cas_secrets = redact;
        self
    }

    /// Attaches a shared CAS hash set (cleared on compaction/rewind by `ToolRegistry::reset_read_tracker`)
    /// so repeated identical large command outputs collapse to a 1-line dedup receipt.
    #[must_use]
    pub fn with_seen_cas_hashes(mut self, seen_cas_hashes: Arc<Mutex<HashSet<String>>>) -> Self {
        self.seen_cas_hashes = seen_cas_hashes;
        self
    }

    fn format_output(
        &self,
        command: &str,
        raw: &str,
        allow_dedup: bool,
    ) -> Result<FormattedBashOutput> {
        let shaped = shape_command_output_with_redaction(
            Some(command),
            raw,
            self.cas_enabled.then_some(self.cas.as_ref()),
            self.spill_threshold,
            self.preview_max,
            self.redact_cas_secrets,
        )?;
        if allow_dedup
            && let Some(ref handle) = shaped.spilled
            && let Ok(mut seen) = self.seen_cas_hashes.lock()
            && !seen.insert(handle.hash.clone())
        {
            let dedup_text = format!(
                "[identical output to earlier run — {} B ({} lines) archived at {}; inspect with Read/Grep if needed]",
                handle.bytes, handle.lines, handle.rel_path
            );
            let saved = shaped.raw_bytes.saturating_sub(dedup_text.len());
            return Ok(FormattedBashOutput {
                text: dedup_text,
                handle: Some(handle.rel_path.clone()),
                bytes_saved: saved,
                compressor: shaped.compressor,
                deduplicated: true,
            });
        }
        let saved = shaped.raw_bytes.saturating_sub(shaped.shaped_bytes);
        let handle_path = shaped.spilled.map(|h| h.rel_path);
        Ok(FormattedBashOutput {
            text: shaped.text,
            handle: handle_path,
            bytes_saved: saved,
            compressor: shaped.compressor,
            deduplicated: false,
        })
    }

    fn spawn_pty_job(
        &self,
        command: &str,
        exec_dir: &std::path::Path,
        hard_timeout_ms: u64,
        cancel: &CancellationToken,
        progress: Option<ToolProgressSink>,
    ) -> Result<(Arc<BackgroundJob>, watch::Receiver<Option<i32>>)> {
        let mut pty_env = self.extra_env.clone();
        pty_env
            .entry("TERM".into())
            .or_insert_with(|| "xterm-256color".into());
        let spec = build_sandboxed_shell_spec_in_dir(&self.sandbox, command, exec_dir, &pty_env)
            .map_err(CarnotError::Sandbox)?;

        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows: DEFAULT_PTY_ROWS,
                cols: DEFAULT_PTY_COLS,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| CarnotError::Tool {
                tool: "Bash".into(),
                message: format!("failed to open PTY: {e}"),
            })?;

        let mut cmd_builder = CommandBuilder::new(&spec.program);
        for arg in &spec.args {
            cmd_builder.arg(arg);
        }
        cmd_builder.cwd(&spec.cwd);
        if spec.clear_env {
            cmd_builder.env_clear();
        }
        for (k, v) in &spec.envs {
            match v {
                Some(val) => cmd_builder.env(k, val),
                None => cmd_builder.env_remove(k),
            }
        }

        let mut child = pair
            .slave
            .spawn_command(cmd_builder)
            .map_err(|e| CarnotError::Tool {
                tool: "Bash".into(),
                message: format!("failed to spawn PTY shell in {}: {e}", exec_dir.display()),
            })?;
        // Drop slave in parent so master observes EOF when child exits.
        drop(pair.slave);

        let mut reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| CarnotError::Tool {
                tool: "Bash".into(),
                message: format!("failed to clone PTY reader: {e}"),
            })?;
        let writer = pair.master.take_writer().map_err(|e| CarnotError::Tool {
            tool: "Bash".into(),
            message: format!("failed to take PTY writer: {e}"),
        })?;
        let mut killer = child.clone_killer();
        #[cfg(unix)]
        let child_pid = child.process_id();

        let job_cancel = cancel.child_token();
        let (exit_tx, exit_rx) = watch::channel::<Option<i32>>(None);
        let job = Arc::new(BackgroundJob {
            command: command.to_string(),
            is_pty: true,
            redact_secrets: self.redact_cas_secrets,
            started: Instant::now(),
            output: Mutex::new(BoundedOutput::default()),
            progress: Mutex::new(ProgressSlot::new(progress)),
            pty: Some(Mutex::new(Some(PtyHandle {
                writer,
                _master: pair.master,
            }))),
            exit_rx: exit_rx.clone(),
            cancel: job_cancel.clone(),
        });

        // Blocking PTY reader thread -> pushes chunks into ring buffer & live progress sink.
        let job_read = Arc::clone(&job);
        let read_handle = tokio::task::spawn_blocking(move || {
            let mut buf = [0u8; 4096];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => job_read.push_and_notify(&buf[..n]),
                }
            }
        });

        // Blocking child wait thread.
        let (wait_tx, wait_rx) = tokio::sync::oneshot::channel::<i32>();
        tokio::task::spawn_blocking(move || {
            let code = match child.wait() {
                Ok(status) => pty_exit_status_to_code(&status),
                Err(_) => -1,
            };
            let _ = wait_tx.send(code);
        });

        let job_bg = Arc::clone(&job);
        tokio::spawn(async move {
            let exit = tokio::select! {
                () = job_cancel.cancelled() => {
                    #[cfg(unix)]
                    if let Some(pid) = child_pid {
                        signal_process_group(pid, rustix::process::Signal::TERM);
                        tokio::time::sleep(Duration::from_millis(80)).await;
                        signal_process_group(pid, rustix::process::Signal::KILL);
                    }
                    let _ = killer.kill();
                    137
                }
                () = tokio::time::sleep(Duration::from_millis(hard_timeout_ms)) => {
                    #[cfg(unix)]
                    if let Some(pid) = child_pid {
                        signal_process_group(pid, rustix::process::Signal::TERM);
                        tokio::time::sleep(Duration::from_millis(80)).await;
                        signal_process_group(pid, rustix::process::Signal::KILL);
                    }
                    let _ = killer.kill();
                    124
                }
                res = wait_rx => {
                    res.unwrap_or(-1)
                }
            };

            let read_abort = read_handle.abort_handle();
            if tokio::time::timeout(Duration::from_millis(500), read_handle)
                .await
                .is_err()
            {
                read_abort.abort();
            }
            // Drop PTY master + writer immediately on exit so no stale fd/reader lingers.
            job_bg.close_pty();
            job_bg.flush_progress();
            let _ = exit_tx.send(Some(exit));
        });

        Ok((job, exit_rx))
    }

    async fn run_impl(
        &self,
        input: Value,
        cancel: CancellationToken,
        progress: Option<ToolProgressSink>,
    ) -> Result<ToolOutput> {
        // Mode A: Collect, write stdin to, or kill an existing yielded background job
        if let Some(job_id) = input
            .get("job")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let Some(job) = self.jobs.get(job_id) else {
                return Ok(ToolOutput::err_text(format!("Unknown job id: {job_id}")));
            };
            if input.get("kill").and_then(Value::as_bool).unwrap_or(false) {
                job.cancel.cancel();
                self.jobs.remove(job_id);
                return Ok(ToolOutput::ok_text(format!(
                    "Killed {job_id} ({})",
                    job.command
                )));
            }
            let wait_ms = input
                .get("timeout")
                .and_then(Value::as_u64)
                .map_or(self.yield_ms, |s| s.saturating_mul(1000))
                .min(120_000);
            if let Some(stdin_text) = input.get("stdin").and_then(Value::as_str)
                && !stdin_text.is_empty()
                && let Err(e) = job
                    .write_stdin_async(
                        stdin_text.to_string(),
                        Duration::from_millis(wait_ms.clamp(1_000, 10_000)),
                        &cancel,
                    )
                    .await
            {
                return Ok(ToolOutput::err_text(e));
            }

            job.attach_progress(progress);
            let mut rx = job.exit_rx.clone();
            if rx.borrow().is_none() {
                tokio::select! {
                    () = cancel.cancelled() => {
                        job.attach_progress(None);
                        return Ok(ToolOutput::err_text(format!("Cancelled while polling {job_id}")));
                    }
                    _ = tokio::time::timeout(
                        Duration::from_millis(wait_ms),
                        rx.wait_for(std::option::Option::is_some),
                    ) => {}
                }
            }
            job.attach_progress(None);

            let exit = *rx.borrow();
            let delta = job
                .output
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take_delta();
            let FormattedBashOutput {
                text: mut shaped,
                handle,
                bytes_saved,
                compressor,
                deduplicated,
            } = self.format_output(&job.command, &delta, exit.is_some())?;
            if let Some(code) = exit {
                if code != 0
                    && let Some(note) = annotate_sandbox_denial(&delta, &self.sandbox)
                {
                    shaped.push_str(&note);
                }
                self.jobs.remove(job_id);
            }
            let status = match exit {
                Some(code) => format!(
                    "[{job_id} finished exit={code} in {:.1}s]",
                    job.started.elapsed().as_secs_f64()
                ),
                None => format!(
                    "[{job_id} still running ({:.1}s) — call Bash{{\"job\":\"{job_id}\"}} again]",
                    job.started.elapsed().as_secs_f64()
                ),
            };
            let body = if shaped.is_empty() {
                status
            } else {
                format!("{shaped}\n{status}")
            };
            return Ok(ToolOutput {
                blocks: vec![carnot_core::ToolResultBlock::text(body)],
                is_error: exit.is_some_and(|c| c != 0),
                details: json!({
                    "job_id": job_id,
                    "exit_code": exit,
                    "handle": handle,
                    "command": job.command,
                    "pty": job.is_pty,
                    "bytes_saved": bytes_saved,
                    "compressor": compressor,
                    "deduplicated": deduplicated,
                }),
            });
        }

        // Mode B: Launch a new command with adaptive yield (pipe or interactive PTY)
        let Some(command) = input.get("command").and_then(Value::as_str) else {
            return Ok(ToolOutput::err_text("Bash requires `command` or `job`"));
        };
        let use_pty = input.get("pty").and_then(Value::as_bool).unwrap_or(false);
        let initial_stdin = input.get("stdin").and_then(Value::as_str);
        if initial_stdin.is_some_and(|s| !s.is_empty()) && !use_pty {
            return Ok(ToolOutput::err_text(
                "`stdin` requires `pty: true` (or an existing PTY `job`)",
            ));
        }

        let explicit_timeout_ms = input
            .get("timeout")
            .and_then(Value::as_u64)
            .filter(|&s| s > 0)
            .map(|s| s.saturating_mul(1000).min(self.max_timeout_ms));
        let hard_timeout_ms = explicit_timeout_ms.unwrap_or(self.max_timeout_ms);
        // When the caller sets an explicit timeout, wait past `hard_timeout_ms` + kill/drain grace
        // so a timed-out command reports clean `exit=124` instead of racing into a phantom yielded job.
        let yield_ms = match explicit_timeout_ms {
            Some(t) => t.saturating_add(1_500),
            None => self.yield_ms.min(hard_timeout_ms),
        };

        let exec_dir = if let Some(raw_cwd) = input
            .get("cwd")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let resolved = match resolve_confined_path(
                &self.workspace_root,
                &self.sandbox.extra_writable_roots,
                raw_cwd,
            ) {
                Ok(p) => p,
                Err(msg) => {
                    return Ok(ToolOutput::err_text(format!("Invalid Bash cwd: {msg}")));
                }
            };
            if !resolved.is_dir() {
                return Ok(ToolOutput::err_text(format!(
                    "Bash cwd is not an existing directory: {raw_cwd}"
                )));
            }
            resolved
        } else {
            self.workspace_root.clone()
        };

        let (job, exit_rx) = if use_pty {
            let (pty_job, pty_exit_rx) =
                self.spawn_pty_job(command, &exec_dir, hard_timeout_ms, &cancel, progress)?;
            if let Some(input_text) = initial_stdin
                && !input_text.is_empty()
                && let Err(e) = pty_job
                    .write_stdin_async(
                        input_text.to_string(),
                        Duration::from_millis(hard_timeout_ms.clamp(1_000, 10_000)),
                        &cancel,
                    )
                    .await
            {
                pty_job.attach_progress(None);
                pty_job.cancel.cancel();
                return Ok(ToolOutput::err_text(e));
            }
            (pty_job, pty_exit_rx)
        } else {
            let mut cmd =
                build_sandboxed_shell_in_dir(&self.sandbox, command, &exec_dir, &self.extra_env)
                    .map_err(CarnotError::Sandbox)?;
            let mut child = cmd.spawn().map_err(|e| CarnotError::Tool {
                tool: "Bash".into(),
                message: format!(
                    "failed to spawn shell in {}: {e}",
                    self.workspace_root.display()
                ),
            })?;

            let stdout = child.stdout.take();
            let stderr = child.stderr.take();
            let job_cancel = cancel.child_token();
            let (exit_tx, exit_rx) = watch::channel::<Option<i32>>(None);
            let job = Arc::new(BackgroundJob {
                command: command.to_string(),
                is_pty: false,
                redact_secrets: self.redact_cas_secrets,
                started: Instant::now(),
                output: Mutex::new(BoundedOutput::default()),
                progress: Mutex::new(ProgressSlot::new(progress)),
                pty: None,
                exit_rx: exit_rx.clone(),
                cancel: job_cancel.clone(),
            });

            // Background pump: live chunked readers + native process-group kill + 500ms pipe drain grace!
            let job_bg = Arc::clone(&job);
            tokio::spawn(async move {
                let job_out = Arc::clone(&job_bg);
                let out_task = tokio::spawn(async move {
                    if let Some(mut s) = stdout {
                        let mut buf = [0u8; 8192];
                        loop {
                            match s.read(&mut buf).await {
                                Ok(0) | Err(_) => break,
                                Ok(n) => job_out.push_and_notify(&buf[..n]),
                            }
                        }
                    }
                });

                let job_err = Arc::clone(&job_bg);
                let err_task = tokio::spawn(async move {
                    if let Some(mut s) = stderr {
                        let mut buf = [0u8; 8192];
                        loop {
                            match s.read(&mut buf).await {
                                Ok(0) | Err(_) => break,
                                Ok(n) => job_err.push_and_notify(&buf[..n]),
                            }
                        }
                    }
                });

                let exit = tokio::select! {
                    () = job_cancel.cancelled() => {
                        kill_process_group(&mut child).await;
                        137
                    }
                    () = tokio::time::sleep(Duration::from_millis(hard_timeout_ms)) => {
                        kill_process_group(&mut child).await;
                        124
                    }
                    res = child.wait() => {
                        res.map_or(-1, status_to_exit_code)
                    }
                };

                // Drain stdout/stderr up to 500 ms so backgrounded grandchildren never hang Bash.
                let out_abort = out_task.abort_handle();
                let err_abort = err_task.abort_handle();
                let drain = async {
                    let _ = out_task.await;
                    let _ = err_task.await;
                };
                if tokio::time::timeout(Duration::from_millis(500), drain)
                    .await
                    .is_err()
                {
                    out_abort.abort();
                    err_abort.abort();
                }

                job_bg.flush_progress();
                let _ = exit_tx.send(Some(exit));
            });

            (job, exit_rx)
        };

        // Wait up to `yield_ms` via watch::Receiver (zero lost wakeup).
        let mut wait_rx = exit_rx;
        let waited_done = tokio::select! {
            () = cancel.cancelled() => {
                job.attach_progress(None);
                job.cancel.cancel();
                return Err(CarnotError::Cancelled);
            }
            res = wait_rx.wait_for(std::option::Option::is_some) => res.is_ok(),
            () = tokio::time::sleep(Duration::from_millis(yield_ms)) => false,
        };
        job.attach_progress(None);
        let completed = waited_done || wait_rx.borrow().is_some();

        if completed {
            let exit = wait_rx.borrow().unwrap_or(-1);
            let raw = job
                .output
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take_all();
            let FormattedBashOutput {
                text: mut shaped,
                handle,
                bytes_saved,
                compressor,
                deduplicated,
            } = self.format_output(command, &raw, true)?;
            if exit != 0
                && let Some(note) = annotate_sandbox_denial(&raw, &self.sandbox)
            {
                shaped.push_str(&note);
            }
            let elapsed = job.started.elapsed().as_secs_f64();
            let text = if shaped.is_empty() {
                format!("[exit={exit} in {elapsed:.2}s]")
            } else {
                format!("{shaped}\n[exit={exit} in {elapsed:.2}s]")
            };
            Ok(ToolOutput {
                blocks: vec![carnot_core::ToolResultBlock::text(text)],
                is_error: exit != 0,
                details: json!({
                    "command": command,
                    "exit_code": exit,
                    "duration_s": elapsed,
                    "handle": handle,
                    "pty": use_pty,
                    "bytes_saved": bytes_saved,
                    "compressor": compressor,
                    "deduplicated": deduplicated,
                    "yielded": false,
                }),
            })
        } else {
            // Adaptive Yield: register background job + return any partial output already produced!
            let partial = job
                .output
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take_delta();
            let job_id = self.jobs.insert(Arc::clone(&job));
            let header = if partial.is_empty() {
                String::new()
            } else {
                let p = self.format_output(command, &partial, false)?;
                format!("{}\n", p.text)
            };
            let hint = if use_pty {
                format!(
                    "call Bash{{\"job\":\"{job_id}\"}} to collect, {{\"job\":\"{job_id}\",\"stdin\":\"...\"}} to send input, or {{\"job\":\"{job_id}\",\"kill\":true}} to stop"
                )
            } else {
                format!(
                    "call Bash{{\"job\":\"{job_id}\"}} to collect or {{\"job\":\"{job_id}\",\"kill\":true}} to stop"
                )
            };
            Ok(ToolOutput::ok_text(format!(
                "{header}[yielded after {yield_ms}ms -> {job_id} still running: `{command}` — {hint}]"
            ))
            .with_details(json!({
                "command": command,
                "job_id": job_id,
                "pty": use_pty,
                "yielded": true,
            })))
        }
    }
}

impl Tool for BashTool {
    fn spec(&self) -> ToolSpec {
        crate::canonical_tool_specs()
            .into_iter()
            .find(|s| s.name == "Bash")
            .expect("Bash spec")
    }

    fn effects(&self, _input: &Value) -> ToolEffects {
        ToolEffects::serial()
    }

    fn execute(
        &self,
        input: Value,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<ToolOutput>> {
        Box::pin(async move { self.run_impl(input, cancel, None).await })
    }

    fn execute_with_progress<'a>(
        &'a self,
        _call_id: &'a str,
        input: Value,
        cancel: CancellationToken,
        progress: Option<ToolProgressSink>,
    ) -> BoxFuture<'a, Result<ToolOutput>> {
        Box::pin(async move { self.run_impl(input, cancel, progress).await })
    }
}
