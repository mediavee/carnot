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
use portable_pty::{CommandBuilder, MasterPty, PtySize, native_pty_system};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read as _, Write as _};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

const MAX_JOB_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const MAX_RETAINED_JOBS: usize = 64;
const PROGRESS_THROTTLE_MS: u64 = 120;
const PROGRESS_MAX_LINES: usize = 4;
const PROGRESS_MAX_CHARS: usize = 400;
const DEFAULT_PTY_COLS: u16 = 120;
const DEFAULT_PTY_ROWS: u16 = 36;

#[derive(Debug, Default)]
struct BoundedOutput {
    /// Raw unread byte ring (O(1) front eviction capped at `MAX_JOB_BUFFER_BYTES`).
    pending_bytes: std::collections::VecDeque<u8>,
    omitted_bytes: usize,
    /// Small rolling tail used solely for live progress previews (never drained by `take_delta`).
    live_tail: std::collections::VecDeque<u8>,
}

const LIVE_TAIL_CAP_BYTES: usize = 4096;

impl BoundedOutput {
    fn push_bytes(&mut self, chunk: &[u8]) {
        self.pending_bytes.extend(chunk);
        if self.pending_bytes.len() > MAX_JOB_BUFFER_BYTES {
            let excess = self.pending_bytes.len() - MAX_JOB_BUFFER_BYTES;
            self.pending_bytes.drain(..excess);
            self.omitted_bytes += excess;
        }
        self.live_tail.extend(chunk);
        if self.live_tail.len() > LIVE_TAIL_CAP_BYTES {
            let excess = self.live_tail.len() - LIVE_TAIL_CAP_BYTES;
            self.live_tail.drain(..excess);
        }
    }

    fn live_preview(&self) -> String {
        if self.live_tail.is_empty() {
            return String::new();
        }
        let bytes: Vec<u8> = self.live_tail.iter().copied().collect();
        format_live_preview(&String::from_utf8_lossy(&bytes))
    }

    fn take_delta(&mut self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        if self.omitted_bytes > 0 {
            let _ = writeln!(
                s,
                "[... {} older bytes omitted after 8 MiB ring cap ...]",
                self.omitted_bytes
            );
            self.omitted_bytes = 0;
        }
        if !self.pending_bytes.is_empty() {
            let bytes: Vec<u8> = self.pending_bytes.drain(..).collect();
            s.push_str(&String::from_utf8_lossy(&bytes));
        }
        s
    }

    fn take_all(&mut self) -> String {
        self.take_delta()
    }
}

/// Strips ANSI CSI/OSC escape sequences and carriage returns, returning up to the last
/// `PROGRESS_MAX_LINES` non-empty lines capped at `PROGRESS_MAX_CHARS`.
fn format_live_preview(raw: &str) -> String {
    let cleaned = strip_ansi_escapes(raw);
    let mut lines: Vec<&str> = cleaned
        .lines()
        .map(|l| l.rsplit('\r').next().unwrap_or(l).trim_end())
        .filter(|l| !l.trim().is_empty())
        .collect();
    if lines.len() > PROGRESS_MAX_LINES {
        lines = lines.split_off(lines.len() - PROGRESS_MAX_LINES);
    }
    let joined = lines.join("\n");
    if joined.chars().count() > PROGRESS_MAX_CHARS {
        let tail: String = joined
            .chars()
            .rev()
            .take(PROGRESS_MAX_CHARS)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        format!("…{tail}")
    } else {
        joined
    }
}

fn strip_ansi_escapes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            match chars.peek().copied() {
                Some('[') => {
                    let _ = chars.next();
                    for ch in chars.by_ref() {
                        if ('@'..='~').contains(&ch) {
                            break;
                        }
                    }
                }
                Some(']') => {
                    let _ = chars.next();
                    while let Some(ch) = chars.next() {
                        if ch == '\x07' {
                            break;
                        }
                        if ch == '\x1b' && chars.peek() == Some(&'\\') {
                            let _ = chars.next();
                            break;
                        }
                    }
                }
                Some(_) => {
                    let _ = chars.next();
                }
                None => {}
            }
        } else if c != '\0' {
            out.push(c);
        }
    }
    out
}

struct ProgressSlot {
    sink: Option<ToolProgressSink>,
    last_emit: Option<Instant>,
    last_text: String,
}

impl ProgressSlot {
    fn new(sink: Option<ToolProgressSink>) -> Self {
        Self {
            sink,
            last_emit: None,
            last_text: String::new(),
        }
    }

    fn set_sink(&mut self, sink: Option<ToolProgressSink>) {
        self.sink = sink;
        self.last_emit = None;
        self.last_text.clear();
    }

    fn maybe_emit(&mut self, preview: &str, force: bool) {
        let Some(ref sink) = self.sink else {
            return;
        };
        if preview.is_empty() || preview == self.last_text {
            return;
        }
        let now = Instant::now();
        if !force
            && let Some(prev) = self.last_emit
            && now.duration_since(prev) < Duration::from_millis(PROGRESS_THROTTLE_MS)
        {
            return;
        }
        self.last_emit = Some(now);
        self.last_text = preview.to_string();
        sink(self.last_text.clone());
    }
}

struct PtyHandle {
    writer: Box<dyn std::io::Write + Send>,
    _master: Box<dyn MasterPty + Send>,
}

struct BackgroundJob {
    command: String,
    is_pty: bool,
    started: Instant,
    output: Mutex<BoundedOutput>,
    progress: Mutex<ProgressSlot>,
    pty: Option<Mutex<PtyHandle>>,
    exit_rx: watch::Receiver<Option<i32>>,
    cancel: CancellationToken,
}

impl std::fmt::Debug for BackgroundJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundJob")
            .field("command", &self.command)
            .field("is_pty", &self.is_pty)
            .field("started", &self.started)
            .finish_non_exhaustive()
    }
}

impl BackgroundJob {
    fn push_and_notify(&self, chunk: &[u8]) {
        let preview = {
            let mut out = self
                .output
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            out.push_bytes(chunk);
            out.live_preview()
        };
        self.progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .maybe_emit(&preview, false);
    }

    fn flush_progress(&self) {
        let preview = self
            .output
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .live_preview();
        self.progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .maybe_emit(&preview, true);
    }

    fn attach_progress(&self, sink: Option<ToolProgressSink>) {
        self.progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .set_sink(sink);
    }

    fn write_stdin(&self, data: &str) -> std::result::Result<(), String> {
        let Some(ref pty_lock) = self.pty else {
            return Err("Job was not started with `pty: true`; `stdin` requires a PTY job".into());
        };
        let mut guard = pty_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard
            .writer
            .write_all(data.as_bytes())
            .and_then(|()| guard.writer.flush())
            .map_err(|e| format!("failed to write to PTY stdin: {e}"))
    }
}

#[derive(Debug, Default)]
pub struct JobRegistry {
    jobs: Mutex<HashMap<String, Arc<BackgroundJob>>>,
    counter: Mutex<u64>,
}

impl JobRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    fn insert(&self, job: Arc<BackgroundJob>) -> String {
        let mut c = self
            .counter
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *c += 1;
        let id = format!("job-{}", *c);
        let mut map = self
            .jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Evict oldest finished jobs if cap exceeded.
        if map.len() >= MAX_RETAINED_JOBS {
            let finished: Vec<String> = map
                .iter()
                .filter(|(_, j)| j.exit_rx.borrow().is_some())
                .map(|(k, _)| k.clone())
                .collect();
            for k in finished {
                map.remove(&k);
            }
        }
        map.insert(id.clone(), job);
        id
    }

    fn get(&self, id: &str) -> Option<Arc<BackgroundJob>> {
        self.jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(id)
            .cloned()
    }

    fn remove(&self, id: &str) {
        self.jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id);
    }

    pub fn kill_all(&self) {
        if let Ok(map) = self.jobs.lock() {
            for j in map.values() {
                j.cancel.cancel();
            }
        }
    }
}

impl Drop for JobRegistry {
    fn drop(&mut self) {
        self.kill_all();
    }
}

#[cfg(unix)]
fn status_to_exit_code(status: std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    if let Some(code) = status.code() {
        code
    } else if let Some(sig) = status.signal() {
        128 + sig
    } else {
        -1
    }
}

#[cfg(not(unix))]
fn status_to_exit_code(status: std::process::ExitStatus) -> i32 {
    status.code().unwrap_or(-1)
}

#[cfg(unix)]
fn signal_process_group(raw_pid: u32, sig: rustix::process::Signal) {
    if let Ok(signed) = i32::try_from(raw_pid)
        && let Some(pid) = rustix::process::Pid::from_raw(signed)
    {
        let _ = rustix::process::kill_process_group(pid, sig);
    }
}

/// Kills the child's entire process group (`process_group(0)` => pgid == pid) via native
/// `rustix::process::kill_process_group` syscalls — zero `unsafe` and zero `/bin/kill` subprocess.
async fn kill_process_group(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        signal_process_group(pid, rustix::process::Signal::TERM);
        tokio::time::sleep(Duration::from_millis(80)).await;
        signal_process_group(pid, rustix::process::Signal::KILL);
    }
    let _ = child.kill().await;
    let _ = child.wait().await;
}

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
        initial_stdin: Option<&str>,
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
                message: format!(
                    "failed to spawn PTY shell in {}: {e}",
                    exec_dir.display()
                ),
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
        let mut writer = pair.master.take_writer().map_err(|e| CarnotError::Tool {
            tool: "Bash".into(),
            message: format!("failed to take PTY writer: {e}"),
        })?;
        if let Some(input_text) = initial_stdin
            && !input_text.is_empty()
        {
            let _ = writer.write_all(input_text.as_bytes());
            let _ = writer.flush();
        }
        let mut killer = child.clone_killer();
        let child_pid = child.process_id();

        let job_cancel = cancel.child_token();
        let (exit_tx, exit_rx) = watch::channel::<Option<i32>>(None);
        let job = Arc::new(BackgroundJob {
            command: command.to_string(),
            is_pty: true,
            started: Instant::now(),
            output: Mutex::new(BoundedOutput::default()),
            progress: Mutex::new(ProgressSlot::new(progress)),
            pty: Some(Mutex::new(PtyHandle {
                writer,
                _master: pair.master,
            })),
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
                Ok(status) => i32::try_from(status.exit_code()).unwrap_or(-1),
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
            if let Some(stdin_text) = input.get("stdin").and_then(Value::as_str)
                && !stdin_text.is_empty()
                && let Err(e) = job.write_stdin(stdin_text)
            {
                return Ok(ToolOutput::err_text(e));
            }
            let wait_ms = input
                .get("timeout")
                .and_then(Value::as_u64)
                .map_or(self.yield_ms, |s| s.saturating_mul(1000))
                .min(120_000);

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
            self.spawn_pty_job(
                command,
                &exec_dir,
                hard_timeout_ms,
                initial_stdin,
                &cancel,
                progress,
            )?
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn repeated_spilled_bash_output_deduplicates_and_resets_after_clear() {
        let dir = std::env::temp_dir().join(format!("carnot-bash-dedup-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let mut cfg = carnot_core::CarnotConfig::new(dir.clone());
        cfg.modules.sandbox = false;
        cfg.cas_spill_threshold_bytes = 256;
        cfg.cas_preview_max_bytes = 128;

        let cas = Arc::new(CasStore::open(&dir).expect("cas"));
        let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let bash = reg.get("Bash").expect("Bash");

        let cmd = "python3 -c \"print('\\n'.join(f'entry-{i:04d}-payload-abcdefghij' for i in range(40)))\"";
        let first = bash
            .execute(json!({"command": cmd}), CancellationToken::new())
            .await
            .expect("first run");
        assert!(!first.is_error, "first run failed: {}", first.text_only());
        assert_eq!(first.details["deduplicated"], false, "first not deduped");
        assert!(
            first.text_only().contains(".carnot/h/"),
            "expected CAS handle: {}",
            first.text_only()
        );

        let second = bash
            .execute(json!({"command": cmd}), CancellationToken::new())
            .await
            .expect("second run");
        assert!(
            !second.is_error,
            "second run failed: {}",
            second.text_only()
        );
        assert_eq!(second.details["deduplicated"], true, "second deduped");
        assert!(
            second
                .text_only()
                .contains("identical output to earlier run"),
            "got: {}",
            second.text_only()
        );
        assert!(
            second.text_only().len() < first.text_only().len(),
            "dedup receipt must be shorter"
        );

        // Compaction / rewind calls `reset_read_tracker()` which must clear seen CAS hashes too!
        reg.reset_read_tracker();
        let third = bash
            .execute(json!({"command": cmd}), CancellationToken::new())
            .await
            .expect("third run after reset");
        assert_eq!(third.details["deduplicated"], false, "reset clears dedup");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn spilled_bash_output_redacts_secrets_before_writing_to_cas_disk() {
        let dir = std::env::temp_dir().join(format!("carnot-bash-redact-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let mut cfg = carnot_core::CarnotConfig::new(dir.clone());
        cfg.modules.sandbox = false;
        cfg.cas_spill_threshold_bytes = 256;
        cfg.cas_preview_max_bytes = 128;
        cfg.redact_cas_secrets = true;

        let cas = Arc::new(CasStore::open(&dir).expect("cas"));
        let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let bash = reg.get("Bash").expect("Bash");

        let secret = "sk-ant-api03-secretvalue1234567890abcdef";
        let cmd = format!(
            "python3 -c \"print('TOKEN={secret}'); print('\\n'.join(f'line-{{i:04d}}-padding-abcdef' for i in range(40)))\""
        );
        let out = bash
            .execute(json!({"command": cmd}), CancellationToken::new())
            .await
            .expect("run");
        assert!(!out.is_error, "run failed: {}", out.text_only());
        assert!(
            !out.text_only().contains(secret),
            "secret leaked in preview"
        );
        let rel = out.details["handle"].as_str().expect("spilled CAS handle");
        let on_disk = std::fs::read_to_string(dir.join(rel)).expect("read CAS blob");
        assert!(
            !on_disk.contains(secret),
            "raw secret must NEVER be written to .carnot/h/ on disk"
        );
        assert!(
            on_disk.contains("[REDACTED:ANTHROPIC_KEY]"),
            "expected redaction marker"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn live_progress_sink_receives_incremental_bash_output() {
        let dir =
            std::env::temp_dir().join(format!("carnot-bash-stream-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let mut cfg = carnot_core::CarnotConfig::new(dir.clone());
        cfg.modules.sandbox = false;

        let cas = Arc::new(CasStore::open(&dir).expect("cas"));
        let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let bash = reg.get("Bash").expect("Bash");

        let updates: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let updates_cb = Arc::clone(&updates);
        let sink: ToolProgressSink = Arc::new(move |msg: String| {
            updates_cb.lock().expect("lock").push(msg);
        });

        let out = bash
            .execute_with_progress(
                "call-stream-1",
                json!({"command": "echo step-one; sleep 0.2; echo step-two"}),
                CancellationToken::new(),
                Some(sink),
            )
            .await
            .expect("run");
        assert!(!out.is_error, "command failed: {}", out.text_only());
        let seen = updates.lock().expect("lock").clone();
        assert!(
            !seen.is_empty(),
            "expected at least one live progress update"
        );
        assert!(
            seen.iter().any(|m| m.contains("step-one")),
            "expected step-one in live updates: {seen:?}"
        );
        assert!(
            seen.iter().any(|m| m.contains("step-two")),
            "expected step-two in live updates: {seen:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn pty_mode_provides_real_tty_and_accepts_interactive_stdin() {
        let dir = std::env::temp_dir().join(format!("carnot-bash-pty-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let mut cfg = carnot_core::CarnotConfig::new(dir.clone());
        cfg.modules.sandbox = false;
        cfg.bash_yield_ms = 250;

        let cas = Arc::new(CasStore::open(&dir).expect("cas"));
        let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let bash = reg.get("Bash").expect("Bash");

        // 1. Verify PTY allocates a real TTY (`test -t 1`).
        let tty_out = bash
            .execute(
                json!({"command": "test -t 1 && echo IS_TTY || echo NOT_TTY", "pty": true}),
                CancellationToken::new(),
            )
            .await
            .expect("pty tty check");
        assert!(!tty_out.is_error, "tty check failed: {}", tty_out.text_only());
        assert!(
            tty_out.text_only().contains("IS_TTY"),
            "expected real TTY in pty mode, got: {}",
            tty_out.text_only()
        );
        assert_eq!(tty_out.details["pty"], true, "pty flag in details");

        // 2. Interactive command yields while waiting for stdin, then completes once stdin is sent.
        let yielded = bash
            .execute(
                json!({"command": "printf 'PROMPT> '; read line; echo \"GOT:$line\"", "pty": true}),
                CancellationToken::new(),
            )
            .await
            .expect("pty yield");
        assert_eq!(yielded.details["yielded"], true, "expected yielded PTY job");
        let job_id = yielded.details["job_id"]
            .as_str()
            .expect("job_id")
            .to_string();

        let done = bash
            .execute(
                json!({"job": job_id, "stdin": "hello-pty\n", "timeout": 5}),
                CancellationToken::new(),
            )
            .await
            .expect("pty stdin reply");
        assert!(!done.is_error, "pty reply failed: {}", done.text_only());
        assert!(
            done.text_only().contains("GOT:hello-pty"),
            "expected interactive echo, got: {}",
            done.text_only()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn timeout_kills_process_group_via_native_syscall() {
        let dir = std::env::temp_dir().join(format!("carnot-bash-kill-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let mut cfg = carnot_core::CarnotConfig::new(dir.clone());
        cfg.modules.sandbox = false;

        let cas = Arc::new(CasStore::open(&dir).expect("cas"));
        let reg = crate::ToolRegistry::with_canonical_lid_tools(&cfg, cas);
        let bash = reg.get("Bash").expect("Bash");

        let out = bash
            .execute(
                json!({"command": "sleep 30", "timeout": 1}),
                CancellationToken::new(),
            )
            .await
            .expect("timeout run");
        assert!(out.is_error, "timeout must mark error");
        assert_eq!(out.details["exit_code"], 124, "timeout exit code 124");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
