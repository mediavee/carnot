//! Background job registry, bounded output ring buffer, PTY handles, live progress throttling,
//! and safe process-group signal cleanup for `BashTool`.

use crate::ToolProgressSink;
use portable_pty::MasterPty;
use std::collections::HashMap;
use std::io::Write as _;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

pub(super) const MAX_JOB_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const MAX_RETAINED_JOBS: usize = 64;
const PROGRESS_THROTTLE_MS: u64 = 120;
const PROGRESS_MAX_LINES: usize = 4;
const PROGRESS_MAX_CHARS: usize = 400;
pub(super) const DEFAULT_PTY_COLS: u16 = 120;
pub(super) const DEFAULT_PTY_ROWS: u16 = 36;

#[derive(Debug, Default)]
pub(super) struct BoundedOutput {
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

    fn live_preview(&self, redact_secrets: bool) -> String {
        if self.live_tail.is_empty() {
            return String::new();
        }
        let bytes: Vec<u8> = self.live_tail.iter().copied().collect();
        format_live_preview(&String::from_utf8_lossy(&bytes), redact_secrets)
    }

    pub(super) fn take_delta(&mut self) -> String {
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

    pub(super) fn take_all(&mut self) -> String {
        self.take_delta()
    }
}

/// Strips ANSI CSI/OSC escape sequences and carriage returns, optionally redacts secrets across
/// the full rolling buffer BEFORE truncating, and returns up to the last `PROGRESS_MAX_LINES`
/// non-empty lines capped at `PROGRESS_MAX_CHARS`.
pub(super) fn format_live_preview(raw: &str, redact_secrets: bool) -> String {
    let cleaned = strip_ansi_escapes(raw);
    let safe = if redact_secrets {
        carnot_cas::redact_secrets_cow(&cleaned)
    } else {
        std::borrow::Cow::Borrowed(cleaned.as_str())
    };
    let mut lines: Vec<&str> = safe
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

pub(super) fn strip_ansi_escapes(s: &str) -> String {
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

pub(super) struct ProgressSlot {
    sink: Option<ToolProgressSink>,
    last_emit: Option<Instant>,
    last_text: String,
}

impl ProgressSlot {
    pub(super) fn new(sink: Option<ToolProgressSink>) -> Self {
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

    pub(super) fn maybe_emit(&mut self, preview: &str, force: bool) {
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

pub(super) struct PtyHandle {
    pub(super) writer: Box<dyn std::io::Write + Send>,
    pub(super) _master: Box<dyn MasterPty + Send>,
}

pub(super) struct BackgroundJob {
    pub(super) command: String,
    pub(super) is_pty: bool,
    pub(super) redact_secrets: bool,
    pub(super) started: Instant,
    pub(super) output: Mutex<BoundedOutput>,
    pub(super) progress: Mutex<ProgressSlot>,
    pub(super) pty: Option<Mutex<Option<PtyHandle>>>,
    pub(super) exit_rx: watch::Receiver<Option<i32>>,
    pub(super) cancel: CancellationToken,
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
    pub(super) fn push_and_notify(&self, chunk: &[u8]) {
        let preview = {
            let mut out = self
                .output
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            out.push_bytes(chunk);
            out.live_preview(self.redact_secrets)
        };
        self.progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .maybe_emit(&preview, false);
    }

    pub(super) fn flush_progress(&self) {
        let preview = self
            .output
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .live_preview(self.redact_secrets);
        self.progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .maybe_emit(&preview, true);
    }

    pub(super) fn attach_progress(&self, sink: Option<ToolProgressSink>) {
        self.progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .set_sink(sink);
    }

    fn write_stdin_blocking(&self, data: &str) -> std::result::Result<(), String> {
        let Some(ref pty_lock) = self.pty else {
            return Err("Job was not started with `pty: true`; `stdin` requires a PTY job".into());
        };
        let mut guard = pty_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(ref mut handle) = *guard else {
            return Err("PTY job has already exited; stdin is closed".into());
        };
        handle
            .writer
            .write_all(data.as_bytes())
            .and_then(|()| handle.writer.flush())
            .map_err(|e| format!("failed to write to PTY stdin: {e}"))
    }

    /// Writes `data` to the PTY stdin off the async executor with a bounded timeout and
    /// cancellation check so a stalled child can never freeze a single-threaded Tokio runtime.
    pub(super) async fn write_stdin_async(
        self: &Arc<Self>,
        data: String,
        timeout: Duration,
        cancel: &CancellationToken,
    ) -> std::result::Result<(), String> {
        let job = Arc::clone(self);
        let mut task = tokio::task::spawn_blocking(move || job.write_stdin_blocking(&data));
        tokio::select! {
            biased;
            () = cancel.cancelled() => Err("Cancelled while writing to PTY stdin".into()),
            res = &mut task => res.map_err(|e| format!("PTY stdin task failed: {e}"))?,
            () = tokio::time::sleep(timeout) => {
                Err(format!(
                    "Timed out after {}ms writing to PTY stdin (child not reading input)",
                    timeout.as_millis()
                ))
            }
        }
    }

    pub(super) fn close_pty(&self) {
        if let Some(ref pty_lock) = self.pty {
            let _ = pty_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
        }
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

    pub(super) fn insert(&self, job: Arc<BackgroundJob>) -> String {
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

    pub(super) fn get(&self, id: &str) -> Option<Arc<BackgroundJob>> {
        self.jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(id)
            .cloned()
    }

    pub(super) fn remove(&self, id: &str) {
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
pub(super) fn status_to_exit_code(status: std::process::ExitStatus) -> i32 {
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
pub(super) fn status_to_exit_code(status: std::process::ExitStatus) -> i32 {
    status.code().unwrap_or(-1)
}

pub(super) fn pty_exit_status_to_code(status: &portable_pty::ExitStatus) -> i32 {
    if let Some(sig_name) = status.signal() {
        let lower = sig_name.to_ascii_lowercase();
        let sig_num = if let Some(rest) = lower.strip_prefix("signal ")
            && let Ok(n) = rest.trim().parse::<i32>()
        {
            n
        } else if lower.contains("hangup") || lower == "sighup" {
            1
        } else if lower.contains("interrupt") || lower == "sigint" {
            2
        } else if lower.contains("quit") || lower == "sigquit" {
            3
        } else if lower.contains("abort") || lower == "sigabrt" {
            6
        } else if lower.contains("killed") || lower == "sigkill" {
            9
        } else if lower.contains("segmentation") || lower == "sigsegv" {
            11
        } else if lower.contains("broken pipe") || lower == "sigpipe" {
            13
        } else {
            15
        };
        return 128 + sig_num;
    }
    i32::try_from(status.exit_code()).unwrap_or(-1)
}

#[cfg(unix)]
pub(super) fn signal_process_group(raw_pid: u32, sig: rustix::process::Signal) {
    if let Ok(signed) = i32::try_from(raw_pid)
        && let Some(pid) = rustix::process::Pid::from_raw(signed)
    {
        let _ = rustix::process::kill_process_group(pid, sig);
    }
}

/// Kills the child's entire process group (`process_group(0)` => pgid == pid) via native
/// `rustix::process::kill_process_group` syscalls — zero `unsafe` and zero `/bin/kill` subprocess.
pub(super) async fn kill_process_group(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        signal_process_group(pid, rustix::process::Signal::TERM);
        tokio::time::sleep(Duration::from_millis(80)).await;
        signal_process_group(pid, rustix::process::Signal::KILL);
    }
    let _ = child.kill().await;
    let _ = child.wait().await;
}
