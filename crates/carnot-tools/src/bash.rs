//! Adaptive-Yield Sandboxed Shell (`Bash`) + Background `JobRegistry` — hardened (Opus B1–B6):
//!   - `watch::channel<Option<i32>>` eliminates lost-wakeup race on fast commands (B1)
//!   - Live streaming chunked stdout/stderr into bounded 8 MiB ring/head+tail buffer so polling
//!     a yielded job returns partial output immediately (B2)
//!   - 500 ms post-exit pipe drain grace so backgrounded grandchildren (`sleep 8 &`) never hang (B3)
//!   - Entire process-group SIGTERM/SIGKILL cleanup without `unsafe` (`kill -KILL -<pgid>`) (B4)
//!   - Signal exit codes reported as `128 + sig`, saturating timeout arithmetic (B5)
//!   - Finished jobs pruned on collection & bounded to 64 retained entries (B6)

use crate::files::resolve_confined_path;
use crate::{BoxFuture, Tool};
use carnot_cas::{CasStore, shape_command_output};
use carnot_core::{CarnotError, Result, ToolEffects, ToolOutput, ToolSpec};
use carnot_sandbox::{SandboxPolicy, annotate_sandbox_denial, build_sandboxed_shell_in_dir};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

const MAX_JOB_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const MAX_RETAINED_JOBS: usize = 64;

#[derive(Debug, Default)]
struct BoundedOutput {
    /// Raw unread byte ring (capped at `MAX_JOB_BUFFER_BYTES`, keeping latest tail bytes — fixes B3 & N15!)
    pending_bytes: Vec<u8>,
    omitted_bytes: usize,
}

impl BoundedOutput {
    fn push_bytes(&mut self, chunk: &[u8]) {
        self.pending_bytes.extend_from_slice(chunk);
        if self.pending_bytes.len() > MAX_JOB_BUFFER_BYTES {
            let excess = self.pending_bytes.len() - MAX_JOB_BUFFER_BYTES;
            self.pending_bytes.drain(..excess);
            self.omitted_bytes += excess;
        }
    }

    fn take_delta(&mut self) -> String {
        let mut s = String::new();
        if self.omitted_bytes > 0 {
            s.push_str(&format!(
                "[... {} older bytes omitted after 8 MiB ring cap ...]\n",
                self.omitted_bytes
            ));
            self.omitted_bytes = 0;
        }
        if !self.pending_bytes.is_empty() {
            let bytes = std::mem::take(&mut self.pending_bytes);
            s.push_str(&String::from_utf8_lossy(&bytes));
        }
        s
    }

    fn take_all(&mut self) -> String {
        self.take_delta()
    }
}

#[derive(Debug)]
struct BackgroundJob {
    command: String,
    started: Instant,
    output: Mutex<BoundedOutput>,
    exit_rx: watch::Receiver<Option<i32>>,
    cancel: CancellationToken,
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
        let mut c = self.counter.lock().expect("lock");
        *c += 1;
        let id = format!("job-{}", *c);
        let mut map = self.jobs.lock().expect("lock");
        // Evict oldest finished jobs if cap exceeded (B6)
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
        self.jobs.lock().expect("lock").get(id).cloned()
    }

    fn remove(&self, id: &str) {
        self.jobs.lock().expect("lock").remove(id);
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

/// Kills the child's entire process group (`process_group(0)` => pgid == pid) without `unsafe`.
async fn kill_process_group(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        let pgid = format!("-{pid}");
        let _ = tokio::process::Command::new("/bin/kill")
            .args(["-TERM", &pgid])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await;
        tokio::time::sleep(Duration::from_millis(80)).await;
        let _ = tokio::process::Command::new("/bin/kill")
            .args(["-KILL", &pgid])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await;
    }
    let _ = child.kill().await;
    let _ = child.wait().await;
}

#[derive(Debug)]
pub struct BashTool {
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
}

impl BashTool {
    #[allow(clippy::too_many_arguments)]
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
            cas_enabled,
            spill_threshold,
            preview_max,
            yield_ms,
            max_timeout_ms,
            extra_env,
        }
    }

    fn format_output(
        &self,
        command: &str,
        raw: &str,
    ) -> Result<(String, Option<String>, usize, Option<&'static str>)> {
        let shaped = shape_command_output(
            Some(command),
            raw,
            self.cas_enabled.then_some(self.cas.as_ref()),
            self.spill_threshold,
            self.preview_max,
        )?;
        let saved = shaped.raw_bytes.saturating_sub(shaped.shaped_bytes);
        let handle_path = shaped.spilled.map(|h| h.rel_path);
        Ok((shaped.text, handle_path, saved, shaped.compressor))
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
        Box::pin(async move {
            // Mode A: Collect or kill an existing yielded background job
            if let Some(job_id) = input.get("job").and_then(Value::as_str) {
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
                    .map(|s| s.saturating_mul(1000))
                    .unwrap_or(self.yield_ms)
                    .min(120_000);

                let mut rx = job.exit_rx.clone();
                if rx.borrow().is_none() {
                    let _ = tokio::time::timeout(
                        Duration::from_millis(wait_ms),
                        rx.wait_for(|v| v.is_some()),
                    )
                    .await;
                }
                let exit = *rx.borrow();
                let delta = job.output.lock().expect("lock").take_delta();
                let (mut shaped, handle, bytes_saved, compressor) =
                    self.format_output(&job.command, &delta)?;
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
                    is_error: exit.map(|c| c != 0).unwrap_or(false),
                    details: json!({
                        "job_id": job_id,
                        "exit_code": exit,
                        "handle": handle,
                        "command": job.command,
                        "bytes_saved": bytes_saved,
                        "compressor": compressor,
                    }),
                });
            }

            // Mode B: Launch a new command with adaptive yield
            let Some(command) = input.get("command").and_then(Value::as_str) else {
                return Ok(ToolOutput::err_text("Bash requires `command` or `job`"));
            };
            let explicit_timeout_ms = input
                .get("timeout")
                .and_then(Value::as_u64)
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
                started: Instant::now(),
                output: Mutex::new(BoundedOutput::default()),
                exit_rx: exit_rx.clone(),
                cancel: job_cancel.clone(),
            });

            // Background pump: live chunked readers + process-group kill + 500ms pipe drain grace!
            let job_bg = job.clone();
            tokio::spawn(async move {
                let job_out = job_bg.clone();
                let out_task = tokio::spawn(async move {
                    if let Some(mut s) = stdout {
                        let mut buf = [0u8; 8192];
                        loop {
                            match s.read(&mut buf).await {
                                Ok(0) | Err(_) => break,
                                Ok(n) => {
                                    job_out.output.lock().expect("lock").push_bytes(&buf[..n]);
                                }
                            }
                        }
                    }
                });

                let job_err = job_bg.clone();
                let err_task = tokio::spawn(async move {
                    if let Some(mut s) = stderr {
                        let mut buf = [0u8; 8192];
                        loop {
                            match s.read(&mut buf).await {
                                Ok(0) | Err(_) => break,
                                Ok(n) => {
                                    job_err.output.lock().expect("lock").push_bytes(&buf[..n]);
                                }
                            }
                        }
                    }
                });

                let exit = tokio::select! {
                    _ = job_cancel.cancelled() => {
                        kill_process_group(&mut child).await;
                        137
                    }
                    _ = tokio::time::sleep(Duration::from_millis(hard_timeout_ms)) => {
                        kill_process_group(&mut child).await;
                        124
                    }
                    res = child.wait() => {
                        res.map(status_to_exit_code).unwrap_or(-1)
                    }
                };

                // Drain stdout/stderr up to 500 ms so backgrounded grandchildren never hang Bash (B3)!
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

                let _ = exit_tx.send(Some(exit));
            });

            // Wait up to `yield_ms` via watch::Receiver (zero lost wakeup — B1!)
            let mut wait_rx = exit_rx;
            let waited_done = tokio::select! {
                _ = cancel.cancelled() => {
                    job.cancel.cancel();
                    return Err(CarnotError::Cancelled);
                }
                res = wait_rx.wait_for(|v| v.is_some()) => res.is_ok(),
                _ = tokio::time::sleep(Duration::from_millis(yield_ms)) => false,
            };
            let completed = waited_done || wait_rx.borrow().is_some();

            if completed {
                let exit = wait_rx.borrow().unwrap_or(-1);
                let raw = job.output.lock().expect("lock").take_all();
                let (mut shaped, handle, bytes_saved, compressor) =
                    self.format_output(command, &raw)?;
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
                        "bytes_saved": bytes_saved,
                        "compressor": compressor,
                        "yielded": false,
                    }),
                })
            } else {
                // Adaptive Yield: register background job + return any partial output already produced!
                let partial = job.output.lock().expect("lock").take_delta();
                let job_id = self.jobs.insert(job.clone());
                let header = if partial.is_empty() {
                    String::new()
                } else {
                    let (p_shaped, _, _, _) = self.format_output(command, &partial)?;
                    format!("{p_shaped}\n")
                };
                Ok(ToolOutput::ok_text(format!(
                    "{header}[yielded after {}ms -> {job_id} still running: `{command}` — call Bash{{\"job\":\"{job_id}\"}} to collect or {{\"job\":\"{job_id}\",\"kill\":true}} to stop]",
                    yield_ms
                ))
                .with_details(json!({
                    "command": command,
                    "job_id": job_id,
                    "yielded": true,
                })))
            }
        })
    }
}
