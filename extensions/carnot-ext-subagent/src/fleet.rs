//! Stateful in-process multi-agent `Fleet` registry for long-running orchestration.
//!
//! Design properties:
//!   - **Zero polling / 0% idle CPU**: shared `tokio::sync::Notify` + monotonic `epoch` wakes
//!     `wait(any|all)` and `on_turn_boundary` in <1 ms on worker completion, failure, or incoming
//!     `SendMessage`.
//!   - **Multi-turn persistent workers (`keep_alive: true`)**: keeps live `Arc<Agent>` + isolated
//!     worktree alive between turns so follow-up `SendMessage` hits warm prompt caches & same repo state.
//!   - **Bounded memory for multi-hour runs**: caps archived closed workers (`MAX_RETAINED_WORKERS`)
//!     and coordinator inbox (`MAX_INBOX_MESSAGES`); terminates event observers cleanly on close.
//!   - **Permission safety**: `DelegatingPermissionBroker` forwards worker tool approvals to the
//!     parent's `PermissionBroker` with the original `tool_name` preserved (so read-only auto-allow,
//!     `CallDeferredTool` unwrapping, and `allow_always` work identically) and `"_carnot_worker"`
//!     attached to the cloned input for UI attribution.

use crate::budget::FleetBudget;
use carnot_agent::Agent;
use carnot_cas::CasStore;
use carnot_core::{AgentEvent, CarnotError, Result};
use carnot_ext::{
    BoxFuture as ExtFuture, PermissionBroker, ToolGateDecision,
    warm_fork::{IsolatedForkWorktree, LandOutcome},
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};
use tokio::sync::{Notify, watch};
use tokio_util::sync::CancellationToken;

pub const MAX_SUBAGENT_SUMMARY_CHARS: usize = 6_000;
const MAX_RETAINED_WORKERS: usize = 32;
const MAX_INBOX_MESSAGES: usize = 64;
const DEFAULT_MAX_CONCURRENT_WORKERS: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerState {
    Running,
    Idle,
    Failed(String),
    Closed,
}

impl WorkerState {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Idle => "idle",
            Self::Failed(_) => "failed",
            Self::Closed => "closed",
        }
    }

    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running)
    }
}

#[derive(Debug, Clone)]
pub struct WorkerReport {
    pub worker_name: String,
    pub report: String,
    pub is_error: bool,
    pub halted_reason: Option<String>,
    pub provider: String,
    pub model: String,
    pub forked: bool,
    pub keep_alive: bool,
    pub turns: u64,
    pub cost_usd: f64,
    pub isolated: bool,
    pub diff_handle: Option<String>,
}

#[derive(Debug, Clone)]
pub struct InboxMessage {
    pub from: String,
    pub message: String,
}

/// Permission broker attached to child workers that delegates to the parent's broker while
/// preserving the original `tool_name` verbatim and annotating cloned `input` with
/// `"_carnot_worker": "<worker_name>"` for UI display.
#[derive(Debug)]
pub struct DelegatingPermissionBroker {
    worker_name: String,
    parent_broker: Arc<dyn PermissionBroker>,
}

impl DelegatingPermissionBroker {
    pub fn new(worker_name: String, parent_broker: Arc<dyn PermissionBroker>) -> Self {
        Self {
            worker_name,
            parent_broker,
        }
    }
}

impl PermissionBroker for DelegatingPermissionBroker {
    fn request_permission(
        &self,
        call_id: &str,
        tool_name: &str,
        input: &Value,
    ) -> ExtFuture<'_, Result<ToolGateDecision>> {
        let tool_name = tool_name.to_string();
        let call_id = call_id.to_string();
        let mut tagged_input = input.clone();
        if let Some(obj) = tagged_input.as_object_mut() {
            obj.insert(
                "_carnot_worker".to_string(),
                Value::String(self.worker_name.clone()),
            );
        }
        Box::pin(async move {
            self.parent_broker
                .request_permission(&call_id, &tool_name, &tagged_input)
                .await
        })
    }
}

#[derive(Debug)]
pub struct WorkerHandle {
    pub name: String,
    pub task_preview: String,
    pub provider_id: String,
    pub model: String,
    pub forked: bool,
    pub keep_alive: bool,
    pub isolated: bool,
    pub started: Instant,
    pub agent: Arc<Agent>,
    pub parent_agent: Option<Weak<Agent>>,
    pub parent_workspace: PathBuf,
    pub worktree: Mutex<Option<IsolatedForkWorktree>>,
    pub turns_completed: Arc<AtomicU64>,
    pub tools_called: Arc<AtomicU64>,
    pub cost_micros: Arc<AtomicU64>,
    pub last_activity: Arc<Mutex<String>>,
    pub cancel: CancellationToken,
    pub state_lock: Mutex<()>,
    pub state_tx: watch::Sender<WorkerState>,
    pub state_rx: watch::Receiver<WorkerState>,
    pub last_report: Mutex<Option<WorkerReport>>,
    /// True once a turn finished and its report has NOT yet been delivered/collected by coordinator.
    pub report_unread: AtomicBool,
    /// Serializes sequential turns on this worker (`send` while idle vs initial task).
    pub run_lock: tokio::sync::Mutex<()>,
}

impl WorkerHandle {
    pub fn state(&self) -> WorkerState {
        self.state_rx.borrow().clone()
    }

    /// Atomically transitions `Idle | Failed(_)` -> `Running` under `state_lock`.
    /// Returns `true` if this caller claimed the transition and should spawn the turn.
    pub fn try_begin_turn(&self) -> bool {
        let _g = self.state_lock.lock().unwrap_or_else(|e| e.into_inner());
        if self.cancel.is_cancelled() {
            return false;
        }
        match self.state() {
            WorkerState::Idle | WorkerState::Failed(_) => {
                let _ = self.state_tx.send(WorkerState::Running);
                true
            }
            WorkerState::Running | WorkerState::Closed => false,
        }
    }

    /// Updates worker state under `state_lock`. `WorkerState::Closed` is strictly terminal and
    /// can never be overwritten back to `Running`, `Idle`, or `Failed`.
    /// Returns `true` if the new state was applied, `false` if the worker was already `Closed`.
    pub fn set_state(&self, next: WorkerState) -> bool {
        let _g = self.state_lock.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(self.state(), WorkerState::Closed) && !matches!(next, WorkerState::Closed) {
            return false;
        }
        let _ = self.state_tx.send(next);
        true
    }

    pub fn cost_usd(&self) -> f64 {
        self.cost_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Emits a lightweight `AgentEvent::WorkerUpdated` onto the parent coordinator's `EventLog`
    /// so TUI, ACP, HTTP SSE, and RPC clients observe live worker state with zero polling.
    pub fn emit_status_event(&self) {
        let Some(parent) = self.parent_agent.as_ref().and_then(Weak::upgrade) else {
            return;
        };
        let activity = self
            .last_activity
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "starting".into());
        parent.emit_event(AgentEvent::WorkerUpdated {
            worker: self.name.clone(),
            state: self.state().label().to_string(),
            model: self.model.clone(),
            turns: self.turns_completed.load(Ordering::Relaxed),
            tools_called: self.tools_called.load(Ordering::Relaxed),
            cost_usd: self.cost_usd(),
            activity,
        });
    }

    pub fn has_worktree(&self) -> bool {
        self.worktree.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    pub fn progress_line(&self) -> String {
        let elapsed = self.started.elapsed().as_secs_f64();
        let turns = self.turns_completed.load(Ordering::Relaxed);
        let tools = self.tools_called.load(Ordering::Relaxed);
        let cost = self.cost_usd();
        let last = self
            .last_activity
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "starting".into());
        let st = self.state();
        let state_str = match st {
            WorkerState::Running => "running",
            WorkerState::Idle if self.keep_alive => "idle (keep_alive)",
            WorkerState::Idle => "finished (ready to collect)",
            WorkerState::Failed(_) => "failed",
            WorkerState::Closed => "closed",
        };
        let mut flags = String::new();
        if self.forked {
            flags.push_str(" | fork");
        }
        if self.isolated && self.has_worktree() {
            flags.push_str(" | worktree");
        }
        format!(
            "{} [{} | {:.1}s | {}/{}{} | turns={} tools={} cost=${:.4} | last={}] — {}",
            self.name,
            state_str,
            elapsed,
            self.provider_id,
            self.model,
            flags,
            turns,
            tools,
            cost,
            last,
            self.task_preview
        )
    }

    /// Exports strictly this worker's changes (`base_sha..worker`) into parent's CAS store.
    pub fn export_diff_to_cas(&self) -> Option<String> {
        let diff_bytes = {
            let guard = self.worktree.lock().ok()?;
            let wt = guard.as_ref()?;
            wt.export_worker_diff().ok()??
        };
        let cas = CasStore::open(&self.parent_workspace).ok()?;
        let text = String::from_utf8_lossy(&diff_bytes);
        let handle = cas.put_text(&text).ok()?;
        Some(handle.rel_path)
    }

    /// Atomically checks & lands this worker's worktree changes onto the parent repository,
    /// advancing the worktree baseline commit on success so future lands are incremental.
    pub fn land_worktree(&self) -> Result<LandOutcome> {
        let mut guard = self.worktree.lock().unwrap_or_else(|e| e.into_inner());
        let Some(wt) = guard.as_mut() else {
            return Err(CarnotError::Tool {
                tool: "Subagent".into(),
                message: format!(
                    "worker `{}` has no active isolated worktree to land",
                    self.name
                ),
            });
        };
        wt.land_into_parent()
    }

    pub fn changed_files_stat(&self) -> Option<String> {
        let guard = self.worktree.lock().ok()?;
        guard.as_ref()?.changed_files_summary()
    }

    pub fn drop_worktree(&self) {
        if let Ok(mut guard) = self.worktree.lock() {
            let _ = guard.take();
        }
    }

    /// Drops the worktree on tokio's blocking pool when inside an async runtime (so removing a
    /// large `CARGO_TARGET_DIR` never stalls tokio worker threads).
    pub fn drop_worktree_bg(&self) {
        let wt = self.worktree.lock().ok().and_then(|mut g| g.take());
        if let Some(wt) = wt {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn_blocking(move || drop(wt));
            } else {
                drop(wt);
            }
        }
    }

    /// Drops the worktree immediately (offloaded to blocking pool if async) only if no turn
    /// currently holds `run_lock`; otherwise the running worker task drops it deterministically
    /// as soon as `prompt_with_cancel()` unwinds.
    pub fn drop_worktree_if_idle(&self, blocking_sync: bool) {
        if let Ok(_guard) = self.run_lock.try_lock() {
            if blocking_sync {
                self.drop_worktree();
            } else {
                self.drop_worktree_bg();
            }
        }
    }
}

/// Shared fleet registry owned by the coordinator's `SubagentModule`.
#[derive(Debug)]
pub struct Fleet {
    workers: Mutex<BTreeMap<String, Arc<WorkerHandle>>>,
    counter: AtomicU64,
    epoch: AtomicU64,
    coordinator_inbox: Mutex<VecDeque<InboxMessage>>,
    pub notify: Arc<Notify>,
    pub budget: Arc<FleetBudget>,
    pub max_concurrent: usize,
}

impl Fleet {
    pub fn new(fleet_max_cost_usd: Option<f64>) -> Self {
        let max_concurrent = std::env::var("CARNOT_MAX_WORKERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|n| n.clamp(1, 16))
            .unwrap_or(DEFAULT_MAX_CONCURRENT_WORKERS);
        Self {
            workers: Mutex::new(BTreeMap::new()),
            counter: AtomicU64::new(0),
            epoch: AtomicU64::new(0),
            coordinator_inbox: Mutex::new(VecDeque::new()),
            notify: Arc::new(Notify::new()),
            budget: Arc::new(FleetBudget::new(fleet_max_cost_usd)),
            max_concurrent,
        }
    }

    pub fn bump_epoch_and_notify(&self) {
        self.epoch.fetch_add(1, Ordering::Release);
        self.notify.notify_waiters();
    }

    pub fn next_auto_name(&self) -> String {
        let n = self.counter.fetch_add(1, Ordering::Relaxed) + 1;
        format!("sub-{n}")
    }

    pub fn running_count(&self) -> usize {
        self.workers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter(|w| w.state().is_running())
            .count()
    }

    pub fn has_running_workers(&self) -> bool {
        self.running_count() > 0
    }

    pub fn insert(&self, worker: Arc<WorkerHandle>) -> Result<()> {
        let mut map = self.workers.lock().unwrap_or_else(|e| e.into_inner());
        let active_running = map.values().filter(|w| w.state().is_running()).count();
        if active_running >= self.max_concurrent {
            return Err(CarnotError::Tool {
                tool: "Subagent".into(),
                message: format!(
                    "max concurrent workers ({}) reached; wait for or close an existing worker first",
                    self.max_concurrent
                ),
            });
        }
        if let Some(existing) = map.get(&worker.name)
            && !matches!(existing.state(), WorkerState::Closed)
        {
            return Err(CarnotError::Tool {
                tool: "Subagent".into(),
                message: format!(
                    "worker `{}` already exists (state={}); use `SendMessage` to message it or `Subagent{{\"job\":\"{}\",\"kill\":true}}` to close it first",
                    worker.name,
                    existing.state().label(),
                    worker.name
                ),
            });
        }

        // Bound retained archive for very long runs (multi-hour / dozens of waves)
        if map.len() >= MAX_RETAINED_WORKERS {
            let evictable: Vec<String> = map
                .iter()
                .filter(|(_, w)| {
                    matches!(w.state(), WorkerState::Closed)
                        || (!w.keep_alive
                            && !w.state().is_running()
                            && !w.report_unread.load(Ordering::Relaxed))
                })
                .map(|(k, _)| k.clone())
                .collect();
            for k in evictable {
                map.remove(&k);
                if map.len() < MAX_RETAINED_WORKERS {
                    break;
                }
            }
        }
        map.insert(worker.name.clone(), worker);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Arc<WorkerHandle>> {
        self.workers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .cloned()
    }

    pub fn remove(&self, name: &str) -> Option<Arc<WorkerHandle>> {
        self.workers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(name)
    }

    /// Atomically checks `max_concurrent` under the `workers` lock and claims an `Idle | Failed`
    /// worker's transition to `Running`. Returns `Err(max_concurrent)` if the concurrency cap is
    /// already full, `Ok(true)` if claimed, or `Ok(false)` if the worker is already running/closed.
    pub fn try_claim_wake(&self, worker: &WorkerHandle) -> std::result::Result<bool, usize> {
        let map = self.workers.lock().unwrap_or_else(|e| e.into_inner());
        let active_running = map.values().filter(|w| w.state().is_running()).count();
        if active_running >= self.max_concurrent {
            return Err(self.max_concurrent);
        }
        Ok(worker.try_begin_turn())
    }

    pub fn close_worker(&self, name: &str) -> Option<(Arc<WorkerHandle>, f64)> {
        let w = self.remove(name)?;
        let cost = w.cost_usd();
        w.cancel.cancel();
        w.set_state(WorkerState::Closed);
        w.emit_status_event();
        w.drop_worktree_if_idle(false);
        self.bump_epoch_and_notify();
        Some((w, cost))
    }

    pub fn push_coordinator_message(&self, from: String, message: String) {
        {
            let mut q = self
                .coordinator_inbox
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if q.len() >= MAX_INBOX_MESSAGES {
                q.pop_front();
            }
            q.push_back(InboxMessage { from, message });
        }
        self.bump_epoch_and_notify();
    }

    pub fn has_unread_inbox(&self) -> bool {
        !self
            .coordinator_inbox
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    pub fn has_unread_updates(&self) -> bool {
        if self.has_unread_inbox() {
            return true;
        }
        self.workers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .any(|w| w.report_unread.load(Ordering::Acquire))
    }

    /// Drains only pending worker -> coordinator `SendMessage` inbox items without touching
    /// worker reports or evicting workers (safe during active tool turns).
    pub fn drain_coordinator_inbox(&self) -> Vec<InboxMessage> {
        let mut q = self
            .coordinator_inbox
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        q.drain(..).collect()
    }

    /// Drains all pending worker->coordinator messages AND newly completed unread worker reports.
    /// Non-`keep_alive` workers whose reports are drained are cleaned up cleanly so memory stays bounded.
    pub fn drain_coordinator_updates(&self) -> (Vec<InboxMessage>, Vec<WorkerReport>) {
        let inbox = self.drain_coordinator_inbox();

        let mut reports = Vec::new();
        let mut remove_oneshot = Vec::new();
        {
            let map = self.workers.lock().unwrap_or_else(|e| e.into_inner());
            for (name, w) in map.iter() {
                if w.report_unread.swap(false, Ordering::AcqRel)
                    && let Ok(guard) = w.last_report.lock()
                    && let Some(ref rep) = *guard
                {
                    reports.push(rep.clone());
                    if !w.keep_alive && !w.state().is_running() {
                        remove_oneshot.push(name.clone());
                    }
                }
            }
        }
        for name in remove_oneshot {
            if let Some(w) = self.remove(&name) {
                w.drop_worktree_if_idle(false);
            }
        }
        (inbox, reports)
    }

    /// Single-call fleet dashboard overview (workers, states, turns, individual & total cost).
    /// Drops the `workers` lock before inspecting each worker handle so locks never nest.
    pub fn list_overview(&self) -> (Vec<String>, f64, usize) {
        let snapshot: Vec<Arc<WorkerHandle>> = {
            let map = self.workers.lock().unwrap_or_else(|e| e.into_inner());
            map.values().cloned().collect()
        };
        let rows: Vec<String> = snapshot.iter().map(|w| w.progress_line()).collect();
        let running = snapshot.iter().filter(|w| w.state().is_running()).count();
        (rows, self.budget.spent_usd(), running)
    }

    /// Zero-CPU reactive wait barrier supporting `any` or `all` over specific worker names (or all workers).
    ///
    /// In `any` mode, already-idle `keep_alive` workers whose reports were previously read do NOT
    /// cause a zero-delay spin: `wait_for` blocks until at least one currently-running target
    /// transitions out of `Running`, or an unread report / coordinator inbox message arrives.
    pub async fn wait_for(
        &self,
        targets: &[String],
        wait_all: bool,
        timeout: Duration,
        cancel: &CancellationToken,
    ) -> bool {
        let initially_running = self.snapshot_running_targets(targets);
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            // Register notified future BEFORE checking condition to guarantee ZERO missed wakeups!
            let notified = self.notify.notified();
            if self.wait_condition_met(targets, wait_all, &initially_running) {
                return true;
            }
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return false;
            }
            tokio::select! {
                _ = cancel.cancelled() => return false,
                _ = notified => {}
                _ = tokio::time::sleep(remaining) => {
                    return self.wait_condition_met(targets, wait_all, &initially_running);
                }
            }
        }
    }

    fn snapshot_running_targets(&self, targets: &[String]) -> Vec<String> {
        let map = self.workers.lock().unwrap_or_else(|e| e.into_inner());
        let iter: Vec<&Arc<WorkerHandle>> = if targets.is_empty() {
            map.values().collect()
        } else {
            targets.iter().filter_map(|n| map.get(n)).collect()
        };
        iter.into_iter()
            .filter(|w| w.state().is_running())
            .map(|w| w.name.clone())
            .collect()
    }

    fn wait_condition_met(
        &self,
        targets: &[String],
        wait_all: bool,
        initially_running: &[String],
    ) -> bool {
        let has_inbox = self.has_unread_inbox();
        let map = self.workers.lock().unwrap_or_else(|e| e.into_inner());
        let selected: Vec<&Arc<WorkerHandle>> = if targets.is_empty() {
            map.values().collect()
        } else {
            targets.iter().filter_map(|n| map.get(n)).collect()
        };
        if selected.is_empty() {
            return true;
        }
        if wait_all {
            selected.iter().all(|w| !w.state().is_running())
        } else {
            if has_inbox
                || selected
                    .iter()
                    .any(|w| w.report_unread.load(Ordering::Acquire))
            {
                return true;
            }
            if initially_running.is_empty() {
                // None of the targeted workers were running when `wait_for` began -> don't hang.
                return true;
            }
            // Wake as soon as any worker that was running at wait start settles, fails, or closes.
            initially_running
                .iter()
                .any(|name| map.get(name).is_none_or(|w| !w.state().is_running()))
        }
    }

    pub fn kill_all(&self) {
        self.kill_all_inner(false);
    }

    fn kill_all_inner(&self, blocking_sync: bool) {
        let all: Vec<Arc<WorkerHandle>> = {
            let mut map = self.workers.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *map).into_values().collect()
        };
        for w in all {
            w.cancel.cancel();
            w.set_state(WorkerState::Closed);
            w.emit_status_event();
            w.drop_worktree_if_idle(blocking_sync);
        }
        self.bump_epoch_and_notify();
    }
}

impl Drop for Fleet {
    fn drop(&mut self) {
        // Synchronous cleanup on drop guarantees worktrees are removed before process exit / drop return.
        self.kill_all_inner(true);
    }
}

/// Spawns a turn execution loop on `worker`, updating live metrics, rolling up cost into
/// `fleet.budget` + parent `Agent`, recording the finished `WorkerReport`, and notifying waiters.
///
/// Holds `Weak<Fleet>` (never `Arc<Fleet>`) so a background task can never extend `Fleet` past
/// coordinator drop.
pub fn spawn_worker_turn(
    fleet_weak: Weak<Fleet>,
    budget: Arc<FleetBudget>,
    worker: Arc<WorkerHandle>,
    prompt: String,
    parent_agent: Option<Weak<Agent>>,
) {
    if !worker.set_state(WorkerState::Running) {
        return;
    }
    worker.emit_status_event();
    if let Some(f) = fleet_weak.upgrade() {
        f.notify.notify_waiters();
    }

    tokio::spawn(async move {
        let _turn_guard = worker.run_lock.lock().await;
        if worker.cancel.is_cancelled() {
            worker.set_state(WorkerState::Closed);
            worker.drop_worktree_bg();
            if let Some(f) = fleet_weak.upgrade() {
                f.bump_epoch_and_notify();
            }
            return;
        }

        let mut current_prompt = prompt;
        loop {
            // Subscribe from latest event seq so observer tracks ONLY this turn and exits cleanly!
            let mut ev_stream = worker.agent.subscribe_from(worker.agent.latest_event_seq());
            let tc = Arc::clone(&worker.turns_completed);
            let tools_c = Arc::clone(&worker.tools_called);
            let cm = Arc::clone(&worker.cost_micros);
            let la = Arc::clone(&worker.last_activity);
            let worker_obs = Arc::downgrade(&worker);
            let budget_obs = Arc::clone(&budget);
            let fleet_obs = fleet_weak.clone();
            let parent_obs = parent_agent.clone();
            let obs_task = tokio::spawn(async move {
                loop {
                    let env = ev_stream.next().await;
                    match env.event {
                        AgentEvent::TurnStart { turn, .. } => {
                            tc.fetch_add(1, Ordering::Relaxed);
                            if let Ok(mut g) = la.lock() {
                                *g = format!("turn:{turn}");
                            }
                            if let Some(w) = worker_obs.upgrade() {
                                w.emit_status_event();
                            }
                        }
                        AgentEvent::ToolStart { tool, .. } => {
                            tools_c.fetch_add(1, Ordering::Relaxed);
                            if let Ok(mut g) = la.lock() {
                                *g = format!("tool:{tool}");
                            }
                            if let Some(w) = worker_obs.upgrade() {
                                w.emit_status_event();
                            }
                        }
                        AgentEvent::UsageRecorded { usage, .. } => {
                            let delta = (usage.cost_usd.max(0.0) * 1_000_000.0).round() as u64;
                            cm.fetch_add(delta, Ordering::Relaxed);
                            budget_obs
                                .record_worker_cost(usage.cost_usd, parent_obs.as_ref())
                                .await;
                            if budget_obs.exceeded().is_some()
                                && let Some(f) = fleet_obs.upgrade()
                            {
                                f.kill_all();
                            }
                        }
                        AgentEvent::RunSettled { .. } | AgentEvent::Error { .. } => {
                            break;
                        }
                        _ => {}
                    }
                }
            });

            let ca_abort = Arc::clone(&worker.agent);
            let wc = worker.cancel.clone();
            let cancel_watcher = tokio::spawn(async move {
                wc.cancelled().await;
                ca_abort.abort().await;
            });

            // Bind `worker.cancel` directly into `prompt_with_cancel` so cancellation arriving
            // before `prompt_inner` installs `cancel_current` is never lost.
            let run_res = worker
                .agent
                .prompt_with_cancel(current_prompt, worker.cancel.clone())
                .await;
            cancel_watcher.abort();
            // Give observer a brief window to flush final `UsageRecorded` / `RunSettled`
            let _ = tokio::time::timeout(Duration::from_millis(150), obs_task).await;

            // If worker was killed/cancelled while running, safely drop its worktree now that
            // `prompt_with_cancel()` has exited and leave state as `Closed`.
            if worker.cancel.is_cancelled() {
                worker.drop_worktree_bg();
                worker.set_state(WorkerState::Closed);
                if let Some(f) = fleet_weak.upgrade() {
                    f.bump_epoch_and_notify();
                }
                return;
            }

            // If a follow-up/steer arrived during the post-turn window, process it right away
            // before transitioning to `Idle` so it is never stranded.
            if run_res.is_ok()
                && let Some(next_msg) = worker.agent.take_pending_wakeup().await
            {
                current_prompt = next_msg;
                continue;
            }

            // Run blocking git diff/stat calls on tokio's blocking thread pool.
            let (diff_handle, changed_stat) = if worker.isolated {
                let w_clone = Arc::clone(&worker);
                tokio::task::spawn_blocking(move || {
                    let dh = w_clone.export_diff_to_cas();
                    let st = w_clone.changed_files_stat();
                    (dh, st)
                })
                .await
                .unwrap_or((None, None))
            } else {
                (None, None)
            };

            // For non-keep_alive one-shot workers, clean up worktree right after exporting diff to CAS.
            // For keep_alive workers, keep worktree alive so coordinator can `land` or `SendMessage` fixes.
            if !worker.keep_alive || worker.cancel.is_cancelled() {
                worker.drop_worktree_bg();
            }

            let (report, next_state) = match run_res {
                Ok(outcome) => {
                    let mut summary = outcome.final_text;
                    if summary.chars().count() > MAX_SUBAGENT_SUMMARY_CHARS {
                        summary = summary.chars().take(MAX_SUBAGENT_SUMMARY_CHARS).collect();
                        summary.push_str("\n[... subagent summary truncated ...]");
                    }
                    let halted_tag = outcome
                        .halted_reason
                        .as_deref()
                        .map(|h| format!(" halted=\"{h}\""))
                        .unwrap_or_default();
                    let mut text = format!(
                        "[subagent-report worker={} provider={} model={} turns={} cost=${:.4}{}{}{}{}]\n{}",
                        worker.name,
                        worker.provider_id,
                        worker.model,
                        outcome.metrics.turns,
                        worker.cost_usd(),
                        if worker.forked { " fork=true" } else { "" },
                        if worker.keep_alive {
                            " keep_alive=true"
                        } else {
                            ""
                        },
                        if worker.isolated {
                            " isolated_worktree=true"
                        } else {
                            ""
                        },
                        halted_tag,
                        summary
                    );
                    if let Some(ref hr) = outcome.halted_reason {
                        text.push_str(&format!("\n[subagent-halted] {hr}"));
                    }
                    if let Some(ref st) = changed_stat {
                        text.push_str(&format!("\n[subagent-workspace-changes]\n{st}"));
                    }
                    if let Some(ref dh) = diff_handle {
                        if worker.keep_alive {
                            text.push_str(&format!(
                                "\n[subagent-worktree] Worktree kept alive — call Subagent{{\"job\":\"{}\",\"land\":true}} to apply atomically (patch also saved in `{dh}`)",
                                worker.name
                            ));
                        } else {
                            text.push_str(&format!(
                                "\n[subagent-diff-patch] Saved isolated worktree diff to `{dh}` (apply via `git apply {dh}`)"
                            ));
                        }
                    }
                    let rep = WorkerReport {
                        worker_name: worker.name.clone(),
                        report: text,
                        is_error: false,
                        halted_reason: outcome.halted_reason,
                        provider: worker.provider_id.clone(),
                        model: worker.model.clone(),
                        forked: worker.forked,
                        keep_alive: worker.keep_alive,
                        turns: outcome.metrics.turns,
                        cost_usd: worker.cost_usd(),
                        isolated: worker.isolated,
                        diff_handle,
                    };
                    (rep, WorkerState::Idle)
                }
                Err(e) => {
                    let msg = format!("Subagent `{}` failed: {e}", worker.name);
                    let rep = WorkerReport {
                        worker_name: worker.name.clone(),
                        report: msg.clone(),
                        is_error: true,
                        halted_reason: None,
                        provider: worker.provider_id.clone(),
                        model: worker.model.clone(),
                        forked: worker.forked,
                        keep_alive: worker.keep_alive,
                        turns: worker.turns_completed.load(Ordering::Relaxed),
                        cost_usd: worker.cost_usd(),
                        isolated: worker.isolated,
                        diff_handle,
                    };
                    (rep, WorkerState::Failed(msg))
                }
            };

            if worker.cancel.is_cancelled() || !worker.set_state(next_state.clone()) {
                worker.drop_worktree_bg();
                worker.set_state(WorkerState::Closed);
                worker.emit_status_event();
                if let Some(f) = fleet_weak.upgrade() {
                    f.bump_epoch_and_notify();
                }
                return;
            }
            if let Ok(mut g) = worker.last_report.lock() {
                *g = Some(report);
            }
            worker.report_unread.store(true, Ordering::Release);
            let is_idle = matches!(next_state, WorkerState::Idle);
            if let Ok(mut g) = worker.last_activity.lock() {
                *g = if worker.keep_alive {
                    "idle".into()
                } else {
                    "done".into()
                };
            }
            worker.emit_status_event();
            if let Some(f) = fleet_weak.upgrade() {
                f.bump_epoch_and_notify();
            }

            // Dekker post-state-publish recheck: if a `SendMessage` arrived right as we published
            // `Idle`, claim the turn atomically and process it without ever stranding the message.
            if is_idle
                && worker.keep_alive
                && !worker.cancel.is_cancelled()
                && let Some(late_msg) = worker.agent.take_pending_wakeup().await
            {
                if worker.try_begin_turn() {
                    current_prompt = late_msg;
                    continue;
                } else {
                    // Re-queue if another wake already claimed the turn
                    worker.agent.follow_up(late_msg).await;
                }
            }
            break;
        }
    });
}

pub fn report_to_details(rep: &WorkerReport) -> Value {
    json!({
        "job_id": rep.worker_name,
        "provider": rep.provider,
        "model": rep.model,
        "fork": rep.forked,
        "keep_alive": rep.keep_alive,
        "turns": rep.turns,
        "cost_usd": rep.cost_usd,
        "isolated_worktree": rep.isolated,
        "diff_handle": rep.diff_handle,
        "halted_reason": rep.halted_reason,
        "finished": true,
    })
}
