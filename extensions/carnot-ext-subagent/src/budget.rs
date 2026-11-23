//! Shared atomic fleet USD budget guardrail across coordinator + all active & historical workers.
//!
//! Links directly to the parent coordinator `Agent`'s lock-free `total_cost_micros` counter
//! (which tracks parent turns, compaction calls, and live rolled-up worker spend) so the fleet
//! cap always reflects exact combined parent + worker spend without double counting.

use carnot_agent::Agent;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

#[derive(Debug)]
pub struct FleetBudget {
    /// Worker-only spend in micro-dollars (fallback when no parent counter is attached).
    worker_spent_micros: AtomicU64,
    /// Shared monotonic total cost counter owned by the parent coordinator `Agent`.
    parent_total_micros: OnceLock<Arc<AtomicU64>>,
    /// Optional hard fleet USD limit in micro-dollars (1 USD = 1_000_000 micros). `0` = unlimited.
    limit_micros: u64,
}

impl FleetBudget {
    pub fn new(limit_usd: Option<f64>) -> Self {
        let limit_micros = limit_usd
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| (v * 1_000_000.0).round() as u64)
            .unwrap_or(0);
        Self {
            worker_spent_micros: AtomicU64::new(0),
            parent_total_micros: OnceLock::new(),
            limit_micros,
        }
    }

    /// Attaches the parent coordinator `Agent`'s unified `total_cost_micros` counter so
    /// `spent_usd()` and `exceeded()` always inspect combined parent + worker spend.
    pub fn attach_parent_counter(&self, counter: Arc<AtomicU64>) {
        let _ = self.parent_total_micros.set(counter);
    }

    pub fn limit_usd(&self) -> Option<f64> {
        (self.limit_micros > 0).then(|| self.limit_micros as f64 / 1_000_000.0)
    }

    fn effective_spent_micros(&self) -> u64 {
        if let Some(parent_counter) = self.parent_total_micros.get() {
            parent_counter.load(Ordering::Relaxed)
        } else {
            self.worker_spent_micros.load(Ordering::Relaxed)
        }
    }

    pub fn spent_usd(&self) -> f64 {
        self.effective_spent_micros() as f64 / 1_000_000.0
    }

    /// Atomically records a worker's cost delta and rolls it up into the parent `Agent`'s metrics
    /// and shared `total_cost_micros` counter.
    pub async fn record_worker_cost(&self, delta_usd: f64, parent: Option<&Weak<Agent>>) {
        if !delta_usd.is_finite() || delta_usd <= 0.0 {
            return;
        }
        let delta_micros = (delta_usd * 1_000_000.0).round() as u64;
        if delta_micros > 0 {
            self.worker_spent_micros
                .fetch_add(delta_micros, Ordering::Relaxed);
        }
        if let Some(p) = parent.and_then(Weak::upgrade) {
            p.record_external_cost(delta_usd).await;
        }
    }

    /// Returns `Some((spent_usd, limit_usd))` if the shared parent + workers fleet cap is reached.
    pub fn exceeded(&self) -> Option<(f64, f64)> {
        if self.limit_micros == 0 {
            return None;
        }
        let spent = self.effective_spent_micros();
        if spent >= self.limit_micros {
            Some((
                spent as f64 / 1_000_000.0,
                self.limit_micros as f64 / 1_000_000.0,
            ))
        } else {
            None
        }
    }
}
