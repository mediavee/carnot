//! Lossless monotone-`seq` ring-buffered `EventLog` with explicit ring-gap signaling.
//!
//! Replaces `tokio::sync::broadcast(512)` which silently disconnected slow SSE/ACP/RPC
//! clients on `RecvError::Lagged`. Subscribers read from a 4,096-event ring buffer with
//! `subscribe_from(after_seq)`. If `after_seq > 0` fell behind the oldest retained event,
//! emits a synthetic `AgentEvent::EventsTruncated { dropped_through_seq }` so clients resync!

use carnot_core::{AgentEvent, EventEnvelope};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

const RING_CAPACITY: usize = 4_096;

#[derive(Debug)]
struct EventLogState {
    next_seq: u64,
    ring: VecDeque<EventEnvelope>,
}

/// Thread-safe lossless event log with monotone sequence numbers and explicit gap signaling.
#[derive(Debug)]
pub struct EventLog {
    state: Mutex<EventLogState>,
    notify: Notify,
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new()
    }
}

impl EventLog {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(EventLogState {
                next_seq: 1,
                ring: VecDeque::with_capacity(256),
            }),
            notify: Notify::new(),
        }
    }

    pub fn push(&self, event: AgentEvent) -> EventEnvelope {
        let env = {
            let mut st = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let seq = st.next_seq;
            st.next_seq += 1;
            let env = EventEnvelope { seq, event };
            if st.ring.len() >= RING_CAPACITY {
                st.ring.pop_front();
            }
            st.ring.push_back(env.clone());
            env
        };
        self.notify.notify_waiters();
        env
    }

    pub fn latest_seq(&self) -> u64 {
        let st = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        st.next_seq.saturating_sub(1)
    }

    /// Subscribes starting from future events only (equivalent to `subscribe_from(latest_seq())`).
    pub fn subscribe(self: &Arc<Self>) -> EventStream {
        EventStream {
            log: Arc::clone(self),
            cursor: self.latest_seq(),
        }
    }

    /// Subscribes replaying all retained events with `seq > after_seq`, then streaming live events.
    pub fn subscribe_from(self: &Arc<Self>, after_seq: u64) -> EventStream {
        EventStream {
            log: Arc::clone(self),
            cursor: after_seq,
        }
    }

    /// Snapshot of all retained events with `seq > after_seq` (O(1) indexing + `EventsTruncated` on wrap).
    pub fn replay_after(&self, after_seq: u64) -> Vec<EventEnvelope> {
        let st = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut out = Vec::new();
        let Some(oldest) = st.ring.front() else {
            return out;
        };
        if oldest.seq > 1 && after_seq + 1 < oldest.seq {
            out.push(EventEnvelope {
                seq: oldest.seq - 1,
                event: AgentEvent::EventsTruncated {
                    dropped_through_seq: oldest.seq - 1,
                },
            });
        }
        let start = if after_seq < oldest.seq {
            0
        } else {
            (after_seq + 1 - oldest.seq) as usize
        };
        if start < st.ring.len() {
            out.extend(st.ring.range(start..).cloned());
        }
        out
    }
}

/// Async cursor over `EventLog` — never disconnects on `Lagged` and explicitly signals ring gaps.
#[derive(Debug)]
pub struct EventStream {
    log: Arc<EventLog>,
    cursor: u64,
}

impl EventStream {
    pub fn cursor(&self) -> u64 {
        self.cursor
    }

    /// Returns the next `EventEnvelope` with `seq > cursor` in O(1), waiting cleanly without lost wakeups.
    pub async fn next(&mut self) -> EventEnvelope {
        loop {
            let notified = self.log.notify.notified();
            tokio::pin!(notified);
            // CRITICAL: register in Tokio's Notify waiter queue BEFORE locking `state` so a
            // concurrent `notify_waiters()` between unlock and `.await` can NEVER be lost!
            notified.as_mut().enable();
            {
                let st = self
                    .log
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(oldest) = st.ring.front() {
                    // Emit a gap signal even when `cursor == 0` if seq 1 has already been evicted.
                    if oldest.seq > 1 && self.cursor + 1 < oldest.seq {
                        let dropped = oldest.seq - 1;
                        self.cursor = dropped;
                        return EventEnvelope {
                            seq: dropped,
                            event: AgentEvent::EventsTruncated {
                                dropped_through_seq: dropped,
                            },
                        };
                    }
                    let idx = (self.cursor + 1).saturating_sub(oldest.seq) as usize;
                    if let Some(ev) = st.ring.get(idx) {
                        self.cursor = ev.seq;
                        return ev.clone();
                    }
                }
            }
            notified.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn replays_from_after_seq_and_streams_live_without_dropping() {
        let log = Arc::new(EventLog::new());
        let e1 = log.push(AgentEvent::TextDelta { delta: "a".into() });
        let e2 = log.push(AgentEvent::TextDelta { delta: "b".into() });
        assert_eq!(e1.seq, 1);
        assert_eq!(e2.seq, 2);

        let mut stream = log.subscribe_from(1);
        let got2 = stream.next().await;
        assert_eq!(got2.seq, 2);

        let log2 = Arc::clone(&log);
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            log2.push(AgentEvent::TextDelta { delta: "c".into() });
        });
        let got3 = stream.next().await;
        assert_eq!(got3.seq, 3);
    }

    #[tokio::test]
    async fn emits_events_truncated_when_ring_wraps_past_cursor() {
        let log = Arc::new(EventLog::new());
        for i in 0..(RING_CAPACITY + 10) {
            log.push(AgentEvent::TextDelta {
                delta: format!("{i}"),
            });
        }
        let mut stream = log.subscribe_from(2);
        let first = stream.next().await;
        assert!(matches!(
            first.event,
            AgentEvent::EventsTruncated {
                dropped_through_seq: 10
            }
        ));
        let next = stream.next().await;
        assert_eq!(next.seq, 11);

        // Even `after_seq == 0` signals truncation once seq 1 has been evicted!
        let mut from_zero = log.subscribe_from(0);
        let z_first = from_zero.next().await;
        assert!(matches!(
            z_first.event,
            AgentEvent::EventsTruncated {
                dropped_through_seq: 10
            }
        ));
    }
}
