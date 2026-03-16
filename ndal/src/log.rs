//! DNOS Non-Determinism Abstraction Layer — Replay Log
//!
//! The replay log is the core data structure of the NDAL. Every oracle
//! response is appended here, hash-chained to its predecessor. This
//! makes any execution replayable: feed the same log → get the same state.
//!
//! Properties:
//! - Append-only: entries are never modified
//! - Hash-chained: tamper-evident
//! - Content-addressed: large responses stored by hash
//! - Prunable: entries before a snapshot can be archived
//!
//! Modes:
//! - Live: oracles query the real universe, responses are logged
//! - Replay: oracles read from log, return recorded responses
//! - Verify: query both universe and log, flag divergence

use crate::types::*;

// ─────────────────────────────────────────────────────────────────────────────
// Log Mode
// ─────────────────────────────────────────────────────────────────────────────

/// Operating mode of the replay log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogMode {
    /// Live: oracle responses come from the real universe and are logged.
    Live,
    /// Replay: oracle responses come from the log. Queries are verified
    /// against stored queries but the response is always the logged one.
    Replay,
    /// Verify: oracle responses come from the real universe AND the log.
    /// If they differ, a divergence is recorded. Useful for detecting
    /// non-reproducible behavior.
    Verify,
}

// ─────────────────────────────────────────────────────────────────────────────
// Divergence Record
// ─────────────────────────────────────────────────────────────────────────────

/// Records a point where live and replay responses differed.
#[derive(Debug, Clone)]
pub struct Divergence {
    /// Sequence number of the divergent entry
    pub sequence: u64,
    /// Epoch when divergence occurred
    pub epoch: Epoch,
    /// Which oracle diverged
    pub oracle: OracleId,
    /// The logged (expected) response
    pub expected: OracleResponse,
    /// The live (actual) response
    pub actual: OracleResponse,
}

// ─────────────────────────────────────────────────────────────────────────────
// Snapshot
// ─────────────────────────────────────────────────────────────────────────────

/// A snapshot captures the complete system state at a given epoch,
/// allowing the log to be pruned up to that point.
///
/// New nodes can sync from: snapshot + log entries after snapshot epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    /// Epoch at which this snapshot was taken
    pub epoch: Epoch,
    /// Sequence number of the last log entry incorporated
    pub last_sequence: u64,
    /// Hash chain value at the snapshot point
    pub chain_hash: [u8; 8],
    /// Hash of the neural substrate's complete weight state
    pub weights_hash: [u8; 32],
    /// Hash of the neural substrate's activation/memory state
    pub state_hash: [u8; 32],
    /// IAL stream digest at this point
    pub ial_digest: (u64, u64),
}

// ─────────────────────────────────────────────────────────────────────────────
// Replay Log
// ─────────────────────────────────────────────────────────────────────────────

/// The replay log. Append-only, hash-chained storage of all oracle responses.
pub struct ReplayLog {
    /// The log entries (in-memory for now; production would use mmap or disk)
    entries: Vec<LogEntry>,
    /// Current mode
    mode: LogMode,
    /// Head of the hash chain (hash of most recent entry)
    chain_head: [u8; 8],
    /// Next sequence number
    next_sequence: u64,
    /// Replay cursor (for Replay and Verify modes)
    replay_cursor: usize,
    /// Divergences detected in Verify mode
    divergences: Vec<Divergence>,
    /// Snapshots taken
    snapshots: Vec<Snapshot>,
    /// Configuration
    config: NdalConfig,
    /// Total bytes of response data logged (for monitoring)
    total_data_bytes: u64,
}

impl ReplayLog {
    /// Create a new empty log in Live mode.
    pub fn new(config: NdalConfig) -> Self {
        ReplayLog {
            entries: Vec::new(),
            mode: LogMode::Live,
            chain_head: [0u8; 8], // Genesis hash
            next_sequence: 0,
            replay_cursor: 0,
            divergences: Vec::new(),
            snapshots: Vec::new(),
            config,
            total_data_bytes: 0,
        }
    }

    /// Create a log preloaded with entries for replay.
    pub fn from_entries(entries: Vec<LogEntry>, config: NdalConfig) -> Option<Self> {
        // Verify the hash chain
        let mut prev_hash = [0u8; 8];
        for entry in &entries {
            if !entry.verify(&prev_hash) {
                return None; // Chain broken
            }
            prev_hash = entry.chain_hash;
        }

        let next_seq = entries.last().map(|e| e.sequence + 1).unwrap_or(0);

        Some(ReplayLog {
            chain_head: prev_hash,
            next_sequence: next_seq,
            entries,
            mode: LogMode::Replay,
            replay_cursor: 0,
            divergences: Vec::new(),
            snapshots: Vec::new(),
            config,
            total_data_bytes: 0,
        })
    }

    /// Set the log mode.
    pub fn set_mode(&mut self, mode: LogMode) {
        self.mode = mode;
        if mode == LogMode::Replay || mode == LogMode::Verify {
            self.replay_cursor = 0;
        }
    }

    pub fn mode(&self) -> LogMode {
        self.mode
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Live operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Append an oracle response to the log (Live mode).
    /// Returns the log entry and an oracle token for the neural substrate.
    pub fn append(
        &mut self,
        epoch: Epoch,
        oracle: OracleId,
        query: QueryParams,
        response: OracleResponse,
    ) -> (LogEntry, OracleToken) {
        let sequence = self.next_sequence;

        let chain_hash = LogEntry::compute_chain_hash(
            sequence,
            &epoch,
            oracle,
            &response,
            &self.chain_head,
        );

        let token = OracleToken::new(epoch, oracle, 0, &response);

        let data_size = match &response.data {
            ResponseData::Bytes(b) => b.len(),
            ResponseData::ContentRef { size_bytes, .. } => *size_bytes as usize,
            _ => 8,
        };
        self.total_data_bytes += data_size as u64;

        let entry = LogEntry {
            sequence,
            epoch,
            oracle,
            query,
            response,
            chain_hash,
        };

        self.chain_head = chain_hash;
        self.next_sequence += 1;
        self.entries.push(entry.clone());

        (entry, token)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Replay operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Read the next oracle response from the log (Replay mode).
    /// Returns None if the log is exhausted.
    pub fn replay_next(&mut self) -> Option<(LogEntry, OracleToken)> {
        if self.replay_cursor >= self.entries.len() {
            return None;
        }

        let entry = self.entries[self.replay_cursor].clone();
        let token = OracleToken::new(entry.epoch, entry.oracle, 0, &entry.response);
        self.replay_cursor += 1;

        Some((entry, token))
    }

    /// Peek at the next replay entry without advancing the cursor.
    pub fn replay_peek(&self) -> Option<&LogEntry> {
        self.entries.get(self.replay_cursor)
    }

    /// Check if the replay cursor matches an expected query.
    /// In Verify mode, compare the live response against the logged one.
    pub fn verify_or_replay(
        &mut self,
        epoch: Epoch,
        oracle: OracleId,
        query: &QueryParams,
        live_response: Option<&OracleResponse>,
    ) -> Option<(OracleResponse, OracleToken)> {
        match self.mode {
            LogMode::Live => {
                // Shouldn't be called in live mode via this path
                None
            }
            LogMode::Replay => {
                // Return logged response, ignore live
                let (entry, token) = self.replay_next()?;
                Some((entry.response, token))
            }
            LogMode::Verify => {
                // Compare live vs logged
                let (entry, token) = self.replay_next()?;
                if let Some(live) = live_response {
                    if live != &entry.response {
                        self.divergences.push(Divergence {
                            sequence: entry.sequence,
                            epoch,
                            oracle,
                            expected: entry.response.clone(),
                            actual: live.clone(),
                        });
                    }
                }
                // Always return logged response for determinism
                Some((entry.response, token))
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Snapshot and pruning
    // ─────────────────────────────────────────────────────────────────────────

    /// Take a snapshot at the current state.
    pub fn take_snapshot(
        &mut self,
        weights_hash: [u8; 32],
        state_hash: [u8; 32],
        ial_digest: (u64, u64),
    ) -> Snapshot {
        let last_entry = self.entries.last();
        let snapshot = Snapshot {
            epoch: Epoch(last_entry.map(|e| e.epoch.0).unwrap_or(0)),
            last_sequence: self.next_sequence.saturating_sub(1),
            chain_hash: self.chain_head,
            weights_hash,
            state_hash,
            ial_digest,
        };
        self.snapshots.push(snapshot.clone());
        snapshot
    }

    /// Prune log entries up to (but not including) the given sequence number.
    /// Only safe if a snapshot exists at or after that sequence.
    pub fn prune_before(&mut self, sequence: u64) -> usize {
        // Verify a snapshot exists that covers the pruned range
        let has_covering_snapshot = self.snapshots.iter()
            .any(|s| s.last_sequence >= sequence);

        if !has_covering_snapshot {
            return 0;
        }

        let original_len = self.entries.len();
        self.entries.retain(|e| e.sequence >= sequence);

        // Adjust replay cursor
        if self.replay_cursor > 0 {
            let pruned = original_len - self.entries.len();
            self.replay_cursor = self.replay_cursor.saturating_sub(pruned);
        }

        original_len - self.entries.len()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Verification
    // ─────────────────────────────────────────────────────────────────────────

    /// Verify the entire hash chain from genesis.
    pub fn verify_chain(&self) -> Result<(), (u64, &str)> {
        let mut prev_hash = [0u8; 8];
        for entry in &self.entries {
            if !entry.verify(&prev_hash) {
                return Err((entry.sequence, "chain hash mismatch"));
            }
            prev_hash = entry.chain_hash;
        }
        Ok(())
    }

    /// Compare this log against another log, returning the first divergence point.
    pub fn find_divergence_with(&self, other: &ReplayLog) -> Option<u64> {
        let min_len = self.entries.len().min(other.entries.len());
        for i in 0..min_len {
            if self.entries[i].chain_hash != other.entries[i].chain_hash {
                return Some(self.entries[i].sequence);
            }
        }
        // If one is longer, divergence is at the end of the shorter
        if self.entries.len() != other.entries.len() {
            return Some(min_len as u64);
        }
        None // Identical
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────────────────

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn chain_head(&self) -> &[u8; 8] {
        &self.chain_head
    }

    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    pub fn divergences(&self) -> &[Divergence] {
        &self.divergences
    }

    pub fn snapshots(&self) -> &[Snapshot] {
        &self.snapshots
    }

    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    pub fn stats(&self) -> LogStats {
        LogStats {
            entries: self.entries.len() as u64,
            chain_head: self.chain_head,
            next_sequence: self.next_sequence,
            divergences: self.divergences.len(),
            snapshots: self.snapshots.len(),
            total_data_bytes: self.total_data_bytes,
            mode: self.mode,
        }
    }
}

/// Log statistics for monitoring.
#[derive(Debug, Clone)]
pub struct LogStats {
    pub entries: u64,
    pub chain_head: [u8; 8],
    pub next_sequence: u64,
    pub divergences: usize,
    pub snapshots: usize,
    pub total_data_bytes: u64,
    pub mode: LogMode,
}

impl std::fmt::Display for LogStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NDAL Log: {} entries, chain={:02X?}, mode={:?}, divergences={}, snapshots={}, data={}B",
            self.entries,
            &self.chain_head[..4],
            self.mode,
            self.divergences,
            self.snapshots,
            self.total_data_bytes,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_random_response(value: u64) -> OracleResponse {
        OracleResponse {
            data: ResponseData::U64(value),
            status: OracleStatus::Ok,
        }
    }

    #[test]
    fn log_append_and_chain() {
        let mut log = ReplayLog::new(NdalConfig::default());

        let (e1, _) = log.append(
            Epoch(0),
            OracleId::Random,
            QueryParams::Random { n_bytes: 4 },
            make_random_response(42),
        );
        assert_eq!(e1.sequence, 0);

        let (e2, _) = log.append(
            Epoch(1),
            OracleId::Random,
            QueryParams::Random { n_bytes: 4 },
            make_random_response(99),
        );
        assert_eq!(e2.sequence, 1);

        // Chain should be valid
        assert!(log.verify_chain().is_ok());
    }

    #[test]
    fn log_replay_produces_same_tokens() {
        let config = NdalConfig::default();
        let mut live_log = ReplayLog::new(config.clone());

        // Live session: append 5 entries
        let mut live_tokens = Vec::new();
        for i in 0..5 {
            let (_, token) = live_log.append(
                Epoch(i),
                OracleId::Random,
                QueryParams::Random { n_bytes: 4 },
                make_random_response(i * 100 + 7),
            );
            live_tokens.push(token);
        }

        // Replay session: read from the same entries
        let entries = live_log.entries().to_vec();
        let mut replay_log = ReplayLog::from_entries(entries, config).unwrap();

        let mut replay_tokens = Vec::new();
        while let Some((_, token)) = replay_log.replay_next() {
            replay_tokens.push(token);
        }

        assert_eq!(live_tokens, replay_tokens,
            "Replay must produce identical oracle tokens");
    }

    #[test]
    fn log_verify_mode_detects_divergence() {
        let config = NdalConfig::default();
        let mut log = ReplayLog::new(config.clone());

        // Record a live session
        log.append(
            Epoch(0),
            OracleId::Random,
            QueryParams::Random { n_bytes: 4 },
            make_random_response(42),
        );

        // Switch to verify mode
        let entries = log.entries().to_vec();
        let mut verify_log = ReplayLog::from_entries(entries, config).unwrap();
        verify_log.set_mode(LogMode::Verify);

        // Simulate a live response that differs from logged
        let live_response = make_random_response(99); // Different!

        let result = verify_log.verify_or_replay(
            Epoch(0),
            OracleId::Random,
            &QueryParams::Random { n_bytes: 4 },
            Some(&live_response),
        );

        assert!(result.is_some());
        // Should have recorded a divergence
        assert_eq!(verify_log.divergences().len(), 1);
        assert_eq!(verify_log.divergences()[0].sequence, 0);
    }

    #[test]
    fn two_logs_divergence_detection() {
        let config = NdalConfig::default();

        let mut log_a = ReplayLog::new(config.clone());
        let mut log_b = ReplayLog::new(config);

        // Both get the same first entry
        log_a.append(Epoch(0), OracleId::Random,
            QueryParams::Random { n_bytes: 4 }, make_random_response(42));
        log_b.append(Epoch(0), OracleId::Random,
            QueryParams::Random { n_bytes: 4 }, make_random_response(42));

        // Second entry differs (different universe)
        log_a.append(Epoch(1), OracleId::Random,
            QueryParams::Random { n_bytes: 4 }, make_random_response(100));
        log_b.append(Epoch(1), OracleId::Random,
            QueryParams::Random { n_bytes: 4 }, make_random_response(200));

        let divergence = log_a.find_divergence_with(&log_b);
        assert_eq!(divergence, Some(1), "Should detect divergence at sequence 1");
    }

    #[test]
    fn identical_logs_no_divergence() {
        let config = NdalConfig::default();
        let mut log_a = ReplayLog::new(config.clone());
        let mut log_b = ReplayLog::new(config);

        for i in 0..10 {
            let resp = make_random_response(i * 7 + 3);
            log_a.append(Epoch(i), OracleId::Random,
                QueryParams::Random { n_bytes: 4 }, resp.clone());
            log_b.append(Epoch(i), OracleId::Random,
                QueryParams::Random { n_bytes: 4 }, resp);
        }

        assert_eq!(log_a.find_divergence_with(&log_b), None);
    }

    #[test]
    fn snapshot_and_prune() {
        let config = NdalConfig::default();
        let mut log = ReplayLog::new(config);

        // Append 100 entries
        for i in 0..100 {
            log.append(Epoch(i), OracleId::Random,
                QueryParams::Random { n_bytes: 4 },
                make_random_response(i));
        }
        assert_eq!(log.len(), 100);

        // Take snapshot at current state
        log.take_snapshot([0u8; 32], [0u8; 32], (0, 0));

        // Prune entries before sequence 50
        let pruned = log.prune_before(50);
        assert_eq!(pruned, 50);
        assert_eq!(log.len(), 50);

        // Chain should still be verifiable from the remaining entries
        // (chain verification starts from genesis, so partial verification
        // would need the snapshot's chain_hash as starting point.
        // For now, verify the entries are correctly retained.)
        assert_eq!(log.entries()[0].sequence, 50);
    }

    #[test]
    fn log_from_entries_rejects_broken_chain() {
        let config = NdalConfig::default();
        let mut log = ReplayLog::new(config.clone());

        log.append(Epoch(0), OracleId::Random,
            QueryParams::Random { n_bytes: 4 }, make_random_response(42));
        log.append(Epoch(1), OracleId::Random,
            QueryParams::Random { n_bytes: 4 }, make_random_response(99));

        let mut entries = log.entries().to_vec();
        // Tamper with entry 1's chain hash
        entries[1].chain_hash[0] ^= 0xFF;

        let result = ReplayLog::from_entries(entries, config);
        assert!(result.is_none(), "Broken chain should be rejected");
    }
}
