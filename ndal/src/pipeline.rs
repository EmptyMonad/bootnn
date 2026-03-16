//! DNOS Non-Determinism Abstraction Layer — Pipeline
//!
//! Top-level interface. Accepts oracle queries from the neural substrate
//! (or its control logic), routes them through the appropriate oracle,
//! logs the response, and emits deterministic tokens.
//!
//! Usage:
//! ```ignore
//! let mut ndal = NdalPipeline::new(NdalConfig::default());
//!
//! // Live mode: query the universe
//! let token = ndal.query(Epoch(0), QueryParams::Random { n_bytes: 4 });
//!
//! // Switch to replay
//! ndal.set_mode(LogMode::Replay);
//!
//! // Replay: same tokens from log
//! let token = ndal.replay_next();
//!
//! // Verify state across nodes
//! let digest = ndal.log_digest();
//! ```

use crate::types::*;
use crate::log::*;
use crate::oracles::*;

/// The NDAL pipeline. Top-level interface for all non-deterministic operations.
pub struct NdalPipeline {
    /// Oracle registry (routes queries to implementations)
    oracles: OracleRegistry,
    /// Replay log (records all oracle responses)
    log: ReplayLog,
    /// Configuration
    config: NdalConfig,
    /// Total oracle tokens emitted
    tokens_emitted: u64,
    /// Total queries processed
    queries_processed: u64,
}

impl NdalPipeline {
    /// Create a new NDAL pipeline in Live mode.
    pub fn new(config: NdalConfig) -> Self {
        let oracles = OracleRegistry::new(config.clone());
        let log = ReplayLog::new(config.clone());

        NdalPipeline {
            oracles,
            log,
            config,
            tokens_emitted: 0,
            queries_processed: 0,
        }
    }

    /// Create a pipeline preloaded with a log for replay.
    pub fn from_log(entries: Vec<LogEntry>, config: NdalConfig) -> Option<Self> {
        let log = ReplayLog::from_entries(entries, config.clone())?;
        let oracles = OracleRegistry::new(config.clone());

        Some(NdalPipeline {
            oracles,
            log,
            config,
            tokens_emitted: 0,
            queries_processed: 0,
        })
    }

    /// Set the operating mode.
    pub fn set_mode(&mut self, mode: LogMode) {
        self.log.set_mode(mode);
    }

    pub fn mode(&self) -> LogMode {
        self.log.mode()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Live queries
    // ─────────────────────────────────────────────────────────────────────────

    /// Query an oracle and get a deterministic token.
    ///
    /// In Live mode: queries the real universe, logs the response.
    /// In Replay mode: reads from log, ignores the query params.
    /// In Verify mode: queries both, logs divergence if different.
    ///
    /// Returns None if rate-limited or oracle unavailable.
    pub fn query(&mut self, epoch: Epoch, params: QueryParams) -> Option<OracleToken> {
        self.queries_processed += 1;

        match self.log.mode() {
            LogMode::Live => {
                let (oracle_id, response) = self.oracles.query_live(epoch, &params)?;
                let (_, token) = self.log.append(epoch, oracle_id, params, response);
                self.tokens_emitted += 1;
                Some(token)
            }
            LogMode::Replay => {
                let (_, token) = self.log.replay_next()?;
                self.tokens_emitted += 1;
                Some(token)
            }
            LogMode::Verify => {
                // Get live response
                let oracle_id = match &params {
                    QueryParams::Random { .. }      => OracleId::Random,
                    QueryParams::Clock { .. }       => OracleId::Clock,
                    QueryParams::Network { .. }     => OracleId::Network,
                    QueryParams::Consensus { .. }   => OracleId::Consensus,
                    QueryParams::Environment { .. } => OracleId::Environment,
                    QueryParams::User { .. }        => OracleId::User,
                    QueryParams::FileState { .. }   => OracleId::FileState,
                    QueryParams::SystemState { .. } => OracleId::SystemState,
                };

                let live_resp = self.oracles.query_live(epoch, &params)
                    .map(|(_, r)| r);

                let (response, token) = self.log.verify_or_replay(
                    epoch,
                    oracle_id,
                    &params,
                    live_resp.as_ref(),
                )?;

                self.tokens_emitted += 1;
                Some(token)
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Convenience methods for common oracle queries
    // ─────────────────────────────────────────────────────────────────────────

    /// Get N bytes of randomness.
    pub fn random(&mut self, epoch: Epoch, n_bytes: u32) -> Option<OracleToken> {
        self.query(epoch, QueryParams::Random { n_bytes })
    }

    /// Get wall-clock time at configured resolution.
    pub fn clock(&mut self, epoch: Epoch) -> Option<OracleToken> {
        self.query(epoch, QueryParams::Clock {
            resolution: self.config.clock_resolution,
        })
    }

    /// Query an environment value.
    pub fn env(&mut self, epoch: Epoch, key: EnvKey) -> Option<OracleToken> {
        self.query(epoch, QueryParams::Environment { key })
    }

    /// Ask peers for consensus on a question.
    pub fn consensus(&mut self, epoch: Epoch, question: ConsensusQuestion, quorum: u32) -> Option<OracleToken> {
        self.query(epoch, QueryParams::Consensus { question, quorum })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Deterministic training support
    // ─────────────────────────────────────────────────────────────────────────

    /// Get a deterministic shuffle seed for training.
    /// The seed is random in live mode, but recorded so replay
    /// produces the same shuffle order.
    pub fn training_shuffle_seed(&mut self, epoch: Epoch) -> Option<u64> {
        let token = self.random(epoch, 8)?;
        Some(u64::from_le_bytes(token.payload))
    }

    /// Get a deterministic initialization seed for weight init.
    pub fn weight_init_seed(&mut self, epoch: Epoch) -> Option<u64> {
        let token = self.random(epoch, 8)?;
        Some(u64::from_le_bytes(token.payload))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Snapshot and state management
    // ─────────────────────────────────────────────────────────────────────────

    /// Take a snapshot of the current state for pruning/sync.
    pub fn snapshot(
        &mut self,
        weights_hash: [u8; 32],
        state_hash: [u8; 32],
        ial_digest: (u64, u64),
    ) -> Snapshot {
        self.log.take_snapshot(weights_hash, state_hash, ial_digest)
    }

    /// Prune log entries before a sequence number.
    pub fn prune_before(&mut self, sequence: u64) -> usize {
        self.log.prune_before(sequence)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Verification
    // ─────────────────────────────────────────────────────────────────────────

    /// Get the log's chain head hash for cross-node comparison.
    pub fn log_chain_head(&self) -> &[u8; 8] {
        self.log.chain_head()
    }

    /// Verify the log's hash chain integrity.
    pub fn verify_chain(&self) -> Result<(), (u64, &str)> {
        self.log.verify_chain()
    }

    /// Find divergence point with another NDAL's log.
    pub fn find_divergence_with(&self, other: &NdalPipeline) -> Option<u64> {
        self.log.find_divergence_with(&other.log)
    }

    /// Get divergences detected in Verify mode.
    pub fn divergences(&self) -> &[Divergence] {
        self.log.divergences()
    }

    /// Configuration hash for cross-node compatibility check.
    pub fn config_hash(&self) -> u64 {
        self.config.config_hash()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────────────────

    pub fn log(&self) -> &ReplayLog {
        &self.log
    }

    pub fn log_entries(&self) -> &[LogEntry] {
        self.log.entries()
    }

    /// Seed the random oracle (for testing/deterministic scenarios).
    pub fn seed_random(&mut self, seed: u64) {
        self.oracles.seed_random(seed);
    }

    pub fn stats(&self) -> NdalStats {
        NdalStats {
            queries_processed: self.queries_processed,
            tokens_emitted: self.tokens_emitted,
            log_stats: self.log.stats(),
            config_hash: self.config.config_hash(),
        }
    }
}

/// NDAL statistics.
#[derive(Debug, Clone)]
pub struct NdalStats {
    pub queries_processed: u64,
    pub tokens_emitted: u64,
    pub log_stats: LogStats,
    pub config_hash: u64,
}

impl std::fmt::Display for NdalStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NDAL: {} queries → {} tokens, config={:016X}, {}",
            self.queries_processed,
            self.tokens_emitted,
            self.config_hash,
            self.log_stats,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_live_then_replay_produces_identical_tokens() {
        let config = NdalConfig::default();

        // Live session
        let mut live = NdalPipeline::new(config.clone());
        live.seed_random(12345);

        let mut live_tokens = Vec::new();
        for i in 0..10 {
            if let Some(token) = live.random(Epoch(i), 4) {
                live_tokens.push(token);
            }
        }

        // Replay session from live's log
        let entries = live.log_entries().to_vec();
        let mut replay = NdalPipeline::from_log(entries, config).unwrap();

        let mut replay_tokens = Vec::new();
        for i in 0..10 {
            if let Some(token) = replay.random(Epoch(i), 4) {
                replay_tokens.push(token);
            }
        }

        assert_eq!(live_tokens, replay_tokens,
            "Replay must produce identical oracle tokens as live session");
    }

    #[test]
    fn two_live_sessions_same_seed_same_output() {
        let config = NdalConfig::default();

        let mut p1 = NdalPipeline::new(config.clone());
        let mut p2 = NdalPipeline::new(config);

        p1.seed_random(42);
        p2.seed_random(42);

        let mut t1 = Vec::new();
        let mut t2 = Vec::new();

        for i in 0..20 {
            t1.push(p1.random(Epoch(i), 4));
            t2.push(p2.random(Epoch(i), 4));
        }

        assert_eq!(t1, t2, "Same seed must produce same oracle output");
    }

    #[test]
    fn different_seeds_produce_different_logs() {
        let config = NdalConfig::default();

        let mut p1 = NdalPipeline::new(config.clone());
        let mut p2 = NdalPipeline::new(config);

        p1.seed_random(42);
        p2.seed_random(43);

        for i in 0..5 {
            p1.random(Epoch(i), 4);
            p2.random(Epoch(i), 4);
        }

        let divergence = p1.find_divergence_with(&p2);
        assert_eq!(divergence, Some(0), "Different seeds should diverge at first entry");
    }

    #[test]
    fn verify_mode_detects_live_divergence() {
        let config = NdalConfig::default();

        // Record a session
        let mut live = NdalPipeline::new(config.clone());
        live.seed_random(42);
        for i in 0..5 {
            live.random(Epoch(i), 4);
        }

        // Create verifier with the recorded log but a DIFFERENT seed
        let entries = live.log_entries().to_vec();
        let mut verifier = NdalPipeline::from_log(entries, config).unwrap();
        verifier.seed_random(99); // Different seed!
        verifier.set_mode(LogMode::Verify);

        for i in 0..5 {
            verifier.random(Epoch(i), 4);
        }

        // Should have detected divergences
        assert!(!verifier.divergences().is_empty(),
            "Different seed in verify mode should produce divergences");
    }

    #[test]
    fn training_shuffle_seed_is_replayable() {
        let config = NdalConfig::default();

        let mut live = NdalPipeline::new(config.clone());
        live.seed_random(42);

        let seed1 = live.training_shuffle_seed(Epoch(0)).unwrap();
        let seed2 = live.training_shuffle_seed(Epoch(1)).unwrap();

        // Replay
        let entries = live.log_entries().to_vec();
        let mut replay = NdalPipeline::from_log(entries, config).unwrap();

        let r_seed1 = replay.training_shuffle_seed(Epoch(0)).unwrap();
        let r_seed2 = replay.training_shuffle_seed(Epoch(1)).unwrap();

        assert_eq!(seed1, r_seed1);
        assert_eq!(seed2, r_seed2);
    }

    #[test]
    fn snapshot_and_prune() {
        let config = NdalConfig::default();
        let mut ndal = NdalPipeline::new(config);
        ndal.seed_random(42);

        // Generate 50 oracle queries
        for i in 0..50 {
            ndal.random(Epoch(i), 4);
        }
        assert_eq!(ndal.log().len(), 50);

        // Snapshot
        ndal.snapshot([0u8; 32], [0u8; 32], (0, 0));

        // Prune first 25 entries
        let pruned = ndal.prune_before(25);
        assert_eq!(pruned, 25);
        assert_eq!(ndal.log().len(), 25);
    }

    #[test]
    fn chain_integrity() {
        let config = NdalConfig::default();
        let mut ndal = NdalPipeline::new(config);
        ndal.seed_random(42);

        for i in 0..100 {
            ndal.random(Epoch(i), 4);
        }

        assert!(ndal.verify_chain().is_ok());
    }

    #[test]
    fn env_oracle_returns_value() {
        let config = NdalConfig::default();
        let mut ndal = NdalPipeline::new(config);

        let token = ndal.env(Epoch(0), EnvKey::CpuCount);
        assert!(token.is_some(), "CPU count should be available");
    }

    #[test]
    fn clock_oracle_returns_truncated_time() {
        let config = NdalConfig {
            clock_resolution: ClockResolution::Hour,
            ..NdalConfig::default()
        };
        let mut ndal = NdalPipeline::new(config);

        let token = ndal.clock(Epoch(0));
        assert!(token.is_some());

        let ts = u64::from_le_bytes(token.unwrap().payload);
        assert_eq!(ts % 3600, 0, "Timestamp should be truncated to hour");
    }
}
