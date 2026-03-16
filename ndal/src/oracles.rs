//! DNOS Non-Determinism Abstraction Layer — Oracle Implementations
//!
//! Each oracle is a named interface to a specific source of non-determinism.
//! In Live mode, it queries the real universe. In Replay mode, it reads
//! from the log. In both cases, the neural substrate receives the same
//! type of token.
//!
//! The oracle trait defines the interface. Concrete implementations
//! handle the actual non-deterministic queries.

use crate::types::*;
use crate::log::*;

// ─────────────────────────────────────────────────────────────────────────────
// Oracle Trait
// ─────────────────────────────────────────────────────────────────────────────

/// An oracle wraps a non-deterministic source and provides a deterministic
/// interface to the neural substrate via the replay log.
pub trait Oracle {
    /// The oracle's identity.
    fn id(&self) -> OracleId;

    /// Query the oracle in live mode. This actually touches the
    /// non-deterministic source (RNG, network, clock, etc.).
    fn query_live(&mut self, params: &QueryParams) -> OracleResponse;

    /// Whether this oracle is available in the current environment.
    fn available(&self) -> bool;
}

// ─────────────────────────────────────────────────────────────────────────────
// Random Oracle
// ─────────────────────────────────────────────────────────────────────────────

/// Source of entropy. Uses a deterministic PRNG seeded from system entropy
/// so that the stream is reproducible from the seed.
///
/// In live mode: seeds from system entropy, logs the seed.
/// In replay mode: seeds from logged seed, produces same stream.
pub struct RandomOracle {
    /// Xorshift64 state — deterministic after seeding
    state: u64,
    /// Whether the oracle has been seeded
    seeded: bool,
}

impl RandomOracle {
    pub fn new() -> Self {
        RandomOracle {
            state: 0,
            seeded: false,
        }
    }

    /// Seed from system entropy (live mode).
    pub fn seed_from_system(&mut self) {
        // Use time-based entropy since we can't guarantee /dev/urandom
        use std::time::{SystemTime, UNIX_EPOCH};
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        self.state = t.as_nanos() as u64;
        if self.state == 0 {
            self.state = 0xDEADBEEFCAFEBABE; // Fallback
        }
        self.seeded = true;
    }

    /// Seed from a known value (replay mode).
    pub fn seed_from_value(&mut self, seed: u64) {
        self.state = if seed == 0 { 1 } else { seed };
        self.seeded = true;
    }

    /// Generate next random u64 (xorshift64).
    fn next_u64(&mut self) -> u64 {
        if !self.seeded {
            self.seed_from_system();
        }
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generate N random bytes.
    fn random_bytes(&mut self, n: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(n);
        let mut remaining = n;
        while remaining > 0 {
            let val = self.next_u64();
            let bytes = val.to_le_bytes();
            let take = remaining.min(8);
            result.extend_from_slice(&bytes[..take]);
            remaining -= take;
        }
        result.truncate(n);
        result
    }
}

impl Oracle for RandomOracle {
    fn id(&self) -> OracleId {
        OracleId::Random
    }

    fn query_live(&mut self, params: &QueryParams) -> OracleResponse {
        match params {
            QueryParams::Random { n_bytes } => {
                let bytes = self.random_bytes(*n_bytes as usize);
                OracleResponse {
                    data: if bytes.len() <= 8 {
                        let mut val = 0u64;
                        for (i, &b) in bytes.iter().enumerate() {
                            val |= (b as u64) << (i * 8);
                        }
                        ResponseData::U64(val)
                    } else {
                        ResponseData::Bytes(bytes)
                    },
                    status: OracleStatus::Ok,
                }
            }
            _ => OracleResponse {
                data: ResponseData::Empty,
                status: OracleStatus::Error,
            },
        }
    }

    fn available(&self) -> bool {
        true // Random is always available
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Clock Oracle
// ─────────────────────────────────────────────────────────────────────────────

/// Wall-clock time at configurable resolution.
///
/// The resolution parameter prevents leaking sub-epoch timing back into
/// the deterministic layer. The network sees "it's March 16, 2025" not
/// "it's 14:23:07.847291".
pub struct ClockOracle {
    resolution: ClockResolution,
}

impl ClockOracle {
    pub fn new(resolution: ClockResolution) -> Self {
        ClockOracle { resolution }
    }

    fn truncate_timestamp(&self, unix_secs: u64) -> u64 {
        match self.resolution {
            ClockResolution::Day    => (unix_secs / 86400) * 86400,
            ClockResolution::Hour   => (unix_secs / 3600) * 3600,
            ClockResolution::Minute => (unix_secs / 60) * 60,
            ClockResolution::Second => unix_secs,
        }
    }
}

impl Oracle for ClockOracle {
    fn id(&self) -> OracleId {
        OracleId::Clock
    }

    fn query_live(&mut self, params: &QueryParams) -> OracleResponse {
        match params {
            QueryParams::Clock { resolution } => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                // Use the more restrictive of configured and requested resolution
                let effective_res = if (*resolution as u8) < (self.resolution as u8) {
                    self.resolution
                } else {
                    *resolution
                };

                let truncated = match effective_res {
                    ClockResolution::Day    => (now / 86400) * 86400,
                    ClockResolution::Hour   => (now / 3600) * 3600,
                    ClockResolution::Minute => (now / 60) * 60,
                    ClockResolution::Second => now,
                };

                OracleResponse {
                    data: ResponseData::U64(truncated),
                    status: OracleStatus::Ok,
                }
            }
            _ => OracleResponse {
                data: ResponseData::Empty,
                status: OracleStatus::Error,
            },
        }
    }

    fn available(&self) -> bool {
        true // Clock is always available
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Environment Oracle
// ─────────────────────────────────────────────────────────────────────────────

/// Queries hardware/environment capabilities. Typically queried once at boot
/// and cached. Values are constant for a given boot but differ between machines.
pub struct EnvironmentOracle {
    /// Cached environment values (query once, return from cache thereafter)
    cache: std::collections::HashMap<EnvKey, u64>,
}

impl EnvironmentOracle {
    pub fn new() -> Self {
        EnvironmentOracle {
            cache: std::collections::HashMap::new(),
        }
    }

    fn probe_value(&self, key: &EnvKey) -> u64 {
        match key {
            EnvKey::CpuCount => {
                // Portable: std::thread::available_parallelism
                std::thread::available_parallelism()
                    .map(|n| n.get() as u64)
                    .unwrap_or(1)
            }
            EnvKey::HasGpu => 0,     // TODO: detect via sysfs
            EnvKey::HasTpm => 0,     // TODO: check /dev/tpm0
            EnvKey::BootTimestamp => {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            }
            // For values we can't easily probe, return 0
            _ => 0,
        }
    }
}

impl Oracle for EnvironmentOracle {
    fn id(&self) -> OracleId {
        OracleId::Environment
    }

    fn query_live(&mut self, params: &QueryParams) -> OracleResponse {
        match params {
            QueryParams::Environment { key } => {
                let value = if let Some(&cached) = self.cache.get(key) {
                    cached
                } else {
                    let v = self.probe_value(key);
                    self.cache.insert(*key, v);
                    v
                };

                OracleResponse {
                    data: ResponseData::U64(value),
                    status: OracleStatus::Ok,
                }
            }
            _ => OracleResponse {
                data: ResponseData::Empty,
                status: OracleStatus::Error,
            },
        }
    }

    fn available(&self) -> bool {
        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Network Oracle (stub)
// ─────────────────────────────────────────────────────────────────────────────

/// External network communication oracle. Currently stubbed — returns
/// Unavailable in environments without network access.
pub struct NetworkOracle {
    enabled: bool,
}

impl NetworkOracle {
    pub fn new(enabled: bool) -> Self {
        NetworkOracle { enabled }
    }
}

impl Oracle for NetworkOracle {
    fn id(&self) -> OracleId {
        OracleId::Network
    }

    fn query_live(&mut self, params: &QueryParams) -> OracleResponse {
        match params {
            QueryParams::Network { method, target_hash, timeout_epochs } => {
                if !self.enabled {
                    return OracleResponse {
                        data: ResponseData::Empty,
                        status: OracleStatus::Unavailable,
                    };
                }
                // Stub: real implementation would use async HTTP/TCP
                OracleResponse {
                    data: ResponseData::Empty,
                    status: OracleStatus::Unavailable,
                }
            }
            _ => OracleResponse {
                data: ResponseData::Empty,
                status: OracleStatus::Error,
            },
        }
    }

    fn available(&self) -> bool {
        self.enabled
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Consensus Oracle (stub)
// ─────────────────────────────────────────────────────────────────────────────

/// Peer consensus oracle. Queries the P2P network for state agreement.
/// Currently stubbed — requires dnosd P2P layer integration.
pub struct ConsensusOracle {
    enabled: bool,
}

impl ConsensusOracle {
    pub fn new(enabled: bool) -> Self {
        ConsensusOracle { enabled }
    }
}

impl Oracle for ConsensusOracle {
    fn id(&self) -> OracleId {
        OracleId::Consensus
    }

    fn query_live(&mut self, params: &QueryParams) -> OracleResponse {
        OracleResponse {
            data: ResponseData::Empty,
            status: OracleStatus::Unavailable,
        }
    }

    fn available(&self) -> bool {
        self.enabled
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Oracle Registry — manages all oracle instances
// ─────────────────────────────────────────────────────────────────────────────

/// Registry that holds all oracle instances and routes queries.
pub struct OracleRegistry {
    random: RandomOracle,
    clock: ClockOracle,
    environment: EnvironmentOracle,
    network: NetworkOracle,
    consensus: ConsensusOracle,
    config: NdalConfig,
    /// Queries this epoch (for rate limiting)
    epoch_query_count: u32,
    current_epoch: Epoch,
}

impl OracleRegistry {
    pub fn new(config: NdalConfig) -> Self {
        let clock_res = config.clock_resolution;
        OracleRegistry {
            random: RandomOracle::new(),
            clock: ClockOracle::new(clock_res),
            environment: EnvironmentOracle::new(),
            network: NetworkOracle::new(config.enabled_oracles[OracleId::Network as usize]),
            consensus: ConsensusOracle::new(config.enabled_oracles[OracleId::Consensus as usize]),
            config,
            epoch_query_count: 0,
            current_epoch: Epoch(0),
        }
    }

    /// Route a query to the appropriate oracle in live mode.
    /// Returns None if rate-limited or oracle disabled.
    pub fn query_live(
        &mut self,
        epoch: Epoch,
        query: &QueryParams,
    ) -> Option<(OracleId, OracleResponse)> {
        // Reset rate limit counter on epoch boundary
        if epoch != self.current_epoch {
            self.epoch_query_count = 0;
            self.current_epoch = epoch;
        }

        // Rate limit
        if self.epoch_query_count >= self.config.max_queries_per_epoch {
            return None;
        }

        let oracle_id = query_to_oracle_id(query);

        // Check if oracle is enabled
        if !self.config.enabled_oracles[oracle_id as usize] {
            return Some((oracle_id, OracleResponse {
                data: ResponseData::Empty,
                status: OracleStatus::Unavailable,
            }));
        }

        let response = match oracle_id {
            OracleId::Random      => self.random.query_live(query),
            OracleId::Clock       => self.clock.query_live(query),
            OracleId::Environment => self.environment.query_live(query),
            OracleId::Network     => self.network.query_live(query),
            OracleId::Consensus   => self.consensus.query_live(query),
            _ => OracleResponse {
                data: ResponseData::Empty,
                status: OracleStatus::Unavailable,
            },
        };

        self.epoch_query_count += 1;
        Some((oracle_id, response))
    }

    /// Seed the random oracle with a known value (for replay/testing).
    pub fn seed_random(&mut self, seed: u64) {
        self.random.seed_from_value(seed);
    }
}

/// Map a query to its oracle ID.
fn query_to_oracle_id(query: &QueryParams) -> OracleId {
    match query {
        QueryParams::Random { .. }      => OracleId::Random,
        QueryParams::Clock { .. }       => OracleId::Clock,
        QueryParams::Network { .. }     => OracleId::Network,
        QueryParams::Consensus { .. }   => OracleId::Consensus,
        QueryParams::Environment { .. } => OracleId::Environment,
        QueryParams::User { .. }        => OracleId::User,
        QueryParams::FileState { .. }   => OracleId::FileState,
        QueryParams::SystemState { .. } => OracleId::SystemState,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_oracle_deterministic_from_seed() {
        let mut o1 = RandomOracle::new();
        let mut o2 = RandomOracle::new();

        o1.seed_from_value(42);
        o2.seed_from_value(42);

        let query = QueryParams::Random { n_bytes: 8 };

        let r1 = o1.query_live(&query);
        let r2 = o2.query_live(&query);

        assert_eq!(r1, r2, "Same seed must produce same random output");
    }

    #[test]
    fn random_oracle_different_seeds_differ() {
        let mut o1 = RandomOracle::new();
        let mut o2 = RandomOracle::new();

        o1.seed_from_value(42);
        o2.seed_from_value(43);

        let query = QueryParams::Random { n_bytes: 8 };

        let r1 = o1.query_live(&query);
        let r2 = o2.query_live(&query);

        assert_ne!(r1, r2);
    }

    #[test]
    fn clock_oracle_truncates_to_resolution() {
        let mut clock = ClockOracle::new(ClockResolution::Hour);

        let resp = clock.query_live(&QueryParams::Clock {
            resolution: ClockResolution::Second,
        });

        // The response should be truncated to hour (more restrictive wins)
        if let OracleResponse { data: ResponseData::U64(ts), status: OracleStatus::Ok } = resp {
            assert_eq!(ts % 3600, 0, "Should be truncated to hour boundary");
        } else {
            panic!("Expected U64 response");
        }
    }

    #[test]
    fn environment_oracle_caches() {
        let mut env = EnvironmentOracle::new();

        let query = QueryParams::Environment { key: EnvKey::CpuCount };
        let r1 = env.query_live(&query);
        let r2 = env.query_live(&query);

        assert_eq!(r1, r2, "Environment values should be cached");
    }

    #[test]
    fn registry_rate_limits() {
        let config = NdalConfig {
            max_queries_per_epoch: 2,
            ..NdalConfig::default()
        };
        let mut reg = OracleRegistry::new(config);
        reg.seed_random(42);

        let query = QueryParams::Random { n_bytes: 4 };
        let epoch = Epoch(0);

        assert!(reg.query_live(epoch, &query).is_some());
        assert!(reg.query_live(epoch, &query).is_some());
        assert!(reg.query_live(epoch, &query).is_none(), "Third query should be rate-limited");

        // New epoch resets counter
        assert!(reg.query_live(Epoch(1), &query).is_some());
    }

    #[test]
    fn registry_disabled_oracle_returns_unavailable() {
        let mut config = NdalConfig::default();
        config.enabled_oracles[OracleId::Network as usize] = false;

        let mut reg = OracleRegistry::new(config);

        let query = QueryParams::Network {
            method: NetMethod::Get,
            target_hash: 0xBEEF,
            timeout_epochs: 100,
        };

        let (_, resp) = reg.query_live(Epoch(0), &query).unwrap();
        assert_eq!(resp.status, OracleStatus::Unavailable);
    }
}
