//! DNOS Non-Determinism Abstraction Layer — Core Types
//!
//! Defines the oracle vocabulary, log entry structure, and the interfaces
//! through which the neural substrate interacts with irreducible randomness.
//!
//! Design principle: Non-determinism is never raw. It is always:
//! 1. Requested through a named oracle
//! 2. Recorded in a hash-chained log
//! 3. Returned as a deterministic token
//!
//! The log makes any execution replayable. Two NDAL instances fed the same
//! log will produce bit-identical oracle token streams.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// Re-use IAL types for token compatibility
// In a real build these would come from the dnos-ial crate dependency.
// Here we define compatible standalone versions.

/// Epoch (matches IAL definition)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epoch(pub u64);

// ─────────────────────────────────────────────────────────────────────────────
// Oracle Identity
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies which oracle is being queried. Oracles are the named
/// interfaces to non-determinism. Each oracle type has a fixed query
/// and response schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum OracleId {
    /// Entropy source (RDRAND, /dev/urandom, thermal noise)
    Random      = 0,
    /// Wall-clock time at configurable resolution
    Clock       = 1,
    /// External network communication
    Network     = 2,
    /// Peer consensus queries
    Consensus   = 3,
    /// Hardware/environment capabilities
    Environment = 4,
    /// Direct human input (modal, non-keyboard)
    User        = 5,
    /// Filesystem state queries (existence, size, mtime)
    FileState   = 6,
    /// Process/system state queries (load, uptime, pid status)
    SystemState = 7,
}

impl OracleId {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(OracleId::Random),
            1 => Some(OracleId::Clock),
            2 => Some(OracleId::Network),
            3 => Some(OracleId::Consensus),
            4 => Some(OracleId::Environment),
            5 => Some(OracleId::User),
            6 => Some(OracleId::FileState),
            7 => Some(OracleId::SystemState),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            OracleId::Random      => "RANDOM",
            OracleId::Clock       => "CLOCK",
            OracleId::Network     => "NET",
            OracleId::Consensus   => "CONSENSUS",
            OracleId::Environment => "ENV",
            OracleId::User        => "USER",
            OracleId::FileState   => "FILESTATE",
            OracleId::SystemState => "SYSSTATE",
        }
    }
}

impl fmt::Display for OracleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Oracle Query
// ─────────────────────────────────────────────────────────────────────────────

/// A query to an oracle. Describes what non-deterministic information
/// is being requested.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OracleQuery {
    /// Which oracle to ask
    pub oracle: OracleId,
    /// Query-specific parameters (compact, oracle-dependent encoding)
    pub params: QueryParams,
}

/// Oracle-specific query parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QueryParams {
    /// Random: request N bytes of entropy
    Random {
        n_bytes: u32,
    },
    /// Clock: request time at a given resolution
    Clock {
        resolution: ClockResolution,
    },
    /// Network: external request
    Network {
        method: NetMethod,
        target_hash: u32,    // FNV hash of URL/address
        timeout_epochs: u32,
    },
    /// Consensus: ask peers a question
    Consensus {
        question: ConsensusQuestion,
        quorum: u32,
    },
    /// Environment: query a system capability
    Environment {
        key: EnvKey,
    },
    /// User: present a prompt and collect a choice
    User {
        prompt_hash: u32,
        n_options: u8,
    },
    /// File state: check existence/size/mtime of a file
    FileState {
        path_hash: u32,
        query_type: FileStateQuery,
    },
    /// System state: load, uptime, etc.
    SystemState {
        key: SysStateKey,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Query parameter enums
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClockResolution {
    Day,        // Truncated to midnight UTC
    Hour,       // Truncated to hour
    Minute,     // Truncated to minute
    Second,     // Truncated to second
    // No sub-second: that would leak timing back through the IAL
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NetMethod {
    Get,
    Post,
    Ping,       // Latency probe only
    Dns,        // DNS resolution
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConsensusQuestion {
    StateHash,          // "What is your state hash at current epoch?"
    WeightChecksum,     // "What is your weight matrix checksum?"
    PatternVote(u32),   // "Do you have pattern with hash X?"
    EpochSync,          // "What epoch are you on?"
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnvKey {
    TotalMemory,
    AvailableMemory,
    CpuCount,
    CpuFeatures,       // Bitmask: SSE, AVX, etc.
    StorageBytes,
    HasGpu,
    HasTpm,
    BootTimestamp,
    NodeId,             // Ed25519 public key hash
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileStateQuery {
    Exists,
    SizeBytes,
    ModifiedEpoch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SysStateKey {
    LoadAverage,        // Quantized to integer percentage
    Uptime,             // In epochs
    ProcessCount,
    NetworkUp,          // Boolean
}

// ─────────────────────────────────────────────────────────────────────────────
// Oracle Response
// ─────────────────────────────────────────────────────────────────────────────

/// A response from an oracle. This is the non-deterministic payload
/// that gets logged and converted to a token.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OracleResponse {
    /// Compact response data (up to 32 bytes).
    /// For larger responses (network bodies), this contains a content hash
    /// and the full data goes into a content-addressed store.
    pub data: ResponseData,
    /// Whether the query succeeded
    pub status: OracleStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResponseData {
    /// Raw bytes (up to 32 bytes inline)
    Bytes(Vec<u8>),
    /// Single unsigned integer
    U64(u64),
    /// Boolean
    Bool(bool),
    /// Hash reference to large content (stored separately)
    ContentRef {
        hash: [u8; 32],
        size_bytes: u64,
    },
    /// No data (for pure status responses)
    Empty,
}

impl ResponseData {
    /// Compact encoding to at most 8 bytes for token payload.
    pub fn to_payload_bytes(&self) -> [u8; 8] {
        let mut buf = [0u8; 8];
        match self {
            ResponseData::Bytes(b) => {
                let len = b.len().min(8);
                buf[..len].copy_from_slice(&b[..len]);
            }
            ResponseData::U64(v) => {
                buf.copy_from_slice(&v.to_le_bytes());
            }
            ResponseData::Bool(v) => {
                buf[0] = if *v { 1 } else { 0 };
            }
            ResponseData::ContentRef { hash, .. } => {
                // First 8 bytes of hash
                buf.copy_from_slice(&hash[..8]);
            }
            ResponseData::Empty => {}
        }
        buf
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OracleStatus {
    /// Query succeeded, data is valid
    Ok,
    /// Query timed out (e.g., network request)
    Timeout,
    /// Query failed (e.g., network error, file not found)
    Error,
    /// Oracle is unavailable (e.g., no TPM, no network)
    Unavailable,
    /// Query was denied (e.g., permission, rate limit)
    Denied,
}

// ─────────────────────────────────────────────────────────────────────────────
// Replay Log Entry
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry in the replay log. Contains everything needed to
/// deterministically reproduce the oracle's response during replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogEntry {
    /// Monotonic sequence number (0-indexed, no gaps)
    pub sequence: u64,
    /// Epoch when the query was made
    pub epoch: Epoch,
    /// Which oracle was queried
    pub oracle: OracleId,
    /// The query parameters (needed for verification during replay)
    pub query: QueryParams,
    /// The response from the oracle (the non-deterministic payload)
    pub response: OracleResponse,
    /// Hash chain link: H(entry_data || prev_hash)
    pub chain_hash: [u8; 8],
}

impl LogEntry {
    /// Compute the chain hash for this entry given the previous hash.
    pub fn compute_chain_hash(
        sequence: u64,
        epoch: &Epoch,
        oracle: OracleId,
        response: &OracleResponse,
        prev_hash: &[u8; 8],
    ) -> [u8; 8] {
        // FNV-1a hash of entry data + previous hash
        let mut h: u64 = 0xcbf29ce484222325;
        let prime: u64 = 0x100000001b3;

        let feed = |h: &mut u64, bytes: &[u8]| {
            for &b in bytes {
                *h ^= b as u64;
                *h = h.wrapping_mul(prime);
            }
        };

        feed(&mut h, &sequence.to_le_bytes());
        feed(&mut h, &epoch.0.to_le_bytes());
        feed(&mut h, &[oracle as u8]);
        feed(&mut h, &response.data.to_payload_bytes());
        feed(&mut h, &[response.status as u8]);
        feed(&mut h, prev_hash);

        h.to_le_bytes()
    }

    /// Verify this entry's chain hash against its predecessor.
    pub fn verify(&self, prev_hash: &[u8; 8]) -> bool {
        let expected = Self::compute_chain_hash(
            self.sequence,
            &self.epoch,
            self.oracle,
            &self.response,
            prev_hash,
        );
        self.chain_hash == expected
    }

    /// Serialize to bytes for storage/transmission.
    /// Format: [seq:8][epoch:8][oracle:1][status:1][payload:8][chain:8] = 34 bytes fixed
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(34);
        buf.extend_from_slice(&self.sequence.to_le_bytes());
        buf.extend_from_slice(&self.epoch.0.to_le_bytes());
        buf.push(self.oracle as u8);
        buf.push(self.response.status as u8);
        buf.extend_from_slice(&self.response.data.to_payload_bytes());
        buf.extend_from_slice(&self.chain_hash);
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < 34 {
            return None;
        }
        let sequence = u64::from_le_bytes(buf[0..8].try_into().ok()?);
        let epoch = Epoch(u64::from_le_bytes(buf[8..16].try_into().ok()?));
        let oracle = OracleId::from_u8(buf[16])?;
        let status = match buf[17] {
            0 => OracleStatus::Ok,
            1 => OracleStatus::Timeout,
            2 => OracleStatus::Error,
            3 => OracleStatus::Unavailable,
            4 => OracleStatus::Denied,
            _ => return None,
        };
        let mut payload = [0u8; 8];
        payload.copy_from_slice(&buf[18..26]);
        let mut chain_hash = [0u8; 8];
        chain_hash.copy_from_slice(&buf[26..34]);

        Some(LogEntry {
            sequence,
            epoch,
            oracle,
            query: QueryParams::Random { n_bytes: 0 }, // Query not stored in compact format
            response: OracleResponse {
                data: ResponseData::Bytes(payload.to_vec()),
                status,
            },
            chain_hash,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Oracle Token (what the neural substrate sees)
// ─────────────────────────────────────────────────────────────────────────────

/// A token emitted by the NDAL for consumption by the neural substrate.
/// Compatible with the IAL token format (19 bytes, same Ord semantics).
///
/// The neural substrate receives these interleaved with IAL tokens.
/// It doesn't know (or care) that the data came from a non-deterministic
/// source. It just sees: "at epoch E, oracle O said X."
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OracleToken {
    pub epoch: Epoch,
    /// Oracle ID encoded as channel-like discriminator (0x80 | oracle_id)
    /// to distinguish from IAL tokens (channels 0-7) in the merged stream.
    pub oracle_channel: u8,
    /// Oracle-specific sub-type (maps to IAL EventType)
    pub sub_type: u16,
    /// Quantized response data (8 bytes, matching IAL Payload)
    pub payload: [u8; 8],
}

impl OracleToken {
    pub fn new(epoch: Epoch, oracle: OracleId, sub_type: u16, response: &OracleResponse) -> Self {
        OracleToken {
            epoch,
            oracle_channel: 0x80 | (oracle as u8),
            sub_type,
            payload: response.data.to_payload_bytes(),
        }
    }

    /// Serialize to 19 bytes (IAL-compatible format).
    pub fn to_bytes(&self) -> [u8; 19] {
        let mut buf = [0u8; 19];
        buf[0..8].copy_from_slice(&self.epoch.0.to_le_bytes());
        buf[8] = self.oracle_channel;
        buf[9..11].copy_from_slice(&self.sub_type.to_le_bytes());
        buf[11..19].copy_from_slice(&self.payload);
        buf
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NDAL Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the NDAL. Controls which oracles are available,
/// rate limits, and log behavior.
#[derive(Debug, Clone)]
pub struct NdalConfig {
    /// Maximum oracle queries per epoch (rate limiting)
    pub max_queries_per_epoch: u32,

    /// Clock oracle resolution (prevents leaking fine timing)
    pub clock_resolution: ClockResolution,

    /// Network oracle timeout in epochs
    pub network_timeout_epochs: u32,

    /// Whether to log full query parameters (verbose) or just hashes (compact)
    pub verbose_log: bool,

    /// Maximum log entries before requiring a snapshot/prune
    pub max_log_entries: u64,

    /// Snapshot interval in epochs (0 = no automatic snapshots)
    pub snapshot_interval_epochs: u64,

    /// Enabled oracles (disabled oracles always return Unavailable)
    pub enabled_oracles: [bool; 8],
}

impl Default for NdalConfig {
    fn default() -> Self {
        NdalConfig {
            max_queries_per_epoch: 64,
            clock_resolution: ClockResolution::Second,
            network_timeout_epochs: 100,
            verbose_log: false,
            max_log_entries: 1_000_000,
            snapshot_interval_epochs: 10_000,
            enabled_oracles: [true; 8], // All enabled by default
        }
    }
}

impl NdalConfig {
    /// Bare-metal configuration: minimal oracles, no network.
    pub fn bare_metal() -> Self {
        let mut enabled = [false; 8];
        enabled[OracleId::Random as usize] = true;
        enabled[OracleId::Clock as usize] = true;
        enabled[OracleId::Environment as usize] = true;

        NdalConfig {
            max_queries_per_epoch: 8,
            clock_resolution: ClockResolution::Second,
            network_timeout_epochs: 0,
            verbose_log: false,
            max_log_entries: 10_000,
            snapshot_interval_epochs: 1_000,
            enabled_oracles: enabled,
        }
    }

    /// Distributed configuration: all oracles, longer logs.
    pub fn distributed() -> Self {
        NdalConfig {
            max_queries_per_epoch: 256,
            clock_resolution: ClockResolution::Second,
            network_timeout_epochs: 500,
            verbose_log: true,
            max_log_entries: 10_000_000,
            snapshot_interval_epochs: 100_000,
            enabled_oracles: [true; 8],
        }
    }

    /// FNV-1a hash of config for cross-node verification.
    pub fn config_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let prime: u64 = 0x100000001b3;

        let feed = |h: &mut u64, bytes: &[u8]| {
            for &b in bytes {
                *h ^= b as u64;
                *h = h.wrapping_mul(prime);
            }
        };

        feed(&mut h, &self.max_queries_per_epoch.to_le_bytes());
        feed(&mut h, &[self.clock_resolution as u8]);
        feed(&mut h, &self.network_timeout_epochs.to_le_bytes());
        feed(&mut h, &[self.verbose_log as u8]);
        for &enabled in &self.enabled_oracles {
            feed(&mut h, &[enabled as u8]);
        }

        h
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Content-Addressed Store interface
// ─────────────────────────────────────────────────────────────────────────────

/// Interface for storing large oracle responses (network bodies, etc.)
/// by content hash. The log stores only the hash; full data lives here.
pub trait ContentStore {
    /// Store data, returns its hash.
    fn put(&mut self, data: &[u8]) -> [u8; 32];
    /// Retrieve data by hash.
    fn get(&self, hash: &[u8; 32]) -> Option<Vec<u8>>;
    /// Check if data exists.
    fn has(&self, hash: &[u8; 32]) -> bool;
}

/// In-memory content store (for testing and bare-metal).
#[derive(Debug, Default)]
pub struct MemoryContentStore {
    store: std::collections::HashMap<[u8; 32], Vec<u8>>,
}

impl ContentStore for MemoryContentStore {
    fn put(&mut self, data: &[u8]) -> [u8; 32] {
        let hash = content_hash(data);
        self.store.insert(hash, data.to_vec());
        hash
    }

    fn get(&self, hash: &[u8; 32]) -> Option<Vec<u8>> {
        self.store.get(hash).cloned()
    }

    fn has(&self, hash: &[u8; 32]) -> bool {
        self.store.contains_key(hash)
    }
}

/// FNV-1a based content hash (not cryptographic, fine for integrity checking).
/// In production, use SHA-256.
pub fn content_hash(data: &[u8]) -> [u8; 32] {
    let mut result = [0u8; 32];
    // Use 4 independent FNV-1a hashes for 32 bytes
    for chunk in 0..4 {
        let mut h: u64 = 0xcbf29ce484222325u64.wrapping_add(chunk as u64);
        for &b in data {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        result[chunk * 8..(chunk + 1) * 8].copy_from_slice(&h.to_le_bytes());
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_entry_chain_hash_is_deterministic() {
        let prev = [0u8; 8];
        let resp = OracleResponse {
            data: ResponseData::U64(42),
            status: OracleStatus::Ok,
        };

        let h1 = LogEntry::compute_chain_hash(0, &Epoch(100), OracleId::Random, &resp, &prev);
        let h2 = LogEntry::compute_chain_hash(0, &Epoch(100), OracleId::Random, &resp, &prev);
        assert_eq!(h1, h2, "Same inputs must produce same chain hash");
    }

    #[test]
    fn log_entry_chain_hash_changes_with_sequence() {
        let prev = [0u8; 8];
        let resp = OracleResponse {
            data: ResponseData::U64(42),
            status: OracleStatus::Ok,
        };

        let h1 = LogEntry::compute_chain_hash(0, &Epoch(100), OracleId::Random, &resp, &prev);
        let h2 = LogEntry::compute_chain_hash(1, &Epoch(100), OracleId::Random, &resp, &prev);
        assert_ne!(h1, h2, "Different sequence must produce different hash");
    }

    #[test]
    fn log_entry_verify_valid_chain() {
        let prev = [0u8; 8];
        let resp = OracleResponse {
            data: ResponseData::U64(42),
            status: OracleStatus::Ok,
        };

        let chain_hash = LogEntry::compute_chain_hash(
            0, &Epoch(100), OracleId::Random, &resp, &prev
        );

        let entry = LogEntry {
            sequence: 0,
            epoch: Epoch(100),
            oracle: OracleId::Random,
            query: QueryParams::Random { n_bytes: 4 },
            response: resp,
            chain_hash,
        };

        assert!(entry.verify(&prev), "Valid chain should verify");
    }

    #[test]
    fn log_entry_detect_tampered_chain() {
        let prev = [0u8; 8];
        let resp = OracleResponse {
            data: ResponseData::U64(42),
            status: OracleStatus::Ok,
        };

        let mut chain_hash = LogEntry::compute_chain_hash(
            0, &Epoch(100), OracleId::Random, &resp, &prev
        );
        chain_hash[0] ^= 0xFF; // Tamper

        let entry = LogEntry {
            sequence: 0,
            epoch: Epoch(100),
            oracle: OracleId::Random,
            query: QueryParams::Random { n_bytes: 4 },
            response: resp,
            chain_hash,
        };

        assert!(!entry.verify(&prev), "Tampered chain should fail verification");
    }

    #[test]
    fn log_entry_serialization_roundtrip() {
        let resp = OracleResponse {
            data: ResponseData::U64(0xDEADBEEF),
            status: OracleStatus::Ok,
        };
        let chain_hash = LogEntry::compute_chain_hash(
            7, &Epoch(42), OracleId::Clock, &resp, &[0u8; 8]
        );

        let entry = LogEntry {
            sequence: 7,
            epoch: Epoch(42),
            oracle: OracleId::Clock,
            query: QueryParams::Clock { resolution: ClockResolution::Second },
            response: resp,
            chain_hash,
        };

        let bytes = entry.to_bytes();
        assert_eq!(bytes.len(), 34, "Compact entry should be 34 bytes");

        let restored = LogEntry::from_bytes(&bytes).unwrap();
        assert_eq!(restored.sequence, entry.sequence);
        assert_eq!(restored.epoch, entry.epoch);
        assert_eq!(restored.oracle, entry.oracle);
        assert_eq!(restored.chain_hash, entry.chain_hash);
    }

    #[test]
    fn oracle_token_is_ial_compatible() {
        let resp = OracleResponse {
            data: ResponseData::U64(42),
            status: OracleStatus::Ok,
        };
        let token = OracleToken::new(Epoch(100), OracleId::Random, 0, &resp);
        let bytes = token.to_bytes();
        assert_eq!(bytes.len(), 19, "Oracle token must be 19 bytes (IAL compatible)");

        // Channel byte should have high bit set to distinguish from IAL
        assert!(bytes[8] & 0x80 != 0, "Oracle channel must have high bit set");
    }

    #[test]
    fn content_store_put_get() {
        let mut store = MemoryContentStore::default();
        let data = b"hello, non-deterministic world";
        let hash = store.put(data);
        assert!(store.has(&hash));
        assert_eq!(store.get(&hash).unwrap(), data);
    }

    #[test]
    fn config_hash_sensitive() {
        let c1 = NdalConfig::default();
        let mut c2 = NdalConfig::default();
        c2.max_queries_per_epoch = 128;
        assert_ne!(c1.config_hash(), c2.config_hash());
    }
}
