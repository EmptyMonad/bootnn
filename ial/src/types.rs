//! DNOS Input Abstraction Layer — Core Types
//!
//! Defines the deterministic token vocabulary that sits between
//! raw physical observations and the neural substrate.
//!
//! Design invariant: Two IAL instances with the same configuration,
//! processing the same physical events, produce bit-identical token streams.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

// ─────────────────────────────────────────────────────────────────────────────
// Epoch — discrete time quantum
// ─────────────────────────────────────────────────────────────────────────────

/// A discrete time unit. All events within the same epoch are treated
/// as simultaneous and subject to canonical ordering.
///
/// Epoch boundaries are absolute: epoch N starts at N * epoch_duration_us
/// from the reference timestamp (UNIX epoch or boot time).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epoch(pub u64);

impl Epoch {
    /// Compute epoch from a microsecond timestamp and epoch duration.
    ///
    /// This is the core quantization operation. Two timestamps that differ
    /// by less than `epoch_duration_us` AND fall within the same epoch
    /// boundary produce the same Epoch value.
    #[inline]
    pub fn from_timestamp_us(timestamp_us: u64, epoch_duration_us: u64) -> Self {
        debug_assert!(epoch_duration_us > 0, "epoch duration must be positive");
        Epoch(timestamp_us / epoch_duration_us)
    }

    /// Returns the start of this epoch in microseconds.
    #[inline]
    pub fn start_us(&self, epoch_duration_us: u64) -> u64 {
        self.0 * epoch_duration_us
    }

    /// Returns the end of this epoch in microseconds (exclusive).
    #[inline]
    pub fn end_us(&self, epoch_duration_us: u64) -> u64 {
        (self.0 + 1) * epoch_duration_us
    }
}

impl fmt::Display for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{}", self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Channel — input source identifier
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies the source of an observation.
/// Channels are ordered for canonical sorting within an epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Channel {
    /// Keyboard events (key down, key up, repeat)
    Keyboard    = 0,
    /// Mouse movement and button events
    Mouse       = 1,
    /// File system events (open, close, modify, create, delete)
    FileSystem  = 2,
    /// Network events (connect, disconnect, packet)
    Network     = 3,
    /// System events (process start/stop, signal, timer)
    System      = 4,
    /// Sensor data (temperature, accelerometer, etc.)
    Sensor      = 5,
    /// Explicit timeout / heartbeat (no event occurred)
    Heartbeat   = 6,
    /// Dropped/missed event indicator
    Miss        = 7,
}

impl Channel {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Channel::Keyboard),
            1 => Some(Channel::Mouse),
            2 => Some(Channel::FileSystem),
            3 => Some(Channel::Network),
            4 => Some(Channel::System),
            5 => Some(Channel::Sensor),
            6 => Some(Channel::Heartbeat),
            7 => Some(Channel::Miss),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EventType — semantic classification within a channel
// ─────────────────────────────────────────────────────────────────────────────

/// Semantic event type. Combined with Channel, this uniquely identifies
/// what kind of thing happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventType(pub u16);

// Keyboard event types
pub const EVT_KEY_DOWN:   EventType = EventType(0x0001);
pub const EVT_KEY_UP:     EventType = EventType(0x0002);
pub const EVT_KEY_REPEAT: EventType = EventType(0x0003);
pub const EVT_WORD:       EventType = EventType(0x0010); // Semantic: word recognized

// Mouse event types
pub const EVT_MOUSE_MOVE:    EventType = EventType(0x0100);
pub const EVT_MOUSE_CLICK:   EventType = EventType(0x0101);
pub const EVT_MOUSE_RELEASE: EventType = EventType(0x0102);
pub const EVT_MOUSE_SCROLL:  EventType = EventType(0x0103);

// File system event types
pub const EVT_FILE_OPEN:    EventType = EventType(0x0200);
pub const EVT_FILE_CLOSE:   EventType = EventType(0x0201);
pub const EVT_FILE_MODIFY:  EventType = EventType(0x0202);
pub const EVT_FILE_CREATE:  EventType = EventType(0x0203);
pub const EVT_FILE_DELETE:  EventType = EventType(0x0204);

// System event types
pub const EVT_PROC_START:   EventType = EventType(0x0300);
pub const EVT_PROC_EXIT:    EventType = EventType(0x0301);
pub const EVT_SIGNAL:       EventType = EventType(0x0302);

// Meta event types
pub const EVT_HEARTBEAT:    EventType = EventType(0x0F00);
pub const EVT_MISS:         EventType = EventType(0x0F01);
pub const EVT_EPOCH_BOUNDARY: EventType = EventType(0x0FFF);

// ─────────────────────────────────────────────────────────────────────────────
// Payload — quantized event data
// ─────────────────────────────────────────────────────────────────────────────

/// Fixed-size payload for event data. 8 bytes, interpreted according to
/// the Channel + EventType.
///
/// For keyboard: [scancode, 0, 0, 0, 0, 0, 0, 0]
/// For mouse click: [button, 0, x_bucket_lo, x_bucket_hi, y_bucket_lo, y_bucket_hi, 0, 0]
/// For file events: [hash of filename, first 4 bytes]
/// For heartbeat: [0; 8]
#[derive(Debug, Clone, Copy, Eq)]
pub struct Payload(pub [u8; 8]);

impl Payload {
    pub const ZERO: Payload = Payload([0u8; 8]);

    /// Create from a single byte value (e.g., scancode).
    pub fn from_byte(b: u8) -> Self {
        let mut p = [0u8; 8];
        p[0] = b;
        Payload(p)
    }

    /// Create from two u16 values (e.g., quantized x,y coordinates).
    pub fn from_u16_pair(a: u16, b: u16) -> Self {
        let mut p = [0u8; 8];
        p[0..2].copy_from_slice(&a.to_le_bytes());
        p[2..4].copy_from_slice(&b.to_le_bytes());
        Payload(p)
    }

    /// Create from a u32 (e.g., filename hash).
    pub fn from_u32(v: u32) -> Self {
        let mut p = [0u8; 8];
        p[0..4].copy_from_slice(&v.to_le_bytes());
        Payload(p)
    }

    /// Create from raw bytes (truncated or zero-padded to 8).
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut p = [0u8; 8];
        let len = bytes.len().min(8);
        p[..len].copy_from_slice(&bytes[..len]);
        Payload(p)
    }
}

impl PartialEq for Payload {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for Payload {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Payload {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl Hash for Payload {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Token — the fundamental unit of deterministic input
// ─────────────────────────────────────────────────────────────────────────────

/// A Token is the atomic unit of input to the DNOS neural substrate.
///
/// Tokens are:
/// - Fixed-size (19 bytes serialized)
/// - Totally ordered (for canonical sorting)
/// - Hashable (for state verification)
/// - Self-describing (channel + event_type + payload)
///
/// The canonical ordering is: (epoch, channel, event_type, payload).
/// This is a pure lexicographic sort on the binary representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Token {
    pub epoch:      Epoch,
    pub channel:    Channel,
    pub event_type: EventType,
    pub payload:    Payload,
}

impl Token {
    pub fn new(epoch: Epoch, channel: Channel, event_type: EventType, payload: Payload) -> Self {
        Token { epoch, channel, event_type, payload }
    }

    /// Heartbeat token: "nothing happened in this epoch."
    pub fn heartbeat(epoch: Epoch) -> Self {
        Token {
            epoch,
            channel: Channel::Heartbeat,
            event_type: EVT_HEARTBEAT,
            payload: Payload::ZERO,
        }
    }

    /// Miss token: "an event was expected but not received."
    pub fn miss(epoch: Epoch, channel: Channel) -> Self {
        Token {
            epoch,
            channel: Channel::Miss,
            event_type: EVT_MISS,
            payload: Payload::from_byte(channel as u8),
        }
    }

    /// Serialize to exactly 19 bytes, canonical binary representation.
    pub fn to_bytes(&self) -> [u8; 19] {
        let mut buf = [0u8; 19];
        buf[0..8].copy_from_slice(&self.epoch.0.to_le_bytes());
        buf[8] = self.channel as u8;
        buf[9..11].copy_from_slice(&self.event_type.0.to_le_bytes());
        buf[11..19].copy_from_slice(&self.payload.0);
        buf
    }

    /// Deserialize from exactly 19 bytes.
    pub fn from_bytes(buf: &[u8; 19]) -> Option<Self> {
        let epoch = Epoch(u64::from_le_bytes(buf[0..8].try_into().ok()?));
        let channel = Channel::from_u8(buf[8])?;
        let event_type = EventType(u16::from_le_bytes(buf[9..11].try_into().ok()?));
        let mut payload = [0u8; 8];
        payload.copy_from_slice(&buf[11..19]);
        Some(Token {
            epoch,
            channel,
            event_type,
            payload: Payload(payload),
        })
    }
}

impl Ord for Token {
    fn cmp(&self, other: &Self) -> Ordering {
        self.epoch.cmp(&other.epoch)
            .then(self.channel.cmp(&other.channel))
            .then(self.event_type.cmp(&other.event_type))
            .then(self.payload.cmp(&other.payload))
    }
}

impl PartialOrd for Token {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} {:?} {:04X} {:02X?}]",
            self.epoch, self.channel, self.event_type.0, &self.payload.0[..4])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IAL Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Input Abstraction Layer.
///
/// All fields that affect token generation must be identical across nodes
/// for deterministic equivalence.
#[derive(Debug, Clone, PartialEq)]
pub struct IalConfig {
    /// Epoch duration in microseconds. Events within the same epoch are
    /// treated as simultaneous. Must be > 0.
    ///
    /// Recommended values:
    ///   10_000 (10ms) — human perceptual window, good default
    ///   1_000  (1ms)  — high-resolution, requires tight clock sync
    ///   100_000 (100ms) — coarse, very jitter-tolerant
    pub epoch_duration_us: u64,

    /// Spatial quantization grid size for mouse/cursor events.
    /// Coordinates are divided by this value and truncated.
    pub spatial_grid_size: u16,

    /// Sensor quantization resolution. Raw sensor values are divided
    /// by this and rounded to the nearest integer.
    pub sensor_resolution: f64,

    /// Maximum number of tokens per epoch before overflow handling.
    /// If more than this many events arrive in a single epoch, excess
    /// events are collapsed into a single OVERFLOW token.
    pub max_tokens_per_epoch: usize,

    /// Whether to emit heartbeat tokens for empty epochs.
    /// If true, every epoch produces at least one token.
    /// This provides a guaranteed clock signal to the neural substrate.
    pub emit_heartbeats: bool,

    /// Number of consecutive empty epochs before emitting a heartbeat.
    /// Only relevant if emit_heartbeats is true.
    /// Set to 1 for every-epoch heartbeats, higher for sparser.
    pub heartbeat_interval: u64,

    /// Multi-resolution layers. If non-empty, tokens are generated at
    /// multiple temporal resolutions simultaneously.
    /// Each entry is an epoch duration in microseconds.
    /// The primary epoch_duration_us is always layer 0.
    pub multi_resolution_layers: Vec<u64>,
}

impl Default for IalConfig {
    fn default() -> Self {
        IalConfig {
            epoch_duration_us: 10_000,  // 10ms
            spatial_grid_size: 10,
            sensor_resolution: 0.5,
            max_tokens_per_epoch: 256,
            emit_heartbeats: true,
            heartbeat_interval: 10,     // Every 100ms at default epoch
            multi_resolution_layers: vec![],
        }
    }
}

impl IalConfig {
    /// Configuration hash — must match across nodes for deterministic
    /// equivalence. Uses a simple FNV-1a hash of the config fields.
    pub fn config_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
        let prime: u64 = 0x100000001b3;

        let mix = |h: &mut u64, bytes: &[u8]| {
            for &b in bytes {
                *h ^= b as u64;
                *h = h.wrapping_mul(prime);
            }
        };

        mix(&mut h, &self.epoch_duration_us.to_le_bytes());
        mix(&mut h, &self.spatial_grid_size.to_le_bytes());
        mix(&mut h, &self.sensor_resolution.to_le_bytes());
        mix(&mut h, &(self.max_tokens_per_epoch as u64).to_le_bytes());
        mix(&mut h, &[self.emit_heartbeats as u8]);
        mix(&mut h, &self.heartbeat_interval.to_le_bytes());
        for layer in &self.multi_resolution_layers {
            mix(&mut h, &layer.to_le_bytes());
        }

        h
    }

    /// Bare-metal configuration: tight epochs, no heartbeats,
    /// keyboard-only. For Tier 1/2 assembly integration.
    pub fn bare_metal() -> Self {
        IalConfig {
            epoch_duration_us: 10_000,  // PIT tick = 10ms at 100Hz
            spatial_grid_size: 1,       // Pixel-level
            sensor_resolution: 1.0,
            max_tokens_per_epoch: 16,
            emit_heartbeats: false,
            heartbeat_interval: 0,
            multi_resolution_layers: vec![],
        }
    }

    /// Distributed configuration: wider epochs for jitter tolerance,
    /// heartbeats for liveness, multi-resolution.
    pub fn distributed() -> Self {
        IalConfig {
            epoch_duration_us: 10_000,
            spatial_grid_size: 10,
            sensor_resolution: 0.5,
            max_tokens_per_epoch: 256,
            emit_heartbeats: true,
            heartbeat_interval: 10,
            multi_resolution_layers: vec![
                100,        // 100μs — raw capture
                1_000,      // 1ms — fine motor
                10_000,     // 10ms — perceptual (primary)
                100_000,    // 100ms — action level
                1_000_000,  // 1s — behavioral
            ],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Token Stream hash — for state verification between nodes
// ─────────────────────────────────────────────────────────────────────────────

/// Rolling hash of a token stream. Two streams that have processed the
/// same tokens in the same canonical order will have the same hash.
///
/// Uses FNV-1a for simplicity and speed. Not cryptographic — use SHA-256
/// for adversarial verification contexts.
#[derive(Debug, Clone)]
pub struct StreamHash {
    state: u64,
    count: u64,
}

impl StreamHash {
    pub fn new() -> Self {
        StreamHash {
            state: 0xcbf29ce484222325,
            count: 0,
        }
    }

    /// Incorporate a token into the rolling hash.
    pub fn feed(&mut self, token: &Token) {
        let bytes = token.to_bytes();
        for &b in &bytes {
            self.state ^= b as u64;
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
        self.count += 1;
    }

    /// Current hash value.
    pub fn hash_value(&self) -> u64 {
        self.state
    }

    /// Number of tokens hashed.
    pub fn token_count(&self) -> u64 {
        self.count
    }

    /// Digest: (hash, count) pair for comparison.
    pub fn digest(&self) -> (u64, u64) {
        (self.state, self.count)
    }
}

impl Default for StreamHash {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_quantization_collapses_jitter() {
        let epoch_us = 10_000; // 10ms

        // Two events 3μs apart → same epoch
        let e1 = Epoch::from_timestamp_us(1_000_000, epoch_us);
        let e2 = Epoch::from_timestamp_us(1_000_003, epoch_us);
        assert_eq!(e1, e2);

        // Two events 15ms apart → different epochs
        let e3 = Epoch::from_timestamp_us(1_015_000, epoch_us);
        assert_ne!(e1, e3);
    }

    #[test]
    fn token_serialization_roundtrip() {
        let tok = Token::new(
            Epoch(42),
            Channel::Keyboard,
            EVT_KEY_DOWN,
            Payload::from_byte(0x62), // 'b'
        );
        let bytes = tok.to_bytes();
        let restored = Token::from_bytes(&bytes).unwrap();
        assert_eq!(tok, restored);
    }

    #[test]
    fn canonical_ordering_is_deterministic() {
        let mut tokens = vec![
            Token::new(Epoch(1), Channel::Mouse, EVT_MOUSE_CLICK, Payload::ZERO),
            Token::new(Epoch(1), Channel::Keyboard, EVT_KEY_DOWN, Payload::from_byte(0x62)),
            Token::new(Epoch(0), Channel::Keyboard, EVT_KEY_DOWN, Payload::from_byte(0x70)),
        ];

        tokens.sort();

        // Epoch 0 first, then epoch 1 keyboard before mouse
        assert_eq!(tokens[0].epoch, Epoch(0));
        assert_eq!(tokens[1].channel, Channel::Keyboard);
        assert_eq!(tokens[2].channel, Channel::Mouse);
    }

    #[test]
    fn stream_hash_is_order_dependent() {
        let t1 = Token::new(Epoch(0), Channel::Keyboard, EVT_KEY_DOWN, Payload::from_byte(0x61));
        let t2 = Token::new(Epoch(0), Channel::Keyboard, EVT_KEY_DOWN, Payload::from_byte(0x62));

        let mut h1 = StreamHash::new();
        h1.feed(&t1);
        h1.feed(&t2);

        let mut h2 = StreamHash::new();
        h2.feed(&t2);
        h2.feed(&t1);

        // Different order → different hash (this is intentional:
        // the canonicalizer ensures order before hashing)
        assert_ne!(h1.hash_value(), h2.hash_value());
    }

    #[test]
    fn stream_hash_identical_for_identical_streams() {
        let tokens = vec![
            Token::new(Epoch(0), Channel::Keyboard, EVT_KEY_DOWN, Payload::from_byte(0x62)),
            Token::new(Epoch(1), Channel::Keyboard, EVT_KEY_DOWN, Payload::from_byte(0x6F)),
            Token::new(Epoch(2), Channel::Keyboard, EVT_KEY_DOWN, Payload::from_byte(0x78)),
        ];

        let mut h1 = StreamHash::new();
        let mut h2 = StreamHash::new();
        for t in &tokens {
            h1.feed(t);
            h2.feed(t);
        }

        assert_eq!(h1.digest(), h2.digest());
    }

    #[test]
    fn config_hash_sensitive_to_changes() {
        let c1 = IalConfig::default();
        let mut c2 = IalConfig::default();
        c2.epoch_duration_us = 20_000;

        assert_ne!(c1.config_hash(), c2.config_hash());
    }
}
