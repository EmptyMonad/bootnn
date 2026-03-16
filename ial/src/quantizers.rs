//! DNOS Input Abstraction Layer — Quantizers
//!
//! Three quantizers that transform continuous, noisy physical observations
//! into discrete, deterministic values:
//!
//! - TemporalQuantizer: collapses microsecond timestamps into epochs
//! - SpatialQuantizer: buckets continuous coordinates into grid cells
//! - SemanticEncoder: maps raw events to higher-level semantic tokens
//!
//! The key invariant: two quantizer instances with the same configuration
//! produce bit-identical outputs for the same inputs, regardless of the
//! order in which inputs are presented (ordering is handled by the
//! Canonicalizer, not the quantizers).

use crate::types::*;

// ─────────────────────────────────────────────────────────────────────────────
// Raw Observation — what the observer produces
// ─────────────────────────────────────────────────────────────────────────────

/// A raw observation from the system observer.
/// This is the input to the IAL pipeline. It contains physical timestamps
/// and unquantized values — everything the IAL will strip away.
#[derive(Debug, Clone)]
pub struct RawObservation {
    /// Microsecond timestamp (CLOCK_MONOTONIC or equivalent)
    pub timestamp_us: u64,
    /// Source channel
    pub channel: Channel,
    /// Raw event data
    pub data: RawEventData,
}

/// Raw event data, channel-specific.
#[derive(Debug, Clone)]
pub enum RawEventData {
    /// Keyboard: scancode + press/release
    Key { scancode: u8, pressed: bool },
    /// Mouse: raw x,y coordinates + optional button
    MouseMove { x: i32, y: i32 },
    MouseButton { button: u8, pressed: bool, x: i32, y: i32 },
    MouseScroll { delta: i32 },
    /// File system: path + operation
    FileEvent { path_hash: u32, event_type: EventType },
    /// Process: pid + event
    ProcessEvent { pid: u32, event_type: EventType },
    /// Sensor: channel_id + raw value
    Sensor { sensor_id: u8, value: f64 },
    /// Heartbeat (no data)
    Heartbeat,
}

// ─────────────────────────────────────────────────────────────────────────────
// Temporal Quantizer
// ─────────────────────────────────────────────────────────────────────────────

/// Collapses microsecond timestamps into discrete epochs.
///
/// This is where jitter dies. Events separated by microseconds but within
/// the same epoch boundary become simultaneous. The network never knows
/// which arrived first.
pub struct TemporalQuantizer {
    epoch_duration_us: u64,
    /// Track the current epoch for heartbeat generation.
    last_epoch: Option<Epoch>,
    /// Count of empty epochs since last event (for heartbeat spacing).
    empty_epoch_count: u64,
    heartbeat_interval: u64,
    emit_heartbeats: bool,
}

impl TemporalQuantizer {
    pub fn new(config: &IalConfig) -> Self {
        TemporalQuantizer {
            epoch_duration_us: config.epoch_duration_us,
            last_epoch: None,
            empty_epoch_count: 0,
            heartbeat_interval: config.heartbeat_interval,
            emit_heartbeats: config.emit_heartbeats,
        }
    }

    /// Quantize a timestamp to its epoch.
    pub fn quantize(&mut self, timestamp_us: u64) -> Epoch {
        let epoch = Epoch::from_timestamp_us(timestamp_us, self.epoch_duration_us);

        // Track epoch transitions for heartbeat generation
        if let Some(last) = self.last_epoch {
            if epoch.0 > last.0 + 1 {
                // Gap detected — there were empty epochs between last event
                // and this one. The canonicalizer will insert heartbeats.
                self.empty_epoch_count += epoch.0 - last.0 - 1;
            }
        }
        self.last_epoch = Some(epoch);
        self.empty_epoch_count = 0;
        epoch
    }

    /// Generate heartbeat tokens for empty epochs between `from` and `to`.
    /// Returns heartbeats spaced according to heartbeat_interval.
    pub fn generate_heartbeats(&self, from_epoch: Epoch, to_epoch: Epoch) -> Vec<Token> {
        if !self.emit_heartbeats || from_epoch.0 >= to_epoch.0 {
            return vec![];
        }

        let mut heartbeats = Vec::new();
        let mut e = from_epoch.0 + 1;
        while e < to_epoch.0 {
            if (e - from_epoch.0) % self.heartbeat_interval == 0 {
                heartbeats.push(Token::heartbeat(Epoch(e)));
            }
            e += 1;
        }
        heartbeats
    }

    pub fn epoch_duration_us(&self) -> u64 {
        self.epoch_duration_us
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial Quantizer
// ─────────────────────────────────────────────────────────────────────────────

/// Buckets continuous spatial coordinates into discrete grid cells.
///
/// A mouse at (157, 94) with grid_size=10 becomes bucket (15, 9).
/// This eliminates sub-pixel noise and makes coordinate comparison exact.
pub struct SpatialQuantizer {
    grid_size: u16,
}

impl SpatialQuantizer {
    pub fn new(config: &IalConfig) -> Self {
        SpatialQuantizer {
            grid_size: config.spatial_grid_size,
        }
    }

    /// Quantize a coordinate pair to grid bucket.
    pub fn quantize(&self, x: i32, y: i32) -> (u16, u16) {
        let gx = if x < 0 { 0 } else { (x as u32 / self.grid_size as u32) as u16 };
        let gy = if y < 0 { 0 } else { (y as u32 / self.grid_size as u32) as u16 };
        (gx, gy)
    }

    /// Quantize a sensor value to discrete level.
    pub fn quantize_sensor(&self, value: f64, resolution: f64) -> i32 {
        if resolution <= 0.0 {
            return value as i32;
        }
        (value / resolution).round() as i32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Semantic Encoder
// ─────────────────────────────────────────────────────────────────────────────

/// Maps raw events to semantic tokens, performing behavioral abstraction.
///
/// This is where "user pressed b, o, x" becomes "user typed word 'box'",
/// and "mouse jittered around then clicked" becomes "click at grid (15,9)".
///
/// The initial implementation uses fixed rules. Future versions could
/// use a meta-network to learn useful abstractions.
pub struct SemanticEncoder {
    /// Key buffer for word recognition
    key_buffer: Vec<u8>,
    /// Maximum key buffer size before forced flush
    max_word_len: usize,
    /// Last mouse position for movement collapse
    last_mouse_bucket: Option<(u16, u16)>,
    /// Spatial quantizer reference
    grid_size: u16,
}

impl SemanticEncoder {
    pub fn new(config: &IalConfig) -> Self {
        SemanticEncoder {
            key_buffer: Vec::with_capacity(16),
            max_word_len: 8,  // Match DNOS history width
            last_mouse_bucket: None,
            grid_size: config.spatial_grid_size,
        }
    }

    /// Process a raw observation and emit zero or more tokens.
    ///
    /// May buffer events (e.g., accumulating keystrokes into words)
    /// and emit them later via flush().
    pub fn encode(
        &mut self,
        epoch: Epoch,
        obs: &RawObservation,
        spatial: &SpatialQuantizer,
    ) -> Vec<Token> {
        match &obs.data {
            RawEventData::Key { scancode, pressed } => {
                self.encode_key(epoch, *scancode, *pressed)
            }
            RawEventData::MouseMove { x, y } => {
                self.encode_mouse_move(epoch, *x, *y, spatial)
            }
            RawEventData::MouseButton { button, pressed, x, y } => {
                self.encode_mouse_button(epoch, *button, *pressed, *x, *y, spatial)
            }
            RawEventData::MouseScroll { delta } => {
                vec![Token::new(
                    epoch,
                    Channel::Mouse,
                    EVT_MOUSE_SCROLL,
                    Payload::from_byte(if *delta > 0 { 1 } else { 0xFF }),
                )]
            }
            RawEventData::FileEvent { path_hash, event_type } => {
                vec![Token::new(
                    epoch,
                    Channel::FileSystem,
                    *event_type,
                    Payload::from_u32(*path_hash),
                )]
            }
            RawEventData::ProcessEvent { pid, event_type } => {
                vec![Token::new(
                    epoch,
                    Channel::System,
                    *event_type,
                    Payload::from_u32(*pid),
                )]
            }
            RawEventData::Sensor { sensor_id, value } => {
                let quantized = spatial.quantize_sensor(*value, 0.5);
                let mut payload = [0u8; 8];
                payload[0] = *sensor_id;
                payload[1..5].copy_from_slice(&(quantized as i32).to_le_bytes());
                vec![Token::new(
                    epoch,
                    Channel::Sensor,
                    EventType(*sensor_id as u16),
                    Payload(payload),
                )]
            }
            RawEventData::Heartbeat => {
                vec![Token::heartbeat(epoch)]
            }
        }
    }

    fn encode_key(&mut self, epoch: Epoch, scancode: u8, pressed: bool) -> Vec<Token> {
        let mut tokens = Vec::new();

        if pressed {
            // Always emit the raw key_down token
            tokens.push(Token::new(
                epoch,
                Channel::Keyboard,
                EVT_KEY_DOWN,
                Payload::from_byte(scancode),
            ));

            // Buffer for word recognition (printable ASCII only)
            if scancode >= 0x20 && scancode < 0x7F {
                self.key_buffer.push(scancode);

                // Flush if buffer full
                if self.key_buffer.len() >= self.max_word_len {
                    tokens.extend(self.flush_word_buffer(epoch));
                }
            } else {
                // Non-printable key breaks word accumulation
                tokens.extend(self.flush_word_buffer(epoch));
            }
        }
        // Key-up events are elided at the semantic level.
        // The network doesn't need to know about releases for behavioral
        // abstraction. If raw key-up tracking is needed, use a lower
        // abstraction layer.

        tokens
    }

    fn encode_mouse_move(
        &mut self,
        epoch: Epoch,
        x: i32,
        y: i32,
        spatial: &SpatialQuantizer,
    ) -> Vec<Token> {
        let bucket = spatial.quantize(x, y);

        // Only emit if the mouse moved to a different grid cell.
        // This collapses sub-grid jitter into nothing.
        if self.last_mouse_bucket == Some(bucket) {
            return vec![];
        }

        self.last_mouse_bucket = Some(bucket);
        vec![Token::new(
            epoch,
            Channel::Mouse,
            EVT_MOUSE_MOVE,
            Payload::from_u16_pair(bucket.0, bucket.1),
        )]
    }

    fn encode_mouse_button(
        &mut self,
        epoch: Epoch,
        button: u8,
        pressed: bool,
        x: i32,
        y: i32,
        spatial: &SpatialQuantizer,
    ) -> Vec<Token> {
        if !pressed {
            return vec![]; // Elide releases at semantic level
        }

        let bucket = spatial.quantize(x, y);
        self.last_mouse_bucket = Some(bucket);

        let mut payload = [0u8; 8];
        payload[0] = button;
        payload[2..4].copy_from_slice(&bucket.0.to_le_bytes());
        payload[4..6].copy_from_slice(&bucket.1.to_le_bytes());

        vec![Token::new(
            epoch,
            Channel::Mouse,
            EVT_MOUSE_CLICK,
            Payload(payload),
        )]
    }

    /// Flush the key buffer, emitting a WORD token if >= 3 chars accumulated.
    fn flush_word_buffer(&mut self, epoch: Epoch) -> Vec<Token> {
        if self.key_buffer.len() < 3 {
            self.key_buffer.clear();
            return vec![];
        }

        // Hash the word to a fixed-size payload
        let word_hash = fnv1a_bytes(&self.key_buffer);
        let len = self.key_buffer.len() as u8;

        let mut payload = [0u8; 8];
        payload[0..4].copy_from_slice(&word_hash.to_le_bytes());
        payload[4] = len;
        // Store first 3 chars for debugging
        for (i, &ch) in self.key_buffer.iter().take(3).enumerate() {
            payload[5 + i] = ch;
        }

        self.key_buffer.clear();

        vec![Token::new(
            epoch,
            Channel::Keyboard,
            EVT_WORD,
            Payload(payload),
        )]
    }

    /// Force-flush any buffered state. Call at epoch boundaries.
    pub fn flush(&mut self, epoch: Epoch) -> Vec<Token> {
        self.flush_word_buffer(epoch)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Canonicalizer — deterministic ordering of same-epoch tokens
// ─────────────────────────────────────────────────────────────────────────────

/// Collects tokens within an epoch and emits them in canonical order.
///
/// The canonical order is the natural Ord implementation on Token:
/// (epoch, channel, event_type, payload) — pure lexicographic.
///
/// This ensures two nodes that received the same events in the same epoch
/// but in different arrival orders produce identical token streams.
pub struct Canonicalizer {
    /// Buffer for current epoch
    current_epoch: Option<Epoch>,
    buffer: Vec<Token>,
    max_tokens: usize,
    /// Rolling hash of the output stream
    stream_hash: StreamHash,
}

impl Canonicalizer {
    pub fn new(config: &IalConfig) -> Self {
        Canonicalizer {
            current_epoch: None,
            buffer: Vec::with_capacity(config.max_tokens_per_epoch),
            max_tokens: config.max_tokens_per_epoch,
            stream_hash: StreamHash::new(),
        }
    }

    /// Add a token. If it belongs to a new epoch, flush the previous
    /// epoch's tokens first and return them in canonical order.
    pub fn push(&mut self, token: Token) -> Vec<Token> {
        let mut flushed = Vec::new();

        if let Some(current) = self.current_epoch {
            if token.epoch != current {
                // New epoch — flush previous
                flushed = self.flush_buffer();
            }
        }

        self.current_epoch = Some(token.epoch);

        if self.buffer.len() < self.max_tokens {
            self.buffer.push(token);
        }
        // Overflow tokens are silently dropped. In production, emit an
        // overflow token instead. For now, the max_tokens_per_epoch
        // limit is generous enough that this shouldn't happen.

        flushed
    }

    /// Flush the current epoch buffer, returning tokens in canonical order.
    pub fn flush_buffer(&mut self) -> Vec<Token> {
        if self.buffer.is_empty() {
            return vec![];
        }

        // Sort into canonical order
        self.buffer.sort();

        // Deduplicate identical tokens (same event in same epoch)
        self.buffer.dedup();

        // Feed into rolling hash
        for token in &self.buffer {
            self.stream_hash.feed(token);
        }

        let result = std::mem::take(&mut self.buffer);
        self.buffer = Vec::with_capacity(self.max_tokens);
        result
    }

    /// Force flush and return remaining tokens.
    pub fn finish(&mut self) -> Vec<Token> {
        self.flush_buffer()
    }

    /// Get the current stream hash digest.
    pub fn stream_digest(&self) -> (u64, u64) {
        self.stream_hash.digest()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

fn fnv1a_bytes(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> IalConfig {
        IalConfig {
            epoch_duration_us: 10_000,
            spatial_grid_size: 10,
            ..IalConfig::default()
        }
    }

    #[test]
    fn temporal_quantizer_collapses_jitter() {
        let config = make_config();
        let mut tq = TemporalQuantizer::new(&config);

        let e1 = tq.quantize(1_000_000);
        let e2 = tq.quantize(1_000_003); // 3μs later
        assert_eq!(e1, e2, "3μs jitter should collapse to same epoch");
    }

    #[test]
    fn spatial_quantizer_buckets_correctly() {
        let config = make_config();
        let sq = SpatialQuantizer::new(&config);

        assert_eq!(sq.quantize(157, 94), (15, 9));
        assert_eq!(sq.quantize(159, 99), (15, 9)); // Same bucket
        assert_eq!(sq.quantize(160, 100), (16, 10)); // Different bucket
    }

    #[test]
    fn mouse_jitter_within_grid_cell_is_elided() {
        let config = make_config();
        let sq = SpatialQuantizer::new(&config);
        let mut se = SemanticEncoder::new(&config);

        let epoch = Epoch(100);

        // First move: emits token
        let obs1 = RawObservation {
            timestamp_us: 1_000_000,
            channel: Channel::Mouse,
            data: RawEventData::MouseMove { x: 155, y: 93 },
        };
        let t1 = se.encode(epoch, &obs1, &sq);
        assert_eq!(t1.len(), 1, "First move should emit token");

        // Jitter within same grid cell: no token
        let obs2 = RawObservation {
            timestamp_us: 1_000_050,
            channel: Channel::Mouse,
            data: RawEventData::MouseMove { x: 157, y: 94 },
        };
        let t2 = se.encode(epoch, &obs2, &sq);
        assert_eq!(t2.len(), 0, "Jitter within grid cell should be elided");

        // Move to different cell: emits token
        let obs3 = RawObservation {
            timestamp_us: 1_000_100,
            channel: Channel::Mouse,
            data: RawEventData::MouseMove { x: 165, y: 105 },
        };
        let t3 = se.encode(epoch, &obs3, &sq);
        assert_eq!(t3.len(), 1, "Move to new grid cell should emit token");
    }

    #[test]
    fn canonicalizer_sorts_same_epoch_events() {
        let config = make_config();
        let mut canon = Canonicalizer::new(&config);

        let epoch = Epoch(42);

        // Push events in reverse channel order
        canon.push(Token::new(epoch, Channel::Mouse, EVT_MOUSE_CLICK, Payload::ZERO));
        canon.push(Token::new(epoch, Channel::Keyboard, EVT_KEY_DOWN, Payload::from_byte(0x62)));

        // Trigger flush by pushing to a new epoch
        let flushed = canon.push(Token::new(Epoch(43), Channel::Keyboard, EVT_KEY_DOWN, Payload::from_byte(0x70)));

        // Verify canonical order: Keyboard before Mouse
        assert_eq!(flushed.len(), 2);
        assert_eq!(flushed[0].channel, Channel::Keyboard);
        assert_eq!(flushed[1].channel, Channel::Mouse);
    }

    #[test]
    fn two_canonicalizers_produce_identical_output() {
        let config = make_config();
        let epoch = Epoch(1);

        let tokens_node_a = vec![
            Token::new(epoch, Channel::Mouse, EVT_MOUSE_CLICK, Payload::ZERO),
            Token::new(epoch, Channel::Keyboard, EVT_KEY_DOWN, Payload::from_byte(0x62)),
            Token::new(epoch, Channel::FileSystem, EVT_FILE_MODIFY, Payload::from_u32(0xDEAD)),
        ];

        // Node B receives same events but in different order
        let tokens_node_b = vec![
            Token::new(epoch, Channel::FileSystem, EVT_FILE_MODIFY, Payload::from_u32(0xDEAD)),
            Token::new(epoch, Channel::Keyboard, EVT_KEY_DOWN, Payload::from_byte(0x62)),
            Token::new(epoch, Channel::Mouse, EVT_MOUSE_CLICK, Payload::ZERO),
        ];

        let mut canon_a = Canonicalizer::new(&config);
        let mut canon_b = Canonicalizer::new(&config);

        for t in tokens_node_a {
            canon_a.push(t);
        }
        for t in tokens_node_b {
            canon_b.push(t);
        }

        let out_a = canon_a.finish();
        let out_b = canon_b.finish();

        assert_eq!(out_a, out_b, "Same events in different order must produce identical output");
        assert_eq!(canon_a.stream_digest(), canon_b.stream_digest(),
            "Stream hashes must match");
    }

    #[test]
    fn word_recognition_from_keystrokes() {
        let config = make_config();
        let sq = SpatialQuantizer::new(&config);
        let mut se = SemanticEncoder::new(&config);

        let epoch = Epoch(0);
        let mut all_tokens = Vec::new();

        // Type "box"
        for &ch in b"box" {
            let obs = RawObservation {
                timestamp_us: 0,
                channel: Channel::Keyboard,
                data: RawEventData::Key { scancode: ch, pressed: true },
            };
            all_tokens.extend(se.encode(epoch, &obs, &sq));
        }

        // Flush at epoch boundary
        all_tokens.extend(se.flush(epoch));

        // Should have 3 KEY_DOWN tokens + 1 WORD token
        let key_tokens: Vec<_> = all_tokens.iter()
            .filter(|t| t.event_type == EVT_KEY_DOWN)
            .collect();
        let word_tokens: Vec<_> = all_tokens.iter()
            .filter(|t| t.event_type == EVT_WORD)
            .collect();

        assert_eq!(key_tokens.len(), 3);
        assert_eq!(word_tokens.len(), 1);
    }
}
