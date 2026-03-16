//! DNOS Input Abstraction Layer — Pipeline
//!
//! The complete transform: RawObservation → deterministic Token stream.
//!
//! This is the membrane between physical reality and the DNOS neural substrate.
//! Everything that enters the pipeline comes out as a canonical, hashable,
//! replayable sequence of tokens.

use crate::types::*;
use crate::quantizers::*;

/// The IAL pipeline. Accepts raw observations, emits deterministic tokens.
///
/// Usage:
/// ```ignore
/// let mut ial = Pipeline::new(IalConfig::default());
///
/// // Feed raw observations from the system observer
/// let tokens = ial.process(observation);
///
/// // At the end of processing, flush remaining buffered state
/// let remaining = ial.flush();
///
/// // Verify determinism across nodes
/// let (hash, count) = ial.stream_digest();
/// ```
pub struct Pipeline {
    config: IalConfig,
    temporal: TemporalQuantizer,
    spatial: SpatialQuantizer,
    semantic: SemanticEncoder,
    canonicalizer: Canonicalizer,
    /// Total tokens emitted
    tokens_emitted: u64,
    /// Total observations processed
    observations_processed: u64,
    /// Multi-resolution pipelines (optional)
    multi_res: Vec<MultiResLayer>,
}

/// A secondary temporal resolution layer.
/// Operates on the same observations but with a different epoch duration.
struct MultiResLayer {
    epoch_duration_us: u64,
    canonicalizer: Canonicalizer,
    /// Prefix applied to tokens from this layer so the neural substrate
    /// can distinguish which resolution a token came from.
    layer_id: u8,
}

impl Pipeline {
    pub fn new(config: IalConfig) -> Self {
        let temporal = TemporalQuantizer::new(&config);
        let spatial = SpatialQuantizer::new(&config);
        let semantic = SemanticEncoder::new(&config);
        let canonicalizer = Canonicalizer::new(&config);

        // Build multi-resolution layers
        let multi_res = config.multi_resolution_layers.iter()
            .enumerate()
            .filter(|(_, &dur)| dur != config.epoch_duration_us)
            .map(|(i, &dur)| {
                let layer_config = IalConfig {
                    epoch_duration_us: dur,
                    max_tokens_per_epoch: config.max_tokens_per_epoch,
                    ..config.clone()
                };
                MultiResLayer {
                    epoch_duration_us: dur,
                    canonicalizer: Canonicalizer::new(&layer_config),
                    layer_id: (i + 1) as u8,
                }
            })
            .collect();

        Pipeline {
            config,
            temporal,
            spatial,
            semantic,
            canonicalizer,
            tokens_emitted: 0,
            observations_processed: 0,
            multi_res,
        }
    }

    /// Process a single raw observation through the full pipeline.
    ///
    /// Returns zero or more canonical tokens. Tokens may be buffered
    /// (e.g., word accumulation) and emitted on subsequent calls.
    ///
    /// The returned tokens are in canonical order for their epoch.
    /// When an epoch boundary is crossed, the previous epoch's tokens
    /// are flushed in sorted order.
    pub fn process(&mut self, obs: RawObservation) -> Vec<Token> {
        self.observations_processed += 1;

        // Step 1: Temporal quantization
        let epoch = self.temporal.quantize(obs.timestamp_us);

        // Step 2: Semantic encoding (includes spatial quantization)
        let tokens = self.semantic.encode(epoch, &obs, &self.spatial);

        // Step 3: Canonicalization
        let mut output = Vec::new();
        for token in &tokens {
            let flushed = self.canonicalizer.push(*token);
            output.extend(flushed);
        }

        // Step 4: Multi-resolution layers
        for layer in &mut self.multi_res {
            let layer_epoch = Epoch::from_timestamp_us(
                obs.timestamp_us,
                layer.epoch_duration_us,
            );
            for token in &tokens {
                // Remap token to this layer's epoch
                let remapped = Token::new(
                    layer_epoch,
                    token.channel,
                    token.event_type,
                    token.payload,
                );
                let flushed = layer.canonicalizer.push(remapped);
                // Tag multi-res tokens so the network knows which layer
                // they came from. We encode the layer_id in the payload's
                // last byte (which is otherwise unused for most events).
                for mut ft in flushed {
                    ft.payload.0[7] = layer.layer_id;
                    output.push(ft);
                }
            }
        }

        self.tokens_emitted += output.len() as u64;
        output
    }

    /// Process a batch of observations. Convenience method.
    pub fn process_batch(&mut self, observations: Vec<RawObservation>) -> Vec<Token> {
        let mut all_tokens = Vec::new();
        for obs in observations {
            all_tokens.extend(self.process(obs));
        }
        all_tokens
    }

    /// Flush all buffered state (semantic encoder + canonicalizer).
    /// Call this at the end of an observation session or at explicit
    /// sync points.
    pub fn flush(&mut self) -> Vec<Token> {
        let epoch = self.canonicalizer_current_epoch();

        // Flush semantic encoder (e.g., partial word buffer)
        let semantic_tokens = self.semantic.flush(epoch);
        for token in &semantic_tokens {
            self.canonicalizer.push(*token);
        }

        // Flush main canonicalizer
        let mut output = self.canonicalizer.finish();

        // Flush multi-resolution canonicalizers
        for layer in &mut self.multi_res {
            let mut flushed = layer.canonicalizer.finish();
            for ft in &mut flushed {
                ft.payload.0[7] = layer.layer_id;
            }
            output.extend(flushed);
        }

        self.tokens_emitted += output.len() as u64;
        output
    }

    /// Get the current stream hash digest (hash, token_count).
    /// Two pipelines that have processed the same observations
    /// (regardless of arrival order within epochs) will produce
    /// the same digest.
    pub fn stream_digest(&self) -> (u64, u64) {
        self.canonicalizer.stream_digest()
    }

    /// Get the configuration hash. Must match across nodes for
    /// deterministic equivalence.
    pub fn config_hash(&self) -> u64 {
        self.config.config_hash()
    }

    /// Statistics.
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            observations_processed: self.observations_processed,
            tokens_emitted: self.tokens_emitted,
            compression_ratio: if self.observations_processed > 0 {
                self.tokens_emitted as f64 / self.observations_processed as f64
            } else {
                0.0
            },
            config_hash: self.config.config_hash(),
            stream_digest: self.canonicalizer.stream_digest(),
        }
    }

    fn canonicalizer_current_epoch(&self) -> Epoch {
        // Default to epoch 0 if no tokens have been processed yet
        Epoch(0)
    }
}

/// Pipeline statistics for monitoring and debugging.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub observations_processed: u64,
    pub tokens_emitted: u64,
    pub compression_ratio: f64,
    pub config_hash: u64,
    pub stream_digest: (u64, u64),
}

impl std::fmt::Display for PipelineStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IAL: {} obs → {} tokens ({:.1}x), config={:016X}, stream=({:016X}, {})",
            self.observations_processed,
            self.tokens_emitted,
            self.compression_ratio,
            self.config_hash,
            self.stream_digest.0,
            self.stream_digest.1,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Token → Neural Input encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a token stream into the fixed-size input vector expected by
/// the DNOS neural network.
///
/// This bridges the IAL output to the neural substrate's input layer.
/// The encoding is deterministic: same tokens → same input vector.
pub struct TokenEncoder {
    /// Size of the neural network's input layer
    input_size: usize,
    /// Number of token history slots
    history_depth: usize,
    /// Circular buffer of recent tokens
    history: Vec<Option<Token>>,
    /// Write pointer into history
    write_ptr: usize,
}

impl TokenEncoder {
    /// Create an encoder for a network with `input_size` input neurons
    /// and `history_depth` token history slots.
    ///
    /// Each token occupies `input_size / history_depth` neurons.
    /// For the Tier 2 network (256 inputs, 8 history slots), each token
    /// gets 32 neurons.
    pub fn new(input_size: usize, history_depth: usize) -> Self {
        assert!(input_size >= history_depth, "input_size must be >= history_depth");
        assert!(input_size % history_depth == 0, "input_size must be divisible by history_depth");

        TokenEncoder {
            input_size,
            history_depth,
            history: vec![None; history_depth],
            write_ptr: 0,
        }
    }

    /// Push a token into history and return the encoded input vector.
    ///
    /// The input vector is in Q8.8 fixed-point format (i16 values)
    /// matching the assembly implementation.
    pub fn push(&mut self, token: &Token) -> Vec<i16> {
        self.history[self.write_ptr] = Some(*token);
        self.write_ptr = (self.write_ptr + 1) % self.history_depth;
        self.encode()
    }

    /// Encode current history to input vector.
    fn encode(&self) -> Vec<i16> {
        let slot_size = self.input_size / self.history_depth;
        let mut input = vec![0i16; self.input_size];

        for slot in 0..self.history_depth {
            // Read from history in reverse chronological order
            let idx = (self.write_ptr + self.history_depth - 1 - slot) % self.history_depth;
            let offset = slot * slot_size;

            if let Some(token) = &self.history[idx] {
                self.encode_token_to_slot(token, &mut input[offset..offset + slot_size]);
            }
            // Empty slots remain zero (no event)
        }

        input
    }

    /// Encode a single token into a fixed-size slot.
    ///
    /// Slot layout (for 32-neuron slots):
    ///   [0..8]   - channel one-hot (8 channels)
    ///   [8..10]  - event_type high bits
    ///   [10..18] - payload bytes as normalized values
    ///   [18..26] - payload bytes as binary decomposition
    ///   [26..32] - reserved / zero
    fn encode_token_to_slot(&self, token: &Token, slot: &mut [i16]) {
        let q88_max: i16 = 256; // 1.0 in Q8.8

        // Channel one-hot
        if (token.channel as usize) < 8 && (token.channel as usize) < slot.len() {
            slot[token.channel as usize] = q88_max;
        }

        // Event type (normalized to 0..1 range, split into two neurons)
        if slot.len() > 9 {
            slot[8] = ((token.event_type.0 >> 8) as i16).min(q88_max);
            slot[9] = ((token.event_type.0 & 0xFF) as i16).min(q88_max);
        }

        // Payload: direct byte values normalized to 0..256 (Q8.8)
        for i in 0..8 {
            if 10 + i < slot.len() {
                slot[10 + i] = token.payload.0[i] as i16;
            }
        }

        // Payload: binary decomposition of first 8 bytes
        // Each bit of the payload maps to one neuron (0 or 256)
        if slot.len() >= 26 {
            for i in 0..8 {
                let byte = token.payload.0[i];
                // Pack 1 bit per neuron for the first byte
                // (we only have room for 8 bits in neurons 18-25)
                if 18 + i < slot.len() {
                    slot[18 + i] = if byte > 127 { q88_max } else { 0 };
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key_obs(timestamp_us: u64, scancode: u8) -> RawObservation {
        RawObservation {
            timestamp_us,
            channel: Channel::Keyboard,
            data: RawEventData::Key { scancode, pressed: true },
        }
    }

    #[test]
    fn pipeline_deterministic_across_instances() {
        let config = IalConfig::default();

        let observations = vec![
            make_key_obs(1_000_000, b'b'),
            make_key_obs(1_000_003, b'o'),  // 3μs later, same epoch
            make_key_obs(1_010_500, b'x'),  // Different epoch
        ];

        // Run through two independent pipelines
        let mut p1 = Pipeline::new(config.clone());
        let mut p2 = Pipeline::new(config);

        let mut tokens1 = Vec::new();
        let mut tokens2 = Vec::new();

        for obs in &observations {
            tokens1.extend(p1.process(obs.clone()));
            tokens2.extend(p2.process(obs.clone()));
        }

        tokens1.extend(p1.flush());
        tokens2.extend(p2.flush());

        assert_eq!(tokens1, tokens2, "Two pipelines must produce identical token streams");
        assert_eq!(p1.stream_digest(), p2.stream_digest(), "Stream hashes must match");
    }

    #[test]
    fn pipeline_absorbs_jitter() {
        let config = IalConfig {
            epoch_duration_us: 10_000,
            ..IalConfig::default()
        };

        // Node A: events at t=1000000, t=1000003
        let obs_a = vec![
            make_key_obs(1_000_000, b'b'),
            make_key_obs(1_000_003, b'o'),
        ];

        // Node B: same events at t=1000002, t=1000005 (2μs shifted)
        let obs_b = vec![
            make_key_obs(1_000_002, b'b'),
            make_key_obs(1_000_005, b'o'),
        ];

        let mut pa = Pipeline::new(config.clone());
        let mut pb = Pipeline::new(config);

        let mut tokens_a = Vec::new();
        let mut tokens_b = Vec::new();

        for obs in obs_a { tokens_a.extend(pa.process(obs)); }
        for obs in obs_b { tokens_b.extend(pb.process(obs)); }

        tokens_a.extend(pa.flush());
        tokens_b.extend(pb.flush());

        assert_eq!(tokens_a, tokens_b,
            "Same events with microsecond jitter must produce identical tokens");
    }

    #[test]
    fn pipeline_different_arrival_order_same_epoch() {
        let config = IalConfig::default();

        // Same three events, same epoch, different arrival order
        let obs_set_a = vec![
            make_key_obs(1_000_000, b'z'),
            make_key_obs(1_000_001, b'a'),
            make_key_obs(1_000_002, b'm'),
        ];

        let obs_set_b = vec![
            make_key_obs(1_000_002, b'm'),
            make_key_obs(1_000_000, b'z'),
            make_key_obs(1_000_001, b'a'),
        ];

        let mut pa = Pipeline::new(config.clone());
        let mut pb = Pipeline::new(config);

        let mut ta = Vec::new();
        let mut tb = Vec::new();

        for obs in obs_set_a { ta.extend(pa.process(obs)); }
        for obs in obs_set_b { tb.extend(pb.process(obs)); }

        ta.extend(pa.flush());
        tb.extend(pb.flush());

        // After canonicalization, both should be sorted identically
        assert_eq!(ta, tb,
            "Different arrival order within same epoch must produce identical output");
    }

    #[test]
    fn token_encoder_produces_fixed_size_output() {
        let mut encoder = TokenEncoder::new(256, 8);

        let token = Token::new(
            Epoch(0),
            Channel::Keyboard,
            EVT_KEY_DOWN,
            Payload::from_byte(b'p'),
        );

        let input = encoder.push(&token);
        assert_eq!(input.len(), 256);

        // Keyboard channel should be hot in first slot
        assert_eq!(input[0], 256); // Channel::Keyboard = 0, Q8.8 1.0 = 256
    }

    #[test]
    fn compression_ratio_for_mouse_jitter() {
        let config = IalConfig {
            spatial_grid_size: 10,
            ..IalConfig::default()
        };

        let mut pipeline = Pipeline::new(config);

        // Simulate 100 mouse move events within the same grid cell
        for i in 0..100u64 {
            let obs = RawObservation {
                timestamp_us: 1_000_000 + i * 10,
                channel: Channel::Mouse,
                data: RawEventData::MouseMove { x: 155 + (i as i32 % 5), y: 93 },
            };
            pipeline.process(obs);
        }

        let remaining = pipeline.flush();
        let stats = pipeline.stats();

        // 100 observations should compress to very few tokens
        // (1 initial move + maybe a few grid boundary crossings)
        assert!(stats.tokens_emitted < 10,
            "100 mouse jitter events in same grid cell should compress heavily, got {}",
            stats.tokens_emitted);
    }
}
