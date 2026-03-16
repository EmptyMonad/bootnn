//! DNOS Input Abstraction Layer
//!
//! The determinism membrane between physical reality and the neural substrate.
//!
//! Transforms raw, jitter-laden, non-deterministically-ordered observations
//! into canonical, hashable, replayable token streams.
//!
//! # Architecture
//!
//! ```text
//! RawObservation → [Temporal Quantizer] → [Semantic Encoder] → [Canonicalizer] → Token
//!                  [Spatial Quantizer] ↗
//! ```
//!
//! # Guarantees
//!
//! - Same physical events → same token stream (regardless of μs timing)
//! - Same epoch events → same canonical order (regardless of arrival order)
//! - Stream hash verification (two nodes can compare digests)
//! - Fixed-size token encoding (19 bytes, totally ordered)
//!
//! # Usage
//!
//! ```ignore
//! use dnos_ial::{Pipeline, IalConfig, RawObservation};
//!
//! let mut ial = Pipeline::new(IalConfig::default());
//! let tokens = ial.process(observation);
//! let digest = ial.stream_digest();
//! ```

pub mod types;
pub mod quantizers;
pub mod pipeline;

// Re-exports for convenience
pub use types::*;
pub use quantizers::{RawObservation, RawEventData};
pub use pipeline::{Pipeline, PipelineStats, TokenEncoder};
