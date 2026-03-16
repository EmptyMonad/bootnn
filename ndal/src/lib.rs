//! DNOS Non-Determinism Abstraction Layer
//!
//! Contains irreducible non-determinism without losing the deterministic
//! guarantee. Every interaction with the non-deterministic universe goes
//! through a named oracle, is recorded in a hash-chained replay log, and
//! is returned to the neural substrate as a deterministic token.
//!
//! # Architecture
//!
//! ```text
//! Neural Substrate
//!       ↑ OracleTokens
//!       │
//! ┌─────────────────────────────────────────────┐
//! │                NDAL Pipeline                 │
//! │                                             │
//! │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │
//! │  │Random│ │Clock │ │ Net  │ │ Env  │ ...   │
//! │  │Oracle│ │Oracle│ │Oracle│ │Oracle│        │
//! │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘      │
//! │     └────┬───┘────┬───┘────┬───┘           │
//! │          ▼        ▼        ▼                │
//! │     ┌──────────────────────────┐            │
//! │     │      Replay Log          │            │
//! │     │ (hash-chained, prunable) │            │
//! │     └──────────────────────────┘            │
//! └─────────────────────────────────────────────┘
//!       ↑ raw queries
//!       │
//! Physical Universe (RNG, network, clocks, sensors)
//! ```
//!
//! # Modes
//!
//! - **Live**: Oracles query the real universe; responses are logged.
//! - **Replay**: Oracles read from log; execution is deterministic.
//! - **Verify**: Both; divergences are flagged.
//!
//! # Guarantees
//!
//! - Same log → same oracle tokens → same neural substrate behavior
//! - Hash-chained log is tamper-evident
//! - Divergence between nodes detectable by comparing chain hashes
//! - Snapshot + partial log sufficient for new node sync

pub mod types;
pub mod log;
pub mod oracles;
pub mod pipeline;

pub use types::*;
pub use log::{ReplayLog, LogMode, Divergence, Snapshot, LogStats};
pub use oracles::{Oracle, OracleRegistry, RandomOracle, ClockOracle, EnvironmentOracle};
pub use pipeline::{NdalPipeline, NdalStats};
