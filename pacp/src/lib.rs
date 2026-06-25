//! DNOS PAC-Privacy layer (Privacy Track M1).
//!
//! A calibrated, oracle-logged perturbation on the neural substrate's output.
//! It bounds how much an observer of the *released* output can reconstruct
//! about the private keystroke history, while preserving the system's core
//! guarantee that replay/verify are bit-exact.
//!
//! # The mechanism (and why it doesn't break determinism)
//!
//! DNOS is engineered so that a peer holding the NDAL replay log can
//! reconstruct execution bit-for-bit. Naive output noise would desync every
//! node. The resolution: draw every noise sample from the **NDAL Random
//! oracle**, so the draw is recorded in the replay log.
//!
//! - **Replay / Verify** read the same draw from the log ⇒ identical noise ⇒
//!   execution stays bit-exact.
//! - **Live, fresh** observation sees genuinely fresh noise ⇒ an observer
//!   *without the log* cannot reconstruct the private history.
//!
//! Privacy is therefore conditional on who holds the log — which makes the log
//! itself a secret to protect (Privacy Track M4), orthogonal to this layer.
//!
//! # What is and isn't perturbed
//!
//! - The **command** is the argmax over outputs `[0..20]` — the overt action.
//!   It is released **exactly**; this layer never touches those logits.
//! - The **cursor delta** (outputs 20 and 22) is real-valued and was *not*
//!   trained to be history-invariant (it is the HIGH-leakage channel measured
//!   by M0). This is the channel we noise.
//!
//! # Pipeline
//!
//! ```text
//!   LeakageTable (offline, M0)  ─┐
//!                                ├─► PrivacyRiskEngine ─► σ per channel
//!   PrivacyBudget (SNR target)  ─┘
//!                                          │
//!   outputs[0..32] ──► PrivacyLayer.release ──► perturb (NDAL-logged noise)
//!                                          │      command logits untouched
//!                                          ▼
//!                                   ReleaseOutcome
//! ```

pub mod table;
pub mod risk;
pub mod perturb;
pub mod history;

pub use table::{LeakageTable, CURSOR_DX, CURSOR_DY, OUTPUT_MIDPOINT, OUTPUT_SIZE};
pub use risk::{PrivacyBudget, PrivacyRiskEngine};
pub use history::{QueryHistory, BudgetExhausted};

use dnos_ndal::{Epoch, NdalPipeline};

/// Default per-principal input-history window for the tracker.
const HISTORY_WINDOW: usize = 64;

/// What happened to a release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseOutcome {
    /// Cursor channel was perturbed with NDAL-logged noise.
    Perturbed,
    /// Perturbation is disabled by the budget; output released unchanged.
    Exact,
    /// Failed closed: cursor delta forced to neutral (no movement, no leak)
    /// because the principal's release budget was exhausted.
    Suppressed,
}

/// A release could not be performed safely. Returned (not swallowed) so the
/// caller must decide — the layer never silently degrades.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivacyError {
    /// The NDAL Random oracle yielded no token (rate-limited or disabled), so
    /// no logged noise could be drawn. Continuing would either leak (release
    /// un-noised) or desync replay (suppress without a logged draw), so this is
    /// surfaced as a hard error. The NDAL config must budget at least one
    /// Random query per privacy release per epoch (see `NdalConfig`).
    EntropyUnavailable,
    /// The leakage table was measured on a different weight blob than the one
    /// in use (CRC32 mismatch). Sizing noise to the wrong signal would silently
    /// under- or over-noise, so construction fails closed.
    WeightsMismatch { table: u32, weights: u32 },
}

/// The PAC-Privacy layer: holds the calibrated σ and per-principal budget.
pub struct PrivacyLayer {
    budget: PrivacyBudget,
    sigma_dx: i64,
    sigma_dy: i64,
    history: QueryHistory,
    /// CRC32 of the weights the leakage table was measured on (for checking).
    table_crc32: u32,
}

impl PrivacyLayer {
    /// Build a layer from an offline leakage table and a budget. σ is computed
    /// once here (integer, deterministic) from the table's measured variances.
    ///
    /// Prefer [`PrivacyLayer::for_weights`] in any context where the running
    /// weights' CRC32 is known: it fails closed on a stale table rather than
    /// silently sizing noise to the wrong signal.
    pub fn new(table: &LeakageTable, budget: PrivacyBudget) -> Self {
        let sigma_dx = PrivacyRiskEngine::sigma_output_units(table.within_var_out2_dx, &budget);
        let sigma_dy = PrivacyRiskEngine::sigma_output_units(table.within_var_out2_dy, &budget);
        PrivacyLayer {
            budget,
            sigma_dx,
            sigma_dy,
            history: QueryHistory::new(HISTORY_WINDOW),
            table_crc32: table.weights_crc32,
        }
    }

    /// Build a layer, verifying the table was measured on the weights now in
    /// use. Fails closed on mismatch so a stale `leakage.tbl` cannot silently
    /// mis-size the noise. `weights_crc32` is the CRC32 from the weight-blob
    /// header (the same value `tools/leakage_eval.py` bakes into the table).
    pub fn for_weights(
        table: &LeakageTable,
        budget: PrivacyBudget,
        weights_crc32: u32,
    ) -> Result<Self, PrivacyError> {
        if table.weights_crc32 != weights_crc32 {
            return Err(PrivacyError::WeightsMismatch {
                table: table.weights_crc32,
                weights: weights_crc32,
            });
        }
        Ok(Self::new(table, budget))
    }

    /// Re-seed a principal's spent-release count (snapshot restore / mid-stream
    /// replay). See [`QueryHistory::restore`].
    pub fn restore_releases(&mut self, principal: u64, releases_spent: u64) {
        self.history.restore(principal, releases_spent);
    }

    /// The per-channel noise σ (output units) this layer applies.
    pub fn sigma(&self) -> (i64, i64) {
        (self.sigma_dx, self.sigma_dy)
    }

    /// The leakage bound (bits) the configured budget targets.
    pub fn target_bits(&self) -> f64 {
        self.budget.target_bits()
    }

    /// CRC32 of the weights this layer's table was measured on.
    pub fn table_crc32(&self) -> u32 {
        self.table_crc32
    }

    /// Releases consumed by `principal` so far.
    pub fn releases_spent(&self, principal: u64) -> u64 {
        self.history.spent(principal)
    }

    /// Perturb (or suppress) the cursor channel of one inference's output.
    ///
    /// `outputs` is the substrate's 32-element output vector (the piecewise-
    /// sigmoid range, as the kernel produces it). `epoch`/`principal` scope the
    /// NDAL draw and the budget; `input_digest` records the (private) input in
    /// the history window for composition accounting.
    ///
    /// The command logits `[0..20]` are never modified.
    ///
    /// ## Determinism contract
    ///
    /// When perturbation is enabled this draws the oracle **exactly once per
    /// release, before any budget branch**. So the number and order of NDAL
    /// draws is a pure function of the release sequence — independent of the
    /// budget meter, the rate limiter, or any other runtime-divergent state.
    /// That is what keeps replay log-aligned: a budget-exhausted release still
    /// draws (and discards) its sample, so live and replay consume the log in
    /// lock-step. The only state that gates perturb-vs-suppress is the budget
    /// count, which both sides compute identically from the same sequence
    /// (restore it on a mid-stream resume via [`Self::restore_releases`]).
    ///
    /// Returns `Err(EntropyUnavailable)` if the oracle yields no token, rather
    /// than silently leaking or desyncing — the caller must handle it.
    pub fn release(
        &mut self,
        outputs: &mut [i32; OUTPUT_SIZE],
        ndal: &mut NdalPipeline,
        epoch: Epoch,
        principal: u64,
        input_digest: u64,
    ) -> Result<ReleaseOutcome, PrivacyError> {
        if !self.budget.perturb_cursor {
            return Ok(ReleaseOutcome::Exact);
        }

        // Draw first, unconditionally: one logged oracle query per release, so
        // the log stays aligned across live/replay no matter what follows.
        let token = ndal.random(epoch, 8).ok_or(PrivacyError::EntropyUnavailable)?;
        let p = token.payload;
        let u_dx = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
        let u_dy = u32::from_le_bytes([p[4], p[5], p[6], p[7]]);

        // Charge the budget (deterministic from the release sequence). On
        // exhaustion, fail closed: the draw above is discarded and the cursor
        // is frozen — but the log entry was still produced, so replay matches.
        if self
            .history
            .charge(principal, self.budget.max_releases, input_digest)
            .is_err()
        {
            suppress(outputs);
            return Ok(ReleaseOutcome::Suppressed);
        }

        perturb::apply(outputs, u_dx, u_dy, self.sigma_dx, self.sigma_dy);
        Ok(ReleaseOutcome::Perturbed)
    }
}

/// Fail-closed: force the cursor delta to neutral so the decoded movement is
/// zero — no cursor motion, and nothing about private history is released.
fn suppress(outputs: &mut [i32; OUTPUT_SIZE]) {
    outputs[CURSOR_DX] = OUTPUT_MIDPOINT;
    outputs[CURSOR_DY] = OUTPUT_MIDPOINT;
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnos_ndal::NdalConfig;

    fn table() -> LeakageTable {
        // ~ the M0 numbers: dx ±0.37px, dy ±0.48px of private-history jitter.
        LeakageTable::from_px2(0x1234_5678, 0.137, 0.230)
    }

    #[test]
    fn exact_when_perturb_disabled() {
        let mut b = PrivacyBudget::from_target_bits(0.5, u64::MAX);
        b.perturb_cursor = false;
        let mut layer = PrivacyLayer::new(&table(), b);
        let mut ndal = NdalPipeline::new(NdalConfig::default());
        ndal.seed_random(7);

        let mut out = [0i32; OUTPUT_SIZE];
        out[CURSOR_DX] = 16384;
        out[CURSOR_DY] = 16384;
        let before = out;
        let outcome = layer.release(&mut out, &mut ndal, Epoch(0), 1, 0xab).unwrap();
        assert_eq!(outcome, ReleaseOutcome::Exact);
        assert_eq!(out, before);
    }

    #[test]
    fn suppresses_when_budget_exhausted() {
        let budget = PrivacyBudget::from_target_bits(0.5, 0); // 0 releases allowed
        let mut layer = PrivacyLayer::new(&table(), budget);
        let mut ndal = NdalPipeline::new(NdalConfig::default());
        ndal.seed_random(7);

        let mut out = [0i32; OUTPUT_SIZE];
        out[CURSOR_DX] = 20000;
        out[CURSOR_DY] = 9000;
        let outcome = layer.release(&mut out, &mut ndal, Epoch(0), 1, 0xab).unwrap();
        assert_eq!(outcome, ReleaseOutcome::Suppressed);
        assert_eq!(out[CURSOR_DX], OUTPUT_MIDPOINT);
        assert_eq!(out[CURSOR_DY], OUTPUT_MIDPOINT);
    }

    #[test]
    fn perturbs_and_leaves_command_exact() {
        let budget = PrivacyBudget::from_target_bits(0.5, u64::MAX);
        let mut layer = PrivacyLayer::new(&table(), budget);
        let mut ndal = NdalPipeline::new(NdalConfig::default());
        ndal.seed_random(7);

        let mut out = [0i32; OUTPUT_SIZE];
        for i in 0..20 {
            out[i] = 1000 + i as i32; // argmax at index 19
        }
        out[CURSOR_DX] = 16384;
        out[CURSOR_DY] = 16384;
        let before = out;
        let outcome = layer.release(&mut out, &mut ndal, Epoch(0), 1, 0xab).unwrap();
        assert_eq!(outcome, ReleaseOutcome::Perturbed);
        assert_eq!(out[..20], before[..20], "command logits must be exact");
    }

    #[test]
    fn for_weights_rejects_stale_table() {
        let budget = PrivacyBudget::from_target_bits(0.5, u64::MAX);
        // table() is keyed to crc 0x1234_5678; ask for different weights.
        let res = PrivacyLayer::for_weights(&table(), budget.clone(), 0xDEAD_BEEF);
        assert!(matches!(
            res,
            Err(PrivacyError::WeightsMismatch { table: 0x1234_5678, weights: 0xDEAD_BEEF })
        ));
        // matching crc constructs fine
        assert!(PrivacyLayer::for_weights(&table(), budget, 0x1234_5678).is_ok());
    }

    #[test]
    fn missing_entropy_fails_loud_not_silent() {
        // One Random query per epoch; the second draw in the same epoch is
        // rate-limited (random() == None) and must surface as a hard error,
        // never a silent suppress (which would desync replay).
        let cfg = NdalConfig { max_queries_per_epoch: 1, ..NdalConfig::default() };
        let mut layer = PrivacyLayer::new(&table(), PrivacyBudget::from_target_bits(0.5, u64::MAX));
        let mut ndal = NdalPipeline::new(cfg);
        ndal.seed_random(7);

        let mut out = [0i32; OUTPUT_SIZE];
        out[CURSOR_DX] = 16384;
        out[CURSOR_DY] = 16384;
        assert_eq!(
            layer.release(&mut out, &mut ndal, Epoch(0), 1, 0x1),
            Ok(ReleaseOutcome::Perturbed)
        );
        assert_eq!(
            layer.release(&mut out, &mut ndal, Epoch(0), 1, 0x2),
            Err(PrivacyError::EntropyUnavailable),
        );
    }
}
