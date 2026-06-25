//! Privacy Risk Engine — turns a measured signal + a budget into a noise scale.
//!
//! PAC-Privacy bounds how much an observer learns about the private input by
//! adding noise sized to the signal. For one Gaussian channel the leakage bound
//! is `bits = 0.5·log2(1 + SNR)` where `SNR = signal_var / noise_var`. Inverting:
//! `noise_var = signal_var / SNR`, and the target SNR fixes the bits.
//!
//! Determinism note: the σ a node applies must be identical across nodes (or
//! replay/verify diverges). So the *canonical* knob is an **integer SNR ratio**
//! (folded into `config_hash`, like NDAL/IAL config), and σ is derived by
//! **integer** sqrt — no float in the determinism-relevant path. `from_target_
//! bits` is a host-side convenience for picking that ratio from a bits target;
//! the resulting integer ratio is what actually governs behavior.

/// Per-principal privacy budget + the noise calibration knob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrivacyBudget {
    /// Target signal-to-noise ratio = `snr_num / snr_den`. Smaller ⇒ more noise
    /// ⇒ less leakage. This integer ratio is the canonical, hashed knob.
    pub snr_num: u64,
    pub snr_den: u64,
    /// Max sensitive-channel releases per principal before failing closed.
    pub max_releases: u64,
    /// Whether the cursor channel is perturbed at all (command/argmax is always
    /// released exactly, regardless).
    pub perturb_cursor: bool,
}

impl PrivacyBudget {
    /// Canonical constructor: the integer SNR ratio directly. This is the only
    /// value that governs σ and `config_hash`, so distributed nodes MUST agree
    /// on `(snr_num, snr_den)` — exchange the integer ratio over the wire, not a
    /// bits target (see `from_target_bits`).
    pub fn from_snr_ratio(snr_num: u64, snr_den: u64, max_releases: u64) -> Self {
        PrivacyBudget {
            snr_num: snr_num.max(1), // never zero: avoids divide-by-zero, caps noise
            snr_den: snr_den.max(1),
            max_releases,
            perturb_cursor: true,
        }
    }

    /// Pick the SNR ratio from a per-channel leakage target in bits.
    ///
    /// `SNR = 2^(2·bits) − 1`, quantized to /1000. Uses float **here only**
    /// (config time, host), and `powf`/`log2` are not bit-identical across
    /// platforms — so two nodes that each call this independently can land on
    /// different `snr_num` and then disagree on σ and `config_hash`. Run this on
    /// ONE host, then distribute the resulting integer ratio via
    /// `from_snr_ratio`. The integer ratio — never the bits target — is the
    /// determinism-relevant, hashed value.
    pub fn from_target_bits(bits: f64, max_releases: u64) -> Self {
        let snr = (2.0_f64.powf(2.0 * bits) - 1.0).max(0.0);
        let den = 1000u64;
        let num = (snr * den as f64).round() as u64;
        PrivacyBudget::from_snr_ratio(num, den, max_releases)
    }

    /// The leakage bound (bits) this budget targets, for reporting.
    pub fn target_bits(&self) -> f64 {
        0.5 * (1.0 + self.snr_num as f64 / self.snr_den.max(1) as f64).log2()
    }

    /// FNV-1a hash of the budget for cross-node verification (mirrors
    /// NdalConfig::config_hash so mismatched privacy settings are detectable).
    pub fn config_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let prime: u64 = 0x100000001b3;
        let mut feed = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(prime);
            }
        };
        feed(&self.snr_num.to_le_bytes());
        feed(&self.snr_den.to_le_bytes());
        feed(&self.max_releases.to_le_bytes());
        feed(&[self.perturb_cursor as u8]);
        h
    }
}

/// Stateless calibrator: signal variance + budget → integer noise σ.
pub struct PrivacyRiskEngine;

impl PrivacyRiskEngine {
    /// σ (output units) for a channel, fully integer and deterministic.
    ///
    /// `σ² = signal_var · snr_den / snr_num`, then integer sqrt. u128 keeps the
    /// intermediate product from overflowing for realistic variances.
    pub fn sigma_output_units(signal_var_out2: u64, b: &PrivacyBudget) -> i64 {
        let num = b.snr_num.max(1) as u128;
        let sigma2 = (signal_var_out2 as u128) * (b.snr_den as u128) / num;
        // u128::isqrt is stable; result fits i64 for any realistic variance.
        sigma2.isqrt() as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snr_one_means_sigma_equals_signal_std() {
        // SNR = 1 ⇒ noise_var = signal_var ⇒ σ = sqrt(signal_var).
        let b = PrivacyBudget { snr_num: 1, snr_den: 1, max_releases: u64::MAX, perturb_cursor: true };
        let sigma = PrivacyRiskEngine::sigma_output_units(144, &b);
        assert_eq!(sigma, 12); // isqrt(144)
    }

    #[test]
    fn smaller_snr_gives_more_noise() {
        let v = 1_000_000u64;
        let loose = PrivacyBudget { snr_num: 4, snr_den: 1, max_releases: 0, perturb_cursor: true };
        let tight = PrivacyBudget { snr_num: 1, snr_den: 4, max_releases: 0, perturb_cursor: true };
        let s_loose = PrivacyRiskEngine::sigma_output_units(v, &loose);
        let s_tight = PrivacyRiskEngine::sigma_output_units(v, &tight);
        assert!(s_tight > s_loose, "tighter budget (less SNR) must add more noise");
    }

    #[test]
    fn from_target_bits_roundtrips_through_target_bits() {
        for bits in [0.1, 0.5, 1.0, 2.0] {
            let b = PrivacyBudget::from_target_bits(bits, 100);
            assert!((b.target_bits() - bits).abs() < 0.05,
                    "bits {bits} → {} via SNR {}/{}", b.target_bits(), b.snr_num, b.snr_den);
        }
    }

    #[test]
    fn config_hash_is_sensitive() {
        let a = PrivacyBudget::from_target_bits(0.5, 100);
        let mut b = a.clone();
        b.max_releases = 99;
        assert_ne!(a.config_hash(), b.config_hash());
    }
}
