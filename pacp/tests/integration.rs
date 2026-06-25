//! Integration tests for the PAC-Privacy layer (Privacy Track M1).
//!
//! The headline property: privacy noise is drawn through the NDAL Random
//! oracle, so it is recorded in the replay log. Therefore replaying the log
//! reproduces the *same* perturbed output bit-for-bit (determinism preserved),
//! while a fresh live session with a different seed produces different output
//! (a fresh observer cannot reconstruct).

use dnos_ndal::{Epoch, LogMode, NdalConfig, NdalPipeline};
use dnos_pacp::{LeakageTable, PrivacyBudget, PrivacyLayer, ReleaseOutcome, OUTPUT_SIZE,
                CURSOR_DX, CURSOR_DY};

fn table() -> LeakageTable {
    // Approximate the canonical M0 measurement (dx ±0.37px, dy ±0.48px).
    LeakageTable::from_px2(0x1234_5678, 0.137, 0.230)
}

/// Build a deterministic-looking output vector for inference number `k`.
/// The cursor channel carries a value; the command logits have a fixed argmax.
fn make_output(k: i32) -> [i32; OUTPUT_SIZE] {
    let mut out = [0i32; OUTPUT_SIZE];
    for i in 0..20 {
        out[i] = 500 + i as i32; // argmax at 19, stable
    }
    out[CURSOR_DX] = 16384 + (k * 37) % 4000;
    out[CURSOR_DY] = 16384 - (k * 19) % 4000;
    out
}

/// Run a session of `n` releases against the given pipeline, returning the
/// perturbed cursor outputs per step. Each release uses a distinct epoch.
fn run_session(layer: &mut PrivacyLayer, ndal: &mut NdalPipeline, n: i32) -> Vec<(i32, i32)> {
    let mut released = Vec::new();
    for k in 0..n {
        let mut out = make_output(k);
        let outcome = layer
            .release(&mut out, ndal, Epoch(k as u64), 1, k as u64)
            .expect("entropy available");
        assert_eq!(outcome, ReleaseOutcome::Perturbed);
        released.push((out[CURSOR_DX], out[CURSOR_DY]));
    }
    released
}

/// Run `n` releases that all share a SINGLE epoch — the case that used to
/// desync replay (the noise draw could be skipped on a live-only condition).
fn run_session_one_epoch(layer: &mut PrivacyLayer, ndal: &mut NdalPipeline, n: i32) -> Vec<(i32, i32)> {
    let mut released = Vec::new();
    for k in 0..n {
        let mut out = make_output(k);
        let outcome = layer
            .release(&mut out, ndal, Epoch(0), 1, k as u64)
            .expect("entropy available");
        assert_eq!(outcome, ReleaseOutcome::Perturbed);
        released.push((out[CURSOR_DX], out[CURSOR_DY]));
    }
    released
}

#[test]
fn replay_reproduces_perturbation_bit_for_bit() {
    let cfg = NdalConfig::default();
    let budget = PrivacyBudget::from_target_bits(0.3, u64::MAX);

    // ── Live session: noise drawn fresh, recorded in the log.
    let mut live = NdalPipeline::new(cfg.clone());
    live.seed_random(0xC0FFEE);
    let mut layer_live = PrivacyLayer::new(&table(), budget.clone());
    let live_out = run_session(&mut layer_live, &mut live, 12);

    // ── Replay session: same log, fresh privacy layer.
    let entries = live.log_entries().to_vec();
    let mut replay = NdalPipeline::from_log(entries, cfg).unwrap();
    assert_eq!(replay.mode(), LogMode::Replay);
    let mut layer_replay = PrivacyLayer::new(&table(), budget);
    let replay_out = run_session(&mut layer_replay, &mut replay, 12);

    assert_eq!(live_out, replay_out,
        "replay must reproduce the perturbed output bit-for-bit");
}

#[test]
fn fresh_seed_diverges_from_observer() {
    let cfg = NdalConfig::default();
    let budget = PrivacyBudget::from_target_bits(0.3, u64::MAX);

    let mut a = NdalPipeline::new(cfg.clone());
    a.seed_random(1);
    let mut layer_a = PrivacyLayer::new(&table(), budget.clone());
    let out_a = run_session(&mut layer_a, &mut a, 12);

    let mut b = NdalPipeline::new(cfg);
    b.seed_random(2); // different universe / no shared log
    let mut layer_b = PrivacyLayer::new(&table(), budget);
    let out_b = run_session(&mut layer_b, &mut b, 12);

    assert_ne!(out_a, out_b,
        "without the log, a different draw must yield different output \
         (otherwise there is no privacy)");
}

#[test]
fn same_seed_two_nodes_agree() {
    // Two nodes with the same seed and config agree — the determinism the
    // verification path relies on still holds for shared-state nodes.
    let cfg = NdalConfig::default();
    let budget = PrivacyBudget::from_target_bits(0.3, u64::MAX);

    let mut p1 = NdalPipeline::new(cfg.clone());
    let mut p2 = NdalPipeline::new(cfg);
    p1.seed_random(42);
    p2.seed_random(42);
    let mut l1 = PrivacyLayer::new(&table(), budget.clone());
    let mut l2 = PrivacyLayer::new(&table(), budget);

    assert_eq!(run_session(&mut l1, &mut p1, 16),
               run_session(&mut l2, &mut p2, 16));
}

#[test]
fn budget_exhaustion_fails_closed_midstream() {
    let cfg = NdalConfig::default();
    // Allow only 3 releases for the principal.
    let budget = PrivacyBudget::from_target_bits(0.3, 3);
    let mut ndal = NdalPipeline::new(cfg);
    ndal.seed_random(9);
    let mut layer = PrivacyLayer::new(&table(), budget);

    let mut outcomes = Vec::new();
    for k in 0..6 {
        let mut out = make_output(k);
        let outcome = layer
            .release(&mut out, &mut ndal, Epoch(k as u64), 1, k as u64)
            .expect("entropy available");
        outcomes.push(outcome);
        // After exhaustion the cursor is forced neutral.
        if outcome == ReleaseOutcome::Suppressed {
            assert_eq!(out[CURSOR_DX], dnos_pacp::OUTPUT_MIDPOINT);
            assert_eq!(out[CURSOR_DY], dnos_pacp::OUTPUT_MIDPOINT);
        }
    }

    assert_eq!(outcomes[0], ReleaseOutcome::Perturbed);
    assert_eq!(outcomes[2], ReleaseOutcome::Perturbed);
    assert_eq!(outcomes[3], ReleaseOutcome::Suppressed);
    assert_eq!(outcomes[5], ReleaseOutcome::Suppressed);
    assert_eq!(layer.releases_spent(1), 3);
}

#[test]
fn replay_bit_exact_with_many_releases_in_one_epoch() {
    // Regression for the rate-limit/desync finding: many releases share one
    // epoch. Because the layer draws exactly once per release (before any
    // budget branch), every draw is logged and replay stays aligned.
    let cfg = NdalConfig::default(); // max_queries_per_epoch = 64
    let budget = PrivacyBudget::from_target_bits(0.3, u64::MAX);

    let mut live = NdalPipeline::new(cfg.clone());
    live.seed_random(0xABCD);
    let mut layer_live = PrivacyLayer::new(&table(), budget.clone());
    let live_out = run_session_one_epoch(&mut layer_live, &mut live, 50);

    let entries = live.log_entries().to_vec();
    let mut replay = NdalPipeline::from_log(entries, cfg).unwrap();
    let mut layer_replay = PrivacyLayer::new(&table(), budget);
    let replay_out = run_session_one_epoch(&mut layer_replay, &mut replay, 50);

    assert_eq!(live_out, replay_out,
        "50 same-epoch releases must replay bit-for-bit");
}

#[test]
fn rate_limited_release_fails_loud() {
    // With only 1 Random query allowed per epoch, the 2nd same-epoch release
    // gets no entropy and must error (never silently suppress, which would
    // desync replay against a log that has no entry for the skipped draw).
    let cfg = NdalConfig { max_queries_per_epoch: 1, ..NdalConfig::default() };
    let budget = PrivacyBudget::from_target_bits(0.3, u64::MAX);
    let mut ndal = NdalPipeline::new(cfg);
    ndal.seed_random(5);
    let mut layer = PrivacyLayer::new(&table(), budget);

    let mut out0 = make_output(0);
    assert!(layer.release(&mut out0, &mut ndal, Epoch(0), 1, 0).is_ok());

    let mut out1 = make_output(1);
    assert_eq!(
        layer.release(&mut out1, &mut ndal, Epoch(0), 1, 1),
        Err(dnos_pacp::PrivacyError::EntropyUnavailable),
    );
}
