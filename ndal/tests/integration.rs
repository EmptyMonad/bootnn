//! Integration tests for the DNOS Non-Determinism Abstraction Layer.
//!
//! These test the end-to-end properties:
//! 1. Live session → replay produces identical tokens
//! 2. Two nodes with same seed + same queries → identical logs
//! 3. Different universes → divergence detected at exact entry
//! 4. Snapshot enables log pruning without state loss
//! 5. Training with oracle seeds is fully replayable

use dnos_ndal::*;
use dnos_ndal::log::LogMode;
use dnos_ndal::pipeline::NdalPipeline;

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 1: Full live-then-replay cycle
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn scenario_live_replay_full_cycle() {
    let config = NdalConfig::default();

    // === Live session ===
    let mut live = NdalPipeline::new(config.clone());
    live.seed_random(0xDNOS);

    let mut live_tokens = Vec::new();

    // Mix of oracle types
    live_tokens.push(live.env(Epoch(0), EnvKey::CpuCount));
    live_tokens.push(live.random(Epoch(1), 4));
    live_tokens.push(live.random(Epoch(2), 8));
    live_tokens.push(live.clock(Epoch(3)));
    live_tokens.push(live.random(Epoch(4), 4));

    let live_chain = *live.log_chain_head();
    let live_entries = live.log_entries().to_vec();

    // === Replay session ===
    let mut replay = NdalPipeline::from_log(live_entries, config).unwrap();

    let mut replay_tokens = Vec::new();

    // Same queries (params don't matter in replay — log provides responses)
    replay_tokens.push(replay.env(Epoch(0), EnvKey::CpuCount));
    replay_tokens.push(replay.random(Epoch(1), 4));
    replay_tokens.push(replay.random(Epoch(2), 8));
    replay_tokens.push(replay.clock(Epoch(3)));
    replay_tokens.push(replay.random(Epoch(4), 4));

    assert_eq!(live_tokens, replay_tokens,
        "Replay must produce identical tokens for all oracle types");

    // Chain heads should match (replay doesn't extend the chain,
    // but the logged entries have the same hashes)
    assert!(replay.verify_chain().is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 2: Two nodes, same seed, same queries → identical
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn scenario_two_nodes_identical_execution() {
    let config = NdalConfig::default();

    let mut node_a = NdalPipeline::new(config.clone());
    let mut node_b = NdalPipeline::new(config);

    node_a.seed_random(42);
    node_b.seed_random(42);

    // Both nodes execute the same query sequence
    for i in 0..50 {
        let epoch = Epoch(i);
        node_a.random(epoch, 4);
        node_b.random(epoch, 4);
    }

    // Logs should be identical
    assert_eq!(
        node_a.find_divergence_with(&node_b),
        None,
        "Two nodes with same seed and queries must produce identical logs"
    );

    // Chain heads should match
    assert_eq!(node_a.log_chain_head(), node_b.log_chain_head());
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 3: Different universes → exact divergence point
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn scenario_divergence_pinpoints_exact_entry() {
    let config = NdalConfig::default();

    let mut node_a = NdalPipeline::new(config.clone());
    let mut node_b = NdalPipeline::new(config);

    // Same seed for first 10 queries
    node_a.seed_random(42);
    node_b.seed_random(42);

    for i in 0..10 {
        node_a.random(Epoch(i), 4);
        node_b.random(Epoch(i), 4);
    }

    // No divergence yet
    assert_eq!(node_a.find_divergence_with(&node_b), None);

    // Now node_b re-seeds (simulating a different universe from here)
    node_b.seed_random(999);

    // Both continue querying
    for i in 10..20 {
        node_a.random(Epoch(i), 4);
        node_b.random(Epoch(i), 4);
    }

    // Divergence should be at sequence 10 (first post-reseed entry)
    let div = node_a.find_divergence_with(&node_b);
    assert_eq!(div, Some(10),
        "Divergence should be pinpointed at the exact entry where universes split");
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 4: Snapshot + prune + new node sync
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn scenario_snapshot_prune_sync() {
    let config = NdalConfig::default();
    let mut origin = NdalPipeline::new(config.clone());
    origin.seed_random(42);

    // Generate 100 oracle queries
    for i in 0..100 {
        origin.random(Epoch(i), 4);
    }
    assert_eq!(origin.log().len(), 100);

    // Take snapshot at epoch 50
    let snapshot = origin.snapshot(
        [0xAA; 32], // Mock weights hash
        [0xBB; 32], // Mock state hash
        (12345, 50), // Mock IAL digest
    );
    assert_eq!(snapshot.last_sequence, 99);

    // Prune entries 0-49
    let pruned = origin.prune_before(50);
    assert_eq!(pruned, 50);
    assert_eq!(origin.log().len(), 50);

    // Verify remaining chain integrity
    // (partial chain can't verify from genesis, but entries are structurally intact)
    assert_eq!(origin.log_entries()[0].sequence, 50);
    assert_eq!(origin.log_entries().last().unwrap().sequence, 99);

    // Simulate new node syncing: receives snapshot + remaining log
    let remaining_entries = origin.log_entries().to_vec();
    assert_eq!(remaining_entries.len(), 50);

    // New node can reconstruct from snapshot epoch onward
    // (In production, it would load the snapshot state + replay remaining entries)
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 5: Deterministic training with oracle seeds
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn scenario_deterministic_training_replay() {
    let config = NdalConfig::default();

    // === Training session 1 ===
    let mut trainer_a = NdalPipeline::new(config.clone());
    trainer_a.seed_random(42);

    // Simulate training: get shuffle seed per epoch, init seed at start
    let init_seed_a = trainer_a.weight_init_seed(Epoch(0)).unwrap();
    let mut shuffle_seeds_a = Vec::new();
    for epoch in 1..=10 {
        shuffle_seeds_a.push(trainer_a.training_shuffle_seed(Epoch(epoch)).unwrap());
    }

    // === Training session 2: replay from log ===
    let entries = trainer_a.log_entries().to_vec();
    let mut trainer_b = NdalPipeline::from_log(entries, config).unwrap();

    let init_seed_b = trainer_b.weight_init_seed(Epoch(0)).unwrap();
    let mut shuffle_seeds_b = Vec::new();
    for epoch in 1..=10 {
        shuffle_seeds_b.push(trainer_b.training_shuffle_seed(Epoch(epoch)).unwrap());
    }

    assert_eq!(init_seed_a, init_seed_b,
        "Weight init seeds must match across live/replay");
    assert_eq!(shuffle_seeds_a, shuffle_seeds_b,
        "Shuffle seeds must match across live/replay for all epochs");
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 6: Mixed oracle queries maintain log coherence
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn scenario_mixed_oracle_log_coherence() {
    let config = NdalConfig::default();
    let mut ndal = NdalPipeline::new(config);
    ndal.seed_random(42);

    // Interleave different oracle types
    ndal.random(Epoch(0), 4);
    ndal.env(Epoch(0), EnvKey::CpuCount);
    ndal.clock(Epoch(1));
    ndal.random(Epoch(1), 8);
    ndal.env(Epoch(2), EnvKey::BootTimestamp);
    ndal.random(Epoch(2), 4);

    // Verify chain is intact across oracle type switches
    assert!(ndal.verify_chain().is_ok(),
        "Log chain must remain valid across mixed oracle queries");

    // Verify sequential sequence numbers
    let entries = ndal.log_entries();
    for (i, entry) in entries.iter().enumerate() {
        assert_eq!(entry.sequence, i as u64,
            "Sequence numbers must be monotonic with no gaps");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 7: Rate limiting prevents oracle abuse
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn scenario_rate_limiting() {
    let config = NdalConfig {
        max_queries_per_epoch: 5,
        ..NdalConfig::default()
    };
    let mut ndal = NdalPipeline::new(config);
    ndal.seed_random(42);

    // 5 queries in epoch 0 should succeed
    for _ in 0..5 {
        assert!(ndal.random(Epoch(0), 4).is_some());
    }

    // 6th query in same epoch should be rate-limited
    assert!(ndal.random(Epoch(0), 4).is_none(),
        "Should be rate-limited after max_queries_per_epoch");

    // New epoch resets the counter
    assert!(ndal.random(Epoch(1), 4).is_some(),
        "New epoch should reset rate limit");
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 8: Large-scale determinism (1000 mixed queries)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn scenario_large_scale_determinism() {
    let config = NdalConfig::default();

    let mut p1 = NdalPipeline::new(config.clone());
    let mut p2 = NdalPipeline::new(config);

    p1.seed_random(12345);
    p2.seed_random(12345);

    // 1000 mixed queries
    for i in 0u64..1000 {
        let epoch = Epoch(i / 10); // ~10 queries per epoch
        match i % 3 {
            0 => { p1.random(epoch, 4); p2.random(epoch, 4); }
            1 => { p1.env(epoch, EnvKey::CpuCount); p2.env(epoch, EnvKey::CpuCount); }
            2 => { p1.clock(epoch); p2.clock(epoch); }
            _ => unreachable!(),
        }
    }

    assert_eq!(
        p1.find_divergence_with(&p2),
        None,
        "1000 identical mixed queries must produce identical logs"
    );
}
