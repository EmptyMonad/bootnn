//! Inspect a leakage table and the noise it implies for a budget.
//!
//!   cargo run --example load_table -- ../leakage.tbl [target_bits]
//!
//! Reads a `dnos-pacp` leakage table (as emitted by
//! `tools/leakage_eval.py --table`), prints the parsed measurement, and shows
//! the per-channel σ the Risk Engine derives for a target leakage in bits.

use std::env;
use std::fs;

use dnos_pacp::{LeakageTable, PrivacyBudget, PrivacyLayer};

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        eprintln!("usage: load_table <table-file> [target_bits]");
        std::process::exit(2);
    });
    let target_bits: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.3);

    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let table: LeakageTable = text.parse().unwrap_or_else(|e| {
        eprintln!("cannot parse {path}: {e}");
        std::process::exit(1);
    });

    println!("parsed table:");
    println!("  weights crc32      : 0x{:08X}", table.weights_crc32);
    println!("  within var (out²)  : dx {}  dy {}",
             table.within_var_out2_dx, table.within_var_out2_dy);

    let budget = PrivacyBudget::from_target_bits(target_bits, u64::MAX);
    let layer = PrivacyLayer::new(&table, budget.clone());
    let (sdx, sdy) = layer.sigma();
    // 1024 output units = 1 px (decode is (out-16384) >> 10).
    println!("\nfor target ≈ {:.2} bits  (SNR {}/{}):",
             budget.target_bits(), budget.snr_num, budget.snr_den);
    println!("  σ dx : {sdx} output units  (≈ {:.3} px)", sdx as f64 / 1024.0);
    println!("  σ dy : {sdy} output units  (≈ {:.3} px)", sdy as f64 / 1024.0);
}
