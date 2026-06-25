//! Leakage table — the offline-measured signal the Risk Engine sizes noise to.
//!
//! PAC-Privacy calibrates noise to *measured* leakage. `tools/leakage_eval.py`
//! (Privacy Track M0) measures, over resampled private histories, how much the
//! cursor-delta outputs move with the command held fixed — the variance of the
//! release attributable to private history. That measurement is this table.
//!
//! The runtime never re-measures: estimation is offline (M0), the kernel/host
//! just reads the baked table. The table is keyed by the weights' CRC32 (the
//! same checksum carried in the weight-blob header) so a table can be checked
//! against the model it was measured on.
//!
//! Units: variance is in **output units²**. The cursor outputs live in the
//! piecewise-sigmoid range [0, 32767] with midpoint 16384; the kernel decodes
//! a delta as `(out - 16384) >> 10`, so 1024 output units = 1 px of cursor
//! delta and 1 px² = 1024² output units². Keeping the table in output units
//! lets the whole perturbation path stay integer and bit-reproducible.

use std::fmt;
use std::str::FromStr;

// Output-layer geometry, mirroring dnos.asm decode_output_32.
pub const OUTPUT_SIZE: usize = 32;
pub const CURSOR_DX: usize = 20; // outputs[20] → dx
pub const CURSOR_DY: usize = 22; // outputs[22] → dy
pub const OUTPUT_MIDPOINT: i32 = 16384; // piecewise-sigmoid "zero delta"
pub const OUTPUT_MAX: i32 = 32767;
pub const OUTPUT_MIN: i32 = 0;
pub const PX_SHIFT: u32 = 10; // (out - 16384) >> 10  → 1024 units = 1 px

/// One output-channel's private-history signal, measured offline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeakageTable {
    /// CRC32 of the weight blob this was measured on (header offset 21).
    pub weights_crc32: u32,
    /// Variance of cursor dx (output[20]) due to private history, output-units².
    pub within_var_out2_dx: u64,
    /// Variance of cursor dy (output[22]) due to private history, output-units².
    pub within_var_out2_dy: u64,
}

impl LeakageTable {
    pub fn new(weights_crc32: u32, within_var_out2_dx: u64, within_var_out2_dy: u64) -> Self {
        LeakageTable { weights_crc32, within_var_out2_dx, within_var_out2_dy }
    }

    /// Build from per-channel pixel² variances. Convenience/test constructor —
    /// the **canonical** production table is written by `tools/leakage_eval.py
    /// --table` (one rounding authority) and read via [`FromStr`]; don't mix
    /// that path with this one for the same weights, as the two languages'
    /// round-half rules can differ by 1 unit on an exact-half boundary.
    /// 1 px² = 1024² output units².
    pub fn from_px2(weights_crc32: u32, within_px2_dx: f64, within_px2_dy: f64) -> Self {
        let scale = (1u64 << PX_SHIFT) as f64; // 1024
        let conv = |v: f64| (v.max(0.0) * scale * scale).round() as u64;
        LeakageTable::new(weights_crc32, conv(within_px2_dx), conv(within_px2_dy))
    }
}

// Text format (std-parseable, no serde — keeps the crate dependency-free):
//
//   # dnos-pacp leakage table v1
//   crc32 0xXXXXXXXX
//   within_out2 <dx> <dy>
//
// Emitted by `tools/leakage_eval.py --table <path>`.
impl fmt::Display for LeakageTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "# dnos-pacp leakage table v1")?;
        writeln!(f, "crc32 0x{:08X}", self.weights_crc32)?;
        write!(f, "within_out2 {} {}", self.within_var_out2_dx, self.within_var_out2_dy)
    }
}

impl FromStr for LeakageTable {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut crc: Option<u32> = None;
        let mut within: Option<(u64, u64)> = None;

        for line in s.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split_whitespace();
            match it.next() {
                Some("crc32") => {
                    let tok = it.next().ok_or("crc32: missing value")?;
                    let hex = tok.strip_prefix("0x").or_else(|| tok.strip_prefix("0X"));
                    crc = Some(match hex {
                        Some(h) => u32::from_str_radix(h, 16).map_err(|e| e.to_string())?,
                        None => tok.parse::<u32>().map_err(|e| e.to_string())?,
                    });
                }
                Some("within_out2") => {
                    let dx = it.next().ok_or("within_out2: missing dx")?
                        .parse::<u64>().map_err(|e| e.to_string())?;
                    let dy = it.next().ok_or("within_out2: missing dy")?
                        .parse::<u64>().map_err(|e| e.to_string())?;
                    within = Some((dx, dy));
                }
                Some(other) => return Err(format!("unknown table key: {other}")),
                None => {}
            }
        }

        let crc = crc.ok_or("table missing crc32 line")?;
        let (dx, dy) = within.ok_or("table missing within_out2 line")?;
        Ok(LeakageTable::new(crc, dx, dy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn px2_conversion_scales_by_1024_squared() {
        let t = LeakageTable::from_px2(0xABCD, 1.0, 0.25);
        assert_eq!(t.within_var_out2_dx, 1024 * 1024);
        assert_eq!(t.within_var_out2_dy, (0.25 * 1024.0 * 1024.0) as u64);
    }

    #[test]
    fn text_format_roundtrips() {
        let t = LeakageTable::new(0xDEADBEEF, 143_700, 251_000);
        let s = t.to_string();
        let back: LeakageTable = s.parse().unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn parses_with_comments_and_blank_lines() {
        let s = "# header\n\ncrc32 0x00000010\nwithin_out2 5 7\n";
        let t: LeakageTable = s.parse().unwrap();
        assert_eq!(t, LeakageTable::new(16, 5, 7));
    }

    #[test]
    fn rejects_missing_fields() {
        assert!("crc32 0x1".parse::<LeakageTable>().is_err());
        assert!("within_out2 1 2".parse::<LeakageTable>().is_err());
    }
}
