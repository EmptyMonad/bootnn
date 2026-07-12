//! S1 v1 wire framing - the Rust emitter.
//!
//! `tools/dnos_client.py::frame` is the wire-protocol AUTHORITY; this
//! module is a second, independent implementation whose byte-level
//! agreement with the authority is enforced by `tools/frame_test.py`
//! on every push (all 256 events). The marker byte doubles as the
//! version: 0xD1 sits above the ASCII event range, so desynced or
//! unframed streams self-correct by discard on the kernel side
//! (counted in `serial_bad_frames`).

/// v1 frame marker/version byte. Must equal `dnos_client.FRAME_V1`;
/// the differential gate, not this comment, enforces that.
pub const FRAME_V1: u8 = 0xD1;

/// One event on the wire: `[FRAME_V1][event]`. Mirrors the authority
/// exactly - no validation, no opinions; byte parity is the contract.
pub fn frame(event: u8) -> [u8; 2] {
    [FRAME_V1, event]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_two_bytes_marker_first() {
        for e in 0..=255u8 {
            assert_eq!(frame(e), [0xD1, e]);
        }
    }

    #[test]
    fn marker_sits_above_the_ascii_event_range() {
        // The self-correcting-discard property depends on this.
        assert!(FRAME_V1 >= 0x80);
    }
}
