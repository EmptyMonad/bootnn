//! Print all 256 v1 frames as hex, one per line - consumed by
//! tools/frame_test.py for the cross-language differential against
//! the Python wire-protocol authority.

use dnos_ial::wire::frame;

fn main() {
    for e in 0..=255u8 {
        let f = frame(e);
        println!("{:02x}{:02x}", f[0], f[1]);
    }
}
