//! DNOS law runner for bare-metal RISC-V (`qemu-system-riscv32 -M virt`).
//!
//! A LAW RUNNER, not a kernel port: it boots, validates the v5 blob
//! (magic, version, payload CRC, reserved ternary code 10), self-plays
//! the same `pboxline` boot demo as the x86 kernel, then serves S1 v1
//! frames from the UART - one law step per event, exactly
//! `S(n+1) = f(S(n), input(n))`. The law-visible state (last_cmd, h,
//! counters) sits in a fixed block at 0x8020_0000 so the portability
//! harness reads it over QMP, the same way swarm_test reads the x86
//! kernel. `tools/train.py::SsmMachine` is the reference this binary
//! must match bit for bit - the portability faithfulness equality is
//! the port's acceptance test.
//!
//! Input is the wire, not a keyboard: `virt` has no PS/2, and the S1
//! frame [0xD1][event] is the substrate-neutral input surface. This is
//! the second implementation of frame-DECODE semantics (the x86 kernel
//! is the first): a byte that is not a marker when a marker is
//! expected is discarded and counted in serial_bad_frames.
//!
//! The v5 blob is supplied VERBATIM by `-device loader` at
//! 0x8040_0000 - no image patching; any artifact CRC boots unchanged.

#![no_std]
#![no_main]

use core::arch::global_asm;
use core::panic::PanicInfo;
use core::ptr::{read_volatile, write_volatile};

global_asm!(
    r#"
    .section .text.init
    .globl _start
_start:
    la sp, {stack_top}
    j main_rv
    "#,
    stack_top = const STACK_TOP,
);

const STACK_TOP: u32 = 0x8018_0000;

// ── the fixed state block the harness reads over QMP ────────────────
const STATE_BASE: u32 = 0x8020_0000;
const MAGIC: u32 = 0x5256_4E44; // "DNVR"
// offsets: +0 magic, +4 hdr_status, +8 last_cmd, +12 step_count,
//          +16 serial_rx_count, +20 serial_bad_frames, +24 h[512] i16

const BLOB_BASE: u32 = 0x8040_0000;
const HEADER_SIZE: u32 = 128;

// 16550 UART on the virt machine.
const UART_BASE: u32 = 0x1000_0000;
const UART_LSR: u32 = UART_BASE + 5;

const FRAME_V1: u8 = 0xD1;
const DEMO_KEYS: &[u8] = b"pboxline"; // must match src/dnos.asm

const H_MAX: usize = 512;
const N_CMD: usize = 20;
// Substrate limits: the readout buffers (z1/z2 below) and the drive
// loop are sized for these. The runner is self-validating like the
// x86 kernel's validate_weights - a blob past any bound is REFUSED
// (hdr_status=2), never run into an out-of-bounds panic that would
// hang looking healthy.
const FEAT_MAX: usize = 8; // key is u8: `key >> i` needs i < 8
const R1_MAX: usize = 1024;
const R2_MAX: usize = 512;
const N_OUT_MAX: usize = 64;

#[inline(always)]
fn rd8(a: u32) -> u8 {
    unsafe { read_volatile(a as *const u8) }
}
#[inline(always)]
fn wr32(a: u32, v: u32) {
    unsafe { write_volatile(a as *mut u32, v) }
}
#[inline(always)]
fn rd32(a: u32) -> u32 {
    unsafe { read_volatile(a as *const u32) }
}
#[inline(always)]
fn rd16(a: u32) -> u16 {
    unsafe { read_volatile(a as *const u16) }
}

fn state_set(off: u32, v: u32) {
    wr32(STATE_BASE + off, v);
}
fn state_get(off: u32) -> u32 {
    rd32(STATE_BASE + off)
}
fn h_get(c: usize) -> i32 {
    unsafe { read_volatile((STATE_BASE + 24 + 2 * c as u32) as *const i16) as i32 }
}
fn h_set(c: usize, v: i32) {
    unsafe { write_volatile((STATE_BASE + 24 + 2 * c as u32) as *mut i16, v as i16) }
}

// ── v5 blob layout (tools/train.py::save_v5) ────────────────────────
struct Law {
    h: usize,       // channels
    r1: usize,
    r2: usize,
    n_out: usize,
    n_in: usize,
    shifts: [u32; 4], // k_in, k1, k2, k3
    d_base: u32,      // decay byte table, h entries
    mat_base: [u32; 4],
    mat_di: [usize; 4],
}

fn crc32(mut addr: u32, len: u32) -> u32 {
    // Nibble-table CRC-32 (zlib polynomial), the same construction the
    // x86 kernel uses at boot.
    let mut tab = [0u32; 16];
    for (i, t) in tab.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..4 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *t = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for _ in 0..len {
        let b = rd8(addr) as u32;
        crc = tab[((crc ^ b) & 0xF) as usize] ^ (crc >> 4);
        crc = tab[((crc ^ (b >> 4)) & 0xF) as usize] ^ (crc >> 4);
        addr += 1;
    }
    !crc
}

fn parse_and_validate() -> Option<Law> {
    // magic 'D','N', version 5.
    if rd8(BLOB_BASE) != b'D' || rd8(BLOB_BASE + 1) != b'N' || rd8(BLOB_BASE + 2) != 5 {
        return None;
    }
    let sz = |i: u32| -> usize { rd16(BLOB_BASE + 5 + 2 * i) as usize };
    let (n_in, h, r1, r2, n_out) = (sz(0), sz(1), sz(2), sz(3), sz(4));
    if h == 0
        || h > H_MAX
        || n_in == 0
        || n_in > FEAT_MAX
        || r1 > R1_MAX
        || r2 > R2_MAX
        || n_out == 0
        || n_out > N_OUT_MAX
    {
        return None;
    }
    let n_weights = rd32(BLOB_BASE + 16);
    let expect = (n_in * h + h * r1 + r1 * r2 + r2 * n_out) as u32;
    if n_weights != expect {
        return None;
    }
    // Payload CRC: decay table + packed matrices, exactly as written.
    let payload_len = h as u32 + n_weights / 4;
    if crc32(BLOB_BASE + HEADER_SIZE, payload_len) != rd32(BLOB_BASE + 21) {
        return None;
    }
    // Reserved ternary code 10 is refused before any inference.
    let packed_base = BLOB_BASE + HEADER_SIZE + h as u32;
    for i in 0..(n_weights / 4) {
        let b = rd8(packed_base + i);
        // code 10 in any of the 4 lanes: (b >> 2k) & 3 == 2
        if (b & 3) == 2 || ((b >> 2) & 3) == 2 || ((b >> 4) & 3) == 2 || ((b >> 6) & 3) == 2 {
            return None;
        }
    }
    let counts = [n_in * h, h * r1, r1 * r2, r2 * n_out];
    let mut mat_base = [0u32; 4];
    let mut acc = packed_base;
    for (i, c) in counts.iter().enumerate() {
        mat_base[i] = acc;
        acc += (*c as u32) / 4;
    }
    Some(Law {
        h,
        r1,
        r2,
        n_out,
        n_in,
        shifts: [
            rd8(BLOB_BASE + 25) as u32,
            rd8(BLOB_BASE + 26) as u32,
            rd8(BLOB_BASE + 27) as u32,
            rd8(BLOB_BASE + 28) as u32,
        ],
        d_base: BLOB_BASE + HEADER_SIZE,
        mat_base,
        mat_di: [n_in, h, r1, r2],
    })
}

/// Signed ternary code at flat index `o*di + i` of matrix m:
/// 2-bit codes, weight i in bits 2*(i mod 4), 00=0 01=+1 11=-1.
#[inline(always)]
fn tern(law: &Law, m: usize, o: usize, i: usize) -> i32 {
    let idx = o * law.mat_di[m] + i;
    let code = (rd8(law.mat_base[m] + (idx >> 2) as u32) >> (2 * (idx & 3))) & 3;
    match code {
        1 => 1,
        3 => -1,
        _ => 0,
    }
}

/// One tick of the v5 law: decay, drive, clamp, 3-layer ternary
/// readout, piecewise sigmoid, argmax[:20] -> last_cmd. Must equal
/// SsmMachine.step bit for bit.
fn law_step(law: &Law, key: u8) {
    let [k_in, k1, k2, k3] = law.shifts;
    // leak + drive + clamp
    for c in 0..law.h {
        let d = rd8(law.d_base + c as u32) as u32;
        let hv = h_get(c);
        let hl = hv - (hv >> d);
        let mut u = 0i32;
        for i in 0..law.n_in {
            if (key >> i) & 1 != 0 {
                u += 256 * tern(law, 0, c, i);
            }
        }
        h_set(c, (hl + (u >> k_in)).clamp(-32768, 32767));
    }
    // readout: h -> r1 -> r2 -> out
    let mut z1 = [0i32; 1024];
    for o in 0..law.r1 {
        let mut a = 0i32;
        for c in 0..law.h {
            match tern(law, 1, o, c) {
                1 => a += h_get(c),
                -1 => a -= h_get(c),
                _ => {}
            }
        }
        z1[o] = (a >> k1).clamp(0, 32767);
    }
    let mut z2 = [0i32; 512];
    for o in 0..law.r2 {
        let mut a = 0i32;
        for c in 0..law.r1 {
            match tern(law, 2, o, c) {
                1 => a += z1[c],
                -1 => a -= z1[c],
                _ => {}
            }
        }
        z2[o] = (a >> k2).clamp(0, 32767);
    }
    let mut best_cmd = 0usize;
    let mut best_val = i32::MIN;
    for o in 0..law.n_out.min(N_CMD) {
        let mut a = 0i32;
        for c in 0..law.r2 {
            match tern(law, 3, o, c) {
                1 => a += z2[c],
                -1 => a -= z2[c],
                _ => {}
            }
        }
        let acc = a >> k3;
        let out = if acc < -8192 {
            0
        } else if acc > 8192 {
            32767
        } else {
            (acc + 8192) * 2
        };
        if out > best_val {
            best_val = out;
            best_cmd = o;
        }
    }
    state_set(8, best_cmd as u32); // last_cmd
    state_set(12, state_get(12) + 1); // step_count
}

#[no_mangle]
extern "C" fn main_rv() -> ! {
    // Zero the state block (magic last, so a half-initialized block is
    // never mistaken for a live one).
    for off in (4..24 + 2 * H_MAX as u32).step_by(4) {
        state_set(off, 0);
    }
    let law = match parse_and_validate() {
        Some(l) => l,
        None => {
            state_set(4, 2); // hdr_status: BAD - refuse to serve
            state_set(0, MAGIC);
            loop {
                core::hint::spin_loop();
            }
        }
    };
    state_set(4, 1); // hdr_status: validated
    state_set(0, MAGIC);

    // The same boot demo the x86 kernel plays, so one simulator path
    // predicts both profiles.
    for &k in DEMO_KEYS {
        law_step(&law, k);
    }

    // Serve the wire: v1 frames, one law step per event.
    let mut awaiting_event = false;
    loop {
        if rd8(UART_LSR) & 1 == 0 {
            core::hint::spin_loop();
            continue;
        }
        let b = rd8(UART_BASE);
        if awaiting_event {
            state_set(16, state_get(16) + 1); // serial_rx_count
            law_step(&law, b);
            awaiting_event = false;
        } else if b == FRAME_V1 {
            awaiting_event = true;
        } else {
            state_set(20, state_get(20) + 1); // serial_bad_frames
        }
    }
}

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
