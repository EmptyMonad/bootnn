#!/usr/bin/env python3
"""
Leakage eval for DNOS — Privacy Track milestone M0 (see ROADMAP "Privacy
Track" and docs/PAC_PRIVACY_FEASIBILITY.md).

PAC-Privacy calibrates noise to *measured* information leakage. This tool
does the measuring — and nothing else. It adds NO noise; it only quantifies
how much the observable outputs move as the (private) keystroke history
varies, which is the signal covariance any future perturbation layer (M1)
will be sized against.

Two channels, matching the feasibility report:

  * command channel  — argmax over outputs[0:20]. Expected LOW leakage: the
    network was trained to be history-invariant on single-key commands
    (ROADMAP backlog #7 / context_eval.py), so private history should barely
    move which command is drawn.
  * cursor channel   — outputs[20] -> dx, outputs[22] -> dy (the kernel's
    decode_output_32: (out - 16384) >> 10). Real-valued, NOT hardened
    against history → expected HIGH leakage. Reported in actual pixels.

Method: hold the current key fixed (the command the user is overtly issuing)
and resample the prior history; whatever the output still does is leakage of
PRIVATE history. Decompose total output variance into "which command you're
issuing" (between-key) vs "your private history" (within-key).

The forward pass is the assembly-exact integer path (vectorized twin of
train.simulate_assembly_forward, asserted equal at startup), so the numbers
reflect what bare metal computes.

Usage:
  python tools/leakage_eval.py                       # weights.bin, 64 hist/key
  python tools/leakage_eval.py --per-key 200
  python tools/leakage_eval.py --json leakage.json   # dump Sigma + metrics
  python tools/leakage_eval.py --max-history-bits 0.5  # optional CI gate
"""

import argparse
import json
import math
import string
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from train import (  # noqa: E402
    INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE,
    HEADER_SIZE, WEIGHT_DATA_SIZE, W1_COUNT, W2_COUNT,
    Q88_SCALE, Q88_MIN, Q88_MAX, CONTEXT_EVENTS,
    sequence_to_input, simulate_assembly_forward,
)

EVAL_SEED = 0x10C0FFEE  # independent of training (1337) and context_eval (99173)

# Decode constants from dnos.asm decode_output_32.
CURSOR_OUTPUTS = [20, 21, 22, 23]   # 20->dx, 22->dy used by the kernel
DX_IDX, DY_IDX = 20, 22
SIGMOID_MID = 16384                 # piecewise-sigmoid "zero delta" midpoint
DECODE_SHIFT = 10                   # sar 10  → 1 output unit = 1/1024 px-ish

# The single-key command contract (mirror of context_eval.SINGLE_KEYS, which
# mirrors train.single_keys). These are the keys a user overtly issues.
SINGLE_KEYS = {
    'p': 'pixel', 'd': 'pixel', '.': 'pixel',
    'b': 'rect', 'r': 'rect',
    'l': 'hline', 'h': 'hline', '-': 'hline',
    'v': 'vline', '|': 'vline',
    'o': 'circle', 'f': 'fill', 'c': 'clear', 'u': 'undo',
    ' ': 'nop',
    'w': 'move_up', 'k': 'move_up',
    's': 'move_down', 'j': 'move_down',
    'a': 'move_left', '+': 'color_next',
}

# Prior over private history — arbitrary plausible typing, incl. untrained
# keys. NOTE: PAC privacy holds only w.r.t. this modeled prior; an adversary
# with a sharper prior sees more (see feasibility doc, risk #3).
HISTORY_ALPHABET = [ord(c) for c in
                    string.ascii_lowercase + string.digits + " .,-+|"]


# ─────────────────────────────────────────────────────────────────────────────
# Assembly-exact integer forward (vectorized) — matches the kernel bit-for-bit
# ─────────────────────────────────────────────────────────────────────────────

def read_crc32(weights_file):
    """Weight-blob CRC32 from the 128-byte header (offset 21, little-endian
    uint32; see train.save). Lets a leakage table be checked against the model
    it was measured on."""
    with open(weights_file, 'rb') as f:
        header = f.read(HEADER_SIZE)
    return int.from_bytes(header[21:25], 'little')


def load_weights(weights_file):
    """Load the Q8.8 int16 blob and reshape to (in,out) matrices.

    train.save() stores each layer as `for j in OUT: for i in IN:` i.e.
    flat[j*IN+i] = W[i,j]; so reshape(OUT, IN).T recovers W[i,j]."""
    with open(weights_file, 'rb') as f:
        f.read(HEADER_SIZE)
        raw = f.read(WEIGHT_DATA_SIZE)
    w = np.frombuffer(raw, dtype=np.int16).astype(np.int64)
    w1 = w[:W1_COUNT].reshape(HIDDEN1_SIZE, INPUT_SIZE).T            # (256,128)
    w2 = w[W1_COUNT:W1_COUNT + W2_COUNT].reshape(HIDDEN2_SIZE, HIDDEN1_SIZE).T
    w3 = w[W1_COUNT + W2_COUNT:].reshape(OUTPUT_SIZE, HIDDEN2_SIZE).T  # (64,32)
    return w1, w2, w3


def _layer(act, w):
    """One layer: per-element (a*w)>>8 (arithmetic/floor, == SAR), then sum.
    NOT a plain matmul — each product is floored before accumulation, exactly
    as the kernel's two-operand imul + sar 8 does."""
    # act: (N, IN) int64 ; w: (IN, OUT) int64 → (N, OUT)
    return ((act[:, :, None] * w[None, :, :]) >> 8).sum(axis=1)


def forward_batch(w1, w2, w3, X, chunk=256):
    """Bit-exact integer forward over a batch of float inputs X (N, 256).
    Returns int32 outputs (N, 32) including the piecewise-sigmoid output."""
    N = X.shape[0]
    out = np.empty((N, OUTPUT_SIZE), dtype=np.int64)
    for s in range(0, N, chunk):
        Xc = X[s:s + chunk]
        Xr = np.clip(np.round(Xc * Q88_SCALE), Q88_MIN, Q88_MAX).astype(np.int64)
        a1 = np.clip(_layer(Xr, w1), 0, Q88_MAX)
        a2 = np.clip(_layer(a1, w2), 0, Q88_MAX)
        z3 = _layer(a2, w3)
        # piecewise sigmoid (matches kernel + simulate_assembly_forward)
        o = np.where(z3 < -8192, 0,
                     np.where(z3 > 8192, 32767, (z3 + 8192) * 2))
        out[s:s + chunk] = o
    return out


def decode_cursor(outputs):
    """Outputs (N,32) → integer (dx, dy) exactly as decode_output_32:
    ((out - 16384) >> 10), arithmetic shift."""
    dx = (outputs[:, DX_IDX].astype(np.int64) - SIGMOID_MID) >> DECODE_SHIFT
    dy = (outputs[:, DY_IDX].astype(np.int64) - SIGMOID_MID) >> DECODE_SHIFT
    return dx, dy


def self_check(weights_file, w1, w2, w3, rng):
    """Assert the vectorized path equals the canonical per-element simulator
    so leakage numbers are trustworthy / metal-faithful. Probes must mirror the
    inputs evaluate() actually feeds — `[key] + history` with a leading command
    key in slot 0 — not just bare histories, or a divergence that only shows up
    with a command byte present would slip through."""
    probes = []
    keys = sorted(SINGLE_KEYS)
    for i in range(12):
        n = int(rng.integers(0, CONTEXT_EVENTS))
        hist = [int(rng.choice(HISTORY_ALPHABET)) for _ in range(n)]
        # Half with a real command key in slot 0 (as evaluate does), half pure
        # history, so both the keyed and unkeyed input shapes are covered.
        if i % 2 == 0:
            key = ord(keys[i % len(keys)])
            probes.append(sequence_to_input([key] + hist))
        else:
            probes.append(sequence_to_input(hist if hist else [ord(keys[0])]))
    X = np.stack(probes)
    batch = forward_batch(w1, w2, w3, X)
    for i, inp in enumerate(probes):
        ref = simulate_assembly_forward(weights_file, inp)
        if not np.array_equal(batch[i], ref.astype(np.int64)):
            raise SystemExit("[leakage_eval] FATAL: vectorized forward diverges "
                             "from simulate_assembly_forward — numbers would be "
                             "untrustworthy. Aborting.")


# ─────────────────────────────────────────────────────────────────────────────
# Leakage measurement
# ─────────────────────────────────────────────────────────────────────────────

def shannon_bits(labels):
    """Entropy (bits) of an integer label array — how much an observer learns
    about private history by watching this discrete channel."""
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def evaluate(weights_file, per_key=64, rng=None):
    rng = rng or np.random.default_rng(EVAL_SEED)
    w1, w2, w3 = load_weights(weights_file)
    self_check(weights_file, w1, w2, w3, rng)

    pooled_out = []            # all outputs across the deployment prior
    grp_cmd_bits = []          # per-key command-channel entropy over histories
    grp_dx_std, grp_dy_std = [], []   # per-key cursor std over histories (px)
    grp_dx_mean, grp_dy_mean = [], [] # per-key cursor mean (px) → between-key

    for key in sorted(SINGLE_KEYS):
        # Hold the overt command key fixed; resample PRIVATE history.
        inputs = []
        for _ in range(per_key):
            hist_len = int(rng.integers(0, CONTEXT_EVENTS))
            hist = [int(rng.choice(HISTORY_ALPHABET)) for _ in range(hist_len)]
            inputs.append(sequence_to_input([ord(key)] + hist))
        X = np.stack(inputs)
        out = forward_batch(w1, w2, w3, X)
        pooled_out.append(out)

        cmds = np.argmax(out[:, :20], axis=1)
        dx, dy = decode_cursor(out)
        grp_cmd_bits.append(shannon_bits(cmds))
        grp_dx_std.append(float(np.std(dx)))
        grp_dy_std.append(float(np.std(dy)))
        grp_dx_mean.append(float(np.mean(dx)))
        grp_dy_mean.append(float(np.mean(dy)))

    pooled = np.concatenate(pooled_out, axis=0)

    # ── Cursor covariance Σ (raw output units) over the deployment prior.
    #    This is exactly what the M1 Risk Engine sizes anisotropic noise to.
    sigma = np.cov(pooled[:, CURSOR_OUTPUTS].astype(np.float64), rowvar=False)

    # ── Variance decomposition (law of total variance), in pixels²:
    #    within-key  = driven by PRIVATE history   (the leak we care about)
    #    between-key = driven by WHICH command      (overt, less private)
    within_dx = float(np.mean(np.square(grp_dx_std)))
    within_dy = float(np.mean(np.square(grp_dy_std)))
    between_dx = float(np.var(grp_dx_mean))
    between_dy = float(np.var(grp_dy_mean))

    return {
        "weights_crc32": read_crc32(weights_file),
        "per_key": per_key,
        "n_samples": int(pooled.shape[0]),
        "n_keys": len(SINGLE_KEYS),
        # command channel (LOW expected)
        "cmd_history_bits_mean": float(np.mean(grp_cmd_bits)),
        "cmd_history_bits_max": float(np.max(grp_cmd_bits)),
        # cursor channel (HIGH expected), pixels
        "cursor_hist_std_px": {"dx": float(np.sqrt(within_dx)),
                               "dy": float(np.sqrt(within_dy))},
        "cursor_between_std_px": {"dx": float(np.sqrt(between_dx)),
                                  "dy": float(np.sqrt(between_dy))},
        "cursor_history_var_frac": {
            "dx": within_dx / (within_dx + between_dx + 1e-12),
            "dy": within_dy / (within_dy + between_dy + 1e-12),
        },
        # raw signal variance (px²) attributable to private history — the
        # quantity M1 will noise against, per channel.
        "_within_px2": {"dx": within_dx, "dy": within_dy},
        "cursor_sigma_raw": sigma.tolist(),
    }


def mi_bridge(within_px2):
    """Illustrative ONLY — no noise is applied. For candidate Gaussian noise
    σ (in pixels), the PAC Gaussian-MI upper bound on private-history leakage
    through one cursor channel is 0.5*log2(1 + signal_var/σ²). Shows the σ a
    future M1 layer would need to hit a bits target. The mechanism here stays
    deterministic; this is a sizing aid, not a measurement of this run."""
    rows = []
    for sigma_px in (0.5, 1.0, 2.0, 4.0):
        bits_dx = 0.5 * math.log2(1 + within_px2["dx"] / (sigma_px ** 2))
        bits_dy = 0.5 * math.log2(1 + within_px2["dy"] / (sigma_px ** 2))
        rows.append((sigma_px, bits_dx, bits_dy))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # UTF-8 console so the ─/σ/Σ/± glyphs don't crash on legacy Windows (cp1252).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="DNOS leakage eval (Privacy M0)")
    ap.add_argument("--weights", default=str(ROOT / "weights.bin"))
    ap.add_argument("--per-key", type=int, default=64,
                    help="resampled private histories per command key")
    ap.add_argument("--json", default=None,
                    help="write metrics + cursor Σ to this file (for M1/CI)")
    ap.add_argument("--table", default=None,
                    help="write the compact leakage table the pacp crate reads "
                         "(dnos-pacp LeakageTable text format)")
    ap.add_argument("--max-history-bits", type=float, default=None,
                    help="optional gate: fail if mean command-channel history "
                         "leakage exceeds this many bits (regression guard)")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        sys.exit(f"[leakage_eval] weights not found: {args.weights} "
                 f"(run `python tools/train.py` first)")

    print(f"[leakage_eval] {args.per_key} held-out histories per key, "
          f"seed {EVAL_SEED:#x}")
    print("[leakage_eval] measuring only — NO noise is applied (that's M1).")
    m = evaluate(args.weights, args.per_key)

    print(f"\nprior: uniform over {m['n_keys']} command keys × random "
          f"held-out histories  (N={m['n_samples']})")

    print("\n── command channel (argmax) — leakage of PRIVATE history ──")
    print(f"  mean entropy over histories : {m['cmd_history_bits_mean']:.3f} bits")
    print(f"  worst key                   : {m['cmd_history_bits_max']:.3f} bits")
    print("  (≈0 ⇒ which command is drawn barely depends on history — the "
          "trained-in privacy win)")

    print("\n── cursor channel (dx=out20, dy=out22) — leakage of PRIVATE history ──")
    ch = m["cursor_hist_std_px"]; bw = m["cursor_between_std_px"]
    fr = m["cursor_history_var_frac"]
    print(f"  history-driven jitter (std) : dx ±{ch['dx']:.2f} px   "
          f"dy ±{ch['dy']:.2f} px   ← with the command held FIXED")
    print(f"  command-driven spread (std) : dx ±{bw['dx']:.2f} px   "
          f"dy ±{bw['dy']:.2f} px")
    print(f"  frac of cursor variance from private history : "
          f"dx {100*fr['dx']:.0f}%   dy {100*fr['dy']:.0f}%")

    print("\n── cursor covariance Σ (raw output units, [20,21,22,23]) ──")
    for row in m["cursor_sigma_raw"]:
        print("  [" + "  ".join(f"{v:10.1f}" for v in row) + "]")
    print("  (pooled over all command keys ⇒ dominated by which-command spread,")
    print("   not private history; 21/23 duplicate 20/22 and decode ignores them.")
    print("   M1 sizes noise to the WITHIN-history dx/dy variances above, which")
    print("   the --table output carries — not this pooled Σ.)")

    print("\n── noise sizing bridge (illustrative; NOT applied this run) ──")
    print("  σ (px)   MI bound dx (bits)   MI bound dy (bits)")
    for sigma_px, bdx, bdy in mi_bridge(m["_within_px2"]):
        print(f"   {sigma_px:4.1f}        {bdx:6.2f}             {bdy:6.2f}")
    print("  (0.5·log2(1+var/σ²): the σ a future layer needs to hit a target.)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(m, f, indent=2)
        print(f"\n[leakage_eval] wrote {args.json}")

    if args.table:
        # Compact format consumed by dnos-pacp LeakageTable::from_str.
        # within_out2 = within-history variance in OUTPUT units² (px² × 1024²),
        # since the cursor decode is (out - 16384) >> 10  (1024 units = 1 px).
        scale = (1 << 10) ** 2  # 1024²
        dx = round(m["_within_px2"]["dx"] * scale)
        dy = round(m["_within_px2"]["dy"] * scale)
        with open(args.table, "w") as f:
            f.write("# dnos-pacp leakage table v1\n")
            f.write(f"crc32 0x{m['weights_crc32']:08X}\n")
            f.write(f"within_out2 {dx} {dy}\n")
        print(f"[leakage_eval] wrote {args.table} "
              f"(crc 0x{m['weights_crc32']:08X}, within_out2 {dx} {dy})")

    if args.max_history_bits is not None:
        if m["cmd_history_bits_mean"] > args.max_history_bits:
            print(f"\n[leakage_eval] FAIL: command-channel history leakage "
                  f"{m['cmd_history_bits_mean']:.3f} > "
                  f"{args.max_history_bits} bits")
            sys.exit(1)
        print(f"\n[leakage_eval] PASS: command-channel history leakage "
              f"≤ {args.max_history_bits} bits")


if __name__ == "__main__":
    main()
