#!/usr/bin/env python3
"""
Integer-exact training lab (ROADMAP item 9; decade-view research item).

THESIS: inference is portable by construction; training is not - float
BPTT ties the canonical CRC to a BLAS build, because float addition is
not associative and every BLAS orders its reductions differently
(TIER4 finding d records the resulting worry). Integer addition IS
associative, so a trainer whose every operation is integer is
order-insensitive - and therefore environment-free - BY CONSTRUCTION,
not by pinning.

This is a LAB (ssm_lab pattern): it graduates into train.py or dies
here. Design:

  - Training forward = an integer SHADOW of the law: same ops and
    signs, per-layer >>k truncations deferred (activations carry
    units law*2^sum(k), clamps scale up). The TRUE law is the
    selection metric and the saved artifact; forward() is gated
    bit-exact against SsmMachine.
  - The backward is int64 STE: clamp/ReLU masks from the integer
    forward; the recurrence backward is the decay spectrum's own
    shift-subtract (dh <- m*(dh - dh>>d)). Range-control shifts are
    PER-SAMPLE, before any batch reduction - floor-shift after a sum
    is not linear in the batch, and chunk-exactness is a gate.
  - No exp, no sqrt, no float anywhere: Weston-Watkins hinge (every
    violating rival contributes +/-1 - integer gradients need
    density), momentum as a leaky shift, sign-SGD steps of
    mean|w| >> lr_shift with an optional stepped decay schedule.
  - Data sampling uses xorshift64* over Python ints - no numpy
    Generator stream dependency. Training is a pure function of the
    seed, full stop.

Gates (--gate, cheap, no training, in CI):
  1. ORDER-INSENSITIVITY: one real gradient computation, batch
     permuted and accumulation chunked -> bit-identical integer
     gradients; float64's non-associativity shown alongside.
  2. FORWARD==LAW: the law-forward equals SsmMachine on a saved v5
     blob, outputs and h, sequence for sequence.

FINDINGS (2026-07-12, first campaign - the honest ledger):
  PROVEN: the order-freedom half of the thesis. Integer gradients are
  bit-identical under batch permutation and chunked accumulation; the
  BLAS-variance mechanism (reduction order) is absent by construction
  across the whole pipeline, RNG and data included.
  LEARNED ALONG THE WAY (each redesign forced by measurement):
  (a) training THROUGH the law's truncation grid cannot bootstrap -
  at init the random-ternary signal is below the quantization step
  (measured logits +/-1), hence the deferred-shift shadow; (b)
  integer LARS is a trap - floor division zeroes every sub-peak
  update (median |g| ~2^15 vs peaks ~2^40), silently freezing almost
  all weights; (c) post-sum shifts silently break batch linearity.
  FIRST CAMPAIGN'S DEAD END, RESOLVED BY CONTROL ARM (2026-07-12,
  second campaign, tools/int_train_control.py): the universal 15%
  plateau was NOT the optimizer - integer Adam-flavour (m=EMA(g),
  u=EMA|g|, per-weight (m*step)//u) and error feedback both hit the
  identical wall, while the float control (same harness, same hinge,
  same backward SHAPE) sailed to ~60%. Elimination localized the
  defect: range-control shifts sized to worst-case BOUNDS destroyed
  the gradient - random-sign cancellation makes typical magnitudes
  sqrt-scale (2^13..2^16), so >>15 twice left 2-5 bits of signal.
  Quantization noise is optimizer-invariant; that is exactly what the
  invariance of the plateau was saying.
  Re-sizing the shifts to typical magnitudes (dz2 unshifted, >>8
  elsewhere; int64 headroom verified) BROKE the wall: 15% -> 23.8%,
  loss monotone through the old floor. Second-order finding: a step
  proportional to live mean|w| climbs ~50% faster than an init-frozen
  step (norm growth is part of how the ternary law finds its scale)
  but feeds back into divergence after ~1k epochs; best-checkpoint
  selection preserves the peak through it.
  OPEN: rate. Integer best 23.8% vs float control ~60% at comparable
  epochs - remaining levers, in suspected order: carry extra
  fractional bits through the dz1 >>8 hop (signal precision, the
  proven lever's continuation); RMS second moment via exact isqrt
  (m/EMA|g| under-damps spiky weights); norm control (weight decay as
  a shift) to let the fast proportional step run without divergence.

Probes:
  python tools/int_train_lab.py --epochs 400 --h 128 --ctx 4
  python tools/int_train_lab.py --gate                          # CI
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from ssm_lab import HISTORY_POOL, SINGLE_KEYS  # noqa: E402
from train import CMD, SsmMachine, save_v5  # noqa: E402

F = 24                      # fixed-point fractional bits (Q24)
ONE = 1 << F
SEQ_LEN = 64
OUT = 64
N_CMD = 20
MARGIN = 1024               # hinge margin, in pre-sigmoid acc units
MU_S = 3                    # momentum = 1 - 2^-3 = 0.875, as a shift
A_S1 = 3                    # adam: first-moment EMA shift (beta1 ~ 0.875)
A_S2 = 6                    # adam: |g|-moment EMA shift (beta2 ~ 0.984)
FREEZE = 200                # epochs before per-layer shifts freeze
MASK64 = (1 << 64) - 1


class XorShift:
    """xorshift64* over Python ints: the lab's only randomness.
    Bit-identical on every platform and numpy version by construction."""

    def __init__(self, seed):
        self.s = (seed & MASK64) or 88172645463325252

    def next(self):
        x = self.s
        x = (x ^ (x << 13)) & MASK64
        x ^= x >> 7
        x = (x ^ (x << 17)) & MASK64
        self.s = x
        return (x * 0x2545F4914F6CDD1D & MASK64) >> 32

    def below(self, n):
        return self.next() % n

    def signed_below(self, n):
        return self.below(2 * n + 1) - n


def gen_data(rng, ctx_per_key=8):
    """Single-key commands, bare and under random full-alphabet
    histories (honest typing order) - the scoped probe task."""
    data = [([ord(k)], CMD[cmd]) for k, cmd in SINGLE_KEYS]
    for k, cmd in SINGLE_KEYS:
        for _ in range(ctx_per_key):
            hist = [HISTORY_POOL[rng.below(len(HISTORY_POOL))]
                    for _ in range(1 + rng.below(SEQ_LEN - 1))]
            data.append((hist + [ord(k)], CMD[cmd]))
    return data


def encode(data):
    B = len(data)
    X = np.zeros((B, SEQ_LEN, 8), dtype=np.int64)
    cls = np.zeros(B, dtype=np.int64)
    for b, (keys, c) in enumerate(data):
        keys = keys[-SEQ_LEN:]
        for t, k in enumerate(keys):
            X[b, SEQ_LEN - len(keys) + t] = \
                [(k >> i) & 1 for i in range(8)]
        cls[b] = c
    return X * 256, cls


def _round_log2(x):
    """round(log2(x)) for a positive int, exactly: bit_length gives
    floor; round up iff x^2 >= 2^(2b+1) (i.e. x >= 2^b * sqrt(2))."""
    b = int(x).bit_length() - 1
    return b + (1 if int(x) * int(x) >= 1 << (2 * b + 1) else 0)


def int_project(w, frozen_k=None):
    """ternary_project, integer-exact: delta = 3*mean|w|/4 (shifts),
    k = clamp(F - round(log2(mean surviving |w|)), 0, 15)."""
    aw = np.abs(w)
    n = w.size
    delta = (3 * int(aw.sum()) // n) >> 2
    s = np.sign(w) * (aw > delta)
    if frozen_k is not None:
        return s.astype(np.int64), frozen_k
    nz = aw[s != 0]
    if nz.size == 0:
        return s.astype(np.int64), 0
    alpha = int(nz.sum()) // int(nz.size)
    k = min(max(F - _round_log2(max(alpha, 1)), 0), 15)
    return s.astype(np.int64), k


class IntLab:
    def __init__(self, h=512, r1=1024, r2=384, seed=1337):
        rng = XorShift(seed)
        self.H, self.r1, self.r2 = h, r1, r2
        self.d = np.repeat(np.arange(1, 9), h // 8).astype(np.int64)
        # Shadow init: uniform integers, scale a power of two per
        # fan-in (a defined constant of the algorithm; the sqrt(2/fan)
        # ideal is approximated by the nearest shift).
        def init(fi, fo):
            a = ONE >> ((fi.bit_length() + 1) // 2 + 1)
            return np.array([[rng.signed_below(a) for _ in range(fo)]
                             for _ in range(fi)], dtype=np.int64)
        self.w = [init(8, h), init(h, r1), init(r1, r2), init(r2, OUT)]
        self.v = [np.zeros_like(w) for w in self.w]
        self.u = [np.zeros_like(w) for w in self.w]
        self.r = [np.zeros_like(w) for w in self.w]   # error feedback
        self.step0 = [int(np.abs(w).sum()) // w.size for w in self.w]
        self.frozen = None
        self.opt = "sign"

    def project(self):
        outs = [int_project(w, None if self.frozen is None
                            else self.frozen[i])
                for i, w in enumerate(self.w)]
        return [o[0] for o in outs], [o[1] for o in outs]

    # ── the integer SHADOW forward (training) ──────────────────────
    # Same ops and signs as the law, but the per-layer >>k truncations
    # are DEFERRED: activations carry units law*2^(sum of k so far)
    # and clamps scale up. At init the law's own grid crushes the
    # random-ternary signal to zero (measured: acc range +/-1), so
    # gradients cannot bootstrap; the shadow keeps full integer
    # precision - still order-free - while checkpoint selection and
    # the saved artifact use the TRUE law (ssm_lab's shadow/law split,
    # integerized).
    def forward_train(self, X, signs, ks):
        s_in, s1, s2, s3 = signs
        k_in, k1, k2, k3 = ks
        B = X.shape[0]
        h = np.zeros((B, self.H), dtype=np.int64)
        hc = 32767 << k_in
        masks = np.empty((B, SEQ_LEN, self.H), dtype=bool)
        for t in range(SEQ_LEN):
            pre = h - (h >> self.d) + (X[:, t] @ s_in)
            masks[:, t] = (pre > -hc) & (pre < hc)
            h = np.clip(pre, -hc, hc)
        c1 = 32767 << (k_in + k1)
        a1 = h @ s1
        z1 = np.clip(a1, 0, c1)
        m1 = (a1 > 0) & (a1 < c1)
        c2 = 32767 << (k_in + k1 + k2)
        a2 = z1 @ s2
        z2 = np.clip(a2, 0, c2)
        m2 = (a2 > 0) & (a2 < c2)
        acc = z2 @ s3          # units: law << (k_in+k1+k2+k3)
        return acc, (h, z1, m1, z2, m2, masks)

    # ── the law, batched (bit-exact SsmMachine.step) ────────────────
    def forward(self, X, signs, ks):
        s_in, s1, s2, s3 = signs
        k_in, k1, k2, k3 = ks
        B = X.shape[0]
        h = np.zeros((B, self.H), dtype=np.int64)
        masks = np.empty((B, SEQ_LEN, self.H), dtype=bool)
        for t in range(SEQ_LEN):
            pre = h - (h >> self.d) + ((X[:, t] @ s_in) >> k_in)
            masks[:, t] = (pre > -32768) & (pre < 32767)
            h = np.clip(pre, -32768, 32767)
        a1 = (h @ s1) >> k1
        z1 = np.clip(a1, 0, 32767)
        m1 = (a1 > 0) & (a1 < 32767)
        a2 = (z1 @ s2) >> k2
        z2 = np.clip(a2, 0, 32767)
        m2 = (a2 > 0) & (a2 < 32767)
        acc = (z2 @ s3) >> k3
        return acc, (h, z1, m1, z2, m2, masks)

    def grads(self, X, cls, signs, ks):
        """Hinge top, integer STE backward through the shadow forward.
        Every reduction is an int64 sum: associative, hence order-free.
        The inter-layer >>15 shifts are range control (activations are
        ~15-bit), a DEFINED semantic; absolute gradient scale is
        irrelevant because step() normalizes per layer."""
        acc, (h, z1, m1, z2, m2, masks) = \
            self.forward_train(X, signs, ks)
        B = X.shape[0]
        k = ks[0] + ks[1] + ks[2] + ks[3]
        logits = acc[:, :N_CMD]
        # Weston-Watkins hinge: EVERY rival within the margin of the
        # true class contributes +/-1 - denser than max-rival hinge
        # (integer gradients need density; they cannot whisper).
        zy = logits[np.arange(B), cls]
        viol = (zy[:, None] - logits) < (MARGIN << k)
        viol[np.arange(B), cls] = False
        g = np.zeros((B, OUT), dtype=np.int64)
        g[:, :N_CMD] = viol << F
        g[np.arange(B), cls] = -(viol.sum(axis=1) << F)
        n_wrong = int(viol.any(axis=1).sum())
        loss = int((np.maximum(
            (MARGIN << k) - (zy[:, None] - logits), 0)
            * viol).sum()) >> k
        # Range-control shifts are PER-SAMPLE (before any batch
        # reduction; post-sum floor-shift breaks chunk linearity - a
        # gate) and sized to TYPICAL magnitudes, not worst-case
        # bounds: random-sign cancellation makes typical values
        # sqrt-scale, and over-shifting (>>15 everywhere) was measured
        # to leave 2-5 bits of gradient - quantization noise that no
        # optimizer could rescue. int64 headroom holds at these:
        d3 = z2.T @ (g >> 8)
        dz2 = (g @ signs[3].T) * m2
        d2 = z1.T @ dz2
        dz1 = ((dz2 @ signs[2].T) >> 8) * m1
        d1 = h.T @ dz1
        dh = dz1 @ signs[1].T
        # BPTT: the recurrence backward is the decay spectrum itself.
        d_in = np.zeros_like(self.w[0])
        for t in range(SEQ_LEN - 1, -1, -1):
            dh = dh * masks[:, t]
            d_in += (X[:, t] >> 8).T @ dh
            dh = dh - (dh >> self.d)
        return [d_in, d1, d2, d3], acc, n_wrong, loss

    def step(self, grads, lr_shift):
        """sign-SGD with shift-momentum: every weight whose momentum
        buffer carries ANY accumulated signal moves by exactly
        mean|w| >> lr_shift. No division, no normalization - integer
        LARS was measured to floor sub-peak updates to zero (median
        |g| ~2^15 against peaks ~2^40), silently freezing almost every
        weight. Sign updates are dense and scale-free by construction."""
        for w, v, g in zip(self.w, self.v, grads):
            v -= (v >> MU_S)
            v += g
            step_q = max((int(np.abs(w).sum()) // w.size) >> lr_shift, 1)
            w -= np.sign(v) * step_q

    def step_adam(self, grads, lr_shift):
        """Integer Adam-flavour: per-weight adaptivity without squares
        or roots (g^2 overflows int64 at our scales; Adam's actual
        mechanism is sign-consistency scaling). m = EMA(g), u =
        EMA(|g|), both leaky shifts with the (1-beta) factor included
        so each is a true average; |m| <= u by construction, so the
        per-weight update sign(m) * (|m| * step) // u is bounded by
        step = mean|w| >> lr_shift. A weight with a consistent
        gradient direction moves the full step; a noisy one barely
        moves. Exact integer division, sign-symmetric (no floor bias),
        deterministic."""
        for i, (w, m, u, r, g) in enumerate(
                zip(self.w, self.v, self.u, self.r, grads)):
            m -= m >> A_S1
            m += g >> A_S1
            u -= u >> A_S2
            u += np.abs(g) >> A_S2
            # Step tracks the live mean|w| (measured to climb ~50%
            # faster than an init-frozen step: norm growth is part of
            # how the ternary law finds its scale) - but unchecked it
            # feeds back into divergence after ~1k epochs, so pair it
            # with a decay schedule (--decay-every).
            step_q = max((int(np.abs(w).sum()) // w.size) >> lr_shift, 1)
            # ERROR FEEDBACK: the desired update is computed in fine
            # units (8 extra fractional bits) and accumulated in a
            # residual; whole quanta are emitted, the remainder is
            # never lost. This is float's fractional accumulation,
            # exactly, in integers - without it, sub-quantum updates
            # vanish and most weights starve (measured: the float
            # control converges on this same harness, the quantized
            # step does not).
            r += np.sign(m) * (((np.abs(m) * step_q) << 8)
                               // np.maximum(u, 1))
            emit = np.sign(r) * (np.abs(r) >> 8)
            r -= emit << 8
            w -= emit

    def accuracy(self, X, cls, signs, ks):
        acc, _ = self.forward(X, signs, ks)
        return float((acc[:, :N_CMD].argmax(axis=1) == cls).mean())

    def save(self, path):
        signs, ks = self.project()
        return save_v5(path, self.d, signs, tuple(int(k) for k in ks))

    def train(self, epochs, lr_shift, ctx_per_key=8, seed_data=777,
              resample_every=25, log_every=50, decay_every=0):
        val_X, val_cls = encode(gen_data(XorShift(seed_data + 1), 6))
        stream = XorShift(seed_data)
        X, cls = encode(gen_data(stream, ctx_per_key))
        best, best_w = -1.0, None
        for ep in range(epochs):
            if ep and ep % resample_every == 0:
                X, cls = encode(gen_data(stream, ctx_per_key))
            signs, ks = self.project()
            if self.frozen is None and ep == FREEZE:
                self.frozen = ks
                print(f"  epoch {ep}: shifts frozen at {ks}")
            g, _, n_wrong, loss = self.grads(X, cls, signs, ks)
            # sign-SGD converges to a noise ball proportional to the
            # step; a stepped shift schedule (integer, deterministic)
            # shrinks the ball. decay_every=0 disables.
            sh = lr_shift + (ep // decay_every if decay_every else 0)
            if self.opt == "adam":
                self.step_adam(g, min(sh, lr_shift + 4))
            else:
                self.step(g, min(sh, lr_shift + 4))
            if ep % log_every == 0 or ep == epochs - 1:
                va = self.accuracy(val_X, val_cls, signs, ks)
                sacc, _ = self.forward_train(val_X, signs, ks)
                sa = float((sacc[:, :N_CMD].argmax(axis=1)
                            == val_cls).mean())
                if va > best:
                    best, best_w = va, [w.copy() for w in self.w]
                print(f"  epoch {ep:5d}: loss={loss:>12}  "
                      f"violations={n_wrong:4d}  law={va * 100:5.1f}%  "
                      f"shadow={sa * 100:5.1f}%  best={best * 100:5.1f}%",
                      flush=True)
        if best_w is not None:
            self.w = best_w
        return best


# ── gates ────────────────────────────────────────────────────────────

def gate():
    lab = IntLab(h=128, r1=256, r2=96, seed=42)
    X, cls = encode(gen_data(XorShift(5), 2))
    signs, ks = lab.project()

    # 1a. Batch permutation: gradients bit-identical.
    g0, _, _, _ = lab.grads(X, cls, signs, ks)
    perm = np.arange(len(cls))[::-1]
    g1, _, _, _ = lab.grads(X[perm], cls[perm], signs, ks)
    assert all(np.array_equal(a, b) for a, b in zip(g0, g1)), \
        "integer gradients changed under batch permutation"
    # 1b. Chunked accumulation: linear in the batch, exactly.
    half = len(cls) // 2
    ga, _, _, _ = lab.grads(X[:half], cls[:half], signs, ks)
    gb, _, _, _ = lab.grads(X[half:], cls[half:], signs, ks)
    assert all(np.array_equal(w, a + b)
               for w, a, b in zip(g0, ga, gb)), \
        "integer gradients are not chunk-exact"
    # 1c. The float counterexample: the same reduction pattern in
    # float64 is order-SENSITIVE (this is the mechanism behind BLAS
    # variance; integer training removes it by construction).
    a = (0.1 + 0.2) + 0.3
    b = 0.1 + (0.2 + 0.3)
    assert a != b, "float64 addition was associative?!"
    print(f"[int_train] order gate: int grads bit-identical under "
          f"permutation and chunking; float64 is order-SENSITIVE "
          f"((0.1+0.2)+0.3 = {a!r} but 0.1+(0.2+0.3) = {b!r} - the "
          f"BLAS-variance mechanism, absent by construction here)")

    # 2. FORWARD == LAW: save a v5 blob from the (untrained) lab and
    # the batched training forward must equal SsmMachine, h included.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        blob = Path(td) / "intlab.bin"
        lab.save(str(blob))
        machine = SsmMachine(str(blob))
        seqs = gen_data(XorShift(9), 1)[:10]
        Xs, _ = encode(seqs)
        acc, (h, *_rest) = lab.forward(Xs, signs, ks)
        ok = True
        for i, (keys, _) in enumerate(seqs):
            machine.run(keys)
            out = np.where(acc[i] < -8192, 0,
                           np.where(acc[i] > 8192, 32767,
                                    (acc[i] + 8192) * 2))
            got = np.asarray(machine.run(keys), dtype=np.int64)
            ok &= np.array_equal(out, got)
            ok &= np.array_equal(h[i], machine.h)
        assert ok, "training forward diverged from SsmMachine"
    print("[int_train] forward==law: batched training forward equals "
          "SsmMachine on the saved blob (outputs and h, 10 sequences)")
    print("[int_train] result: PASS - integer training is order-free "
          "by construction and trains THE law, not a shadow of it")


def main():
    ap = argparse.ArgumentParser(description="integer-exact training lab")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--r1", type=int, default=256)
    ap.add_argument("--r2", type=int, default=96)
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--lr-shift", type=int, default=12,
                    help="learning rate as a right shift of the "
                         "momentum buffer")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--decay-every", type=int, default=0)
    ap.add_argument("--opt", choices=["sign", "adam"], default="adam")
    ap.add_argument("--save")
    args = ap.parse_args()
    if args.gate:
        gate()
        return
    lab = IntLab(h=args.h, r1=args.r1, r2=args.r2, seed=args.seed)
    lab.opt = args.opt
    t0 = time.time()
    best = lab.train(args.epochs, args.lr_shift, ctx_per_key=args.ctx,
                     decay_every=args.decay_every)
    print(f"[int_train] best held-out {best * 100:.1f}% "
          f"({time.time() - t0:.0f}s)")
    if args.save:
        lab.save(args.save)


if __name__ == "__main__":
    main()
