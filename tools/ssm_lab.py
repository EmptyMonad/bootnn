#!/usr/bin/env python3
"""
Tier 4 Part B feasibility lab: diagonal-SSM recurrent core with ternary
weights (docs/TIER4_DESIGN.md, "Part B integer law").

This is a LAB: it either graduates into train.py as format v5 or dies
here. It duplicates data-generation *content* from train.py on purpose,
because the lab fixes two things the window law cannot express:

  1. Recency is architectural: h[c] decays by lambda_c = 1 - 2^-d_c
     per tick (shift-subtract, exact in int16). Ternary weights encode
     WHAT to detect; the decay spectrum encodes WHEN it mattered.
     (Finding f: uniform-magnitude weights cannot express a fade-out
     across a spatial window.)
  2. Sequences are in honest typing order. The window generator encodes
     word patterns most-recent-first, so "box" was trained as typed
     "x,o,b" (the demo's own "line" entry contradicts it). Here a word
     is its letters, oldest first, label at the end.

Integer law (the law; float BPTT is its shadow):
    hl[c] = h[c] - (h[c] >> d[c])
    u[c]  = (x_raw @ s_in)[c] >> k_in          x_raw in {0, 256}^8
    h'[c] = clamp(hl + u, -32768, 32767)
    readout: Tier 4a ternary MLP on h' (H -> 1024 -> 384 -> 64),
    sum-then-sar per layer, ReLU clamps, piecewise sigmoid at the end.

Usage:
  python tools/ssm_lab.py --epochs 3000            # feasibility probe
  python tools/ssm_lab.py --epochs 3000 --h 256    # width study
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from train import CMD, CMD_NAMES, save_v5, ternary_project  # noqa: E402

FEAT = 8
SEQ_LEN = 64
OUT = 64
N_CMD = 20
ACT_MAX_RAW = 32767

SINGLE_KEYS = [
    ('p', 'pixel'), ('d', 'pixel'), ('.', 'pixel'), ('b', 'rect'),
    ('r', 'rect'), ('l', 'hline'), ('h', 'hline'), ('-', 'hline'),
    ('v', 'vline'), ('|', 'vline'), ('o', 'circle'), ('f', 'fill'),
    ('c', 'clear'), ('u', 'undo'), (' ', 'nop'), ('w', 'move_up'),
    ('k', 'move_up'), ('s', 'move_down'), ('j', 'move_down'),
    ('a', 'move_left'), ('+', 'color_next'),
]
WORDS = [
    ("box", 'rect'), ("rect", 'rect'), ("line", 'hline'),
    ("hline", 'hline'), ("vline", 'vline'), ("dot", 'pixel'),
    ("pixel", 'pixel'), ("fill", 'fill'), ("clear", 'clear'),
    ("cls", 'clear'), ("circ", 'circle'), ("ring", 'circle'),
    ("undo", 'undo'), ("color", 'color_next'), ("up", 'move_up'),
    ("down", 'move_down'), ("left", 'move_left'),
    ("right", 'move_right'), ("save", 'save'), ("text", 'text'),
]
HISTORY_POOL = sorted({ord(k) for k, _ in SINGLE_KEYS}
                      | {ord(c) for c in 'boxlinerectpixfilcsudwnghty'})


def key_feats(key):
    return [(key >> i) & 1 for i in range(FEAT)]


def gen_sequences(ctx_rng=None, ctx_per_key=16):
    """Examples as (keys oldest->newest, class, desc). Content mirrors
    train.generate_training_data; order is honest typing order."""
    if ctx_rng is None:
        ctx_rng = np.random.default_rng(20260610)
    data = []
    for k, cmd in SINGLE_KEYS:
        data.append(([ord(k)], CMD[cmd], f"{k} -> {cmd}"))
    for k, cmd in SINGLE_KEYS:
        for v in range(ctx_per_key):
            hist_len = int(ctx_rng.integers(1, SEQ_LEN))
            hist = [int(ctx_rng.choice(HISTORY_POOL))
                    for _ in range(hist_len)]
            data.append((hist + [ord(k)], CMD[cmd], f"{k} -> {cmd} +ctx{v}"))
    for k in 'bplvc':
        cmd = dict(SINGLE_KEYS)[k]
        data.append(([ord(k)] * 2, CMD[cmd], f"{k}{k} -> {cmd}"))
    for word, cmd in WORDS:
        data.append(([ord(c) for c in word], CMD[cmd], f"{word} -> {cmd}"))
    for i, k in enumerate('pboxline'):        # demo, as the demo types it
        expect = {'p': 'pixel', 'b': 'rect', 'x': 'rect', 'l': 'hline',
                  'e': 'hline'}.get(k)
        if expect:
            data.append(([ord(c) for c in 'pboxline'[:i + 1]], CMD[expect],
                         f"demo[{i}] {k} -> {expect}"))
    return data


def encode(data):
    """Left-pad to SEQ_LEN with zero events (zero drive on zero state is
    identity, so padding is exact). Returns X (B,L,8) float 0/1, cls."""
    B = len(data)
    X = np.zeros((B, SEQ_LEN, FEAT), dtype=np.float32)
    cls = np.zeros(B, dtype=np.int64)
    for b, (keys, c, _) in enumerate(data):
        keys = keys[-SEQ_LEN:]
        for t, k in enumerate(keys):
            X[b, SEQ_LEN - len(keys) + t] = key_feats(k)
        cls[b] = c
    return X, cls


class SsmLab:
    def __init__(self, h=512, thresh=0.75, seed=1337, r1=1024, r2=384):
        # Readout widths r1/r2 are instance parameters: the v5 format
        # reads topology from the header and SsmMachine follows it, so
        # equal-total-params comparisons can size the readout, not
        # just h. Defaults preserve the canonical rng draw order
        # exactly (same shapes, same draws).
        rng = np.random.RandomState(seed)
        self.H = h
        # Structural decay spectrum: 8 log-spaced horizons, equal blocks.
        self.d = np.repeat(np.arange(1, 9), h // 8).astype(np.int64)
        self.lam = (1.0 - 2.0 ** -self.d).astype(np.float64)
        self.thresh = thresh
        self.win = rng.randn(FEAT, h) * np.sqrt(2.0 / FEAT) * 0.5
        self.w1 = rng.randn(h, r1) * np.sqrt(2.0 / h) * 0.5
        self.w2 = rng.randn(r1, r2) * np.sqrt(2.0 / r1) * 0.5
        self.w3 = rng.randn(r2, OUT) * np.sqrt(2.0 / r2) * 0.5
        self._frozen = None

    def _proj(self, w, i):
        s, k = ternary_project(w, self.thresh)
        if self._frozen is not None:
            k = self._frozen[i]
        return s, k

    # ── the integer law ─────────────────────────────────────────────────
    def int_logits(self, X):
        """Stateful bit-exact twin: X (B,L,8) in {0,1} -> final logits."""
        (si, ki), (s1, k1), (s2, k2), (s3, k3) = (
            self._proj(w, i) for i, w in
            enumerate((self.win, self.w1, self.w2, self.w3)))
        si = si.astype(np.int64)
        Xr = (X * 256).astype(np.int64)            # raw Q8.8 features
        B = X.shape[0]
        hh = np.zeros((B, self.H), dtype=np.int64)
        for t in range(X.shape[1]):
            hl = hh - (hh >> self.d)
            u = (Xr[:, t] @ si) >> ki
            hh = np.clip(hl + u, -32768, 32767)
        z1 = np.clip((hh @ s1.astype(np.int64)) >> k1, 0, ACT_MAX_RAW)
        z2 = np.clip((z1 @ s2.astype(np.int64)) >> k2, 0, ACT_MAX_RAW)
        return (z2 @ s3.astype(np.int64)) >> k3

    def int_accuracy(self, X, cls):
        z3 = self.int_logits(X)
        out = np.where(z3 < -8192, 0,
                       np.where(z3 > 8192, 32767, (z3 + 8192) * 2))
        return 100 * np.mean(np.argmax(out[:, :N_CMD], axis=1) == cls)

    def save(self, filename):
        """Emit the projected law as a format v5 blob (train.save_v5)."""
        projs = [self._proj(w, i) for i, w in
                 enumerate((self.win, self.w1, self.w2, self.w3))]
        return save_v5(filename, self.d, [p[0] for p in projs],
                       tuple(int(p[1]) for p in projs))

    # ── float shadow (BPTT) ─────────────────────────────────────────────
    def _ste(self, w, i):
        s, k = self._proj(w, i)
        return (2.0 ** -np.asarray(k, dtype=np.float64)) * s

    def train(self, epochs, lr, resample, resample_every=25,
              freeze=200, log_every=500, val_data=None):
        val = val_data or gen_sequences(np.random.default_rng(555000))
        Xv, cv = encode(val)
        data = resample()
        X, cls = encode(data)
        ws = [self.win, self.w1, self.w2, self.w3]
        ms = [np.zeros_like(w) for w in ws]
        vs = [np.zeros_like(w) for w in ws]
        b1, b2, eps = 0.9, 0.999, 1e-8
        best_acc, best = 0, None

        for ep in range(epochs):
            if self._frozen is None and ep == freeze:
                self._frozen = [
                    ternary_project(w, self.thresh)[1] for w in ws]
                print(f"  Epoch {ep:5d}: shifts frozen at sar {self._frozen}")
            if ep and ep % resample_every == 0:
                X, cls = encode(resample())
            wi, w1, w2, w3 = (self._ste(w, i) for i, w in enumerate(ws))
            B = len(X)

            # forward through time (float, 1.0 == raw 256)
            hs = np.zeros((B, self.H))
            drive = X @ wi                        # (B,L,H), precomputed
            H_last = None
            for t in range(SEQ_LEN):
                hs = np.clip(self.lam * hs + drive[:, t], -128, 128)
            H_last = hs
            z1 = H_last @ w1
            a1 = np.clip(z1, 0, 127.996)
            z2 = a1 @ w2
            a2 = np.clip(z2, 0, 127.996)
            logits = a2 @ w3

            l20 = logits[:, :N_CMD]
            l20 = l20 - l20.max(axis=1, keepdims=True)
            ex = np.exp(l20)
            sm = ex / ex.sum(axis=1, keepdims=True)
            loss = -np.mean(np.log(sm[np.arange(B), cls] + 1e-9))
            dlog = np.zeros_like(logits)
            dlog[:, :N_CMD] = sm
            dlog[np.arange(B), cls] -= 1.0
            dlog /= B
            CAP, CL = 28.0, 1e-3
            over = np.clip(np.abs(logits) - CAP, 0, None)
            loss += CL * np.mean(np.sum(over ** 2, axis=1))
            dlog += CL * 2.0 * over * np.sign(logits) / B

            dw3 = a2.T @ dlog
            da2 = dlog @ w3.T
            dz2 = da2 * ((z2 > 0) & (z2 < 127.996))
            dw2 = a1.T @ dz2
            da1 = dz2 @ w2.T
            dz1 = da1 * ((z1 > 0) & (z1 < 127.996))
            dw1 = H_last.T @ dz1
            dh = dz1 @ w1.T
            # back through time: dh_{t-1} = lam*dh_t; dWin += x_t^T dh_t
            dwi = np.zeros_like(self.win)
            for t in range(SEQ_LEN - 1, -1, -1):
                dwi += X[:, t].T @ dh
                dh = dh * self.lam
            cur_lr = lr * (0.5 if ep > epochs * 2 // 3 else 1.0)
            for i, (w, dw) in enumerate(zip(ws, (dwi, dw1, dw2, dw3))):
                ms[i] = b1 * ms[i] + (1 - b1) * dw
                vs[i] = b2 * vs[i] + (1 - b2) * dw ** 2
                w -= cur_lr * (ms[i] / (1 - b1 ** (ep + 1))) / \
                    (np.sqrt(vs[i] / (1 - b2 ** (ep + 1))) + eps)

            if ep % 100 == 0 or ep == epochs - 1:
                acc = self.int_accuracy(Xv, cv)
                if acc >= best_acc:
                    best_acc = acc
                    best = [w.copy() for w in ws]
                if ep % log_every == 0 or ep == epochs - 1:
                    print(f"  Epoch {ep:5d}: loss={loss:.4f}  "
                          f"heldout_int={acc:.1f}%  best={best_acc:.1f}%")
        if best:
            self.win, self.w1, self.w2, self.w3 = best
        print(f"  Best held-out (integer law): {best_acc:.1f}%")


def main():
    ap = argparse.ArgumentParser(description="Part B SSM feasibility lab")
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--h', type=int, default=512)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--save', help='write the trained law as a v5 blob')
    args = ap.parse_args()

    np.random.seed(args.seed)
    stream = np.random.default_rng(20260610)
    lab = SsmLab(h=args.h, seed=args.seed)
    print(f"[ssm_lab] h={args.h}, decay spectrum d=1..8, "
          f"readout {args.h}->1024->384->64, all ternary")
    lab.train(args.epochs, args.lr, resample=lambda: gen_sequences(stream))

    # ── gates, on the integer law only ──────────────────────────────────
    canon = gen_sequences()                       # canonical draw
    Xc, cc = encode(canon)
    acc_c = lab.int_accuracy(Xc, cc)
    n_bad = int(round((100 - acc_c) / 100 * len(canon)))
    print(f"[ssm_lab] canonical suite: {len(canon) - n_bad}/{len(canon)} "
          f"({acc_c:.1f}%)")
    if n_bad:
        z3 = lab.int_logits(Xc)
        out = np.where(z3 < -8192, 0,
                       np.where(z3 > 8192, 32767, (z3 + 8192) * 2))
        pred = np.argmax(out[:, :N_CMD], axis=1)
        for i in np.flatnonzero(pred != cc)[:8]:
            print(f"    x {canon[i][2]:28s} got={CMD_NAMES[pred[i]]}")

    erng = np.random.default_rng(99173)           # CI eval seed, held out
    ev = []
    for k, cmd in SINGLE_KEYS:
        for _ in range(50):
            hl = int(erng.integers(0, SEQ_LEN))
            hist = [int(erng.choice(HISTORY_POOL)) for _ in range(hl)]
            ev.append((hist + [ord(k)], CMD[cmd], k))
    Xe, ce = encode(ev)
    acc_e = lab.int_accuracy(Xe, ce)
    print(f"[ssm_lab] held-out streams: {acc_e:.1f}% "
          f"({int(acc_e / 100 * len(ev))}/{len(ev)})")
    words = [w for w in canon if w[2].split()[0] in dict(WORDS)]
    Xw, cw = encode(words)
    print(f"[ssm_lab] word commands (typing order): "
          f"{lab.int_accuracy(Xw, cw):.1f}% of {len(words)}")
    if args.save:
        lab.save(args.save)
    ok = acc_c == 100.0 and acc_e >= 95.0
    print(f"[ssm_lab] result: {'GATES MET' if ok else 'below gates'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
