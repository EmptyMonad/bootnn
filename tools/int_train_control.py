#!/usr/bin/env python3
"""CONTROL ARM for int_train_lab (diagnostic reference - deliberately
FLOAT, never a product path): the integer lab's exact harness - same
xorshift data, same shadow-forward structure, same Weston-Watkins
hinge, same STE backward SHAPE - but float64 arithmetic + float Adam.

Every integer-lab failure is judged against this: if the control
converges where the integer lab stalls, the wall is integer-specific;
if both stall, the harness or gradient path is the bug. It settled
the first campaign's plateau (control ~60% by epoch 800 while every
integer optimizer sat at 15%), localizing the defect to over-sized
integer range-control shifts - not the optimizer, loss, or task."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from int_train_lab import XorShift, gen_data, encode, SEQ_LEN, N_CMD, MARGIN, OUT
from train import ternary_project

H, R1, R2 = 128, 256, 96
rng = np.random.default_rng(7)
d = np.repeat(np.arange(1, 9), H // 8)
w = [rng.standard_normal((8, H)) * 0.35,
     rng.standard_normal((H, R1)) * 0.09,
     rng.standard_normal((R1, R2)) * 0.09,
     rng.standard_normal((R2, OUT)) * 0.1]
m = [np.zeros_like(x) for x in w]
v = [np.zeros_like(x) for x in w]
lam = 1.0 - 2.0 ** -d.astype(np.float64)

def project():
    outs = [ternary_project(x) for x in w]
    return [o[0].astype(np.float64) for o in outs], [o[1] for o in outs]

def forward(X, signs, ks):
    s_in, s1, s2, s3 = signs
    k_in, k1, k2, k3 = ks
    B = X.shape[0]
    h = np.zeros((B, H))
    hc = 32767.0 * 2 ** k_in
    masks = np.empty((B, SEQ_LEN, H), dtype=bool)
    for t in range(SEQ_LEN):
        pre = h * lam + X[:, t].astype(np.float64) @ s_in
        masks[:, t] = np.abs(pre) < hc
        h = np.clip(pre, -hc, hc)
    c1 = 32767.0 * 2 ** (k_in + k1)
    a1 = h @ s1; z1 = np.clip(a1, 0, c1); m1 = (a1 > 0) & (a1 < c1)
    c2 = 32767.0 * 2 ** (k_in + k1 + k2)
    a2 = z1 @ s2; z2 = np.clip(a2, 0, c2); m2 = (a2 > 0) & (a2 < c2)
    return z2 @ s3, (h, z1, m1, z2, m2, masks)

def grads(X, cls, signs, ks):
    acc, (h, z1, m1, z2, m2, masks) = forward(X, signs, ks)
    B = X.shape[0]
    k = sum(ks)
    logits = acc[:, :N_CMD]
    zy = logits[np.arange(B), cls]
    viol = (zy[:, None] - logits) < MARGIN * 2.0 ** k
    viol[np.arange(B), cls] = False
    g = np.zeros((B, OUT))
    g[:, :N_CMD] = viol
    g[np.arange(B), cls] = -viol.sum(axis=1)
    loss = float((np.maximum(MARGIN * 2.0 ** k
                             - (zy[:, None] - logits), 0) * viol).sum()
                 / 2.0 ** k)
    d3 = z2.T @ g
    dz2 = (g @ signs[3].T) * m2
    d2 = z1.T @ dz2
    dz1 = (dz2 @ signs[2].T) * m1
    d1 = h.T @ dz1
    dh = dz1 @ signs[1].T
    d_in = np.zeros_like(w[0])
    for t in range(SEQ_LEN - 1, -1, -1):
        dh = dh * masks[:, t]
        d_in += (X[:, t].astype(np.float64) / 256.0).T @ dh
        dh = dh * lam
    return [d_in, d1, d2, d3], loss, int(viol.any(axis=1).sum())

stream = XorShift(777)
X, cls = encode(gen_data(stream, 4))
val_X, val_cls = encode(gen_data(XorShift(778), 6))
LR, B1, B2, EPS = 3e-3, 0.9, 0.999, 1e-8
frozen = None
for ep in range(801):
    if ep and ep % 25 == 0:
        X, cls = encode(gen_data(stream, 4))
    signs, ks = project()
    if frozen is None and ep == 200:
        frozen = ks
        print(f"  freeze {ks}")
    if frozen is not None:
        ks = frozen
    g, loss, nw = grads(X, cls, signs, ks)
    for wi, mi, vi, gi in zip(w, m, v, g):
        mi *= B1; mi += (1 - B1) * gi
        vi *= B2; vi += (1 - B2) * gi * gi
        wi -= LR * mi / (np.sqrt(vi) + EPS)
    if ep % 50 == 0:
        acc, _ = forward(val_X, signs, ks)
        va = float((acc[:, :N_CMD].argmax(axis=1) == val_cls).mean())
        print(f"  ep {ep:4d}: loss={loss:12.0f} viol={nw:4d} val={va*100:5.1f}%")
