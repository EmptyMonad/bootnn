#!/usr/bin/env python3
"""
S3 spike: branching-leaf specialists, structure-first
(docs/SWARM_DESIGN.md S3; ROADMAP item 5).

Two 512-channel v5 leaves behind a HARD deterministic router — no
blending, no softmax, no parameters in the routing decision. The
router is a static total function key -> leaf, so the path is a pure
function of the event log and replay retraces it bit-exactly.

Structure under test (width held at 512; topology is the only change):

  event ->  route(key)  ->  leaf 0 (draw specialist)   ; its own h
                        ->  leaf 1 (motion specialist) ; its own h

Each leaf SEES only its routed subsequence, in training and in
inference identically (the structural smoke test asserts ForestMachine
== isolated per-leaf machines, bit for bit, untrained — structure is
gated without any training).

Law shape (the question this spike answers empirically): each leaf
trains independently from its own seed and saves its own v5 blob with
its own CRC. The composite exists only as a MANIFEST — canonical JSON
of (router table, leaf CRCs) — whose CRC commits to the structure.
Replay therefore naturally produces ONE LAW PER LEAF; the composite
canonical law is a derived commitment over parts, not a training
artifact. Contributions of examples remain inline deltas targeting
whichever leaf their trigger routes to; creating/scoping a leaf is NOT
expressible as a delta and needs a structural claim field.

Spike scope: single-key commands + random contexts. Word commands
stay a generalist concern (letters of one word route to different
leaves; a phrase-level router is future structure, out of scope).

Usage:
  python tools/s3_forest.py --smoke          # structure gate, no training
  python tools/s3_forest.py --epochs 4000    # the spike (trains 2 leaves)
"""

import argparse
import json
import sys
from pathlib import Path
from zlib import crc32

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from ssm_lab import (HISTORY_POOL, SINGLE_KEYS, SsmLab,  # noqa: E402
                     encode)
from train import CMD, CMD_NAMES, SsmMachine, save_v5  # noqa: E402

# ── the router: a static TOTAL function, versioned by its content ──────
DRAW_KEYS = set("pd.brlh-v|ofcu ")          # trigger keys, draw/meta class
MOTION_KEYS = set("wksja+")                 # trigger keys, motion/color
ROUTER_VERSION = 1


def route(key):
    """key (int) -> leaf id. Hard, deterministic, total: trigger keys by
    class; everything else (history junk) by parity. No state, no
    parameters - the path is a pure function of the event."""
    c = chr(key) if 0 <= key < 128 else ""
    if c in DRAW_KEYS:
        return 0
    if c in MOTION_KEYS:
        return 1
    return key % 2


def leaf_view(keys, leaf):
    """The subsequence a leaf sees: exactly the events routed to it."""
    return [k for k in keys if route(k) == leaf]


def gen_scoped(leaf, ctx_rng=None, ctx_per_key=16):
    """Scoped training data: this leaf's trigger keys, each under random
    FULL-WORLD histories filtered to the leaf's view - training sees
    precisely what inference will see."""
    if ctx_rng is None:
        ctx_rng = np.random.default_rng(20260610)
    triggers = [(k, cmd) for k, cmd in SINGLE_KEYS
                if route(ord(k)) == leaf]
    data = []
    for k, cmd in triggers:
        data.append(([ord(k)], CMD[cmd], f"{k} -> {cmd}"))
    for k, cmd in triggers:
        for v in range(ctx_per_key):
            hist_len = int(ctx_rng.integers(1, 64))
            hist = [int(ctx_rng.choice(HISTORY_POOL))
                    for _ in range(hist_len)]
            seq = leaf_view(hist, leaf) + [ord(k)]
            data.append((seq, CMD[cmd], f"{k} -> {cmd} +ctx{v}"))
    return data


class ForestMachine:
    """Two stateful laws behind the router. step(key) advances EXACTLY
    ONE leaf (single path) and returns its decision."""

    def __init__(self, blob0, blob1):
        self.leaves = [SsmMachine(str(blob0)), SsmMachine(str(blob1))]

    def reset(self):
        for m in self.leaves:
            m.reset()

    def step(self, key):
        return self.leaves[route(key)].step(key)

    def run(self, keys):
        self.reset()
        out = None
        for k in keys:
            out = self.step(k)
        return out


def manifest_for(blobs):
    """The composite law: canonical JSON committing to router + leaves.
    Its CRC is DERIVED from parts - there is no monolithic artifact."""
    m = {"router_version": ROUTER_VERSION,
         "draw_keys": sorted(DRAW_KEYS), "motion_keys": sorted(MOTION_KEYS),
         "leaves": [{"leaf": i,
                     "crc": crc32(Path(b).read_bytes()) & 0xFFFFFFFF}
                    for i, b in enumerate(blobs)]}
    blob = json.dumps(m, sort_keys=True, separators=(",", ":"))
    return m, crc32(blob.encode()) & 0xFFFFFFFF


def eval_forest(machine_run, data):
    hits = 0
    for keys, cls, _ in data:
        out = machine_run(keys)
        hits += int(np.argmax(out[:20])) == cls
    return 100.0 * hits / len(data)


def heldout_scoped(leaf, per_key=50):
    rng = np.random.default_rng(99173)
    data = []
    for k, cmd in [(k, c) for k, c in SINGLE_KEYS if route(ord(k)) == leaf]:
        for _ in range(per_key):
            hl = int(rng.integers(0, 64))
            hist = [int(rng.choice(HISTORY_POOL)) for _ in range(hl)]
            data.append((hist + [ord(k)], CMD[cmd], k))
    return data


def smoke():
    """Structure gate, zero training: ForestMachine must equal isolated
    per-leaf machines fed their own views, bit for bit."""
    tmp = ROOT / "swarm_shots"
    tmp.mkdir(exist_ok=True)
    blobs = []
    for i in (0, 1):
        np.random.seed(4000 + i)
        lab = SsmLab(h=512, seed=4000 + i)
        p = tmp / f"smoke_leaf{i}.bin"
        lab.save(str(p))
        blobs.append(p)
    forest = ForestMachine(*blobs)
    iso = [SsmMachine(str(b)) for b in blobs]
    rng = np.random.default_rng(7)
    ok = True
    for _ in range(20):
        seq = [int(rng.choice(HISTORY_POOL))
               for _ in range(int(rng.integers(1, 64)))]
        out_f = forest.run(seq)
        views = [leaf_view(seq, i) for i in (0, 1)]
        last_leaf = route(seq[-1])
        out_i = iso[last_leaf].run(views[last_leaf])
        ok &= np.array_equal(np.asarray(out_f), np.asarray(out_i))
        # And the NON-deciding leaf's state must equal its isolated twin
        other = 1 - last_leaf
        iso[other].run(views[other])
        ok &= np.array_equal(forest.leaves[other].h, iso[other].h)
    m, ccrc = manifest_for(blobs)
    print(f"[s3] smoke: routed forest == isolated leaf views: "
          f"{'BIT-EXACT' if ok else 'DIVERGED'} (20 random logs)")
    print(f"[s3] composite commitment (derived): {ccrc:#010x} over "
          f"{[hex(x['crc']) for x in m['leaves']]}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="S3 branching-leaf spike")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--smoke", action="store_true",
                    help="structure gate only (no training)")
    args = ap.parse_args()

    if args.smoke:
        sys.exit(smoke())

    blobs = []
    for leaf in (0, 1):
        print(f"[s3] training leaf {leaf} "
              f"({'draw' if leaf == 0 else 'motion'} specialist, h=512)...")
        np.random.seed(1337 + leaf)
        stream = np.random.default_rng(20260610)
        lab = SsmLab(h=512, seed=1337 + leaf)
        lab.train(args.epochs, args.lr,
                  resample=lambda l=leaf: gen_scoped(l, stream),
                  val_data=gen_scoped(leaf, np.random.default_rng(555000)))
        p = ROOT / f"weights_leaf{leaf}.bin"
        lab.save(str(p))
        blobs.append(p)

    m, ccrc = manifest_for(blobs)
    (ROOT / "forest_manifest.json").write_text(
        json.dumps(m, sort_keys=True, indent=1) + "\n")
    forest = ForestMachine(*blobs)
    base = SsmMachine(str(ROOT / "weights_ssm.bin"))

    # Held-out per leaf (the leaf's own law shape) and composite.
    print("\n[s3] held-out (single-key + random full-world contexts):")
    all_data = []
    for leaf in (0, 1):
        data = heldout_scoped(leaf)
        all_data += data
        acc = eval_forest(forest.run, data)
        print(f"  leaf {leaf}: {acc:.1f}%  ({len(data)} cases)")
    comp = eval_forest(forest.run, all_data)
    gen = eval_forest(base.run, all_data)
    print(f"  composite (routed forest): {comp:.1f}%")
    print(f"  baseline (single 512 generalist, same suite): {gen:.1f}%")
    print(f"\n[s3] LAW SHAPE: one law PER LEAF "
          f"({', '.join(hex(x['crc']) for x in m['leaves'])}); composite "
          f"is a derived commitment {ccrc:#010x} over router+leaves - "
          f"not a training artifact.")
    ok = comp >= 95.0
    print(f"[s3] result: {'GATES MET' if ok else 'below gates'} "
          f"(composite vs 95% bar)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
