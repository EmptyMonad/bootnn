#!/usr/bin/env python3
"""
S3 economics: do specialists beat a generalist at EQUAL TOTAL PARAMS?
(docs/SWARM_DESIGN.md S3 exit criterion; the open question left by the
spike.)

The spike held width (2x512ch leaves = ~2x the generalist's weights)
and used fewer epochs (4k vs 10k) - confounded in both directions.
This campaign removes both confounds:

  generalist:  the canonical v5 law - 946,176 ternary weights,
               10k epochs (already trained; scored, not retrained)
  specialists: 2 leaves, h=512, readout 608/232 = 471,296 weights
               each, 942,592 total - 99.6% of the generalist's budget
               (specialists argue from UNDER budget) - 10k epochs
               EACH on the spike's scoped data.

Suite = the spike's: scoped held-out per leaf (seed 99173), composite
= the routed forest, the generalist scored on the identical cases.

NOT a CI gate (hours of training). The verdict gets recorded in the
ROADMAP; the run is deterministic and replayable like everything else
- fixed seeds, fixed data streams, artifacts CRC'd.

Usage:
  python tools/s3_economics.py                # the campaign (hours)
  python tools/s3_economics.py --epochs 20    # plumbing smoke
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import s3_forest  # noqa: E402
from ssm_lab import SsmLab  # noqa: E402
from train import SsmMachine  # noqa: E402

H, R1, R2 = 512, 608, 232


def n_params(r1, r2):
    return 8 * H + H * r1 + r1 * r2 + r2 * 64


def main():
    ap = argparse.ArgumentParser(description="S3 equal-params economics")
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()

    total = 2 * n_params(R1, R2)
    gen_params = n_params(1024, 384)
    assert total <= gen_params, "specialists must argue from UNDER budget"

    blobs, t0 = [], time.time()
    for leaf in (0, 1):
        print(f"[econ] training leaf {leaf} "
              f"({'draw' if leaf == 0 else 'motion'} specialist, h={H}, "
              f"readout {R1}/{R2}, {n_params(R1, R2):,} weights, "
              f"{args.epochs} epochs)...", flush=True)
        np.random.seed(1337 + leaf)
        stream = np.random.default_rng(20260610)
        lab = SsmLab(h=H, seed=1337 + leaf, r1=R1, r2=R2)
        lab.train(args.epochs, args.lr,
                  resample=lambda l=leaf: s3_forest.gen_scoped(l, stream),
                  val_data=s3_forest.gen_scoped(
                      leaf, np.random.default_rng(555000)))
        p = ROOT / f"weights_leaf{leaf}_econ.bin"
        lab.save(str(p))
        blobs.append(p)
        print(f"[econ] leaf {leaf} done at t+{time.time() - t0:.0f}s",
              flush=True)

    print(f"\n[econ] budget: specialists {total:,} vs generalist "
          f"{gen_params:,} ({100.0 * total / gen_params:.1f}%)")
    forest = s3_forest.ForestMachine(blobs)
    base = SsmMachine(str(ROOT / "weights_ssm.bin"))
    all_data = []
    print("[econ] held-out (single-key + random full-world contexts):")
    for leaf in (0, 1):
        data = s3_forest.heldout_scoped(leaf)
        all_data += data
        acc = s3_forest.eval_forest(forest.run, data)
        print(f"  leaf {leaf}: {acc:.1f}%  ({len(data)} cases)")
    comp = s3_forest.eval_forest(forest.run, all_data)
    gen = s3_forest.eval_forest(base.run, all_data)
    m, ccrc = s3_forest.manifest_for(blobs)
    print(f"  composite (routed forest): {comp:.1f}%")
    print(f"  generalist (canonical law, same suite): {gen:.1f}%")
    print(f"[econ] manifest {ccrc:#010x} over "
          f"{[hex(x['crc']) for x in m['leaves']]}")
    verdict = "SPECIALISTS PAY" if comp >= gen else "GENERALIST HOLDS"
    print(f"[econ] verdict at equal total params, equal epochs: {verdict} "
          f"(composite {comp:.1f}% vs generalist {gen:.1f}%)")


if __name__ == "__main__":
    main()
