#!/usr/bin/env python3
"""
Context-generalization eval for DNOS (ROADMAP backlog #7).

The training set teaches single-key commands under sampled histories; this
eval measures how well those mappings hold under *held-out* random
histories the network has never seen — i.e. whether the law generalizes or
memorizes. Uses the assembly-exact simulator, so the number reflects what
the metal will do.

The eval set is seeded (different seed from training) and fixed, so the
score is comparable across training runs.

Usage:
  python tools/context_eval.py                  # weights.bin, 50 ctx/key
  python tools/context_eval.py --per-key 100 --min-accuracy 90
"""

import argparse
import string
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from train import (CMD, CMD_NAMES, CONTEXT_EVENTS,  # noqa: E402
                   sequence_to_input, simulate_assembly_forward)

EVAL_SEED = 99173  # independent of training seeds

# The single-key command contract (must mirror train.py single_keys).
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

# Histories are drawn from everything a user might plausibly type,
# including keys the network was never trained on.
HISTORY_ALPHABET = [ord(c) for c in
                    string.ascii_lowercase + string.digits + " .,-+|"]


def evaluate(weights_file, per_key=50, rng=None):
    rng = rng or np.random.default_rng(EVAL_SEED)
    per_cmd_hits = {}
    per_cmd_total = {}
    confusions = {}

    for key, cmd_name in sorted(SINGLE_KEYS.items()):
        expected = CMD[cmd_name]
        hits = 0
        for _ in range(per_key):
            hist_len = int(rng.integers(0, CONTEXT_EVENTS))
            hist = [int(rng.choice(HISTORY_ALPHABET)) for _ in range(hist_len)]
            inp = sequence_to_input([ord(key)] + hist)
            out = simulate_assembly_forward(weights_file, inp)
            got = int(np.argmax(out[:20]))
            if got == expected:
                hits += 1
            else:
                pair = (cmd_name, CMD_NAMES[got])
                confusions[pair] = confusions.get(pair, 0) + 1
        per_cmd_hits[cmd_name] = per_cmd_hits.get(cmd_name, 0) + hits
        per_cmd_total[cmd_name] = per_cmd_total.get(cmd_name, 0) + per_key

    total = sum(per_cmd_total.values())
    correct = sum(per_cmd_hits.values())
    return correct, total, per_cmd_hits, per_cmd_total, confusions


def main():
    ap = argparse.ArgumentParser(description="DNOS context-generalization eval")
    ap.add_argument("--weights", default=str(ROOT / "weights.bin"))
    ap.add_argument("--per-key", type=int, default=50,
                    help="held-out random histories per key")
    ap.add_argument("--min-accuracy", type=float, default=None,
                    help="exit non-zero below this percentage")
    args = ap.parse_args()

    print(f"[context_eval] {args.per_key} held-out random histories per key, "
          f"seed {EVAL_SEED}")
    correct, total, hits, totals, confusions = evaluate(
        args.weights, args.per_key)

    print(f"\n{'command':<12s} {'accuracy':>10s}")
    for cmd in sorted(totals, key=lambda c: hits[c] / totals[c]):
        pct = 100 * hits[cmd] / totals[cmd]
        print(f"{cmd:<12s} {hits[cmd]:>4d}/{totals[cmd]:<4d} {pct:5.1f}%")

    if confusions:
        print("\ntop confusions (expected -> got):")
        for (exp, got), n in sorted(confusions.items(),
                                    key=lambda kv: -kv[1])[:8]:
            print(f"  {exp:>12s} -> {got:<12s} x{n}")

    pct = 100 * correct / total
    print(f"\nContext-generalization accuracy: {correct}/{total} ({pct:.1f}%)")

    if args.min_accuracy is not None and pct < args.min_accuracy:
        print(f"[context_eval] FAIL: below {args.min_accuracy}%")
        sys.exit(1)


if __name__ == "__main__":
    main()
