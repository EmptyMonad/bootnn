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


def _is_v5(weights_file):
    with open(weights_file, 'rb') as f:
        return f.read(3)[2] == 5


def _v5_batch_logits(weights_file, seqs):
    """Vectorized v5 law over left-zero-padded sequences (exact: zero
    drive on zero state is identity). Same arithmetic as SsmMachine,
    batched — the per-tick readout is skipped because only the final
    output is compared."""
    from train import SsmMachine
    m = SsmMachine(weights_file)
    win, w1, w2, w3 = m.mats
    k_in, k1, k2, k3 = m.shifts
    L = CONTEXT_EVENTS
    B = len(seqs)
    X = np.zeros((B, L, 8), dtype=np.int64)
    for b, keys in enumerate(seqs):
        keys = keys[-L:]
        for t, k in enumerate(keys):
            X[b, L - len(keys) + t] = [(k >> i) & 1 for i in range(8)]
    X *= 256
    h = np.zeros((B, m.H), dtype=np.int64)
    for t in range(L):
        hl = h - (h >> m.d)
        h = np.clip(hl + ((X[:, t] @ win) >> k_in), -32768, 32767)
    z1 = np.clip((h @ w1) >> k1, 0, 32767)
    z2 = np.clip((z1 @ w2) >> k2, 0, 32767)
    acc = (z2 @ w3) >> k3
    return np.where(acc < -8192, 0,
                    np.where(acc > 8192, 32767, (acc + 8192) * 2))


def evaluate(weights_file, per_key=50, rng=None):
    rng = rng or np.random.default_rng(EVAL_SEED)
    per_cmd_hits = {}
    per_cmd_total = {}
    confusions = {}
    v5 = _is_v5(weights_file)

    if v5:
        # Sequences in typing order (oldest first), trigger key last.
        seqs, expected_all, names = [], [], []
        for key, cmd_name in sorted(SINGLE_KEYS.items()):
            for _ in range(per_key):
                hist_len = int(rng.integers(0, CONTEXT_EVENTS))
                hist = [int(rng.choice(HISTORY_ALPHABET))
                        for _ in range(hist_len)]
                seqs.append(hist + [ord(key)])
                expected_all.append(CMD[cmd_name])
                names.append(cmd_name)
        out = _v5_batch_logits(weights_file, seqs)
        got_all = np.argmax(out[:, :20], axis=1)
        for cmd_name, exp, got in zip(names, expected_all, got_all):
            per_cmd_total[cmd_name] = per_cmd_total.get(cmd_name, 0) + 1
            if got == exp:
                per_cmd_hits[cmd_name] = per_cmd_hits.get(cmd_name, 0) + 1
            else:
                per_cmd_hits.setdefault(cmd_name, 0)
                pair = (cmd_name, CMD_NAMES[int(got)])
                confusions[pair] = confusions.get(pair, 0) + 1
        total = sum(per_cmd_total.values())
        correct = sum(per_cmd_hits.values())
        return correct, total, per_cmd_hits, per_cmd_total, confusions

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
