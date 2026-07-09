#!/usr/bin/env python3
"""
Tier 4a format gate: the ternary law is bit-exact and self-rejecting,
independent of any trained blob.

  1. Bit-exactness: for seeded untrained networks (canonical and wide
     topologies), the packed weight file replayed through
     simulate_assembly_forward equals the in-memory integer law,
     output for output. The packed file is parsed by the same code
     every consumer uses, so this covers pack/unpack agreement, the
     header contract (sizes and shifts read from the header, not from
     module constants), and sum-then-sar semantics.
  2. Reserved-code rejection: 2-bit code 10 must never be written
     (writer asserts) and must never be accepted (a blob with an
     injected 10 is refused before any inference).

Runs in seconds; the full training gate is separate.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from train import (HEADER_SIZE, Tier2Network,  # noqa: E402
                   generate_training_data, simulate_assembly_forward)

N_EXAMPLES = 30


def bit_exact(net, path, X):
    net.save(str(path))
    z3 = net._exact_int_logits(X)
    mem = np.where(z3 < -8192, 0,
                   np.where(z3 > 8192, 32767, (z3 + 8192) * 2))
    for i in range(len(X)):
        sim = np.asarray(simulate_assembly_forward(str(path), X[i]),
                         dtype=np.int64)
        if not np.array_equal(sim, mem[i].astype(np.int64)):
            return False
    return True


def main():
    data = generate_training_data()[:N_EXAMPLES]
    X = np.array([d[0] for d in data], dtype=np.float32)
    tmp = Path(tempfile.mkdtemp())
    failures = []

    for name, hidden in [("canonical", (1024, 384)), ("wide", (2048, 512))]:
        np.random.seed(1234)
        net = Tier2Network(quant='ternary', hidden=hidden)
        ok = bit_exact(net, tmp / f"{name}.bin", X)
        print(f"[ternary_format] {name} {hidden}: "
              f"{'BIT-EXACT' if ok else 'DIVERGED'} "
              f"({N_EXAMPLES} examples, file vs memory)")
        if not ok:
            failures.append(f"{name}: packed file diverges from the law")

    # Reserved code 10: inject into the first weight byte and expect
    # refusal before inference.
    blob = bytearray((tmp / "canonical.bin").read_bytes())
    blob[HEADER_SIZE] = (blob[HEADER_SIZE] & ~0x3) | 0x2
    bad = tmp / "reserved.bin"
    bad.write_bytes(blob)
    try:
        simulate_assembly_forward(str(bad), X[0])
        failures.append("reserved code 10 was accepted")
        print("[ternary_format] reserved code 10: ACCEPTED (must refuse)")
    except ValueError:
        print("[ternary_format] reserved code 10: refused before inference")

    if failures:
        print("[ternary_format] result: FAIL")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("[ternary_format] result: PASS - the packed format IS the law")


if __name__ == "__main__":
    main()
