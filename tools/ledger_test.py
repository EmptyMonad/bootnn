#!/usr/bin/env python3
"""
S2 gate: mint-on-verified-replay, end to end.

  1. alice trains a (CI-sized) contribution and submits the claim.
     Verification replays the training: her CRC reproduces -> minted.
  2. mallory submits the same tuple with a falsified CRC ("I did the
     work, trust me"): the replay disagrees -> rejected, no mint.
  3. alice pays bob; bob cannot overdraw.
  4. Tampering with any byte of history breaks the hash chain: audit
     must fail loudly.
  5. A second verdict on an already-verdicted claim is refused
     (no double minting).

Determinism is the proof-of-training; the wallet is a fold over the
log; supply is a pure function of history.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from zlib import crc32

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "tools" / "ledger.py"
TRAIN = ROOT / "tools" / "train.py"

SEED, EPOCHS, LR = 4242, 30, 0.02


def led(ledger, *argv, expect_fail=False):
    r = subprocess.run([sys.executable, str(LEDGER), "--ledger",
                        str(ledger), *argv],
                       capture_output=True, text=True, timeout=3600)
    for ln in (r.stdout + r.stderr).strip().splitlines():
        print(f"  {ln}")
    if (r.returncode != 0) != expect_fail:
        print(f"[ledger_test] result: FAIL (exit {r.returncode}, "
              f"expected {'non-zero' if expect_fail else '0'}: {argv})")
        sys.exit(1)
    return r.stdout


def balances(ledger):
    out = led(ledger, "balances")
    return {ln.split(":")[0]: int(ln.split(":")[1])
            for ln in out.strip().splitlines() if ":" in ln
            and not ln.startswith("supply")}


def main():
    tmp = Path(tempfile.mkdtemp())
    ledger = tmp / "ledger.jsonl"

    # ── alice actually does the work ─────────────────────────────────────
    blob = tmp / "alice.bin"
    print(f"[ledger_test] alice trains (seed={SEED} epochs={EPOCHS})...")
    subprocess.run([sys.executable, str(TRAIN), "--seed", str(SEED),
                    "--epochs", str(EPOCHS), "--lr", str(LR),
                    "--output", str(blob)],
                   capture_output=True, timeout=3600)
    crc = crc32(blob.read_bytes()) & 0xFFFFFFFF
    print(f"[ledger_test] alice's blob crc={crc:#010x}")

    # ── honest but WEAK: replay reproduces, quality bar rejects ──────────
    # Reproducibility is not worthiness: a 30-epoch blob is honest work
    # and still mints nothing at the default bar.
    led(ledger, "submit", "--account", "alice", "--seed", str(SEED),
        "--epochs", str(EPOCHS), "--lr", str(LR), "--claimed-crc", hex(crc))
    out = led(ledger, "verify", "--claim", "0")
    if "below quality bar" not in out:
        print("[ledger_test] result: FAIL (weak claim not quality-rejected)")
        sys.exit(1)
    if balances(ledger).get("alice", 0) != 0:
        print("[ledger_test] result: FAIL (weak claim minted)")
        sys.exit(1)

    # ── same work under an explicit policy bar: mints ────────────────────
    led(ledger, "submit", "--account", "alice", "--seed", str(SEED),
        "--epochs", str(EPOCHS), "--lr", str(LR), "--claimed-crc", hex(crc))
    out = led(ledger, "verify", "--claim", "2", "--min-heldout", "20")
    if "VERIFIED" not in out:
        print("[ledger_test] result: FAIL (honest claim not verified)")
        sys.exit(1)

    # ── mallory claims the same work with a falsified CRC ────────────────
    led(ledger, "submit", "--account", "mallory", "--seed", str(SEED),
        "--epochs", str(EPOCHS), "--lr", str(LR),
        "--claimed-crc", hex(crc ^ 1))
    out = led(ledger, "verify", "--claim", "4")
    if "dishonest" not in out:
        print("[ledger_test] result: FAIL (false claim not rejected)")
        sys.exit(1)

    b = balances(ledger)
    if b.get("alice") != 100 or b.get("mallory", 0) != 0:
        print(f"[ledger_test] result: FAIL (balances {b})")
        sys.exit(1)

    # ── transfers: valid applies, overdraw refused ───────────────────────
    led(ledger, "transfer", "--src", "alice", "--dst", "bob",
        "--amount", "40")
    led(ledger, "transfer", "--src", "bob", "--dst", "alice",
        "--amount", "1000", expect_fail=True)
    b = balances(ledger)
    if b.get("alice") != 60 or b.get("bob") != 40:
        print(f"[ledger_test] result: FAIL (post-transfer balances {b})")
        sys.exit(1)

    # ── dataset-delta claim: contribute new EXAMPLES, not just knobs ─────
    # carol teaches a key the base set never had; the delta rides inline
    # in the claim, so a verifier reproduces her exact CRC by replay.
    delta = tmp / "carol_delta.jsonl"
    delta.write_text(json.dumps({"keys": [ord("q")], "cmd": "rect"}) + "\n")
    carol_blob = tmp / "carol.bin"
    subprocess.run([sys.executable, str(TRAIN), "--seed", str(SEED),
                    "--epochs", str(EPOCHS), "--lr", str(LR),
                    "--data-delta", str(delta), "--output", str(carol_blob)],
                   capture_output=True, timeout=3600)
    ccrc = crc32(carol_blob.read_bytes()) & 0xFFFFFFFF
    # The delta must actually move the artifact (else it is not a
    # contribution): a delta'd blob differs from the no-delta blob.
    if ccrc == crc:
        print("[ledger_test] result: FAIL (delta did not change the law)")
        sys.exit(1)
    led(ledger, "submit", "--account", "carol", "--seed", str(SEED),
        "--epochs", str(EPOCHS), "--lr", str(LR),
        "--data-delta", str(delta), "--claimed-crc", hex(ccrc))
    out = led(ledger, "verify", "--claim", "7", "--min-heldout", "0")
    if "VERIFIED" not in out:
        print("[ledger_test] result: FAIL (delta claim not verified by replay)")
        sys.exit(1)
    if balances(ledger).get("carol", 0) != 100:
        print("[ledger_test] result: FAIL (delta claim did not mint)")
        sys.exit(1)

    # ── no double mint ───────────────────────────────────────────────────
    led(ledger, "verify", "--claim", "2", expect_fail=True)

    # ── audit passes clean, fails tampered ───────────────────────────────
    led(ledger, "audit")
    lines = ledger.read_text().splitlines()
    e = json.loads(lines[0])
    e["data"]["account"] = "eve"          # rewrite history
    lines[0] = json.dumps(e, sort_keys=True, separators=(",", ":"))
    tampered = tmp / "tampered.jsonl"
    tampered.write_text("\n".join(lines) + "\n")
    led(tampered, "audit", expect_fail=True)

    print("[ledger_test] result: PASS - work mints, lies don't, "
          "history is tamper-evident")


if __name__ == "__main__":
    main()
