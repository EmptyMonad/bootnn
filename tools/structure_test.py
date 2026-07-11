#!/usr/bin/env python3
"""
Structural-claim gate: architecture is a fold over the log.

A create_leaf claim is the one contribution an inline delta cannot
express (the S3 law-shape verdict). This gate proves:

  1. GRAMMAR: malformed structural claims are refused with a
     deterministic reason - wrong op, wrong leaf index, empty or
     duplicate or untrainable scope, missing fields, tampered inline
     base manifest, broken chain.
  2. REPLAY: an honest claim reproduces BOTH commitments (leaf blob
     CRC and derived composite manifest CRC) and mints; a dishonest
     commitment is exposed by structural replay and rejected.
  3. CHAIN: a structure claim must extend the last VERIFIED structure;
     rejected claims never advance the chain - a rejected or
     wrongly-based claim cannot mutate the architecture.
  4. DERIVED COMPOSITE: the verdict's manifest CRC equals an
     independent re-derivation from parts - the composite is never
     asserted, only derived.
  5. The mint is the same identity-blind mint as every other claim.

Training here is tiny (seconds): the gate is about structure, not
quality - the quality bar is exercised at min-heldout 0, as
ledger_test does for training claims.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ledger  # noqa: E402
import s3_forest  # noqa: E402
from ssm_lab import SsmLab  # noqa: E402

LEDGER = ROOT / "tools" / "ledger.py"
EPOCHS = 20


def cli(*argv, expect_fail=False):
    r = subprocess.run([sys.executable, str(LEDGER), *argv],
                       capture_output=True, text=True, timeout=1200)
    if (r.returncode != 0) != expect_fail:
        print(r.stdout, r.stderr)
        raise AssertionError(
            f"cli {'succeeded' if expect_fail else 'failed'}: {argv}")
    return r.stdout


def build_base(tmp):
    """Two random-init leaves under the base table: the structure the
    claims will extend. Built fresh in-test - nothing pinned."""
    blobs = []
    for i in (0, 1):
        np.random.seed(7000 + i)
        lab = SsmLab(h=512, seed=7000 + i)
        p = tmp / f"base_leaf{i}.bin"
        lab.save(str(p))
        blobs.append(p)
    m, mcrc = s3_forest.manifest_for(blobs)
    mf = tmp / "base_manifest.json"
    mf.write_text(json.dumps(m, sort_keys=True, separators=(",", ":")))
    return m, mcrc, mf


def grammar(m):
    good = {"account": "gina", "op": "create_leaf", "leaf": 2,
            "scope": ["f", "u"], "base_manifest": m,
            "base_manifest_crc": s3_forest.manifest_crc(m),
            "seed": 9, "epochs": 1, "lr": 0.02,
            "claimed_crc": 0, "claimed_manifest_crc": 0}
    assert s3_forest.validate_structure(good) is None, \
        s3_forest.validate_structure(good)

    def refused(frag, **kw):
        r = s3_forest.validate_structure(dict(good, **kw))
        assert r and frag in r, f"expected {frag!r}, got {r!r}"

    refused("unknown structural op", op="delete_leaf")
    refused("next index", leaf=5)
    refused("non-empty", scope=[])
    refused("duplicate", scope=["f", "f"])
    refused("single characters", scope=["fu"])
    refused("trainable trigger", scope=["z"])
    tampered = dict(m, router_version=99)
    refused("does not hash", base_manifest=tampered)
    incomplete = dict(good)
    del incomplete["claimed_crc"]
    r = s3_forest.validate_structure(incomplete)
    assert r and "missing field" in r, r
    r = s3_forest.validate_structure(good, prior_manifest_crc=123)
    assert r and "chain" in r, r
    print("[structure] grammar: 9 malformed claims refused, each with "
          "its reason")


def economy(tmp, m, mcrc, mf):
    """Honest end-to-end through the CLI: contributor runs the same
    deterministic pipeline the verifier will, submits, verifier
    replays, structure mints."""
    claim = {"account": "gina", "op": "create_leaf", "leaf": 2,
             "scope": ["f", "u"], "base_manifest": m,
             "base_manifest_crc": mcrc, "seed": 4242, "epochs": EPOCHS,
             "lr": 0.02, "claimed_crc": 0, "claimed_manifest_crc": 0}
    blob = tmp / "gina_leaf2.bin"
    leaf_crc, man_crc, table = s3_forest.replay_structure(claim, blob)

    led_path = tmp / "econ.jsonl"
    cli("--ledger", str(led_path), "submit-structure",
        "--account", "gina", "--leaf", "2", "--scope", "fu",
        "--base-manifest", str(mf), "--seed", "4242",
        "--epochs", str(EPOCHS), "--lr", "0.02",
        "--claimed-crc", hex(leaf_crc),
        "--claimed-manifest-crc", hex(man_crc))
    out = cli("--ledger", str(led_path), "verify", "--claim", "0",
              "--min-heldout", "0")
    assert "VERIFIED" in out, out

    led = ledger.Ledger(led_path)
    vd = next(e["data"] for e in led.entries if e["type"] == "verdict")
    assert vd["computed_manifest_crc"] == man_crc
    # The composite is DERIVED: re-derive it independently from parts.
    _, man_crc2 = s3_forest.manifest_of(
        table, [x["crc"] for x in m["leaves"]] + [leaf_crc])
    assert man_crc2 == man_crc, "manifest commitment not re-derivable"
    balances, errors = led.fold()
    assert not errors, errors
    assert balances["gina"] == ledger.MINT_PER_VERIFIED_CLAIM
    print(f"[structure] economy: create_leaf replayed bit-exact (leaf "
          f"{leaf_crc:#010x}, manifest {man_crc:#010x}), minted "
          f"{balances['gina']}, audit clean")
    return led_path, man_crc


def dishonest_and_chain(tmp, led_path, m, mcrc, verified_man_crc):
    led = ledger.Ledger(led_path)
    # Dishonest: commitments that replay cannot reproduce. The claim
    # chains correctly (base = gina's verified manifest) so only the
    # replay itself can expose the lie.
    gina = led.entries[0]["data"]
    new_table = s3_forest.derive_table(
        s3_forest.table_of_manifest(m), gina["scope"])
    new_manifest, _ = s3_forest.manifest_of(
        new_table, [x["crc"] for x in m["leaves"]]
        + [gina["claimed_crc"]])
    lie = {"account": "hank", "op": "create_leaf", "leaf": 3,
           "scope": ["+"], "base_manifest": new_manifest,
           "base_manifest_crc": verified_man_crc,
           "seed": 5, "epochs": 8, "lr": 0.02,
           "claimed_crc": 0xDEAD, "claimed_manifest_crc": 0xBEEF}
    assert s3_forest.validate_structure(lie, verified_man_crc) is None
    led.append("claim", lie)
    idx = len(led.entries) - 1
    out = cli("--ledger", str(led_path), "verify", "--claim", str(idx),
              "--min-heldout", "0")
    assert "REJECTED (dishonest" in out, out
    # The rejected claim must NOT advance the structure chain.
    led = ledger.Ledger(led_path)
    assert ledger.prior_structure_crc(led.entries) == verified_man_crc, \
        "a rejected structural claim mutated the architecture"
    # Wrong base: a claim extending the ORIGINAL manifest instead of
    # the verified chain head is refused on grammar, before training.
    stale = {"account": "ivan", "op": "create_leaf", "leaf": 2,
             "scope": ["+"], "base_manifest": m,
             "base_manifest_crc": mcrc, "seed": 5, "epochs": 8,
             "lr": 0.02, "claimed_crc": 1, "claimed_manifest_crc": 2}
    led.append("claim", stale)
    idx = len(led.entries) - 1
    out = cli("--ledger", str(led_path), "verify", "--claim", str(idx),
              "--min-heldout", "0")
    assert "REJECTED (grammar" in out and "chain" in out, out
    balances, errors = ledger.Ledger(led_path).fold()
    assert not errors, errors
    assert "hank" not in balances and "ivan" not in balances
    print("[structure] negatives: dishonest commitment exposed by "
          "replay; stale base refused on chain; neither minted, "
          "neither mutated structure")


def main():
    tmp = Path(tempfile.mkdtemp())
    m, mcrc, mf = build_base(tmp)
    grammar(m)
    led_path, man_crc = economy(tmp, m, mcrc, mf)
    dishonest_and_chain(tmp, led_path, m, mcrc, man_crc)
    print("[structure] result: PASS - architecture, like weights, is a "
          "pure function of the log")


if __name__ == "__main__":
    main()
