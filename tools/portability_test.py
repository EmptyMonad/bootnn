#!/usr/bin/env python3
"""
Portability gate: substrate coverage is a mintable, replay-verifiable
contribution (S2 hardware-profile reward class).

  1. GRAMMAR: unknown profile, empty or unsafe probe, missing fields -
     refused with deterministic reasons, no boot.
  2. HONEST COVERAGE: measure the canonical artifact's trajectory on
     qemu-tcg-x86 (boot 1, via the submit CLI's --measure), verify
     with a SECOND independent boot -> VERIFIED: metal == claim ==
     law, coverage mints, audit clean. Cross-boot digest equality is
     the determinism claim itself.
  3. DISHONESTY: a wrong claimed digest is exposed by the metal
     replay (boot 3) and rejected.
  4. UNIQUENESS: a duplicate (artifact, profile) claim is rejected
     without booting, and the fold mints a pair exactly once even if
     a duplicate verified verdict is forged directly into a log.
  5. FAITHFULNESS WIRING: verify_portability must fail law_ok when
     the law's prediction differs from metal (forced by monkeypatch -
     manufacturing a genuinely unfaithful profile means breaking the
     kernel, which is integrity_test's business).

Three QEMU boots total; requires dnos.img (tools/build.py).
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from zlib import crc32

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ledger  # noqa: E402
import portability  # noqa: E402

LEDGER = ROOT / "tools" / "ledger.py"
ARTIFACT = ROOT / "weights_ssm.bin"
PROFILE = "qemu-tcg-x86"
PROBE = "pblofc"


def cli(*argv, expect_fail=False):
    r = subprocess.run([sys.executable, str(LEDGER), *argv],
                       capture_output=True, text=True, timeout=600)
    if (r.returncode != 0) != expect_fail:
        print(r.stdout, r.stderr)
        raise AssertionError(
            f"cli {'succeeded' if expect_fail else 'failed'}: {argv}")
    return r.stdout


def grammar():
    acrc = crc32(ARTIFACT.read_bytes()) & 0xFFFFFFFF
    good = {"account": "gina", "op": "portability", "artifact_crc": acrc,
            "profile": PROFILE, "probe": PROBE, "claimed_digest": 0}
    assert portability.validate_portability(good) is None

    def refused(frag, **kw):
        d = dict(good, **kw)
        for k, v in list(kw.items()):
            if v is None:
                del d[k]
        r = portability.validate_portability(d)
        assert r and frag in r, f"expected {frag!r}, got {r!r}"

    refused("unknown profile", profile="riscv-not-yet")
    refused("non-empty", probe="")
    refused("qcode-safe", probe="p+b")
    refused("missing field", claimed_digest=None)
    print("[portability_test] grammar: malformed claims refused, no boot")


def honest_and_dishonest(tmp):
    led_path = tmp / "coverage.jsonl"
    # Boot 1: the contributor measures through the CLI.
    out = cli("--ledger", str(led_path), "submit-portability",
              "--account", "gina", "--artifact", str(ARTIFACT),
              "--profile", PROFILE, "--probe", PROBE, "--measure")
    assert "portability claim recorded" in out, out
    # Boot 2: the verifier replays independently.
    out = cli("--ledger", str(led_path), "verify", "--claim", "0",
              "--artifact", str(ARTIFACT))
    assert "VERIFIED (metal == claim == law" in out, out
    led = ledger.Ledger(led_path)
    balances, errors = led.fold()
    assert not errors, errors
    assert balances["gina"] == ledger.MINT_PER_PORTABILITY
    measured = led.entries[0]["data"]["claimed_digest"]
    print(f"[portability_test] honest: two independent boots agree "
          f"({measured:#010x}), coverage minted {balances['gina']}")

    # Dishonest: a digest metal cannot reproduce (boot 3).
    lie = dict(led.entries[0]["data"], account="hank",
               claimed_digest=(measured + 1) & 0xFFFFFFFF)
    # A different pair, or it would be rejected as duplicate first:
    # same artifact, same profile IS the pair - so the duplicate check
    # must be beaten by... nothing. Use a fresh ledger for the lie.
    led2_path = tmp / "lie.jsonl"
    ledger.Ledger(led2_path).append("claim", lie)
    out = cli("--ledger", str(led2_path), "verify", "--claim", "0",
              "--artifact", str(ARTIFACT))
    assert "REJECTED (dishonest" in out, out
    _, errors = ledger.Ledger(led2_path).fold()
    assert not errors
    print("[portability_test] dishonest: wrong digest exposed by metal "
          "replay, no mint")
    return led_path, measured


def uniqueness(tmp, led_path, measured):
    # A duplicate claim for an already-minted pair: rejected without
    # booting (the verdict text carries the reason).
    led = ledger.Ledger(led_path)
    dup = dict(led.entries[0]["data"], account="ivan")
    led.append("claim", dup)
    idx = len(led.entries) - 1
    out = cli("--ledger", str(led_path), "verify", "--claim", str(idx),
              "--artifact", str(ARTIFACT))
    assert "already minted" in out, out
    balances, errors = ledger.Ledger(led_path).fold()
    assert not errors and "ivan" not in balances

    # Fold-level: even a FORGED duplicate verified verdict mints the
    # pair once. (Verdicts here are appended directly, as
    # authorship_test does for its fake gauntlet.)
    led3_path = tmp / "forged.jsonl"
    led3 = ledger.Ledger(led3_path)
    base = dict(dup, account="mallory")
    led3.append("claim", base)
    led3.append("verdict", {"claim_idx": 0, "verified": True,
                            "replay_ok": True, "law_ok": True,
                            "computed_digest": measured,
                            "law_digest": measured})
    led3.append("claim", dict(base))
    led3.append("verdict", {"claim_idx": 2, "verified": True,
                            "replay_ok": True, "law_ok": True,
                            "computed_digest": measured,
                            "law_digest": measured})
    balances, errors = led3.fold()
    assert not errors, errors
    assert balances["mallory"] == ledger.MINT_PER_PORTABILITY, \
        f"pair minted more than once: {balances}"
    print("[portability_test] uniqueness: duplicate rejected without "
          "boot; forged double-verdict still mints the pair once")


def faithfulness_wiring():
    """No boot: both sides canned, the decision logic must reject a
    profile whose metal disagrees with the law."""
    claim = {"op": "portability", "profile": PROFILE, "probe": PROBE,
             "artifact_crc": 0, "claimed_digest": 0xAAAA}
    real_sim, real_metal = (portability.sim_trajectory,
                            portability.metal_trajectory)
    try:
        portability.sim_trajectory = lambda w, p: ([], 0xBBBB)
        portability.metal_trajectory = \
            lambda pr, w, p, port=0: ([], 0xAAAA)
        replay_ok, law_ok, _, _ = portability.verify_portability(
            claim, ARTIFACT)
    finally:
        portability.sim_trajectory = real_sim
        portability.metal_trajectory = real_metal
    assert replay_ok and not law_ok, \
        "an unfaithful profile was not flagged"
    print("[portability_test] faithfulness: honest-but-divergent metal "
          "fails law_ok - a substrate that isn't the law covers nothing")


def main():
    if not (ROOT / "dnos.img").is_file():
        sys.exit("[portability_test] dnos.img missing - run "
                 "tools/build.py first")
    tmp = Path(tempfile.mkdtemp())
    grammar()
    faithfulness_wiring()
    led_path, measured = honest_and_dishonest(tmp)
    uniqueness(tmp, led_path, measured)
    print("[portability_test] result: PASS - substrate coverage is "
          "verified by replay and minted once per (artifact, profile)")


if __name__ == "__main__":
    main()
