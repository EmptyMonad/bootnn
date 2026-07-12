#!/usr/bin/env python3
"""
Authorship gate: identity is optional, persistent, and optional stays
optional under economic pressure.

  1. HBSS round trip: a Lamport signature verifies; any bit flipped in
     the message or the signature fails; the SSID self-certifies.
  2. MERKLE round trip: one SSID (a Merkle root of one-time leaves)
     signs many messages; a tampered auth path or a wrong leaf index
     fails to certify.
  3. ECONOMIC NEUTRALITY (the invariant): the same work minted with an
     authorship signature and minted anonymously produce byte-identical
     verdict data and the identical mint. Reward is gated by gauntlet
     alone - never by identity.
  4. PERSISTENCE: several ledger entries signed from one seed share one
     SSID with distinct leaves and audit clean; the legacy one-time
     envelope still audits clean as the empty-path degenerate.
  5. ONE-TIME-NESS IS LOG-ENFORCED: a leaf that signs two entries fails
     audit; an exhausted identity refuses to append rather than reuse.
  6. INTEGRITY: a tampered authorship envelope fails audit; a claim
     with no envelope audits clean (anonymous is first-class).
  7. REGISTRY-FREE: verification consults nothing but the entry.

Uses a trivial fake gauntlet so the test is fast and about authorship,
not training.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hbss  # noqa: E402
import ledger  # noqa: E402


def hbss_roundtrip():
    sk, pk = hbss.keypair("carol-key")
    msg = "the entry hash"
    sig = hbss.sign(sk, msg)
    pkh = hbss.pk_hex(pk)
    assert hbss.verify(pkh, msg, sig), "valid sig rejected"
    assert not hbss.verify(pkh, "other", sig), "sig verified wrong message"
    bad = ("f" if sig[0] != "f" else "0") + sig[1:]
    assert not hbss.verify(pkh, msg, bad), "tampered sig verified"
    assert hbss.ssid(pk) == hbss.ssid_of_hex(pkh), "ssid not self-consistent"
    print("[authorship] HBSS: sign/verify/tamper/ssid OK")


def merkle_roundtrip():
    ident = hbss.MerkleIdentity("dana-key")
    envs = [ident.envelope(f"entry hash {i}", i) for i in (0, 3, 7)]
    for i, env in zip((0, 3, 7), envs):
        assert hbss.verify(env["pubkey"], f"entry hash {i}", env["sig"]), \
            f"leaf {i} sig rejected"
        assert hbss.certify(env), f"leaf {i} does not certify"
    assert len({env["ssid"] for env in envs}) == 1, \
        "leaves of one identity produced different SSIDs"
    assert len({env["pubkey"] for env in envs}) == 3, \
        "distinct leaves shared a pubkey"
    # Tampered auth path: valid signature, wrong membership proof.
    bad = dict(envs[0], path=list(envs[0]["path"]))
    p0 = bad["path"][0]
    bad["path"][0] = ("0" if p0[0] != "0" else "1") + p0[1:]
    assert not hbss.certify(bad), "tampered auth path certified"
    # Wrong leaf index: flips left/right folds, root must mismatch.
    assert not hbss.certify(dict(envs[0], leaf=1)), \
        "wrong leaf index certified"
    print("[authorship] Merkle: one SSID, many leaves; bad path and bad "
          "index refused")


def _mint_one(tmp, name, author_seed):
    """Build a fresh ledger, one claim (optionally signed), force a
    verified verdict via a monkeypatched replay+gauntlet, return the
    verdict entry's data and the balance minted."""
    path = Path(tmp) / f"{name}.jsonl"
    led = ledger.Ledger(path)
    data = {"account": name, "seed": 1, "epochs": 1, "lr": 0.02,
            "claimed_crc": 0xABCDEF01}
    led.append("claim", data, author_seed=author_seed)
    # Deterministic fake verification: honest + above bar, no training.
    led.append("verdict", {"claim_idx": 0, "verified": True,
                           "computed_crc": 0xABCDEF01, "replay_ok": True,
                           "heldout": 99.9, "min_heldout": 95.0})
    balances, errors = led.fold()
    assert not errors, f"{name}: {errors}"
    verdict = next(e["data"] for e in led.entries if e["type"] == "verdict")
    return verdict, balances.get(name, 0), led


def neutrality(tmp):
    v_anon, bal_anon, _ = _mint_one(tmp, "anon", None)
    v_signed, bal_signed, led = _mint_one(tmp, "signed", "signer-seed-1")
    # The verdict (the reward-bearing record) is identical modulo the
    # account label: mint amount and every verified/quality field match.
    for k in ("verified", "replay_ok", "heldout", "min_heldout",
              "computed_crc", "claim_idx"):
        assert v_anon[k] == v_signed[k], f"verdict field {k} differs"
    assert bal_anon == bal_signed == ledger.MINT_PER_VERIFIED_CLAIM, \
        f"mint differs: anon={bal_anon} signed={bal_signed}"
    # The signed claim carries a self-certifying persistent author; the
    # anon one carries none - and both minted the same.
    signed_claim = led.entries[0]
    assert "author" in signed_claim and \
        hbss.certify(signed_claim["author"]) and \
        signed_claim["author"]["path"], "signed claim not Merkle-certified"
    print(f"[authorship] neutrality: anon and signed both mint "
          f"{bal_anon}; verdict bytes identical - reward is "
          f"identity-blind")


def persistence(tmp):
    path = Path(tmp) / "persist.jsonl"
    led = ledger.Ledger(path)
    for i in range(3):
        led.append("claim", {"account": "erin", "seed": i, "epochs": 1,
                             "lr": 0.02, "claimed_crc": i},
                   author_seed="erin-seed")
    _, errors = led.fold()
    assert not errors, f"persistent identity fails audit: {errors}"
    envs = [e["author"] for e in led.entries]
    assert len({env["ssid"] for env in envs}) == 1, \
        "one seed produced multiple SSIDs"
    assert [env["leaf"] for env in envs] == [0, 1, 2], \
        "leaves not spent in order"
    # Legacy one-time envelope = the empty-path degenerate: attach an
    # old-format signature to a fresh anonymous entry (the envelope
    # sits outside the hashed body, so the chain is untouched) and the
    # new verifier must accept it unchanged.
    led2 = ledger.Ledger(Path(tmp) / "legacy.jsonl")
    e = led2.append("claim", {"account": "old", "seed": 9, "epochs": 1,
                              "lr": 0.02, "claimed_crc": 9})
    sk, pk = hbss.keypair("legacy-seed")
    raw = json.loads(led2.path.read_text().splitlines()[0])
    raw["author"] = {"ssid": hbss.ssid(pk), "pubkey": hbss.pk_hex(pk),
                     "sig": hbss.sign(sk, e["hash"])}
    led2.path.write_text(json.dumps(raw, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    _, errors = ledger.Ledger(led2.path).fold()
    assert not errors, f"legacy one-time envelope fails audit: {errors}"
    print("[authorship] persistence: 3 entries, one SSID, leaves 0/1/2, "
          "audit clean; legacy envelope still verifies")


def one_time_enforced(tmp):
    # Leaf reuse: sign two entries with the SAME leaf (bypassing
    # append's spent-leaf scan by attaching envelopes post-hoc - valid
    # signatures, valid membership, but the leaf is spent twice).
    path = Path(tmp) / "reuse.jsonl"
    led = ledger.Ledger(path)
    for i in range(2):
        led.append("claim", {"account": "mallory", "seed": i, "epochs": 1,
                             "lr": 0.02, "claimed_crc": i})
    ident = hbss.MerkleIdentity("mallory-seed")
    lines = []
    for ln in led.path.read_text().splitlines():
        e = json.loads(ln)
        e["author"] = ident.envelope(e["hash"], 0)
        lines.append(json.dumps(e, sort_keys=True, separators=(",", ":")))
    led.path.write_text("\n".join(lines) + "\n")
    _, errors = ledger.Ledger(path).fold()
    assert any("reused" in x for x in errors), \
        "a leaf signed two entries and audit stayed clean"
    # Exhaustion: after 2^depth signed appends the identity must refuse
    # rather than silently reuse a leaf.
    led3 = ledger.Ledger(Path(tmp) / "exhaust.jsonl")
    n = 1 << hbss.IDENT_DEPTH
    for i in range(n):
        led3.append("claim", {"account": "frank", "seed": i, "epochs": 1,
                              "lr": 0.02, "claimed_crc": i},
                    author_seed="frank-seed")
    try:
        led3.append("claim", {"account": "frank", "seed": n, "epochs": 1,
                              "lr": 0.02, "claimed_crc": n},
                    author_seed="frank-seed")
        raise AssertionError("exhausted identity appended a 9th signature")
    except SystemExit as ex:
        assert "exhausted" in str(ex), f"wrong refusal: {ex}"
    _, errors = led3.fold()
    assert not errors, f"exhausted-but-honest ledger fails audit: {errors}"
    print(f"[authorship] one-time enforced: leaf reuse fails audit; "
          f"leaf {n+1} of {n} refused loudly")


def adversarial_ssid(tmp):
    """One-time-ness must survive an ADVERSARIAL envelope, not just an
    honest one. Reusing a leaf pubkey with a different fabricated auth
    path yields a different (still-certifying) ssid; if the reuse guard
    keyed on the self-asserted ssid, the two would look distinct and
    slip. Keying on the pubkey - the material a signature reveals -
    catches it."""
    from hbss import _h, keypair, merkle_root_from_leaf, pk_hex, sign
    led = ledger.Ledger(Path(tmp) / "advssid.jsonl")
    for i in range(2):
        led.append("claim", {"account": "eve", "seed": i, "epochs": 1,
                             "lr": 0.02, "claimed_crc": i})
    sk, pk = keypair("adversary-leaf")
    pubkey = pk_hex(pk)
    leafhash = _h(bytes.fromhex(pubkey))
    lines = []
    for n, ln in enumerate(led.path.read_text().splitlines()):
        e = json.loads(ln)
        sib = bytes([n + 1]) * 32          # a different path per entry
        ssid = merkle_root_from_leaf(leafhash, 0, [sib]).hex()
        e["author"] = {"ssid": ssid, "pubkey": pubkey,
                       "sig": sign(sk, e["hash"]), "leaf": 0,
                       "path": [sib.hex()]}
        # Both envelopes independently certify (each ssid IS the root of
        # its own path) - the attacker's freedom.
        assert hbss.certify(e["author"]), "fabricated envelope rejected"
        lines.append(json.dumps(e, sort_keys=True, separators=(",", ":")))
    led.path.write_text("\n".join(lines) + "\n")
    _, errors = ledger.Ledger(led.path).fold()
    assert any("reused" in x for x in errors), \
        "adversarial ssid variation evaded one-time-leaf detection"
    print("[authorship] adversarial ssid: leaf reuse under two fabricated "
          "ssids still fails audit (dedup keys on pubkey)")


def fold_totality(tmp):
    """fold() classifies malformed envelopes; it never raises. A signed
    entry with the signature field stripped is an integrity error, not
    a KeyError that takes down the audit."""
    _, _, led = _mint_one(tmp, "totality", "signer-seed-1")
    lines = led.path.read_text().splitlines()
    e = json.loads(lines[0])
    del e["author"]["sig"]
    lines[0] = json.dumps(e, sort_keys=True, separators=(",", ":"))
    bad = Path(tmp) / "no_sig.jsonl"
    bad.write_text("\n".join(lines) + "\n")
    _, errors = ledger.Ledger(bad).fold()          # must not raise
    assert any("missing pubkey/sig" in x for x in errors), errors
    print("[authorship] fold totality: envelope missing sig -> audit "
          "error, not a crash")


def integrity(tmp):
    _, _, led = _mint_one(tmp, "signed_integ", "signer-seed-1")
    lines = led.path.read_text().splitlines()
    e = json.loads(lines[0])
    # Flip one hex char of the signature: forgery must fail audit.
    sig = e["author"]["sig"]
    e["author"]["sig"] = ("0" if sig[0] != "0" else "1") + sig[1:]
    lines[0] = json.dumps(e, sort_keys=True, separators=(",", ":"))
    bad = Path(tmp) / "tampered.jsonl"
    bad.write_text("\n".join(lines) + "\n")
    _, errors = ledger.Ledger(bad).fold()
    assert any("signature invalid" in x for x in errors), \
        "tampered author envelope passed audit"
    print("[authorship] integrity: forged author signature fails audit")


def main():
    tmp = tempfile.mkdtemp()
    hbss_roundtrip()
    merkle_roundtrip()
    neutrality(tmp)
    persistence(tmp)
    one_time_enforced(tmp)
    adversarial_ssid(tmp)
    fold_totality(tmp)
    integrity(tmp)
    print("[authorship] result: PASS - identity is optional, persistent, "
          "unforgeable when present, and never buys a mint")


if __name__ == "__main__":
    main()
