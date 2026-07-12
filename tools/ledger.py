#!/usr/bin/env python3
"""
S2 prototype: the contribution ledger (docs/SWARM_DESIGN.md, S2).

An append-only, hash-chained event log where:

  - a *claim* is a training-contribution tuple
    (account, seed, epochs, lr, claimed_crc) — "I ran this training
    and got this blob" — or a STRUCTURAL claim (op=create_leaf,
    grammar in s3_forest.validate_structure): "I carved this scope
    into a new leaf and got this blob AND this composite commitment" —
    or a PORTABILITY claim (op=portability, grammar in
    portability.validate_portability): "this artifact computes the law
    on this hardware profile", verified by booting the named profile
    and minted once per (artifact, profile) pair.
    Weights, architecture, and substrate coverage are all pure
    functions of the log;
  - a *verdict* is produced by replaying the claim: rerun train.py
    with the tuple, CRC32 the output, compare. Verification IS
    deterministic replay — no committee, no proof system, binary;
  - *issuance is a pure function of the log*: every claim whose first
    verdict verified mints MINT_PER_VERIFIED_CLAIM to its account
    during the fold. There is no mint event to forge — you cannot
    write a balance, only a history that folds to one;
  - *transfers* are validated at append time and re-validated by
    audit (no balance may ever fold negative);
  - *audit* re-derives everything from genesis: chain hashes, verdict
    uniqueness, balances, total supply. Tampering with any byte of
    history breaks the chain.

No blockchain, no kernel changes, no cryptographic signatures yet
(those live at this boundary later, per SWARM_DESIGN — ML-DSA per
event, SPHINCS+ for identities; hashes only in the hot path).

Usage:
  python tools/ledger.py --ledger L.jsonl submit --account alice \
      --seed 4242 --epochs 30 --lr 0.02 --claimed-crc 0x1becb214
  python tools/ledger.py --ledger L.jsonl verify --claim 0
  python tools/ledger.py --ledger L.jsonl transfer --src alice --dst bob --amount 40
  python tools/ledger.py --ledger L.jsonl balances
  python tools/ledger.py --ledger L.jsonl audit
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from zlib import crc32

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "tools" / "train.py"

MINT_PER_VERIFIED_CLAIM = 100
# The hardware-profile reward class: verified coverage of one
# (artifact, profile) pair mints once - substrate coverage is a
# contribution, but the same coverage twice is worth nothing.
MINT_PER_PORTABILITY = 100
GENESIS = "dnos-ledger-v0"


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def entry_hash(idx, prev, etype, data):
    body = canonical({"idx": idx, "prev": prev, "type": etype, "data": data})
    return hashlib.blake2b(body.encode(), digest_size=32).hexdigest()


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.entries = []
        if self.path.is_file():
            self.entries = [json.loads(ln) for ln in
                            self.path.read_text().splitlines() if ln.strip()]

    # ── the fold: state is derived, never stored ────────────────────────
    def fold(self):
        """Replay the log from genesis. Returns (balances, errors).

        Authorship anchors to THIS log: precedence is append order, and
        integrity is the hash chain below - identity-free by default. An
        optional `author` envelope (hash-based signature under a
        persistent Merkle-Lamport identity; the legacy one-time envelope
        is its empty-path degenerate) rides alongside; when present it
        is verified for integrity, but it NEVER gates reward. The mint reads only
        `verdict.verified` (= replay_ok AND gauntlet) - so an anonymous
        claim and a signed one mint identically, and optional stays
        optional under economic pressure."""
        balances = {}
        errors = []
        claims = {}
        verdicted = set()
        leaves_spent = set()
        covered = set()
        prev = GENESIS
        for e in self.entries:
            if e["prev"] != prev or \
               e["hash"] != entry_hash(e["idx"], e["prev"], e["type"], e["data"]):
                errors.append(f"entry {e['idx']}: hash chain broken")
                prev = e["hash"]
                continue
            prev = e["hash"]
            # Optional authorship proof: registry-free, provable without
            # lookup. Absent -> anonymous (no error). Present -> must
            # verify against the entry hash and its own embedded pubkey,
            # and its ssid must certify (Merkle leaf->root fold; the
            # legacy one-time envelope is the empty-path degenerate).
            # A forged/tampered envelope is an integrity error, not a
            # reward event. One-time-ness is enforced by the log, not
            # by signer discipline: a leaf that signs two entries has
            # revealed too many preimages, so reuse fails audit.
            author = e.get("author")
            if author is not None:
                from hbss import certify, verify
                pubkey, sig = author.get("pubkey"), author.get("sig")
                if not isinstance(pubkey, str) or not isinstance(sig, str):
                    # fold is TOTAL over adversarial logs: a malformed
                    # envelope is an integrity error, never an exception.
                    errors.append(f"entry {e['idx']}: author envelope "
                                  f"missing pubkey/sig")
                elif not verify(pubkey, e["hash"], sig):
                    errors.append(f"entry {e['idx']}: author signature invalid")
                elif not certify(author):
                    errors.append(f"entry {e['idx']}: ssid does not certify")
                else:
                    # One-time-ness keys on the LEAF PUBKEY - the secret
                    # material a signature reveals - NOT the self-asserted
                    # ssid. An adversary can vary the ssid by fabricating
                    # an auth path that still certifies, but reusing a
                    # leaf means reusing its pubkey; two signatures under
                    # one pubkey reveal preimages twice, so reuse fails
                    # audit regardless of the ssid claimed.
                    if pubkey in leaves_spent:
                        errors.append(f"entry {e['idx']}: one-time leaf "
                                      f"reused (pubkey {pubkey[:12]}...)")
                    leaves_spent.add(pubkey)
            d = e["data"]
            if e["type"] == "claim":
                claims[e["idx"]] = d
            elif e["type"] == "verdict":
                ci = d["claim_idx"]
                if ci not in claims:
                    errors.append(f"entry {e['idx']}: verdict for unknown claim {ci}")
                elif ci in verdicted:
                    errors.append(f"entry {e['idx']}: double verdict for claim {ci}")
                else:
                    verdicted.add(ci)
                    if d["verified"]:
                        c = claims[ci]
                        acct = c.get("account")
                        if acct is None:
                            errors.append(f"entry {e['idx']}: verified "
                                          f"claim {ci} has no account")
                        elif c.get("op") == "portability":
                            # Coverage mints once per (artifact, profile)
                            # - even a forged duplicate verdict buys
                            # nothing at the fold. Missing pair fields on
                            # a forged claim are an audit error, not a
                            # crash (fold stays total).
                            if "artifact_crc" not in c or "profile" not in c:
                                errors.append(f"entry {e['idx']}: "
                                              f"portability claim {ci} "
                                              f"missing artifact_crc/profile")
                            else:
                                pair = (c["artifact_crc"], c["profile"])
                                if pair not in covered:
                                    covered.add(pair)
                                    balances[acct] = balances.get(acct, 0) \
                                        + MINT_PER_PORTABILITY
                        else:
                            balances[acct] = balances.get(acct, 0) \
                                + MINT_PER_VERIFIED_CLAIM
            elif e["type"] == "transfer":
                src, dst, amt = d["src"], d["dst"], d["amount"]
                if amt <= 0 or balances.get(src, 0) < amt:
                    errors.append(f"entry {e['idx']}: invalid transfer "
                                  f"{src}->{dst} {amt}")
                else:
                    balances[src] -= amt
                    balances[dst] = balances.get(dst, 0) + amt
        return balances, errors

    def append(self, etype, data, author_seed=None):
        # A transfer must be valid against the folded state *now*;
        # a verdict must be the first for its claim.
        balances, errors = self.fold()
        if errors:
            sys.exit(f"ERROR: refusing to append to a ledger that fails "
                     f"audit: {errors[0]}")
        if etype == "transfer":
            if data["amount"] <= 0 or \
               balances.get(data["src"], 0) < data["amount"]:
                sys.exit(f"ERROR: transfer rejected - {data['src']} has "
                         f"{balances.get(data['src'], 0)}, "
                         f"wants to send {data['amount']}")
        if etype == "verdict":
            for e in self.entries:
                if e["type"] == "verdict" and \
                   e["data"]["claim_idx"] == data["claim_idx"]:
                    sys.exit(f"ERROR: claim {data['claim_idx']} already "
                             f"has a verdict (entry {e['idx']})")
        idx = len(self.entries)
        prev = self.entries[-1]["hash"] if self.entries else GENESIS
        h = entry_hash(idx, prev, etype, data)
        e = {"idx": idx, "prev": prev, "type": etype, "data": data,
             "hash": h}
        # Optional authorship proof: signs the entry HASH (which binds
        # idx+prev+type+data), so the signature covers position and
        # content but sits OUTSIDE the hashed body - it cannot perturb
        # the chain, and anonymous vs signed produce the same chain.
        # The seed derives a persistent Merkle identity (one SSID,
        # 2^depth one-time leaves); the signer is stateless - which
        # leaves are spent is read back from this log, and exhaustion
        # refuses loudly rather than ever reusing a leaf.
        if author_seed is not None:
            from hbss import MerkleIdentity
            ident = MerkleIdentity(author_seed)
            spent = {a.get("leaf", 0) for x in self.entries
                     for a in (x.get("author"),)
                     if a and a["ssid"] == ident.ssid}
            free = next((i for i in range(ident.n_leaves)
                         if i not in spent), None)
            if free is None:
                sys.exit(f"ERROR: identity {ident.ssid[:12]}... exhausted "
                         f"- all {ident.n_leaves} one-time leaves spent on "
                         f"this ledger; derive a new identity")
            e["author"] = ident.envelope(h, free)
        self.entries.append(e)
        with self.path.open("a") as f:
            f.write(canonical(e) + "\n")
        return e


# The economy's default quality bar: an honest replay mints only if the
# replayed artifact ALSO clears this held-out generalization threshold.
# CRC-match proves honesty (they ran what they said); the gauntlet
# proves worth. Reproducibility is not worthiness.
MINT_MIN_HELDOUT = 95.0


def replay_claim(claim, out):
    """Verification IS replay: rerun the training, CRC the artifact.
    A dataset-delta claim carries its examples inline, so the delta is
    part of the log and the replay is fully determined by it."""
    cmd = [sys.executable, str(TRAIN),
           "--seed", str(claim["seed"]),
           "--epochs", str(claim["epochs"]),
           "--lr", str(claim["lr"]),
           "--output", str(out)]
    delta = claim.get("data_delta") or []
    delta_ctx = tempfile.TemporaryDirectory() if delta else None
    if delta:
        dpath = Path(delta_ctx.name) / "delta.jsonl"
        dpath.write_text("".join(canonical(r) + "\n" for r in delta))
        cmd += ["--data-delta", str(dpath)]
    print(f"[ledger] replaying: {' '.join(cmd[1:])}")
    # train.py's exit code also gates accuracy/divergence; the ledger
    # asks separately (a) is the artifact reproduced, (b) does it pass
    # the quality gauntlet.
    try:
        subprocess.run(cmd, capture_output=True, timeout=3600)
    finally:
        if delta_ctx:
            delta_ctx.cleanup()
    if not out.is_file():
        return None
    return crc32(out.read_bytes()) & 0xFFFFFFFF


def prior_structure_crc(entries):
    """The manifest commitment of the LAST VERIFIED structural claim
    in the log, or None. Structure claims must chain to it: rejected
    claims never mutate the architecture."""
    prior = None
    claims = {e["idx"]: e["data"] for e in entries if e["type"] == "claim"}
    for e in entries:
        if e["type"] == "verdict" and e["data"].get("verified"):
            c = claims.get(e["data"]["claim_idx"], {})
            if c.get("op") == "create_leaf":
                prior = c["claimed_manifest_crc"]
    return prior


def verify_structure(claim, entries):
    """Structural verification IS structural replay: the grammar and
    chain are checked, the new leaf is retrained deterministically on
    its scoped view, and BOTH commitments - the leaf blob CRC and the
    derived composite manifest CRC - must reproduce. Worth is the new
    leaf's scoped held-out (the software impl of its oracle port).
    Returns (replay_ok, leaf_crc, manifest_crc, heldout, reason)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import s3_forest
    reason = s3_forest.validate_structure(claim, prior_structure_crc(entries))
    if reason is not None:
        return False, None, None, None, reason
    with tempfile.TemporaryDirectory() as td:
        blob = Path(td) / "leaf.bin"
        print(f"[ledger] structural replay: create_leaf {claim['leaf']} "
              f"scope {''.join(claim['scope'])!r} (seed {claim['seed']}, "
              f"{claim['epochs']} epochs)")
        leaf_crc, mcrc, table = s3_forest.replay_structure(claim, blob)
        replay_ok = (leaf_crc == claim["claimed_crc"]
                     and mcrc == claim["claimed_manifest_crc"])
        heldout = None
        if replay_ok:
            heldout = round(s3_forest.leaf_gauntlet(blob, claim["leaf"],
                                                    table), 1)
    return replay_ok, leaf_crc, mcrc, heldout, None


def gauntlet(blob, per_key=20):
    """Held-out generalization of an artifact, percent. Deterministic
    (context_eval's fixed seed), format-aware (window or v5)."""
    sys.path.insert(0, str(ROOT / "tools"))
    from context_eval import evaluate
    correct, total, _, _, _ = evaluate(str(blob), per_key=per_key)
    return 100.0 * correct / total


class WorthOracle:
    """The worth-measurement PORT (NDAL seam). Everything the economy
    knows about an artifact's value enters through here, so the
    measurement's *provenance* is swappable without touching the
    ledger: today a deterministic software gauntlet; later a TPM-,
    PUF-, or throughput-attested oracle. One physical root per
    substrate; logical ports are derived per specialist, so a
    branching law measures each leaf through its own named port."""

    def port(self, specialist_id):
        return _OraclePort(self, specialist_id)

    def measure(self, blob, specialist_id="root"):
        raise NotImplementedError


class _OraclePort:
    def __init__(self, root, specialist_id):
        self.root, self.specialist_id = root, specialist_id

    def measure(self, blob):
        return self.root.measure(blob, self.specialist_id)


class SoftwareOracle(WorthOracle):
    """Null/software implementation: worth == the deterministic
    held-out gauntlet. No attestation - the root of trust is replay."""

    def measure(self, blob, specialist_id="root"):
        return gauntlet(blob)


WORTH_ORACLE = SoftwareOracle()


def main():
    ap = argparse.ArgumentParser(description="DNOS contribution ledger (S2)")
    ap.add_argument("--ledger", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("submit", help="record a training-contribution claim")
    p.add_argument("--account", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--claimed-crc", required=True,
                   help="CRC32 of the claimed weight blob (hex ok)")
    p.add_argument("--data-delta",
                   help="JSONL of contributed examples "
                        "({\"keys\":[..],\"cmd\":name}); stored inline in "
                        "the claim so replay is fully determined by the log")
    p.add_argument("--author-seed",
                   help="optional self-sovereign authorship: sign the "
                        "claim with a persistent Merkle-Lamport identity "
                        "derived from this seed - one SSID, many entries, "
                        "each leaf one-time (registry-free; never gates "
                        "reward)")

    p = sub.add_parser("submit-structure",
                       help="record a create_leaf structural claim - the "
                            "contribution a delta cannot express")
    p.add_argument("--account", required=True)
    p.add_argument("--leaf", type=int, required=True,
                   help="new leaf id (must be the next index)")
    p.add_argument("--scope", required=True,
                   help="keys carved for the new leaf, e.g. 'uf'")
    p.add_argument("--base-manifest", required=True,
                   help="JSON file of the composite manifest this claim "
                        "extends (carried inline, CRC-bound)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--claimed-crc", required=True,
                   help="CRC32 of the new leaf's v5 blob (hex ok)")
    p.add_argument("--claimed-manifest-crc", required=True,
                   help="CRC32 of the new composite manifest (hex ok)")
    p.add_argument("--author-seed",
                   help="optional persistent Merkle-Lamport authorship "
                        "(never gates reward)")

    p = sub.add_parser("submit-portability",
                       help="record a hardware-profile coverage claim: "
                            "this artifact computes the law on this "
                            "profile")
    p.add_argument("--account", required=True)
    p.add_argument("--artifact", required=True,
                   help="path to the v5 blob the claim covers")
    p.add_argument("--profile", required=True,
                   help="profile id (see portability.PROFILES)")
    p.add_argument("--probe", default="pblofc",
                   help="probe event log, qcode-safe [a-z0-9]")
    p.add_argument("--claimed-digest",
                   help="trajectory digest measured on the profile "
                        "(hex ok); omit with --measure")
    p.add_argument("--measure", action="store_true",
                   help="boot the profile now and measure the digest "
                        "instead of passing --claimed-digest")
    p.add_argument("--author-seed",
                   help="optional persistent Merkle-Lamport authorship "
                        "(never gates reward)")

    p = sub.add_parser("verify", help="replay a claim and record the verdict")
    p.add_argument("--claim", type=int, required=True)
    p.add_argument("--min-heldout", type=float, default=MINT_MIN_HELDOUT,
                   help="quality bar: honest replays mint only at or "
                        "above this held-out accuracy (policy knob, "
                        "recorded in the verdict)")
    p.add_argument("--artifact", default=str(ROOT / "weights_ssm.bin"),
                   help="artifact file for portability claims (its CRC "
                        "must match the claim's artifact_crc)")

    p = sub.add_parser("transfer")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--amount", type=int, required=True)

    sub.add_parser("balances")
    sub.add_parser("audit")

    args = ap.parse_args()
    led = Ledger(args.ledger)

    if args.cmd == "submit":
        delta = []
        if args.data_delta:
            for ln in Path(args.data_delta).read_text().splitlines():
                if ln.strip():
                    delta.append(json.loads(ln))
        data = {"account": args.account, "seed": args.seed,
                "epochs": args.epochs, "lr": args.lr,
                "claimed_crc": int(args.claimed_crc, 0)}
        if delta:
            data["data_delta"] = delta
        e = led.append("claim", data, author_seed=args.author_seed)
        who = (f" signed ssid {e['author']['ssid'][:12]}..."
               if "author" in e else " (anonymous)")
        print(f"[ledger] claim recorded as entry {e['idx']}"
              + (f" (+{len(delta)} contributed examples)" if delta else "")
              + who)

    elif args.cmd == "submit-structure":
        sys.path.insert(0, str(ROOT / "tools"))
        import s3_forest
        base = json.loads(Path(args.base_manifest).read_text())
        data = {"account": args.account, "op": "create_leaf",
                "leaf": args.leaf, "scope": sorted(args.scope),
                "base_manifest": base,
                "base_manifest_crc": s3_forest.manifest_crc(base),
                "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
                "claimed_crc": int(args.claimed_crc, 0),
                "claimed_manifest_crc": int(args.claimed_manifest_crc, 0)}
        # Fail fast on grammar (chain position is re-checked at verify;
        # a hand-crafted malformed claim still gets a rejected verdict).
        reason = s3_forest.validate_structure(data)
        if reason is not None:
            sys.exit(f"ERROR: structural claim refused: {reason}")
        e = led.append("claim", data, author_seed=args.author_seed)
        print(f"[ledger] structural claim recorded as entry {e['idx']} "
              f"(create_leaf {args.leaf}, scope {args.scope!r})")

    elif args.cmd == "submit-portability":
        sys.path.insert(0, str(ROOT / "tools"))
        import portability
        acrc = crc32(Path(args.artifact).read_bytes()) & 0xFFFFFFFF
        data = {"account": args.account, "op": "portability",
                "artifact_crc": acrc, "profile": args.profile,
                "probe": args.probe}
        if args.measure:
            _, d = portability.metal_trajectory(args.profile,
                                                args.artifact, args.probe)
            data["claimed_digest"] = d
        elif args.claimed_digest:
            data["claimed_digest"] = int(args.claimed_digest, 0)
        else:
            sys.exit("ERROR: pass --claimed-digest or --measure")
        reason = portability.validate_portability(data)
        if reason is not None:
            sys.exit(f"ERROR: portability claim refused: {reason}")
        e = led.append("claim", data, author_seed=args.author_seed)
        print(f"[ledger] portability claim recorded as entry {e['idx']} "
              f"(artifact {acrc:#010x} on {args.profile}, digest "
              f"{data['claimed_digest']:#010x})")

    elif args.cmd == "verify":
        claim = next((e["data"] for e in led.entries
                      if e["idx"] == args.claim and e["type"] == "claim"),
                     None)
        if claim is None:
            sys.exit(f"ERROR: no claim at entry {args.claim}")
        for e in led.entries:
            if e["type"] == "verdict" and \
               e["data"]["claim_idx"] == args.claim:
                sys.exit(f"ERROR: claim {args.claim} already has a "
                         f"verdict (entry {e['idx']}) - not replaying")
        if claim.get("op") == "portability":
            sys.path.insert(0, str(ROOT / "tools"))
            import portability
            reason = portability.validate_portability(claim)
            if reason is None:
                # Coverage already minted for this pair? Deterministic
                # from the log; settled without booting anything.
                claims_by_idx = {e["idx"]: e["data"] for e in led.entries
                                 if e["type"] == "claim"}
                for e in led.entries:
                    if e["type"] != "verdict" or \
                       not e["data"].get("verified"):
                        continue
                    c2 = claims_by_idx.get(e["data"]["claim_idx"], {})
                    if c2.get("op") == "portability" and \
                       c2["artifact_crc"] == claim["artifact_crc"] and \
                       c2["profile"] == claim["profile"]:
                        reason = "coverage already minted for this pair"
                        break
            if reason is not None:
                led.append("verdict", {"claim_idx": args.claim,
                                       "verified": False,
                                       "replay_ok": False,
                                       "reason": reason})
                print(f"[ledger] portability claim {args.claim}: -> "
                      f"REJECTED ({reason})")
                return
            acrc = crc32(Path(args.artifact).read_bytes()) & 0xFFFFFFFF
            if acrc != claim["artifact_crc"]:
                sys.exit(f"ERROR: artifact {args.artifact} has crc "
                         f"{acrc:#010x}, claim covers "
                         f"{claim['artifact_crc']:#010x} - not verdictable "
                         f"with this file")
            replay_ok, law_ok, metal_d, law_d = \
                portability.verify_portability(claim, args.artifact)
            verified = bool(replay_ok and law_ok)
            led.append("verdict", {"claim_idx": args.claim,
                                   "verified": verified,
                                   "replay_ok": replay_ok,
                                   "law_ok": law_ok,
                                   "computed_digest": metal_d,
                                   "law_digest": law_d})
            if verified:
                outcome = (f"VERIFIED (metal == claim == law, "
                           f"{metal_d:#010x}; coverage minted, "
                           f"+{MINT_PER_PORTABILITY})")
            elif not replay_ok:
                outcome = (f"REJECTED (dishonest: metal {metal_d:#010x} "
                           f"!= claimed {claim['claimed_digest']:#010x})")
            else:
                outcome = (f"REJECTED (profile diverges from law: metal "
                           f"{metal_d:#010x} != law {law_d:#010x})")
            print(f"[ledger] portability claim {args.claim}: -> {outcome}")
            return
        if claim.get("op") == "create_leaf":
            replay_ok, leaf_crc, mcrc, heldout, reason = \
                verify_structure(claim, led.entries)
            verified = bool(replay_ok and heldout is not None
                            and heldout >= args.min_heldout)
            vd = {"claim_idx": args.claim, "verified": verified,
                  "computed_crc": leaf_crc,
                  "computed_manifest_crc": mcrc,
                  "replay_ok": replay_ok, "heldout": heldout,
                  "min_heldout": args.min_heldout}
            if reason is not None:
                vd["reason"] = reason
            led.append("verdict", vd)
            if reason is not None:
                outcome = f"REJECTED (grammar: {reason})"
            elif not replay_ok:
                outcome = "REJECTED (dishonest: structural replay disagrees)"
            elif verified:
                outcome = (f"VERIFIED (leaf heldout {heldout}% >= "
                           f"{args.min_heldout}%, structure minted, "
                           f"+{MINT_PER_VERIFIED_CLAIM})")
            else:
                outcome = (f"REJECTED (honest but below quality bar: "
                           f"heldout {heldout}% < {args.min_heldout}%)")
            print(f"[ledger] structural claim {args.claim}: -> {outcome}")
            return
        with tempfile.TemporaryDirectory() as td:
            blob = Path(td) / "replayed.bin"
            computed = replay_claim(claim, blob)
            replay_ok = computed == claim["claimed_crc"]
            heldout = None
            if replay_ok:
                # Honest - now is it WORTH anything? Worth enters through
                # the oracle port named for the specialist (dishonest
                # claims skip it: nothing of theirs to measure).
                port = WORTH_ORACLE.port(claim.get("leaf", "root"))
                heldout = round(port.measure(blob), 1)
        verified = bool(replay_ok and heldout is not None
                        and heldout >= args.min_heldout)
        e = led.append("verdict", {
            "claim_idx": args.claim, "verified": verified,
            "computed_crc": computed, "replay_ok": replay_ok,
            "heldout": heldout, "min_heldout": args.min_heldout})
        crc_s = f"{computed:#010x}" if computed is not None else "none"
        if not replay_ok:
            outcome = "REJECTED (dishonest: replay disagrees)"
        elif verified:
            outcome = (f"VERIFIED (heldout {heldout}% >= "
                       f"{args.min_heldout}%, +{MINT_PER_VERIFIED_CLAIM} "
                       f"minted)")
        else:
            outcome = (f"REJECTED (honest but below quality bar: "
                       f"heldout {heldout}% < {args.min_heldout}%)")
        print(f"[ledger] claim {args.claim}: claimed "
              f"{claim['claimed_crc']:#010x}, replay produced "
              f"{crc_s} -> {outcome}")

    elif args.cmd == "transfer":
        led.append("transfer", {"src": args.src, "dst": args.dst,
                                "amount": args.amount})
        print(f"[ledger] transfer recorded")

    elif args.cmd == "balances":
        balances, errors = led.fold()
        for a in sorted(balances):
            print(f"{a}: {balances[a]}")
        print(f"supply: {sum(balances.values())}")
        if errors:
            sys.exit("ERROR: ledger fails audit; balances not trustworthy")

    elif args.cmd == "audit":
        balances, errors = led.fold()
        n_claims = sum(1 for e in led.entries if e["type"] == "claim")
        n_ok = sum(1 for e in led.entries
                   if e["type"] == "verdict" and e["data"]["verified"])
        print(f"[ledger] {len(led.entries)} entries, {n_claims} claims, "
              f"{n_ok} verified, supply {sum(balances.values())}")
        if errors:
            for err in errors:
                print("  -", err)
            print("[ledger] audit: FAIL")
            sys.exit(1)
        print("[ledger] audit: PASS - state is a pure function of history")


if __name__ == "__main__":
    main()
