#!/usr/bin/env python3
"""
S2 prototype: the contribution ledger (docs/SWARM_DESIGN.md, S2).

An append-only, hash-chained event log where:

  - a *claim* is a training-contribution tuple
    (account, seed, epochs, lr, claimed_crc) — "I ran this training
    and got this blob";
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
        """Replay the log from genesis. Returns (balances, errors)."""
        balances = {}
        errors = []
        claims = {}
        verdicted = set()
        prev = GENESIS
        for e in self.entries:
            if e["prev"] != prev or \
               e["hash"] != entry_hash(e["idx"], e["prev"], e["type"], e["data"]):
                errors.append(f"entry {e['idx']}: hash chain broken")
                prev = e["hash"]
                continue
            prev = e["hash"]
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
                        acct = claims[ci]["account"]
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

    def append(self, etype, data):
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
        e = {"idx": idx, "prev": prev, "type": etype, "data": data,
             "hash": entry_hash(idx, prev, etype, data)}
        self.entries.append(e)
        with self.path.open("a") as f:
            f.write(canonical(e) + "\n")
        return e


def replay_claim(claim):
    """Verification IS replay: rerun the training, CRC the artifact."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "replayed.bin"
        cmd = [sys.executable, str(TRAIN),
               "--seed", str(claim["seed"]),
               "--epochs", str(claim["epochs"]),
               "--lr", str(claim["lr"]),
               "--output", str(out)]
        print(f"[ledger] replaying: {' '.join(cmd[1:])}")
        # train.py's exit code also gates accuracy/divergence; the ledger
        # only asks whether the *artifact* is reproduced. The economy
        # gates on the full gauntlet separately before a blob becomes law.
        subprocess.run(cmd, capture_output=True, timeout=3600)
        if not out.is_file():
            return None
        return crc32(out.read_bytes()) & 0xFFFFFFFF


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

    p = sub.add_parser("verify", help="replay a claim and record the verdict")
    p.add_argument("--claim", type=int, required=True)

    p = sub.add_parser("transfer")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--amount", type=int, required=True)

    sub.add_parser("balances")
    sub.add_parser("audit")

    args = ap.parse_args()
    led = Ledger(args.ledger)

    if args.cmd == "submit":
        e = led.append("claim", {
            "account": args.account, "seed": args.seed,
            "epochs": args.epochs, "lr": args.lr,
            "claimed_crc": int(args.claimed_crc, 0)})
        print(f"[ledger] claim recorded as entry {e['idx']}")

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
        computed = replay_claim(claim)
        verified = computed == claim["claimed_crc"]
        e = led.append("verdict", {
            "claim_idx": args.claim, "verified": verified,
            "computed_crc": computed})
        print(f"[ledger] claim {args.claim}: claimed "
              f"{claim['claimed_crc']:#010x}, replay produced "
              f"{computed:#010x} -> "
              f"{'VERIFIED (+%d minted)' % MINT_PER_VERIFIED_CLAIM
                 if verified else 'REJECTED'}")

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
