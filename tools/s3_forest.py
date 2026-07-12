#!/usr/bin/env python3
"""
S3 spike: branching-leaf specialists, structure-first
(docs/SWARM_DESIGN.md S3; ROADMAP item 5).

Two 512-channel v5 leaves behind a HARD deterministic router — no
blending, no softmax, no parameters in the routing decision. The
router is a static total function key -> leaf, so the path is a pure
function of the event log and replay retraces it bit-exactly.

Structure under test (width held at 512; topology is the only change):

  event ->  route(key)  ->  leaf 0 (draw specialist)   ; its own h
                        ->  leaf 1 (motion specialist) ; its own h

Each leaf SEES only its routed subsequence, in training and in
inference identically (the structural smoke test asserts ForestMachine
== isolated per-leaf machines, bit for bit, untrained — structure is
gated without any training).

Law shape (the question this spike answers empirically): each leaf
trains independently from its own seed and saves its own v5 blob with
its own CRC. The composite exists only as a MANIFEST — canonical JSON
of (router table, leaf CRCs) — whose CRC commits to the structure.
Replay therefore naturally produces ONE LAW PER LEAF; the composite
canonical law is a derived commitment over parts, not a training
artifact. Contributions of examples remain inline deltas targeting
whichever leaf their trigger routes to; creating/scoping a leaf is NOT
expressible as a delta and needs a structural claim field.

Spike scope: single-key commands + random contexts. Word commands
stay a generalist concern (letters of one word route to different
leaves; a phrase-level router is future structure, out of scope).

Usage:
  python tools/s3_forest.py --smoke          # structure gate, no training
  python tools/s3_forest.py --epochs 4000    # the spike (trains 2 leaves)
"""

import argparse
import json
import sys
from pathlib import Path
from zlib import crc32

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from ssm_lab import (HISTORY_POOL, SINGLE_KEYS, SsmLab,  # noqa: E402
                     encode)
from train import CMD, CMD_NAMES, SsmMachine, save_v5  # noqa: E402

# ── the router: a static TOTAL function, versioned by its content ──────
# The table is DATA, not code: structure claims (create_leaf) derive new
# tables from it, so the swarm's architecture is a fold over the log.
DRAW_KEYS = set("pd.brlh-v|ofcu ")          # trigger keys, draw/meta class
MOTION_KEYS = set("wksja+")                 # trigger keys, motion/color
ROUTER_VERSION = 1
FALLBACK_MOD = 2      # junk-key routing, PINNED: adding leaves must not
                      # silently re-route history noise


def base_table():
    """Router version 1: the spike's two-leaf structure, as data."""
    return {"version": ROUTER_VERSION,
            "scopes": [sorted(DRAW_KEYS), sorted(MOTION_KEYS)],
            "fallback_mod": FALLBACK_MOD}


def route(key, table=None):
    """key (int) -> leaf id. Hard, deterministic, total: explicitly
    scoped keys to their leaf; everything else (history junk) by parity
    over the PINNED fallback modulus. No state, no parameters - the
    path is a pure function of the event and the table."""
    if table is None:
        table = base_table()
    c = chr(key) if 0 <= key < 128 else ""
    for leaf, scope in enumerate(table["scopes"]):
        if c in scope:
            return leaf
    return key % table["fallback_mod"]


def derive_table(base, scope):
    """A create_leaf claim's table derivation: the new leaf's scope is
    CARVED from wherever its keys lived (explicit scopes or the
    fallback pool); the fallback modulus is pinned; the version
    increments. Pure - the new structure is a function of (base, scope)."""
    return {"version": base["version"] + 1,
            "scopes": [[k for k in s if k not in scope]
                       for s in base["scopes"]] + [sorted(scope)],
            "fallback_mod": base["fallback_mod"]}


def leaf_view(keys, leaf, table=None):
    """The subsequence a leaf sees: exactly the events routed to it."""
    return [k for k in keys if route(k, table) == leaf]


def gen_scoped(leaf, ctx_rng=None, ctx_per_key=16, table=None):
    """Scoped training data: this leaf's trigger keys, each under random
    FULL-WORLD histories filtered to the leaf's view - training sees
    precisely what inference will see."""
    if ctx_rng is None:
        ctx_rng = np.random.default_rng(20260610)
    triggers = [(k, cmd) for k, cmd in SINGLE_KEYS
                if route(ord(k), table) == leaf]
    data = []
    for k, cmd in triggers:
        data.append(([ord(k)], CMD[cmd], f"{k} -> {cmd}"))
    for k, cmd in triggers:
        for v in range(ctx_per_key):
            hist_len = int(ctx_rng.integers(1, 64))
            hist = [int(ctx_rng.choice(HISTORY_POOL))
                    for _ in range(hist_len)]
            seq = leaf_view(hist, leaf, table) + [ord(k)]
            data.append((seq, CMD[cmd], f"{k} -> {cmd} +ctx{v}"))
    return data


class ForestMachine:
    """N stateful laws behind the router. step(key) advances EXACTLY
    ONE leaf (single path) and returns its decision."""

    def __init__(self, blobs, table=None):
        self.table = table if table is not None else base_table()
        self.leaves = [SsmMachine(str(b)) for b in blobs]

    def reset(self):
        for m in self.leaves:
            m.reset()

    def step(self, key):
        return self.leaves[route(key, self.table)].step(key)

    def run(self, keys):
        self.reset()
        out = None
        for k in keys:
            out = self.step(k)
        return out


def manifest_of(table, leaf_crcs):
    """The composite law: canonical JSON committing to router + leaves.
    Its CRC is DERIVED from parts - there is no monolithic artifact.
    The commitment goes through manifest_crc (the single serialization
    authority) so builder and verifier can never disagree on the bytes."""
    m = {"router_version": table["version"],
         "scopes": table["scopes"], "fallback_mod": table["fallback_mod"],
         "leaves": [{"leaf": i, "crc": c} for i, c in enumerate(leaf_crcs)]}
    return m, manifest_crc(m)


def manifest_for(blobs, table=None):
    """manifest_of over blob files (CRCs read from disk)."""
    return manifest_of(table if table is not None else base_table(),
                       [crc32(Path(b).read_bytes()) & 0xFFFFFFFF
                        for b in blobs])


def eval_forest(machine_run, data):
    hits = 0
    for keys, cls, _ in data:
        out = machine_run(keys)
        hits += int(np.argmax(out[:20])) == cls
    return 100.0 * hits / len(data)


def heldout_scoped(leaf, per_key=50, table=None):
    rng = np.random.default_rng(99173)
    data = []
    for k, cmd in [(k, c) for k, c in SINGLE_KEYS
                   if route(ord(k), table) == leaf]:
        for _ in range(per_key):
            hl = int(rng.integers(0, 64))
            hist = [int(rng.choice(HISTORY_POOL)) for _ in range(hl)]
            data.append((hist + [ord(k)], CMD[cmd], k))
    return data


# ── the structural claim grammar (the one thing a delta can't say) ─────
# A create_leaf claim is the S2 contribution that changes STRUCTURE:
# it carries the manifest it extends (inline, CRC-bound), the carved
# scope, and the new leaf's training tuple. Verification is replay -
# re-derive the table, retrain the leaf, re-derive the composite
# commitment - so architecture, like weights, is a pure function of
# the log.

def table_of_manifest(m):
    return {"version": m["router_version"], "scopes": m["scopes"],
            "fallback_mod": m["fallback_mod"]}


def manifest_crc(m):
    blob = json.dumps(m, sort_keys=True, separators=(",", ":"))
    return crc32(blob.encode()) & 0xFFFFFFFF


def validate_structure(data, prior_manifest_crc=None):
    """The grammar. Returns None for a well-formed create_leaf claim,
    else the reason it is refused. Deterministic: validity is a pure
    function of (claim, chain position)."""
    if data.get("op") != "create_leaf":
        return f"unknown structural op: {data.get('op')!r}"
    base = data.get("base_manifest")
    if not isinstance(base, dict):
        return "base_manifest must be carried inline"
    if manifest_crc(base) != data.get("base_manifest_crc"):
        return "base_manifest does not hash to base_manifest_crc"
    if not isinstance(base.get("scopes"), list) or \
       "fallback_mod" not in base:
        # A validator returns the reason it refuses; it never raises.
        # An old-schema (draw_keys/motion_keys) or malformed manifest
        # that happens to hash consistently is refused here, not on a
        # KeyError three lines down.
        return "base_manifest is not the current schema (no scopes)"
    if prior_manifest_crc is not None and \
       data["base_manifest_crc"] != prior_manifest_crc:
        return "claim does not chain to the last verified structure"
    if data.get("leaf") != len(base["scopes"]):
        return (f"leaf must be the next index {len(base['scopes'])}, "
                f"got {data.get('leaf')}")
    scope = data.get("scope")
    if not scope or not isinstance(scope, list):
        return "scope must be a non-empty list of keys"
    if any(not isinstance(k, str) or len(k) != 1 for k in scope):
        return "scope keys must be single characters"
    if len(set(scope)) != len(scope):
        return "scope has duplicate keys"
    if not any(k in scope for k, _ in SINGLE_KEYS):
        return "scope contains no trainable trigger key"
    for f in ("seed", "epochs", "lr", "claimed_crc",
              "claimed_manifest_crc"):
        if f not in data:
            return f"missing field: {f}"
    return None


def replay_structure(claim, out):
    """Verification IS replay, structural form: re-derive the table,
    retrain the new leaf deterministically on its scoped view, save
    the blob to `out`, and re-derive the composite commitment.
    Returns (leaf_crc, manifest_crc, table); the caller compares both
    CRCs to the claimed values."""
    base = claim["base_manifest"]
    table = derive_table(table_of_manifest(base), claim["scope"])
    leaf = claim["leaf"]
    np.random.seed(claim["seed"])
    stream = np.random.default_rng(20260610)
    lab = SsmLab(h=512, seed=claim["seed"])
    lab.train(claim["epochs"], claim["lr"],
              resample=lambda: gen_scoped(leaf, stream, table=table),
              val_data=gen_scoped(leaf, np.random.default_rng(555000),
                                  table=table))
    lab.save(str(out))
    leaf_crc = crc32(Path(out).read_bytes()) & 0xFFFFFFFF
    _, mcrc = manifest_of(table,
                          [x["crc"] for x in base["leaves"]] + [leaf_crc])
    return leaf_crc, mcrc, table


def leaf_gauntlet(blob, leaf, table, per_key=50):
    """Scoped worth: held-out full-world histories, the leaf measured
    on ITS OWN VIEW (bit-identical to the routed path, per the smoke
    gate). This is the software implementation behind the leaf's
    oracle port."""
    machine = SsmMachine(str(blob))
    data = heldout_scoped(leaf, per_key, table)
    hits = 0
    for keys, cls, _ in data:
        out = machine.run(leaf_view(keys, leaf, table))
        hits += int(np.argmax(out[:20])) == cls
    return 100.0 * hits / len(data)


def smoke():
    """Structure gate, zero training: ForestMachine must equal isolated
    per-leaf machines fed their own views, bit for bit - for the base
    2-leaf table AND for a table derived by a create_leaf carve. The
    table-driven router must also reproduce the spike's hardcoded
    routing key for key."""
    # Legacy invariance: the table IS the old code path.
    for k in range(128):
        c = chr(k)
        want = 0 if c in DRAW_KEYS else 1 if c in MOTION_KEYS else k % 2
        assert route(k) == want, f"table-driven route diverged on key {k}"
    tmp = ROOT / "swarm_shots"
    tmp.mkdir(exist_ok=True)
    blobs = []
    for i in (0, 1, 2):
        np.random.seed(4000 + i)
        lab = SsmLab(h=512, seed=4000 + i)
        p = tmp / f"smoke_leaf{i}.bin"
        lab.save(str(p))
        blobs.append(p)

    def bit_exact(blob_list, table, runs):
        forest = ForestMachine(blob_list, table)
        iso = [SsmMachine(str(b)) for b in blob_list]
        rng = np.random.default_rng(7)
        ok = True
        for _ in range(runs):
            seq = [int(rng.choice(HISTORY_POOL))
                   for _ in range(int(rng.integers(1, 64)))]
            out_f = forest.run(seq)
            views = [leaf_view(seq, i, table)
                     for i in range(len(blob_list))]
            last_leaf = route(seq[-1], table)
            out_i = iso[last_leaf].run(views[last_leaf])
            ok &= np.array_equal(np.asarray(out_f), np.asarray(out_i))
            # And every NON-deciding leaf's state == its isolated twin
            for other in range(len(blob_list)):
                if other == last_leaf:
                    continue
                iso[other].run(views[other])
                ok &= np.array_equal(forest.leaves[other].h, iso[other].h)
        return ok

    ok = bit_exact(blobs[:2], base_table(), 20)
    m, ccrc = manifest_for(blobs[:2])
    print(f"[s3] smoke: routed forest == isolated leaf views: "
          f"{'BIT-EXACT' if ok else 'DIVERGED'} (20 random logs)")
    print(f"[s3] composite commitment (derived): {ccrc:#010x} over "
          f"{[hex(x['crc']) for x in m['leaves']]}")
    # Structural carve: a third leaf takes undo+fill from the draw
    # scope; the 3-leaf routed forest must stay bit-exact against
    # isolated views under the DERIVED table.
    t3 = derive_table(base_table(), ["u", "f"])
    assert "u" not in t3["scopes"][0] and "f" not in t3["scopes"][0], \
        "carve left keys behind"
    assert route(ord("u"), t3) == 2 and route(ord("f"), t3) == 2
    ok3 = bit_exact(blobs, t3, 20)
    m3, ccrc3 = manifest_for(blobs, t3)
    print(f"[s3] smoke: derived 3-leaf table (carve u,f): "
          f"{'BIT-EXACT' if ok3 else 'DIVERGED'} (20 random logs), "
          f"commitment {ccrc3:#010x}")
    return 0 if (ok and ok3) else 1


def main():
    ap = argparse.ArgumentParser(description="S3 branching-leaf spike")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--smoke", action="store_true",
                    help="structure gate only (no training)")
    args = ap.parse_args()

    if args.smoke:
        sys.exit(smoke())

    blobs = []
    for leaf in (0, 1):
        print(f"[s3] training leaf {leaf} "
              f"({'draw' if leaf == 0 else 'motion'} specialist, h=512)...")
        np.random.seed(1337 + leaf)
        stream = np.random.default_rng(20260610)
        lab = SsmLab(h=512, seed=1337 + leaf)
        lab.train(args.epochs, args.lr,
                  resample=lambda l=leaf: gen_scoped(l, stream),
                  val_data=gen_scoped(leaf, np.random.default_rng(555000)))
        p = ROOT / f"weights_leaf{leaf}.bin"
        lab.save(str(p))
        blobs.append(p)

    m, ccrc = manifest_for(blobs)
    (ROOT / "forest_manifest.json").write_text(
        json.dumps(m, sort_keys=True, indent=1) + "\n")
    forest = ForestMachine(blobs)
    base = SsmMachine(str(ROOT / "weights_ssm.bin"))

    # Held-out per leaf (the leaf's own law shape) and composite.
    print("\n[s3] held-out (single-key + random full-world contexts):")
    all_data = []
    for leaf in (0, 1):
        data = heldout_scoped(leaf)
        all_data += data
        acc = eval_forest(forest.run, data)
        print(f"  leaf {leaf}: {acc:.1f}%  ({len(data)} cases)")
    comp = eval_forest(forest.run, all_data)
    gen = eval_forest(base.run, all_data)
    print(f"  composite (routed forest): {comp:.1f}%")
    print(f"  baseline (single 512 generalist, same suite): {gen:.1f}%")
    print(f"\n[s3] LAW SHAPE: one law PER LEAF "
          f"({', '.join(hex(x['crc']) for x in m['leaves'])}); composite "
          f"is a derived commitment {ccrc:#010x} over router+leaves - "
          f"not a training artifact.")
    ok = comp >= 95.0
    print(f"[s3] result: {'GATES MET' if ok else 'below gates'} "
          f"(composite vs 95% bar)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
