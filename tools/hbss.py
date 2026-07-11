#!/usr/bin/env python3
"""
Hash-based one-time signatures (Lamport over blake2b) - the optional
self-sovereign authorship proof for the ledger.

Registry-free and provable-without-lookup: the public key is carried
in the signed entry; the identity (SSID) is blake2b(pubkey), so it is
self-certifying - a verifier recomputes it, never consults a registry
or CA. Post-quantum by construction (security rests only on the hash).
Stdlib only - no dependency, per the deferred-crypto decision.

One-time: each keypair signs exactly one message. A PERSISTENT
identity (MerkleIdentity) is a Merkle tree of 2^depth such leaves:
the SSID is the root, each signature reveals one leaf pubkey plus its
authentication path, and the verifier folds leaf->root from the
envelope alone. The legacy per-entry envelope is the depth-0
degenerate case - an empty path folds to blake2b(pubkey), which IS
the legacy SSID - so old envelopes verify unchanged.
"""

import hashlib

N = 256                      # message-digest bits = Lamport chains
HASHLEN = 32


def _h(b):
    return hashlib.blake2b(b, digest_size=HASHLEN).digest()


def keypair(seed):
    """Deterministic keypair from a seed (bytes/str). Returns
    (sk, pk): sk[i][b], pk[i][b] are 2xN preimages / their hashes."""
    seed = seed.encode() if isinstance(seed, str) else seed
    sk, pk = [], []
    for i in range(N):
        pair_sk, pair_pk = [], []
        for b in (0, 1):
            x = _h(seed + i.to_bytes(2, "big") + bytes([b]))
            pair_sk.append(x)
            pair_pk.append(_h(x))
        sk.append(pair_sk)
        pk.append(pair_pk)
    return sk, pk


def _bits(msg):
    d = hashlib.blake2b(msg, digest_size=N // 8).digest()
    return [(d[i // 8] >> (i % 8)) & 1 for i in range(N)]


def sign(sk, msg):
    """Reveal, per digest bit, the matching secret preimage. Hex-joined."""
    msg = msg.encode() if isinstance(msg, str) else msg
    return "".join(sk[i][bit].hex() for i, bit in enumerate(_bits(msg)))


def pk_hex(pk):
    return "".join(pk[i][b].hex() for i in range(N) for b in (0, 1))


def ssid(pk):
    """Self-certifying identity: blake2b of the public key."""
    return _h(bytes.fromhex(pk_hex(pk))).hex()


def verify(pk_hex_str, msg, sig_hex):
    """True iff sig's revealed preimages hash to the pubkey halves the
    message digest selects. pk_hex_str may be a pubkey or its hex."""
    msg = msg.encode() if isinstance(msg, str) else msg
    try:
        pk_flat = bytes.fromhex(pk_hex_str)
        sig_flat = bytes.fromhex(sig_hex)
    except ValueError:
        return False
    if len(pk_flat) != 2 * N * HASHLEN or len(sig_flat) != N * HASHLEN:
        return False
    bits = _bits(msg)
    for i, bit in enumerate(bits):
        reveal = sig_flat[i * HASHLEN:(i + 1) * HASHLEN]
        want = pk_flat[(2 * i + bit) * HASHLEN:(2 * i + bit + 1) * HASHLEN]
        if _h(reveal) != want:
            return False
    return True


def ssid_of_hex(pk_hex_str):
    return _h(bytes.fromhex(pk_hex_str)).hex()


# ── persistent identity: a Merkle tree of one-time leaves ────────────
# One SSID = the Merkle root over 2^depth leaf-pubkey hashes. Every
# leaf stays one-time; the identity signs up to 2^depth entries. No
# leaf/node domain-separation prefix is needed: a leaf preimage is a
# 16 KB pubkey and an internal preimage is 64 bytes of child hashes -
# the input lengths cannot collide, so the roles are separated by
# construction (and the depth-0 degenerate case stays byte-identical
# to the legacy one-time SSID).

IDENT_DEPTH = 3                  # 2^3 = 8 signatures per identity


def merkle_root_from_leaf(leaf_hash, index, path):
    """Fold a leaf hash up its authentication path (index selects
    left/right at each level). An empty path - the legacy one-time
    envelope - returns the leaf hash itself."""
    node = leaf_hash
    for sib in path:
        node = _h(node + sib) if index & 1 == 0 else _h(sib + node)
        index >>= 1
    return node


def certify(envelope):
    """True iff the envelope's SSID is the Merkle root that its own
    pubkey + path fold to. Consults nothing but the envelope:
    registry-free, provable without lookup. Handles both forms -
    persistent (leaf + path present) and legacy one-time (absent)."""
    try:
        leaf = _h(bytes.fromhex(envelope["pubkey"]))
        path = [bytes.fromhex(p) for p in envelope.get("path", [])]
    except (ValueError, KeyError):
        return False
    root = merkle_root_from_leaf(leaf, envelope.get("leaf", 0), path)
    return root.hex() == envelope.get("ssid")


class MerkleIdentity:
    """Deterministic persistent identity from one seed: 2^depth
    one-time Lamport leaves under one root. The seed alone recreates
    the whole identity, so the signer stores nothing; which leaves
    are already spent is read back from the log they signed."""

    def __init__(self, seed, depth=IDENT_DEPTH):
        seed = seed.encode() if isinstance(seed, str) else seed
        self.n_leaves = 1 << depth
        self.keys = [keypair(seed + b"/leaf/" + i.to_bytes(2, "big"))
                     for i in range(self.n_leaves)]
        self.levels = [[_h(bytes.fromhex(pk_hex(pk)))
                        for _, pk in self.keys]]
        while len(self.levels[-1]) > 1:
            lv = self.levels[-1]
            self.levels.append([_h(lv[i] + lv[i + 1])
                                for i in range(0, len(lv), 2)])
        self.ssid = self.levels[-1][0].hex()

    def envelope(self, msg, leaf_index):
        """Sign msg with one leaf; the envelope carries the leaf
        pubkey, its index, and the authentication path that proves
        leaf-in-SSID membership."""
        sk, pk = self.keys[leaf_index]
        path, idx = [], leaf_index
        for lv in self.levels[:-1]:
            path.append(lv[idx ^ 1].hex())
            idx >>= 1
        return {"ssid": self.ssid, "pubkey": pk_hex(pk),
                "sig": sign(sk, msg), "leaf": leaf_index, "path": path}
