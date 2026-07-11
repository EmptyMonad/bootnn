#!/usr/bin/env python3
"""
Hash-based one-time signatures (Lamport over blake2b) - the optional
self-sovereign authorship proof for the ledger.

Registry-free and provable-without-lookup: the public key is carried
in the signed entry; the identity (SSID) is blake2b(pubkey), so it is
self-certifying - a verifier recomputes it, never consults a registry
or CA. Post-quantum by construction (security rests only on the hash).
Stdlib only - no dependency, per the deferred-crypto decision.

One-time: each keypair signs exactly one message (here, one ledger
entry body). A persistent multi-signature identity is a Merkle tree of
these leaves - deferred, flagged where it bites.
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
