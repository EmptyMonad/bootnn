#!/usr/bin/env python3
"""
Hardware-profile portability harness (S2 reward class; ROADMAP item 7).

A portability claim certifies "artifact X computes the law on profile
P". The verifier patches the artifact into a copy of the image, boots
the NAMED profile, drives the claim's probe keys, and digests the
law-visible trajectory: (key, last_cmd, h_crc) per key - h included,
so coverage extends to the working memory itself. Two equalities must
hold for the mint:

  honesty:      metal digest == claimed digest
  faithfulness: metal digest == the simulator's prediction
                (the law is the reference: a substrate that boots but
                diverges from the law covers nothing)

The profile table is the verifier's capability list; a claim naming a
profile this verifier cannot boot is not verdictable here. The mint
prices VERIFIED COVERAGE of the (artifact, profile) pair - once per
pair, identity-blind like every mint. Adding a real second profile
(RISC-V, per the decade view) means adding a table entry and a boot
recipe; the claim grammar and the digests do not change.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from zlib import crc32

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from boot_test import QMP, find_qemu  # noqa: E402
from build import SECTOR_SIZE, WEIGHT_SECTOR  # noqa: E402
from interactive_test import DEMO_KEYS  # noqa: E402
from train import SsmMachine  # noqa: E402

IMAGE = ROOT / "dnos.img"
SYMBOLS = ROOT / "dnos_symbols.json"

# The verifier's capability list. Keys are profile ids carried in
# claims; values describe how THIS verifier boots that profile.
PROFILES = {
    "qemu-tcg-x86": {
        "desc": "QEMU TCG emulation, 32-bit x86 PC (the CI profile)",
        "boot_seconds": 6.0,
        "key_seconds": 0.8,
    },
}

PROBE_SAFE = set("abcdefghijklmnopqrstuvwxyz0123456789")


def validate_portability(data):
    """The grammar. None for a well-formed portability claim, else the
    deterministic reason it is refused."""
    if data.get("op") != "portability":
        return f"unknown portability op: {data.get('op')!r}"
    if data.get("profile") not in PROFILES:
        return (f"unknown profile {data.get('profile')!r} "
                f"(this verifier boots: {sorted(PROFILES)})")
    probe = data.get("probe")
    if not probe or not isinstance(probe, str):
        return "probe must be a non-empty event string"
    if not set(probe) <= PROBE_SAFE:
        return "probe keys must be qcode-safe [a-z0-9]"
    for f in ("artifact_crc", "claimed_digest"):
        if f not in data:
            return f"missing field: {f}"
    return None


def _h_crc(h):
    """CRC32 of h as unsigned little-endian int16 words - the exact
    byte order swarm_test reads from guest memory."""
    raw = b"".join((int(v) & 0xFFFF).to_bytes(2, "little") for v in h)
    return crc32(raw) & 0xFFFFFFFF


def digest_of(records):
    """The trajectory commitment: canonical text over (key, last_cmd,
    h_crc) per probe key, CRC32'd."""
    blob = "".join(f"{k}:{cmd}:{h:08x}\n" for k, cmd, h in records)
    return crc32(blob.encode()) & 0xFFFFFFFF


def sim_trajectory(weights_path, probe):
    """The law's prediction for the probe: the same boot demo the
    kernel plays, then the probe keys, on a fresh SsmMachine."""
    machine = SsmMachine(str(weights_path))
    machine.run([ord(c) for c in DEMO_KEYS])
    records = []
    for k in probe:
        out = machine.step(ord(k))
        records.append((k, int(np.argmax(out[:20])), _h_crc(machine.h)))
    return records, digest_of(records)


def metal_trajectory(profile_id, artifact_path, probe, port=55670,
                     qemu=None):
    """Boot the named profile with the artifact patched into a COPY of
    the image, drive the probe over QMP, read (last_cmd, h_crc) after
    each key. Returns (records, digest)."""
    profile = PROFILES[profile_id]
    if not IMAGE.is_file() or not SYMBOLS.is_file():
        sys.exit("ERROR: dnos.img / dnos_symbols.json missing - run "
                 "tools/build.py first")
    sym = json.loads(SYMBOLS.read_text())
    from swarm_test import h_digest, read_sym  # noqa: E402 (needs sym file)

    blob = Path(artifact_path).read_bytes()
    with tempfile.TemporaryDirectory() as td:
        img = Path(td) / "portability.img"
        shutil.copyfile(IMAGE, img)
        with img.open("r+b") as f:
            f.seek(WEIGHT_SECTOR * SECTOR_SIZE)
            f.write(blob)
        cmd = [qemu or find_qemu(None), "-drive",
               f"file={img},format=raw", "-m", "16M", "-display", "none",
               "-qmp", f"tcp:127.0.0.1:{port},server,nowait",
               "-no-reboot", "-no-shutdown"]
        print(f"[portability] booting {profile_id}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        records = []
        try:
            qmp = QMP("127.0.0.1", port)
            time.sleep(profile["boot_seconds"])
            for k in probe:
                qmp.execute("send-key", keys=[{"type": "qcode", "data": k}])
                time.sleep(profile["key_seconds"])
                records.append((k, read_sym(qmp, sym["last_cmd"], 4),
                                h_digest(qmp, sym)))
            try:
                qmp.execute("quit", expect_reply=False)
            except OSError:
                pass
        finally:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    return records, digest_of(records)


def verify_portability(claim, artifact_path, port=55670):
    """The verdict core: one metal replay, one law prediction.
    Returns (replay_ok, law_ok, metal_digest, law_digest)."""
    _, law_d = sim_trajectory(artifact_path, claim["probe"])
    _, metal_d = metal_trajectory(claim["profile"], artifact_path,
                                  claim["probe"], port=port)
    return (metal_d == claim["claimed_digest"], metal_d == law_d,
            metal_d, law_d)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="measure a portability trajectory digest")
    ap.add_argument("--artifact", default=str(ROOT / "weights_ssm.bin"))
    ap.add_argument("--profile", default="qemu-tcg-x86")
    ap.add_argument("--probe", default="pblofc")
    ap.add_argument("--port", type=int, default=55670)
    args = ap.parse_args()
    _, sim = sim_trajectory(args.artifact, args.probe)
    _, metal = metal_trajectory(args.profile, args.artifact, args.probe,
                                port=args.port)
    crc = crc32(Path(args.artifact).read_bytes()) & 0xFFFFFFFF
    print(f"[portability] artifact {crc:#010x} on {args.profile}: "
          f"metal {metal:#010x}, law {sim:#010x} "
          f"({'FAITHFUL' if metal == sim else 'DIVERGED'})")


if __name__ == "__main__":
    main()
