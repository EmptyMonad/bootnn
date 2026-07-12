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
    "qemu-tcg-riscv32-virt": {
        "desc": "QEMU TCG, RISC-V rv32imac virt machine - the dnos-rv "
                "law runner (rv/), events over the S1 wire",
        "boot_seconds": 2.5,
        "key_seconds": 0.4,
    },
}

# The rv runner's fixed state block (rv/src/main.rs) - the RISC-V
# equivalent of dnos_symbols.json, known by construction.
RV_ELF = ROOT / "rv" / "target" / "riscv32imac-unknown-none-elf" \
    / "release" / "dnos-rv"
RV_SYM = {"magic": 0x80200000, "hdr_status": 0x80200004,
          "last_cmd": 0x80200008, "h_state": 0x80200018}
RV_MAGIC = 0x52564E44

PROBE_SAFE = set("abcdefghijklmnopqrstuvwxyz0123456789")

# Coverage is only as deep as the probe: a claim whose probe grazes
# one key certifies almost nothing. Policy floor - the probe must
# exercise at least this many DISTINCT trigger keys (keys the law maps
# to commands), so a mint always covers a real slice of behavior.
MIN_PROBE_TRIGGERS = 6


def _trigger_keys():
    from ssm_lab import SINGLE_KEYS
    return {k for k, _ in SINGLE_KEYS}


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
    n_triggers = len(set(probe) & _trigger_keys())
    if n_triggers < MIN_PROBE_TRIGGERS:
        return (f"probe exercises {n_triggers} distinct trigger keys; "
                f"coverage floor is {MIN_PROBE_TRIGGERS}")
    for f in ("artifact_crc", "claimed_digest"):
        if f not in data:
            return f"missing field: {f}"
    return None


# What each profile's SUBSTRATE can actually run. The x86 kernel's
# buffers are compiled for the canonical topology (its own
# validate_weights rejects anything else at boot - this check just
# fails fast and legibly, before a boot that would halt red). The rv
# runner reads topology from the header but its h block and the
# digest domain are 512 channels, and its readout buffers cap r1/r2.
X86_TOPOLOGY = (8, 512, 1024, 384, 64)


def _check_topology(profile_id, artifact_path):
    import struct
    hdr = Path(artifact_path).read_bytes()[:16]
    if hdr[:2] != b"DN" or hdr[2] != 5:
        sys.exit(f"ERROR: {artifact_path} is not a v5 blob")
    sizes = struct.unpack_from("<HHHHH", hdr, 5)
    if profile_id == "qemu-tcg-x86" and sizes != X86_TOPOLOGY:
        sys.exit(f"ERROR: topology {sizes} is not runnable on "
                 f"{profile_id} (kernel is compiled for {X86_TOPOLOGY})")
    if profile_id == "qemu-tcg-riscv32-virt" and \
       (sizes[1] != 512 or sizes[2] > 1024 or sizes[3] > 512):
        sys.exit(f"ERROR: topology {sizes} is not runnable on "
                 f"{profile_id} (h must be 512; readout caps 1024/512)")


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
    """Boot the named profile with the artifact supplied verbatim,
    drive the probe, read (last_cmd, h_crc) after each key over QMP.
    Returns (records, digest). Each profile is a boot recipe; the
    digest domain is identical across profiles - that is the claim."""
    _check_topology(profile_id, artifact_path)
    if profile_id == "qemu-tcg-riscv32-virt":
        return _riscv_trajectory(artifact_path, probe, port, qemu)
    return _x86_trajectory(profile_id, artifact_path, probe, port, qemu)


def _sibling_qemu(name):
    """Find a qemu system binary next to the discovered i386 one, or
    on PATH (CI installs qemu-system-misc there)."""
    sib = Path(find_qemu(None)).parent / f"{name}.exe"
    if sib.is_file():
        return str(sib)
    found = shutil.which(name)
    if found:
        return found
    sys.exit(f"ERROR: {name} not found (next to qemu-system-i386 or "
             f"on PATH)")


def _riscv_trajectory(artifact_path, probe, port, qemu):
    """The rv law runner: no image, no patching - QEMU's loader places
    the artifact verbatim at the blob address; events travel as S1 v1
    frames over the UART (virt has no keyboard, and the wire is the
    substrate-neutral input surface anyway)."""
    from dnos_client import frame  # noqa: E402
    from swarm_test import h_digest, read_sym  # noqa: E402
    profile = PROFILES["qemu-tcg-riscv32-virt"]
    if not RV_ELF.is_file():
        print("[portability] building rv law runner...")
        r = subprocess.run(
            ["cargo", "build", "--release", "--manifest-path",
             str(ROOT / "rv" / "Cargo.toml"),
             "--target", "riscv32imac-unknown-none-elf"],
            capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            sys.exit("ERROR: rv runner build failed")
    serial_port = port + 1
    cmd = [qemu or _sibling_qemu("qemu-system-riscv32"),
           "-M", "virt", "-bios", "none", "-kernel", str(RV_ELF),
           "-device", f"loader,file={artifact_path},addr=0x80400000",
           "-m", "32M", "-display", "none",
           "-qmp", f"tcp:127.0.0.1:{port},server,nowait",
           "-serial", f"tcp:127.0.0.1:{serial_port},server=on,wait=off",
           "-no-reboot", "-no-shutdown"]
    print(f"[portability] booting qemu-tcg-riscv32-virt: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    records = []
    try:
        qmp = QMP("127.0.0.1", port)
        time.sleep(profile["boot_seconds"])
        if read_sym(qmp, RV_SYM["magic"], 4) != RV_MAGIC:
            sys.exit("ERROR: rv runner state block missing after boot")
        if read_sym(qmp, RV_SYM["hdr_status"], 4) != 1:
            sys.exit("ERROR: rv runner refused the artifact "
                     "(header/CRC validation failed)")
        import socket
        wire = socket.create_connection(("127.0.0.1", serial_port),
                                        timeout=3)
        for k in probe:
            wire.sendall(frame(ord(k)))
            time.sleep(profile["key_seconds"])
            records.append((k, read_sym(qmp, RV_SYM["last_cmd"], 4),
                            h_digest(qmp, RV_SYM)))
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


def _x86_trajectory(profile_id, artifact_path, probe, port, qemu):
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
