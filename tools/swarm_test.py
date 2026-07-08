#!/usr/bin/env python3
"""
Swarm Tier 0: deterministic replication test (docs/SWARM_DESIGN.md, S0).

Boots N identical DNOS nodes, feeds every node the same event log, and
asserts that all N occupy the same state — and that the state is the one
the law predicts:

  1. After boot (the demo is kernel-internal and therefore already part
     of the shared log), all N canonical state vectors are equal.
  2. After every injected key, all N state vectors are equal AND
     last_cmd equals simulate_assembly_forward's prediction for the
     identical history (the law gate, per interactive_test.py).
  3. Framebuffer digests agree across nodes at every checkpoint. The
     status-bar telemetry rows are cropped before hashing: tick_count is
     wall-clock-relative per node and is *excluded* from replication
     (recorded, not compared) — nodes are equal in state, not in age.
  4. Negative test (divergence localization): one extra key is sent to
     node 0 only. Node 0's digest must diverge; the remaining nodes must
     still agree with each other. A replication check that cannot see a
     one-event fork is not a replication check.

All digests are computed host-side from QMP reads; the kernel does not
attest to itself in S0 (zero kernel changes).

Usage:
  python tools/swarm_test.py
  python tools/swarm_test.py --nodes 3 --keys pblofc
"""

import argparse
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from boot_test import QMP, find_qemu            # noqa: E402
from interactive_test import DEMO_KEYS          # noqa: E402
from train import (CMD_NAMES, sequence_to_input,  # noqa: E402
                   simulate_assembly_forward)

IMAGE = ROOT / "dnos.img"
WEIGHTS = ROOT / "weights.bin"
SYMBOLS = ROOT / "dnos_symbols.json"
SHOTS = ROOT / "swarm_shots"

BASE_PORT = 55600

# The canonical state vector: (symbol, byte width). tick_count/step_count
# are deliberately absent — see module docstring.
STATE_VECTOR = [
    ("hdr_status", 4), ("k_video_mode", 1),
    ("cursor_x", 4), ("cursor_y", 4),
    ("color_idx", 1), ("draw_size", 4),
    ("last_cmd", 4), ("last_confidence", 4),
    ("kb_read_idx", 2), ("kb_write_idx", 2),
]
XP_FMT = {1: "b", 2: "h", 4: "w"}

# Rows cropped from the top and bottom of every screendump before
# hashing: the status bar renders T:<tick> telemetry, which is per-node
# wall-clock state, not replicated state.
CROP_ROWS = 40


def read_sym(qmp, addr, width):
    r = qmp.execute("human-monitor-command",
                    **{"command-line": f"xp /1{XP_FMT[width]}x {addr:#x}"})
    return int(r["return"].strip().split(":")[1].strip(), 16)


def state_vector(qmp, sym):
    return tuple((name, read_sym(qmp, sym[name], width))
                 for name, width in STATE_VECTOR)


def read_counters(qmp, sym):
    return (read_sym(qmp, sym["tick_count"], 4),
            read_sym(qmp, sym["step_count"], 4))


def fb_digest(qmp, node, tag):
    """CRC32 of the framebuffer with telemetry rows cropped."""
    SHOTS.mkdir(exist_ok=True)
    path = SHOTS / f"node{node}_{tag}.ppm"
    qmp.execute("screendump", filename=str(path))
    data = path.read_bytes()
    # P6\n<w> <h>\n255\n<pixels>
    _, dims, _, px = data.split(b"\n", 3)
    w, h = (int(v) for v in dims.split())
    row = w * 3
    body = px[CROP_ROWS * row:(h - CROP_ROWS) * row]
    return zlib.crc32(body) & 0xFFFFFFFF


def send_key(qmp, key):
    qmp.execute("send-key", keys=[{"type": "qcode", "data": key}])


def all_equal(values):
    return len(set(values)) == 1


def main():
    ap = argparse.ArgumentParser(description="DNOS swarm replication test")
    ap.add_argument("--qemu")
    ap.add_argument("--nodes", type=int, default=3)
    ap.add_argument("--keys", default="pblofc")
    ap.add_argument("--boot-seconds", type=float, default=6.0)
    ap.add_argument("--settle-seconds", type=float, default=0.8)
    args = ap.parse_args()

    if not SYMBOLS.is_file():
        sys.exit("ERROR: dnos_symbols.json missing - run tools/build.py first")
    sym = json.loads(SYMBOLS.read_text())
    qemu = find_qemu(args.qemu)

    failures = []
    procs, qmps = [], []
    try:
        for i in range(args.nodes):
            procs.append(subprocess.Popen(
                [qemu, "-drive", f"file={IMAGE},format=raw", "-m", "16M",
                 "-display", "none", "-snapshot",
                 "-qmp", f"tcp:127.0.0.1:{BASE_PORT + i},server,nowait",
                 "-no-reboot", "-no-shutdown"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for i in range(args.nodes):
            qmps.append(QMP("127.0.0.1", BASE_PORT + i))
        print(f"[swarm] {args.nodes} nodes launched; waiting "
              f"{args.boot_seconds}s for boot + demo...")
        time.sleep(args.boot_seconds)

        def checkpoint(tag):
            states = [state_vector(q, sym) for q in qmps]
            digests = [fb_digest(q, i, tag) for i, q in enumerate(qmps)]
            ticks = [read_counters(q, sym) for q in qmps]
            ok = all_equal(states) and all_equal(digests)
            print(f"[swarm] {tag:12s} state={'==' if all_equal(states) else 'DIVERGED'} "
                  f"fb={'==' if all_equal(digests) else 'DIVERGED'} "
                  f"fb_crc={digests[0]:#010x} "
                  f"ticks={[t for t, _ in ticks]}")
            if not ok:
                for i, (s, d) in enumerate(zip(states, digests)):
                    print(f"[swarm]   node{i}: fb={d:#010x} {dict(s)}")
                failures.append(f"nodes diverged at '{tag}'")
            return states

        # ── gate 1: post-demo, all nodes equal ──────────────────────────
        states = checkpoint("post-demo")
        for name, val in states[0]:
            if name == "hdr_status" and val != 1:
                failures.append("weights not validated at boot")
            if name == "k_video_mode" and val != 1:
                failures.append("expected 800x600x32 under QEMU stdvga")

        # ── gate 2: shared log, per-key law==metal==metal==... ──────────
        history = [ord(c) for c in DEMO_KEYS[::-1]]
        for key in args.keys:
            expected = int(np.argmax(simulate_assembly_forward(
                WEIGHTS, sequence_to_input([ord(key)] + history))[:20]))
            for q in qmps:
                send_key(q, key)
            time.sleep(args.settle_seconds)
            states = checkpoint(f"after_{key}")
            actual = dict(states[0])["last_cmd"]
            act_name = (CMD_NAMES[actual]
                        if actual < len(CMD_NAMES) else f"cmd{actual}")
            if actual != expected:
                failures.append(f"'{key}': law={CMD_NAMES[expected]} "
                                f"metal={act_name}")
                print(f"[swarm]   law MISMATCH: expected "
                      f"{CMD_NAMES[expected]}, got {act_name}")
            history = [ord(key)] + history

        # ── gate 3: negative test — fork node 0, must be localized ──────
        send_key(qmps[0], "p")
        time.sleep(args.settle_seconds)
        states = [state_vector(q, sym) for q in qmps]
        digests = [fb_digest(q, i, "forked") for i, q in enumerate(qmps)]
        forked = (states[0], digests[0]) != (states[1], digests[1])
        rest_ok = all_equal(states[1:]) and all_equal(digests[1:])
        print(f"[swarm] fork test: node0 "
              f"{'diverged' if forked else 'DID NOT diverge'}, "
              f"others {'still agree' if rest_ok else 'DIVERGED'}")
        if not forked:
            failures.append("fork was not detected (node 0 should diverge)")
        if not rest_ok:
            failures.append("unforked nodes diverged during negative test")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    if failures:
        print("[swarm] result: FAIL")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"[swarm] result: PASS - {args.nodes} nodes, one log, one state")


if __name__ == "__main__":
    main()
