#!/usr/bin/env python3
"""
S1 differential test: the input surface is universal
(docs/SWARM_DESIGN.md, S1 — v0 byte framing).

Boots two identical nodes. Node K is driven through the PS/2 keyboard
(QMP send-key); node S is driven through COM1 (raw ASCII bytes over the
serial chardev socket). Both receive the same semantic event log, and
at every step the two nodes must occupy the same state:

  1. Equal canonical state vectors — including the ring indexes, since
     both producers convert to ASCII at the edge and feed one shared
     ring: keyboard and serial are indistinguishable to f.
  2. Equal cropped-framebuffer digests.
  3. last_cmd equal to the simulator's prediction for the identical
     history (the law gate) — law == keyboard == wire.
  4. Provenance: node S consumed its events from the UART
     (serial_rx_count == len(keys)) and node K from the ISR
     (serial_rx_count == 0). Same state, different producers — that is
     the claim.

Humans and agents are peers above the same deterministic interface;
this test is that sentence, executable.

Usage:
  python tools/serial_test.py
  python tools/serial_test.py --keys pblofc
"""

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from boot_test import QMP, find_qemu            # noqa: E402
from interactive_test import DEMO_KEYS          # noqa: E402
from swarm_test import (fb_digest, read_sym,    # noqa: E402
                        send_key, state_vector)
from train import (CMD_NAMES, sequence_to_input,  # noqa: E402
                   simulate_assembly_forward)

IMAGE = ROOT / "dnos.img"
WEIGHTS = ROOT / "weights.bin"
SYMBOLS = ROOT / "dnos_symbols.json"

QMP_K, QMP_S, SERIAL_PORT = 55640, 55641, 55642


def main():
    ap = argparse.ArgumentParser(description="DNOS serial differential test")
    ap.add_argument("--qemu")
    ap.add_argument("--keys", default="pblofc")
    ap.add_argument("--boot-seconds", type=float, default=6.0)
    ap.add_argument("--settle-seconds", type=float, default=0.8)
    args = ap.parse_args()

    if not SYMBOLS.is_file():
        sys.exit("ERROR: dnos_symbols.json missing - run tools/build.py first")
    sym = json.loads(SYMBOLS.read_text())
    qemu = find_qemu(args.qemu)

    base = ["-drive", f"file={IMAGE},format=raw", "-m", "16M",
            "-display", "none", "-snapshot", "-no-reboot", "-no-shutdown"]
    failures = []
    procs = []
    try:
        procs.append(subprocess.Popen(
            [qemu, *base, "-qmp", f"tcp:127.0.0.1:{QMP_K},server,nowait"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen(
            [qemu, *base, "-qmp", f"tcp:127.0.0.1:{QMP_S},server,nowait",
             "-serial", f"tcp:127.0.0.1:{SERIAL_PORT},server=on,wait=off"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        qk, qs = QMP("127.0.0.1", QMP_K), QMP("127.0.0.1", QMP_S)

        deadline = time.time() + 10
        while True:
            try:
                wire = socket.create_connection(("127.0.0.1", SERIAL_PORT),
                                                timeout=2)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.2)

        print(f"[serial] node K (keyboard) + node S (COM1) launched; "
              f"waiting {args.boot_seconds}s for boot + demo...")
        time.sleep(args.boot_seconds)

        def checkpoint(tag):
            sk, ss = state_vector(qk, sym), state_vector(qs, sym)
            dk = fb_digest(qk, "K", f"serial_{tag}")
            ds = fb_digest(qs, "S", f"serial_{tag}")
            ok = sk == ss and dk == ds
            print(f"[serial] {tag:12s} state={'==' if sk == ss else 'DIVERGED'} "
                  f"fb={'==' if dk == ds else 'DIVERGED'} fb_crc={dk:#010x}")
            if not ok:
                print(f"[serial]   K: fb={dk:#010x} {dict(sk)}")
                print(f"[serial]   S: fb={ds:#010x} {dict(ss)}")
                failures.append(f"paths diverged at '{tag}'")
            return sk

        if read_sym(qs, sym["serial_present"], 1) != 1:
            failures.append("node S did not detect its UART")

        checkpoint("post-demo")

        history = [ord(c) for c in DEMO_KEYS[::-1]]
        for key in args.keys:
            expected = int(np.argmax(simulate_assembly_forward(
                WEIGHTS, sequence_to_input([ord(key)] + history))[:20]))
            send_key(qk, key)                 # node K: PS/2
            wire.sendall(key.encode("ascii"))  # node S: the wire
            time.sleep(args.settle_seconds)
            states = checkpoint(f"after_{key}")
            actual = dict(states)["last_cmd"]
            if actual != expected:
                act = (CMD_NAMES[actual]
                       if actual < len(CMD_NAMES) else f"cmd{actual}")
                failures.append(f"'{key}': law={CMD_NAMES[expected]} metal={act}")
            history = [ord(key)] + history

        # ── provenance: same state, different producers ──────────────────
        rx_k = read_sym(qk, sym["serial_rx_count"], 4)
        rx_s = read_sym(qs, sym["serial_rx_count"], 4)
        print(f"[serial] provenance: node K serial_rx={rx_k} (want 0), "
              f"node S serial_rx={rx_s} (want {len(args.keys)})")
        if rx_k != 0:
            failures.append(f"node K consumed {rx_k} serial bytes (want 0)")
        if rx_s != len(args.keys):
            failures.append(f"node S serial_rx_count={rx_s}, "
                            f"want {len(args.keys)}")
        wire.close()
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    if failures:
        print("[serial] result: FAIL")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("[serial] result: PASS - keyboard and wire are the same interface")


if __name__ == "__main__":
    main()
