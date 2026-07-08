#!/usr/bin/env python3
"""
DNOS thin client (SWARM_DESIGN S1): one deterministic interface,
two kinds of peer.

Launches a node and drives it over COM1 — the same wire an autonomous
agent would use. Humans get an interactive terminal; agents get
--send/--stdin. Either way the session is an *event log*: every byte
sent is recorded, and the log replays to the same machine state,
bit for bit (--replay verifies this).

The client is also a live differential monitor: for every event it
shows the law's prediction next to what the metal decoded. If they
ever disagree, the session exits non-zero.

Usage:
  python tools/dnos_client.py                    # interactive terminal
  python tools/dnos_client.py --send pblofc      # agent one-shot
  echo -n pblofc | python tools/dnos_client.py --stdin
  python tools/dnos_client.py --replay sessions/session_*.jsonl

Interactive keys: printable ASCII is sent as events; ESC halts the
node (byte 27 is the kernel's halt event) and ends the session;
Ctrl+C ends the session without halting the node.
"""

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zlib import crc32

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from boot_test import QMP, find_qemu            # noqa: E402
from interactive_test import DEMO_KEYS          # noqa: E402
from swarm_test import fb_digest, state_vector  # noqa: E402
from train import (CMD_NAMES, sequence_to_input,  # noqa: E402
                   simulate_assembly_forward)

IMAGE = ROOT / "dnos.img"
WEIGHTS = ROOT / "weights.bin"
SYMBOLS = ROOT / "dnos_symbols.json"
SESSIONS = ROOT / "sessions"

QMP_PORT, SERIAL_PORT = 55660, 55661


class Node:
    """A booted DNOS node reachable over QMP (state) and COM1 (events)."""

    def __init__(self, qemu, boot_seconds):
        self.proc = subprocess.Popen(
            [qemu, "-drive", f"file={IMAGE},format=raw", "-m", "16M",
             "-display", "none", "-snapshot", "-no-reboot", "-no-shutdown",
             "-qmp", f"tcp:127.0.0.1:{QMP_PORT},server,nowait",
             "-serial", f"tcp:127.0.0.1:{SERIAL_PORT},server=on,wait=off"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.qmp = QMP("127.0.0.1", QMP_PORT)
        deadline = time.time() + 10
        while True:
            try:
                self.wire = socket.create_connection(
                    ("127.0.0.1", SERIAL_PORT), timeout=2)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.2)
        print(f"[client] node booting ({boot_seconds}s: boot + demo)...")
        time.sleep(boot_seconds)

    def close(self):
        try:
            self.wire.close()
        except OSError:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def main():
    ap = argparse.ArgumentParser(description="DNOS thin client")
    ap.add_argument("--qemu")
    ap.add_argument("--send", help="send these events and exit (agent mode)")
    ap.add_argument("--stdin", action="store_true",
                    help="read events from stdin until EOF (agent mode)")
    ap.add_argument("--replay", help="replay a session log and verify its "
                                     "recorded final state")
    ap.add_argument("--log", help="session log path "
                                  "(default sessions/session_<utc>.jsonl)")
    ap.add_argument("--boot-seconds", type=float, default=6.0)
    ap.add_argument("--settle-seconds", type=float, default=0.5)
    args = ap.parse_args()

    if not SYMBOLS.is_file():
        sys.exit("ERROR: dnos_symbols.json missing - run tools/build.py first")
    sym = json.loads(SYMBOLS.read_text())
    weights_crc = crc32(WEIGHTS.read_bytes()) & 0xFFFFFFFF

    # ── choose the event source ─────────────────────────────────────────
    replay_footer = None
    if args.replay:
        lines = [json.loads(ln) for ln in
                 Path(args.replay).read_text().splitlines() if ln.strip()]
        replay_footer = next(e for e in lines if e.get("type") == "final")
        events = [e["event"] for e in lines if e.get("type") == "event"]
        if replay_footer["weights_crc"] != weights_crc:
            sys.exit(f"ERROR: log was recorded against weights CRC "
                     f"{replay_footer['weights_crc']:#010x}, current blob is "
                     f"{weights_crc:#010x} - not the same law")
        source = iter(events)
        print(f"[client] replaying {len(events)} events from {args.replay}")
    elif args.send is not None:
        source = iter(ord(c) for c in args.send)
    elif args.stdin:
        source = iter(sys.stdin.buffer.read())
    else:
        source = None                       # interactive

    # ── session log ─────────────────────────────────────────────────────
    log_path = None
    log_f = None
    if not args.replay:
        SESSIONS.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = Path(args.log) if args.log else \
            SESSIONS / f"session_{stamp}.jsonl"
        log_f = log_path.open("w")

    node = Node(find_qemu(args.qemu), args.boot_seconds)
    history = [ord(c) for c in DEMO_KEYS[::-1]]
    mismatches = 0
    sent = 0

    def step(byte):
        nonlocal history, mismatches, sent
        expected = int(np.argmax(simulate_assembly_forward(
            WEIGHTS, sequence_to_input([byte] + history))[:20]))
        node.wire.sendall(bytes([byte]))
        sent += 1
        if log_f:
            log_f.write(json.dumps({"type": "event", "event": byte}) + "\n")
        time.sleep(args.settle_seconds)
        st = dict(state_vector(node.qmp, sym))
        actual = st["last_cmd"]
        act = CMD_NAMES[actual] if actual < len(CMD_NAMES) else f"cmd{actual}"
        ok = actual == expected
        if not ok:
            mismatches += 1
        print(f"[client] '{chr(byte) if 32 <= byte < 127 else byte}' "
              f"law={CMD_NAMES[expected]:12s} metal={act:12s} "
              f"@({st['cursor_x']},{st['cursor_y']}) "
              f"{'OK' if ok else 'MISMATCH'}")

    try:
        if source is not None:
            for byte in source:
                step(byte)
        else:                               # interactive terminal
            print("[client] interactive: type to drive the node "
                  "(ESC halts + exits, Ctrl+C exits without halting)")
            try:
                if sys.platform == "win32":
                    import msvcrt
                    while True:
                        ch = msvcrt.getch()
                        b = ch[0]
                        if b == 27:
                            step(27)
                            break
                        if 32 <= b < 127 or b == 13:
                            step(b)
                else:
                    import termios
                    import tty
                    fd = sys.stdin.fileno()
                    old = termios.tcgetattr(fd)
                    tty.setraw(fd)
                    try:
                        while True:
                            b = sys.stdin.buffer.read(1)[0]
                            if b == 27:
                                step(27)
                                break
                            if 32 <= b < 127 or b == 13:
                                step(b)
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except KeyboardInterrupt:
                print("\n[client] session ended (node left running state)")

        # ── final state: the session's verifiable summary ───────────────
        final_state = dict(state_vector(node.qmp, sym))
        final_fb = fb_digest(node.qmp, "client", "final")
        print(f"[client] {sent} events; final fb_crc={final_fb:#010x} "
              f"last_cmd={final_state['last_cmd']}")
        if log_f:
            log_f.write(json.dumps({
                "type": "final", "weights_crc": weights_crc,
                "state": final_state, "fb_crc": final_fb}) + "\n")
            log_f.close()
            print(f"[client] session log: {log_path}")
        if replay_footer:
            same = (final_state == replay_footer["state"]
                    and final_fb == replay_footer["fb_crc"])
            print(f"[client] replay verification: "
                  f"{'PASS - identical state' if same else 'FAIL - diverged'}")
            if not same:
                print(f"[client]   recorded: fb={replay_footer['fb_crc']:#010x}"
                      f" {replay_footer['state']}")
                print(f"[client]   replayed: fb={final_fb:#010x} {final_state}")
                sys.exit(1)
    finally:
        if log_f and not log_f.closed:
            log_f.close()
        node.close()

    if mismatches:
        print(f"[client] result: FAIL ({mismatches} law-vs-metal mismatches)")
        sys.exit(1)


if __name__ == "__main__":
    main()
