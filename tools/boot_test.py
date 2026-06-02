#!/usr/bin/env python3
"""
DNOS headless boot smoke-test.

Boots dnos.img in QEMU with no display, drives the QMP monitor to grab a
screenshot after the demo has had time to run, then quits. Also captures an
interrupt/reset trace so a triple-fault or reboot loop is visible.

This is the automatable core of a CI check: build -> boot_test -> inspect.

Usage:
  python tools/boot_test.py                       # boot, screenshot after 4s
  python tools/boot_test.py --seconds 6 --out shot.png
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "dnos.img"

QEMU_CANDIDATES = [
    r"V:\Program Files\qemu\qemu-system-i386.exe",
    r"C:\Program Files\qemu\qemu-system-i386.exe",
    "qemu-system-i386",
]


def find_qemu(cli):
    import shutil
    for cand in [cli, os.environ.get("QEMU")] + QEMU_CANDIDATES:
        if cand and (Path(cand).exists() or shutil.which(cand)):
            return cand
    sys.exit("ERROR: qemu-system-i386 not found (pass --qemu or set QEMU).")


class QMP:
    """Minimal QMP client over TCP."""

    def __init__(self, host, port, timeout=30):
        deadline = time.time() + timeout
        while True:
            try:
                self.sock = socket.create_connection((host, port), timeout=5)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.1)
        self.f = self.sock.makefile("rwb", buffering=0)
        self._read()                       # greeting
        self.execute("qmp_capabilities")

    def _read(self):
        while True:
            line = self.f.readline()
            if not line:
                raise EOFError("QMP closed")
            msg = json.loads(line)
            if "event" in msg and "return" not in msg:
                continue                   # skip async events
            return msg

    def execute(self, cmd, **args):
        payload = {"execute": cmd}
        if args:
            payload["arguments"] = args
        self.f.write((json.dumps(payload) + "\r\n").encode())
        return self._read()


def main():
    ap = argparse.ArgumentParser(description="DNOS QEMU boot smoke-test")
    ap.add_argument("--qemu")
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="run time before screenshot")
    ap.add_argument("--out", default=str(ROOT / "boot_screenshot.png"))
    ap.add_argument("--port", type=int, default=55556)
    args = ap.parse_args()

    qemu = find_qemu(args.qemu)
    intlog = ROOT / "qemu_int.log"
    out = Path(args.out).resolve()

    cmd = [
        qemu, "-drive", f"file={IMAGE},format=raw", "-m", "16M",
        "-display", "none",
        "-qmp", f"tcp:127.0.0.1:{args.port},server,nowait",
        "-d", "int,cpu_reset", "-D", str(intlog),
        "-no-reboot", "-no-shutdown",
    ]
    print("[boot_test] launching:", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    try:
        qmp = QMP("127.0.0.1", args.port)
        print(f"[boot_test] connected; running {args.seconds}s...")
        time.sleep(args.seconds)
        r = qmp.execute("screendump", filename=str(out), format="png")
        print("[boot_test] screendump:", r)
        qmp.execute("quit")
    finally:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Summarize the interrupt trace for triple-fault / reset detection.
    resets = faults = 0
    if intlog.exists():
        text = intlog.read_text(errors="ignore")
        resets = text.count("CPU Reset")
        faults = text.lower().count("triple")
    print(f"[boot_test] screenshot: {out} "
          f"({out.stat().st_size if out.exists() else 0} bytes)")
    print(f"[boot_test] cpu resets in trace: {resets}, triple-fault hits: {faults}")


if __name__ == "__main__":
    main()
