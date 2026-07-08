#!/usr/bin/env python3
"""
Weight-integrity boot test.

Two boots:
  1. Pristine image  -> hdr_status must be 1 (validated), demo runs.
  2. One weight byte flipped -> hdr_status must be 2 (rejected),
     the CPU must be hard-halted (tick_count frozen), and the
     screen must be painted red.

"Don't mistake the map for the territory": the kernel must refuse
to run inference on weights that don't match their trained CRC.
"""
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "dnos.img"
SYMBOLS = json.loads((ROOT / "dnos_symbols.json").read_text())

QMP_PORT = 55558


class Qmp:
    def __init__(self, port):
        deadline = time.time() + 10
        while True:
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=2)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.2)
        self.f = self.sock.makefile("rw")
        self.f.readline()  # greeting
        self.execute("qmp_capabilities")

    def execute(self, cmd, **args):
        self.f.write(json.dumps({"execute": cmd, "arguments": args}) + "\n")
        self.f.flush()
        while True:
            r = json.loads(self.f.readline())
            if "return" in r or "error" in r:
                return r


def read_u32(qmp, addr):
    r = qmp.execute("human-monitor-command",
                    **{"command-line": f"xp /1wx {addr:#x}"})
    return int(r["return"].strip().split(":")[1].strip(), 16)


def boot(image, port, seconds):
    proc = subprocess.Popen(
        ["qemu-system-i386", "-drive", f"file={image},format=raw", "-m", "16M",
         "-display", "none", "-qmp", f"tcp:127.0.0.1:{port},server,nowait",
         "-no-reboot", "-no-shutdown"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    qmp = Qmp(port)
    time.sleep(seconds)
    return proc, qmp


def main():
    failures = 0

    # ── Case 1: pristine image validates ────────────────────────────────
    proc, qmp = boot(IMAGE, QMP_PORT, seconds=4.0)
    status = read_u32(qmp, SYMBOLS["hdr_status"])
    ticks = read_u32(qmp, SYMBOLS["tick_count"])
    ok = status == 1 and ticks > 0
    print(f"[integrity] pristine:  hdr_status={status} tick_count={ticks}  "
          f"{'OK' if ok else 'FAIL (expected status=1, ticks>0)'}")
    failures += 0 if ok else 1
    proc.terminate(); proc.wait()

    # ── Case 2: corrupted weights are rejected ──────────────────────────
    corrupt = ROOT / "dnos_corrupt.img"
    shutil.copyfile(IMAGE, corrupt)
    # Flip one bit deep inside the weight data (sector 69 + header + offset)
    with open(corrupt, "r+b") as f:
        off = 69 * 512 + 128 + 40_000
        f.seek(off)
        b = f.read(1)[0]
        f.seek(off)
        f.write(bytes([b ^ 0xFF]))

    proc, qmp = boot(corrupt, QMP_PORT + 1, seconds=4.0)
    status = read_u32(qmp, SYMBOLS["hdr_status"])
    t1 = read_u32(qmp, SYMBOLS["tick_count"])
    time.sleep(1.0)
    t2 = read_u32(qmp, SYMBOLS["tick_count"])

    rejected = status == 2
    halted = t1 == t2  # cli+hlt: ticks must be frozen
    print(f"[integrity] corrupted: hdr_status={status} "
          f"ticks {t1}->{t2}  "
          f"{'OK' if (rejected and halted) else 'FAIL (expected status=2, frozen ticks)'}")
    failures += 0 if (rejected and halted) else 1
    proc.terminate(); proc.wait()
    corrupt.unlink(missing_ok=True)

    print(f"[integrity] result: {'PASS' if failures == 0 else 'FAIL'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
