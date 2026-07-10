#!/usr/bin/env python3
"""
DNOS interactive differential test: law vs. metal.

Boots dnos.img headless, replays the boot demo in the Python assembly
simulator to reconstruct the network's input history, then feeds real
keystrokes through QEMU's PS/2 keyboard and asserts, for every key, that
the command the kernel decoded (read from guest memory via QMP) equals
the command the simulator predicts for the identical history:

    STATE(t+1) = f(STATE(t), INPUT(t))   must hold on metal.

A secondary gate requires that at least one drawing command visibly
changed the framebuffer, proving the decode → draw path.

Requires dnos_symbols.json (written by tools/build.py).

Usage:
  python tools/interactive_test.py
  python tools/interactive_test.py --keys pblofcvh
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from boot_test import QMP, find_qemu  # noqa: E402
from train import CMD_NAMES, SsmMachine  # noqa: E402

IMAGE = ROOT / "dnos.img"
WEIGHTS = ROOT / "weights_ssm.bin"
SYMBOLS = ROOT / "dnos_symbols.json"
SHOTS = ROOT / "interactive_shots"

DEMO_KEYS = "pboxline"          # must match demo_keys in src/dnos.asm
DRAW_CMDS = {"pixel", "hline", "vline", "rect", "filled_rect", "circle",
             "fill", "clear"}


def read_u32(qmp, addr):
    r = qmp.execute("human-monitor-command",
                    **{"command-line": f"xp /1wx {addr:#x}"})
    return int(r["return"].strip().split(":")[1].strip(), 16)


def read_ppm(path):
    data = Path(path).read_bytes()
    _, dims, _, px = data.split(b"\n", 3)
    return px


def screendump(qmp, name):
    SHOTS.mkdir(exist_ok=True)
    path = SHOTS / f"{name}.ppm"
    qmp.execute("screendump", filename=str(path))
    return read_ppm(path)


def diff_pixels(a, b):
    n = min(len(a), len(b))
    return sum(1 for i in range(0, n, 3) if a[i:i + 3] != b[i:i + 3])


def main():
    ap = argparse.ArgumentParser(description="DNOS law-vs-metal test")
    ap.add_argument("--qemu")
    ap.add_argument("--port", type=int, default=55557)
    ap.add_argument("--keys", default="pblofc",
                    help="keys to send after the boot demo")
    ap.add_argument("--boot-seconds", type=float, default=6.0)
    args = ap.parse_args()

    if not SYMBOLS.is_file():
        sys.exit("ERROR: dnos_symbols.json missing - run tools/build.py first")
    sym = json.loads(SYMBOLS.read_text())

    qemu = find_qemu(args.qemu)
    cmd = [
        qemu, "-drive", f"file={IMAGE},format=raw", "-m", "16M",
        "-display", "none",
        "-qmp", f"tcp:127.0.0.1:{args.port},server,nowait",
        "-no-reboot", "-no-shutdown",
    ]
    print("[interactive] launching:", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    mismatches = []
    draw_diffs = []
    try:
        qmp = QMP("127.0.0.1", args.port)
        print(f"[interactive] waiting {args.boot_seconds}s for boot demo...")
        time.sleep(args.boot_seconds)

        # The law is now stateful (v5 recurrent): mirror the metal by
        # stepping SsmMachine through the same boot demo, then key by key.
        machine = SsmMachine(str(WEIGHTS))
        machine.run([ord(c) for c in DEMO_KEYS])
        prev_shot = screendump(qmp, "baseline")

        for key in args.keys:
            out = machine.step(ord(key))
            expected = int(np.argmax(out[:20]))

            qmp.execute("send-key",
                        keys=[{"type": "qcode", "data": key}])
            time.sleep(0.8)

            actual = read_u32(qmp, sym["last_cmd"])
            shot = screendump(qmp, f"after_{key}")
            changed = diff_pixels(prev_shot, shot)
            prev_shot = shot

            exp_name = CMD_NAMES[expected]
            act_name = (CMD_NAMES[actual]
                        if actual < len(CMD_NAMES) else f"cmd{actual}")
            ok = actual == expected
            print(f"[interactive] '{key}': law={exp_name:12s} "
                  f"metal={act_name:12s} fb_delta={changed:6d} px  "
                  f"{'OK' if ok else 'MISMATCH'}")
            if not ok:
                mismatches.append(key)
            if exp_name in DRAW_CMDS:
                draw_diffs.append(changed)

        try:
            qmp.execute("quit", expect_reply=False)
        except OSError:
            pass
    finally:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    drew = any(d > 0 for d in draw_diffs)
    print(f"[interactive] law-vs-metal mismatches: {len(mismatches)} "
          f"{mismatches or ''}")
    print(f"[interactive] draw commands issued: {len(draw_diffs)}, "
          f"visible: {sum(1 for d in draw_diffs if d > 0)}")
    if mismatches:
        print("[interactive] result: FAIL (metal diverged from law)")
        sys.exit(1)
    if draw_diffs and not drew:
        print("[interactive] result: FAIL (no draw command changed the screen)")
        sys.exit(1)
    print("[interactive] result: PASS - substrate matches its law end-to-end")
    sys.exit(0)


if __name__ == "__main__":
    main()
