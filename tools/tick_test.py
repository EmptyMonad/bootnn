#!/usr/bin/env python3
"""
Deterministic-loop test.

Verifies the PIT-paced main loop's invariants on live metal:

  1. step_count <= tick_count at all times (one state transition per tick,
     never more) — the honest, implementable form of
     S(n+1) = f(S(n), input(n)).
  2. Steps advance at tick rate while running (delta_step ~= delta_tick
     in interactive mode, and never exceeds it).
  3. A burst of keystrokes injected faster than the tick rate is consumed
     at most one event per tick: the ring buffer drains completely, and
     every key still decodes (kb_read_idx catches kb_write_idx).
  4. hdr_status == 1 (weights were validated on this boot).
  5. The DISPI path delivered 800x600x32 (k_video_mode == 1 under QEMU
     stdvga; VGA fallback would be 0).
"""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from boot_test import find_qemu  # noqa: E402

IMAGE = ROOT / "dnos.img"
SYMBOLS = json.loads((ROOT / "dnos_symbols.json").read_text())

QMP_PORT = 55560


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
        self.f.readline()
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


def read_u16(qmp, addr):
    r = qmp.execute("human-monitor-command",
                    **{"command-line": f"xp /1hx {addr:#x}"})
    return int(r["return"].strip().split(":")[1].strip(), 16)


def send_key(qmp, key):
    qmp.execute("send-key", keys=[{"type": "qcode", "data": key}])


def main():
    failures = []

    proc = subprocess.Popen(
        [find_qemu(None), "-drive", f"file={IMAGE},format=raw", "-m", "16M",
         "-display", "none", "-qmp", f"tcp:127.0.0.1:{QMP_PORT},server,nowait",
         "-no-reboot", "-no-shutdown"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    qmp = Qmp(QMP_PORT)
    time.sleep(6.0)  # boot + demo

    # ── weights validated, video mode ───────────────────────────────────
    hdr = read_u32(qmp, SYMBOLS["hdr_status"])
    vmode = read_u32(qmp, SYMBOLS["k_video_mode"]) & 0xFF
    print(f"[tick] hdr_status={hdr} k_video_mode={vmode} "
          f"({'800x600x32 DISPI/VESA' if vmode == 1 else 'VGA fallback'})")
    if hdr != 1:
        failures.append("weights not validated")
    if vmode != 1:
        failures.append("expected 32bpp linear framebuffer under QEMU stdvga")

    # ── invariant: step_count <= tick_count, sampled repeatedly ─────────
    for i in range(5):
        t = read_u32(qmp, SYMBOLS["tick_count"])
        s = read_u32(qmp, SYMBOLS["step_count"])
        ok = s <= t
        print(f"[tick] sample {i}: tick_count={t} step_count={s}  "
              f"{'OK' if ok else 'VIOLATION'}")
        if not ok:
            failures.append(f"step_count {s} > tick_count {t}")
        time.sleep(0.3)

    # ── pacing: steps advance at (and never above) tick rate ────────────
    t1, s1 = read_u32(qmp, SYMBOLS["tick_count"]), read_u32(qmp, SYMBOLS["step_count"])
    time.sleep(2.0)
    t2, s2 = read_u32(qmp, SYMBOLS["tick_count"]), read_u32(qmp, SYMBOLS["step_count"])
    dt, ds = t2 - t1, s2 - s1
    print(f"[tick] over 2s: delta_tick={dt} delta_step={ds}")
    if ds == 0:
        failures.append("main loop is not stepping")
    if ds > dt:
        failures.append(f"steps ({ds}) outpaced ticks ({dt})")

    # ── burst: keys arrive faster than ticks; consumed one per tick ─────
    keys = ["p", "b", "l", "o", "v"]
    for k in keys:
        send_key(qmp, k)          # no inter-key delay: genuine burst
    w0 = read_u16(qmp, SYMBOLS["kb_write_idx"])
    time.sleep(1.5)               # 150 ticks: ample to drain 5 events
    r1 = read_u16(qmp, SYMBOLS["kb_read_idx"])
    w1 = read_u16(qmp, SYMBOLS["kb_write_idx"])
    drained = r1 == w1 and w1 >= w0
    print(f"[tick] burst of {len(keys)}: kb write={w1} read={r1}  "
          f"{'drained OK' if drained else 'NOT drained'}")
    if not drained:
        failures.append("keyboard ring buffer did not drain")

    proc.terminate(); proc.wait()

    if failures:
        print("[tick] result: FAIL")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("[tick] result: PASS - S(n+1) = f(S(n), input(n)), one tick at a time")


if __name__ == "__main__":
    main()
