#!/usr/bin/env python3
"""
DNOS cross-platform build script.

Replaces the Makefile's Linux-only `dd`/`truncate` so the image can be built
on Windows (and anywhere Python + nasm exist). Steps:

  1. Assemble src/dnos.asm  -> dnos.img            (nasm -f bin)
  2. Ensure weights.bin exists (train if missing)
  3. Patch weights.bin into the image at the weight sector
  4. Pad the image to a 1.44 MB floppy

Tool discovery order for nasm / qemu:
  1. --nasm / --qemu CLI args
  2. NASM / QEMU environment variables
  3. PATH
  4. Known Windows install locations

Usage:
  python tools/build.py                 # build dnos.img
  python tools/build.py --run           # build, then boot in QEMU
  python tools/build.py --train         # force retrain weights first
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Layout constants — must match src/dnos.asm and Makefile.
SECTOR_SIZE   = 512
WEIGHT_SECTOR = 69            # 0-indexed block; weights begin at sector 70
FLOPPY_SIZE   = 1474560       # 1.44 MB

ROOT     = Path(__file__).resolve().parent.parent
ASM_SRC  = ROOT / "src" / "dnos.asm"
TRAIN    = ROOT / "tools" / "train.py"
IMAGE    = ROOT / "dnos.img"
WEIGHTS  = ROOT / "weights.bin"
LISTING  = ROOT / "dnos.lst"
SYMBOLS  = ROOT / "dnos_symbols.json"
ORG_BASE = 0x7C00             # whole image is org 0x7C00; phys = org + file offset

# Kernel variables exported to dnos_symbols.json for the boot/interactive
# tests to inspect via the QEMU monitor.
SYMBOL_NAMES = {
    "tick_count", "last_cmd", "last_confidence", "cursor_x", "cursor_y",
    "color_idx", "draw_size", "kb_read_idx", "kb_write_idx", "k_video_mode",
    "last_tick", "step_count", "hdr_status",
}

# Fallback install locations (this machine + common defaults).
NASM_CANDIDATES = [
    r"V:\Program Files\NASM\nasm.exe",
    r"C:\Program Files\NASM\nasm.exe",
]
QEMU_CANDIDATES = [
    r"V:\Program Files\qemu\qemu-system-i386.exe",
    r"C:\Program Files\qemu\qemu-system-i386.exe",
]


def find_tool(name, cli_value, env_var, candidates):
    """Locate an executable via CLI arg, env var, PATH, then known paths."""
    if cli_value:
        return cli_value
    env = os.environ.get(env_var)
    if env:
        return env
    on_path = shutil.which(name)
    if on_path:
        return on_path
    for cand in candidates:
        if Path(cand).exists():
            return cand
    sys.exit(f"ERROR: could not find '{name}'. Pass --{name} or set {env_var}.")


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def assemble(nasm):
    print(f"[DNOS] Assembling {ASM_SRC.relative_to(ROOT)} -> {IMAGE.name}")
    run([nasm, "-f", "bin", str(ASM_SRC), "-o", str(IMAGE), "-l", str(LISTING)])
    export_symbols()


def export_symbols():
    """Parse the NASM listing for known data labels and write their physical
    addresses to dnos_symbols.json so tests can inspect kernel state."""
    import json
    import re
    symbols = {}
    pattern = re.compile(r"^\s*\d+\s+([0-9A-F]{8})\s+\S+\s+(\w+):")
    for line in LISTING.read_text(errors="ignore").splitlines():
        m = pattern.match(line)
        if m and m.group(2) in SYMBOL_NAMES:
            symbols[m.group(2)] = ORG_BASE + int(m.group(1), 16)
    SYMBOLS.write_text(json.dumps(symbols, indent=2) + "\n")
    print(f"[DNOS] Exported {len(symbols)} symbols -> {SYMBOLS.name}")


def ensure_weights(python, force):
    if force or not WEIGHTS.exists():
        print("[DNOS] Training weights...")
        run([python, str(TRAIN), "--output", str(WEIGHTS)])
    else:
        print(f"[DNOS] Using existing {WEIGHTS.name}")


def patch_weights():
    """Write weights.bin into the image at the weight sector (replaces dd)."""
    offset = WEIGHT_SECTOR * SECTOR_SIZE
    data = WEIGHTS.read_bytes()
    print(f"[DNOS] Patching {len(data):,} bytes at sector {WEIGHT_SECTOR} "
          f"(offset 0x{offset:X})")
    with open(IMAGE, "r+b") as f:
        f.seek(offset)
        f.write(data)


def pad_image():
    """Pad/truncate the image to floppy size (replaces truncate)."""
    size = IMAGE.stat().st_size
    if size < FLOPPY_SIZE:
        with open(IMAGE, "r+b") as f:
            f.seek(FLOPPY_SIZE - 1)
            f.write(b"\x00")
    elif size > FLOPPY_SIZE:
        print(f"  WARNING: image is {size:,} bytes (> {FLOPPY_SIZE:,}); "
              f"truncating.")
        with open(IMAGE, "r+b") as f:
            f.truncate(FLOPPY_SIZE)
    print(f"[DNOS] Image padded to {FLOPPY_SIZE:,} bytes")


def boot(qemu, extra):
    print("[DNOS] Booting in QEMU...")
    # Boot as a hard disk so the bootloader can use INT 13h LBA extensions.
    run([qemu, "-drive", f"file={IMAGE},format=raw", "-m", "16M", *extra])


def main():
    ap = argparse.ArgumentParser(description="DNOS cross-platform builder")
    ap.add_argument("--nasm", help="path to nasm")
    ap.add_argument("--qemu", help="path to qemu-system-i386")
    ap.add_argument("--python", default=sys.executable, help="python for training")
    ap.add_argument("--train", action="store_true", help="force retrain weights")
    ap.add_argument("--run", action="store_true", help="boot image in QEMU after build")
    ap.add_argument("--qemu-arg", action="append", default=[],
                    help="extra QEMU argument (repeatable)")
    args = ap.parse_args()

    nasm = find_tool("nasm", args.nasm, "NASM", NASM_CANDIDATES)

    assemble(nasm)
    ensure_weights(args.python, args.train)
    patch_weights()
    pad_image()
    print(f"[DNOS] Build complete: {IMAGE}")

    if args.run:
        qemu = find_tool("qemu", args.qemu, "QEMU", QEMU_CANDIDATES)
        boot(qemu, args.qemu_arg)


if __name__ == "__main__":
    main()
