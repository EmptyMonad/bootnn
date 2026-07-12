#!/usr/bin/env python3
"""
Wire-frame differential gate: two independent implementations of the
S1 v1 frame - dnos_client.frame (Python, the wire-protocol AUTHORITY)
and dnos_ial::wire::frame (Rust, the canonical agent-side encoder) -
must agree byte-for-byte on all 256 events. The protocol layer gets
the same discipline as the law: agreement is tested, never assumed.

Also the constants-agreement check: FRAME_V1 and the boot-demo key
sequence each exist in multiple sources that cannot share a definition
across languages (Python tools, the ial and rv crates, the kernel
asm). Hand-synchronized copies drift; this gate pins them to the
authority textually, so a bump that misses a copy fails the push
instead of surfacing as a digest mismatch misdiagnosed as law
divergence.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import dnos_client  # noqa: E402
from interactive_test import DEMO_KEYS  # noqa: E402


def constants_agree():
    marker = f"0x{dnos_client.FRAME_V1:02X}"
    for path, pattern, what in [
        (ROOT / "ial" / "src" / "wire.rs",
         rf"FRAME_V1: u8 = {marker}", "ial FRAME_V1"),
        (ROOT / "rv" / "src" / "main.rs",
         rf"FRAME_V1: u8 = {marker}", "rv FRAME_V1"),
        (ROOT / "rv" / "src" / "main.rs",
         rf'DEMO_KEYS: &\[u8\] = b"{DEMO_KEYS}"', "rv DEMO_KEYS"),
        (ROOT / "src" / "dnos.asm",
         r"demo_keys:\s*db\s*"
         + r",\s*".join(f"'{c}'" for c in DEMO_KEYS), "asm demo_keys"),
    ]:
        if not re.search(pattern, path.read_text()):
            sys.exit(f"[frame_test] FAIL: {what} in {path.name} does "
                     f"not match the authority ({pattern!r})")
    print(f"[frame_test] constants: FRAME_V1 {marker} and demo "
          f"{DEMO_KEYS!r} agree across python/ial/rv/asm sources")


def main():
    constants_agree()
    r = subprocess.run(
        ["cargo", "run", "-q", "--manifest-path",
         str(ROOT / "ial" / "Cargo.toml"), "--example", "frames"],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit("[frame_test] FAIL: rust emitter did not run")
    rust = r.stdout.split()
    if len(rust) != 256:
        sys.exit(f"[frame_test] FAIL: expected 256 frames, "
                 f"got {len(rust)}")
    for e in range(256):
        py = dnos_client.frame(e).hex()
        if rust[e] != py:
            sys.exit(f"[frame_test] FAIL: event {e:#04x}: "
                     f"rust {rust[e]} != python {py}")
    print("[frame_test] result: PASS - 256/256 frames byte-identical; "
          "the Rust encoder speaks the authority's wire language")


if __name__ == "__main__":
    main()
