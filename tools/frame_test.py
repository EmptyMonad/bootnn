#!/usr/bin/env python3
"""
Wire-frame differential gate: two independent implementations of the
S1 v1 frame - dnos_client.frame (Python, the wire-protocol AUTHORITY)
and dnos_ial::wire::frame (Rust, the canonical agent-side encoder) -
must agree byte-for-byte on all 256 events. The protocol layer gets
the same discipline as the law: agreement is tested, never assumed.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import dnos_client  # noqa: E402


def main():
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
