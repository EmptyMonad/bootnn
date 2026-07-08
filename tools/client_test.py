#!/usr/bin/env python3
"""
S1 client gate: record a session through the thin client (agent mode),
then replay its log and require bit-identical final state.

This is the user-facing form of the determinism claim: any session,
human or agent, is an event log, and the log IS the session — replay
reconstructs the machine exactly. Exercises dnos_client.py end to end
(law==metal per event, session logging, replay verification).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIENT = ROOT / "tools" / "dnos_client.py"
LOG = ROOT / "sessions" / "ci_session.jsonl"


def run(label, *extra):
    cmd = [sys.executable, str(CLIENT), *extra]
    print(f"[client_test] {label}: {' '.join(cmd[1:])}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    tail = "\n".join(r.stdout.strip().splitlines()[-3:])
    print("\n".join(f"  {ln}" for ln in tail.splitlines()))
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        print(f"[client_test] result: FAIL ({label})")
        sys.exit(1)
    return r.stdout


def main():
    run("record", "--send", "pblofc", "--log", str(LOG))
    out = run("replay", "--replay", str(LOG))
    if "PASS - identical state" not in out:
        print("[client_test] result: FAIL (replay did not verify)")
        sys.exit(1)
    print("[client_test] result: PASS - the log IS the session")


if __name__ == "__main__":
    main()
