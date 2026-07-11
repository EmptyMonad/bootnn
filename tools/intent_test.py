#!/usr/bin/env python3
"""
Intent-compiler gate: the conversation is ephemeral, the configuration
is law. Same answers -> byte-identical config log; different answers
-> different; unanswered/invalid intent refused; tampered log fails
audit; the log references laws by stable commitment.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IC = ROOT / "tools" / "intent_compiler.py"


def run(*argv, expect_fail=False):
    r = subprocess.run([sys.executable, str(IC), *argv],
                       capture_output=True, text=True, timeout=120)
    if (r.returncode != 0) != expect_fail:
        print(r.stdout, r.stderr)
        print(f"[intent_test] FAIL (exit {r.returncode}: {argv})")
        sys.exit(1)
    return r.stdout


def main():
    tmp = Path(tempfile.mkdtemp())
    ans = tmp / "a.json"
    ans.write_text(json.dumps({"shell": "tiling", "priority":
                               "development", "input": "both"}))
    run("--answers", str(ans), "--out", str(tmp / "c1.jsonl"))
    run("--answers", str(ans), "--out", str(tmp / "c2.jsonl"))
    b1, b2 = (tmp / "c1.jsonl").read_bytes(), (tmp / "c2.jsonl").read_bytes()
    if b1 != b2:
        print("[intent_test] FAIL (same answers, different logs)")
        sys.exit(1)
    if b'"law.v5.crc"' not in b1:
        print("[intent_test] FAIL (no law reference in config)")
        sys.exit(1)

    ans.write_text(json.dumps({"shell": "win95", "priority":
                               "development", "input": "both"}))
    run("--answers", str(ans), "--out", str(tmp / "c3.jsonl"))
    if (tmp / "c3.jsonl").read_bytes() == b1:
        print("[intent_test] FAIL (different answers, same log)")
        sys.exit(1)

    ans.write_text(json.dumps({"shell": "vista"}))          # not an intent
    run("--answers", str(ans), "--out", str(tmp / "c4.jsonl"),
        expect_fail=True)

    run("--audit", str(tmp / "c1.jsonl"))
    lines = (tmp / "c1.jsonl").read_text().splitlines()
    e = json.loads(lines[1])
    e["data"]["value"] = "evil"
    lines[1] = json.dumps(e, sort_keys=True, separators=(",", ":"))
    (tmp / "bad.jsonl").write_text("\n".join(lines) + "\n")
    run("--audit", str(tmp / "bad.jsonl"), expect_fail=True)

    print("[intent_test] result: PASS - the conversation is ephemeral, "
          "the configuration is law")


if __name__ == "__main__":
    main()
