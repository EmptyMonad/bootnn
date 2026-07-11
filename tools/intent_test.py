#!/usr/bin/env python3
"""
Intent-compiler gate: the conversation is ephemeral, the configuration
is law. Same answers -> byte-identical config log; different answers
-> different; unanswered/invalid intent refused; tampered log fails
audit; the log references laws by stable commitment.

Provision mode: a specialist answer compiles into a provision event
that IS a create_leaf claim template - proven by handing the compiled
template (plus the inline base manifest and dummy claimed CRCs) to
s3_forest.validate_structure, which must accept it. Provisioning with
no base manifest to extend is refused; "none" compiles no provision.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IC = ROOT / "tools" / "intent_compiler.py"
sys.path.insert(0, str(ROOT / "tools"))


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
    base_answers = {"shell": "tiling", "priority": "development",
                    "input": "both", "specialist": "none"}
    ans.write_text(json.dumps(base_answers))
    run("--answers", str(ans), "--out", str(tmp / "c1.jsonl"))
    run("--answers", str(ans), "--out", str(tmp / "c2.jsonl"))
    b1, b2 = (tmp / "c1.jsonl").read_bytes(), (tmp / "c2.jsonl").read_bytes()
    if b1 != b2:
        print("[intent_test] FAIL (same answers, different logs)")
        sys.exit(1)
    if b'"law.v5.crc"' not in b1:
        print("[intent_test] FAIL (no law reference in config)")
        sys.exit(1)
    if b'"provision"' in b1:
        print("[intent_test] FAIL (specialist=none compiled a provision)")
        sys.exit(1)

    ans.write_text(json.dumps(dict(base_answers, shell="win95")))
    run("--answers", str(ans), "--out", str(tmp / "c3.jsonl"))
    if (tmp / "c3.jsonl").read_bytes() == b1:
        print("[intent_test] FAIL (different answers, same log)")
        sys.exit(1)

    ans.write_text(json.dumps({"shell": "vista"}))          # not an intent
    run("--answers", str(ans), "--out", str(tmp / "c4.jsonl"),
        expect_fail=True)

    # ── provision mode ──────────────────────────────────────────────
    import s3_forest
    table = s3_forest.base_table()
    fixture = {"router_version": table["version"],
               "scopes": table["scopes"],
               "fallback_mod": table["fallback_mod"],
               "leaves": [{"leaf": 0, "crc": 1}, {"leaf": 1, "crc": 2}]}
    mf = tmp / "manifest.json"
    mf.write_text(json.dumps(fixture, sort_keys=True,
                             separators=(",", ":")))
    ans.write_text(json.dumps(dict(base_answers, specialist="undo-fill")))
    run("--answers", str(ans), "--manifest", str(mf),
        "--out", str(tmp / "c5.jsonl"))
    run("--answers", str(ans), "--manifest", str(mf),
        "--out", str(tmp / "c6.jsonl"))
    if (tmp / "c5.jsonl").read_bytes() != (tmp / "c6.jsonl").read_bytes():
        print("[intent_test] FAIL (same provision intent, different logs)")
        sys.exit(1)
    prov = next((json.loads(ln)["data"] for ln in
                 (tmp / "c5.jsonl").read_text().splitlines()
                 if json.loads(ln)["type"] == "provision"), None)
    if prov is None:
        print("[intent_test] FAIL (specialist intent compiled no provision)")
        sys.exit(1)
    # The compiled template must SPEAK THE STRUCTURAL GRAMMAR: complete
    # it with the inline manifest and placeholder outputs, and the
    # claim validator must accept it.
    claim = dict(prov, account="builder", base_manifest=fixture,
                 claimed_crc=0, claimed_manifest_crc=0)
    reason = s3_forest.validate_structure(claim)
    if reason is not None:
        print(f"[intent_test] FAIL (provision template refused by the "
              f"grammar: {reason})")
        sys.exit(1)
    if prov["leaf"] != 2 or prov["scope"] != ["f", "u"]:
        print(f"[intent_test] FAIL (wrong template: {prov})")
        sys.exit(1)
    # Structure-intent with nothing to extend is refused.
    run("--answers", str(ans), "--manifest", str(tmp / "absent.json"),
        "--out", str(tmp / "c7.jsonl"), expect_fail=True)

    run("--audit", str(tmp / "c1.jsonl"))
    lines = (tmp / "c1.jsonl").read_text().splitlines()
    e = json.loads(lines[1])
    e["data"]["value"] = "evil"
    lines[1] = json.dumps(e, sort_keys=True, separators=(",", ":"))
    (tmp / "bad.jsonl").write_text("\n".join(lines) + "\n")
    run("--audit", str(tmp / "bad.jsonl"), expect_fail=True)
    run("--audit", str(tmp / "c5.jsonl"))   # provision logs audit too

    print("[intent_test] result: PASS - the conversation is ephemeral, "
          "the configuration is law, and structure-intent compiles to "
          "a lawful claim template")


if __name__ == "__main__":
    main()
