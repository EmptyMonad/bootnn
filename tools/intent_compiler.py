#!/usr/bin/env python3
"""
Intent compiler, REFERENCE MODE (ROADMAP item 6; SWARM_DESIGN S1).

The user describes intent in an interview; the system compiles it into
configuration. The conversation is ephemeral - the OUTPUT is a
canonical, hash-chained config log: same answers -> byte-identical
log -> same config CRC, forever replayable and auditable.

Reference mode (per the S3 law-shape verdict): the compiled config
REFERENCES laws by stable commitment - the canonical v5 blob's
whole-file CRC and, if present, the forest manifest CRC. It does not
PROVISION structure (leaf creation waits on the structural claim
grammar).

The chain is ledger.Ledger - one hash-chain authority in the repo.

Usage:
  python tools/intent_compiler.py --answers answers.json --out config.jsonl
  python tools/intent_compiler.py --audit config.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from zlib import crc32

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from ledger import Ledger, canonical  # noqa: E402

# The interview: fixed questions, enumerated answers. Free text is the
# host conversation's job; by the time intent reaches the compiler it
# is one of these values or the compilation refuses.
QUESTIONS = {
    "shell": ["win95", "macos", "tiling", "conversational"],
    "priority": ["gaming", "development", "minimalism", "accessibility"],
    "input": ["keyboard", "agent", "both"],
}

# intent -> configuration, deterministically. Order fixed by key sort.
COMPILE_RULES = {
    ("shell", "win95"): {"ui.theme": "win95", "ui.taskbar": "bottom"},
    ("shell", "macos"): {"ui.theme": "macos", "ui.dock": "bottom"},
    ("shell", "tiling"): {"ui.theme": "tiling", "ui.gaps": "2"},
    ("shell", "conversational"): {"ui.theme": "conversational"},
    ("priority", "gaming"): {"host.compat": "isolated", "tick.hz": "20"},
    ("priority", "development"): {"client.log": "verbose", "tick.hz": "20"},
    ("priority", "minimalism"): {"ui.chrome": "none", "tick.hz": "20"},
    ("priority", "accessibility"): {"ui.scale": "1.5", "tick.hz": "20"},
    ("input", "keyboard"): {"wire.frame": "v1", "producers": "ps2"},
    ("input", "agent"): {"wire.frame": "v1", "producers": "com1"},
    ("input", "both"): {"wire.frame": "v1", "producers": "ps2+com1"},
}


def law_references():
    """Stable commitments the config is compiled AGAINST."""
    refs = {}
    blob = ROOT / "weights_ssm.bin"
    if blob.is_file():
        refs["law.v5.crc"] = f"{crc32(blob.read_bytes()) & 0xFFFFFFFF:#010x}"
    manifest = ROOT / "forest_manifest.json"
    if manifest.is_file():
        m = json.loads(manifest.read_text())
        blobm = json.dumps(m, sort_keys=True, separators=(",", ":"))
        refs["law.forest.crc"] = f"{crc32(blobm.encode()) & 0xFFFFFFFF:#010x}"
    return refs


def compile_intent(answers, out_path):
    for q, a in answers.items():
        if q not in QUESTIONS or a not in QUESTIONS[q]:
            sys.exit(f"ERROR: not a compilable intent: {q}={a}")
    for q in QUESTIONS:
        if q not in answers:
            sys.exit(f"ERROR: unanswered question: {q}")

    out = Path(out_path)
    if out.exists():
        out.unlink()
    log = Ledger(out)
    log.append("config", {"references": law_references(),
                          "interview": dict(sorted(answers.items()))})
    settings = {}
    for q in sorted(QUESTIONS):
        settings.update(COMPILE_RULES[(q, answers[q])])
    for key in sorted(settings):
        log.append("config", {"set": key, "value": settings[key]})
    cfg_crc = crc32(out.read_bytes()) & 0xFFFFFFFF
    print(f"[intent] compiled {len(settings)} settings -> {out} "
          f"(config crc {cfg_crc:#010x})")
    return cfg_crc


def main():
    ap = argparse.ArgumentParser(description="DNOS intent compiler")
    ap.add_argument("--answers", help="JSON of interview answers")
    ap.add_argument("--out", default="config.jsonl")
    ap.add_argument("--audit", help="verify a config log's hash chain")
    args = ap.parse_args()

    if args.audit:
        _, errors = Ledger(args.audit).fold()
        print(f"[intent] audit: {'PASS' if not errors else 'FAIL'}")
        sys.exit(1 if errors else 0)
    if not args.answers:
        sys.exit("ERROR: --answers or --audit required")
    compile_intent(json.loads(Path(args.answers).read_text()), args.out)


if __name__ == "__main__":
    main()
