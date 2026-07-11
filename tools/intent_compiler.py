#!/usr/bin/env python3
"""
Intent compiler: REFERENCE + PROVISION modes (ROADMAP item 6).

The user describes intent in an interview; the system compiles it into
configuration. The conversation is ephemeral - the OUTPUT is a
canonical, hash-chained config log: same answers -> byte-identical
log -> same config CRC, forever replayable and auditable.

Reference mode: the compiled config REFERENCES laws by stable
commitment - the canonical v5 blob's whole-file CRC and, if present,
the forest manifest CRC.

Provision mode: a structure-intent answer ("specialist") compiles into
a PROVISION EVENT - the create_leaf claim template in the structural
grammar (s3_forest.validate_structure): op, next leaf index, carved
scope, the base manifest's commitment, and the policy training tuple.
The two claimed CRCs are deliberately absent: they are training
OUTPUTS, so they belong to the fulfillment - a builder trains the
leaf and submits the completed claim to the ledger, where
verification is structural replay. The compiler stays cheap and
deterministic; the economy stays the judge.

The chain is ledger.Ledger - one hash-chain authority in the repo.

Usage:
  python tools/intent_compiler.py --answers answers.json --out config.jsonl
  python tools/intent_compiler.py --answers a.json --manifest forest_manifest.json --out config.jsonl
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
    "specialist": ["none", "undo-fill", "color"],
}

# Provision mode: which scope each specialist intent carves. The
# training tuple is POLICY - fixed here so the same answer always
# compiles to the same lawful template (the spike's canonical config).
PROVISION_SCOPES = {"undo-fill": ["f", "u"], "color": ["+"]}
PROVISION_POLICY = {"seed": 1337, "epochs": 4000, "lr": 0.02}

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


def compile_intent(answers, out_path, manifest_path=None):
    for q, a in answers.items():
        if q not in QUESTIONS or a not in QUESTIONS[q]:
            sys.exit(f"ERROR: not a compilable intent: {q}={a}")
    for q in QUESTIONS:
        if q not in answers:
            sys.exit(f"ERROR: unanswered question: {q}")
    provision = None
    if answers["specialist"] != "none":
        # Structure-intent needs a structure to extend: the base
        # manifest is what the template's commitment binds to.
        if manifest_path is None or not Path(manifest_path).is_file():
            sys.exit("ERROR: specialist provisioning requires a base "
                     "manifest to extend (--manifest)")
        base = json.loads(Path(manifest_path).read_text())
        provision = dict(PROVISION_POLICY,
                         op="create_leaf",
                         leaf=len(base["scopes"]),
                         scope=sorted(PROVISION_SCOPES[answers["specialist"]]),
                         base_manifest_crc=crc32(
                             canonical(base).encode()) & 0xFFFFFFFF)

    out = Path(out_path)
    if out.exists():
        out.unlink()
    log = Ledger(out)
    log.append("config", {"references": law_references(),
                          "interview": dict(sorted(answers.items()))})
    settings = {}
    for q in sorted(QUESTIONS):
        settings.update(COMPILE_RULES.get((q, answers[q]), {}))
    for key in sorted(settings):
        log.append("config", {"set": key, "value": settings[key]})
    if provision is not None:
        # The lawful intent to create structure. Fulfillment = train,
        # fill in the two claimed CRCs, submit-structure to a ledger.
        log.append("provision", provision)
    cfg_crc = crc32(out.read_bytes()) & 0xFFFFFFFF
    print(f"[intent] compiled {len(settings)} settings"
          + (", 1 provision" if provision else "")
          + f" -> {out} (config crc {cfg_crc:#010x})")
    return cfg_crc


def main():
    ap = argparse.ArgumentParser(description="DNOS intent compiler")
    ap.add_argument("--answers", help="JSON of interview answers")
    ap.add_argument("--out", default="config.jsonl")
    ap.add_argument("--manifest",
                    default=str(ROOT / "forest_manifest.json"),
                    help="base composite manifest for provision mode")
    ap.add_argument("--audit", help="verify a config log's hash chain")
    args = ap.parse_args()

    if args.audit:
        _, errors = Ledger(args.audit).fold()
        print(f"[intent] audit: {'PASS' if not errors else 'FAIL'}")
        sys.exit(1 if errors else 0)
    if not args.answers:
        sys.exit("ERROR: --answers or --audit required")
    compile_intent(json.loads(Path(args.answers).read_text()), args.out,
                   manifest_path=args.manifest)


if __name__ == "__main__":
    main()
