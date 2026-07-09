# DNOS Roadmap

> `STATE(t+1) = f(STATE(t), INPUT(t))`
>
> The goal: a system whose behavior *is* its weights — computed on bare metal,
> not managed by an OS. This roadmap tracks the path from a working prototype
> toward that end. Items are grounded in verifiable engineering; speculative
> long-range goals are labeled **Vision** so they are never mistaken for
> committed deliverables.

**Legend:** ✅ done · 🚧 in progress · 📋 planned · 🔬 research/vision

---

## Phase 1 — Canonical weights & the abstraction gap ✅

Make the trained "law" an accurate reflection of what bare metal computes, and
make it reproducible.

- ✅ **Network converges.** Quantization-aware training (Q8.8 STE, ReLU+clamp,
  softmax classification loss). Float and Q8.8 accuracy both **100%** (was
  16% / 8%).
- ✅ **Abstraction gap closed.** `simulate_assembly_forward` reproduces the
  kernel's int16 math bit-for-bit; **0% divergence** between Python and
  assembly. Enforced: `train.py` exits non-zero on any mismatch.
- ✅ **Canonical (deterministic) weights.** Seeded training (`--seed`, default
  1337) → identical weight blob every build (CRC32 stable).
- ✅ **Weight format.** 128-byte header (magic `DN`, version, tier, layer sizes,
  activation types, weight count, CRC32) + 43,008 Q8.8 int16 weights.

## Phase 2 — Weight-as-law / bare-silicon execution 🚧

Move from "executing code" to "being the rule," on real hardware.

- ✅ **Boots on bare metal.** Two-stage boot → 32-bit protected mode. Reaches
  the neural demo and renders. (Previously never booted.)
- ✅ **LBA disk I/O.** INT 13h AH=42h reads replace CHS reads that exceeded
  floppy geometry; boots as a hard disk.
- ✅ **Coherent memory map.** Fixed load-address≠link-address (kernel at
  0x8600), stage2/kernel overlap, weight/scratch overlap (`ACTIV_BASE`), and a
  short weight load that left the output layer unloaded (now all 170 sectors).
- ✅ **Metadata handling.** The 128-byte header is skipped on load; weights map
  exactly to `WEIGHT_BASE`. (Header *validation* — CRC/layer-size check at boot
  — is a planned hardening step, see below.)
- ✅ **CI enforcement.** Train → build → headless boot test on every push.
- ✅ **On-metal inference actually computes the law.** The forward pass used
  one-operand `imul`, which clobbers EDX — the weight pointer — on the first
  multiply-accumulate of every neuron, so the kernel computed garbage (every
  output saturated, argmax always 0) while the simulator validated the
  *intended* math. Fixed with two-operand `imul`. Verified live: kernel
  `last_cmd` (read from guest memory over QMP) now equals
  `simulate_assembly_forward` for the identical input history, key by key.
- ✅ **Interactive differential test (was backlog #4).**
  `tools/interactive_test.py` boots headless, feeds real PS/2 keystrokes over
  QMP, reads the decoded command from guest memory (`dnos_symbols.json` is
  exported at build time from the NASM listing), and asserts metal == law for
  every key, plus that draw commands reach the framebuffer. Runs in CI.
- ✅ **Cursor/decode fixes.** Cursor initializes to screen centre for the
  active mode (it used to clamp into the status bar, which repaints every 16
  ticks and erased everything drawn); the delta decode subtracts the sigmoid
  midpoint (16384) so a trained zero delta no longer drifts the cursor
  +16,+16 per inference; move_down/move_right now clamp; rect's right edge
  uses width, not height.
- ✅ **Context-robust single-key commands.** Single-key mappings are trained
  under seeded random histories spanning the full 32-event window; the
  network previously misclassified e.g. 'b' → move_right after the boot demo.
- ✅ **True 32bpp video.** Bochs DISPI (QEMU stdvga / Bochs / VirtualBox)
  programs 800×600×32 with a linear framebuffer directly via ports
  0x1CE/0x1CF; the LFB physical base is discovered from PCI BAR0 (config
  mechanism #1). Fallback chain is VBE 32bpp → DISPI → VGA 13h. Under QEMU
  the kernel now runs at full resolution (`k_video_mode == 1`, asserted by
  `tools/tick_test.py`).
- ✅ **Boot-time header validation.** `validate_weights` checks magic, all
  four layer sizes, the weight count, and the CRC32 of all 86,016 weight
  bytes (nibble-table CRC-32, zlib polynomial) before any inference. On
  mismatch: `hdr_status = 2`, screen painted red with "BAD WEIGHT CRC",
  hard halt. `tools/integrity_test.py` proves both directions in CI —
  pristine validates, a single flipped weight byte is rejected with the
  tick counter frozen. The map is verified before it is trusted as
  territory.
- ✅ **Deterministic main loop ("tick").** The main loop blocks on the PIT
  and performs exactly one state transition per tick, consuming at most one
  input event per tick: `S(n+1) = f(S(n), input(n))` literally, not
  approximately. Invariant `step_count <= tick_count` is exported and
  asserted live by `tools/tick_test.py`, including under keystroke bursts
  faster than the tick rate (ring buffer drains one event per tick).
- ✅ **Real font rendering.** 8×8 bitmap font (0x20–0x5A), opaque glyph
  cells, VGA and 32bpp paths. Banner, demo/interactive indicators, the
  weight-rejection message, and a live status-bar telemetry line
  (`C:cmd X:x Y:y T:tick` in hex) are all legible on screen.

## Phase 3 — Material synthesis & self-reference 🔬

- ✅ **Tier 3 (942,080 weights).** Implemented per `docs/TIER3_DESIGN.md`
  Strategy A: the full 1.8 MB blob is lifted to 0x200000 at boot
  (INT 13h → bounce buffer → unreal-mode copy) and inference runs from
  high memory with no disk in the loop — paging was explicitly rejected
  because cache-dependent inference latency would bend tick determinism.
  Topology 512→1024→384→64, 64-event context window, PIT at 20 Hz
  (the measured single-transition budget). Every CI gate extended, none
  removed: 386/386 with 0% Q8.8 divergence, ≥95% held-out context
  generalization, boot, integrity (CRC over 1.8 MB, negative-tested),
  tick invariant, and law==metal differential — all passing.
- 🚧 **Swarm (distributed phase).** Designed in `docs/SWARM_DESIGN.md`:
  ✅ **S0 deterministic replication** — `tools/swarm_test.py`, in CI:
  3 nodes fed one event log agree on state vector and cropped-framebuffer
  CRC at every checkpoint, each equal to the simulator's prediction;
  a one-event fork on a single node is detected and localized while the
  others stay in agreement. Node age (tick_count) is wall-clock-relative
  and excluded: nodes are equal in state, not in age.
  🚧 **S1 input surface** — v0 landed, in CI: COM1 is a polled event
  producer feeding the same ring as the keyboard (scancode→ASCII at the
  edge; UART probe-guarded so absent hardware can't flood the ring).
  `tools/serial_test.py` proves a PS/2-driven node and a wire-driven
  node fed the same semantic log occupy identical states — humans and
  agents are peers above one deterministic interface, executable form.
  ✅ **S1 thin client** — `tools/dnos_client.py`: interactive terminal
  for humans, `--send`/`--stdin` for agents, all over the same COM1
  wire; every session is an event log with a recorded final state, and
  `--replay` re-executes the log and verifies bit-identical state
  (`tools/client_test.py`, in CI: "the log IS the session").
  Remaining: versioned 8-byte IAL token framing — **deliberately
  blocked on TIER4_DESIGN OQ1** (the frame carries whatever the Tier 4
  encoder consumes; do not design it twice). Then: S2 a contribution
  economy where verification is deterministic replay and the wallet is a
  region of replicated state, S3 specialization + routing — the emergent
  architecture, gated on 2 specialists + router beating an equal-weight
  generalist. Scale *out*, not up.
- 🔬 **Recursive boot / prehistory.** Formalize training ("prehistory") and boot
  ("birth") as one pipeline; snapshot/replay total state across cycles. The
  Rust IAL/NDAL layers already provide deterministic replay primitives to build
  on.
- 🔬 **Memristive compute (Vision).** Analog ReRAM/PCM crossbars where
  "resistance *is* the weight" and matmul happens as Ohm's Law. Hardware-
  acquisition project; a north star, not a near-term task.

## Phase 4 — Cosmological scaling 🔬 (Vision)

Long-range, exploratory. Kept as motivation, not as tickets:

- 🔬 Internal "tick rate" as a stable physical reference for system time.
- 🔬 Perspective/geometry emerging from informational ratios.
- 🔬 Searching for emergent constants in the recursive structure.

These have no pass/fail test today; the engineering proxy for "tick rate" lives
in Phase 2 (deterministic loop). The rest stays vision until it can be made
falsifiable.

---

## Work queue (self-contained: "continue the DNOS roadmap" resumes here)

Each item is specified well enough to implement without this
conversation. Order is deliberate.

1. 🚧 **Tier 4 Part A — ternary weights** (`docs/TIER4_DESIGN.md`,
   Part A + amendments): **law side landed** — `train.py --ternary`
   (STE QAT, frozen-after-warmup shifts, held-out checkpoint
   selection quarantined from Q8.8, width as an instance parameter),
   packed 2-bit format v4a, format-aware simulator reading topology
   from the header, `tools/ternary_format_test.py` in CI (bit-exact
   file-vs-law, reserved code 10 refused). The campaign's chief find:
   ternary exposed a training-data defect (histories capped at 31 of
   64 events) whose fix **upgraded the canonical Q8.8 law** to 99.9%
   held-out generalization (CRC 0xE07DA759, defaults 6000/25, full
   live gauntlet re-verified). Remaining: a ternary config that
   clears 386/386 + ≥95% on the honest task (width/epoch search in
   progress), then the kernel packed-walk inner loop (commit 2 —
   add/sub/skip, no `imul` in inference, boot-time v4a validation,
   integrity negative tests on metal).
2. 📋 **Tier 4 Part B — diagonal SSM core** (TIER4_DESIGN, Part B):
   resolve OQ1 (event unit) FIRST — it fixes the S1 v1 token frame;
   then `h` (512×int16, `h_base` exported, in the swarm digest),
   decay spectrum `λ = 1-2^-k`, delete the history window, stateful
   simulator, streaming generalization eval.
3. 📋 **S1 v1 — versioned IAL token framing** (unblocked by item 2):
   frame = version byte + the Tier 4 event features; kernel framer on
   COM1; `dnos_client.py` and the Rust IAL emit it; serial_test
   extended to frame level.
4. 🚧 **S2 — contribution log + mint prototype**: core **landed**
   (`tools/ledger.py` + `tools/ledger_test.py`, in CI) — hash-chained
   JSONL log, claims are training tuples, verification is replay
   (rerun `train.py`, compare CRC32), issuance is a pure function of
   the log (no mint event exists to forge), transfers append-validated
   and audit-re-derived, tampering cascades to audit failure.
   Remaining: dataset-delta claims (train.py must accept a data
   contribution, not just hyperparams), gauntlet gating before a
   verified blob becomes law, ML-DSA signatures on entries, SPHINCS+
   identities, hardware-profile reward class (portability mints).
5. 📋 **S3 spike — two specialists + a router** (SWARM_DESIGN, S3):
   train two blobs on disjoint command subsets, host-side router,
   measure against one equal-weight generalist. Falsifies (or funds)
   the emergent-architecture claim at N=2.
6. 🔬 **Intent compiler prototype** (host-side, after S1 v1): NL
   interview → canonical config event log → deterministic client/shell
   configuration; the conversation is ephemeral, the log is law.

## Completed backlog (Tier 2/3 era)

1. ✅ ~~32bpp VESA via Bochs DISPI~~ — 800×600×32 restored under QEMU.
2. ✅ ~~Boot-time header CRC + layer-size validation~~ — with a negative
   test in CI (corrupted weights must be rejected).
3. ✅ ~~Deterministic PIT-paced main loop~~ — invariant asserted on metal.
4. ✅ ~~Interactive boot test~~ — done as a *differential* test: every
   keystroke's decoded command is asserted equal to the simulator's
   prediction for the same history (`tools/interactive_test.py`).
5. ✅ ~~Bitmap font for on-screen text~~ — 8×8, both video paths.
6. ✅ ~~Tier 3 design doc~~ — `docs/TIER3_DESIGN.md`, implemented as designed.
7. ✅ ~~Generalization, not memorization~~ — measured and solved.
   `tools/context_eval.py` scores single-key commands under held-out random
   histories (seeded, assembly-exact simulator): the frozen-augmentation
   model scored **53.9%** (e.g. pixel→hline ×30 confusions). Training now
   *resamples* the random-context examples every 50 epochs, so the network
   learns the history distribution instead of one sample: **100.0%**
   (1050/1050) on the same held-out eval, with the original suite still at
   100% and zero Q8.8 divergence. CI gates at ≥95%. The differential
   interactive test confirms on metal: 12/12 keys law==metal, 9/9 draw
   commands visible.
8. ✅ ~~Demo/test color state~~ — the pen color is pinned to white at the
   end of the demo sequence; network-driven color drift during the demo can
   no longer leave color == background for the interactive session.

## How correctness is enforced

| Gate | Mechanism |
|------|-----------|
| Network learns the task | `train.py` accuracy check |
| Python == assembly math | `simulate_assembly_forward`, non-zero exit on divergence |
| Reproducible weights | seeded RNG, stable CRC32 |
| Image builds | `tools/build.py` (assemble + patch + pad) |
| Boots without faulting | `tools/boot_test.py` (headless QEMU, screenshot + triple-fault scan) |
| Corrupted weights are rejected | `tools/integrity_test.py` (CRC negative test, frozen-tick halt) |
| One state transition per tick | `tools/tick_test.py` (`step_count <= tick_count`, burst drain) |
| Determinism layers hold their contracts | `cargo test` on `ial/` and `ndal/` |
| All of the above, every push | `.github/workflows/ci.yml` |
