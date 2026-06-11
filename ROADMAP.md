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
- 📋 **True 32bpp video.** SeaBIOS offers only 24bpp VBE at 800×600, so the
  kernel falls back to VGA. Add a real 32bpp linear-framebuffer mode (Bochs
  DISPI) or native 24bpp drawing to restore high resolution.
- 📋 **Boot-time header validation.** Read layer sizes from the header and
  verify the CRC32 before inference — "don't mistake the map for the territory."
- 📋 **Deterministic main loop ("tick").** Replace the busy-wait delay with a
  fixed PIT-driven cycle so every inference is a clean state transition
  `S(n+1) = f(S(n), input)` — the honest, implementable form of
  self-reference.
- 📋 **Real font rendering.** `draw_text_simple` currently writes ASCII codes as
  pixel values; add a bitmap font so status/debug text is legible.

## Phase 3 — Material synthesis & self-reference 🔬

- 🔬 **Tier 3 (~1M weights).** Paging + on-disk weight swap so the network
  exceeds low memory. Requires a pager and a streaming forward pass.
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

## Near-term backlog (next concrete steps)

1. 📋 32bpp VESA via Bochs DISPI (restore 800×600) **or** 24bpp drawing path.
2. 📋 Boot-time header CRC + layer-size validation.
3. 📋 Deterministic PIT-paced main loop.
4. ✅ ~~Interactive boot test~~ — done as a *differential* test: every
   keystroke's decoded command is asserted equal to the simulator's
   prediction for the same history (`tools/interactive_test.py`).
5. 📋 Bitmap font for on-screen text.
6. 📋 Tier 3 design doc (paging + weight streaming).
7. 📋 **Generalization, not memorization.** With 43k weights and ~400
   examples the network memorizes; under deep unseen histories some keys
   still decode to the wrong command (the substrate faithfully computes a
   law that is itself wrong). Drawing-board item for emergent neuromorphics:
   train for context invariance explicitly (contrastive histories, dropout
   on history slots, or an architectural attention split between slot 0 and
   context), and quantify it with a held-out random-context eval set.
8. 📋 Demo/test color state: the boot demo can leave color == background
   (white on white), making correct draws invisible. Either pin the palette
   for the demo or have the demo end with a canonical clear + color reset.

## How correctness is enforced

| Gate | Mechanism |
|------|-----------|
| Network learns the task | `train.py` accuracy check |
| Python == assembly math | `simulate_assembly_forward`, non-zero exit on divergence |
| Reproducible weights | seeded RNG, stable CRC32 |
| Image builds | `tools/build.py` (assemble + patch + pad) |
| Boots without faulting | `tools/boot_test.py` (headless QEMU, screenshot + triple-fault scan) |
| All of the above, every push | `.github/workflows/ci.yml` |
