# DNOS — Deterministic Neural Operating System

A neural network that **is** the operating system.

```
STATE(t+1) = f(STATE(t), INPUT(t))
```

No scheduler. No processes. No drivers. The neural network directly controls hardware, interprets input, and produces output. System behavior emerges entirely from learned weights.

## Quick Start

Linux (Make):
```bash
sudo apt install nasm qemu-system-x86 python3-numpy
make && make run
```

Any platform (Windows / macOS / Linux), no `make`/`dd`/`truncate` required:
```bash
python tools/build.py --run      # train + assemble + patch + boot in QEMU
```

## What Happens

DNOS boots from a hard-disk image — stage 1 → stage 2 (16-bit) → 32-bit protected mode, using INT 13h LBA reads (no CHS geometry limits). Stage 2 lifts the 1.8 MB weight blob above the 1 MB line (INT 13h → 32 KB bounce buffer → unreal-mode copy to 0x200000), validates nothing yet — that's the kernel's job — and switches to protected mode. The kernel then verifies the blob's magic, layer sizes, weight count, and CRC32 over all 1,884,160 weight bytes before a single inference runs; corrupted weights get a red screen, "BAD WEIGHT CRC", and a hard halt. Only a verified law is allowed to govern. Then: keyboard input → neural forward pass (942,080 weights) → screen output, exactly one state transition per PIT tick. On first boot, a demo sequence feeds synthetic keypresses through the network to prove the substrate is alive.

Type `box` and the network draws a rectangle. Type `line` and it draws a line. Press `p` for a pixel. The network learned these mappings during offline training — the assembly code contains no `if key == 'b' then draw_rect` logic. It's weights all the way down.

**Display:** true 800×600×32. The fallback chain is VBE 32bpp → Bochs DISPI (ports 0x1CE/0x1CF, LFB base discovered from PCI BAR0) → 320×200 VGA. SeaBIOS only offers 24bpp VBE, so under QEMU the DISPI path delivers the full-resolution linear framebuffer. On-screen text (banner, mode indicator, status-bar telemetry) renders through a real 8×8 bitmap font on both video paths.

**Verified:** training reaches 100% accuracy in both float and Q8.8, with bit-exact Python↔assembly agreement; weights are deterministic (seeded) and CRC-stamped. Commands also hold under *held-out* random input histories (`tools/context_eval.py`, gated ≥95%; the network trains on resampled history contexts, so the mapping generalizes instead of memorizing). CI runs seven gates on every push: the context-generalization eval; a headless QEMU boot test (screenshot + triple-fault scan); a *differential* interactive test that feeds real PS/2 keystrokes over QMP and asserts the command the kernel decodes (read from guest memory) equals the simulator's prediction for the identical input history; a weight-integrity test proving a single corrupted weight byte is rejected at boot with the tick counter frozen; a deterministic-loop test asserting `step_count <= tick_count` live, including under keystroke bursts; and `cargo test` on both Rust determinism layers. The substrate provably computes its law on metal — and refuses to compute a law it can't verify.

## Architecture

```
Keyboard ──→ Input History ──→ Neural Network ──→ Decode ──→ VGA/VESA
  (IRQ1)      (64 events)   (512→1024→384→64)   (argmax)   (framebuffer)
```

The kernel is three things: a boot sequence, a neural forward pass in Q8.8 fixed-point (942,080 weights, ~1.8 MB, resident above the 1 MB line), and graphics primitives. The main loop is PIT-paced: it blocks on the timer and performs exactly one state transition per tick, consuming at most one input event per tick — `S(n+1) = f(S(n), input(n))` literally. The tick (20 Hz) is the measured budget for one full Tier 3 forward pass; determinism is rate-independent. Everything else — what keystrokes mean, what to draw, how to respond — is in the weights.

### Determinism Layers

For distributed DNOS, two abstraction layers guarantee that identical observations produce identical behavior across nodes:

**IAL** (Input Abstraction Layer) — quantizes continuous time into discrete epochs, buckets spatial coordinates into grid cells, and canonically sorts same-epoch events. Eliminates microsecond jitter, arrival order variance, and sensor noise.

**NDAL** (Non-Determinism Abstraction Layer) — wraps genuinely non-deterministic sources (RNG, clocks, network, hardware) in named oracles. Every oracle response is recorded in a hash-chained replay log. Same log → same execution → same state.

```
Physical World
       │
  ┌────┴────┐
  │   IAL   │  kills accidental non-determinism
  └────┬────┘
  ┌────┴────┐
  │  NDAL   │  contains essential non-determinism
  └────┬────┘
       │
  Deterministic Token Stream → Neural Substrate
```

## Project Structure

```
├── src/dnos.asm          Bootable kernel (32-bit PM, VESA/VGA, IRQ, neural core)
├── tools/train.py        Training (quantization-aware, 100% Q8.8 accuracy, seeded)
├── tools/build.py        Cross-platform build: assemble + patch weights + pad
├── tools/boot_test.py    Headless QEMU smoke test (screenshot + fault check)
├── tools/interactive_test.py  Differential test: QMP keystrokes, metal == law
├── Makefile              make && make run (Linux)
├── .github/workflows/    CI: train → build → boot test on every push
│
├── ial/                  Input Abstraction Layer (Rust)
│   ├── src/              Temporal/spatial quantizers, semantic encoder, canonicalizer
│   └── tests/            7 integration scenarios proving jitter absorption
│
├── ndal/                 Non-Determinism Abstraction Layer (Rust)
│   ├── src/              Oracles, replay log, Live/Replay/Verify modes
│   └── tests/            8 scenarios: determinism, divergence detection, snapshots
│
├── docs/                 Design documents
└── legacy/               Tier 1 originals (16-bit, 2,560 weights)
```

## Scaling

| Tier | Weights | Mode | Status |
|------|---------|------|--------|
| 1 | 2,560 | 16-bit real mode | Complete (in `legacy/`) |
| 2 | 43,008 | 32-bit protected mode | **Current — boots, 100% Q8.8 accuracy, CI-tested** |
| 3 | ~1M | Paging + disk swap | Planned |
| 4+ | 10M–10B | GPU / distributed | Research |

See **[ROADMAP.md](ROADMAP.md)** for the detailed, status-tracked plan. Long-term: memristor crossbar arrays where matrix multiplication happens as physics (Ohm's Law), not computation.

## Building & Testing

Linux (Make):
```bash
make              # Assemble + train + patch weights → dnos.img
make run          # Build and launch in QEMU
make train        # Retrain weights only
make validate     # Train + verify Python↔assembly math agreement
make clean        # Remove artifacts
```

Cross-platform (Python — no `make`/`dd`/`truncate`):
```bash
python tools/train.py            # train + validate (exits non-zero on divergence)
python tools/build.py            # assemble + patch weights + pad image
python tools/build.py --run      # ...and boot in QEMU
python tools/boot_test.py        # headless boot smoke test (screenshot + PASS/FAIL)
```

Every push runs the full pipeline (train → build → boot test) in **GitHub Actions**; the boot screenshot is uploaded as a build artifact.

Requires `nasm`, `python3` with `numpy`, and `qemu-system-i386` (`qemu-system-x86` on Debian/Ubuntu).

## License

Public domain.
