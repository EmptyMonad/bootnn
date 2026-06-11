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

DNOS boots from a hard-disk image — stage 1 → stage 2 (16-bit) → 32-bit protected mode, using INT 13h LBA reads (no CHS geometry limits). It loads 43,008 trained weights into memory and enters a loop: keyboard input → neural forward pass → screen output. On first boot, a demo sequence feeds synthetic keypresses through the network to prove the substrate is alive.

Type `box` and the network draws a rectangle. Type `line` and it draws a line. Press `p` for a pixel. The network learned these mappings during offline training — the assembly code contains no `if key == 'b' then draw_rect` logic. It's weights all the way down.

**Display:** the kernel uses a 32bpp VESA linear framebuffer when one is available and falls back to 320×200 VGA otherwise. (SeaBIOS exposes only a 24bpp VBE mode at 800×600, which the 32bpp-only drawing path declines — so under QEMU you currently get the clean VGA fallback.)

**Verified:** training reaches 100% accuracy in both float and Q8.8, with bit-exact Python↔assembly agreement; weights are deterministic (seeded). CI runs two gates on every push: a headless QEMU boot test (screenshot + triple-fault scan), and a *differential* interactive test that feeds real PS/2 keystrokes over QMP and asserts the command the kernel decodes (read from guest memory) equals the simulator's prediction for the identical input history — the substrate provably computes its law on metal, not just in Python.

## Architecture

```
Keyboard ──→ Input History ──→ Neural Network ──→ Decode ──→ VGA/VESA
  (IRQ1)      (32 events)    (256→128→64→32)    (argmax)   (framebuffer)
```

The kernel is three things: a boot sequence, a neural forward pass in Q8.8 fixed-point, and graphics primitives. Everything else — what keystrokes mean, what to draw, how to respond — is in the weights.

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
