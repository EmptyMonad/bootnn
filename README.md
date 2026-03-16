# DNOS — Deterministic Neural Operating System

A neural network that **is** the operating system.

```
STATE(t+1) = f(STATE(t), INPUT(t))
```

No scheduler. No processes. No drivers. The neural network directly controls hardware, interprets input, and produces output. System behavior emerges entirely from learned weights.

## Quick Start

```bash
sudo apt install nasm qemu-system-x86 python3-numpy
make && make run
```

## What Happens

DNOS boots from a floppy image into 32-bit protected mode, loads 43,008 trained weights into memory, and enters a loop: keyboard input → neural forward pass → screen output. On first boot, a demo sequence feeds synthetic keypresses through the network to prove the substrate is alive.

Type `box` and the network draws a rectangle. Type `line` and it draws a line. Press `p` for a pixel. The network learned these mappings during offline training — the assembly code contains no `if key == 'b' then draw_rect` logic. It's weights all the way down.

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
├── src/dnos.asm          Bootable kernel (32-bit PM, VESA, IRQ, neural core)
├── tools/train.py        Training pipeline (Adam, Q8.8-aware, 100% accuracy)
├── Makefile              make && make run
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
| 2 | 43,008 | 32-bit protected mode | Current |
| 3 | ~1M | Paging + disk swap | Planned |
| 4+ | 10M–10B | GPU / distributed | Research |

Long-term: memristor crossbar arrays where matrix multiplication happens as physics (Ohm's Law), not computation.

## Building

```bash
make              # Assemble + train + patch weights → dnos.img
make run          # Build and launch in QEMU
make train        # Retrain weights only
make validate     # Train + verify Python↔assembly math agreement
make clean        # Remove artifacts
```

Requires `nasm`, `python3` with `numpy`, and `qemu-system-i386`.

## License

Public domain.
