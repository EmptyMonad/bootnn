# Tier 3 Design — ~1M Weights via Streaming Inference

> Status: **implemented** (Strategy A). Kept as the design of record.
> Resolved open questions: (1) bounce-copy chosen — implemented as
> unreal-mode a32 copies from a 32 KB bounce buffer, the boring and
> portable option; (2) tick rate fixed at 20 Hz, verified live by
> `tools/tick_test.py` (delta_step == delta_tick under QEMU TCG);
> (3) IAL-token input encoding deferred to the distributed phase.

## Goal

Scale the substrate from Tier 2's 43,008 weights (86 KB, fits below 640 KB
real-mode-loadable memory) to ~1,000,000 weights (~2 MB Q8.8), while
preserving every correctness property Tier 2 established:

- Bit-exact Python↔assembly agreement (`simulate_assembly_forward`)
- Seeded, CRC-stamped canonical weights, validated at boot
- One state transition per PIT tick
- The full CI gauntlet, extended, on every push

## Proposed topology

```
1024 → 768 → 384 → 64        (packed input → wide features → compress → cmd)
W1: 1024×768 = 786,432
W2:  768×384 = 294,912       (exceeds budget)
```

That overshoots. The working proposal is:

```
512 → 1024 → 384 → 64
W1: 512×1024 = 524,288
W2: 1024×384 = 393,216
W3:  384×64  =  24,576
              ─────────
               942,080 weights ≈ 1.80 MB Q8.8   ✔ under 2 MB
```

Input grows from 32 events × 8 features to 64 events × 8 features (512),
doubling the context window — directly useful for longer word commands and
IAL token streams.

## The core problem: weights no longer fit below 1 MB

Tier 2 loads all weights to 0x10000–0x25400 in real mode via INT 13h and
never touches disk again. 1.8 MB cannot live below 0x100000, and INT 13h is
unavailable after the PM switch.

Two candidate strategies:

### Strategy A — load-high once, no paging (preferred)

1. Stage 2 enables **unreal mode** (flat 4 GB data limit while staying in
   RM) — or performs repeated RM↔PM bounce-buffer copies — to place the
   entire weight blob at 0x00200000 (2 MB mark) during boot.
2. The PM kernel reads weights directly from high memory. No pager, no
   streaming, no disk driver.

Costs: a bounce buffer (64 KB at 0x10000, already reserved) and a copy loop;
boot time grows by ~2 MB of INT 13h reads (≈ 1 s). No steady-state cost.
Requires ≥ 4 MB guest RAM (QEMU flag becomes `-m 16M`, already true).

**This preserves determinism trivially** — after boot, the system is exactly
Tier 2 with bigger matrices. `S(n+1) = f(S(n), input(n))` needs no disk in
the loop.

### Strategy B — on-demand weight streaming (rejected for Tier 3)

Page W1 row-blocks from disk during the forward pass. Requires a PM disk
driver (ATA PIO), a block cache, and makes inference latency depend on cache
state — which violates tick determinism unless the tick is slowed to
worst-case. Complexity is Tier 4+ territory (and the memristive vision makes
it moot: conductance doesn't page). Documented here so the decision is
explicit.

## Memory map (Tier 3)

```
0x00008600  Kernel (unchanged, window grows to 61 sectors max)
0x00010000  Bounce buffer (64 KB, boot-time only; reused as scratch after)
0x00030000  Input history (64 events)
0x00060000  Activations (1024+384+64 int16 ≈ 3 KB — unchanged region)
0x00200000  Weight blob: 128 B header + 1,884,160 B weights + CRC
```

## Header v3

Same 128-byte layout; `version = 3`, `tier = 3`, layer sizes
(512, 1024, 384, 64), weight count 942,080, CRC32 over all weight bytes.
`validate_weights` gains nothing new — it already reads sizes from
constants; Tier 3 switches it to read *expected* sizes from build-time
equates exactly as today. Boot-time CRC over 1.8 MB at ~10 cycles/byte is
< 100 ms on any target.

## Forward pass changes

- Accumulators stay 32-bit; the dot-product length grows from 256 to 1024.
  Q8.8 with int16 weights and int16 activations accumulated in int32
  overflows at ~65k terms in the worst case — 1024 terms is safe by a wide
  margin (headroom ×64), but `simulate_assembly_forward` must model the
  int32 accumulator explicitly, as it does today.
- The unrolled per-layer loops become parameterized by the header's layer
  sizes (read once at boot into kernel variables), eliminating the
  hand-duplicated layer code.

## Training

`tools/train.py` scales as-is (numpy); quantization-aware epochs will need
minibatching for the 512×1024 layer but no algorithmic change. Resampled
random-context training (the generalization fix) carries over unchanged.
Seed stays 1337; CRC stays canonical.

## Test plan (all gates extended, none removed)

| Gate | Tier 3 form |
|------|-------------|
| Convergence + 0% divergence | unchanged, larger matrices |
| Context generalization ≥ 95% | unchanged, 64-event windows |
| Boot test | asserts weights land at 0x200000 (new symbol `w_base`) |
| Integrity test | corrupt a byte above the 1 MB mark; must reject |
| Tick test | unchanged invariant; tick budget re-measured (forward pass is ~22× Tier 2 MACs; at 100 Hz the budget is 10 ms — ~1M MACs at even 10 cycles/MAC on a 100 MHz floor is 100 ms, so **the PIT divisor becomes a measured, documented constant**, likely 10–20 Hz for Tier 3 on QEMU TCG; determinism is rate-independent) |
| Differential test | unchanged: law == metal per key |

## Open questions (to resolve before implementation)

1. Unreal mode vs. PM bounce-copy for the high load — unreal mode is less
   code but relies on segment-descriptor cache behavior; PM bounce-copy is
   boring and portable. Lean: bounce-copy.
2. Tick rate: fix at build time after measuring the Tier 3 forward pass
   under QEMU TCG and on one real machine; export as a header field so the
   simulator and kernel agree on the time base.
3. Whether the input encoder consumes IAL tokens directly (8-byte token =
   8 features — a clean fit) — this would make the Rust IAL the canonical
   host-side encoder and the kernel's `encode_input` its bare-metal twin,
   with a shared differential test.

## Exit criteria

Tier 3 closes when the full CI gauntlet passes with the 942,080-weight
network, including the high-memory integrity negative test, and the
README's "Verified" paragraph is true of the new topology with zero edits
other than the numbers.
