# Tier 4 Design — Ternary Weights + a Recurrent Core

> Status: **proposed** (design of record for the engine swap).
> Prerequisites: Tier 3 (implemented), Swarm S0/S1-v0 (implemented —
> the interface this engine sits behind is frozen and gated, so Tier 4
> changes the engine, not the boundary).

## Goal

Retire the two assumptions Tier 3 carried forward (recorded in
`docs/SWARM_DESIGN.md`, "Dated assumptions"):

1. **The sliding window contradicts the creed.** `S(n+1) = f(S(n),
   input(n))` is a recurrence, but the network is a feedforward MLP
   over a replayed 64-event window — the window is a workaround for
   having no recurrent state. Tier 4 gives `f` real state.
2. **Q8.8 int16 weights are two quantization generations old.**
   Ternary weights ({-1, 0, +1}, 2-bit packed) make inference
   multiply-free — add/sub/skip only, retiring `imul` from the
   inference path entirely (the instruction class behind the worst
   bug in the project's history) — and shrink the blob ~8×.

What does *not* change: seeded bit-exact training, CRC-stamped
canonical weights validated at boot, one transition per tick,
`simulate_assembly_forward` as the law, every CI gate extended and
none removed, and the S1 event boundary (a session log recorded on
Tier 3 must still be *syntactically* valid input to Tier 4; the
decoded behavior may differ because the law differs).

## Part A — Ternary weights (independent of Part B; can land first)

- **Format:** weights in {-1, 0, +1}, packed 2 bits each (00=0, 01=+1,
  11=-1; 10 reserved/invalid — the validator rejects it). Per-layer
  power-of-two scale `s = 2^-k` applied to accumulators as `sar k`.
  Activations and accumulators stay Q8.8 int16 / int32 — the
  portability and debuggability properties live there and are kept.
- **Training:** quantization-aware with STE over a ternary projection
  (threshold at ±Δ·mean|w|, BitNet-b1.58 style), seeded as ever.
  `train.py --ternary`; the projection threshold is part of the header
  so the simulator and the metal agree on the law by construction.
- **Kernel:** the inner loop walks packed 2-bit weights: 00 → skip,
  01 → `add`, 11 → `sub`. No unpack buffer — walk packed directly
  (4 weights per byte, shift-and-mask). No `imul` anywhere in
  inference.
- **Capacity note:** 942,080 weights → ~236 KB packed. The blob fits
  below 1 MB again; the Tier 3 high-load machinery becomes headroom
  (~30 M ternary weights fit in the same 1.8 MB budget) rather than a
  necessity. Keep loading high anyway — the headroom is the point.
- **Header v4a:** version 4, `quant = ternary`, per-layer scales,
  projection threshold, CRC32 over the *packed* bytes.

## Part B — Diagonal state-space core (the engine matches the creed)

```
h(t+1) = λ ⊙ h(t) + W_in · x(t)         (the recurrence IS the tick)
y(t)   = MLP(h(t+1))                     (existing readout, smaller)
```

- **State:** `h` = 512 int16 channels at a fixed address (new symbol
  `h_base`), zeroed at boot. `h` joins the canonical state vector —
  swarm digests then cover working memory itself.
- **Decay spectrum:** λ per channel, quantized as `λ = 1 - 2^-k`,
  k ∈ {1..14} log-spaced across channels. The leak is shift-and-
  subtract — multiply-free, exact in int16, and the spectrum gives
  horizons from ~2 ticks to ~16k ticks in one mechanism. This
  replaces the 64-event window with an unbounded exponential-horizon
  memory; the history buffer and its shift-and-re-encode are deleted.
- **Input:** x(t) is one event's features. **Decision OQ1 fixes the
  S1 v1 token frame**: the frame carries exactly the feature bytes
  the encoder consumes, versioned. Until OQ1 is resolved, S1 stays at
  v0 (1 byte = 1 event) — do not design the frame twice.
- **Training:** truncated BPTT (window only during *training*, as a
  gradient horizon — the machine itself has none), ternary STE for
  W_in and the readout, float shadow for λ then snap to the k-grid.
  `simulate_assembly_forward` becomes stateful: it carries `h` across
  calls and is the *only* reference — the training graph must equal
  the simulator equal the metal, bit for bit, as today.
- **Tick budget:** one matvec (W_in, ternary: adds only) + elementwise
  leak + readout. Strictly cheaper than Tier 3's three dense Q8.8
  layers; re-measure and re-pin the PIT divisor (expect ≥ 20 Hz).

## Test plan (all gates extended, none removed)

| Gate | Tier 4 form |
|------|-------------|
| Convergence + 0% divergence | stateful: per-tick `h` trajectory AND outputs, Python == metal |
| Context generalization ≥ 95% | streaming form: held-out event streams, no window to hold out |
| Boot | unchanged + `h` zeroed and `h_base` exported |
| Integrity | CRC over packed ternary bytes; flipped 2-bit field rejected; reserved code 10 rejected |
| Tick | unchanged invariant; budget re-measured, divisor re-pinned |
| Differential (law==metal) | interactive_test carries `h` in the simulator |
| Swarm S0 | state vector grows to include `h` digest — replication now covers memory |
| Serial S1 / client | unchanged (the boundary is frozen; that is the point of having landed it first) |

## Open questions (resolve before implementation)

1. **The event unit** (fixes S1 v1 framing): stay at 8 features = the
   key byte, or widen to an 8-byte token (key + modifiers + device id
   + reserved)? Widening retrains everything; do it here or not at all.
2. **h width:** 512 channels is the working default; measure 256 vs
   512 on the streaming eval before committing the memory map.
3. **Sequencing:** land Part A alone first (same topology, ternary
   weights — smallest diff that retires `imul`), or one retrain for
   A+B? Lean: A first, gauntlet green, then B — two small revolutions
   beat one large one.
4. **Demo sequence:** the boot demo's expected outputs change with the
   new law; regenerate the demo assertions from the simulator rather
   than hand-maintaining them.

## Exit criteria

Tier 4 closes when the full gauntlet passes with ternary weights and
the recurrent core: zero divergence on `h` trajectories, streaming
generalization ≥ 95%, integrity negative tests for the packed format,
swarm replication with `h` in the digest, and the README's "Verified"
paragraph true of the new engine with only the numbers changed. The
session-log format recorded by the S1 client must replay against the
new engine without *syntactic* change.
