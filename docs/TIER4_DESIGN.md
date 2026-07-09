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
  `train.py --ternary`. **Amendment (implementation finding):** the
  per-layer shift k must be *frozen after a short warmup* (200 epochs).
  Recomputing k from the live shadow weights makes the layer scale
  jump 2× whenever mean|w| crosses a power-of-two boundary — observed
  as oscillating accuracy (best 41.7% at 4000 epochs); freezing k
  restores monotonic convergence (100% integer-law accuracy by 1500).
  The threshold Δ stays dynamic — mask churn is local and harmless;
  scale churn is global and fatal. Only the shifts land in the header;
  Δ is a training-time detail invisible to the law.
  **Further campaign findings (2026-07-08, all runs seeded):**
  (a) *Density hurts*: Δ=0.5 (denser ±1 mask) scored strictly worse
  than Δ=0.75 at equal epochs (38.8% vs 49.3% held-out contexts) —
  sparser ternary carries cleaner signal here; Δ=0.75 is the default.
  (b) *Coverage dominates*: halving the resample interval (50→25)
  moved held-out contexts 82.6%→91.4% at equal epochs — worth more
  than +2000 epochs. Ternary defaults to `--resample-every 25`.
  (c) *Checkpoint selection must be held-out and quarantined*:
  selecting on the training stream stops discriminating at saturation
  (canonical suite regressed while contexts improved). Ternary selects
  on a fixed 3-draw held-out context sample (seed 555000 — distinct
  from the CI eval seed 99173; selecting on the CI set would be
  leakage), ties advancing to more-trained weights. Q8.8 keeps its
  historical selection verbatim: the canonical CRC 0xB674A6AF was
  reproduced bit-exactly after the quarantine, proving the shipped
  law is untouched by any ternary-mode code path.
  (e) **Ternary as a bug detector — the campaign's largest finding.**
  The persistent 38x/386 plateau with wandering misses was not a
  ternary capacity ceiling: training histories were capped at 31
  events while the Tier 3 window, the eval, and live use span 64 —
  the deep half of the window trained as permanent zeros. Q8.8 had
  the precision slack to interpolate over the gap; 1.58-bit weights
  did not, and surfaced it. Fixing the cap made the task honest and
  *upgraded the canonical Q8.8 law itself*: 386/386 with held-out
  context generalization 99.9% (was 98.4%), canonical config now
  6000 epochs / resample 25 (the bare `train.py` defaults), CRC
  0xE07DA759 (supersedes 0xB674A6AF), full live gauntlet re-verified.
  Reduced precision is an audit instrument: what interpolation can
  hide, quantization exposes.
  (f) **Windowed-ternary retired as a canonical-law candidate
  (2026-07-09).** After a controlled campaign (epochs to 16k, width to
  2048, resample cadence, freeze timing, density, held-out selection),
  the best windowed-ternary law reached 375/386 canonical / 94.9%
  held-out — under the gates, with a *structural* explanation: the
  task demands that recent events outweigh deep history, Q8.8 encodes
  that ratio in weight magnitudes, and ternary weights have none —
  uniform-loudness connections cannot express a fade-out across the
  64-event window, and no training schedule adds expressiveness the
  format lacks. (Freeze-later was also refuted: pinning at epoch 1000
  let the output scale drift to sar 0 — 272/386. Early freeze wins.)
  Per-neuron shifts (format v4b, header[28]=1, byte table between
  header and weights, CRC over both) are implemented and bit-exact in
  the format gate, but they adjust per-neuron gain, not the
  recent-vs-ancient ratio within a neuron — so the canonical ternary
  law is deferred to **Part B, where decay is architectural**: the
  recurrent state fades old information per tick, ternary weights
  encode only *what* to detect, and the window (the thing demanding
  magnitudes) is deleted. Part A and Part B are one design on this
  task, not two milestones.
  (d) *Cross-environment reproducibility (observed, not guaranteed)*:
  the Q8.8 canonical CRC is bit-identical between the cloud session's
  environment and this machine's numpy 2.5 — strong evidence for S2
  replay-verification across verifiers, but float-trajectory
  portability across BLAS builds is not a theorem; S2's dataset-delta
  design should still pin or attest the verifier environment.
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

## Open questions — resolutions (2026-07-09)

1. **Event unit: RESOLVED — stays 8 features = the key byte.** The S1
   v1 frame is therefore `[version byte][event byte]`; nothing widens.
2. **h width:** 512 channels default; the lab measures 256 vs 512.
3. **Sequencing: RESOLVED by the Part A campaign — one retrain, A+B
   together.** Windowed-ternary is structurally mismatched (finding f);
   the ternary law ships with the recurrent core or not at all.
4. **Demo sequence:** regenerate demo assertions from the simulator.

## Part B integer law (design of record for the lab)

```
h: 512 x int16 (Q8.8), zeroed at boot; JOINS canonical state.
Per tick, event x = 8 features in {0, 256}:
  leak:   hl[c] = h[c] - (h[c] >> d[c])     ; lambda_c = 1 - 2^-d_c
  drive:  u[c]  = (sum_i s_in[i,c] * x[i]) >> k_in
  h'[c]  = clamp(hl[c] + u[c], -32768, 32767)
  y      = Tier 4a ternary MLP readout on h' (512->1024->384->64)
```

- `d` is a **structural decay spectrum**, not trained: 8 log-spaced
  values d in {1..8} (horizons ~2 to ~256 ticks), 64 channels each.
  Stored as a byte table (v4b mechanism); shift-subtract is exact in
  int16, no multiply anywhere.
- `s_in` is ternary 8x512; with <=8 terms of +/-256 the drive needs
  headroom, not precision — k_in is per-layer (v4a mechanism).
- Training is truncated BPTT (float shadow, STE, loss at final step);
  the gradient horizon is a training detail — the machine has none.
- **Known risk for the lab to measure:** h is a leaky superposition —
  order sensitivity (line vs nile) now lives only in decay
  differences, where the window encoded position explicitly. If word
  commands suffer, the cheap remedy is a 2-tap drive (x(t) and
  x(t-1) both feed W_in: 16x512), before anything heavier.

> **Feasibility: CONFIRMED (2026-07-09, `tools/ssm_lab.py`, 10k
> epochs, seed 1337).** Canonical 387/387, held-out streams
> 1049/1050 (99.9% — equal to the Q8.8 canonical law), word commands
> 100% in honest typing order (order-sensitivity risk retired; no
> 2-tap drive needed). 946,176 ternary weights ≈ 236 KB packed, no
> multiplies anywhere in the law. The windowed campaign's structural
> ceiling (finding f) does not exist here: recency-as-architecture
> lets uniform-magnitude weights match full-precision generalization.
> Next: graduation — format v5 in train.py, stateful simulator,
> h into canonical state and the swarm digest, kernel recurrent tick
> with the packed-walk inner loop, demo regenerated from the
> simulator, every gate extended.

## Exit criteria

Tier 4 closes when the full gauntlet passes with ternary weights and
the recurrent core: zero divergence on `h` trajectories, streaming
generalization ≥ 95%, integrity negative tests for the packed format,
swarm replication with `h` in the digest, and the README's "Verified"
paragraph true of the new engine with only the numbers changed. The
session-log format recorded by the S1 client must replay against the
new engine without *syntactic* change.
