# PAC-Privacy Integration Feasibility — DNOS (bootnn)

> Feasibility evaluation, derived from repository contents at the current
> checkout. No code written. Conclusions are grounded in the actual sources
> (`src/dnos.asm`, `tools/train.py`, `ial/`, `ndal/`, `docs/`), not the
> aspirational roadmap. Where a subsystem is *planned* or *vision* rather
> than *implemented*, this is stated explicitly.

---

## Executive Summary

DNOS is, at runtime, a **frozen deterministic function**: a 4-layer Q8.8
fixed-point MLP (256→128→64→32, 43,008 weights) that maps a 32-event
keystroke history to a drawing command, executed directly on bare metal with
no OS underneath it. All learning is **offline** (`tools/train.py`); the
device never updates weights. Around this core sit two Rust determinism
layers — IAL (input canonicalization) and NDAL (oracle-mediated
non-determinism with a hash-chained replay log) — that are **not yet wired
into the kernel** and exist as host-side libraries.

**The central finding governs everything below:** DNOS's foundational design
goal is *maximal state reconstructability*. The state equation
`STATE(t+1) = f(STATE(t), INPUT(t))`, the IAL determinism membrane, and
especially the NDAL **replay log** are all engineered so that an observer
holding the log can reconstruct the system's entire internal trajectory
bit-for-bit. PAC-Privacy's goal is the **exact opposite**: bound an
adversary's ability to reconstruct sensitive inputs/state from observations.

This is not a fatal conflict, but it dictates the architecture. PAC-Privacy
**cannot** be layered on top of the replay-log/verification path without
either (a) breaking cross-node verification, or (b) being applied *before*
data enters the log. The viable design treats privacy as a **transform on
the token/output boundary**, with the noise itself recorded as an oracle
draw so determinism-under-replay is preserved while determinism-under-fresh-
observation is deliberately broken.

**Verdict:** Feasible as a host-side (Rust, IAL/NDAL-adjacent) layer with
moderate effort. **Not** feasible at bare-metal kernel level in the current
Tier 2 (no FPU, no entropy source, no allocator, 100 % of behavior in frozen
weights). Neuromorphic adaptation (Phase 4) is speculative on both sides
(no neuromorphic code exists) but the analysis below shows it would *help*
privacy if the substrate ever drifts.

---

# Phase 1 — Repository Analysis

## 1.1 Core architecture (as implemented)

| Concern | Reality in the repo | Evidence |
|---|---|---|
| **Core** | Single bare-metal x86 image: stage1 → stage2 (16-bit) → 32-bit PM kernel. The kernel *is* a neural forward pass plus graphics primitives. | `src/dnos.asm:39-538` |
| **Execution model** | `STATE(t+1)=f(STATE(t),INPUT(t))` realized as: keystroke → `encode_input_32` (shift history, append) → `neural_forward_32` → `decode_output_32` → draw. Single-threaded `main_loop` with `hlt`. | `src/dnos.asm:802-843`, `855-1037` |
| **Scheduler** | **None.** PIT fires at 100 Hz but `timer_isr` only does `inc [tick_count]` (`dnos.asm:645-653`). The README/source comment "preemptive scheduling" is aspirational — there are no tasks to schedule. Delays are tick busy-waits. | `dnos.asm:630-653`, `772-778` |
| **IPC / message passing** | **None.** The only queue is the 256-byte keyboard ring buffer written by IRQ1 (`keyboard_isr`) and drained by `read_key`. No processes, no channels. | `dnos.asm:659-709` |
| **Memory management** | **Static physical map, no paging, no allocator.** Fixed bases: weights `0x10000`, activations `0x60000`, history `0x30000`, keybuf `0x70000`. Tier 3 (paging + weight swap) is *planned*, not present. | `dnos.asm:22-31`, `430-433`; `ROADMAP.md:80-84` |
| **Driver interfaces** | Direct port I/O only: PS/2 (`0x60/0x64`), PIC (`0x20/0xA0`), PIT (`0x40/0x43`), VGA/VESA LFB, INT 13h LBA disk (real mode). No driver abstraction. | `dnos.asm:313-356`, `596-640`, `214-242` |
| **AI / inference subsystem** | The forward pass *is* the kernel. Q8.8: `imul`/`sar 8` MAC, ReLU+clamp hidden, piecewise-sigmoid output, argmax over outputs `[0:20]`. Bit-exact twin in Python (`simulate_assembly_forward`). | `dnos.asm:892-1037`; `train.py:649-717` |
| **Neuromorphic components** | **None exist.** Memristor/ReRAM/SNN are explicitly labelled *Vision* in the roadmap. | `ROADMAP.md:79-101` |
| **Persistent state** | At runtime: **none persisted.** Weights are read-only from disk. `CMD_SAVE`/`CMD_UNDO` are defined constants but **not dispatched** in `decode_output_32` (no `.cmd_save` handler). NDAL log/snapshots are host-side, in-memory (`Vec`), "production would use mmap or disk". | `dnos.asm:1106-1133` (no save case); `ndal/src/log.rs:86-105` |

### Learning vs. inference (critical for privacy)

All weight modification happens **offline**, in `tools/train.py` (Adam,
quantization-aware STE, softmax-CE over 20 command logits). The bare-metal
runtime contains **no backprop, no weight write path**. Therefore:

- "Learning modifies future behavior" is a **build-time** event, not a
  runtime one. Two boots of the same image are identical functions.
- The runtime's only mutable state is the small set listed in §1.3.

This drastically narrows the runtime leakage surface — but it also means the
*model itself* (the weights) is the real secret, and it is shipped in the
clear inside every disk image (`dnos.asm:1586-1610`, patched by
`tools/build.py`).

## 1.2 Architecture & data-flow map

```
                          BUILD TIME (host, Python)
  train.py: Adam + QAT/STE ──► weights.bin (43,008×i16 + 128B hdr + CRC32)
                                         │ build.py patches at LBA 69
                                         ▼
══════════════════════════════════════════════════════════════════════════
                          RUN TIME (bare metal, frozen)
  PS/2 key ─IRQ1─► ring buf ─► read_key ─► scan_to_ascii
                                              │
                                              ▼
                                   encode_input_32
                          (shift 32-event history, append 8-bit binary)
                                              │  HISTORY_BASE (recurrent state)
                                              ▼
                                   neural_forward_32  ◄── WEIGHT_BASE (RO)
                          256→128(ReLU)→64(ReLU)→32(piecewise-sigmoid)
                                              │  ACTIV_BASE (scratch)
                                              ▼
                                   decode_output_32
                  argmax[0:20]→cmd ; outputs[20:24]→cursor Δ ; clamp
                                              │
                          mutates: cursor_x/y, color_idx, draw_size,
                                   last_cmd, last_confidence
                                              ▼
                                   graphics primitives ─► VGA/VESA framebuffer
══════════════════════════════════════════════════════════════════════════
            DETERMINISM LAYERS (Rust, host-side, NOT yet in kernel)
  RawObservation ─► IAL Pipeline ─► canonical Token stream ─► stream_digest
       (temporal/spatial/semantic quantize + canonical sort)
  oracle query ─► NDAL Pipeline ─► OracleToken + append to hash-chained
       (Random/Clock/Env/Net/Consensus)        Replay Log ─► snapshots
```

## 1.3 Trust boundaries

```
  ┌──────────────────────────────────────────────────────────────┐
  │ T0  Physical world / adversary-observable surface            │
  │     • Screen framebuffer (every drawn pixel)                  │
  │     • Timing of responses (PIT-paced)                         │
  ├──────────────────────────────────────────────────────────────┤
  │ T1  Bare-metal kernel (ring 0, single address space)         │
  │     • NO isolation: weights, history, activations, cursor    │
  │       all in one flat map. IDT at 0x0, stack at 0x90000.      │
  │     • Adversary with code-exec = total state read.           │
  ├──────────────────────────────────────────────────────────────┤
  │ T2  The weights (the "law")                                  │
  │     • Shipped in cleartext in every image. The real secret.  │
  ├──────────────────────────────────────────────────────────────┤
  │ T3  Host determinism layers (IAL/NDAL)                       │
  │     • Replay log: DESIGNED to be exported/shared for sync.   │
  │       → trust boundary is intentionally porous here.         │
  │     • Snapshots carry weights_hash, state_hash, ial_digest.  │
  ├──────────────────────────────────────────────────────────────┤
  │ T4  Peer / distributed nodes (Consensus oracle, log share)  │
  │     • Receive logs & snapshots to verify state equality.     │
  └──────────────────────────────────────────────────────────────┘
```

The privacy-relevant insight: **the trust boundary between T3 and T4 is
deliberately weak** — the whole point of the replay log and `find_divergence_
with` (`ndal/src/log.rs:335-347`) is that peers can reconstruct and compare
each other's execution. Any sensitive input that reaches the log is, by
design, exportable to peers.

## 1.4 State-transition model

Runtime hidden state vector **x(t)**:

```
x = ( H[0..255],      # 32 events × 8 features, the recurrent history
      cursor_x, cursor_y,
      color_idx, draw_size,
      last_cmd, last_confidence,
      tick_count,
      framebuffer )    # technically output, but read back by draws
```

Transition on input key u(t):

```
H'      = shift(H) ++ binary8(u)                       # encode_input_32
a       = forward_Q88(H', W)                           # neural_forward_32
cmd     = argmax(a[0:20])
cursor' = clamp(cursor + scale(a[20:24] - 16384))      # decode_output_32
{color_idx,draw_size,framebuffer}' = dispatch(cmd)
```

W (weights) is constant ⇒ the system is a **time-invariant deterministic
automaton** whose only memory is `H` plus the small modal registers.

## 1.5 State-Leakage Surface report

Every location where internal state influences an observable, where a query
reveals hidden state, where long-term memory lives, and where learning
changes behavior:

| # | Location | Mechanism | Long-term? | Learning? | Leak channel |
|---|---|---|---|---|---|
| L1 | `decode_output_32` argmax → framebuffer | command is drawn; observer reads pixels | no | no | **screen** |
| L2 | outputs `[20:24]` → `cursor_x/y` (`dnos.asm:1072-1080`) | continuous cursor delta leaks raw network output, not just argmax | no | no | **screen (high-bandwidth)** |
| L3 | `H` history buffer (`HISTORY_BASE`) | output depends on last 32 keys | session | no | screen (weak — see §2) |
| L4 | `color_idx`, `draw_size` | drawn color/size reveal modal state | session | no | screen |
| L5 | Weights on disk, cleartext | model = the law | permanent | build-time | **disk image / model extraction** |
| L6 | **NDAL replay log** (`ndal/src/log.rs`) | every oracle response recorded, hash-chained, exportable | permanent | (training RNG) | **peer/export — the dominant surface** |
| L7 | NDAL snapshots (`Snapshot`) | `weights_hash`, `state_hash`, `ial_digest` | permanent | no | equality/membership confirmation |
| L8 | IAL `stream_digest` (`pipeline.rs:182`) | rolling hash of entire token (input) stream | session→export | no | confirms "did user X do action Y" |
| L9 | Env oracle: `BootTimestamp`, `CpuCount` (`oracles.rs:228-246`) | machine fingerprint enters the log | permanent | no | **deanonymization via log** |
| L10 | Clock oracle | wall-clock (truncated) enters log | permanent | no | temporal correlation |
| L11 | Random oracle seed (`seed_from_system`, nanosecond time) | one logged seed determines *all* "random" behavior | permanent | training | seed → full PRNG stream |

The five that matter for a PAC-Privacy effort, in priority order:
**L6 (replay log) > L5 (cleartext weights / extraction) > L2 (cursor-delta
output bandwidth) > L9/L11 (fingerprint+seed in log) > L3 (history)**.

---

# Phase 2 — Observability Analysis

Framing per control theory: treat the runtime as a discrete state system
`x(t+1)=f(x,u)`, `y(t)=h(x)`. "Observable" = can an adversary reconstruct
hidden `x` (or the parameters `W`) from a sequence of `(u, y)` pairs? The
distinction from classic observability is that here `h` is lossy
(argmax collapses 20 logits to one index) for some channels and near-lossless
for others (cursor delta).

### Per-subsystem observability

| Subsystem | Observable output | Hidden state | Reconstruct via probing? | Control-theory note | Risk |
|---|---|---|---|---|---|
| **Cursor (L2)** | drawn cursor position + draw sites | `cursor_x/y` AND raw `a[20],a[22]` | Yes, directly. The cursor delta is `(a-16384)>>10` — observer inverts to recover ~10 bits of two output neurons each step. | Fully observable; `h` near-injective on this channel | **HIGH** |
| **Command (L1,L3)** | which primitive was drawn | argmax index; indirectly `H` and `W` | Partially. argmax leaks ⌈log2 20⌉≈4.3 bits/step about `a[0:20]`. `H`-dependence was *deliberately trained out* (context-robust single keys: 100 % on held-out histories, `ROADMAP.md:115-124`) → `H` barely affects argmax. | Weakly observable: the map H→cmd is near-constant by construction | **LOW** (for H), **MEDIUM** (for W via extraction) |
| **Modal regs (L4)** | color/size of drawn shapes | `color_idx`, `draw_size` | Yes, one draw reveals them. | Fully observable | LOW (not sensitive) |
| **Weights (L5)** | (cmd, cursorΔ) over chosen inputs | `W` (43,008 i16) | Yes — black-box **model extraction**. Input is 256-dim binary, 20 classes + 4 regression outputs. Cursor-delta outputs give *real-valued* labels → far fewer queries than pure classification. | The system is the textbook extraction target: small, deterministic, real-valued outputs | **HIGH** |
| **IAL pipeline** | token stream / `stream_digest` | buffered word `key_buffer`, `last_mouse_bucket`, epoch counters | `stream_digest` is a *hash* → not invertible, but it is an **equality oracle**: adversary who guesses an input stream can confirm it. | Hash output ⇒ one-way; confirmation attack only | MEDIUM |
| **NDAL log (L6,L9-11)** | the exported log itself | RNG seed, clock, env fingerprint, full causal trace | **Trivially — by construction.** Replay reconstructs everything bit-exact (`pipeline.rs:101-105`). No probing needed; the log *is* the reconstruction. | Not "observability" but total disclosure | **CRITICAL** |
| **Snapshots (L7)** | `weights_hash`, `state_hash` | weights, neural state | One-way hashes ⇒ confirmation/membership only, unless state space is small enough to brute-force (modal regs are!). | Membership oracle | MEDIUM |

### Repeated-probing reconstruction summary

- **Single-step state** (cursor, color, size): reconstructable in O(1)
  observations. Not sensitive, but they are the carrier for L2's leak of raw
  output neurons.
- **History `H`**: low reconstruction risk *because the model was trained to
  be history-invariant on the command channel*. This is an accidental
  privacy win from the determinism program — context-robustness reduces the
  mutual information between `H` and the argmax output. The cursor-delta
  channel, however, was **not** hardened this way and may still leak `H`.
- **Weights `W`**: high extraction risk; real-valued cursor outputs make it
  a regression-style extraction, the easy case.
- **NDAL log**: not a probing problem — it is designed disclosure. CRITICAL.

### Classification

| Subsystem | State-reconstruction risk |
|---|---|
| NDAL replay log + snapshots (export path) | **CRITICAL** |
| Cursor-delta output channel (L2) | **HIGH** |
| Weights via black-box extraction (L5) | **HIGH** |
| IAL stream digest (confirmation) (L8) | **MEDIUM** |
| Env fingerprint / RNG seed in log (L9/L11) | **MEDIUM** (HIGH once log is shared) |
| Command/argmax channel & history `H` (L1/L3) | **LOW** |
| Modal registers (L4) | **LOW** |

---

# Phase 3 — PAC-Privacy Layer Design

## 3.0 Why PAC-Privacy fits this codebase unusually well

PAC-Privacy (Xiao & Devadas, MIT 2023) calibrates noise to **empirically
measured** information leakage: run the mechanism on resampled inputs,
measure the covariance of the outputs, add anisotropic Gaussian noise sized
to that covariance to bound an adversary's posterior advantage. It is
*black-box* — it never needs to understand the mechanism internally.

This repo already contains the machinery PAC-Privacy needs:

- **Resampling harness**: `train.py` already resamples random input histories
  every 50 epochs (`train.py:408-416`) and `tools/context_eval.py` already
  scores outputs over held-out random histories. PAC's leakage estimator is
  the same loop with covariance instead of accuracy.
- **An exact, fast mechanism twin**: `_exact_int_logits` / `simulate_
  assembly_forward` (`train.py:366-382`, `649-717`) give a vectorized,
  bit-exact forward pass — ideal for the thousands of mechanism evaluations
  PAC's Monte-Carlo estimator requires, without booting QEMU.
- **A noise-recording channel**: the NDAL Random oracle + replay log is
  *exactly* the primitive needed to make perturbation **deterministic under
  replay but fresh under live observation** — solving the determinism/privacy
  conflict (see §3.6).

## 3.1 Component overview

```
                 (host-side, Rust, sits between IAL/NDAL and substrate)
  Token stream ─► [1 Query History Tracker] ─► [2 Leakage Estimator]
                              │                          │ Σ (output covariance)
                              ▼                          ▼
                    [3 State Reconstruction      [4 Privacy Risk Engine]
                        Simulator]  ───────────────►  budget bookkeeping
                              │                          │ noise scale Λ
                              └──────────────┬───────────┘
                                             ▼
                                  [5 Output Perturbation Layer]
                          (adds NDAL-logged calibrated noise to outputs
                           [20:24] cursor Δ and/or command logits)
                                             ▼
                                   substrate / framebuffer
```

All five are **host-side**. Bare-metal placement is addressed in §3.7 and is
*partial at best* in Tier 2.

---

### Component 1 — Query History Tracker

- **Purpose**: maintain the per-principal record of (input token, output)
  pairs the system has revealed, so the Risk Engine can account for
  *cumulative* leakage (PAC bounds compose over queries). This is the privacy
  analogue of, and can reuse, the IAL `TokenEncoder` history (`pipeline.rs:
  247-304`) and the NDAL log's sequencing.
- **Interfaces**:
  - `record(principal_id, input_digest: u64, output: &Output, epoch: Epoch)`
  - `window(principal_id) -> &[QueryRecord]`
  - `decay(policy)` — sliding window / exponential forgetting.
- **Data structures**: ring buffer per principal of
  `QueryRecord { epoch, input_digest: u64, output_quant: SmallVec<i16>,
  noise_drawn: i16x4 }`. Reuse `StreamHash` from IAL for `input_digest`.
- **Runtime cost**: O(1) per query (hash + push). Negligible.
- **Memory**: `window_len × ~32 B × n_principals`. For a single-user kernel,
  one window — kilobytes.
- **Failure modes**: window too short → underestimates cumulative leakage
  (under-noises, **privacy failure**); unbounded growth if `decay` misconfig
  → memory leak; principal mis-attribution on a shared device → cross-user
  leakage. Fail **closed** (treat unknown principal as worst-case).

### Component 2 — Leakage Estimator

- **Purpose**: estimate, per output channel, how much the output reveals
  about the sensitive input — the empirical covariance Σ that PAC-Privacy
  calibrates noise to.
- **Interfaces**:
  - `estimate(mechanism: &dyn Fn(Input)->Output, prior: &InputSampler,
     n: usize) -> LeakageEstimate { sigma: Mat, mi_bits: f64 }`
  - offline `precompute(weights) -> LeakageTable` (per-command, per-channel).
- **Data structures**: covariance matrix over the perturbable outputs
  (the 4 cursor-delta neurons `[20:23]`, optionally the 20 command logits
  pre-argmax); `LeakageEstimate` cached keyed by weights CRC32 (already in
  the header, `train.py:606-610`).
- **Mechanism evaluations**: use `_exact_int_logits` (vectorized, thousands/s)
  — **do not** boot QEMU. The estimator *is* `context_eval.py`'s loop with
  `np.cov` over outputs instead of `argmax==target`.
- **Runtime cost**: **offline/amortized.** Estimation is `n` (≈10³–10⁴)
  forward passes at build time; runtime just reads the cached table. This is
  the key to "minimize impact on inference quality": the expensive part never
  runs on metal.
- **Memory**: covariance is 4×4 (cursor) or 24×24 (with logits) i32 — bytes.
- **Failure modes**: prior `InputSampler` not matching real input
  distribution → mis-estimated Σ (the classic PAC assumption — privacy holds
  only w.r.t. the modeled prior); too few samples → noisy Σ, under-noising.
  Mitigate by drawing the prior from the same `history_pool` the model was
  trained on (`train.py:161-167`) and reporting a confidence interval.

### Component 3 — State Reconstruction Simulator

- **Purpose**: the *adversary model*. Given the released (noised) outputs,
  attempt to invert to the sensitive input/state; its success rate is the
  empirical privacy loss that validates the Risk Engine's noise choice.
- **Interfaces**:
  - `reconstruct(observations: &[Output]) -> PosteriorEstimate`
  - `advantage(true_input, posterior) -> f64` (≥0; 0 = no advantage).
- **Data structures**: for this small system, a nearest-neighbor / MAP
  inverter over the precomputed input→output table; for the cursor channel,
  literal inversion of `(a-16384)>>10`.
- **Runtime cost**: **offline only** (CI gate / red-team), never on metal.
- **Memory**: holds the forward table (≈ #distinct test inputs × output dim).
- **Failure modes**: a weak simulator gives false confidence (under-noising).
  Treat it as an adversarial *lower bound* on risk and keep a safety margin;
  gate CI on it (analogous to the existing differential test).

### Component 4 — Privacy Risk Engine

- **Purpose**: convert a `LeakageEstimate` + remaining budget into a concrete
  noise covariance Λ, and enforce per-principal **privacy budgets** (PAC's
  composition over queries).
- **Interfaces**:
  - `noise_scale(estimate, budget_remaining, channel) -> NoiseSpec`
  - `charge(principal, spent: f64) -> Result<(), BudgetExhausted>`
  - `config(PrivacyBudget { epsilon_like: f64, per_epoch_cap, deterministic_
     channels: ChannelMask })`
- **Data structures**: `PrivacyBudget` (configurable, the required knob);
  per-principal `f64` accumulator; `ChannelMask` marking which outputs MUST
  stay exact (e.g. `CMD_CLEAR` semantics) vs. may be perturbed.
- **Behavior for "preserve deterministic operation where required"**: channels
  in `deterministic_channels` are **never** noised (argmax command index is a
  natural candidate — keep the *command* exact, perturb only the *cursor
  delta* magnitude). This bounds quality impact to sub-pixel/few-pixel cursor
  drift, leaving the drawn-command semantics intact.
- **Runtime cost**: O(1) table lookup + budget arithmetic.
- **Memory**: tens of bytes per principal.
- **Failure modes**: budget exhaustion policy — fail **closed** (stop
  emitting the sensitive channel, or saturate noise) rather than silently
  leaking; clock/seed reuse across reboots could reset budgets (tie budget
  persistence to the snapshot mechanism).

### Component 5 — Output Perturbation Layer

- **Purpose**: actually add the calibrated noise to the released outputs.
- **Interfaces**:
  - `perturb(output: &mut Output, noise: NoiseSpec, draw: OracleToken)`
  - operates on outputs `[20:23]` (cursor Δ) in Q8.8 *before*
    `decode_output_32`'s `sub 16384; sar 10`.
- **Data structures**: noise vector sampled from N(0, Λ) discretized to Q8.8;
  the sample is drawn from the **NDAL Random oracle** so it is logged.
- **Runtime cost**: a handful of integer ops + one oracle draw per inference.
- **Memory**: negligible.
- **Failure modes**: integer/Q8.8 discretization can bias small noise to zero
  (privacy underflow) → use the same `>>` arithmetic the kernel uses and
  verify with the bit-exact simulator; clamping in `decode_output_32`
  (`dnos.asm:1082-1104`) truncates the tail of the noise distribution near
  screen edges → known PAC "bounded-output" caveat, account for it in Σ.

## 3.5 Configurable privacy budgets (mapped to existing config)

The repo already exposes the right shaped knobs; budgets slot alongside them:

- IAL: `epoch_duration_us`, `spatial_grid_size`, `sensor_resolution`
  (`ial/src/types.rs:313-362`) — **coarser quantization is already a privacy
  dial** (k-anonymity-like). PAC budget extends this with output noise.
- NDAL: `max_queries_per_epoch` (rate-limit = composition cap, `oracles.rs:
  408-410`), `clock_resolution` (already a timing-leak control,
  `oracles.rs:155-162`), `enabled_oracles` (disable fingerprinting oracles).
- New: `PrivacyBudget { epsilon_like, per_epoch_cap, channel_mask,
  prior_model }` — a sibling of `NdalConfig`/`IalConfig`, hashed into
  `config_hash` so peers detect mismatched privacy settings.

## 3.6 Resolving the determinism ↔ privacy conflict (the crux)

Naive output noise breaks the core guarantee: two nodes would diverge, and
`find_divergence_with` would fire on every inference. The resolution, using
**existing NDAL primitives**:

1. Draw every noise sample from the **Random oracle** (`pipeline.rs:140-142`).
2. The draw is appended to the **replay log** like any oracle response.
3. **Replay/Verify mode** reads the *same* noise from the log ⇒ replay stays
   bit-exact; cross-node verification still works for nodes that *share the
   log*.
4. **Live, fresh observation** sees genuinely fresh noise ⇒ the adversary
   without the log cannot reconstruct ⇒ privacy holds.

This makes privacy a property of *who holds the log*. It also forces a policy
decision: **the log itself must now be access-controlled**, because the log
de-noises everything (L6 CRITICAL stands). PAC-Privacy protects the
*screen/peer-output* surface; it does **not** protect the log. Protecting the
log is an orthogonal requirement (encryption-at-rest, redaction of L9/L11
oracle entries) and should be tracked as such.

## 3.7 Bare-metal feasibility ("operate at OS level when possible")

Honest assessment for Tier 2 as it exists:

- **Estimator/Simulator/Risk (1–4 heavy parts)**: offline only. They never
  belong on metal; they produce a static `LeakageTable` baked beside the
  weights (extend the 128-byte header or add a second blob). ✅ feasible.
- **Perturbation (5)**: *could* run on metal — it is a few integer ops. But
  it needs (a) an **entropy source** and (b) **Q8.8 noise sampling**. The
  kernel today has neither: `RandomOracle` uses host `SystemTime`
  (`oracles.rs:57-68`), unavailable in ring 0 without an RTC/RDRAND/RDTSC
  path. RDTSC is available (TSC exists) and is the realistic on-metal entropy
  source. A Box-Muller Gaussian needs no FPU if done via a precomputed Q8.8
  inverse-CDF table. ✅ feasible but ~100–200 lines of new asm + a noise LUT.
- **Budget bookkeeping on metal**: trivial integer accumulator. ✅
- **Verdict**: a *minimal* perturbation layer is implementable on Tier 2
  metal (RDTSC entropy + LUT Gaussian + logged draw); the analysis layers
  stay offline. Full PAC-Privacy "at OS level" is **not** achievable in Tier
  2 and realistically lands with Tier 3's pager/daemon.

---

# Phase 4 — Neuromorphic Adaptation (speculative)

No neuromorphic code exists; this section evaluates the *future* (`ROADMAP.md`
Phase 3/4 Vision: memristive crossbars, SNNs, adaptive routing, distributed
attractor memories) against the four posed questions.

**Does hidden state become *more* observable?**
Mixed. Analog memristive matmul (Ohm's-law crossbars) adds device noise and
drift, which *lowers* the SNR of any single observation → **less** per-query
leakage (privacy help). But adaptive/online plasticity reintroduces the
runtime weight-write path that Tier 2 lacks, creating a *new* leakage surface
(L5 becomes dynamic): outputs now encode the *learning history*, and repeated
probing could reconstruct what the device has recently adapted to. Net:
state becomes **harder to reconstruct per-shot, easier to reconstruct
longitudinally**.

**How does state drift affect reconstruction?**
Drift is a moving target for the adversary's reconstruction simulator: a
posterior built at t is stale by t+Δ. Drift therefore acts as *natural noise*
that decays the value of past observations — a privacy benefit, and one PAC
can quantify (the leakage estimator would measure Σ growing over time).
Caveat: drift breaks DNOS determinism, so it can only live *behind* an NDAL
oracle that snapshots the drifted state, or determinism is abandoned for that
subsystem.

**Does continuous remapping improve privacy?**
Yes, materially. Continuously permuting/rotating the weight↔neuron mapping
(adaptive routing) is a form of **oblivious-RAM-like obfuscation**: it
randomizes the relationship between a physical readout and a logical feature,
defeating side-channel/extraction that assumes a fixed layout. It composes
well with PAC output noise. Cost: the remap schedule must itself be an NDAL
oracle draw to stay replayable.

**Do distributed attractor memories reduce extraction risk?**
Partially. Spreading memory across nodes means no single node holds a
reconstructable copy — extraction requires compromising a quorum (analogous
to the Consensus oracle's quorum, `pipeline.rs:157-159`). This is secret-
sharing-like and **reduces single-node extraction risk**. But attractor
dynamics are *content-addressable*: a partial/noisy query can complete to a
stored pattern, which is itself a reconstruction primitive. So distribution
helps against *node compromise* but the completion property *aids* a query-
based adversary. Net positive only if completion outputs are themselves run
through the Phase-3 perturbation layer.

**Design — privacy-preserving neuromorphic memory subsystem (sketch):**

```
  write:  pattern ─► [remap oracle draw] ─► spread across k nodes
                     (each node stores a noisy share; share noise = PAC Λ)
  read:   cue ─► attractor completion on quorum ─► [perturb completion]
                ─► budget-charged release
  drift:  device drift logged as periodic NDAL snapshot deltas
                (replayable; doubles as natural privacy decay)
```

This is **research-tier** on both axes; flagged Vision, not committed.

---

# Phase 5 — Implementation Roadmap

## 5.1 RFC summary

Add a host-side **PAC-Privacy crate** (`pacp/`, sibling to `ial/`, `ndal/`)
that (a) computes an offline `LeakageTable` from the trained weights using the
existing bit-exact simulator, and (b) perturbs the perturbable output channels
at release time, drawing noise from the NDAL Random oracle so replay/verify
stay exact while fresh observation is privatized. Keep the command (argmax)
channel exact by default; perturb the cursor-delta channel. Treat the replay
log as a separate, access-controlled secret (out of PAC's scope). A minimal
on-metal perturbation path (RDTSC + Q8.8 noise LUT) is optional and lands
with Tier 3.

## 5.2 Subsystem diagram (delivery view)

```
  pacp/ (new Rust crate)
   ├─ estimator.rs   ── offline: Σ + MI from simulate_assembly_forward
   ├─ simulator.rs   ── offline: adversary inverter (CI red-team gate)
   ├─ risk.rs        ── Λ + PrivacyBudget composition/charge
   ├─ perturb.rs     ── output noise via NDAL Random oracle
   ├─ history.rs     ── per-principal query tracker (reuse IAL StreamHash)
   └─ table.rs       ── LeakageTable (de)serialize, keyed by weights CRC32
  tools/
   └─ leakage_eval.py ── build-time Σ estimation (clone of context_eval.py)
  src/dnos.asm (Tier 3+)
   └─ optional perturb_output_32 (RDTSC entropy + Q8.8 inverse-CDF LUT)
```

## 5.3 Milestones

| M | Deliverable | Gate |
|---|---|---|
| **M0** | `tools/leakage_eval.py`: covariance/MI of outputs over resampled histories (extend `context_eval.py`). No noise yet — just *measure* L2/L3 leakage. | report Σ for cursor & command channels |
| **M1** | `pacp` crate: Estimator + Risk + Perturbation (host), command channel exact, cursor channel noised; noise via NDAL oracle. | replay bit-exact; live differs; unit tests mirror `ndal` style |
| **M2** | State Reconstruction Simulator + CI red-team gate (adversary advantage ≤ target at chosen budget). | new CI job beside `ci.yml` boot/diff tests |
| **M3** | `PrivacyBudget` config wired into `config_hash`; per-principal composition; fail-closed exhaustion. | budget exhaustion test |
| **M4** | Log-protection track (orthogonal): redact L9/L11 oracle entries, encrypt-at-rest, document T3↔T4 policy. | log no longer carries cleartext fingerprint/seed |
| **M5** *(opt, Tier 3)* | On-metal `perturb_output_32` (RDTSC + LUT), validated against host via the differential harness (`interactive_test.py`). | metal == host noised law under shared log |
| **M6** *(research)* | Neuromorphic privacy memory sketch → falsifiable prototype only if/when Phase 3 hardware exists. | n/a (Vision) |

## 5.4 Dependencies

- M0 → M1 (Σ feeds noise scale). M1 → M2 → M3. M4 parallel to M1–M3.
- M5 depends on Tier 3 entropy path + pager; **blocked** in Tier 2.
- Hard external dep: a real entropy source on metal for M5 (RDTSC/RDRAND).
- Soft dep: the prior model in the Estimator must track the real input
  distribution; reuse `train.py`'s `history_pool`.

## 5.5 Estimated complexity

| Component | Effort | Risk |
|---|---|---|
| M0 leakage_eval.py | S (≈1 day; it's `context_eval.py` + `np.cov`) | low |
| M1 pacp host crate | M (mirrors `ndal` size, ~800–1200 LoC) | low-med |
| M2 simulator + CI | M | med (defining "good enough" adversary) |
| M3 budget/config | S | low |
| M4 log protection | M | **med-high** (touches the determinism contract) |
| M5 on-metal perturb | L (asm + LUT + entropy + diff test) | high (Tier 2 lacks prereqs) |
| M6 neuromorphic | XL / research | speculative |

## 5.6 Key risks & honest caveats

1. **PAC protects outputs, not the log.** The replay log (L6) remains the
   CRITICAL surface and is *outside* PAC's noise model. Without M4, adding
   output noise yields a false sense of privacy. State this loudly.
2. **Privacy is conditional on log access.** §3.6's design means "private to
   anyone without the log." If logs are freely shared for verification (the
   stated distributed-DNOS goal), the privacy gain is only against pure
   screen-observers and peers who lack that node's log.
3. **Prior-distribution assumption.** PAC bounds hold only w.r.t. the modeled
   input prior. Adversaries with a better prior than the Estimator's see more.
4. **Tier-2 metal can't host the full layer.** Don't promise OS-level PAC in
   Tier 2; scope it to host now, metal-perturbation later.
5. **Determinism CI will fight you.** Any naive noise trips
   `find_divergence_with` and the differential test; the oracle-logged-noise
   design is mandatory, not optional.
6. **Cleartext weights (L5)** are an extraction risk PAC output-noise only
   *dampens* (noised real-valued labels raise query cost) — it does not
   prevent shipping the model in the image.

---

## Appendix — Evidence index

- Kernel forward pass & decode: `src/dnos.asm:892-1037`, `1044-1197`
- Recurrent history encode: `src/dnos.asm:855-884`
- No scheduler (timer only ticks): `src/dnos.asm:645-653`
- Cleartext weights in image: `src/dnos.asm:1586-1610`; `tools/build.py`
- Offline training / QAT-STE / bit-exact twin: `tools/train.py:366-382,
  408-491, 649-717`
- Context-robustness (history privacy win): `ROADMAP.md:115-124`;
  `tools/context_eval.py`
- IAL config dials (quantization = privacy): `ial/src/types.rs:313-362`
- IAL stream digest (confirmation oracle): `ial/src/pipeline.rs:182-205`
- NDAL replay log (CRITICAL surface): `ndal/src/log.rs:86-347`
- NDAL oracles incl. fingerprint/seed: `ndal/src/oracles.rs:41-280`
- NDAL rate-limit / clock-res (existing leak controls):
  `ndal/src/oracles.rs:155-162, 408-410`
- Snapshots (hash membership): `ndal/src/log.rs:65-79, 276-293`
```
