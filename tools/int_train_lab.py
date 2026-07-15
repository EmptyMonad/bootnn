#!/usr/bin/env python3
"""
Integer-exact training lab (ROADMAP item 9; decade-view research item).

THESIS: inference is portable by construction; training is not - float
BPTT ties the canonical CRC to a BLAS build, because float addition is
not associative and every BLAS orders its reductions differently
(TIER4 finding d records the resulting worry). Integer addition IS
associative, so a trainer whose every operation is integer is
order-insensitive - and therefore environment-free - BY CONSTRUCTION,
not by pinning.

This is a LAB (ssm_lab pattern): it graduates into train.py or dies
here. Design:

  - Training forward = an integer SHADOW of the law: same ops and
    signs, per-layer >>k truncations deferred (activations carry
    units law*2^sum(k), clamps scale up). The TRUE law is the
    selection metric and the saved artifact; forward() is gated
    bit-exact against SsmMachine.
  - The backward is int64 STE: clamp/ReLU masks from the integer
    forward; the recurrence backward is the decay spectrum's own
    shift-subtract (dh <- m*(dh - dh>>d)). Range-control shifts are
    PER-SAMPLE, before any batch reduction - floor-shift after a sum
    is not linear in the batch, and chunk-exactness is a gate.
  - No exp, no sqrt, no float anywhere: Weston-Watkins hinge (every
    violating rival contributes +/-1 - integer gradients need
    density), momentum as a leaky shift, sign-SGD steps of
    mean|w| >> lr_shift with an optional stepped decay schedule.
  - Data sampling uses xorshift64* over Python ints - no numpy
    Generator stream dependency. Training is a pure function of the
    seed, full stop.

Gates (--gate, cheap, no training, in CI):
  1. ORDER-INSENSITIVITY: one real gradient computation, batch
     permuted and accumulation chunked -> bit-identical integer
     gradients; float64's non-associativity shown alongside.
  2. FORWARD==LAW: the law-forward equals SsmMachine on a saved v5
     blob, outputs and h, sequence for sequence.

FINDINGS (2026-07-12, first campaign - the honest ledger):
  PROVEN: the order-freedom half of the thesis. Integer gradients are
  bit-identical under batch permutation and chunked accumulation; the
  BLAS-variance mechanism (reduction order) is absent by construction
  across the whole pipeline, RNG and data included.
  LEARNED ALONG THE WAY (each redesign forced by measurement):
  (a) training THROUGH the law's truncation grid cannot bootstrap -
  at init the random-ternary signal is below the quantization step
  (measured logits +/-1), hence the deferred-shift shadow; (b)
  integer LARS is a trap - floor division zeroes every sub-peak
  update (median |g| ~2^15 vs peaks ~2^40), silently freezing almost
  all weights; (c) post-sum shifts silently break batch linearity.
  FIRST CAMPAIGN'S DEAD END, RESOLVED BY CONTROL ARM (2026-07-12,
  second campaign, tools/int_train_control.py): the universal 15%
  plateau was NOT the optimizer - integer Adam-flavour (m=EMA(g),
  u=EMA|g|, per-weight (m*step)//u) and error feedback both hit the
  identical wall, while the float control (same harness, same hinge,
  same backward SHAPE) sailed to ~60%. Elimination localized the
  defect: range-control shifts sized to worst-case BOUNDS destroyed
  the gradient - random-sign cancellation makes typical magnitudes
  sqrt-scale (2^13..2^16), so >>15 twice left 2-5 bits of signal.
  Quantization noise is optimizer-invariant; that is exactly what the
  invariance of the plateau was saying.
  Re-sizing the shifts to typical magnitudes (dz2 unshifted, >>8
  elsewhere; int64 headroom verified) BROKE the wall: 15% -> 23.8%,
  loss monotone through the old floor. Second-order finding: a step
  proportional to live mean|w| climbs ~50% faster than an init-frozen
  step (norm growth is part of how the ternary law finds its scale)
  but feeds back into divergence after ~1k epochs; best-checkpoint
  selection preserves the peak through it.
  THIRD CAMPAIGN (2026-07-13): the precision lever was re-derived as
  stale (post-fix magnitudes through the >>8 hop are ~2^21 typical -
  healthy), so the second-moment and norm levers were run instead.
  RMS via integer Newton isqrt (int_isqrt, bit-exact, committed)
  produced the first sub-105 violation counts in the lab's history
  (82-91: margins actually being satisfied, the float control's
  signature) at 18.4%/800ep. Overflow trap #2 for the ledger:
  weight-level gradients reach ~2^36, and an unclipped (g>>4)^2
  wraps int64 - u goes negative, isqrt degenerates, weights explode
  into frozen saturation; clip-before-square is mandatory. Weight
  decay as a shift (2^-11/step) slows but does not stop the
  norm-feedback divergence (peak 21.8%, then decay; --wd-shift, off
  by default).
  FOURTH CAMPAIGN (2026-07-13) - STABILITY SOLVED, by homeostasis:
  the design think reframed the norm/step feedback correctly.
  sign-SGD is an entropy pump: every confident weight moves a full
  step even on pure noise, so the norm diffuses outward regardless
  of signal. A decay that scales with the norm (like the step does)
  gives a FIXED ratio of flows - no equilibrium exists, only a sign:
  runaway or collapse. The counterbalance must create a restoring
  force around a set-point.
  Open-loop OU (fixed step + derived wd=2*lr_shift): new best 25.9%,
  then COLLAPSE - RMS damping makes actual diffusion ratio*step <<
  step, so the nominal-step-sized decay overpowers it; norms fell
  ~10x below the frozen projection scale, the shadow left its
  frozen-k regime, signal died, pure decay death-spiraled (the norm
  trace in the logs exists because of this run).
  CLOSED-LOOP THERMOSTAT: decay fires only while a layer's live norm
  exceeds its freeze-time set-point. Collapse impossible by
  construction, runaway opposed. Measured: norms flat to ~0.1% for
  2500+ epochs, 26.5%. Step schedule stacked on top (previously dead
  code against the frozen step; wired via the base-shift delta):
  29.9%, with late-run accuracy SUSTAINED at 25-27% and loss below
  every previous floor - refinement inside stability, the float
  control's behavior.
  FIFTH CAMPAIGN (2026-07-13) - WIDTH TRANSFER FAILS; two theories
  falsified; the negative ledger:
  h=512 with the winning h=128 recipe froze shifts at [1,0,0,0],
  peaked 15% by ep500, and decayed to chance by ep1000 with the
  thermostat holding norms perfectly flat - stability transferred,
  learning did not (run killed at ep1000; the set-point mechanism is
  width-independent, the failure is elsewhere). THEORY 1 (k=0 is a
  degenerate hard-clip regime) FALSIFIED: the 29.9% h=128 run itself
  freezes at [3,0,0,0] - reproduced bit-exactly post-revert, ks and
  all. THEORY 2 (cold-start second moment explodes warmup norms;
  cap the update as integer bias correction) produced real evidence
  (freeze-0 norms grew 100-400x past set-points) but the cap CHANGED
  THE WARMUP TRAJECTORY into a worse frozen regime (h=128
  freeze-200: 29.9% -> 9.5%) and was reverted. Freeze-at-0 fails at
  h=128 both ways (16.3% uncapped, 9.5% capped): the warmup
  inflation is not a bug to remove - the system NEEDS to find its
  equilibrium scale before k and the set-points lock.
  SIXTH CAMPAIGN (2026-07-13) - INSTRUMENTED; the fork is found.
  --trace lands (per-layer mean/alpha/k/density/gmed/gmax, per-decay
  h-saturation, and projection SIGN-CHURN per mille). Theory 3
  (clamp saturation kills long decay horizons at k_in=1) was
  falsified in one 4-minute measurement: saturation is 0 at every
  decay block, both widths. The warmup trajectories are
  near-identical across widths; the failure is post-freeze, and the
  discriminator is LOSS DIRECTION: through epochs 300-1000, h=128's
  loss falls 21M->6.3M with churn settling at 20-50 per mille, while
  h=512's loss RISES 216M->891M with violations flat, norms pinned,
  and churn LOWER than the healthy run (4-20 per mille - the law is
  nearly frozen). Reading: at width, STE's identity assumption
  breaks - surviving shadow weights sit ~30x the projection unit, so
  descent moves magnitudes (law-invariant) instead of crossing sign
  thresholds; absolute flip counts are BELOW the healthy run despite
  4x the weights, and the rare monster-gradient flips drift the
  realized loss upward. Shadow descent decouples from law
  improvement: LAW-MOTION STARVATION, measured.
  SEVENTH CAMPAIGN (2026-07-13) - CHURN HOMEOSTASIS RUN AND
  FALSIFIED; the mechanism sharpens. Predictions were written first:
  P1 (h=512: servo lifts churn into band, loss direction flips, best
  > 15%) FAILED on all three - churn kept sinking (L2 to 6-8 per
  mille) while the servo drove steps up 16x, loss still ascends,
  best 14.3%. P2 (h=128 non-regression) partially failed: 27.2% vs
  29.9% (healthy churn grazes the band's lower edge late in runs, so
  the servo perturbs steps it should leave alone).
  WHY THE ACTUATOR FAILED IS THE FINDING: gmed=0 on the readout
  layers means most weights receive EXACTLY ZERO gradient (integer
  cancellation) - the servo multiplies the momentum of a zero.
  Law-motion starvation is GRADIENT-DENSITY-limited, not
  step-limited; step size was the wrong actuator variable. The
  suspect with a measured trail: the per-sample >>8 range shifts
  floor small backward values to exact zero, and h=512's
  cancellation profile floors more of them.
  EIGHTH CAMPAIGN (2026-07-13) - THE CAUSAL CHAIN, MEASURED END TO
  END, AND THE FIRST INTERVENTION THAT WORKS. Density measured
  (gnnz in --trace): h=512 mid-readouts carry ~half of h=128's
  nonzero-gradient density and it DECLINES (L1 102->63 vs steady
  ~174), rank-correlating exactly with the churn measurements.
  Liveness measured (relu-mask trace): the source is STRUCTURAL -
  h=512 ReLU liveness collapses to 2% (24/21 per mille, falling) vs
  h=128's stable 12-17%; the >>8 floor is exonerated as primary.
  The chain: width -> faster L0 warmup (norms 4.2x) -> k_in freezes
  1 vs 3 -> every downstream shadow clamp (32767 << (k_in+...))
  narrows 4x while fan-in sums grow 2x -> liveness collapses 5x ->
  masks zero the gradients -> the law starves. (Theory 3 checked h
  saturation and was rightly falsified; the same clamp mechanism
  operates one level up, at a1/a2, where it wasn't measured.)
  INTERVENTION (--kin-floor, default 3): floor the frozen k_in at
  the healthy operating point - one lawful line. Predictions written
  first, three for three: liveness recovered (m1 166 vs 24; m2 half-
  recovered at 56-84), loss direction FLIPPED (249M -> 134M
  descending; was 216M -> 891M ascending), and the 15% width ceiling
  broke: best 17.0%, monotone through the window.
  NINTH CAMPAIGN (2026-07-13, walk batch): the k1 floor
  (--k1-floor, default 0 - narrow runs untouched, 29.9% reproduced
  bit-exactly as non-regression) compensates fan-in clamp geometry
  one layer up: +1 shift for the 4x-terms/2x-sqrt-scale a2 sums.
  Predictions hit again: m2 liveness fully recovered AND RISING
  (150 -> 187 per mille through the window, above h=128's own ~173;
  units coming alive over time instead of dying) and a new width
  record, 18.4% in the 1001-epoch window.
  TENTH CAMPAIGN (2026-07-13, long horizon) - THE RATE GAP CLOSES,
  AND A METRIC ILLUSION FALLS. 10k epochs at h=512 (both floors;
  interrupted once at ep4600 by a machine shutdown - determinism
  replayed it bit-exactly, and --save now checkpoints on every new
  best so interruption costs only time): lab-metric best 81.0%,
  loss 282k (from 891M in the failure era), violations 9/147. The
  float control (~60%) is decisively surpassed.
  THE ILLUSION: the artifact scored 4.8% under the LAW's decision
  rule. The lab had been argmaxing RAW logits; the kernel argmaxes
  the piecewise-sigmoid outputs, window (-8192, 8192] - and hinge
  loss constrains only differences, so the whole logit distribution
  drifted to ~-531k, UNDER the window: the law output all-zeros,
  argmax collapsed to class 0. Ranking knowledge was intact (raw
  argmax 75.5% on the artifact); the scale was lawless.
  THE ONE-BYTE LAWFUL FIX: k3 is a header shift and arithmetic shift
  is monotone - k3+7 compresses the distribution into the window
  with ranking preserved: 74.8% under the law's rule, and 64.6%
  (678/1050) on the CANONICAL GAUNTLET - the first integer-trained
  artifact to post a real score on the repo's common ground
  (canonical float laws: 99.3%; CI gate: 95). accuracy() now applies
  the law's rule so the lab can never report a lawless number again.
  NEXT: output-scale calibration at save time (deterministic: pick
  k3 from the val logit distribution) or a loss term anchoring
  logits in-window; then the 64.6 -> 95 gap (full task incl. words,
  longer horizons) - engineering now, not mystery.
  ELEVENTH CAMPAIGN (2026-07-14) - ALIGNMENT. Deterministic output
  calibration lands: calib_delta = max(0, bl(max|acc|) - 13), applied
  identically in accuracy() (evaluate as-saved) and save() (written
  into the header k3) - the rule reproduces the empirical sweep's
  plateau (gives 9; plateau was 7-10 at equal accuracy). The
  artifact==metric alignment check then caught a SECOND illusion:
  the checkpoint copied post-step weights, one update past the law
  that earned the score (29.9% reported, 24.5% on disk; a flat
  k3-sweep exonerated calibration and convicted the loop order).
  Eval and checkpoint now precede the step. Verified exact: reported
  best == on-disk artifact == law rule, 29.9% at h=128. Corollary:
  every artifact saved before this fix was one step stale - the
  64.6% canonical score of the 10k h=512 run is a LOWER BOUND;
  an aligned rerun is queued (flag-first, ~4h).
  Prior full-length note (3000 epochs, both floors): WIDTH PAYS.
  Loss monotone 141M -> 58M across the whole run - no divergence, no
  collapse, the homeostat and floors holding end to end - and best
  climbed 17.7 -> 18.4 -> 21.1 -> 27.9 -> 33.3%, STILL RISING at
  epoch 3000. h=512 beats the h=128 state of the art (29.9%) for the
  first time in nine campaigns: NEW STATE OF THE ART 33.3%. The
  trajectory has not plateaued - a longer horizon is the obvious
  next spend (flag-first; ~40 min per 3000 epochs at this width).
  The >>8 residue hypothesis stays in line behind that.
  TWELFTH CAMPAIGN (2026-07-14) - FULL TASK OPENS, AND PROVENANCE
  BITES. Procedural finding first: the aligned 10k rerun was
  launched from argparse defaults and silently diverged (froze
  [4,4,0,0] where the tenth's artifact header says [3,1,0,0]) - the
  winning recipe lived only in a session transcript, nowhere in the
  repo. Recovered by grepping the transcript archive, ~40 min of
  compute lost, and the recipe is now pinned below: FLAGS ARE
  PROVENANCE. The relaunched rerun then reproduced the tenth
  campaign's ep200 freeze line character for character ([3,1,0,0],
  OU steps, decay shift, set-point norms) - warmup replay bit-exact,
  the alignment code proven observational on the real recipe.
  Scientific finding: gen_data(full=True) extends the lab to the
  canonical task (doubled keys, WORDS in honest typing order, demo
  prefixes) as a PURE SUPERSET - the additions are static, consume
  no rng draws, so every scoped campaign stays bit-reproducible.
  h=128 probe (winning recipe + --task full, 3000 epochs, 200 s):
  full-task val best 33.3%, ABOVE the scoped 29.9% baseline despite
  30 harder rows in the metric; the artifact's static subset jumped
  9/30 -> 20/30, words 2/20 -> 12/20 (the h=512 canonical-gauntlet
  artifact scores 0/20 on words). WORDS ARE INTEGER-LEARNABLE and
  the sequence machinery holds. Cost at this width: canonical
  gauntlet 19.0% -> 16.5% (both h=128 artifacts sit far below the
  h=512 artifact's 64.6% - the gauntlet rewards width; h=128 is a
  directional instrument, not a contender). NEXT: full-task at
  h=512 once the aligned rerun lands - the 64.6 -> 95 attempt
  proper (flag-first; measured pace 1.5 s/epoch at width with the
  recipe's ctx 4 - the ~4 h/10k estimate holds; the false start's
  apparent 3.9 s/epoch was the doubled ctx-8 batch, not the machine).
  THIRTEENTH CAMPAIGN (2026-07-14) - ALIGNMENT PAYS, MEASURED. The
  aligned 10k h=512 rerun (winning recipe verbatim, 14461 s = 4.0 h)
  replayed the tenth campaign BIT-EXACTLY end to end: freeze line
  character for character, ep4600/5000/9999 loss and violations
  identical to the digit (281996/9 at the end), norms offset by
  exactly one optimizer step - which is the eval-before-step change
  displaying earlier, i.e. the alignment fix confirming itself.
  Predictions three for three: best under the LAWFUL metric 80.3%
  (vs 81.0 raw / 74.8 post-hoc-lawful on the stale artifact);
  invariant verified end to end at width - the saved artifact
  reproduces 80.3% exactly on the val set through SsmMachine; and
  the CANONICAL GAUNTLET: 65.5% (688/1050), +10 keys over the
  tenth's 64.6% lower bound. The one-step-stale correction plus
  per-epoch lawful selection is worth ~1 point for free. NEW
  INTEGER-TRAINING STATE OF THE ART on common ground (float canon
  99.3, mint gate 95). Words 1/20 as expected (scoped task) - the
  remaining gap is DATA, not alignment: full-task h=512 is the
  named next spend.
  FOURTEENTH CAMPAIGN (2026-07-14) - FULL TASK AT WIDTH: WORDS BUY
  IN, THE COMPOSITE PAYS. 10k full-task h=512 (winning recipe +
  --task full, 24666 s; 2.5 s/epoch - the 135-row batch costs more
  than its row ratio). Freeze [3,1,0,0] as at scoped width (floors
  bind; set-points lawfully differ under the changed warmup).
  Homeostat held end to end; loss fell 179M -> 8.5M and was STILL
  falling at ep9999 - the run ended training, not converged.
  Predictions: stability HIT; words at width: static 21/30 (words
  11/20, demo 5/5, doubled 5/5) - the h=128 probe's level, no width
  bonus on words at this budget; gauntlet MISSED: 38.1% (400/1050)
  vs the scoped artifact's 65.5% - at width and this budget the
  full task TRADES scoped skill for word skill rather than adding
  it (h=128 predicted ~2.5 points of cost; width paid 27).
  Invariant verified again: the artifact reproduces its 53.7% lab
  best exactly through SsmMachine.
  Reading: interference NOT proven - this is an unconverged run.
  The scoped run's leap came in its back half (45.6 -> 80.3 after
  ep5000); this one hit 53.7 with loss still halving per ~2500
  epochs at the end. Candidate next spends, cheapest first:
  (a) h=128 balance probes (static-row weighting, resample cadence)
  - minutes each; (b) longer horizon at width (20k, ~14 h);
  (c) parked in-window loss anchoring. Stop-rule: one width run of
  this shape down.
  HORIZON PROBE (2026-07-14, same day): the discriminator run,
  predictions written first. Full-task h=128 at 10k epochs (563 s):
  val best 33.3 -> 42.4%, static 20/30 -> 26/30 (words 17/20), and
  the canonical gauntlet 16.5 -> 20.9% - ABOVE the scoped baseline's
  19.0%. UNCONVERGED confirmed, interference rejected at this width:
  a longer horizon buys back the scoped cost AND the words, both
  still rising at ep10k. This licenses the width horizon (20k
  full-task h=512; determinism makes its first 10k a free bit-exact
  replay of the fourteenth run, so the spend is +10k new epochs,
  ~13.7 h wall).
  FIFTEENTH CAMPAIGN (2026-07-15) - THE BACK HALF PAYS AT WIDTH;
  THE FULL TASK'S CORE IS PERFECT. 20k full-task h=512 (fourteenth
  verbatim + --epochs 20000, 52691 s): the first 10k epochs
  replayed the fourteenth campaign bit-exactly (ep5000/9900/9950
  loss+violations to the digit), then the back half delivered the
  h=128 pattern at width: val best 53.7 -> 84.7% (ABOVE the scoped
  run's 80.3), loss 8.5M -> 556k, still improving at ep19999.
  The artifact: STATIC SUBSET 30/30 - words 20/20, doubled 5/5,
  demo 5/5, the first integer-trained artifact to fully learn the
  canonical task's core - and canonical gauntlet 63.5% (667/1050),
  up 25 points from the fourteenth artifact and within 2 of the
  scoped 65.5% while knowing everything the scoped artifact does
  not. Invariant verified: 84.7% exact through SsmMachine. Reading:
  horizon closed the trade almost completely and had not plateaued;
  the remaining gauntlet gap to the scoped artifact is 2 points
  against a +47-point static gain. Next levers, cheapest-informative
  first: ctx volume at width (the gauntlet is context
  generalization and the lab trains 4 ctx/key vs the float canon's
  richer stream), a third horizon doubling (40k, ~27 h), or
  graduation work. The 95 mint gate stands 31.5 points away.
  PORTABILITY (2026-07-15, same day): the 20k artifact (whole-file
  CRC 0xd28ac8cb) went through the portability harness on BOTH
  hardware profiles: trajectory digest 0x6581d2a9 on qemu-tcg-x86
  and on qemu-tcg-riscv32-virt, each equal to the simulator's
  prediction - FAITHFUL twice. The first INTEGER-TRAINED law
  verified law==metal on two ISAs: training order-free by
  construction, inference substrate-free by measurement - the
  environment-free loop closes end to end. (The 95 gate is the
  quality bar for TRAINING claims; portability's own verdict is
  honesty + faithfulness, both met.)

Probes:
  python tools/int_train_lab.py --epochs 400 --h 128 --ctx 4
  python tools/int_train_lab.py --gate                          # CI

THE WINNING RECIPE (pinned here because it once lived only in a session
transcript and a rerun launched from argparse defaults silently diverged
- froze [4,4,0,0] instead of [3,1,0,0]; the flags ARE part of the law's
provenance):
  h=128 (29.9%):  --epochs 3000 --h 128 --ctx 4 --lr-shift 5 --opt adam
                  --decay-every 800
  h=512 (tenth):  --epochs 10000 --h 512 --r1 1024 --r2 384 --ctx 4
                  --lr-shift 5 --opt adam --decay-every 800 --k1-floor 1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from ssm_lab import HISTORY_POOL, SINGLE_KEYS, WORDS  # noqa: E402
from train import CMD, SsmMachine, save_v5  # noqa: E402

F = 24                      # fixed-point fractional bits (Q24)
ONE = 1 << F
SEQ_LEN = 64
OUT = 64
N_CMD = 20
MARGIN = 1024               # hinge margin, in pre-sigmoid acc units
MU_S = 3                    # momentum = 1 - 2^-3 = 0.875, as a shift
A_S1 = 3                    # adam: first-moment EMA shift (beta1 ~ 0.875)
A_S2 = 8                    # adam: 2nd-moment EMA shift (beta2 ~ 0.996)
A_GQ = 4                    # pre-shift before squaring (int64 headroom)

FREEZE = 200                # epochs before per-layer shifts freeze
CH_LO, CH_HI = 20, 60       # healthy sign-churn band, per mille (measured)
CH_T = 25                   # churn servo cadence, epochs
ADJ_MIN, ADJ_MAX = -4, 6    # per-layer step adjustment bounds (shifts)
MASK64 = (1 << 64) - 1


class XorShift:
    """xorshift64* over Python ints: the lab's only randomness.
    Bit-identical on every platform and numpy version by construction."""

    def __init__(self, seed):
        self.s = (seed & MASK64) or 88172645463325252

    def next(self):
        x = self.s
        x = (x ^ (x << 13)) & MASK64
        x ^= x >> 7
        x = (x ^ (x << 17)) & MASK64
        self.s = x
        return (x * 0x2545F4914F6CDD1D & MASK64) >> 32

    def below(self, n):
        return self.next() % n

    def signed_below(self, n):
        return self.below(2 * n + 1) - n


def gen_data(rng, ctx_per_key=8, full=False):
    """Single-key commands, bare and under random full-alphabet
    histories (honest typing order) - the scoped probe task.
    full=True appends the REST of the canonical task, mirroring
    ssm_lab.gen_sequences content exactly: doubled keys, WORDS in
    honest typing order, and the demo's typed prefixes. The
    additions are static - they consume no rng draws, so the scoped
    stream (and every prior campaign's trajectory) is untouched."""
    data = [([ord(k)], CMD[cmd]) for k, cmd in SINGLE_KEYS]
    for k, cmd in SINGLE_KEYS:
        for _ in range(ctx_per_key):
            hist = [HISTORY_POOL[rng.below(len(HISTORY_POOL))]
                    for _ in range(1 + rng.below(SEQ_LEN - 1))]
            data.append((hist + [ord(k)], CMD[cmd]))
    if full:
        for k in 'bplvc':
            data.append(([ord(k)] * 2, CMD[dict(SINGLE_KEYS)[k]]))
        for word, cmd in WORDS:
            data.append(([ord(c) for c in word], CMD[cmd]))
        for i, k in enumerate('pboxline'):        # demo, as typed
            expect = {'p': 'pixel', 'b': 'rect', 'x': 'rect',
                      'l': 'hline', 'e': 'hline'}.get(k)
            if expect:
                data.append(([ord(c) for c in 'pboxline'[:i + 1]],
                             CMD[expect]))
    return data


def encode(data):
    B = len(data)
    X = np.zeros((B, SEQ_LEN, 8), dtype=np.int64)
    cls = np.zeros(B, dtype=np.int64)
    for b, (keys, c) in enumerate(data):
        keys = keys[-SEQ_LEN:]
        for t, k in enumerate(keys):
            X[b, SEQ_LEN - len(keys) + t] = \
                [(k >> i) & 1 for i in range(8)]
        cls[b] = c
    return X * 256, cls


def int_isqrt(v):
    """Vectorized integer square root: binary bit-length seed + Newton
    iterations, int64 ops and exact integer division only - bit-exact
    on every platform. Converges quadratically; 6 iterations cover
    64-bit inputs."""
    bl = np.zeros_like(v)
    t = np.maximum(v, 0)
    for s in (32, 16, 8, 4, 2, 1):
        big = t >= (np.int64(1) << s)
        bl += np.where(big, s, 0)
        t = np.where(big, t >> s, t)
    bl += (t > 0)
    x = np.int64(1) << ((bl + 1) >> 1)
    for _ in range(6):
        x = (x + v // np.maximum(x, 1)) >> 1
    return np.maximum(x, 1)


def _round_log2(x):
    """round(log2(x)) for a positive int, exactly: bit_length gives
    floor; round up iff x^2 >= 2^(2b+1) (i.e. x >= 2^b * sqrt(2))."""
    b = int(x).bit_length() - 1
    return b + (1 if int(x) * int(x) >= 1 << (2 * b + 1) else 0)


def int_project(w, frozen_k=None):
    """ternary_project, integer-exact: delta = 3*mean|w|/4 (shifts),
    k = clamp(F - round(log2(mean surviving |w|)), 0, 15)."""
    aw = np.abs(w)
    n = w.size
    delta = (3 * int(aw.sum()) // n) >> 2
    s = np.sign(w) * (aw > delta)
    if frozen_k is not None:
        return s.astype(np.int64), frozen_k
    nz = aw[s != 0]
    if nz.size == 0:
        return s.astype(np.int64), 0
    alpha = int(nz.sum()) // int(nz.size)
    k = min(max(F - _round_log2(max(alpha, 1)), 0), 15)
    return s.astype(np.int64), k


class IntLab:
    def __init__(self, h=512, r1=1024, r2=384, seed=1337):
        rng = XorShift(seed)
        self.H, self.r1, self.r2 = h, r1, r2
        self.d = np.repeat(np.arange(1, 9), h // 8).astype(np.int64)
        # Shadow init: uniform integers, scale a power of two per
        # fan-in (a defined constant of the algorithm; the sqrt(2/fan)
        # ideal is approximated by the nearest shift).
        def init(fi, fo):
            a = ONE >> ((fi.bit_length() + 1) // 2 + 1)
            return np.array([[rng.signed_below(a) for _ in range(fo)]
                             for _ in range(fi)], dtype=np.int64)
        self.w = [init(8, h), init(h, r1), init(r1, r2), init(r2, OUT)]
        self.v = [np.zeros_like(w) for w in self.w]
        self.u = [np.zeros_like(w) for w in self.w]
        self.r = [np.zeros_like(w) for w in self.w]   # error feedback
        self.step0 = [int(np.abs(w).sum()) // w.size for w in self.w]
        self.frozen = None
        self.opt = "sign"
        self.wd = 0
        self.step_f = None
        self.norm_f = None
        self.lr0 = 0
        self.freeze_at = FREEZE
        self.kin_floor = 3
        self.k1_floor = 0
        self.save_path = None
        self.full_task = False
        self.calib = None
        self.trace = 0
        self.churn_servo = False
        self.step_adj = [0, 0, 0, 0]
        self._servo_prev = None
        self.wd_f = 0

    def project(self):
        outs = [int_project(w, None if self.frozen is None
                            else self.frozen[i])
                for i, w in enumerate(self.w)]
        return [o[0] for o in outs], [o[1] for o in outs]

    # ── the integer SHADOW forward (training) ──────────────────────
    # Same ops and signs as the law, but the per-layer >>k truncations
    # are DEFERRED: activations carry units law*2^(sum of k so far)
    # and clamps scale up. At init the law's own grid crushes the
    # random-ternary signal to zero (measured: acc range +/-1), so
    # gradients cannot bootstrap; the shadow keeps full integer
    # precision - still order-free - while checkpoint selection and
    # the saved artifact use the TRUE law (ssm_lab's shadow/law split,
    # integerized).
    def forward_train(self, X, signs, ks):
        s_in, s1, s2, s3 = signs
        k_in, k1, k2, k3 = ks
        B = X.shape[0]
        h = np.zeros((B, self.H), dtype=np.int64)
        hc = 32767 << k_in
        masks = np.empty((B, SEQ_LEN, self.H), dtype=bool)
        for t in range(SEQ_LEN):
            pre = h - (h >> self.d) + (X[:, t] @ s_in)
            masks[:, t] = (pre > -hc) & (pre < hc)
            h = np.clip(pre, -hc, hc)
        c1 = 32767 << (k_in + k1)
        a1 = h @ s1
        z1 = np.clip(a1, 0, c1)
        m1 = (a1 > 0) & (a1 < c1)
        c2 = 32767 << (k_in + k1 + k2)
        a2 = z1 @ s2
        z2 = np.clip(a2, 0, c2)
        m2 = (a2 > 0) & (a2 < c2)
        acc = z2 @ s3          # units: law << (k_in+k1+k2+k3)
        return acc, (h, z1, m1, z2, m2, masks)

    # ── the law, batched (bit-exact SsmMachine.step) ────────────────
    def forward(self, X, signs, ks):
        s_in, s1, s2, s3 = signs
        k_in, k1, k2, k3 = ks
        B = X.shape[0]
        h = np.zeros((B, self.H), dtype=np.int64)
        masks = np.empty((B, SEQ_LEN, self.H), dtype=bool)
        for t in range(SEQ_LEN):
            pre = h - (h >> self.d) + ((X[:, t] @ s_in) >> k_in)
            masks[:, t] = (pre > -32768) & (pre < 32767)
            h = np.clip(pre, -32768, 32767)
        a1 = (h @ s1) >> k1
        z1 = np.clip(a1, 0, 32767)
        m1 = (a1 > 0) & (a1 < 32767)
        a2 = (z1 @ s2) >> k2
        z2 = np.clip(a2, 0, 32767)
        m2 = (a2 > 0) & (a2 < 32767)
        acc = (z2 @ s3) >> k3
        return acc, (h, z1, m1, z2, m2, masks)

    def grads(self, X, cls, signs, ks):
        """Hinge top, integer STE backward through the shadow forward.
        Every reduction is an int64 sum: associative, hence order-free.
        The inter-layer >>15 shifts are range control (activations are
        ~15-bit), a DEFINED semantic; absolute gradient scale is
        irrelevant because step() normalizes per layer."""
        acc, (h, z1, m1, z2, m2, masks) = \
            self.forward_train(X, signs, ks)
        B = X.shape[0]
        k = ks[0] + ks[1] + ks[2] + ks[3]
        logits = acc[:, :N_CMD]
        # Weston-Watkins hinge: EVERY rival within the margin of the
        # true class contributes +/-1 - denser than max-rival hinge
        # (integer gradients need density; they cannot whisper).
        zy = logits[np.arange(B), cls]
        viol = (zy[:, None] - logits) < (MARGIN << k)
        viol[np.arange(B), cls] = False
        g = np.zeros((B, OUT), dtype=np.int64)
        g[:, :N_CMD] = viol << F
        g[np.arange(B), cls] = -(viol.sum(axis=1) << F)
        n_wrong = int(viol.any(axis=1).sum())
        loss = int((np.maximum(
            (MARGIN << k) - (zy[:, None] - logits), 0)
            * viol).sum()) >> k
        # Range-control shifts are PER-SAMPLE (before any batch
        # reduction; post-sum floor-shift breaks chunk linearity - a
        # gate) and sized to TYPICAL magnitudes, not worst-case
        # bounds: random-sign cancellation makes typical values
        # sqrt-scale, and over-shifting (>>15 everywhere) was measured
        # to leave 2-5 bits of gradient - quantization noise that no
        # optimizer could rescue. int64 headroom holds at these:
        d3 = z2.T @ (g >> 8)
        dz2 = (g @ signs[3].T) * m2
        d2 = z1.T @ dz2
        dz1 = ((dz2 @ signs[2].T) >> 8) * m1
        d1 = h.T @ dz1
        dh = dz1 @ signs[1].T
        # BPTT: the recurrence backward is the decay spectrum itself.
        d_in = np.zeros_like(self.w[0])
        for t in range(SEQ_LEN - 1, -1, -1):
            dh = dh * masks[:, t]
            d_in += (X[:, t] >> 8).T @ dh
            dh = dh - (dh >> self.d)
        return [d_in, d1, d2, d3], acc, n_wrong, loss

    def step(self, grads, lr_shift):
        """sign-SGD with shift-momentum: every weight whose momentum
        buffer carries ANY accumulated signal moves by exactly
        mean|w| >> lr_shift. No division, no normalization - integer
        LARS was measured to floor sub-peak updates to zero (median
        |g| ~2^15 against peaks ~2^40), silently freezing almost every
        weight. Sign updates are dense and scale-free by construction."""
        for w, v, g in zip(self.w, self.v, grads):
            v -= (v >> MU_S)
            v += g
            step_q = max((int(np.abs(w).sum()) // w.size) >> lr_shift, 1)
            w -= np.sign(v) * step_q

    def step_adam(self, grads, lr_shift):
        """Integer Adam-flavour: per-weight adaptivity without squares
        or roots (g^2 overflows int64 at our scales; Adam's actual
        mechanism is sign-consistency scaling). m = EMA(g), u =
        EMA(|g|), both leaky shifts with the (1-beta) factor included
        so each is a true average; |m| <= u by construction, so the
        per-weight update sign(m) * (|m| * step) // u is bounded by
        step = mean|w| >> lr_shift. A weight with a consistent
        gradient direction moves the full step; a noisy one barely
        moves. Exact integer division, sign-symmetric (no floor bias),
        deterministic."""
        for i, (w, m, u, r, g) in enumerate(
                zip(self.w, self.v, self.u, self.r, grads)):
            m -= m >> A_S1
            m += g >> A_S1
            # True RMS second moment (integer Newton isqrt): EMA|g|
            # under-damps spiky weights. Pre-shift AND clip before
            # squaring: weight-level gradients reach ~2^36, and an
            # unclipped square wraps int64 (measured: u went negative,
            # isqrt degenerated, weights exploded to saturation). A
            # clipped monster just saturates its own damping.
            gq = np.clip(g >> A_GQ, -(1 << 28), 1 << 28)
            u -= u >> A_S2
            u += (gq * gq) >> A_S2
            # Before the freeze: step tracks the live mean|w| (climbs
            # ~50% faster; norm growth is how the ternary law finds
            # its scale). After: the OU regime - fixed step (set at
            # freeze) + restoring decay below.
            if self.step_f is not None:
                # The decay schedule applies to the frozen step too
                # (lr_shift arrives schedule-incremented), and the
                # churn servo's per-layer adjustment composes with it.
                # A negative total shift means the servo grew the step.
                sh = (lr_shift - self.lr0) + self.step_adj[i]
                step_q = max(self.step_f[i] >> sh if sh >= 0
                             else self.step_f[i] << -sh, 1)
            else:
                step_q = max((int(np.abs(w).sum()) // w.size)
                             >> lr_shift, 1)
            # ERROR FEEDBACK: the desired update is computed in fine
            # units (8 extra fractional bits) and accumulated in a
            # residual; whole quanta are emitted, the remainder is
            # never lost. This is float's fractional accumulation,
            # exactly, in integers - without it, sub-quantum updates
            # vanish and most weights starve (measured: the float
            # control converges on this same harness, the quantized
            # step does not).
            denom = int_isqrt(u) << A_GQ
            # NOTE (fifth campaign): a trust-region cap on this update
            # (min with step_q<<8) was tried to tame the cold-start
            # second moment and REVERTED - it changes the warmup
            # trajectory enough to freeze k in a different, worse
            # regime (h=128 freeze-200 degraded 29.9% -> 9.5% with the
            # cap). The warmup dynamics that select the frozen regime
            # are not yet understood; instrument before re-theorizing.
            r += np.sign(m) * (((np.abs(m) * step_q) << 8)
                               // np.maximum(denom, 1))
            emit = np.sign(r) * (np.abs(r) >> 8)
            r -= emit << 8
            w -= emit
            # Restoring drift as a CLOSED-LOOP thermostat: decay fires
            # only while the layer's live norm exceeds its freeze-time
            # set-point. Open-loop balance (decay always on) was
            # measured to collapse layer norms ~10x below the frozen
            # projection scale - RMS damping makes actual diffusion
            # ratio*step << step, so a decay sized to the nominal step
            # overpowers it, the shadow leaves its frozen-k regime,
            # signal dies, and pure decay death-spirals. Below the
            # set-point the pump switches off: collapse is impossible
            # by construction; above it, runaway is opposed.
            wd = self.wd or (self.wd_f if self.step_f is not None else 0)
            if wd and self.step_f is not None and \
               int(np.abs(w).sum()) // w.size > self.norm_f[i]:
                w -= np.sign(w) * (np.abs(w) >> wd)

    def trace_line(self, ep, grads, sat=None):
        """Warmup instrumentation (fifth campaign's instruction: the
        dynamics that select the frozen regime are decisive - measure
        them instead of theorizing). One line per layer, all integers,
        machine-diffable across widths. `sat` = per-decay-block h
        clamp-saturation per mille (which memory horizons are alive)."""
        if sat is not None:
            print(f"  [trace] ep {ep:4d} h-saturation per mille by "
                  f"decay d=1..8: {sat}", flush=True)
        # Projection sign-churn: how much of the TERNARY LAW flipped
        # since the last trace - the law the val metric runs on can
        # thrash while the shadow drifts smoothly.
        signs_now = [int_project(w)[0] for w in self.w]
        if getattr(self, "_prev_signs", None) is not None:
            churn = [int((s != p).sum()) * 1000 // s.size
                     for s, p in zip(signs_now, self._prev_signs)]
            print(f"  [trace] ep {ep:4d} sign-churn per mille since "
                  f"last trace: {churn}", flush=True)
        self._prev_signs = signs_now
        for i, (w, g) in enumerate(zip(self.w, grads)):
            aw = np.abs(w)
            mean = int(aw.sum()) // aw.size
            delta = (3 * mean) >> 2
            nz = aw[aw > delta]
            alpha = int(nz.sum()) // max(int(nz.size), 1)
            k = min(max(F - _round_log2(max(alpha, 1)), 0), 15)
            ag = np.abs(g)
            print(f"  [trace] ep {ep:4d} L{i}: mean={mean:>10} "
                  f"alpha={alpha:>10} k={k:2d} "
                  f"dens={int(nz.size) * 1000 // aw.size:4d} "
                  f"gnnz={int((ag > 0).sum()) * 1000 // ag.size:4d} "
                  f"gmed={int(np.median(ag)):>12} gmax={int(ag.max()):>15}",
                  flush=True)

    @staticmethod
    def calib_delta(acc):
        """Deterministic output-scale calibration: the smallest k3
        bump that brings max|acc| inside the sigmoid window 2^13.
        Derived from the tenth campaign's underflow (logits at
        ~-531k -> delta 7, matching the empirical sweep exactly)."""
        m = int(np.abs(acc[:, :N_CMD]).max())
        return max(0, m.bit_length() - 13)

    def accuracy(self, X, cls, signs, ks):
        # THE LAW'S decision rule, not the shadow's: the kernel
        # argmaxes the piecewise-sigmoid OUTPUTS, window (-8192, 8192]
        # in acc units (argmaxing raw logits reported 81% on an
        # artifact whose lawful score was 4.8% - tenth campaign).
        # Evaluated AS SAVED: the same calibration save() writes into
        # the header is applied here, so the training metric and the
        # artifact's lawful score cannot diverge again.
        acc, _ = self.forward(X, signs, ks)
        acc = acc >> self.calib_delta(acc)
        out = np.where(acc < -8192, 0,
                       np.where(acc > 8192, 32767, (acc + 8192) * 2))
        return float((out[:, :N_CMD].argmax(axis=1) == cls).mean())

    def save(self, path):
        signs, ks = self.project()
        ks = [int(k) for k in ks]
        # Output-scale calibration into the header: k3 is a shift and
        # shifts are monotone, so ranking is untouched while the
        # logits land in the sigmoid window. Calibrated on self.calib
        # (the val set, fixed at train start); an untrained lab has
        # no calib -> delta 0, keeping the forward==law gate exact.
        if self.calib is not None:
            acc, _ = self.forward(self.calib, signs, ks)
            ks[3] = min(ks[3] + self.calib_delta(acc), 15)
        return save_v5(path, self.d, signs, tuple(ks))

    def train(self, epochs, lr_shift, ctx_per_key=8, seed_data=777,
              resample_every=25, log_every=50, decay_every=0):
        self.lr0 = lr_shift
        val_X, val_cls = encode(gen_data(XorShift(seed_data + 1), 6,
                                         full=self.full_task))
        self.calib = val_X
        stream = XorShift(seed_data)
        X, cls = encode(gen_data(stream, ctx_per_key,
                                 full=self.full_task))
        best, best_w = -1.0, None
        for ep in range(epochs):
            if ep and ep % resample_every == 0:
                X, cls = encode(gen_data(stream, ctx_per_key,
                                         full=self.full_task))
            signs, ks = self.project()
            if self.frozen is None and ep == self.freeze_at:
                # k_in floor (eighth campaign): the drive-layer k sets
                # every downstream shadow clamp (32767 << (k_in+...)).
                # Measured causal chain at width: faster L0 warmup ->
                # k_in freezes low -> clamps narrow 4x while fan-in
                # sums grow 2x -> ReLU liveness collapses to 2% ->
                # gradient masks starve the law. Flooring k_in at the
                # healthy operating point keeps the clamp geometry
                # width-invariant; lawful (any k is format-valid, and
                # the drive bound |x@s| <= 2048 stays meaningful).
                ks = list(ks)
                ks[0] = max(ks[0], self.kin_floor)
                # k1 floor (ninth campaign): with k_in floored, the
                # remaining m2 liveness gap tracks fan-in - a2 sums
                # 4x the terms at h=512, sqrt-scale 2x bigger, against
                # the same clamp. +1 shift compensates. Default 0:
                # narrow runs are untouched.
                ks[1] = max(ks[1], self.k1_floor)
                self.frozen = ks
                # OU homeostasis: from here the step is FIXED at the
                # freeze-time scale and shift-decay opposes diffusion.
                # sign-SGD injects norm like an entropy pump (constant
                # step even on pure noise); a restoring drift
                # proportional to |w| gives each weight a stationary
                # Ornstein-Uhlenbeck distribution instead of a
                # runaway, with equilibrium |w|* ~ step << (wd/2)
                # pinned to the norm the frozen projection shifts
                # were derived from.
                norms = [int(np.abs(w).sum()) // w.size for w in self.w]
                self.step_f = [max(n >> lr_shift, 1) for n in norms]
                self.norm_f = norms          # the homeostatic set-point
                self.wd_f = 2 * lr_shift
                print(f"  epoch {ep}: shifts frozen at {ks}; OU step "
                      f"{self.step_f}, decay shift {self.wd_f}, "
                      f"set-point norms {[n >> 14 for n in norms]}")
            g, _, n_wrong, loss = self.grads(X, cls, signs, ks)
            # Eval and checkpoint BEFORE stepping: accuracy scores the
            # projected law of the CURRENT weights, so the checkpoint
            # must copy those same weights. Copying after step() saved
            # a law one update past the one that earned the score
            # (caught by the artifact==metric alignment check: 29.9%
            # reported, 24.5% on disk).
            if ep % log_every == 0 or ep == epochs - 1:
                va = self.accuracy(val_X, val_cls, signs, ks)
                sacc, _ = self.forward_train(val_X, signs, ks)
                sa = float((sacc[:, :N_CMD].argmax(axis=1)
                            == val_cls).mean())
                if va > best:
                    best, best_w = va, [w.copy() for w in self.w]
                    if self.save_path:
                        self.save(self.save_path)
                norms = [int(np.abs(w).sum()) // w.size for w in self.w]
                print(f"  epoch {ep:5d}: loss={loss:>12}  "
                      f"violations={n_wrong:4d}  law={va * 100:5.1f}%  "
                      f"shadow={sa * 100:5.1f}%  best={best * 100:5.1f}%  "
                      f"norm={[n >> 14 for n in norms]}", flush=True)
            # CHURN HOMEOSTASIS (sixth campaign's derived design): the
            # norm thermostat generalized from scale to LAW MOTION.
            # Every CH_T epochs post-freeze, measure each layer's
            # sign-churn; below the healthy band the layer's step
            # doubles (shadow descent is starving the law of motion -
            # the measured h=512 failure), above it the step halves.
            # Same set-point-servo grammar, one level up.
            if self.churn_servo and self.frozen is not None \
                    and ep % CH_T == 0:
                if self._servo_prev is not None:
                    for i, (s, p) in enumerate(zip(signs,
                                                   self._servo_prev)):
                        churn = int((s != p).sum()) * 1000 // s.size
                        if churn < CH_LO:
                            self.step_adj[i] = max(self.step_adj[i] - 1,
                                                   ADJ_MIN)
                        elif churn > CH_HI:
                            self.step_adj[i] = min(self.step_adj[i] + 1,
                                                   ADJ_MAX)
                self._servo_prev = [s.copy() for s in signs]
            if self.trace and ep % self.trace == 0:
                acc_t, (h_t, _z1, _m1t, _z2, _m2t, _masks) =                     self.forward_train(X[:32], signs, ks)
                _masks_live = (_m1t, _m2t)
                hc = 32767 << ks[0]
                sat = [int((np.abs(h_t[:, blk * (self.H // 8):
                                        (blk + 1) * (self.H // 8)])
                            >= hc).sum()) * 1000
                       // (32 * (self.H // 8)) for blk in range(8)]
                _m1, _m2 = _masks_live
                print(f"  [trace] ep {ep:4d} relu-mask OPEN per mille: "
                      f"m1={int(_m1.sum()) * 1000 // _m1.size} "
                      f"m2={int(_m2.sum()) * 1000 // _m2.size}",
                      flush=True)
                self.trace_line(ep, g, sat)
            # sign-SGD converges to a noise ball proportional to the
            # step; a stepped shift schedule (integer, deterministic)
            # shrinks the ball. decay_every=0 disables.
            sh = lr_shift + (ep // decay_every if decay_every else 0)
            if self.opt == "adam":
                self.step_adam(g, min(sh, lr_shift + 4))
            else:
                self.step(g, min(sh, lr_shift + 4))
        if best_w is not None:
            self.w = best_w
        return best


# ── gates ────────────────────────────────────────────────────────────

def gate():
    lab = IntLab(h=128, r1=256, r2=96, seed=42)
    X, cls = encode(gen_data(XorShift(5), 2))
    signs, ks = lab.project()

    # 1a. Batch permutation: gradients bit-identical.
    g0, _, _, _ = lab.grads(X, cls, signs, ks)
    perm = np.arange(len(cls))[::-1]
    g1, _, _, _ = lab.grads(X[perm], cls[perm], signs, ks)
    assert all(np.array_equal(a, b) for a, b in zip(g0, g1)), \
        "integer gradients changed under batch permutation"
    # 1b. Chunked accumulation: linear in the batch, exactly.
    half = len(cls) // 2
    ga, _, _, _ = lab.grads(X[:half], cls[:half], signs, ks)
    gb, _, _, _ = lab.grads(X[half:], cls[half:], signs, ks)
    assert all(np.array_equal(w, a + b)
               for w, a, b in zip(g0, ga, gb)), \
        "integer gradients are not chunk-exact"
    # 1c. The float counterexample: the same reduction pattern in
    # float64 is order-SENSITIVE (this is the mechanism behind BLAS
    # variance; integer training removes it by construction).
    a = (0.1 + 0.2) + 0.3
    b = 0.1 + (0.2 + 0.3)
    assert a != b, "float64 addition was associative?!"
    print(f"[int_train] order gate: int grads bit-identical under "
          f"permutation and chunking; float64 is order-SENSITIVE "
          f"((0.1+0.2)+0.3 = {a!r} but 0.1+(0.2+0.3) = {b!r} - the "
          f"BLAS-variance mechanism, absent by construction here)")

    # 2. FORWARD == LAW: save a v5 blob from the (untrained) lab and
    # the batched training forward must equal SsmMachine, h included.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        blob = Path(td) / "intlab.bin"
        lab.save(str(blob))
        machine = SsmMachine(str(blob))
        seqs = gen_data(XorShift(9), 1)[:10]
        Xs, _ = encode(seqs)
        acc, (h, *_rest) = lab.forward(Xs, signs, ks)
        ok = True
        for i, (keys, _) in enumerate(seqs):
            machine.run(keys)
            out = np.where(acc[i] < -8192, 0,
                           np.where(acc[i] > 8192, 32767,
                                    (acc[i] + 8192) * 2))
            got = np.asarray(machine.run(keys), dtype=np.int64)
            ok &= np.array_equal(out, got)
            ok &= np.array_equal(h[i], machine.h)
        assert ok, "training forward diverged from SsmMachine"
    print("[int_train] forward==law: batched training forward equals "
          "SsmMachine on the saved blob (outputs and h, 10 sequences)")
    print("[int_train] result: PASS - integer training is order-free "
          "by construction and trains THE law, not a shadow of it")


def main():
    ap = argparse.ArgumentParser(description="integer-exact training lab")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--r1", type=int, default=256)
    ap.add_argument("--r2", type=int, default=96)
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--task", choices=["scoped", "full"],
                    default="scoped",
                    help="scoped = single keys under context (every "
                         "prior campaign); full = + doubled keys, "
                         "WORDS, demo prefixes (the canonical task)")
    ap.add_argument("--lr-shift", type=int, default=12,
                    help="learning rate as a right shift of the "
                         "momentum buffer")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--decay-every", type=int, default=0)
    ap.add_argument("--opt", choices=["sign", "adam"], default="adam")
    ap.add_argument("--kin-floor", type=int, default=3,
                    help="minimum frozen k_in (clamp-geometry floor; "
                         "0 disables)")
    ap.add_argument("--k1-floor", type=int, default=0,
                    help="minimum frozen k1 (fan-in clamp compensation "
                         "for wide runs; 0 = untouched)")
    ap.add_argument("--churn-servo", action="store_true",
                    help="hold per-layer sign-churn in the healthy band "
                         "by adjusting each layer's step (law-motion "
                         "homeostasis)")
    ap.add_argument("--trace", type=int, default=0,
                    help="log per-layer warmup diagnostics every N epochs")
    ap.add_argument("--freeze", type=int, default=FREEZE,
                    help="epoch at which shifts, OU step and set-points "
                         "freeze; 0 = stationary from init (no "
                         "pre-freeze norm inflation)")
    ap.add_argument("--wd-shift", type=int, default=0,
                    help="weight decay as w -= |w|>>shift per step; 0=off")
    ap.add_argument("--save")
    args = ap.parse_args()
    if args.gate:
        gate()
        return
    lab = IntLab(h=args.h, r1=args.r1, r2=args.r2, seed=args.seed)
    lab.opt = args.opt
    lab.wd = args.wd_shift
    lab.freeze_at = args.freeze
    lab.trace = args.trace
    lab.churn_servo = args.churn_servo
    lab.kin_floor = args.kin_floor
    lab.k1_floor = args.k1_floor
    lab.save_path = args.save
    lab.full_task = args.task == "full"
    t0 = time.time()
    best = lab.train(args.epochs, args.lr_shift, ctx_per_key=args.ctx,
                     decay_every=args.decay_every)
    print(f"[int_train] best held-out {best * 100:.1f}% "
          f"({time.time() - t0:.0f}s)")
    if args.save:
        lab.save(args.save)


if __name__ == "__main__":
    main()
