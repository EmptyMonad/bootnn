# Swarm Design — Deterministic Replication to Emergent Specialization

> Status: **draft** (design of record for the distributed phase).
> Prerequisite: Tier 3 (`docs/TIER3_DESIGN.md`, implemented). This document
> resolves Tier 3 open question 3 (IAL-token input encoding) and defines
> the phase it was deferred to.

## Goal

Scale the substrate *out*, not up. One DNOS node is a deterministic state
machine: `S(n+1) = f(S(n), input(n))`, bit-exact, CRC-stamped, verified
law==metal on every push. A swarm is N such nodes plus two new claims:

1. **Replication.** Nodes fed the same input log occupy the same state —
   provably, cheaply, at every tick.
2. **Specialization.** Nodes with *different* verified weight blobs plus a
   deterministic routing rule compose into an architecture no single node
   was designed with — the swarm's structure emerges from which
   contributions its economy accepts.

Everything else — the wallet, the ledger, the contribution economy — is a
corollary of determinism, not an add-on.

## The core insight

Every distributed-training network (Bittensor, Gensyn, et al.) struggles
with one problem: verifying a contribution without redoing all the work.
Their answers are validator committees or cryptographic proof-of-learning —
expensive, probabilistic, or both.

DNOS training is seeded and bit-exact end to end. Therefore:

- A **training contribution** is the tuple
  `(base_crc, dataset_delta, hyperparams, seed, claimed_crc)`.
- **Verification is replay**: rerun `train.py` with the tuple, compare
  CRC32. Binary, objective, no committee. Spot-verification (random
  subset of verifiers) suffices because any single honest replay exposes
  a fraud — the same structure as an optimistic rollup, where the fraud
  proof is "replay the tick."
- A **replicated node set** fed one input log *is* a ledger: the log is
  the chain, the state is the consensus, and a wallet is just a region
  of deterministic state updated by a rule inside `f`.

Determinism is the proof-of-training. The map is verified before it is
trusted as territory; now the territory is shared.

## Swarm tiers

### S0 — Deterministic replication (pure tooling, zero kernel changes)

`tools/swarm_test.py`: boot N QEMU instances of the *identical* image,
feed each the same event log (QMP, as `interactive_test.py` does today),
and after every tick read the canonical state region from guest memory
and assert all N digests are equal — and equal to the simulator's
prediction for the same log.

Canonical state = input ring + activations + cursor + color + last_cmd +
tick/step counters (the regions already exported via `dnos_symbols.json`).
The framebuffer is derived state; digest it at checkpoint cadence (every
256 ticks), not per tick. All digests are computed **host-side** from
QMP memory reads — the kernel does not attest to itself in S0.

This is the load-bearing experiment. If N identical machines diverge,
nothing downstream survives; if they don't, replication is free forever.

### S1 — The input surface (human- and agent-usable)

PS/2 scancodes stop being the input; they become one *producer* of the
input. The wire format is the **IAL token**: an 8-byte frame = 8 input
features = exactly one event slot in the history window (the clean fit
noted in Tier 3 OQ3).

- Kernel: read ≤ 1 frame per tick from COM1 (16550, polled at tick
  boundary — no new interrupts, no new nondeterminism; the serial ring
  drains exactly like the PS/2 ring). The keyboard ISR is rewritten to
  *emit* an IAL token into the same ring, so keyboard and serial are
  indistinguishable to `f`.
- Host: the Rust IAL crate becomes the canonical encoder. A thin client
  (`tools/dnos_client.py`) gives humans a terminal and agents a pipe;
  every session is an event log, hence replayable, hence verifiable.
  Usability and auditability are the same feature.
- Differential gate: host IAL encoder vs. kernel decode, token by token,
  in the style of `interactive_test.py`.

### S2 — The contribution economy (ledger as replicated state)

- **Log**: append-only, Merkle-chained event log. Entries are signed at
  the boundary (SPHINCS+ — large signatures are fine in a log; only
  32-byte hashes ever enter machine state).
- **Wallet**: an account table in a fixed state region. Balances change
  only via `f` applying an accepted-contribution or transfer event from
  the log. No blockchain machinery; consensus is byte-equality of
  replicated state, disputes are settled by replay.
- **Issuance rule**: a verified training contribution (replay matches
  `claimed_crc`, and the new blob passes the full CI gauntlet) mints a
  fixed credit to the contributor's account as part of applying the
  acceptance event. Token supply is therefore a pure function of the
  log — auditable by anyone who can run `git clone` and `make`.

### S3 — Specialization and routing (the emergent architecture)

- Nodes carry *different* verified blobs: specialists, identified by
  weight-CRC lineage back through the contribution log.
- A **router** — itself a small deterministic net or rule — maps the
  token stream to a specialist. Mixture-of-experts, except the experts
  are physical machines and the "architecture" is whatever topology the
  contribution economy has grown.
- Falsifiable proxy before anything grander: two specialists plus a
  router must beat one generalist of equal total weights on a mixed
  task suite, under the same gauntlet. If specialization doesn't pay
  at N=2, it doesn't pay.

## Dated assumptions this phase retires

Recorded so the reasoning is explicit, in the spirit of Strategy B:

1. **Sliding-window MLP vs. the creed.** `S(n+1) = f(S(n), input(n))` is
   the definition of a recurrent cell, yet the network is a feedforward
   MLP over a replayed 64-event window — the window is a workaround for
   having no recurrent state. A recurrent core (GRU-class or state-space)
   would make the state finite and explicit and delete the window
   entirely. Candidate **Tier 4**; the swarm design is deliberately
   agnostic to the core's internals.
2. **Q8.8 int16.** The field moved to int4/int8 and ternary (BitNet
   b1.58-style) weights. Ternary is *more* aligned with bare metal, not
   less: multiply-free inference (add/sub/skip), ~10× smaller blobs, and
   nothing about determinism or CRC canonicalization changes. Candidate
   Tier 4 alongside the recurrent core.
3. **Token-sale economics.** The AiCoin-era framing (supply pegged to
   world GDP, benchmark-triggered releases judged by nobody in
   particular) is 2021 thinking. The 2026-relevant economy is
   machine-to-machine: verifiable compute, data provenance, agent
   payments. S2 keeps the one durable mechanic — contribute, verify,
   credit — and discards the macroeconomics.
4. **Scale-up as the axis of progress.** Tier 4 = "more weights" is the
   pre-MoE instinct. The frontier lesson is many specialists plus
   routing and more computation at inference time. Hence S3 is the
   growth axis; single-node tiers become component upgrades.
5. **Storage-for-tokens.** Commoditized (Filecoin et al.); explicitly
   out of scope. DNOS's scarce verified resource is *training*, not disk.

## Test plan (all gates extended, none removed)

| Gate | Swarm form |
|------|-----------|
| Replication | `swarm_test.py`: N nodes, one log, per-tick state digests all equal, == simulator (S0) |
| Divergence localization | inject one flipped state byte in one node; digest mismatch must identify the node and tick (S0 negative test) |
| Input surface | host IAL encoder == kernel decode per token; keyboard and serial paths produce identical state trajectories for identical semantics (S1) |
| Contribution replay | a valid tuple reproduces `claimed_crc`; a tampered dataset delta must fail replay (S2, negative-tested) |
| Ledger integrity | wallet region digest is a pure function of the log prefix; truncated/reordered logs must produce different digests (S2) |
| Specialization pays | 2 specialists + router ≥ 1 generalist, equal total weights, full gauntlet (S3) |

## Open questions (to resolve before implementation)

1. Serial polling cadence: one frame per tick is the determinism-safe
   default; is a bounded burst drain (as the PS/2 ring does) needed for
   agent-driven input rates, and does `tick_test.py`'s invariant carry
   over unchanged?
2. State-digest algorithm: CRC32 matches the existing toolchain but is
   not collision-resistant against an adversary; S2 likely needs a real
   hash (BLAKE3?) at the log/checkpoint layer while CRC32 remains the
   fast per-tick replication check. Where exactly is the boundary?
3. Router placement: host-side process (cheap, but the router is then
   outside the verified substrate) vs. a dedicated DNOS node whose
   output layer *is* the routing table (pure, but adds a hop). Lean:
   host-side for S3 bring-up, in-substrate as the exit criterion.
4. Contribution granularity: full retrains are replay-verifiable today;
   are delta-updates (fine-tunes from a base CRC) admissible, and does
   seeded minibatch order survive them bit-exactly?

## Exit criteria

S0: the replication gate runs in CI (N=3 under QEMU TCG) and a flipped
byte is localized to node and tick. S1: a scripted agent session over
COM1 reproduces, key for key, the state trajectory of the same session
typed on PS/2. S2: a training contribution submitted as a log event is
verified by replay on a second machine and mints exactly once. S3: the
specialization inequality holds under the full gauntlet. At every stage
the Tier 3 gauntlet still passes unmodified on each individual node.
