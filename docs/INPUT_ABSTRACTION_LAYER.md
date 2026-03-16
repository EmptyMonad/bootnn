# DNOS Input Abstraction Layer (IAL)

## Core Principle

The neural substrate never touches raw reality.

Between the physical world (microsecond jitter, sensor noise, event ordering
variance, hardware timing) and the DNOS state equation `STATE(t+1) = f(STATE(t),
INPUT(t))`, there is a **membrane** that quantizes continuous, noisy phenomena
into discrete, deterministic **tokens**.

Two DNOS nodes observing the same physical event produce the same token.
Not approximately. Exactly.

## Why This Matters

A learning machine that's sensitive to microsecond timing isn't learning behavior
— it's memorizing noise. The distinction between "user typed 'box'" and "user
typed 'box' but the 'o' arrived 3μs later on node B" should be invisible to
the network. The network learns *what happened*, not *when it happened at the
hardware level*.

This is also the key to behavioral abstraction from outcomes: the network
observes that a sequence of actions led to a result, not the precise timing
envelope of those actions. Two different timing profiles that produce the same
behavioral outcome should map to the same input token sequence.

## Architecture

```
Physical World
│
│  (continuous, noisy, non-deterministic)
│
▼
┌─────────────────────────────────────────────────────────┐
│              INPUT ABSTRACTION LAYER                     │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Temporal  │  │ Spatial  │  │ Semantic │              │
│  │ Quantizer │  │ Quantizer│  │ Encoder  │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │              │              │                    │
│       ▼              ▼              ▼                    │
│  ┌──────────────────────────────────────────┐           │
│  │           Token Compositor               │           │
│  │  (merge quantized channels into tokens)  │           │
│  └──────────────┬───────────────────────────┘           │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────┐           │
│  │         Sequence Canonicalizer           │           │
│  │  (order-independent within same epoch)   │           │
│  └──────────────┬───────────────────────────┘           │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────┐           │
│  │            Token Stream                  │           │
│  │  (deterministic, hashable, replayable)   │           │
│  └──────────────────────────────────────────┘           │
│                                                          │
└─────────────────────────────────────────────────────────┘
│
│  (discrete, deterministic, canonical)
│
▼
Neural Substrate: STATE(t+1) = f(STATE(t), INPUT(t))
```

## Components

### 1. Temporal Quantizer

Collapses continuous time into discrete **epochs**. Events within the same
epoch are treated as simultaneous.

```
Physical timeline:
  t=0.000ms  key_down('b')
  t=0.003ms  key_down('o')    ← 3μs later
  t=0.847ms  key_down('x')

Epoch size = 10ms (configurable):
  Epoch 0: {key('b'), key('o'), key('x')}   ← all simultaneous
  
Epoch size = 0.1ms:
  Epoch 0: {key('b'), key('o')}
  Epoch 8: {key('x')}
```

The epoch size is a **resolution parameter** that determines the granularity
of temporal perception. Larger epochs = more jitter tolerance, less temporal
resolution. Smaller epochs = more sensitivity, less jitter tolerance.

Key design choice: epoch boundaries are **absolute**, not relative. Epoch N
starts at `N * epoch_duration` from a fixed reference point. This means two
nodes with synchronized clocks (NTP-level, ~1ms) produce the same epoch
boundaries for events separated by more than 1ms + epoch_duration.

### 2. Spatial Quantizer

For sensor data, cursor positions, screen coordinates — continuous values
are bucketed into discrete levels.

```
Raw cursor: (157.3, 94.7)
Grid size = 10px:
  Bucket: (15, 9)

Raw temperature sensor: 23.847°C
Resolution = 0.5°C:
  Bucket: 24.0
```

### 3. Semantic Encoder

Maps raw events to semantic tokens. This is where behavioral abstraction
happens. The encoder can collapse multiple raw events into a single semantic
event:

```
Raw events:
  mouse_move(100, 200)
  mouse_move(102, 201)
  mouse_move(105, 199)
  mouse_click(105, 199)

Semantic token:
  CLICK_AT(10, 20)    ← quantized position, movement elided

Raw events:
  key_down('b'), key_up('b'), key_down('o'), key_up('o'),
  key_down('x'), key_up('x')

Semantic token:
  WORD("box")          ← key_up events elided, sequence recognized
```

The semantic encoder is where DNOS's learning gets interesting: the encoder
itself can be learned. A meta-network that discovers useful abstractions from
raw event streams. But the initial implementation uses fixed rules.

### 4. Token Compositor

Merges outputs from all quantizer channels into a unified token. Each token
has a canonical binary representation:

```rust
struct Token {
    epoch: u64,            // Temporal position
    channel: u8,           // Input channel (keyboard, mouse, sensor, ...)
    event_type: u16,       // Semantic event type
    payload: [u8; 8],      // Quantized event data
}
// Total: 19 bytes per token, fixed-size
```

### 5. Sequence Canonicalizer

Within a single epoch, events from different channels may arrive in any order.
The canonicalizer sorts them into a deterministic order:

```
Sort key: (epoch, channel, event_type, payload)
```

This is pure lexicographic sort on the binary representation. Two nodes that
received the same events in the same epoch will produce the same token sequence
regardless of arrival order.

## Determinism Guarantees

| Source of non-determinism | IAL mitigation |
|--------------------------|----------------|
| Microsecond timing jitter | Temporal quantizer (epoch binning) |
| Sensor noise | Spatial quantizer (bucketing) |
| Event arrival order | Canonicalizer (deterministic sort) |
| Missing/dropped events | Explicit MISS token + timeout epoch |
| Clock drift between nodes | NTP sync + epoch > drift bound |
| Hardware interrupt timing | Events quantized before reaching network |

## Behavioral Abstraction from Outcomes

This is the key insight you raised: the network should learn to associate
behavioral patterns with outcomes, not raw timing profiles.

The IAL enables this by construction. When the semantic encoder collapses
"user moved mouse erratically then clicked" into `CLICK_AT(x,y)`, the neural
substrate only sees the click. The behavioral outcome (something was clicked)
is preserved; the motor noise (erratic movement) is discarded.

At higher abstraction levels, the semantic encoder could emit:
- `FILE_OPENED("readme.md")` instead of a sequence of double-click events
- `TYPED_WORD("hello")` instead of individual key events
- `DREW_BOX(x,y,w,h)` instead of a sequence of mouse drags

Each level of abstraction lets the network learn higher-order behavioral
patterns. The network operating on `FILE_OPENED` tokens learns file usage
patterns. The network operating on raw key tokens learns typing patterns.
Different abstraction levels, same substrate.

## Epoch Selection Strategy

The epoch size is the critical tuning parameter. Too small and you're back
to timing sensitivity. Too large and you lose causal ordering.

Recommended approach: **adaptive multi-resolution epochs**.

```
Layer 0 (physical):    100μs epochs  — raw event capture
Layer 1 (perceptual):  10ms epochs   — human perceptual window
Layer 2 (action):      100ms epochs  — individual actions
Layer 3 (behavioral):  1s epochs     — behavioral sequences
Layer 4 (session):     10s epochs    — high-level intentions
```

Each layer's tokens are inputs to a different depth of the network. The
network doesn't choose a single temporal resolution — it sees all of them
simultaneously, just at different layers.

## Interaction with Existing DNOS Tiers

### Bare-metal (Tier 1/2)

The IAL replaces the current `encode_input` function in `dnos.asm`. Instead
of directly converting key scancodes to binary vectors, the bootloader runs
a minimal IAL that:
1. Bins keyboard interrupts into PIT-tick epochs
2. Sorts same-epoch events by scancode
3. Encodes the canonical token sequence into the input layer

This is ~50 lines of additional assembly.

### dnosd (Rust daemon)

The IAL is a Rust module that sits between the observer and the neural network.
The observer produces raw `Observation` structs; the IAL converts them to
`Token` streams; the neural network consumes tokens.

### Distributed DNOS

Two nodes running IAL with the same epoch configuration and NTP-synchronized
clocks will produce identical token streams for the same physical events
(within the jitter tolerance of the epoch size). State verification becomes
trivial: hash the token stream, compare hashes.

## What This Does NOT Solve

- **Which events to observe** — that's the observer's job
- **What the network learns** — that's the training pipeline's job
- **How nodes discover each other** — that's the P2P layer's job
- **Malicious input** — IAL doesn't validate, only canonicalizes

The IAL is purely a **determinism membrane**. It guarantees that the same
physical reality maps to the same token stream. Everything downstream of
the token stream is deterministic by construction.
