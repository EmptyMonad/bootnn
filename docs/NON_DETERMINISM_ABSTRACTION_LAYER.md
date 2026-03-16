# DNOS Non-Determinism Abstraction Layer (NDAL)

## The Problem the IAL Doesn't Solve

The IAL eliminates *accidental* non-determinism — microsecond jitter,
arrival order variance, sensor noise. After the IAL, identical physical
events produce identical token streams.

But some non-determinism is *essential*. It's not noise to be filtered.
It's information from a universe that is genuinely unpredictable:

- **Hardware RNG**: RDRAND, /dev/urandom, thermal noise
- **Network latency**: A packet takes 12ms or 47ms — this is real data
- **Clock drift**: Two nodes' clocks diverge by real microseconds
- **Concurrent mutation**: Two users edit the same file simultaneously
- **External API responses**: HTTP 200 with body X, or timeout, or 503
- **Sensor readings below quantization**: The IAL bucket says "23°C" but
  was it 23.1 or 23.4? The difference might matter.
- **Interrupt timing relative to computation**: Where in the neural forward
  pass was the CPU when the interrupt fired?

These aren't bugs. They're the boundary between a deterministic machine
and a non-deterministic universe. DNOS must interact with this universe
without losing its core guarantee.

## The Core Idea: Oracles

An **oracle** is a named interface to non-determinism.

When DNOS needs something non-deterministic, it doesn't access it directly.
It asks an oracle. The oracle:

1. **Queries** the non-deterministic source (RNG, network, clock, etc.)
2. **Records** the query and response in an append-only log
3. **Returns** a deterministic token to the neural substrate

The token says: "I asked oracle X question Q and got answer A."

The answer A is non-deterministic — it could have been different.
But once recorded, it IS A, forever. The log makes it deterministic
*retroactively*.

```
┌─────────────────────────────────────────────────────────────┐
│                   Neural Substrate                          │
│                                                             │
│  Sees: Token(Oracle, RANDOM, answer=0x42)                  │
│  Not:  "some random thing happened"                        │
│  Not:  raw bytes from /dev/urandom                         │
│                                                             │
│  The network learns: "when I ask for randomness,           │
│  I get a value. I can use it or ignore it.                 │
│  The value is a fact about what the universe said."        │
└─────────────────────────────────────────────────────────────┘
         ▲
         │ deterministic tokens
         │
┌─────────────────────────────────────────────────────────────┐
│                      NDAL                                   │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │  Random  │ │  Clock   │ │ Network  │ │  Extern  │     │
│  │  Oracle  │ │  Oracle  │ │  Oracle  │ │  Oracle  │     │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘     │
│       │             │             │             │           │
│       ▼             ▼             ▼             ▼           │
│  ┌──────────────────────────────────────────────────┐      │
│  │              Replay Log                          │      │
│  │  (append-only, content-addressed, hashchained)   │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
         ▲
         │ raw non-determinism
         │
┌─────────────────────────────────────────────────────────────┐
│               Physical Universe                             │
│  (RNG, network, clocks, sensors, users, entropy)           │
└─────────────────────────────────────────────────────────────┘
```

## Determinism Guarantee

**Live execution**: Non-deterministic. Oracle responses are fresh
from the universe. But they are logged.

**Replay execution**: Deterministic. Oracle responses are read from
the log. Same log → same execution → same final state.

**Verification**: Two nodes that share the same log will arrive at
the same state. Verification = log comparison. If logs match,
states match. If they don't, the divergence point is the first
log entry that differs — which tells you exactly which oracle
query saw different universes.

## Oracle Types

### 1. Random Oracle

Source of entropy. Used for weight initialization, exploration
in learning, tie-breaking, nonce generation.

```
Query:   RANDOM(n_bytes=4)
Response: 0xA3F7201C
Log:     [epoch=142, oracle=RANDOM, query=4, response=A3F7201C]
```

The network receives a token encoding the response. It doesn't
know or care that the value was random. It's just a fact.

### 2. Clock Oracle

Wall-clock time. The IAL uses monotonic epochs internally, but
sometimes the system needs real-world time (logging, timestamps
for external communication, TTL calculations).

```
Query:   CLOCK(resolution=SECOND)
Response: 1710547200  (Unix timestamp)
Log:     [epoch=500, oracle=CLOCK, query=SECOND, response=1710547200]
```

Resolution parameter prevents leaking sub-epoch timing back into
the deterministic layer. The network sees "it's March 16, 2025"
not "it's 14:23:07.847291".

### 3. Network Oracle

External network communication. HTTP requests, DNS lookups,
peer-to-peer messages. The most complex oracle because responses
can be large and variable-latency.

```
Query:   NET(method=GET, target_hash=0xBEEF, timeout_epochs=100)
Response: STATUS(200, body_hash=0xDEAD, latency_epochs=3)
Log:     [epoch=1000, oracle=NET, query={...}, response={...}]
```

The network receives a compact token: success/failure, body hash,
quantized latency. Not the full response body — that goes into
a content-addressed store, referenced by hash.

### 4. Consensus Oracle

For distributed DNOS: "what do my peers think?" Queries the P2P
network for state hashes, weight checksums, or pattern votes.

```
Query:   CONSENSUS(question=STATE_HASH, quorum=3)
Response: AGREED(hash=0x1234, participants=5, dissent=0)
Log:     [epoch=2000, oracle=CONSENSUS, ...]
```

### 5. Environment Oracle

Hardware capabilities, available memory, CPU features, connected
devices. Things that are constant for a given boot but differ
between machines.

```
Query:   ENV(key=TOTAL_MEMORY)
Response: 4294967296  (4GB)
Log:     [epoch=0, oracle=ENV, query=TOTAL_MEMORY, response=4294967296]
```

Queried once at boot and cached. Makes the neural substrate
aware of its physical constraints without hardcoding them.

### 6. User Oracle

Direct human input that doesn't come through the normal keyboard/mouse
path. Modal dialogs, configuration choices, permission grants.

```
Query:   USER(prompt_hash=0xABCD, options=[ALLOW, DENY, ASK_LATER])
Response: ALLOW
Log:     [epoch=5000, oracle=USER, ...]
```

## The Replay Log

The log is the NDAL's core data structure. It makes non-determinism
retroactively deterministic.

### Structure

```
┌─────────────────────────────────────────────────────────┐
│ Entry 0: [seq=0, epoch=0, oracle=ENV, query, response]  │
│ Entry 1: [seq=1, epoch=0, oracle=ENV, query, response]  │
│ ...                                                      │
│ Entry N: [seq=N, epoch=T, oracle=X, query, response]    │
│                                                          │
│ Each entry is hash-chained:                             │
│   entry.hash = H(entry.data || prev_entry.hash)         │
│                                                          │
│ Integrity verification: check hash chain from entry 0.  │
│ Divergence detection: first entry where hashes differ.  │
└─────────────────────────────────────────────────────────┘
```

### Properties

- **Append-only**: Entries are never modified or deleted
- **Hash-chained**: Tamper-evident, verifiable
- **Content-addressed**: Large responses stored by hash, log stores hash
- **Replayable**: Feed log entries to oracles in replay mode → same tokens
- **Prunable**: Old entries can be archived once a state snapshot exists
  that incorporates them

### Modes

**Live mode**: Oracles query the real universe, log responses.
**Replay mode**: Oracles read from log, return recorded responses.
**Verification mode**: Oracles query both universe AND log, flag divergence.

## Relationship to IAL

```
Physical World
      │
      ▼
┌──────────┐
│   IAL    │ ← eliminates accidental non-determinism
└────┬─────┘
     │ deterministic tokens (for deterministic events)
     │
     ├─────────────────────────────────────┐
     │                                     │
     ▼                                     ▼
┌──────────┐                        ┌──────────┐
│  Direct  │                        │   NDAL   │ ← contains essential non-determinism
│  Tokens  │                        │  Tokens  │
└────┬─────┘                        └────┬─────┘
     │                                   │
     ▼                                   ▼
┌─────────────────────────────────────────────┐
│           Token Stream Merger               │
│  (interleaves IAL + NDAL tokens by epoch)   │
└────────────────────┬────────────────────────┘
                     │
                     ▼
              Neural Substrate
```

The IAL and NDAL produce parallel token streams. Both are canonical
and hashable. The merger interleaves them by epoch, maintaining total
order. The neural substrate sees a single unified stream.

## What the Neural Substrate Learns

With the NDAL, the network can learn:

- "When I request randomness and get a high value, I should explore.
   When I get a low value, I should exploit." (Exploration strategy)
- "Network requests sometimes fail. When they fail, I should retry
   with backoff." (Fault tolerance from experience)
- "The clock oracle says it's night. User behavior patterns change
   at night." (Temporal behavioral modeling)
- "My environment has 4GB of memory. I should use the smaller weight
   matrix." (Adaptive resource management)
- "Peers report a different state hash. I should request the divergent
   input log segment." (Self-healing)

None of this requires the network to "understand" randomness. It just
sees facts ("oracle said X") and learns correlations with outcomes.
The non-determinism is real, but the network's *response* to it is
learned and, given the same oracle log, deterministic.

## Pruning and Snapshots

The log grows without bound. Pruning strategy:

1. At epoch N, compute STATE_HASH(N) — hash of entire neural state
2. Write a **snapshot entry**: [epoch=N, state_hash=H, weights_hash=W]
3. All log entries before epoch N can be archived/deleted
4. A new node syncs by: download snapshot + replay log from epoch N

Snapshot frequency is configurable. More frequent = faster sync,
more storage. Less frequent = smaller overhead, slower sync.

## Interaction with Deterministic Training

Training introduces its own non-determinism: batch shuffling, dropout,
weight initialization. The NDAL handles all of these through the Random
Oracle:

```
# Instead of: np.random.shuffle(data)
shuffle_seed = random_oracle.query(4)  # Get 4 bytes
rng = deterministic_rng(shuffle_seed)
rng.shuffle(data)

# Log entry: [oracle=RANDOM, query=4, response=shuffle_seed]
# Replay: same seed → same shuffle → same training → same weights
```

This means distributed DNOS nodes that share a log will train
identically, even when training involves "random" operations.
The randomness is real but recorded.
