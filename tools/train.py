#!/usr/bin/env python3
"""
DNOS Tier 2 Weight Generator
4-layer network (256→128→64→32) with Q8.8 quantization-aware training.

Generates weights matching the assembly neural core exactly:
  - Q8.8 fixed-point (int16): multiply → shift right 8
  - ReLU hidden layers, piecewise sigmoid output
  - 128-byte header with CRC32 checksum
  - 43,008 weights = 86,016 bytes + 128 header = 86,144 bytes

Usage:
    python3 train_tier2.py                         # Default training
    python3 train_tier2.py --epochs 5000           # Longer training
    python3 train_tier2.py --validate weights.bin  # Validate exported weights
"""

import numpy as np
import struct
import argparse
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Network Topology (must match dnos.asm exactly)
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_SIZE = 256      # 32 events × 8 features
HIDDEN1_SIZE = 128
HIDDEN2_SIZE = 64
OUTPUT_SIZE = 32

LAYER_SIZES = [INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE]

W1_COUNT = INPUT_SIZE * HIDDEN1_SIZE    # 32768
W2_COUNT = HIDDEN1_SIZE * HIDDEN2_SIZE  #  8192
W3_COUNT = HIDDEN2_SIZE * OUTPUT_SIZE   #  2048
TOTAL_WEIGHTS = W1_COUNT + W2_COUNT + W3_COUNT  # 43008

HEADER_SIZE = 128
WEIGHT_DATA_SIZE = TOTAL_WEIGHTS * 2  # int16
TOTAL_FILE_SIZE = HEADER_SIZE + WEIGHT_DATA_SIZE

# Q8.8 scaling: float value * 256 → int16
Q88_SCALE = 256
Q88_MAX = 32767
Q88_MIN = -32768
Q88_HIGH_ACTIVATION = 0x7F00  # legacy: ~127.0 in Q8.8 (no longer used for input)

# Input active-bit level. MUST match dnos.asm encode_input_32 constant.
#   1.0 float → Q8.8 raw 0x0100 (256). Small inputs keep pre-activations in
#   range so gradients flow and Q8.8 clamping doesn't saturate the network.
INPUT_ACTIVE_LEVEL = 1.0           # asm writes 0x0100 for a set bit

# Maximum representable activation after ReLU clamp (int16 max in Q8.8 units).
#   Assembly clamps every hidden activation to [0, 32767] raw = [0, ~128.0].
ACT_MAX = Q88_MAX / Q88_SCALE      # 127.996

# ═══════════════════════════════════════════════════════════════════════════════
# Commands (must match dnos.asm CMD_* constants)
# ═══════════════════════════════════════════════════════════════════════════════

CMD_NAMES = [
    "nop", "pixel", "hline", "vline", "rect", "filled_rect",
    "circle", "clear", "move_up", "move_down", "move_left", "move_right",
    "color_next", "color_prev", "size_up", "size_down",
    "fill", "text", "undo", "save",
    # outputs 20-31 are cursor deltas / reserved
]

CMD = {name: i for i, name in enumerate(CMD_NAMES)}

# Context
CONTEXT_EVENTS = 32
CONTEXT_FEATURES = 8

# ═══════════════════════════════════════════════════════════════════════════════
# Input Encoding (must match assembly encode_input_32)
# ═══════════════════════════════════════════════════════════════════════════════

def ascii_to_binary_q88(key):
    """Convert ASCII key to 8-element Q8.8 binary vector.
    Matches assembly: bit set → 0x7F00, bit clear → 0x0000."""
    vec = np.zeros(8, dtype=np.float32)
    for i in range(8):
        if (key >> i) & 1:
            vec[i] = INPUT_ACTIVE_LEVEL  # 1.0 → asm 0x0100
    return vec

def sequence_to_input(keys):
    """Convert a list of ASCII keys to 256-element input vector.
    Keys[0] = most recent. Padded to 32 events."""
    inp = np.zeros(INPUT_SIZE, dtype=np.float32)
    for t, key in enumerate(keys[:CONTEXT_EVENTS]):
        offset = t * CONTEXT_FEATURES
        inp[offset:offset+8] = ascii_to_binary_q88(key)
    return inp

def command_to_output(cmd_idx, dx=0, dy=0):
    """Create target output vector. One-hot for command, scaled deltas."""
    out = np.zeros(OUTPUT_SIZE, dtype=np.float32)
    out[cmd_idx] = 1.0

    # Cursor deltas in outputs 20-23 (scaled)
    out[20] = np.clip(dx / 32.0, -1, 1)
    out[21] = np.clip(dx / 32.0, -1, 1)
    out[22] = np.clip(dy / 32.0, -1, 1)
    out[23] = np.clip(dy / 32.0, -1, 1)
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# Training Data Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_training_data():
    """Generate comprehensive training examples."""
    data = []

    # --- Single key commands ---
    single_keys = [
        (ord('p'), CMD['pixel'], "P → pixel"),
        (ord('d'), CMD['pixel'], "D → pixel"),
        (ord('.'), CMD['pixel'], ". → pixel"),
        (ord('b'), CMD['rect'], "B → rect"),
        (ord('r'), CMD['rect'], "R → rect"),
        (ord('l'), CMD['hline'], "L → hline"),
        (ord('h'), CMD['hline'], "H → hline"),
        (ord('-'), CMD['hline'], "- → hline"),
        (ord('v'), CMD['vline'], "V → vline"),
        (ord('|'), CMD['vline'], "| → vline"),
        (ord('o'), CMD['circle'], "O → circle"),
        (ord('f'), CMD['fill'], "F → fill"),
        (ord('c'), CMD['clear'], "C → clear"),
        (ord('u'), CMD['undo'], "U → undo"),
        (ord(' '), CMD['nop'], "SPACE → nop"),
        (ord('w'), CMD['move_up'], "W → move_up"),
        (ord('k'), CMD['move_up'], "K → move_up"),
        (ord('s'), CMD['move_down'], "S → move_down"),
        (ord('j'), CMD['move_down'], "J → move_down"),
        (ord('a'), CMD['move_left'], "A → move_left"),
        (ord('+'), CMD['color_next'], "+ → color_next"),
    ]

    for key, cmd, desc in single_keys:
        inp = sequence_to_input([key])
        out = command_to_output(cmd)
        data.append((inp, out, desc))

    # --- Two-key sequences ---
    seqs = [
        ([ord('b'), ord('b')], CMD['rect'], "BB → rect"),
        ([ord('p'), ord('p')], CMD['pixel'], "PP → pixel"),
        ([ord('l'), ord('l')], CMD['hline'], "LL → hline"),
        ([ord('v'), ord('v')], CMD['vline'], "VV → vline"),
        ([ord('c'), ord('c')], CMD['clear'], "CC → clear"),
    ]

    for seq, cmd, desc in seqs:
        inp = sequence_to_input(seq)
        out = command_to_output(cmd)
        data.append((inp, out, desc))

    # --- Word-like patterns ---
    words = [
        ("box",   CMD['rect'],       "box → rect"),
        ("rect",  CMD['rect'],       "rect → rect"),
        ("line",  CMD['hline'],      "line → hline"),
        ("hline", CMD['hline'],      "hline → hline"),
        ("vline", CMD['vline'],      "vline → vline"),
        ("dot",   CMD['pixel'],      "dot → pixel"),
        ("pixel", CMD['pixel'],      "pixel → pixel"),
        ("fill",  CMD['fill'],       "fill → fill"),
        ("clear", CMD['clear'],      "clear → clear"),
        ("cls",   CMD['clear'],      "cls → clear"),
        ("circ",  CMD['circle'],     "circ → circle"),
        ("ring",  CMD['circle'],     "ring → circle"),
        ("undo",  CMD['undo'],       "undo → undo"),
        ("color", CMD['color_next'], "color → color_next"),
        ("up",    CMD['move_up'],    "up → move_up"),
        ("down",  CMD['move_down'],  "down → move_down"),
        ("left",  CMD['move_left'],  "left → move_left"),
        ("right", CMD['move_right'], "right → move_right"),
        ("save",  CMD['save'],       "save → save"),
        ("text",  CMD['text'],       "text → text"),
    ]

    for word, cmd, desc in words:
        keys = [ord(c) for c in word]
        inp = sequence_to_input(keys)
        out = command_to_output(cmd)
        data.append((inp, out, desc))

    # --- Demo sequence keys (must work for first-boot demo) ---
    demo_keys = [ord(c) for c in 'pboxline']
    for i in range(len(demo_keys)):
        key = demo_keys[i]
        # Match what the demo feeds one at a time
        # After 'p': pixel
        if chr(key) == 'p':
            inp = sequence_to_input([key])
            out = command_to_output(CMD['pixel'])
            data.append((inp, out, "demo: p → pixel"))
        elif chr(key) == 'b':
            inp = sequence_to_input([key])
            out = command_to_output(CMD['rect'])
            data.append((inp, out, "demo: b → rect"))
        elif chr(key) == 'l':
            inp = sequence_to_input([key])
            out = command_to_output(CMD['hline'])
            data.append((inp, out, "demo: l → hline"))
        elif chr(key) == 'e':
            # After 'line' sequence
            inp = sequence_to_input([ord('e'), ord('n'), ord('i'), ord('l')])
            out = command_to_output(CMD['hline'])
            data.append((inp, out, "demo: line → hline"))

    return data

# ═══════════════════════════════════════════════════════════════════════════════
# Neural Network with Q8.8 Quantization-Aware Training
# ═══════════════════════════════════════════════════════════════════════════════

def q88_quantize(x):
    """Simulate Q8.8 quantization: float → int16 → float (round-trip)."""
    scaled = np.round(x * Q88_SCALE).astype(np.int32)
    scaled = np.clip(scaled, Q88_MIN, Q88_MAX)
    return scaled.astype(np.float32) / Q88_SCALE

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float32)

def sigmoid_piecewise(x):
    """Piecewise linear sigmoid matching assembly implementation.
    Maps [-32, 32] → [0, 1] (Q8.8 equivalent)."""
    return np.clip((x + 32.0) / 64.0, 0, 1)

def sigmoid_piecewise_deriv(x):
    """Derivative of piecewise sigmoid."""
    mask = (x > -32.0) & (x < 32.0)
    return mask.astype(np.float32) / 64.0

class Tier2Network:
    """4-layer network with Q8.8 quantization-aware training."""

    def __init__(self):
        # Xavier initialization
        self.w1 = np.random.randn(INPUT_SIZE, HIDDEN1_SIZE) * np.sqrt(2.0 / INPUT_SIZE)
        self.w2 = np.random.randn(HIDDEN1_SIZE, HIDDEN2_SIZE) * np.sqrt(2.0 / HIDDEN1_SIZE)
        self.w3 = np.random.randn(HIDDEN2_SIZE, OUTPUT_SIZE) * np.sqrt(2.0 / HIDDEN2_SIZE)

        # Scale weights to Q8.8 friendly range
        self.w1 *= 0.5
        self.w2 *= 0.5
        self.w3 *= 0.5

    def forward(self, x, quantize=False):
        """Forward pass. If quantize=True, simulate Q8.8 at each step."""
        self.x = x

        # Layer 1: Input → Hidden1 (ReLU)
        w1 = q88_quantize(self.w1) if quantize else self.w1
        self.z1 = x @ w1
        if quantize:
            self.z1 = q88_quantize(self.z1)
        self.a1 = relu(self.z1)

        # Layer 2: Hidden1 → Hidden2 (ReLU)
        w2 = q88_quantize(self.w2) if quantize else self.w2
        self.z2 = self.a1 @ w2
        if quantize:
            self.z2 = q88_quantize(self.z2)
        self.a2 = relu(self.z2)

        # Layer 3: Hidden2 → Output (piecewise sigmoid)
        w3 = q88_quantize(self.w3) if quantize else self.w3
        self.z3 = self.a2 @ w3
        if quantize:
            self.z3 = q88_quantize(self.z3)
        self.a3 = sigmoid_piecewise(self.z3)

        return self.a3

    def backward(self, x, y, lr=0.05):
        """Backprop with straight-through estimator for quantization."""
        # Output layer
        dz3 = (self.a3 - y) * sigmoid_piecewise_deriv(self.z3)
        dw3 = self.a2.reshape(-1, 1) @ dz3.reshape(1, -1)

        # Hidden2
        dz2 = (dz3 @ self.w3.T) * relu_deriv(self.z2)
        dw2 = self.a1.reshape(-1, 1) @ dz2.reshape(1, -1)

        # Hidden1
        dz1 = (dz2 @ self.w2.T) * relu_deriv(self.z1)
        dw1 = x.reshape(-1, 1) @ dz1.reshape(1, -1)

        # Gradient clipping
        max_norm = 1.0
        for dw in [dw1, dw2, dw3]:
            norm = np.linalg.norm(dw)
            if norm > max_norm:
                dw *= max_norm / norm

        # Update
        self.w1 -= lr * dw1
        self.w2 -= lr * dw2
        self.w3 -= lr * dw3

        return np.mean((self.a3 - y) ** 2)

    def _ste_quant(self, w):
        """Quantize weights to Q8.8 (forward). Straight-through on backward:
        callers use the returned (quantized) value for the forward/backprop
        matmuls but apply the gradient to the underlying float weight."""
        return np.clip(np.round(w * Q88_SCALE), Q88_MIN, Q88_MAX) / Q88_SCALE

    def _qat_forward(self, X, quantize=True):
        """Forward pass that mirrors the assembly neural core for gradient
        computation: Q8.8-quantized weights (STE), ReLU + clamp-to-ACT_MAX,
        and activations re-quantized to the Q8.8 grid between layers (STE),
        since the kernel stores each hidden activation as an int16.
        Returns (logits, cache). Output argmax over [:20] equals the
        assembly's because the piecewise sigmoid is monotonic."""
        w1 = self._ste_quant(self.w1) if quantize else self.w1
        w2 = self._ste_quant(self.w2) if quantize else self.w2
        w3 = self._ste_quant(self.w3) if quantize else self.w3

        z1 = X @ w1
        a1 = np.clip(z1, 0, ACT_MAX)
        if quantize:
            a1 = np.round(a1 * Q88_SCALE) / Q88_SCALE   # int16 activation (STE)
        z2 = a1 @ w2
        a2 = np.clip(z2, 0, ACT_MAX)
        if quantize:
            a2 = np.round(a2 * Q88_SCALE) / Q88_SCALE
        logits = a2 @ w3                         # output (pre piecewise-sigmoid)
        cache = (X, w1, w2, w3, z1, a1, z2, a2)
        return logits, cache

    def _exact_int_logits(self, X):
        """Bit-exact integer forward — vectorized twin of
        simulate_assembly_forward (per-element imul/sar-8 then accumulate,
        int16-clamped activations). Used as the ground-truth selection metric
        so training optimizes what the kernel actually computes."""
        def qw(w):
            return np.clip(np.round(w * Q88_SCALE),
                           Q88_MIN, Q88_MAX).astype(np.int64)
        Xr = np.clip(np.round(X * Q88_SCALE), Q88_MIN, Q88_MAX).astype(np.int64)
        w1r, w2r, w3r = qw(self.w1), qw(self.w2), qw(self.w3)
        # >> 8 is arithmetic (floor) on int64, matching SAR in the kernel
        z1 = ((Xr[:, :, None] * w1r[None, :, :]) >> 8).sum(axis=1)
        a1 = np.clip(z1, 0, Q88_MAX)
        z2 = ((a1[:, :, None] * w2r[None, :, :]) >> 8).sum(axis=1)
        a2 = np.clip(z2, 0, Q88_MAX)
        z3 = ((a2[:, :, None] * w3r[None, :, :]) >> 8).sum(axis=1)
        return z3

    def train(self, data, epochs=2000, lr=0.001):
        """Quantization-aware training with a classification objective.

        The forward pass matches the assembly core exactly (ReLU + clamp,
        Q8.8-quantized weights via straight-through estimator). The loss is
        softmax cross-entropy over the 20 command outputs, which trains the
        argmax the kernel actually decodes — instead of MSE against a smooth
        sigmoid the bare-metal code never runs."""
        print(f"Training {TOTAL_WEIGHTS:,} weights on {len(data)} examples "
              f"for {epochs} epochs (quantization-aware)...")

        X = np.array([d[0] for d in data], dtype=np.float32)
        Y = np.array([d[1] for d in data], dtype=np.float32)
        cls = np.argmax(Y[:, :20], axis=1)       # target command class

        ms = [np.zeros_like(w) for w in [self.w1, self.w2, self.w3]]
        vs = [np.zeros_like(w) for w in [self.w1, self.w2, self.w3]]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        best_acc = 0
        best_weights = None
        m = len(X)

        for epoch in range(epochs):
            idx = np.random.permutation(m)
            Xs, cs = X[idx], cls[idx]

            # --- Faithful (quantized) forward ---
            logits, (Xc, w1q, w2q, w3q, z1, a1, z2, a2) = \
                self._qat_forward(Xs, quantize=True)

            # --- Softmax cross-entropy over the 20 command logits ---
            l20 = logits[:, :20]
            l20 = l20 - l20.max(axis=1, keepdims=True)
            ex = np.exp(l20)
            sm = ex / ex.sum(axis=1, keepdims=True)
            loss = -np.mean(np.log(sm[np.arange(m), cs] + 1e-9))

            # --- Backward (straight-through through quantization & clamp) ---
            dlogits = np.zeros_like(logits)
            dlogits[:, :20] = sm
            dlogits[np.arange(m), cs] -= 1.0
            dlogits /= m

            # Soft cap: keep every logit inside the output's piecewise-linear
            # region (|float| < 32 ⇔ |raw| < 8192). Outside it the kernel's
            # sigmoid saturates and argmax ties collapse, so we penalize logits
            # beyond ±CAP. This is what keeps the exact-int metric == validator.
            CAP, CAP_LAMBDA = 28.0, 1e-3
            over = np.clip(np.abs(logits) - CAP, 0, None)
            loss += CAP_LAMBDA * np.mean(np.sum(over ** 2, axis=1))
            dlogits += CAP_LAMBDA * 2.0 * over * np.sign(logits) / m

            dw3 = a2.T @ dlogits
            da2 = dlogits @ w3q.T
            dz2 = da2 * ((z2 > 0) & (z2 < ACT_MAX))
            dw2 = a1.T @ dz2
            da1 = dz2 @ w2q.T
            dz1 = da1 * ((z1 > 0) & (z1 < ACT_MAX))
            dw1 = Xc.T @ dz1

            # LR schedule: decay in the final third
            cur_lr = lr * (0.5 if epoch > epochs * 2 // 3 else 1.0)

            # Adam update (applied to float master weights)
            t_adam = epoch + 1
            for i, (w, dw) in enumerate(
                    [(self.w1, dw1), (self.w2, dw2), (self.w3, dw3)]):
                ms[i] = beta1 * ms[i] + (1 - beta1) * dw
                vs[i] = beta2 * vs[i] + (1 - beta2) * dw ** 2
                m_hat = ms[i] / (1 - beta1 ** t_adam)
                v_hat = vs[i] / (1 - beta2 ** t_adam)
                w -= cur_lr * m_hat / (np.sqrt(v_hat) + eps)

            if epoch % 500 == 0 or epoch == epochs - 1:
                acc_q = self._qat_accuracy(X, cls)
                if acc_q >= best_acc:
                    best_acc = acc_q
                    best_weights = (self.w1.copy(), self.w2.copy(),
                                    self.w3.copy())
                print(f"  Epoch {epoch:5d}: loss={loss:.4f}  "
                      f"qat_acc={acc_q:.1f}%  best={best_acc:.1f}%")

        # Restore best quantized-accuracy weights
        if best_weights:
            self.w1, self.w2, self.w3 = best_weights
        print(f"\n  Best quantization-aware accuracy: {best_acc:.1f}%")

    def _qat_accuracy(self, X, cls):
        """Accuracy under the bit-exact integer forward (the metric the
        bare-metal kernel will reproduce)."""
        z3 = self._exact_int_logits(X)
        # Apply the integer piecewise sigmoid before argmax, exactly as
        # simulate_assembly_forward / the kernel do (saturation + tie order).
        out = np.where(z3 < -8192, 0,
                       np.where(z3 > 8192, 32767, (z3 + 8192) * 2))
        preds = np.argmax(out[:, :20], axis=1)
        return 100 * np.mean(preds == cls)

    def _test_accuracy_piecewise(self, data):
        """Test accuracy using the piecewise sigmoid that matches assembly."""
        X = np.array([d[0] for d in data], dtype=np.float32)
        Y = np.array([d[1] for d in data], dtype=np.float32)
        z1 = X @ self.w1; a1 = relu(z1)
        z2 = a1 @ self.w2; a2 = relu(z2)
        z3 = a2 @ self.w3
        a3 = np.clip((z3 + 32.0) / 64.0, 0, 1)
        preds = np.argmax(a3[:, :20], axis=1)
        exps = np.argmax(Y[:, :20], axis=1)
        return 100 * np.mean(preds == exps)

    def _test_accuracy(self, data, quantize=False):
        correct = 0
        for inp, expected, _ in data:
            out = self.forward(inp, quantize=quantize)
            pred = np.argmax(out[:20])
            exp = np.argmax(expected[:20])
            if pred == exp:
                correct += 1
        return 100 * correct / len(data)

    def test(self, data):
        """Full test report."""
        print("\n" + "=" * 60)
        print("Test Results (float32 / Q8.8 quantized)")
        print("=" * 60)

        correct_f = 0
        correct_q = 0
        divergent = 0

        for inp, expected, desc in data:
            # Float test
            out_f = self.forward(inp, quantize=False)
            pred_f = np.argmax(out_f[:20])

            # Q8.8 test
            out_q = self.forward(inp, quantize=True)
            pred_q = np.argmax(out_q[:20])

            exp = np.argmax(expected[:20])

            f_ok = "✓" if pred_f == exp else "✗"
            q_ok = "✓" if pred_q == exp else "✗"
            div = " DIVERGE" if pred_f != pred_q else ""

            if pred_f == exp: correct_f += 1
            if pred_q == exp: correct_q += 1
            if pred_f != pred_q: divergent += 1

            cmd_f = CMD_NAMES[pred_f] if pred_f < len(CMD_NAMES) else f"cmd{pred_f}"
            cmd_q = CMD_NAMES[pred_q] if pred_q < len(CMD_NAMES) else f"cmd{pred_q}"
            cmd_e = CMD_NAMES[exp] if exp < len(CMD_NAMES) else f"cmd{exp}"

            print(f"  {f_ok}/{q_ok} {desc:30s}  "
                  f"float={cmd_f:12s} q88={cmd_q:12s} exp={cmd_e}{div}")

        n = len(data)
        print(f"\nFloat accuracy:    {correct_f}/{n} ({100*correct_f/n:.1f}%)")
        print(f"Q8.8 accuracy:     {correct_q}/{n} ({100*correct_q/n:.1f}%)")
        print(f"F32/Q8.8 diverge:  {divergent}/{n} ({100*divergent/n:.1f}%)")

        if divergent / n > 0.01:
            print("⚠  >1% divergence — consider increasing quantization-aware epochs")

    def save(self, filename):
        """Save weights in DNOS Tier 2 binary format."""
        weight_data = bytearray()

        # Weights: w1 (input→hidden1) stored as hidden1 rows of input columns
        for j in range(HIDDEN1_SIZE):
            for i in range(INPUT_SIZE):
                val = int(np.round(self.w1[i, j] * Q88_SCALE))
                val = max(Q88_MIN, min(Q88_MAX, val))
                weight_data += struct.pack('<h', val)

        # w2
        for j in range(HIDDEN2_SIZE):
            for i in range(HIDDEN1_SIZE):
                val = int(np.round(self.w2[i, j] * Q88_SCALE))
                val = max(Q88_MIN, min(Q88_MAX, val))
                weight_data += struct.pack('<h', val)

        # w3
        for j in range(OUTPUT_SIZE):
            for i in range(HIDDEN2_SIZE):
                val = int(np.round(self.w3[i, j] * Q88_SCALE))
                val = max(Q88_MIN, min(Q88_MAX, val))
                weight_data += struct.pack('<h', val)

        assert len(weight_data) == WEIGHT_DATA_SIZE, \
            f"Weight data size mismatch: {len(weight_data)} vs {WEIGHT_DATA_SIZE}"

        # CRC32
        crc = crc32(weight_data)

        # Build header (128 bytes)
        header = bytearray(HEADER_SIZE)
        header[0:2] = b'DN'
        header[2] = 2                    # Version
        header[3] = 2                    # Tier
        header[4] = 4                    # Num layers
        # Layer sizes as little-endian uint16
        struct.pack_into('<H', header, 5, INPUT_SIZE)
        struct.pack_into('<H', header, 7, HIDDEN1_SIZE)
        struct.pack_into('<H', header, 9, HIDDEN2_SIZE)
        struct.pack_into('<H', header, 11, OUTPUT_SIZE)
        # Activation types: 0=relu, 1=sigmoid
        header[13] = 0                   # Hidden1: ReLU
        header[14] = 0                   # Hidden2: ReLU
        header[15] = 1                   # Output: sigmoid
        # Total weight count
        struct.pack_into('<I', header, 16, TOTAL_WEIGHTS)
        # Weight format: 0=Q8.8
        header[20] = 0
        # CRC32 of weight data
        struct.pack_into('<I', header, 21, crc)

        # Write file
        with open(filename, 'wb') as f:
            f.write(header)
            f.write(weight_data)

            # Pad to sector-aligned size (512-byte boundary)
            current = HEADER_SIZE + WEIGHT_DATA_SIZE
            target = ((current + 511) // 512) * 512
            f.write(bytes(target - current))

        total = HEADER_SIZE + WEIGHT_DATA_SIZE
        print(f"\nSaved weights to {filename}")
        print(f"  Header:  {HEADER_SIZE} bytes")
        print(f"  Weights: {WEIGHT_DATA_SIZE:,} bytes ({TOTAL_WEIGHTS:,} int16)")
        print(f"  Total:   {total:,} bytes")
        print(f"  CRC32:   0x{crc:08X}")

def crc32(data):
    """CRC32 (same as zlib.crc32 but pure Python for portability)."""
    try:
        import zlib
        return zlib.crc32(bytes(data)) & 0xFFFFFFFF
    except ImportError:
        crc = 0xFFFFFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xEDB88320
                else:
                    crc >>= 1
        return crc ^ 0xFFFFFFFF

# ═══════════════════════════════════════════════════════════════════════════════
# Assembly Math Simulator (for validation)
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_assembly_forward(weights_file, input_vec):
    """Simulate the exact assembly forward pass using int16 arithmetic.
    This catches precision issues before they hit bare metal."""

    with open(weights_file, 'rb') as f:
        header = f.read(HEADER_SIZE)
        raw = f.read(WEIGHT_DATA_SIZE)

    # Parse weights as int16
    weights = np.frombuffer(raw, dtype=np.int16)

    # Input as Q8.8
    inp_q = np.round(input_vec * Q88_SCALE).astype(np.int32)
    inp_q = np.clip(inp_q, Q88_MIN, Q88_MAX)

    # Layer 1: Input→Hidden1
    w1_start = 0
    hidden1 = np.zeros(HIDDEN1_SIZE, dtype=np.int32)
    for j in range(HIDDEN1_SIZE):
        acc = 0
        for i in range(INPUT_SIZE):
            w_idx = w1_start + j * INPUT_SIZE + i
            # imul: int16 * int16 → int32, then >> 8
            prod = int(inp_q[i]) * int(weights[w_idx])
            prod >>= 8  # Assembly: sar eax, 8
            acc += prod
        # ReLU
        if acc < 0:
            acc = 0
        if acc > 32767:
            acc = 32767
        hidden1[j] = acc

    # Layer 2: Hidden1→Hidden2
    w2_start = W1_COUNT
    hidden2 = np.zeros(HIDDEN2_SIZE, dtype=np.int32)
    for j in range(HIDDEN2_SIZE):
        acc = 0
        for i in range(HIDDEN1_SIZE):
            w_idx = w2_start + j * HIDDEN1_SIZE + i
            prod = int(hidden1[i]) * int(weights[w_idx])
            prod >>= 8
            acc += prod
        if acc < 0:
            acc = 0
        if acc > 32767:
            acc = 32767
        hidden2[j] = acc

    # Layer 3: Hidden2→Output (piecewise sigmoid)
    w3_start = W1_COUNT + W2_COUNT
    output = np.zeros(OUTPUT_SIZE, dtype=np.int32)
    for j in range(OUTPUT_SIZE):
        acc = 0
        for i in range(HIDDEN2_SIZE):
            w_idx = w3_start + j * HIDDEN2_SIZE + i
            prod = int(hidden2[i]) * int(weights[w_idx])
            prod >>= 8
            acc += prod
        # Piecewise sigmoid (matches assembly)
        if acc < -8192:
            acc = 0
        elif acc > 8192:
            acc = 32767
        else:
            acc = (acc + 8192) << 1
        output[j] = acc

    return output

# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate(weights_file, data):
    """Compare Python network output vs simulated assembly output."""
    print("\n" + "=" * 60)
    print("Assembly Simulation Validation")
    print("=" * 60)

    mismatches = 0
    for inp, expected, desc in data:
        asm_out = simulate_assembly_forward(weights_file, inp)
        asm_cmd = np.argmax(asm_out[:20])
        exp_cmd = np.argmax(expected[:20])

        status = "✓" if asm_cmd == exp_cmd else "✗"
        if asm_cmd != exp_cmd:
            mismatches += 1

        cmd_a = CMD_NAMES[asm_cmd] if asm_cmd < len(CMD_NAMES) else f"cmd{asm_cmd}"
        cmd_e = CMD_NAMES[exp_cmd] if exp_cmd < len(CMD_NAMES) else f"cmd{exp_cmd}"

        print(f"  {status} {desc:30s}  asm={cmd_a:12s} exp={cmd_e}")

    n = len(data)
    print(f"\nAssembly accuracy: {n-mismatches}/{n} ({100*(n-mismatches)/n:.1f}%)")

    if mismatches == 0:
        print("✓ Perfect Python↔Assembly agreement")
    else:
        print(f"⚠ {mismatches} mismatches — review Q8.8 precision")

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Ensure UTF-8 output even on legacy Windows consoles (cp1252) so the
    # ✓/✗/→ status glyphs don't crash the build.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description='DNOS Tier 2 Weight Generator')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='Training epochs (default: 3000)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate (default: 0.05)')
    parser.add_argument('--output', type=str, default='weights.bin',
                        help='Output weights file')
    parser.add_argument('--validate', type=str, default=None,
                        help='Validate existing weights file')
    parser.add_argument('--seed', type=int, default=1337,
                        help='RNG seed for reproducible canonical weights '
                             '(default: 1337)')
    args = parser.parse_args()

    # Deterministic NOS: fixed seed → identical "canonical" weights every build.
    np.random.seed(args.seed)

    # Generate training data
    data = generate_training_data()
    print(f"Generated {len(data)} training examples")
    print(f"Network: {' → '.join(str(s) for s in LAYER_SIZES)}")
    print(f"Weights: {TOTAL_WEIGHTS:,}")

    if args.validate:
        validate(args.validate, data)
        return

    # Train
    net = Tier2Network()
    net.train(data, epochs=args.epochs, lr=args.lr)
    net.test(data)
    net.save(args.output)

    # Validate against assembly simulation
    validate(args.output, data)

    print("\n" + "=" * 60)
    print("Build instructions:")
    print("=" * 60)
    print("  nasm -f bin src/dnos.asm -o dnos.img")
    print(f"  dd if={args.output} of=dnos.img bs=512 seek=69 conv=notrunc")
    print("  qemu-system-i386 -fda dnos.img -m 16M")

if __name__ == '__main__':
    main()
