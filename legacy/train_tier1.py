#!/usr/bin/env python3
"""
DNOS Weight Generator - Tier 1
Generates trained weights for the unified DNOS build.

Usage:
    python3 train.py                    # Generate default weights
    python3 train.py --epochs 5000      # Train longer
    python3 train.py --output my.bin    # Custom output file
"""

import numpy as np
import struct
import argparse

# Network topology (must match dnos.asm)
INPUT_SIZE = 64      # 8 timesteps × 8 bits
HIDDEN_SIZE = 32
OUTPUT_SIZE = 16
TOTAL_WEIGHTS = (INPUT_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE)  # 2560

# Commands (must match dnos.asm)
CMD_NOP = 0
CMD_PIXEL = 1
CMD_HLINE = 2
CMD_VLINE = 3
CMD_RECT = 4
CMD_FILL = 5
CMD_CLEAR = 6

# Key scancodes
KEY_ESC = 27
KEY_SPACE = ord(' ')
KEY_B = ord('b')
KEY_P = ord('p')
KEY_L = ord('l')
KEY_H = ord('h')
KEY_V = ord('v')
KEY_F = ord('f')
KEY_C = ord('c')
KEY_R = ord('r')

def key_to_binary(key):
    """Convert ASCII key to 8-element binary list."""
    return [(key >> i) & 1 for i in range(8)]

def window_to_input(window):
    """Convert 8-key window to 64-element input vector."""
    result = []
    for key in window:
        result.extend(key_to_binary(key))
    return np.array(result, dtype=np.float32)

def command_to_output(cmd, x_delta=0, y_delta=0):
    """Create output vector for command."""
    out = np.zeros(OUTPUT_SIZE, dtype=np.float32)
    out[cmd] = 1.0
    # Encode x/y deltas in neurons 8-11 (simplified)
    out[8] = np.clip(x_delta / 32.0, -1, 1)
    out[9] = np.clip(x_delta / 32.0, -1, 1)
    out[10] = np.clip(y_delta / 32.0, -1, 1)
    out[11] = np.clip(y_delta / 32.0, -1, 1)
    return out

def generate_training_data():
    """Generate training examples."""
    data = []
    
    # Single key commands (most recent key only, rest zero)
    single_key_mappings = [
        (KEY_P, CMD_PIXEL, "P -> pixel"),
        (KEY_B, CMD_RECT, "B -> rect"),
        (KEY_L, CMD_HLINE, "L -> hline"),
        (KEY_H, CMD_HLINE, "H -> hline"),
        (KEY_V, CMD_VLINE, "V -> vline"),
        (KEY_F, CMD_FILL, "F -> fill"),
        (KEY_C, CMD_CLEAR, "C -> clear"),
        (KEY_R, CMD_RECT, "R -> rect"),
        (KEY_SPACE, CMD_NOP, "SPACE -> nop"),
    ]
    
    for key, cmd, desc in single_key_mappings:
        window = [key] + [0] * 7  # Key at position 0, rest empty
        inp = window_to_input(window)
        out = command_to_output(cmd)
        data.append((inp, out, desc))
    
    # Two-key sequences
    sequences = [
        ([KEY_B, KEY_B], CMD_RECT, "BB -> rect"),
        ([KEY_P, KEY_P], CMD_PIXEL, "PP -> pixel"),
        ([KEY_L, KEY_L], CMD_HLINE, "LL -> hline"),
    ]
    
    for seq, cmd, desc in sequences:
        window = seq + [0] * (8 - len(seq))
        inp = window_to_input(window)
        out = command_to_output(cmd)
        data.append((inp, out, desc))
    
    # Word-like patterns (b-o-x for rect)
    words = [
        ([ord('b'), ord('o'), ord('x')], CMD_RECT, "box -> rect"),
        ([ord('l'), ord('i'), ord('n'), ord('e')], CMD_HLINE, "line -> hline"),
        ([ord('d'), ord('o'), ord('t')], CMD_PIXEL, "dot -> pixel"),
        ([ord('c'), ord('l'), ord('r')], CMD_CLEAR, "clr -> clear"),
    ]
    
    for word, cmd, desc in words:
        window = word + [0] * (8 - len(word))
        inp = window_to_input(window)
        out = command_to_output(cmd)
        data.append((inp, out, desc))
    
    return data

class NeuralNetwork:
    """Simple 2-layer network matching DNOS architecture."""
    
    def __init__(self):
        # Xavier initialization
        self.w1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * np.sqrt(2.0 / (INPUT_SIZE + HIDDEN_SIZE))
        self.w2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * np.sqrt(2.0 / (HIDDEN_SIZE + OUTPUT_SIZE))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def forward(self, x):
        self.z1 = np.dot(x, self.w1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, x, y, lr=0.1):
        m = 1
        
        # Output layer
        dz2 = self.a2 - y
        dw2 = np.outer(self.a1, dz2)
        
        # Hidden layer
        dz1 = np.dot(dz2, self.w2.T) * self.a1 * (1 - self.a1)
        dw1 = np.outer(x, dz1)
        
        # Update
        self.w2 -= lr * dw2
        self.w1 -= lr * dw1
        
        return np.mean(dz2 ** 2)
    
    def train(self, data, epochs=1000, lr=0.1):
        """Train on data."""
        print(f"Training on {len(data)} examples for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(data)
            
            for inp, out, _ in data:
                self.forward(inp)
                loss = self.backward(inp, out, lr)
                total_loss += loss
            
            if epoch % 500 == 0:
                print(f"  Epoch {epoch}: loss = {total_loss/len(data):.6f}")
        
        print(f"  Final loss: {total_loss/len(data):.6f}")
    
    def test(self, data):
        """Test and print results."""
        print("\nTest results:")
        correct = 0
        for inp, expected, desc in data:
            out = self.forward(inp)
            pred_cmd = np.argmax(out[:7])  # Only first 7 are commands
            exp_cmd = np.argmax(expected[:7])
            
            status = "✓" if pred_cmd == exp_cmd else "✗"
            if pred_cmd == exp_cmd:
                correct += 1
            
            print(f"  {status} {desc}: predicted {pred_cmd}, expected {exp_cmd}")
        
        print(f"\nAccuracy: {correct}/{len(data)} ({100*correct/len(data):.1f}%)")
    
    def save(self, filename):
        """Save weights in DNOS format."""
        with open(filename, 'wb') as f:
            # Header (64 bytes)
            header = bytearray(64)
            header[0:2] = b'DN'             # Magic
            header[2] = 1                    # Version
            header[3] = 1                    # Tier
            header[4] = 3                    # Num layers
            header[5] = INPUT_SIZE           # Layer 0 size
            header[6] = HIDDEN_SIZE          # Layer 1 size
            header[7] = OUTPUT_SIZE          # Layer 2 size
            header[14:16] = struct.pack('<H', TOTAL_WEIGHTS)
            header[16] = 0                   # Activation type (sigmoid)
            header[17] = 32                  # Learning rate
            f.write(header)
            
            # Weights: input -> hidden (row-major)
            for j in range(HIDDEN_SIZE):
                for i in range(INPUT_SIZE):
                    # Scale to int16 range
                    val = int(self.w1[i, j] * 256)
                    val = max(-32768, min(32767, val))
                    f.write(struct.pack('<h', val))
            
            # Weights: hidden -> output
            for j in range(OUTPUT_SIZE):
                for i in range(HIDDEN_SIZE):
                    val = int(self.w2[i, j] * 256)
                    val = max(-32768, min(32767, val))
                    f.write(struct.pack('<h', val))
            
            # Pad to 8KB (to match dnos.asm expectation)
            current_size = 64 + TOTAL_WEIGHTS * 2
            padding = 8192 - current_size
            f.write(bytes(padding))
        
        print(f"Saved weights to {filename} ({8192} bytes)")

def main():
    parser = argparse.ArgumentParser(description='DNOS Weight Generator')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--output', type=str, default='weights.bin', help='Output file')
    parser.add_argument('--test-only', action='store_true', help='Test without training')
    args = parser.parse_args()
    
    # Generate data
    data = generate_training_data()
    print(f"Generated {len(data)} training examples")
    
    # Expand data with repetitions
    expanded_data = data * 50  # 50 repetitions of each pattern
    
    # Create and train network
    net = NeuralNetwork()
    
    if not args.test_only:
        net.train(expanded_data, epochs=args.epochs, lr=args.lr)
    
    # Test
    net.test(data)
    
    # Save
    net.save(args.output)
    
    print("\nTo build DNOS:")
    print("  1. nasm -f bin dnos.asm -o dnos.img")
    print("  2. dd if=weights.bin of=dnos.img bs=512 seek=10 conv=notrunc")
    print("  3. qemu-system-i386 -fda dnos.img")

if __name__ == '__main__':
    main()
