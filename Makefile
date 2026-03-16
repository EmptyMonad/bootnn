#===============================================================================
# DNOS Tier 2 — Unified Makefile
#
# Targets:
#   make          — Build dnos.img (assemble + train + patch weights)
#   make run      — Build and launch in QEMU
#   make train    — Train weights only
#   make validate — Train + validate assembly math simulation
#   make clean    — Remove build artifacts
#   make test     — Quick build + QEMU with serial output
#===============================================================================

# Tools
ASM      := nasm
PYTHON   := python3
QEMU     := qemu-system-i386
DD       := dd

# Source files
ASM_SRC  := src/dnos.asm
TRAIN_SRC := tools/train.py

# Output files
IMAGE    := dnos.img
WEIGHTS  := weights.bin

# Build parameters
EPOCHS   := 3000
LR       := 0.05

# Weight sector offset in image
# Stage1=1 sector, Stage2=4 sectors, Kernel=64 sectors → weights at sector 70 (offset=69)
WEIGHT_SECTOR := 69

# QEMU flags
QEMU_FLAGS := -m 16M -serial stdio

.PHONY: all run train validate clean test help

#--- Default target ---
all: $(IMAGE)
	@echo ""
	@echo "╔══════════════════════════════════════════╗"
	@echo "║  DNOS Tier 2 — Build Complete            ║"
	@echo "║  Image: $(IMAGE) ($$(wc -c < $(IMAGE)) bytes)  ║"
	@echo "║  Run:   make run                         ║"
	@echo "╚══════════════════════════════════════════╝"

#--- Build image: assemble + train + patch ---
$(IMAGE): $(ASM_SRC) $(WEIGHTS)
	@echo "[DNOS] Assembling $(ASM_SRC)..."
	$(ASM) -f bin $(ASM_SRC) -o $(IMAGE)
	@echo "[DNOS] Patching weights at sector $(WEIGHT_SECTOR)..."
	$(DD) if=$(WEIGHTS) of=$(IMAGE) bs=512 seek=$(WEIGHT_SECTOR) conv=notrunc 2>/dev/null
	@echo "[DNOS] Padding to floppy size..."
	truncate -s 1474560 $(IMAGE) 2>/dev/null || true
	@echo "[DNOS] Build complete: $(IMAGE)"

#--- Train weights ---
$(WEIGHTS): $(TRAIN_SRC)
	@echo "[DNOS] Training neural network ($(EPOCHS) epochs)..."
	$(PYTHON) $(TRAIN_SRC) --epochs $(EPOCHS) --lr $(LR) --output $(WEIGHTS)

train: $(TRAIN_SRC)
	$(PYTHON) $(TRAIN_SRC) --epochs $(EPOCHS) --lr $(LR) --output $(WEIGHTS)

#--- Validate: train + simulate assembly math ---
validate: $(WEIGHTS)
	$(PYTHON) $(TRAIN_SRC) --validate $(WEIGHTS)

#--- Run in QEMU ---
run: $(IMAGE)
	@echo "[DNOS] Starting QEMU..."
	$(QEMU) -fda $(IMAGE) $(QEMU_FLAGS)

#--- Quick test: build + run with serial debug ---
test: $(IMAGE)
	@echo "[DNOS] Testing with serial output..."
	$(QEMU) -fda $(IMAGE) $(QEMU_FLAGS) -no-reboot -d int 2>qemu_debug.log &
	@echo "[DNOS] QEMU PID: $$!"
	@echo "[DNOS] Debug log: qemu_debug.log"

#--- Clean ---
clean:
	rm -f $(IMAGE) $(WEIGHTS) qemu_debug.log
	@echo "[DNOS] Cleaned"

#--- Help ---
help:
	@echo "DNOS Tier 2 Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make          Build dnos.img (assemble + train + patch)"
	@echo "  make run      Build and launch in QEMU"
	@echo "  make train    Train weights only"
	@echo "  make validate Train + validate assembly simulation"
	@echo "  make clean    Remove build artifacts"
	@echo "  make test     Build + QEMU with debug output"
	@echo ""
	@echo "Parameters:"
	@echo "  EPOCHS=N      Training epochs (default: 3000)"
	@echo "  LR=F          Learning rate (default: 0.05)"
	@echo ""
	@echo "Requirements:"
	@echo "  nasm           apt install nasm"
	@echo "  python3+numpy  apt install python3-numpy"
	@echo "  qemu           apt install qemu-system-x86"
	@echo ""
	@echo "Example:"
	@echo "  make EPOCHS=5000 && make run"
