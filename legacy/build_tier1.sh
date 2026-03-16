#!/bin/bash
#===============================================================================
# DNOS Unified Build Script
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[DNOS]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Check dependencies
check_deps() {
    command -v nasm >/dev/null || error "nasm not found. Install with: apt install nasm"
    command -v python3 >/dev/null || error "python3 not found"
    python3 -c "import numpy" 2>/dev/null || error "numpy not found. Install with: pip install numpy"
    log "Dependencies OK"
}

# Build
build() {
    log "Assembling dnos.asm..."
    nasm -f bin dnos.asm -o dnos.img
    
    log "Training neural network..."
    python3 train.py --epochs 2000 --output weights.bin
    
    log "Patching weights into image..."
    # Weights go at sector 11 (byte offset 5120 = 10 * 512)
    dd if=weights.bin of=dnos.img bs=512 seek=10 conv=notrunc 2>/dev/null
    
    # Pad to floppy size (optional, for compatibility)
    truncate -s 1474560 dnos.img 2>/dev/null || true
    
    log "Build complete: dnos.img ($(wc -c < dnos.img) bytes)"
}

# Run in QEMU
run() {
    if [ ! -f dnos.img ]; then
        error "dnos.img not found. Run: $0 build"
    fi
    
    log "Starting QEMU..."
    qemu-system-i386 -fda dnos.img
}

# Clean
clean() {
    rm -f dnos.img weights.bin
    log "Cleaned"
}

# Help
help() {
    cat << EOF
DNOS Build System

Usage: $0 <command>

Commands:
    build   - Build dnos.img (assemble + train + patch)
    run     - Run in QEMU
    clean   - Remove build artifacts
    help    - Show this help

Example:
    $0 build && $0 run
EOF
}

# Main
case "${1:-help}" in
    build)  check_deps && build ;;
    run)    run ;;
    clean)  clean ;;
    help)   help ;;
    *)      error "Unknown command: $1" ;;
esac
