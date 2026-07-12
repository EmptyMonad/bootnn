/* DNOS RISC-V law runner - qemu -M virt loads us at RAM base.
 * Everything (code, data, stack) lives below 0x8020_0000; the state
 * block the harness reads over QMP is at 0x8020_0000 (see main.rs),
 * and the v5 weight blob is supplied verbatim by QEMU's
 * `-device loader` at 0x8040_0000. */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY {
  RAM : ORIGIN = 0x80000000, LENGTH = 2M
}

SECTIONS {
  .text : { KEEP(*(.text.init)) *(.text*) } > RAM
  .rodata : { *(.rodata*) *(.srodata*) } > RAM
  .data : { *(.data*) *(.sdata*) } > RAM
  .bss : { *(.bss*) *(.sbss*) *(COMMON) } > RAM
  /DISCARD/ : { *(.eh_frame*) }
}
