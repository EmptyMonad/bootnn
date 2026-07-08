;===============================================================================
; DNOS Tier 2 — Deterministic Neural Operating System
;
; Unified build: boot1 + boot2 + neural core + demo mode
;
; Features:
;   - Two-stage boot: real mode → 32-bit protected mode
;   - VESA 800×600×32bpp with VGA 320×200 fallback
;   - IRQ-driven keyboard (ring buffer, 256 entries)
;   - PIT timer at 100Hz for preemptive scheduling
;   - 4-layer neural network: 256→128→64→32
;   - 43,008 Q8.8 fixed-point weights
;   - Auto-demo on boot (proves substrate is alive)
;   - DMA-safe multi-chunk disk loading
;
; Build:
;   nasm -f bin src/dnos.asm -o dnos.img
;   python3 tools/train_tier2.py --output weights.bin
;   dd if=weights.bin of=dnos.img bs=512 seek=70 conv=notrunc
;   qemu-system-i386 -fda dnos.img -m 16M
;
; Memory Layout (Physical):
;   0x00007C00  Stage 1 boot (512B)
;   0x00007E00  Stage 2 boot (2KB, sectors 2-5)
;   0x00008000  Kernel + neural core (32KB, sectors 6-69)
;   0x00010000  Weights (86KB, sectors 70-239)
;   0x00020000  Activations / scratch (32KB)
;   0x00030000  Input history buffer (4KB)
;   0x00070000  Keyboard ring buffer (4KB)
;   0x000A0000  VGA framebuffer (legacy)
;   LFB addr    VESA linear framebuffer (if available)
;
;===============================================================================

;===============================================================================
; SECTION 1: STAGE 1 BOOTLOADER (512 bytes, sector 1)
;===============================================================================

[bits 16]
[org 0x7C00]

STAGE2_SECTORS  equ 4
STAGE2_ADDR     equ 0x7E00
STAGE2_LBA      equ 1           ; LBA of stage 2 (sector 0 = stage 1)
KERNEL_SECTORS  equ 48          ; sectors to load (code+font+CRC table; ceiling 61 before WEIGHT_BASE)
KERNEL_ADDR     equ 0x8600      ; MUST equal linked pm_entry (org 0x7C00 + 0xA00)
KERNEL_LBA      equ 5           ; LBA where the kernel image begins
WEIGHTS_LBA     equ 69          ; LBA where the weight blob begins
WEIGHT_SECTORS  equ 3681        ; 128 B header + 1,884,160 B weights, sector-aligned
CHUNK_SECTORS   equ 64          ; 32 KB per INT 13h read into the bounce buffer

boot_entry:
    cli
    cld
    mov [boot_drive], dl

    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00

    ; Print loading message
    mov si, msg_loading
    call print16

    ;--- Verify INT 13h extensions (LBA addressing) are available ---
    mov ah, 0x41
    mov bx, 0x55AA
    mov dl, [boot_drive]
    int 0x13
    jc .no_lba
    cmp bx, 0xAA55
    jne .no_lba

    ;--- Load stage 2 via LBA (AH=42h) to 0x0000:0x7E00 ---
    mov word [dap1_count], STAGE2_SECTORS
    mov word [dap1_off], STAGE2_ADDR
    mov word [dap1_seg], 0x0000
    mov dword [dap1_lba], STAGE2_LBA
    mov ah, 0x42
    mov dl, [boot_drive]
    mov si, dap1
    int 0x13
    jc .disk_err

    ; Pass boot drive to stage 2 and jump
    mov dl, [boot_drive]
    jmp 0x0000:STAGE2_ADDR

.disk_err:
    mov si, msg_disk_err
    call print16
    jmp $

.no_lba:
    mov si, msg_no_lba
    call print16
    jmp $

print16:
    pusha
    mov ah, 0x0E
.lp:
    lodsb
    test al, al
    jz .dn
    int 0x10
    jmp .lp
.dn:
    popa
    ret

boot_drive:   db 0
msg_loading:  db 'DNOS T2...', 13, 10, 0
msg_disk_err: db 'DISK ERR', 0
msg_no_lba:   db 'NO LBA', 0

; Disk Address Packet for stage 1's LBA read
align 4
dap1:         db 0x10          ; packet size
              db 0             ; reserved
dap1_count:   dw 0             ; sectors to transfer
dap1_off:     dw 0             ; dest offset
dap1_seg:     dw 0             ; dest segment
dap1_lba:     dd 0             ; LBA (low 32)
              dd 0             ; LBA (high 32)

times 510 - ($ - boot_entry) db 0
dw 0xAA55

;===============================================================================
; SECTION 2: STAGE 2 BOOTLOADER (sectors 2-5, 2KB)
; VESA setup, A20 enable, GDT, switch to protected mode
;===============================================================================

[bits 16]

stage2_entry:
    cli
    mov [s2_boot_drive], dl

    ;--- Load the 32-bit kernel via LBA to its linked address (0x8600) ---
    ; Must happen here (not stage 1) so it lands just past stage 2 instead
    ; of overlapping it. Uses INT 13h AH=42h (LBA), so no CHS limits.
    call load_kernel

    ;--- Enable A20 (needed before any copy above 1 MB) ---
    call enable_a20

    ;--- Load weights high: INT 13h → bounce buffer → unreal-mode copy ---
    ; 3,681 sectors land at 0x200000 in 32 KB chunks. Must happen before
    ; the PM switch because we need BIOS INT 13h for the reads.
    call load_weights_high

    ;--- VESA video mode ---
    call setup_vesa

    ;--- Load GDT ---
    lgdt [gdt_descriptor]

    ;--- Switch to protected mode ---
    mov eax, cr0
    or eax, 1
    mov cr0, eax

    jmp 0x08:pm_entry

;---------------------------------------
; load_kernel — KERNEL_SECTORS from KERNEL_LBA → 0x0000:KERNEL_ADDR (0x8600)
; Lands at the kernel's linked address, below 0x10000, single read.
;---------------------------------------
load_kernel:
    pusha
    mov ax, KERNEL_SECTORS
    mov bx, 0x0000           ; dest segment
    mov dx, KERNEL_ADDR      ; dest offset
    mov cx, KERNEL_LBA       ; LBA (low 16)
    call disk_read
    popa
    ret

;---------------------------------------
; enter_unreal — flat 4 GB cached limits on DS/ES while staying in RM.
; Loading a segment register in real mode updates the base but leaves
; the cached limit untouched: after this brief PM round-trip, 32-bit
; addresses work with a32 overrides. Interrupts stay off (stage 2 runs
; under cli throughout; SeaBIOS INT 13h tolerates IF=0).
;---------------------------------------
enter_unreal:
    push ds
    push es
    lgdt [gdt_descriptor]
    mov eax, cr0
    or al, 1
    mov cr0, eax
    jmp short .pm
.pm:
    mov bx, 0x10             ; flat 4 GB data descriptor
    mov ds, bx
    mov es, bx
    and al, 0xFE
    mov cr0, eax
    jmp short .rm
.rm:
    pop es
    pop ds
    ret

;---------------------------------------
; load_weights_high — WEIGHT_SECTORS from WEIGHTS_LBA → WEIGHT_BASE (2 MB).
; Loop: INT 13h reads CHUNK_SECTORS into the bounce buffer at 0x10000,
; then an unreal-mode a32 rep movsd lifts the chunk above 1 MB.
; Boot-time only; determinism is untouched — after boot the system is
; exactly the Tier 2 machine with bigger matrices and no disk in the loop.
;---------------------------------------
load_weights_high:
    pusha
    call enter_unreal

    mov word [lw_lba], WEIGHTS_LBA
    mov dword [lw_dest], WEIGHT_BASE
    mov word [lw_left], WEIGHT_SECTORS

.lw_loop:
    mov ax, [lw_left]
    cmp ax, CHUNK_SECTORS
    jbe .lw_sized
    mov ax, CHUNK_SECTORS
.lw_sized:
    mov [lw_this], ax

    ; Read chunk → bounce buffer
    mov bx, BOUNCE_BASE >> 4 ; dest segment 0x1000
    xor dx, dx               ; dest offset 0
    mov cx, [lw_lba]
    call disk_read

    ; Copy bounce → high destination (flat, unreal)
    push ds
    push es
    xor bx, bx
    mov ds, bx
    mov es, bx
    mov esi, BOUNCE_BASE
    mov edi, [lw_dest]
    movzx ecx, word [lw_this]
    shl ecx, 7               ; sectors × 512 / 4 = dwords
    a32 rep movsd
    pop es
    pop ds

    ; Advance LBA / destination / remaining
    mov ax, [lw_this]
    add [lw_lba], ax
    movzx eax, word [lw_this]
    shl eax, 9
    add [lw_dest], eax
    mov ax, [lw_this]
    sub [lw_left], ax
    jnz .lw_loop

    popa
    ret

lw_lba:  dw 0
lw_left: dw 0
lw_this: dw 0
lw_dest: dd 0

;---------------------------------------
; disk_read — INT 13h AH=42h LBA read
;   AX = sector count, BX = dest segment, DX = dest offset, CX = LBA (low 16)
;---------------------------------------
disk_read:
    push si
    mov [dap_count], ax
    mov [dap_seg], bx
    mov [dap_off], dx
    mov [dap_lba], cx
    mov word [dap_lba+2], 0
    mov ah, 0x42
    mov dl, [s2_boot_drive]
    mov si, dap
    int 0x13
    jc .err
    pop si
    ret
.err:
    mov si, msg_wt_err
    call print16_s2
    jmp $

msg_wt_err: db 'WT ERR', 0

align 4
dap:        db 0x10          ; packet size
            db 0             ; reserved
dap_count:  dw 0
dap_off:    dw 0
dap_seg:    dw 0
dap_lba:    dd 0             ; LBA low 32
            dd 0             ; LBA high 32

;---------------------------------------
; VESA Setup
;---------------------------------------
%macro DISPI_W 2
    mov dx, 0x1CE
    mov ax, %1
    out dx, ax
    mov dx, 0x1CF
    mov ax, %2
    out dx, ax
%endmacro

;---------------------------------------
; pci_find_vga - scan bus 0 for a display-class (0x03) device,
; return EAX = BAR0 physical base (memory BAR, masked), or 0.
; Config mechanism #1 via 0xCF8/0xCFC. 32-bit I/O is fine in RM on i386+.
;---------------------------------------
pci_find_vga:
    push ebx
    push edx
    xor ebx, ebx             ; device 0..31, function 0
.pf_dev:
    mov eax, ebx
    shl eax, 11
    or eax, 0x80000008       ; class-code register
    mov dx, 0xCF8
    out dx, eax
    mov dx, 0xCFC
    in eax, dx
    shr eax, 24
    cmp al, 0x03             ; display controller
    jne .pf_next
    mov eax, ebx
    shl eax, 11
    or eax, 0x80000010       ; BAR0
    mov dx, 0xCF8
    out dx, eax
    mov dx, 0xCFC
    in eax, dx
    test al, 1               ; must be a memory BAR
    jnz .pf_next
    and eax, 0xFFFFFFF0
    jmp .pf_done
.pf_next:
    inc ebx
    cmp ebx, 32
    jb .pf_dev
    xor eax, eax
.pf_done:
    pop edx
    pop ebx
    ret

VESA_MODE_800x600x32 equ 0x4115

setup_vesa:
    pusha

    ; Query mode info
    mov ax, 0x4F01
    mov cx, 0x0115           ; Mode without LFB bit for query
    mov di, vesa_info_buf
    int 0x10

    cmp ax, 0x004F
    jne .vesa_fallback

    ; Check mode is supported (bit 0 of mode attributes)
    test byte [vesa_info_buf], 1
    jz .vesa_fallback

    ; Store LFB address
    mov eax, [vesa_info_buf + 40]
    mov [lfb_addr], eax

    ; Store resolution
    mov ax, [vesa_info_buf + 18]
    mov [scr_width], ax
    mov ax, [vesa_info_buf + 20]
    mov [scr_height], ax
    mov al, [vesa_info_buf + 25]
    mov [scr_bpp], al
    mov ax, [vesa_info_buf + 16]     ; BytesPerScanLine (true pitch)
    mov [scr_pitch], ax

    ; The kernel's drawing code is 32bpp-only. If this mode isn't 32bpp
    ; (VBE 0x115 is 24bpp on many BIOSes), fall back to VGA rather than
    ; shear every scanline.
    cmp byte [scr_bpp], 32
    jne .vesa_fallback

    ; Set the mode
    mov ax, 0x4F02
    mov bx, VESA_MODE_800x600x32
    int 0x10
    cmp ax, 0x004F
    jne .vesa_fallback

    mov byte [video_mode], 1     ; VESA active
    popa
    ret

.vesa_fallback:
    ;--- Bochs DISPI (QEMU stdvga / Bochs / VirtualBox) ---------------
    ; SeaBIOS only offers 24bpp VBE at 800x600; the DISPI interface
    ; gives us a true 32bpp linear framebuffer. Probe ID register.
    mov dx, 0x1CE
    xor ax, ax               ; VBE_DISPI_INDEX_ID
    out dx, ax
    mov dx, 0x1CF
    in ax, dx
    and ax, 0xFFF0
    cmp ax, 0xB0C0
    jne .vga_fallback

    ; LFB physical base = BAR0 of the PCI display controller
    call pci_find_vga
    test eax, eax
    jz .vga_fallback
    mov [lfb_addr], eax

    ; Program 800x600x32 with LFB enabled
    DISPI_W 4, 0x0000        ; ENABLE   <- disabled while reprogramming
    DISPI_W 1, 800           ; XRES
    DISPI_W 2, 600           ; YRES
    DISPI_W 3, 32            ; BPP
    DISPI_W 6, 800           ; VIRT_WIDTH -> pitch = 800*4
    DISPI_W 8, 0             ; X_OFFSET
    DISPI_W 9, 0             ; Y_OFFSET
    DISPI_W 4, 0x0041        ; ENABLE | LFB_ENABLED

    mov word [scr_width], 800
    mov word [scr_height], 600
    mov byte [scr_bpp], 32
    mov word [scr_pitch], 3200
    mov byte [video_mode], 1
    popa
    ret

.vga_fallback:
    ; Fall back to VGA mode 13h (320x200x8bpp)
    mov ax, 0x0013
    int 0x10
    mov byte [video_mode], 0
    mov word [scr_width], 320
    mov word [scr_height], 200
    mov byte [scr_bpp], 8
    mov word [scr_pitch], 320        ; VGA mode 13h: 1 byte/pixel
    mov dword [lfb_addr], 0xA0000

    popa
    ret

;---------------------------------------
; A20 Enable
;---------------------------------------
enable_a20:
    pusha

    ; Method 1: Fast A20
    in al, 0x92
    or al, 2
    out 0x92, al

    ; Method 2: Keyboard controller
    call .a20_wait_cmd
    mov al, 0xAD             ; Disable keyboard
    out 0x64, al
    call .a20_wait_cmd
    mov al, 0xD0             ; Read output port
    out 0x64, al
    call .a20_wait_data
    in al, 0x60
    push ax
    call .a20_wait_cmd
    mov al, 0xD1             ; Write output port
    out 0x64, al
    call .a20_wait_cmd
    pop ax
    or al, 2                ; Set A20 bit
    out 0x60, al
    call .a20_wait_cmd
    mov al, 0xAE             ; Re-enable keyboard
    out 0x64, al
    call .a20_wait_cmd

    popa
    ret

.a20_wait_cmd:
    in al, 0x64
    test al, 2
    jnz .a20_wait_cmd
    ret

.a20_wait_data:
    in al, 0x64
    test al, 1
    jz .a20_wait_data
    ret

;---------------------------------------
; GDT
;---------------------------------------
align 8
gdt_start:
    ; Null descriptor
    dq 0

    ; Code segment: base=0, limit=4GB, 32-bit, execute/read
gdt_code:
    dw 0xFFFF               ; Limit low
    dw 0x0000               ; Base low
    db 0x00                  ; Base mid
    db 10011010b             ; Access: present, ring 0, code, exec/read
    db 11001111b             ; Flags: 4KB granularity, 32-bit + limit high
    db 0x00                  ; Base high

    ; Data segment: base=0, limit=4GB, 32-bit, read/write
gdt_data:
    dw 0xFFFF
    dw 0x0000
    db 0x00
    db 10010010b             ; Access: present, ring 0, data, read/write
    db 11001111b
    db 0x00

gdt_end:

gdt_descriptor:
    dw gdt_end - gdt_start - 1
    dd gdt_start

;---------------------------------------
; Stage 2 data
;---------------------------------------
s2_boot_drive:  db 0
video_mode:     db 0            ; 0=VGA, 1=VESA
scr_width:      dw 0
scr_height:     dw 0
scr_bpp:        db 0
scr_pitch:      dw 0
lfb_addr:       dd 0

print16_s2:
    pusha
    mov ah, 0x0E
.lp:
    lodsb
    test al, al
    jz .dn
    int 0x10
    jmp .lp
.dn:
    popa
    ret

align 16
vesa_info_buf:  times 256 db 0

;---------------------------------------
; Pad stage 2 to exactly 2KB (4 sectors)
;---------------------------------------
times 2560 - ($ - boot_entry) db 0

;===============================================================================
; SECTION 3: 32-BIT KERNEL + NEURAL CORE (sectors 6-69, 32KB)
; Entered from stage 2 after PM switch
;===============================================================================

[bits 32]

;--- Constants ---
WEIGHT_BASE     equ 0x200000    ; Weights: 2 MB mark (1.8 MB blob, loaded high at boot)
BOUNCE_BASE     equ 0x10000     ; 32 KB boot-time bounce buffer (scratch after boot)
ACTIV_BASE      equ 0x60000     ; Activations scratch (moved clear of weights)
HISTORY_BASE    equ 0x30000     ; Input history buffer (above weight region)
KEYBUF_BASE     equ 0x70000     ; Keyboard ring buffer

; Network topology
NET_INPUT       equ 512         ; 64 events × 8 features
NET_HIDDEN1     equ 1024
NET_HIDDEN2     equ 384
NET_OUTPUT      equ 64
NET_HISTORY     equ 64          ; Context events
NET_TOTAL_W     equ (NET_INPUT*NET_HIDDEN1)+(NET_HIDDEN1*NET_HIDDEN2)+(NET_HIDDEN2*NET_OUTPUT)  ; 942,080
NET_FEATURES    equ 8           ; Features per event

; Weight offsets (in bytes, int16 = 2 bytes each)
; Header is 128 bytes
W_HEADER        equ 128
W1_OFF          equ W_HEADER                            ; 512×1024 = 524288 weights
W2_OFF          equ W1_OFF + (NET_INPUT * NET_HIDDEN1 * 2)     ; 1024×384 = 393216
W3_OFF          equ W2_OFF + (NET_HIDDEN1 * NET_HIDDEN2 * 2)   ; 384×64 = 24576

; Activation offsets in ACTIV_BASE
A_INPUT         equ 0
A_HIDDEN1       equ A_INPUT + (NET_INPUT * 2)
A_HIDDEN2       equ A_HIDDEN1 + (NET_HIDDEN1 * 2)
A_OUTPUT        equ A_HIDDEN2 + (NET_HIDDEN2 * 2)

; Commands
CMD_NOP         equ 0
CMD_PIXEL       equ 1
CMD_HLINE       equ 2
CMD_VLINE       equ 3
CMD_RECT        equ 4
CMD_FILLED_RECT equ 5
CMD_CIRCLE      equ 6
CMD_CLEAR       equ 7
CMD_MOVE_UP     equ 8
CMD_MOVE_DOWN   equ 9
CMD_MOVE_LEFT   equ 10
CMD_MOVE_RIGHT  equ 11
CMD_COLOR_NEXT  equ 12
CMD_COLOR_PREV  equ 13
CMD_SIZE_UP     equ 14
CMD_SIZE_DOWN   equ 15
CMD_FILL        equ 16
CMD_TEXT        equ 17
CMD_UNDO        equ 18
CMD_SAVE        equ 19

pm_entry:
    ; Set up data segments
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov esp, 0x90000

    ; Copy video info from stage 2 data area
    ; Stage 2 left video_mode at a known offset in the first 2.5KB
    ; We access it via its physical address
    ; video_mode is at boot_entry + (video_mode - boot_entry)
    ; For simplicity, store copies at kernel data area
    movzx eax, byte [stage2_entry + (video_mode - stage2_entry)]
    mov [k_video_mode], al
    mov eax, [stage2_entry + (lfb_addr - stage2_entry)]
    mov [k_lfb_addr], eax
    movzx eax, word [stage2_entry + (scr_width - stage2_entry)]
    mov [k_scr_width], eax
    movzx eax, word [stage2_entry + (scr_height - stage2_entry)]
    mov [k_scr_height], eax
    movzx eax, byte [stage2_entry + (scr_bpp - stage2_entry)]
    mov [k_scr_bpp], eax
    movzx eax, word [stage2_entry + (scr_pitch - stage2_entry)]
    mov [k_scr_pitch], eax

    ; Start the cursor at screen centre for whatever mode we got.
    ; (The static initializers assumed 800x600; under the VGA fallback
    ; they clamp to the bottom-right corner, inside the status bar.)
    mov eax, [k_scr_width]
    shr eax, 1
    mov [cursor_x], eax
    mov eax, [k_scr_height]
    shr eax, 1
    mov [cursor_y], eax

    ; Validate the weight blob before trusting it as law:
    ; magic, layer sizes, weight count, CRC32 over all 86,016 bytes.
    call validate_weights

    ; Set up IDT
    call setup_idt

    ; Set up PIC (remap IRQs to 32-47)
    call setup_pic

    ; Set up PIT at 20 Hz (Tier 3 inference budget)
    call setup_pit

    ; Set up COM1 (polled event producer; probe-guarded)
    call setup_serial

    ; Clear screen
    call clear_screen

    ; Draw boot banner
    call draw_banner

    ; Enable interrupts
    sti

    ; Run demo sequence to prove the substrate is alive
    call demo_sequence

    ; Then enter interactive main loop
    jmp main_loop

;---------------------------------------
; IDT Setup
;---------------------------------------
setup_idt:
    ; Build 256-entry IDT at 0x00000 (we're in PM, can use low memory)
    mov edi, 0x00000
    mov ecx, 256

    ; Default handler for all
    mov eax, default_isr
    mov edx, eax
    shr edx, 16             ; High 16 bits of offset

.idt_fill:
    mov word [edi], ax       ; Offset low
    mov word [edi+2], 0x08   ; Code segment selector
    mov byte [edi+4], 0      ; Reserved
    mov byte [edi+5], 0x8E   ; Present, ring 0, 32-bit interrupt gate
    mov word [edi+6], dx     ; Offset high
    add edi, 8
    loop .idt_fill

    ; Install specific handlers
    ; IRQ0 (INT 32) = PIT timer
    mov eax, timer_isr
    mov edx, eax
    shr edx, 16
    mov edi, 32 * 8          ; IDT entry 32
    mov word [edi], ax
    mov word [edi+2], 0x08
    mov byte [edi+4], 0
    mov byte [edi+5], 0x8E
    mov word [edi+6], dx

    ; IRQ1 (INT 33) = Keyboard
    mov eax, keyboard_isr
    mov edx, eax
    shr edx, 16
    mov edi, 33 * 8
    mov word [edi], ax
    mov word [edi+2], 0x08
    mov byte [edi+4], 0
    mov byte [edi+5], 0x8E
    mov word [edi+6], dx

    ; Load IDT register
    lidt [idt_descriptor]
    ret

idt_descriptor:
    dw 256 * 8 - 1          ; Limit
    dd 0x00000               ; Base

;---------------------------------------
; PIC Setup — remap IRQs to INT 32-47
;---------------------------------------
setup_pic:
    ; ICW1
    mov al, 0x11
    out 0x20, al             ; Master
    out 0xA0, al             ; Slave

    ; ICW2 — vector offsets
    mov al, 32               ; Master: INT 32-39
    out 0x21, al
    mov al, 40               ; Slave: INT 40-47
    out 0xA1, al

    ; ICW3 — cascade
    mov al, 0x04             ; Master: slave on IRQ2
    out 0x21, al
    mov al, 0x02             ; Slave: cascade identity
    out 0xA1, al

    ; ICW4 — 8086 mode
    mov al, 0x01
    out 0x21, al
    out 0xA1, al

    ; Mask: enable IRQ0 (timer) and IRQ1 (keyboard) only
    mov al, 0xFC             ; 11111100 — IRQ0 and IRQ1 unmasked
    out 0x21, al
    mov al, 0xFF             ; Mask all slave IRQs
    out 0xA1, al

    ret

;---------------------------------------
; PIT Setup — 100Hz timer
;---------------------------------------
setup_pit:
    ; Channel 0, rate generator, lo/hi byte
    mov al, 0x34             ; 00110100b
    out 0x43, al

    ; Divisor for 20 Hz: 1193182 / 20 = 59659. Tier 3's forward pass is
    ; ~22x Tier 2's MAC count; the tick is the measured budget for one
    ; full state transition. Determinism is rate-independent — the
    ; invariant is one transition per tick, not the tick's frequency.
    mov ax, 59659
    out 0x40, al             ; Low byte
    mov al, ah
    out 0x40, al             ; High byte
    ret

;---------------------------------------
; Timer ISR (IRQ0 → INT 32)
;---------------------------------------
timer_isr:
    pushad
    inc dword [tick_count]

    ; Send EOI
    mov al, 0x20
    out 0x20, al
    popad
    iretd

;---------------------------------------
; Keyboard ISR (IRQ1 → INT 33)
; Reads scancode into ring buffer
;---------------------------------------
keyboard_isr:
    pushad
    push ds
    mov ax, 0x10
    mov ds, ax

    ; Read scancode
    in al, 0x60

    ; Ignore key releases (bit 7 set)
    test al, 0x80
    jnz .kb_done

    ; Convert to ASCII at the edge: the ring carries *events*, not
    ; scancodes, so the keyboard and COM1 producers are indistinguishable
    ; to the consumer (SWARM_DESIGN S1). Unmapped keys are discarded here,
    ; exactly as NUL bytes are discarded by serial_drain.
    call scan_to_ascii
    test al, al
    jz .kb_done

    ; Store in ring buffer
    movzx ebx, word [kb_write_idx]
    mov [KEYBUF_BASE + ebx], al
    inc bx
    and bx, 0xFF             ; Wrap at 256
    mov [kb_write_idx], bx

.kb_done:
    ; Send EOI
    mov al, 0x20
    out 0x20, al
    pop ds
    popad
    iretd

;---------------------------------------
; Default ISR
;---------------------------------------
default_isr:
    iretd

;---------------------------------------
; Read key from ring buffer
; Returns: AL = scancode, or 0 if empty
;---------------------------------------
read_key:
    movzx eax, word [kb_read_idx]
    cmp ax, [kb_write_idx]
    je .no_key

    mov al, [KEYBUF_BASE + eax]
    inc word [kb_read_idx]
    and word [kb_read_idx], 0xFF
    ret

.no_key:
    xor eax, eax
    ret

;---------------------------------------
; COM1 setup — polled, no IRQ (SWARM_DESIGN S1, v0 byte framing).
; Scratch-register probe first: on hardware with no UART the bus floats
; 0xFF, LSR would read "data ready" forever, and a blind poll would
; flood the event ring. Absent device => serial_present stays 0 and
; serial_drain is a no-op.
;---------------------------------------
SERIAL_BASE     equ 0x3F8       ; COM1

setup_serial:
    mov dx, SERIAL_BASE + 7     ; scratch register probe
    mov al, 0xA5
    out dx, al
    in al, dx
    cmp al, 0xA5
    jne .no_uart
    mov dx, SERIAL_BASE + 1     ; IER = 0: we poll at the tick boundary
    xor al, al
    out dx, al
    mov dx, SERIAL_BASE + 3     ; LCR: DLAB on
    mov al, 0x80
    out dx, al
    mov dx, SERIAL_BASE         ; divisor 1 = 115200 baud
    mov al, 1
    out dx, al
    mov dx, SERIAL_BASE + 1
    xor al, al
    out dx, al
    mov dx, SERIAL_BASE + 3     ; LCR: 8N1, DLAB off
    mov al, 0x03
    out dx, al
    mov dx, SERIAL_BASE + 2     ; FCR: enable + clear FIFOs, 14-byte trigger
    mov al, 0xC7
    out dx, al
    mov dx, SERIAL_BASE + 4     ; MCR: DTR | RTS
    mov al, 0x03
    out dx, al
    mov byte [serial_present], 1
.no_uart:
    ret

;---------------------------------------
; Drain pending COM1 bytes into the shared event ring (bounded).
; Producer twin of keyboard_isr: bytes arriving on the wire are already
; events (ASCII, v0 framing); NUL is discarded exactly like an unmapped
; scancode. Consumption stays one event per tick in the main loop —
; this only moves bytes from the 16-byte UART FIFO into the 256-byte
; ring so agent-rate bursts are absorbed, not dropped.
; Clobbers: EAX, EBX, ECX, EDX.
;---------------------------------------
serial_drain:
    cmp byte [serial_present], 0
    je .done
    mov ecx, 32                 ; per-tick bound (2x FIFO depth)
.next:
    mov dx, SERIAL_BASE + 5     ; LSR
    in al, dx
    test al, 0x01               ; data ready?
    jz .done
    mov dx, SERIAL_BASE
    in al, dx                   ; RBR
    test al, al                 ; discard NUL
    jz .cont
    movzx ebx, word [kb_write_idx]
    mov [KEYBUF_BASE + ebx], al
    inc bx
    and bx, 0xFF                ; wrap at 256, same ring discipline
    mov [kb_write_idx], bx
    inc dword [serial_rx_count]
.cont:
    loop .next
.done:
    ret

;---------------------------------------
; Scancode → ASCII lookup (simplified)
;---------------------------------------
scan_to_ascii:
    ; Input: AL = scancode
    ; Output: AL = ASCII (or 0 if unmapped)
    cmp al, 58
    ja .unmapped
    movzx eax, al
    mov al, [scancode_table + eax]
    ret
.unmapped:
    xor al, al
    ret

scancode_table:
    db 0, 27                 ; 0x00, 0x01 (ESC)
    db '1234567890-='        ; 0x02-0x0D
    db 8, 9                  ; 0x0E (BS), 0x0F (TAB)
    db 'qwertyuiop[]'        ; 0x10-0x1B
    db 13, 0                 ; 0x1C (Enter), 0x1D (LCtrl)
    db 'asdfghjkl', 0x3B, 0x27  ; 0x1E-0x28 (;')
    db '`', 0                ; 0x29 (`), 0x2A (LShift)
    db '\', 'zxcvbnm,./'     ; 0x2B-0x35
    db 0, '*', 0, ' '        ; 0x36-0x39 (RShift, *, LAlt, Space)

;===============================================================================
; DEMO SEQUENCE
; Runs automatically on boot to prove the neural substrate is alive.
; Feeds canned input through the network and renders the output.
;===============================================================================

demo_sequence:
    pushad

    ; Display "DEMO" indicator
    mov esi, msg_demo
    mov ecx, 4
    mov eax, 80
    mov ebx, 4
    call draw_text

    ; Feed a sequence of synthetic keypresses through the network
    ; This simulates: p, b, o, x, l, i, n, e
    mov esi, demo_keys
    mov ecx, 8

.demo_loop:
    push ecx
    push esi

    ; Encode this key into input history
    movzx eax, byte [esi]
    call encode_input_32

    ; Run neural forward pass
    call neural_forward_32

    ; Decode and execute the output
    call decode_output_32
    inc dword [step_count]

    ; Brief delay (wait ~5 timer ticks = 50ms)
    mov eax, [tick_count]
    add eax, 5
.demo_wait:
    cmp [tick_count], eax
    jb .demo_wait

    pop esi
    pop ecx
    inc esi
    loop .demo_loop

    ; Pin the pen color to a canonical value: the demo's network-driven
    ; color changes must not leave color == background (invisible draws).
    mov byte [color_idx], 15

    ; Switch indicator
    mov esi, msg_interactive
    mov ecx, 11
    mov eax, 80
    mov ebx, 4
    call draw_text

    popad
    ret

demo_keys: db 'p', 'b', 'o', 'x', 'l', 'i', 'n', 'e'
msg_demo: db 'DEMO'
msg_interactive: db 'INTERACTIVE'

;===============================================================================
; MAIN LOOP (interactive mode)
;===============================================================================

main_loop:
    ;--- Deterministic tick pacing -------------------------------------
    ; Block until the PIT advances, then perform exactly one state
    ; transition: S(n+1) = f(S(n), input(n)), consuming at most ONE
    ; input event per tick. step_count <= tick_count is an invariant.
    mov eax, [last_tick]
.tick_wait:
    hlt
    cmp eax, [tick_count]
    je .tick_wait
    mov eax, [tick_count]
    mov [last_tick], eax
    inc dword [step_count]

    ; Drain any pending COM1 bytes into the shared event ring (bounded,
    ; producer side only), then consume at most ONE event this tick.
    ; The ring already holds ASCII: both producers convert at the edge.
    call serial_drain
    call read_key
    test al, al
    jz .no_input

    ; Check ESC
    cmp al, 27
    je .halt

    ; Encode into neural input
    movzx eax, al
    call encode_input_32

    ; Forward pass
    call neural_forward_32

    ; Decode and execute
    call decode_output_32

.no_input:
    ; Draw cursor
    call draw_cursor

    ; Update status bar every 10 ticks
    mov eax, [tick_count]
    and eax, 0x0F
    jnz .no_status
    call draw_status_bar
.no_status:

    jmp main_loop            ; pacing hlt is at the top of the loop

.halt:
    cli
    hlt

;===============================================================================
; INPUT ENCODING (32-bit)
; EAX = ASCII key → shift history, encode as 8-bit binary
;===============================================================================

encode_input_32:
    pushad

    ; Shift history: move events [0..N-2] → [1..N-1]
    ; Each event = NET_FEATURES * 2 bytes = 16 bytes
    mov esi, HISTORY_BASE + (NET_HISTORY - 2) * NET_FEATURES * 2
    mov edi, HISTORY_BASE + (NET_HISTORY - 1) * NET_FEATURES * 2
    mov ecx, (NET_HISTORY - 1) * NET_FEATURES
    std
    rep movsw
    cld

    ; Encode current key at slot 0
    mov edi, HISTORY_BASE
    mov ecx, 8

    ; Feature 0-7: 8-bit binary encoding of ASCII value
.enc_bit:
    shr eax, 1
    jnc .enc_zero
    mov word [edi], 0x0100   ; Q8.8: 1.0 (must match train.py INPUT_ACTIVE_LEVEL)
    jmp .enc_next
.enc_zero:
    mov word [edi], 0x0000   ; Q8.8: 0.0
.enc_next:
    add edi, 2
    loop .enc_bit

    popad
    ret

;===============================================================================
; NEURAL FORWARD PASS (32-bit, 3 weight layers)
; Q8.8 fixed-point: multiply two Q8.8, result >> 8
; Accumulate in 32-bit to avoid overflow
;===============================================================================

neural_forward_32:
    pushad

    ; Copy input history → activation input buffer
    mov esi, HISTORY_BASE
    mov edi, ACTIV_BASE + A_INPUT
    mov ecx, NET_INPUT
    rep movsw

    ;--- Layer 1: Input(256) → Hidden1(128) ---
    mov ebp, 0               ; Neuron index j

.L1_neuron:
    xor ebx, ebx             ; 32-bit accumulator

    ; Weight offset: W_HEADER + j * NET_INPUT * 2 + i * 2
    mov eax, ebp
    imul eax, NET_INPUT * 2
    add eax, W1_OFF
    mov edx, eax             ; EDX = weight pointer offset

    mov esi, ACTIV_BASE + A_INPUT
    mov ecx, NET_INPUT

.L1_sum:
    movsx eax, word [esi]            ; Input activation (Q8.8)
    movsx edi, word [WEIGHT_BASE + edx]  ; Weight (Q8.8)
    imul eax, edi                    ; EAX = a * w (Q16.16); two-operand form —
                                     ; one-operand imul clobbers EDX (weight ptr)
    sar eax, 8                       ; → Q8.8
    add ebx, eax                     ; Accumulate
    add esi, 2
    add edx, 2
    loop .L1_sum

    ; ReLU activation
    test ebx, ebx
    jns .L1_pos
    xor ebx, ebx
.L1_pos:
    ; Clamp to int16 range
    cmp ebx, 32767
    jle .L1_clamp_ok
    mov ebx, 32767
.L1_clamp_ok:

    ; Store
    mov eax, ebp
    shl eax, 1
    mov [ACTIV_BASE + A_HIDDEN1 + eax], bx

    inc ebp
    cmp ebp, NET_HIDDEN1
    jb .L1_neuron

    ;--- Layer 2: Hidden1(128) → Hidden2(64) ---
    mov ebp, 0

.L2_neuron:
    xor ebx, ebx

    mov eax, ebp
    imul eax, NET_HIDDEN1 * 2
    add eax, W2_OFF
    mov edx, eax

    mov esi, ACTIV_BASE + A_HIDDEN1
    mov ecx, NET_HIDDEN1

.L2_sum:
    movsx eax, word [esi]
    movsx edi, word [WEIGHT_BASE + edx]
    imul eax, edi                    ; two-operand: must not clobber EDX
    sar eax, 8
    add ebx, eax
    add esi, 2
    add edx, 2
    loop .L2_sum

    ; ReLU
    test ebx, ebx
    jns .L2_pos
    xor ebx, ebx
.L2_pos:
    cmp ebx, 32767
    jle .L2_clamp_ok
    mov ebx, 32767
.L2_clamp_ok:

    mov eax, ebp
    shl eax, 1
    mov [ACTIV_BASE + A_HIDDEN2 + eax], bx

    inc ebp
    cmp ebp, NET_HIDDEN2
    jb .L2_neuron

    ;--- Layer 3: Hidden2(64) → Output(32) ---
    mov ebp, 0

.L3_neuron:
    xor ebx, ebx

    mov eax, ebp
    imul eax, NET_HIDDEN2 * 2
    add eax, W3_OFF
    mov edx, eax

    mov esi, ACTIV_BASE + A_HIDDEN2
    mov ecx, NET_HIDDEN2

.L3_sum:
    movsx eax, word [esi]
    movsx edi, word [WEIGHT_BASE + edx]
    imul eax, edi                    ; two-operand: must not clobber EDX
    sar eax, 8
    add ebx, eax
    add esi, 2
    add edx, 2
    loop .L3_sum

    ; Output layer: piecewise sigmoid (not ReLU)
    ; Maps roughly to [0, 32767]
    cmp ebx, -8192
    jl .L3_sat_lo
    cmp ebx, 8192
    jg .L3_sat_hi
    add ebx, 8192
    shl ebx, 1
    jmp .L3_store
.L3_sat_lo:
    xor ebx, ebx
    jmp .L3_store
.L3_sat_hi:
    mov ebx, 32767
.L3_store:
    mov eax, ebp
    shl eax, 1
    mov [ACTIV_BASE + A_OUTPUT + eax], bx

    inc ebp
    cmp ebp, NET_OUTPUT
    jb .L3_neuron

    popad
    ret

;===============================================================================
; OUTPUT DECODING
; Argmax over first 20 command outputs, extract cursor delta from 20-31
;===============================================================================

decode_output_32:
    pushad

    ; Argmax over outputs 0-19 (commands)
    mov esi, ACTIV_BASE + A_OUTPUT
    mov ecx, 20
    xor ebx, ebx             ; Best index
    movsx edx, word [esi]    ; Best value
    xor eax, eax

.find_max:
    movsx edi, word [esi]
    cmp edi, edx
    jle .not_better
    mov edx, edi
    mov ebx, eax
.not_better:
    add esi, 2
    inc eax
    loop .find_max

    ; Store last command and confidence
    mov [last_cmd], ebx
    mov [last_confidence], edx

    ; Extract cursor deltas from outputs 20-23. The piecewise sigmoid
    ; maps "zero delta" to its midpoint 16384, not 0 — subtract it
    ; first or every inference drifts the cursor +16,+16.
    movsx eax, word [ACTIV_BASE + A_OUTPUT + 20*2]
    sub eax, 16384
    sar eax, 10              ; Scale to ±16
    add [cursor_x], eax

    movsx eax, word [ACTIV_BASE + A_OUTPUT + 22*2]
    sub eax, 16384
    sar eax, 10
    add [cursor_y], eax

    ; Clamp cursor
    cmp dword [cursor_x], 0
    jge .cx_ok
    mov dword [cursor_x], 0
.cx_ok:
    mov eax, [k_scr_width]
    dec eax
    cmp [cursor_x], eax
    jle .cx_ok2
    mov [cursor_x], eax
.cx_ok2:
    cmp dword [cursor_y], 0
    jge .cy_ok
    mov dword [cursor_y], 0
.cy_ok:
    mov eax, [k_scr_height]
    sub eax, 21              ; Stay above the 20px status bar, which
                             ; repaints every 16 ticks and would erase
                             ; anything drawn inside it
    cmp [cursor_y], eax
    jle .cy_ok2
    mov [cursor_y], eax
.cy_ok2:

    ; Dispatch command
    cmp ebx, CMD_NOP
    je .cmd_done
    cmp ebx, CMD_PIXEL
    je .cmd_pixel
    cmp ebx, CMD_HLINE
    je .cmd_hline
    cmp ebx, CMD_VLINE
    je .cmd_vline
    cmp ebx, CMD_RECT
    je .cmd_rect
    cmp ebx, CMD_FILLED_RECT
    je .cmd_filled_rect
    cmp ebx, CMD_CLEAR
    je .cmd_clear
    cmp ebx, CMD_MOVE_UP
    je .cmd_mv_up
    cmp ebx, CMD_MOVE_DOWN
    je .cmd_mv_down
    cmp ebx, CMD_MOVE_LEFT
    je .cmd_mv_left
    cmp ebx, CMD_MOVE_RIGHT
    je .cmd_mv_right
    cmp ebx, CMD_COLOR_NEXT
    je .cmd_color_next
    cmp ebx, CMD_FILL
    je .cmd_fill
    jmp .cmd_done

.cmd_pixel:
    call draw_pixel_32
    jmp .cmd_done
.cmd_hline:
    mov ecx, [draw_size]
    call draw_hline_32
    jmp .cmd_done
.cmd_vline:
    mov ecx, [draw_size]
    call draw_vline_32
    jmp .cmd_done
.cmd_rect:
    mov ecx, [draw_size]
    mov edx, [draw_size]
    call draw_rect_32
    jmp .cmd_done
.cmd_filled_rect:
    mov ecx, [draw_size]
    mov edx, [draw_size]
    call draw_filled_rect_32
    jmp .cmd_done
.cmd_clear:
    call clear_screen
    jmp .cmd_done
.cmd_mv_up:
    sub dword [cursor_y], 10
    cmp dword [cursor_y], 0
    jge .cmd_done
    mov dword [cursor_y], 0
    jmp .cmd_done
.cmd_mv_down:
    add dword [cursor_y], 10
    mov eax, [k_scr_height]
    sub eax, 21
    cmp [cursor_y], eax
    jle .cmd_done
    mov [cursor_y], eax
    jmp .cmd_done
.cmd_mv_left:
    sub dword [cursor_x], 10
    cmp dword [cursor_x], 0
    jge .cmd_done
    mov dword [cursor_x], 0
    jmp .cmd_done
.cmd_mv_right:
    add dword [cursor_x], 10
    mov eax, [k_scr_width]
    dec eax
    cmp [cursor_x], eax
    jle .cmd_done
    mov [cursor_x], eax
    jmp .cmd_done
.cmd_color_next:
    inc byte [color_idx]
    and byte [color_idx], 0x0F
    jmp .cmd_done
.cmd_fill:
    call fill_screen_32
    jmp .cmd_done

.cmd_done:
    popad
    ret

;===============================================================================
; GRAPHICS PRIMITIVES (32-bit, VESA-aware)
;===============================================================================

; Calculate pixel offset in LFB
; Input: EAX=x, EBX=y
; Output: EDI=byte offset
pixel_offset:
    push eax
    push ebx
    push edx

    ; offset = y * scr_width * (bpp/8) + x * (bpp/8)
    cmp byte [k_video_mode], 1
    je .vesa_calc

    ; VGA: offset = y * 320 + x
    imul ebx, 320
    add ebx, eax
    mov edi, [k_lfb_addr]
    add edi, ebx
    pop edx
    pop ebx
    pop eax
    ret

.vesa_calc:
    ; VESA 32bpp: offset = y * pitch + x * 4
    imul ebx, [k_scr_pitch]
    shl eax, 2               ; x * 4 for 32bpp
    add ebx, eax
    mov edi, [k_lfb_addr]
    add edi, ebx
    pop edx
    pop ebx
    pop eax
    ret

; Get color value for current mode
; Output: EAX = color value
get_color:
    movzx eax, byte [color_idx]
    cmp byte [k_video_mode], 1
    je .vesa_color
    ret                      ; VGA: palette index directly

.vesa_color:
    ; Look up 32-bit ARGB from palette
    shl eax, 2
    add eax, color_palette
    mov eax, [eax]
    ret

; 16-color ARGB palette
color_palette:
    dd 0x00000000            ; 0  Black
    dd 0x000000AA            ; 1  Blue
    dd 0x0000AA00            ; 2  Green
    dd 0x0000AAAA            ; 3  Cyan
    dd 0x00AA0000            ; 4  Red
    dd 0x00AA00AA            ; 5  Magenta
    dd 0x00AA5500            ; 6  Brown
    dd 0x00AAAAAA            ; 7  Light gray
    dd 0x00555555            ; 8  Dark gray
    dd 0x005555FF            ; 9  Light blue
    dd 0x0055FF55            ; 10 Light green
    dd 0x0055FFFF            ; 11 Light cyan
    dd 0x00FF5555            ; 12 Light red
    dd 0x00FF55FF            ; 13 Light magenta
    dd 0x00FFFF55            ; 14 Yellow
    dd 0x00FFFFFF            ; 15 White

; Draw single pixel at cursor
draw_pixel_32:
    pushad
    mov eax, [cursor_x]
    mov ebx, [cursor_y]
    call pixel_offset
    call get_color
    cmp byte [k_video_mode], 1
    je .vesa_px
    mov [edi], al            ; VGA 8bpp
    popad
    ret
.vesa_px:
    mov [edi], eax           ; VESA 32bpp
    popad
    ret

; Draw horizontal line, ECX = length
draw_hline_32:
    pushad
    mov eax, [cursor_x]
    mov ebx, [cursor_y]
    call pixel_offset
    call get_color

    cmp byte [k_video_mode], 1
    je .vesa_hl
    rep stosb                ; VGA
    popad
    ret
.vesa_hl:
    rep stosd                ; VESA 32bpp
    popad
    ret

; Draw vertical line, ECX = length
draw_vline_32:
    pushad
    mov eax, [cursor_x]
    mov ebx, [cursor_y]
    call pixel_offset
    call get_color

    cmp byte [k_video_mode], 1
    je .vesa_vl

    ; VGA: stride = 320
.vga_vl:
    mov [edi], al
    add edi, 320
    loop .vga_vl
    popad
    ret

.vesa_vl:
    mov edx, [k_scr_pitch]
.vesa_vl_loop:
    mov [edi], eax
    add edi, edx
    loop .vesa_vl_loop
    popad
    ret

; Draw rectangle outline, ECX=width, EDX=height
draw_rect_32:
    pushad
    push ecx
    push edx

    ; Top
    call draw_hline_32

    ; Bottom
    push dword [cursor_y]
    add [cursor_y], edx
    dec dword [cursor_y]
    call draw_hline_32
    pop dword [cursor_y]

    ; Left
    mov ecx, edx
    call draw_vline_32

    ; Right
    mov eax, [cursor_x]
    add eax, [esp+4]         ; + width (ECX clobbered by the Left edge)
    dec eax
    push dword [cursor_x]
    mov [cursor_x], eax
    mov ecx, edx
    call draw_vline_32
    pop dword [cursor_x]

    pop edx
    pop ecx
    popad
    ret

; Draw filled rectangle
draw_filled_rect_32:
    pushad
    push dword [cursor_y]
    mov ebp, edx             ; Height counter

.fill_row:
    push ecx
    call draw_hline_32
    pop ecx
    inc dword [cursor_y]
    dec ebp
    jnz .fill_row

    pop dword [cursor_y]
    popad
    ret

; Clear screen to blue
clear_screen:
    pushad
    mov edi, [k_lfb_addr]

    cmp byte [k_video_mode], 1
    je .vesa_cls

    ; VGA: fill with palette 1 (blue)
    mov ecx, 320 * 200 / 4
    mov eax, 0x01010101
    rep stosd
    popad
    ret

.vesa_cls:
    mov ecx, [k_scr_pitch]
    imul ecx, [k_scr_height]
    shr ecx, 2               ; whole framebuffer (pitch*height) in dwords
    mov eax, 0x00102040      ; Dark blue ARGB
    rep stosd
    popad
    ret

; Fill screen with current color
fill_screen_32:
    pushad
    mov edi, [k_lfb_addr]
    call get_color

    cmp byte [k_video_mode], 1
    je .vesa_fill

    mov ecx, 320 * 200 / 4
    mov ah, al
    mov edx, eax
    shl edx, 16
    or eax, edx
    rep stosd
    popad
    ret

.vesa_fill:
    mov ecx, [k_scr_pitch]
    imul ecx, [k_scr_height]
    shr ecx, 2               ; whole framebuffer in dwords
    rep stosd
    popad
    ret

; Draw cursor crosshair
draw_cursor:
    pushad
    mov eax, [cursor_x]
    mov ebx, [cursor_y]

    ; Horizontal tick
    sub eax, 3
    call pixel_offset
    mov ecx, 7
    cmp byte [k_video_mode], 1
    je .vc_vesa

    mov al, 15               ; White
    rep stosb
    popad
    ret

.vc_vesa:
    mov eax, 0x00FFFFFF
    rep stosd
    popad
    ret

; Draw boot banner
draw_banner:
    pushad
    mov esi, msg_banner
    mov ecx, 7
    mov eax, 4
    mov ebx, 4
    call draw_text
    popad
    ret

msg_banner: db 'DNOS T2'

; Draw status bar at bottom of screen
draw_status_bar:
    pushad
    ; Bottom row: show last command index and cursor position
    ; For now, just draw a colored bar
    mov edi, [k_lfb_addr]
    mov eax, [k_scr_height]
    sub eax, 16
    imul eax, [k_scr_width]

    cmp byte [k_video_mode], 1
    je .sb_vesa

    ; VGA
    imul eax, 1              ; 8bpp
    add edi, eax
    mov ecx, 320 * 16 / 4
    mov eax, 0x07070707      ; Gray
    rep stosd
    call draw_status_text
    popad
    ret

.sb_vesa:
    ; Start at (height-16) * pitch; fill 16 rows of width px, stepping by pitch
    mov eax, [k_scr_height]
    sub eax, 16
    imul eax, [k_scr_pitch]
    add edi, eax
    mov edx, 16              ; rows
.sb_vesa_row:
    push edi
    mov ecx, [k_scr_width]
    mov eax, 0x00333333      ; Dark gray ARGB
    rep stosd
    pop edi
    add edi, [k_scr_pitch]
    dec edx
    jnz .sb_vesa_row
    call draw_status_text
    popad
    ret

; Telemetry line rendered into the status bar:
;   C:cc X:xxxx Y:yyyy T:tttttttt   (all hex)
draw_status_text:
    pushad
    mov ebx, [k_scr_height]
    sub ebx, 12              ; text baseline inside the bar

    mov esi, str_c
    mov ecx, 2
    mov eax, 4
    call draw_text
    mov edx, [last_cmd]
    mov ecx, 2
    mov eax, 4 + 2*8
    call draw_hex32

    mov esi, str_x
    mov ecx, 3
    mov eax, 4 + 4*8
    call draw_text
    mov edx, [cursor_x]
    mov ecx, 4
    mov eax, 4 + 7*8
    call draw_hex32

    mov esi, str_y
    mov ecx, 3
    mov eax, 4 + 11*8
    call draw_text
    mov edx, [cursor_y]
    mov ecx, 4
    mov eax, 4 + 14*8
    call draw_hex32

    mov esi, str_t
    mov ecx, 3
    mov eax, 4 + 18*8
    call draw_text
    mov edx, [tick_count]
    mov ecx, 8
    mov eax, 4 + 21*8
    call draw_hex32

    popad
    ret

str_c: db 'C:'
str_x: db ' X:'
str_y: db ' Y:'
str_t: db ' T:'

;---------------------------------------
; draw_char — render one opaque 8x8 glyph (white on black)
;   DL = ASCII (0x20..0x5A rendered; others blank), EAX = x, EBX = y
;---------------------------------------
draw_char:
    pushad
    call pixel_offset        ; EDI = fb ptr for (x,y)
    movzx esi, dl
    sub esi, 0x20
    cmp esi, 0x5A - 0x20
    jbe .dc_have
    xor esi, esi             ; out of range -> space
.dc_have:
    shl esi, 3
    add esi, font8x8
    mov ecx, 8               ; rows
.dc_row:
    mov dl, [esi]
    inc esi
    push edi
    mov ebp, 8               ; cols
.dc_col:
    cmp byte [k_video_mode], 1
    je .dc_vesa_px
    ; VGA 8bpp
    test dl, 0x80
    jz .dc_vga_bg
    mov byte [edi], 15
    jmp .dc_vga_adv
.dc_vga_bg:
    mov byte [edi], 0
.dc_vga_adv:
    inc edi
    jmp .dc_next
.dc_vesa_px:
    test dl, 0x80
    jz .dc_vesa_bg
    mov dword [edi], 0x00FFFFFF
    jmp .dc_vesa_adv
.dc_vesa_bg:
    mov dword [edi], 0x00000000
.dc_vesa_adv:
    add edi, 4
.dc_next:
    shl dl, 1
    dec ebp
    jnz .dc_col
    pop edi
    cmp byte [k_video_mode], 1
    je .dc_pitch_vesa
    add edi, 320
    jmp .dc_row_next
.dc_pitch_vesa:
    add edi, [k_scr_pitch]
.dc_row_next:
    loop .dc_row
    popad
    ret

;---------------------------------------
; draw_text — ESI = string, ECX = length, EAX = x, EBX = y
;---------------------------------------
draw_text:
    pushad
.dt_loop:
    mov dl, [esi]
    call draw_char
    inc esi
    add eax, 8
    loop .dt_loop
    popad
    ret

;---------------------------------------
; draw_hex32 — EDX = value, ECX = digit count (1..8), EAX = x, EBX = y
;---------------------------------------
draw_hex32:
    pushad
    mov ebp, edx
.dh_loop:
    push ecx
    dec ecx
    shl ecx, 2               ; (digits remaining - 1) * 4
    mov edi, ebp
    shr edi, cl
    pop ecx
    and edi, 0x0F
    mov dl, [hex_chars + edi]
    call draw_char
    add eax, 8
    loop .dh_loop
    popad
    ret

hex_chars: db '0123456789ABCDEF'

; 8x8 bitmap font, chars 0x20..0x5A (glyph = char - 0x20), MSB = left pixel
font8x8:
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; ' '
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '!'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '"'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '#'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '$'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '%'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '&'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '''
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '('
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; ')'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '*'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '+'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; ','
    db 0x00, 0x00, 0x00, 0x7C, 0x00, 0x00, 0x00, 0x00   ; '-'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0x60, 0x00   ; '.'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '/'
    db 0x7C, 0xC6, 0xCE, 0xD6, 0xE6, 0xC6, 0x7C, 0x00   ; '0'
    db 0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00   ; '1'
    db 0x7C, 0xC6, 0x06, 0x1C, 0x30, 0x60, 0xFE, 0x00   ; '2'
    db 0x7C, 0xC6, 0x06, 0x3C, 0x06, 0xC6, 0x7C, 0x00   ; '3'
    db 0x0C, 0x1C, 0x3C, 0x6C, 0xFE, 0x0C, 0x0C, 0x00   ; '4'
    db 0xFE, 0xC0, 0xFC, 0x06, 0x06, 0xC6, 0x7C, 0x00   ; '5'
    db 0x3C, 0x60, 0xC0, 0xFC, 0xC6, 0xC6, 0x7C, 0x00   ; '6'
    db 0xFE, 0x06, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00   ; '7'
    db 0x7C, 0xC6, 0xC6, 0x7C, 0xC6, 0xC6, 0x7C, 0x00   ; '8'
    db 0x7C, 0xC6, 0xC6, 0x7E, 0x06, 0x0C, 0x78, 0x00   ; '9'
    db 0x00, 0x60, 0x60, 0x00, 0x60, 0x60, 0x00, 0x00   ; ':'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; ';'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '<'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '='
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '>'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '?'
    db 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   ; '@'
    db 0x38, 0x6C, 0xC6, 0xC6, 0xFE, 0xC6, 0xC6, 0x00   ; 'A'
    db 0xFC, 0xC6, 0xC6, 0xFC, 0xC6, 0xC6, 0xFC, 0x00   ; 'B'
    db 0x7C, 0xC6, 0xC0, 0xC0, 0xC0, 0xC6, 0x7C, 0x00   ; 'C'
    db 0xF8, 0xCC, 0xC6, 0xC6, 0xC6, 0xCC, 0xF8, 0x00   ; 'D'
    db 0xFE, 0xC0, 0xC0, 0xF8, 0xC0, 0xC0, 0xFE, 0x00   ; 'E'
    db 0xFE, 0xC0, 0xC0, 0xF8, 0xC0, 0xC0, 0xC0, 0x00   ; 'F'
    db 0x7C, 0xC6, 0xC0, 0xDE, 0xC6, 0xC6, 0x7C, 0x00   ; 'G'
    db 0xC6, 0xC6, 0xC6, 0xFE, 0xC6, 0xC6, 0xC6, 0x00   ; 'H'
    db 0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00   ; 'I'
    db 0x3E, 0x0C, 0x0C, 0x0C, 0xCC, 0xCC, 0x78, 0x00   ; 'J'
    db 0xC6, 0xCC, 0xD8, 0xF0, 0xD8, 0xCC, 0xC6, 0x00   ; 'K'
    db 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xFE, 0x00   ; 'L'
    db 0xC6, 0xEE, 0xFE, 0xD6, 0xC6, 0xC6, 0xC6, 0x00   ; 'M'
    db 0xC6, 0xE6, 0xF6, 0xDE, 0xCE, 0xC6, 0xC6, 0x00   ; 'N'
    db 0x7C, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x7C, 0x00   ; 'O'
    db 0xFC, 0xC6, 0xC6, 0xFC, 0xC0, 0xC0, 0xC0, 0x00   ; 'P'
    db 0x7C, 0xC6, 0xC6, 0xC6, 0xD6, 0xCC, 0x7A, 0x00   ; 'Q'
    db 0xFC, 0xC6, 0xC6, 0xFC, 0xD8, 0xCC, 0xC6, 0x00   ; 'R'
    db 0x7C, 0xC6, 0xC0, 0x7C, 0x06, 0xC6, 0x7C, 0x00   ; 'S'
    db 0xFC, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x00   ; 'T'
    db 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x7C, 0x00   ; 'U'
    db 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x6C, 0x38, 0x00   ; 'V'
    db 0xC6, 0xC6, 0xC6, 0xD6, 0xFE, 0xEE, 0xC6, 0x00   ; 'W'
    db 0xC6, 0x6C, 0x38, 0x38, 0x38, 0x6C, 0xC6, 0x00   ; 'X'
    db 0xC6, 0xC6, 0x6C, 0x38, 0x18, 0x18, 0x18, 0x00   ; 'Y'
    db 0xFE, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0xFE, 0x00   ; 'Z'

;---------------------------------------
; crc32_region — ESI = start, ECX = length -> EAX = CRC32 (zlib polynomial)
; Reflected CRC-32, nibble-table driven: matches Python zlib.crc32.
;---------------------------------------
crc32_region:
    push ebx
    push ecx
    push edx
    push esi
    mov eax, 0xFFFFFFFF
.cr_byte:
    movzx ebx, byte [esi]
    xor eax, ebx
    mov ebx, eax
    and ebx, 0x0F
    shr eax, 4
    xor eax, [crc32_tab + ebx*4]
    mov ebx, eax
    and ebx, 0x0F
    shr eax, 4
    xor eax, [crc32_tab + ebx*4]
    inc esi
    dec ecx
    jnz .cr_byte
    xor eax, 0xFFFFFFFF
    pop esi
    pop edx
    pop ecx
    pop ebx
    ret

align 4
crc32_tab:
    dd 0x00000000, 0x1DB71064, 0x3B6E20C8, 0x26D930AC
    dd 0x76DC4190, 0x6B6B51F4, 0x4DB26158, 0x5005713C
    dd 0xEDB88320, 0xF00F9344, 0xD6D6A3E8, 0xCB61B38C
    dd 0x9B64C2B0, 0x86D3D2D4, 0xA00AE278, 0xBDBDF21C

;---------------------------------------
; validate_weights — don't mistake the map for the territory.
; Verifies magic, layer sizes, weight count, and CRC32 of all
; 1,884,160 weight bytes against the trained header (~19M nibble
; steps: well under a second, once, at boot). On failure:
; hdr_status = 2, screen painted red, message drawn, hard halt.
;---------------------------------------
validate_weights:
    pushad
    cmp word [WEIGHT_BASE], 0x4E44           ; 'D','N' little-endian
    jne .vw_bad
    cmp word [WEIGHT_BASE + 5], NET_INPUT
    jne .vw_bad
    cmp word [WEIGHT_BASE + 7], NET_HIDDEN1
    jne .vw_bad
    cmp word [WEIGHT_BASE + 9], NET_HIDDEN2
    jne .vw_bad
    cmp word [WEIGHT_BASE + 11], NET_OUTPUT
    jne .vw_bad
    cmp dword [WEIGHT_BASE + 16], NET_TOTAL_W
    jne .vw_bad
    mov esi, WEIGHT_BASE + W_HEADER
    mov ecx, NET_TOTAL_W * 2
    call crc32_region
    cmp eax, [WEIGHT_BASE + 21]
    jne .vw_bad
    mov dword [hdr_status], 1
    popad
    ret

.vw_bad:
    mov dword [hdr_status], 2
    mov edi, [k_lfb_addr]
    cmp byte [k_video_mode], 1
    je .vw_red_vesa
    mov ecx, 320 * 200 / 4
    mov eax, 0x04040404                      ; VGA palette 4 = red
    rep stosd
    jmp .vw_msg
.vw_red_vesa:
    mov ecx, [k_scr_pitch]
    imul ecx, [k_scr_height]
    shr ecx, 2
    mov eax, 0x00AA0000
    rep stosd
.vw_msg:
    mov esi, msg_bad_weights
    mov ecx, 14
    mov eax, 8
    mov ebx, 8
    call draw_text
    cli
.vw_halt:
    hlt
    jmp .vw_halt

msg_bad_weights: db 'BAD WEIGHT CRC'

;===============================================================================
; KERNEL DATA
;===============================================================================

k_video_mode:   db 0
k_lfb_addr:     dd 0
k_scr_width:    dd 0
k_scr_height:   dd 0
k_scr_bpp:      dd 0
k_scr_pitch:    dd 0

cursor_x:       dd 400
cursor_y:       dd 300
color_idx:      db 15           ; White
draw_size:      dd 20
tick_count:     dd 0
last_cmd:       dd 0
last_confidence: dd 0

kb_read_idx:    dw 0
kb_write_idx:   dw 0

serial_present: db 0            ; UART probe result (0 = never poll)
serial_rx_count: dd 0           ; COM1 bytes accepted into the event ring

last_tick:      dd 0            ; tick consumed by the last main-loop step
step_count:     dd 0            ; state transitions performed (<= tick_count)
hdr_status:     dd 0            ; 0=unchecked, 1=weights valid, 2=REJECTED

;===============================================================================
; Pad kernel to exactly 32KB (64 sectors)
;===============================================================================

; Guard: code+data must fit inside the KERNEL_SECTORS load window
; (negative TIMES here = assembly error = the honest failure mode).
times (2560 + KERNEL_SECTORS * 512) - ($ - boot_entry) db 0
times (2560 + 32768) - ($ - boot_entry) db 0

;===============================================================================
; SECTION 4: DEFAULT WEIGHTS (sectors 70+)
; Placeholder — real weights patched by train_tier2.py
;===============================================================================

weights_placeholder:
    ; Zeroed region — tools/build.py patches the trained blob here.
    ; An unpatched image has no 'DN' magic: validate_weights paints it
    ; red and halts. The image never runs on weights it can't verify.
    times (WEIGHT_SECTORS * 512) db 0
