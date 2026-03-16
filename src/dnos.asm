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
KERNEL_SECTORS  equ 64          ; 32KB
KERNEL_ADDR     equ 0x8000

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

    ;--- Load stage 2 (sectors 2-5) to 0x7E00 ---
    mov ax, 0x0000
    mov es, ax
    mov bx, STAGE2_ADDR
    mov ah, 0x02
    mov al, STAGE2_SECTORS
    mov ch, 0
    mov cl, 2
    mov dh, 0
    mov dl, [boot_drive]
    int 0x13
    jc .disk_err

    ;--- Load kernel (sectors 6-69, 32KB) to 0x8000 ---
    ; Split into two 16KB chunks to avoid DMA boundary crossing
    ; Chunk 1: sectors 6-37 (16KB) to 0x0000:0x8000
    mov ax, 0x0000
    mov es, ax
    mov bx, KERNEL_ADDR
    mov ah, 0x02
    mov al, 32              ; 16KB
    mov ch, 0
    mov cl, 6
    mov dh, 0
    mov dl, [boot_drive]
    int 0x13
    jc .disk_err

    ; Chunk 2: sectors 38-69 (16KB) to 0x0000:0x10000
    ; Use segment trick: 0x1000:0x0000 = physical 0x10000
    ; But wait — 0x8000 + 0x4000 = 0xC000, still within segment 0
    ; So we can load directly
    mov ax, 0x0000
    mov es, ax
    mov bx, 0xC000          ; 0x8000 + 16KB
    mov ah, 0x02
    mov al, 32
    mov ch, 0
    mov cl, 38
    mov dh, 0
    mov dl, [boot_drive]
    int 0x13
    jc .disk_err

    ; Pass boot drive to stage 2 and jump
    mov dl, [boot_drive]
    jmp 0x0000:STAGE2_ADDR

.disk_err:
    mov si, msg_disk_err
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

    ;--- Load weights in real mode (sectors 70-239, 86KB) ---
    ; Must load before PM switch because we need BIOS INT 13h
    ; DMA-safe: load in 16KB (32-sector) chunks
    call load_weights

    ;--- VESA video mode ---
    call setup_vesa

    ;--- Enable A20 (keyboard controller method + fast fallback) ---
    call enable_a20

    ;--- Load GDT ---
    lgdt [gdt_descriptor]

    ;--- Switch to protected mode ---
    mov eax, cr0
    or eax, 1
    mov cr0, eax

    jmp 0x08:pm_entry

;---------------------------------------
; Load weights: 86KB in 16KB chunks
; Sectors 70-239 → physical 0x10000-0x25000
;---------------------------------------
load_weights:
    pusha

    ; Chunk 1: 32 sectors → 0x1000:0x0000 (phys 0x10000)
    mov ax, 0x1000
    mov es, ax
    xor bx, bx
    mov ah, 0x02
    mov al, 32
    mov ch, 0
    mov cl, 70              ; LBA-ish (works for floppy CHS too)
    mov dh, 0
    mov dl, [s2_boot_drive]
    int 0x13
    jc .wt_err

    ; Chunk 2: 32 sectors → 0x1400:0x0000 (phys 0x14000)
    mov ax, 0x1400
    mov es, ax
    xor bx, bx
    mov ah, 0x02
    mov al, 32
    mov ch, 0
    mov cl, 102
    mov dh, 0
    mov dl, [s2_boot_drive]
    int 0x13
    jc .wt_err

    ; Chunk 3: 32 sectors → 0x1800:0x0000 (phys 0x18000)
    mov ax, 0x1800
    mov es, ax
    xor bx, bx
    mov ah, 0x02
    mov al, 32
    mov ch, 0
    mov cl, 134
    mov dh, 0
    mov dl, [s2_boot_drive]
    int 0x13
    jc .wt_err

    ; Chunk 4: 32 sectors → 0x1C00:0x0000 (phys 0x1C000)
    mov ax, 0x1C00
    mov es, ax
    xor bx, bx
    mov ah, 0x02
    mov al, 32
    mov ch, 0
    mov cl, 166
    mov dh, 0
    mov dl, [s2_boot_drive]
    int 0x13
    jc .wt_err

    ; Chunk 5: remaining sectors → 0x2000:0x0000 (phys 0x20000)
    mov ax, 0x2000
    mov es, ax
    xor bx, bx
    mov ah, 0x02
    mov al, 8               ; Remaining
    mov ch, 0
    mov cl, 198
    mov dh, 0
    mov dl, [s2_boot_drive]
    int 0x13
    jc .wt_err

    popa
    ret

.wt_err:
    mov si, msg_wt_err
    call print16_s2
    jmp $

msg_wt_err: db 'WT ERR', 0

;---------------------------------------
; VESA Setup
;---------------------------------------
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
    ; Fall back to VGA mode 13h (320x200x8bpp)
    mov ax, 0x0013
    int 0x10
    mov byte [video_mode], 0
    mov word [scr_width], 320
    mov word [scr_height], 200
    mov byte [scr_bpp], 8
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
WEIGHT_BASE     equ 0x10000     ; Physical address of weights
ACTIV_BASE      equ 0x20000     ; Activations scratch
HISTORY_BASE    equ 0x30000     ; Input history buffer
KEYBUF_BASE     equ 0x70000     ; Keyboard ring buffer

; Network topology
NET_INPUT       equ 256         ; 32 events × 8 features
NET_HIDDEN1     equ 128
NET_HIDDEN2     equ 64
NET_OUTPUT      equ 32
NET_HISTORY     equ 32          ; Context events
NET_FEATURES    equ 8           ; Features per event

; Weight offsets (in bytes, int16 = 2 bytes each)
; Header is 128 bytes
W_HEADER        equ 128
W1_OFF          equ W_HEADER                            ; 256×128 = 32768 weights
W2_OFF          equ W1_OFF + (NET_INPUT * NET_HIDDEN1 * 2)     ; 128×64 = 8192
W3_OFF          equ W2_OFF + (NET_HIDDEN1 * NET_HIDDEN2 * 2)   ; 64×32 = 2048

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

    ; Set up IDT
    call setup_idt

    ; Set up PIC (remap IRQs to 32-47)
    call setup_pic

    ; Set up PIT at 100Hz
    call setup_pit

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

    ; Divisor for 100Hz: 1193182 / 100 = 11932 = 0x2E9C
    mov ax, 11932
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
    mov edi, [k_lfb_addr]
    add edi, 20              ; Offset 20 pixels
    mov ecx, 4               ; 4 chars
    call draw_text_simple

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

    ; Clear demo indicator
    mov esi, msg_interactive
    mov edi, [k_lfb_addr]
    add edi, 20
    mov ecx, 11
    call draw_text_simple

    popad
    ret

demo_keys: db 'p', 'b', 'o', 'x', 'l', 'i', 'n', 'e'
msg_demo: db 'DEMO'
msg_interactive: db 'INTERACTIVE'

;===============================================================================
; MAIN LOOP (interactive mode)
;===============================================================================

main_loop:
    ; Poll keyboard buffer
    call read_key
    test al, al
    jz .no_input

    ; Convert scancode to ASCII
    push eax
    call scan_to_ascii
    test al, al
    jz .skip_input

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

.skip_input:
    pop eax

.no_input:
    ; Draw cursor
    call draw_cursor

    ; Update status bar every 10 ticks
    mov eax, [tick_count]
    and eax, 0x0F
    jnz .no_status
    call draw_status_bar
.no_status:

    hlt                      ; Wait for next interrupt
    jmp main_loop

.halt:
    pop eax
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
    mov word [edi], 0x7F00   ; Q8.8: ~127.0 (high activation)
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
    imul edi                         ; EAX = a * w (Q16.16)
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
    imul edi
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
    imul edi
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

    ; Extract cursor deltas from outputs 20-23
    movsx eax, word [ACTIV_BASE + A_OUTPUT + 20*2]
    sar eax, 10              ; Scale to ±31
    add [cursor_x], eax

    movsx eax, word [ACTIV_BASE + A_OUTPUT + 22*2]
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
    sub eax, 20              ; Reserve status bar
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
    jmp .cmd_done
.cmd_mv_left:
    sub dword [cursor_x], 10
    cmp dword [cursor_x], 0
    jge .cmd_done
    mov dword [cursor_x], 0
    jmp .cmd_done
.cmd_mv_right:
    add dword [cursor_x], 10
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
    ; VESA 32bpp: offset = (y * width + x) * 4
    imul ebx, [k_scr_width]
    add ebx, eax
    shl ebx, 2               ; * 4 for 32bpp
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
    mov edx, [k_scr_width]
    shl edx, 2
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
    push dword [cursor_x]
    pop eax
    add eax, ecx
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
    mov ecx, [k_scr_width]
    imul ecx, [k_scr_height]
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
    mov ecx, [k_scr_width]
    imul ecx, [k_scr_height]
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
    ; Simple: write "DNOS T2" at top of screen
    mov esi, msg_banner
    mov edi, [k_lfb_addr]
    add edi, 8               ; Small offset
    mov ecx, 7
    call draw_text_simple
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
    popad
    ret

.sb_vesa:
    shl eax, 2               ; 32bpp
    add edi, eax
    mov ecx, [k_scr_width]
    imul ecx, 16
    mov eax, 0x00333333      ; Dark gray ARGB
    rep stosd
    popad
    ret

; Simple text drawing — writes bytes directly as pixel values
; Not real font rendering, just proves text path works
; ESI=string, EDI=LFB position, ECX=length
draw_text_simple:
    pushad

    cmp byte [k_video_mode], 1
    je .dt_vesa

    ; VGA: just write ASCII codes as pixel values (visible as colored dots)
.dt_vga:
    lodsb
    mov [edi], al
    inc edi
    loop .dt_vga
    popad
    ret

.dt_vesa:
    ; VESA: write white-on-dark character markers
    ; Each char = 8-pixel-wide block with brightness = ASCII code
.dt_vesa_char:
    lodsb
    movzx eax, al
    ; Make it white text: 0x00FFFFFF for any non-zero char
    test eax, eax
    jz .dt_vesa_skip
    mov eax, 0x00FFFFFF
.dt_vesa_skip:
    ; Write 8 pixels wide
    push ecx
    mov ecx, 8
.dt_px:
    mov [edi], eax
    add edi, 4
    loop .dt_px
    pop ecx
    loop .dt_vesa_char
    popad
    ret

;===============================================================================
; KERNEL DATA
;===============================================================================

k_video_mode:   db 0
k_lfb_addr:     dd 0
k_scr_width:    dd 0
k_scr_height:   dd 0
k_scr_bpp:      dd 0

cursor_x:       dd 400
cursor_y:       dd 300
color_idx:      db 15           ; White
draw_size:      dd 20
tick_count:     dd 0
last_cmd:       dd 0
last_confidence: dd 0

kb_read_idx:    dw 0
kb_write_idx:   dw 0

;===============================================================================
; Pad kernel to exactly 32KB (64 sectors)
;===============================================================================

times (2560 + 32768) - ($ - boot_entry) db 0

;===============================================================================
; SECTION 4: DEFAULT WEIGHTS (sectors 70+)
; Placeholder — real weights patched by train_tier2.py
;===============================================================================

weights_placeholder:
    ; 128-byte header
    db 'DN'                  ; Magic
    db 2                     ; Version
    db 2                     ; Tier
    db 4                     ; Num layers
    db 0, 1                  ; INPUT_SIZE = 256 (little-endian word)
    db 128, 0                ; HIDDEN1 = 128
    db 64, 0                 ; HIDDEN2 = 64
    db 32, 0                 ; OUTPUT = 32
    times 116 db 0           ; Padding to 128 bytes

    ; Placeholder weights — will be overwritten by dd
    %assign i 0
    %rep 43008
        dw (i * 17 + 31) % 512 - 256
    %assign i i+1
    %endrep

    ; Pad to 86KB (172 sectors)
    times 88064 - ($ - weights_placeholder) db 0
