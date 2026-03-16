;===============================================================================
; DNOS - Deterministic Neural Operating System
; Unified Build - Tier 1
;
; This file compiles to a complete bootable disk image.
; It includes: boot sector, neural core, and basic I/O.
;
; Build: nasm -f bin dnos.asm -o dnos.img
; Run:   qemu-system-i386 -fda dnos.img
;
; Memory Layout:
;   0x7C00 - Boot sector (512 bytes)
;   0x7E00 - Extended boot/init (512 bytes)  
;   0x8000 - Neural core code
;   0x10000 - Weights (64KB)
;   0x20000 - Activations (32KB)
;   0x30000 - Memory organ (16KB)
;   0xA0000 - VGA framebuffer
;===============================================================================

[bits 16]
[org 0x7C00]

;===============================================================================
; SECTION 1: BOOT SECTOR (512 bytes)
;===============================================================================

boot_start:
    cli
    cld
    
    ; Save boot drive
    mov [boot_drive], dl
    
    ; Set up segments
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00
    
    ; Enable A20
    in al, 0x92
    or al, 2
    out 0x92, al
    
    ; Load extended boot + neural core (sectors 2-10)
    mov ax, 0x0000
    mov es, ax
    mov bx, 0x7E00          ; Load at 0x7E00
    mov ah, 0x02            ; Read sectors
    mov al, 9               ; 9 sectors (4.5KB)
    mov ch, 0               ; Cylinder 0
    mov cl, 2               ; Start at sector 2
    mov dh, 0               ; Head 0
    mov dl, [boot_drive]
    int 0x13
    jc disk_error
    
    ; Load weights (sectors 11-26)
    mov ax, 0x1000
    mov es, ax
    xor bx, bx              ; Load at 0x10000
    mov ah, 0x02
    mov al, 16              ; 16 sectors (8KB)
    mov ch, 0
    mov cl, 11              ; Start at sector 11
    mov dh, 0
    mov dl, [boot_drive]
    int 0x13
    jc disk_error
    
    ; Jump to extended boot
    jmp 0x0000:0x7E00

disk_error:
    mov si, msg_disk_err
    call print_string
    jmp $

print_string:
    mov ah, 0x0E
.loop:
    lodsb
    test al, al
    jz .done
    int 0x10
    jmp .loop
.done:
    ret

boot_drive: db 0
msg_disk_err: db 'Disk error', 0

; Pad to 510 bytes and add boot signature
times 510 - ($ - boot_start) db 0
dw 0xAA55

;===============================================================================
; SECTION 2: EXTENDED BOOT (512 bytes, sector 2)
;===============================================================================

extended_boot:
    ; Set video mode 13h (320x200, 256 colors)
    mov ax, 0x0013
    int 0x10
    
    ; Clear screen to blue
    mov ax, 0xA000
    mov es, ax
    xor di, di
    mov cx, 32000           ; 64000 bytes / 2
    mov ax, 0x0101          ; Blue
    rep stosw
    
    ; Initialize segments for neural core
    mov ax, 0
    mov ds, ax
    
    ; Show ready message
    mov si, msg_ready
    mov di, 0               ; Top-left of screen
    call draw_string
    
    ; Enable interrupts and jump to main loop
    sti
    jmp main_loop

msg_ready: db 'DNOS Ready', 0

; Draw string at DS:SI to screen at ES:DI
draw_string:
    push ax
.loop:
    lodsb
    test al, al
    jz .done
    mov [es:di], al
    inc di
    jmp .loop
.done:
    pop ax
    ret

; Pad sector 2
times 1024 - ($ - boot_start) db 0

;===============================================================================
; SECTION 3: NEURAL CORE (starts at 0x8000, sector 3)
;===============================================================================

neural_core:

;---------------------------------------
; Constants
;---------------------------------------

WEIGHT_SEG      equ 0x1000
ACTIV_SEG       equ 0x2000
MEMORY_SEG      equ 0x3000
VIDEO_SEG       equ 0xA000

; Network topology: 64 -> 32 -> 16
; (Tier 1 minimal: 64 input, 32 hidden, 16 output)
INPUT_SIZE      equ 64
HIDDEN_SIZE     equ 32
OUTPUT_SIZE     equ 16
TOTAL_WEIGHTS   equ (INPUT_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE)
; = 2048 + 512 = 2560 weights

; Input history (8 timesteps of 8 values = 64 inputs)
HISTORY_SIZE    equ 8
INPUT_WIDTH     equ 8

; Output command mappings
CMD_NOP         equ 0
CMD_PIXEL       equ 1
CMD_HLINE       equ 2
CMD_VLINE       equ 3
CMD_RECT        equ 4
CMD_FILL        equ 5
CMD_CLEAR       equ 6

;---------------------------------------
; Variables
;---------------------------------------

cursor_x:       dw 160
cursor_y:       dw 100
current_color:  db 15           ; White

; Input history buffer (8 timesteps × 8 values)
input_history:  times 64 dw 0
history_ptr:    dw 0

; Network activations
input_layer:    times INPUT_SIZE dw 0
hidden_layer:   times HIDDEN_SIZE dw 0
output_layer:   times OUTPUT_SIZE dw 0

; Last key pressed
last_key:       db 0

; Tick counter
tick_count:     dd 0

;---------------------------------------
; Main Loop
;---------------------------------------

main_loop:
    ; Increment tick
    inc dword [tick_count]
    
    ; Poll keyboard
    call poll_keyboard
    
    ; If key pressed, process it
    cmp byte [last_key], 0
    je .no_input
    
    ; Encode key into input history
    call encode_input
    
    ; Run neural inference
    call neural_forward
    
    ; Decode output and execute
    call decode_output
    
    ; Clear last key
    mov byte [last_key], 0

.no_input:
    ; Small delay
    mov cx, 0x1000
.delay:
    loop .delay
    
    jmp main_loop

;---------------------------------------
; Keyboard Input
;---------------------------------------

poll_keyboard:
    mov ah, 0x01
    int 0x16
    jz .no_key
    
    ; Key available, read it
    mov ah, 0x00
    int 0x16
    mov [last_key], al
    
    ; Check for ESC
    cmp al, 27
    je .exit
    
    ret

.no_key:
    ret

.exit:
    ; Return to text mode and halt
    mov ax, 0x0003
    int 0x10
    mov si, msg_exit
    call print_string_bios
    cli
    hlt

msg_exit: db 'DNOS halted.', 13, 10, 0

print_string_bios:
    mov ah, 0x0E
.loop:
    lodsb
    test al, al
    jz .done
    int 0x10
    jmp .loop
.done:
    ret

;---------------------------------------
; Input Encoding
;---------------------------------------

encode_input:
    ; Shift history buffer
    ; Move entries 0-6 to 1-7
    mov si, input_history + (HISTORY_SIZE - 2) * INPUT_WIDTH * 2
    mov di, input_history + (HISTORY_SIZE - 1) * INPUT_WIDTH * 2
    mov cx, (HISTORY_SIZE - 1) * INPUT_WIDTH
    std
    rep movsw
    cld
    
    ; Encode current key into slot 0
    mov al, [last_key]
    mov di, input_history
    
    ; Convert byte to 8 binary neurons
    mov cx, 8
.encode_loop:
    xor ah, ah
    shr al, 1
    jnc .zero
    mov word [di], 32767    ; "1" = max positive
    jmp .next
.zero:
    mov word [di], 0        ; "0" = zero
.next:
    add di, 2
    loop .encode_loop
    
    ret

;---------------------------------------
; Neural Forward Pass
;---------------------------------------

neural_forward:
    pusha
    push es
    push ds
    
    ; Copy input history to input layer
    mov si, input_history
    mov di, input_layer
    mov cx, INPUT_SIZE
    rep movsw
    
    ; Layer 1: Input -> Hidden
    ; hidden[j] = sigmoid(sum(input[i] * weight[i,j]))
    
    mov ax, WEIGHT_SEG
    mov es, ax              ; ES = weight segment
    
    xor bx, bx              ; Weight offset
    mov byte [.neuron_idx], 0
    
.hidden_loop:
    ; Compute weighted sum for this hidden neuron
    xor dx, dx              ; Accumulator high
    xor ax, ax              ; Accumulator low (use BP for sum)
    mov bp, ax
    
    mov si, input_layer
    mov cx, INPUT_SIZE
    
.sum_input:
    mov ax, [si]            ; Input activation
    add si, 2
    
    push bx
    mov bx, [es:bx]         ; Weight (already at correct offset in loop)
    imul bx                 ; DX:AX = input * weight
    add bp, ax              ; Accumulate (simplified, ignore overflow)
    pop bx
    add bx, 2               ; Next weight
    
    loop .sum_input
    
    ; Apply sigmoid approximation
    mov ax, bp
    call sigmoid
    
    ; Store in hidden layer
    mov cl, [.neuron_idx]
    xor ch, ch
    shl cx, 1
    mov di, hidden_layer
    add di, cx
    mov [di], ax
    
    inc byte [.neuron_idx]
    cmp byte [.neuron_idx], HIDDEN_SIZE
    jb .hidden_loop
    
    ; Layer 2: Hidden -> Output
    mov byte [.neuron_idx], 0
    ; BX already points past input->hidden weights
    
.output_loop:
    xor bp, bp              ; Accumulator
    
    mov si, hidden_layer
    mov cx, HIDDEN_SIZE
    
.sum_hidden:
    mov ax, [si]
    add si, 2
    
    push bx
    mov bx, [es:bx]
    imul bx
    add bp, ax
    pop bx
    add bx, 2
    
    loop .sum_hidden
    
    ; Sigmoid
    mov ax, bp
    call sigmoid
    
    ; Store in output layer
    mov cl, [.neuron_idx]
    xor ch, ch
    shl cx, 1
    mov di, output_layer
    add di, cx
    mov [di], ax
    
    inc byte [.neuron_idx]
    cmp byte [.neuron_idx], OUTPUT_SIZE
    jb .output_loop
    
    pop ds
    pop es
    popa
    ret

.neuron_idx: db 0

;---------------------------------------
; Sigmoid Approximation
; Input: AX = value
; Output: AX = sigmoid(value) in range [0, 32767]
;---------------------------------------

sigmoid:
    ; Piecewise linear approximation
    ; if x < -8192: return 0
    ; if x > 8192: return 32767
    ; else: return (x + 8192) * 2
    
    cmp ax, -8192
    jl .saturate_low
    cmp ax, 8192
    jg .saturate_high
    
    ; Linear region
    add ax, 8192
    shl ax, 1
    ret

.saturate_low:
    xor ax, ax
    ret

.saturate_high:
    mov ax, 32767
    ret

;---------------------------------------
; Output Decoding
;---------------------------------------

decode_output:
    pusha
    
    ; Find max output neuron (argmax)
    mov si, output_layer
    mov cx, OUTPUT_SIZE
    xor bx, bx              ; Best index
    mov dx, [si]            ; Best value
    xor ax, ax              ; Current index
    
.find_max:
    cmp [si], dx
    jle .not_better
    mov dx, [si]
    mov bx, ax
.not_better:
    add si, 2
    inc ax
    loop .find_max
    
    ; BX = command (0-15, but we only use 0-6)
    cmp bx, 7
    jae .done               ; Invalid command
    
    ; Decode parameters from other output neurons
    ; output[8-9] = X offset, output[10-11] = Y offset
    mov ax, [output_layer + 8*2]
    sar ax, 10              ; Scale to -32 to +31
    add [cursor_x], ax
    
    mov ax, [output_layer + 10*2]
    sar ax, 10
    add [cursor_y], ax
    
    ; Clamp cursor
    cmp word [cursor_x], 0
    jge .x_ok_low
    mov word [cursor_x], 0
.x_ok_low:
    cmp word [cursor_x], 319
    jle .x_ok_high
    mov word [cursor_x], 319
.x_ok_high:
    cmp word [cursor_y], 0
    jge .y_ok_low
    mov word [cursor_y], 0
.y_ok_low:
    cmp word [cursor_y], 199
    jle .y_ok_high
    mov word [cursor_y], 199
.y_ok_high:
    
    ; Execute command
    cmp bx, CMD_NOP
    je .done
    cmp bx, CMD_PIXEL
    je .do_pixel
    cmp bx, CMD_HLINE
    je .do_hline
    cmp bx, CMD_VLINE
    je .do_vline
    cmp bx, CMD_RECT
    je .do_rect
    cmp bx, CMD_FILL
    je .do_fill
    cmp bx, CMD_CLEAR
    je .do_clear
    jmp .done

.do_pixel:
    call draw_pixel
    jmp .done

.do_hline:
    mov cx, 20
    call draw_hline
    jmp .done

.do_vline:
    mov cx, 20
    call draw_vline
    jmp .done

.do_rect:
    mov cx, 30              ; Width
    mov dx, 20              ; Height
    call draw_rect
    jmp .done

.do_fill:
    call fill_screen
    jmp .done

.do_clear:
    mov byte [current_color], 1  ; Blue
    call fill_screen
    mov byte [current_color], 15 ; White
    jmp .done

.done:
    popa
    ret

;---------------------------------------
; Graphics Primitives
;---------------------------------------

; Draw pixel at (cursor_x, cursor_y)
draw_pixel:
    push es
    push di
    push ax
    
    mov ax, VIDEO_SEG
    mov es, ax
    
    ; Calculate offset: y * 320 + x
    mov ax, [cursor_y]
    mov di, 320
    mul di
    add ax, [cursor_x]
    mov di, ax
    
    mov al, [current_color]
    mov [es:di], al
    
    pop ax
    pop di
    pop es
    ret

; Draw horizontal line, CX = length
draw_hline:
    push es
    push di
    push cx
    push ax
    
    mov ax, VIDEO_SEG
    mov es, ax
    
    mov ax, [cursor_y]
    mov di, 320
    mul di
    add ax, [cursor_x]
    mov di, ax
    
    mov al, [current_color]
    rep stosb
    
    pop ax
    pop cx
    pop di
    pop es
    ret

; Draw vertical line, CX = length
draw_vline:
    push es
    push di
    push cx
    push ax
    
    mov ax, VIDEO_SEG
    mov es, ax
    
    mov ax, [cursor_y]
    mov di, 320
    mul di
    add ax, [cursor_x]
    mov di, ax
    
    mov al, [current_color]
.vloop:
    mov [es:di], al
    add di, 320
    loop .vloop
    
    pop ax
    pop cx
    pop di
    pop es
    ret

; Draw rectangle, CX = width, DX = height
draw_rect:
    push cx
    push dx
    
    ; Top edge
    call draw_hline
    
    ; Bottom edge
    push word [cursor_y]
    add [cursor_y], dx
    dec word [cursor_y]
    call draw_hline
    pop word [cursor_y]
    
    ; Left edge
    mov cx, dx
    call draw_vline
    
    ; Right edge
    push word [cursor_x]
    pop ax
    add ax, cx
    dec ax
    push word [cursor_x]
    mov [cursor_x], ax
    pop ax
    push ax
    mov cx, dx
    call draw_vline
    pop ax
    mov [cursor_x], ax
    
    pop dx
    pop cx
    ret

; Fill screen with current color
fill_screen:
    push es
    push di
    push cx
    push ax
    
    mov ax, VIDEO_SEG
    mov es, ax
    xor di, di
    mov cx, 32000
    mov al, [current_color]
    mov ah, al
    rep stosw
    
    pop ax
    pop cx
    pop di
    pop es
    ret

;---------------------------------------
; Pad to fill sectors 3-10 (4KB)
;---------------------------------------

times 5120 - ($ - boot_start) db 0

;===============================================================================
; SECTION 4: DEFAULT WEIGHTS (sectors 11-26, 8KB)
; These are placeholder random weights. Real weights should be trained.
;===============================================================================

weights_start:

; Header (64 bytes)
db 'DN'                     ; Magic
db 1                        ; Version
db 1                        ; Tier
db 3                        ; Num layers
db 64, 32, 16, 0, 0, 0, 0, 0  ; Layer sizes
dw TOTAL_WEIGHTS            ; Weight count
db 0                        ; Activation type (sigmoid)
db 32                       ; Learning rate (unused for now)
times 46 db 0               ; Padding

; Weights: Input (64) -> Hidden (32) = 2048 weights
; Initialize with small random-ish values
%assign i 0
%rep 2048
    dw (i * 17 + 31) % 256 - 128  ; Pseudo-random in range [-128, 127]
%assign i i+1
%endrep

; Weights: Hidden (32) -> Output (16) = 512 weights
%assign i 0
%rep 512
    dw (i * 23 + 47) % 256 - 128
%assign i i+1
%endrep

; Pad weights section to 8KB (16 sectors)
times 8192 - ($ - weights_start) db 0

;===============================================================================
; END - Total image size: 512 + 512 + 4096 + 8192 = 13312 bytes
;===============================================================================
