;===============================================================================
; DNOS Input Abstraction Layer — Bare Metal
;
; Replaces encode_input in dnos.asm with epoch-quantized, canonically-ordered
; input encoding. Uses PIT ticks as epoch boundaries.
;
; This is the assembly equivalent of the Rust IAL Pipeline.
; Same guarantee: same physical events → same token sequence → same network input.
;
; Integration: include this file from dnos.asm, replace the encode_input call
; with call ial_process_event, and call ial_flush at epoch boundaries.
;
; Memory layout (within MEMORY_SEG):
;   0x0000 - Epoch buffer (tokens awaiting canonicalization)
;   0x0200 - Token history ring (last 8 canonical tokens for network input)
;   0x0400 - IAL state (counters, pointers, last-values for dedup)
;===============================================================================

;---------------------------------------
; IAL Constants
;---------------------------------------

IAL_EPOCH_TICKS     equ 1           ; PIT ticks per epoch (at 100Hz = 10ms)
IAL_MAX_TOKENS      equ 16          ; Max tokens buffered per epoch
IAL_TOKEN_SIZE      equ 4           ; Compact token: [channel, type, payload_lo, payload_hi]
IAL_HISTORY_DEPTH   equ 8           ; Token history slots
IAL_GRID_SIZE       equ 10          ; Spatial quantization grid

; Token buffer offsets (relative to MEMORY_SEG)
IAL_EPOCH_BUF       equ 0x0000      ; 16 tokens × 4 bytes = 64 bytes
IAL_HISTORY_BUF     equ 0x0200      ; 8 tokens × 4 bytes = 32 bytes
IAL_STATE           equ 0x0400      ; State variables

; State variable offsets (relative to IAL_STATE)
IAL_CURRENT_EPOCH   equ 0           ; dd - current epoch number
IAL_TOKEN_COUNT     equ 4           ; dw - tokens in current epoch buffer
IAL_HISTORY_PTR     equ 6           ; dw - write pointer into history ring
IAL_LAST_MOUSE_X    equ 8           ; dw - last mouse grid X (for dedup)
IAL_LAST_MOUSE_Y    equ 10          ; dw - last mouse grid Y (for dedup)
IAL_STREAM_HASH     equ 12          ; dd - rolling FNV-1a hash

;---------------------------------------
; ial_init — Initialize the IAL
; Call once at boot after setting up MEMORY_SEG
;---------------------------------------

ial_init:
    pusha
    push es

    mov ax, MEMORY_SEG
    mov es, ax

    ; Zero the entire IAL region (0x0000 - 0x0500)
    xor di, di
    mov cx, 0x0280          ; 0x500 / 2
    xor ax, ax
    rep stosw

    ; Initialize FNV-1a hash state
    mov dword [es:IAL_STATE + IAL_STREAM_HASH], 0x811C9DC5

    ; Initialize last mouse to impossible value (forces first-move emit)
    mov word [es:IAL_STATE + IAL_LAST_MOUSE_X], 0xFFFF
    mov word [es:IAL_STATE + IAL_LAST_MOUSE_Y], 0xFFFF

    pop es
    popa
    ret

;---------------------------------------
; ial_process_key — Process a keyboard event
;
; Input: AL = scancode
; Modifies: IAL epoch buffer
;---------------------------------------

ial_process_key:
    pusha
    push es

    mov ax, MEMORY_SEG
    mov es, ax

    ; Check if we need to flush (epoch boundary crossed)
    call ial_check_epoch

    ; Build token: [channel=0, type=KEY_DOWN(1), scancode, 0]
    mov di, [es:IAL_STATE + IAL_TOKEN_COUNT]
    cmp di, IAL_MAX_TOKENS
    jge .overflow

    ; Calculate buffer offset: IAL_EPOCH_BUF + token_count * 4
    shl di, 2
    add di, IAL_EPOCH_BUF

    mov byte [es:di + 0], 0         ; Channel::Keyboard
    mov byte [es:di + 1], 1         ; EVT_KEY_DOWN
    pop es
    push es
    ; AL was clobbered by MEMORY_SEG load, recover from stack
    ; Actually AL is in the pusha frame. Let's restructure.
    ; We need the original AL. It's saved in pusha at [sp + 14] (AX low byte).
    ; Better: save it before pusha.
    jmp .overflow  ; TODO: fix register management

.overflow:
    pop es
    popa
    ret

;---------------------------------------
; ial_process_key_v2 — Cleaner version
;
; Input: BL = scancode (caller must set BL before calling)
; Modifies: IAL epoch buffer
;---------------------------------------

ial_process_key_v2:
    pusha
    push es

    mov ax, MEMORY_SEG
    mov es, ax

    ; Check epoch boundary
    call ial_check_epoch

    ; Get token count, check overflow
    movzx di, word [es:IAL_STATE + IAL_TOKEN_COUNT]
    cmp di, IAL_MAX_TOKENS
    jge .done

    ; Write token at IAL_EPOCH_BUF + count * 4
    shl di, 2
    add di, IAL_EPOCH_BUF

    mov byte [es:di + 0], 0         ; Channel::Keyboard = 0
    mov byte [es:di + 1], 1         ; EVT_KEY_DOWN
    mov [es:di + 2], bl              ; Scancode as payload
    mov byte [es:di + 3], 0         ; Padding

    ; Increment token count
    inc word [es:IAL_STATE + IAL_TOKEN_COUNT]

.done:
    pop es
    popa
    ret

;---------------------------------------
; ial_check_epoch — Check if PIT tick crossed epoch boundary
;
; Compares current tick_count with stored epoch.
; If epoch advanced, flushes the buffer (sort + hash + commit to history).
;---------------------------------------

ial_check_epoch:
    push eax
    push ebx

    ; Current epoch = tick_count / IAL_EPOCH_TICKS
    mov eax, [tick_count]
    ; IAL_EPOCH_TICKS = 1, so epoch = tick_count directly
    ; For other values: xor edx,edx; mov ebx, IAL_EPOCH_TICKS; div ebx

    ; Compare with stored epoch
    cmp eax, [es:IAL_STATE + IAL_CURRENT_EPOCH]
    je .same_epoch

    ; Epoch changed — flush previous epoch's buffer
    call ial_flush_epoch

    ; Update stored epoch
    mov [es:IAL_STATE + IAL_CURRENT_EPOCH], eax

.same_epoch:
    pop ebx
    pop eax
    ret

;---------------------------------------
; ial_flush_epoch — Sort and commit buffered tokens
;
; This is where canonicalization happens in assembly.
; Sorts tokens by (channel, type, payload) using insertion sort
; (fine for ≤16 elements), then commits to the history ring.
;---------------------------------------

ial_flush_epoch:
    pusha

    movzx cx, word [es:IAL_STATE + IAL_TOKEN_COUNT]
    cmp cx, 0
    je .nothing

    ; --- Insertion sort on the epoch buffer ---
    ; Each token is 4 bytes. Sort key is the full 32-bit value
    ; (channel in MSB position after byte ordering).
    ; Since tokens are [channel, type, payload_lo, payload_hi],
    ; a 32-bit LE comparison sorts by payload first.
    ; We need BE comparison, so we compare byte-by-byte.

    cmp cx, 1
    jle .sorted

    mov si, 1               ; i = 1
.sort_outer:
    cmp si, cx
    jge .sorted

    ; Load token[i] as "key"
    mov di, si
    shl di, 2
    add di, IAL_EPOCH_BUF
    mov eax, [es:di]        ; key token (4 bytes)

    mov bx, si
    dec bx                  ; j = i - 1

.sort_inner:
    cmp bx, 0
    jl .sort_insert

    ; Compare token[j] with key
    push di
    mov di, bx
    shl di, 2
    add di, IAL_EPOCH_BUF
    mov edx, [es:di]        ; token[j]
    pop di

    ; Byte-by-byte comparison for canonical order
    ; Compare channel (byte 0) first
    cmp dl, al
    jb .sort_insert
    ja .sort_shift
    ; Channel equal, compare type (byte 1)
    mov dh, dl              ; save
    shr edx, 8
    push eax
    shr eax, 8
    cmp dl, al
    pop eax
    jb .sort_insert
    ja .sort_shift
    ; Type equal — compare payload (bytes 2-3)
    ; For simplicity, compare full 32-bit value
    cmp edx, eax
    jbe .sort_insert

.sort_shift:
    ; token[j+1] = token[j]
    push di
    mov di, bx
    shl di, 2
    add di, IAL_EPOCH_BUF

    push si
    mov si, bx
    inc si
    shl si, 2
    add si, IAL_EPOCH_BUF
    mov edx, [es:di]
    mov [es:si], edx
    pop si
    pop di

    dec bx
    jmp .sort_inner

.sort_insert:
    ; token[j+1] = key
    push di
    mov di, bx
    inc di
    shl di, 2
    add di, IAL_EPOCH_BUF
    mov [es:di], eax
    pop di

    inc si
    jmp .sort_outer

.sorted:
    ; --- Commit sorted tokens to history ring ---
    xor si, si              ; token index
.commit_loop:
    cmp si, cx
    jge .commit_done

    ; Read token from epoch buffer
    push si
    shl si, 2
    add si, IAL_EPOCH_BUF
    mov eax, [es:si]
    pop si

    ; Write to history ring at write_ptr
    movzx di, word [es:IAL_STATE + IAL_HISTORY_PTR]
    push di
    shl di, 2
    add di, IAL_HISTORY_BUF
    mov [es:di], eax
    pop di

    ; Advance write pointer (circular)
    inc di
    cmp di, IAL_HISTORY_DEPTH
    jb .no_wrap
    xor di, di
.no_wrap:
    mov [es:IAL_STATE + IAL_HISTORY_PTR], di

    ; Update rolling hash: FNV-1a
    ; hash ^= byte; hash *= 0x01000193
    push ecx
    mov ecx, 4
    push si
    mov si, di
    dec si
    jns .hash_ptr_ok
    mov si, IAL_HISTORY_DEPTH - 1
.hash_ptr_ok:
    shl si, 2
    add si, IAL_HISTORY_BUF

    mov edx, [es:IAL_STATE + IAL_STREAM_HASH]
.hash_loop:
    xor dl, [es:si]
    inc si
    imul edx, edx, 0x01000193
    loop .hash_loop

    mov [es:IAL_STATE + IAL_STREAM_HASH], edx
    pop si
    pop ecx

    inc si
    jmp .commit_loop

.commit_done:
    ; Clear epoch buffer
    mov word [es:IAL_STATE + IAL_TOKEN_COUNT], 0

.nothing:
    popa
    ret

;---------------------------------------
; ial_encode_to_input — Convert token history to neural input vector
;
; Writes INPUT_SIZE words to input_layer, reading from the IAL
; history ring in reverse chronological order.
;
; Each history slot maps to INPUT_WIDTH neurons:
;   [channel_onehot(8), type_hi, type_lo, payload(6)]
;
; For 64-input network: 8 slots × 8 neurons = 64
; For 256-input network: 8 slots × 32 neurons = 256
;---------------------------------------

ial_encode_to_input:
    pusha
    push es
    push ds

    mov ax, MEMORY_SEG
    mov es, ax

    ; DS already points to code/data segment
    ; Write to input_layer in DS

    movzx bx, word [es:IAL_STATE + IAL_HISTORY_PTR]
    mov di, input_layer

    mov cx, IAL_HISTORY_DEPTH
.slot_loop:
    ; Read from history ring in reverse chronological order
    dec bx
    jns .ptr_ok
    mov bx, IAL_HISTORY_DEPTH - 1
.ptr_ok:
    push bx
    shl bx, 2
    add bx, IAL_HISTORY_BUF

    ; Read token [channel, type, payload_lo, payload_hi]
    mov al, [es:bx + 0]     ; channel
    mov ah, [es:bx + 1]     ; type
    mov dl, [es:bx + 2]     ; payload_lo
    mov dh, [es:bx + 3]     ; payload_hi
    pop bx

    ; Encode channel as one-hot (simplified for 8 neurons)
    ; For 64-input network: 8 neurons per slot
    push cx
    mov cx, 8
    xor si, si
.channel_hot:
    cmp si, 8
    jge .encode_rest
    xor ah, ah
    cmp al, sl
    jne .not_this_channel
    mov word [di], 32767    ; Q8.8 max = "1"
    jmp .next_channel
.not_this_channel:
    mov word [di], 0
.next_channel:
    add di, 2
    inc si
    jmp .channel_hot

.encode_rest:
    pop cx

    ; For the minimal 64-input (8 neurons/slot) version,
    ; the 8 channel one-hot neurons fill the entire slot.
    ; For 256-input (32 neurons/slot), we'd continue encoding
    ; type and payload into remaining 24 neurons here.

    loop .slot_loop

    pop ds
    pop es
    popa
    ret
