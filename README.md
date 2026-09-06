# jitasm

A Python x86-64 JIT assembler.

I've tested on Windows and Linux. I have no macOS computer for testing, but theoretically it should also work on Intel macOS.

```
pip install jitasm
```

## Example

The following example creates an `int64_t max(int64_t a, int64_t b)` function equivalent to:

```c
int64_t max(int64_t a, int64_t b) {
    if (a >= b) return a;
    return b;
}
```

On Linux, the generated instructions are:

```asm
max:
    mov rax, rdi
    cmp rdi, rsi
    jge .done
    mov rax, rsi
.done:
    ret
```

And we provide a pseudo-instruction `branch.cc` to do `CMP` and `Jcc`:

```asm
max:
    mov rax, rdi
    bge rdi, rsi, .done
    mov rax, rsi
.done:
    ret
```

And we implement it in Python using `jitasm`:

```python
from jitasm.x86_64 import *
from jitasm.utils import ccall

e = Emitter()
(e.label("max"),
    e.mov(RAX, RDI),
    e.bge(RDI, RSI, ".done"),
    e.mov(RAX, RSI),
 e.label(".done"),
    e.ret())
e.finalize()
max_fn_ptr = e.symbol("max")

assert ccall(max_fn_ptr, 3, 7)   == 7
assert ccall(max_fn_ptr, 42, -1) == 42
```

`finalize()` allocates executable memory and resolves labels. `symbol()` returns
the address of a public label, and `ctypes.CFUNCTYPE` converts that address into
a callable Python object. Call `unmap()` when the generated code is no longer
needed. Functions created from `symbol()` must not be called after unmapping.
The emitter can be finalized again to create a fresh mapping.

See [test_qsort.py](./tests/test_qsort.py) for a larger example on Linux.

## Windows

Windows is using a different ABI, registers for function arguments are `RCX`, `RDX`, `R8`, `R9`:

```python
from jitasm.x86_64 import *
from jitasm.utils import ccall

e = Emitter()

e.label('add_two')
e.mov(RAX, RCX)
e.add(RAX, RDX)
e.ret()

e.finalize()
assert ccall(e.symbol('add_two'), 20, 22) == 42
```

## Assembly Spec

```
`m8/m16/m32/m64/m128/m256/m*`: `[r64]`
                   | `[r64 + r64 * scale +/- simm32]` (scale = 1/2/4/8)
                   | `[r64 * scale +/- simm32]` (scale = 1/2/4/8)
                   | `[r64 +/- simm32]`
                   | `[rip + rel32]`
`rel32`: label 
`cond`: EQ | NE | LT | GT | LE | GE | LTU | GTU | GEU | LEU | P | NP | O | NO | S | NS
```

### Implemented Instructions

Only an essential subset of the x86-64 instruction set with several pseudo-instructions is implemented.

#### Data directives

- `DB`:     `db int8...`
- `DW`:     `dw int16...`
- `DD`:     `dd int32...` / `dd float32...` / `dd (target_label, base_label)...`
- `DQ`:     `dq int64...` / `dq float64...`
- `ASCII`:  `ascii str`
- `ASCIZ`:  `asciz str`
- `ALIGN`:  `align bytes`                    // DATA section zero-padding

#### Data movement and addressing

- `MOV`:    `mov r64, r64`
- `MOV`:    `mov r64, imm64`                  // zero uses `XOR r64, r64`
- `MOV`:    `mov r64, m64`
- `MOV`:    `mov m64, r64`
- `MOV`:    `mov m32, r64`  // low bits
- `MOV`:    `mov m16, r64`  // low bits
- `MOV`:    `mov m8, r64`   // low bits
- `MOVZX`:  `movzx r64, r8`
- `MOVZX`:  `movzx r64, r16`
- `MOVZX`:  `movzx r64, r32`
- `MOVZX`:  `movzx r64, m8`
- `MOVZX`:  `movzx r64, m16`
- `MOVZX`:  `movzx r64, m32`
- `MOVSX`:  `movsx r64, r8`
- `MOVSX`:  `movsx r64, r16`
- `MOVSX`:  `movsx r64, r32`
- `MOVSX`:  `movsx r64, m8`
- `MOVSX`:  `movsx r64, m16`
- `MOVSX`:  `movsx r64, m32`
- `LEA`:    `lea r64, m*`

#### Integer arithmetic and bitwise operations

- `ADD`:    `add r64, r64` / `add r64, simm32`
- `SUB`:    `sub r64, r64` / `sub r64, simm32`
- `BITAND`: `bitand r64, r64` / `bitand r64, simm32`
- `BITOR`:  `bitor r64, r64` / `bitor r64, simm32`
- `XOR`:    `xor r64, r64` / `xor r64, simm32`
- `BITNOT`: `bitnot r64`                     // `XOR r64, -1`
- `NEG`:    `neg r64`
- `IMUL`:   `imul r64, r64` / `imul r64, simm32`
- `IDIV`:   `idiv r64, r64`                  // quotient, remainder; clobbers RAX and RDX
- `DIV`:    `div r64, r64`                   // quotient, remainder; clobbers RAX and RDX
- `SHL`:    `shl r64, r64` / `shl r64, uimm8` // register form clobbers RCX
- `SAR`:    `sar r64, r64` / `sar r64, uimm8` // register form clobbers RCX
- `SHR`:    `shr r64, r64` / `shr r64, uimm8` // register form clobbers RCX
- `ROR`:    `ror r64, r64` / `ror r64, uimm8` // register form clobbers RCX
- `ROL`:    `rol r64, r64` / `rol r64, uimm8` // register form clobbers RCX

#### Scalar floating point

`MOVSS`, `MOVSD`, and scalar arithmetic select AVX when available and otherwise use SSE.

- `MOVSS`:  `movss xmm, xmm`
- `MOVSS`:  `movss xmm, m32`
- `MOVSS`:  `movss m32, xmm`
- `MOVSD`:  `movsd xmm, xmm`
- `MOVSD`:  `movsd xmm, m64`
- `MOVSD`:  `movsd m64, xmm`
- `ADDSS`:  `addss xmm, xmm`
- `SUBSS`:  `subss xmm, xmm`
- `MULSS`:  `mulss xmm, xmm`
- `DIVSS`:  `divss xmm, xmm`
- `ADDSD`:  `addsd xmm, xmm`
- `SUBSD`:  `subsd xmm, xmm`
- `MULSD`:  `mulsd xmm, xmm`
- `DIVSD`:  `divsd xmm, xmm`
- `CVTSI2SS`:  `cvtsi2ss xmm, r64`
- `CVTTSS2SI`: `cvttss2si r64, xmm`
- `CVTSI2SD`:  `cvtsi2sd xmm, r64`
- `CVTTSD2SI`: `cvttsd2si r64, xmm`
- `ROUNDS`: `rounds xmm, xmm`                 // single precision, round to nearest, ties to even
- `CEILS`:  `ceils xmm, xmm`
- `FLOORS`: `floors xmm, xmm`
- `TRUNCS`: `truncs xmm, xmm`
- `ROUNDD`: `roundd xmm, xmm`                 // double precision, round to nearest, ties to even
- `CEILD`:  `ceild xmm, xmm`
- `FLOORD`: `floord xmm, xmm`
- `TRUNCD`: `truncd xmm, xmm`

#### AVX SIMD

These instructions require AVX. Memory operands for `vmovaps` must be aligned
to 16 bytes for XMM or 32 bytes for YMM; `vmovups` has no alignment requirement.

- `VMOVAPS`: `vmovaps xmm, xmm` / `vmovaps ymm, ymm`
- `VMOVAPS`: `vmovaps xmm, m128` / `vmovaps m128, xmm`
- `VMOVAPS`: `vmovaps ymm, m256` / `vmovaps m256, ymm`
- `VMOVUPS`: `vmovups xmm, xmm` / `vmovups ymm, ymm`
- `VMOVUPS`: `vmovups xmm, m128` / `vmovups m128, xmm`
- `VMOVUPS`: `vmovups ymm, m256` / `vmovups m256, ymm`
- `VADDPS`:  `vaddps xmm, xmm, xmm` / `vaddps ymm, ymm, ymm`
- `VSUBPS`:  `vsubps xmm, xmm, xmm` / `vsubps ymm, ymm, ymm`
- `VMULPS`:  `vmulps xmm, xmm, xmm` / `vmulps ymm, ymm, ymm`
- `VDIVPS`:  `vdivps xmm, xmm, xmm` / `vdivps ymm, ymm, ymm`
- `VADDSUBPS`: `vaddsubps xmm, xmm, xmm` / `vaddsubps ymm, ymm, ymm`
- `VSQRTPS`: `vsqrtps xmm, xmm` / `vsqrtps ymm, ymm`
- `VMAXPS`:  `vmaxps xmm, xmm, xmm` / `vmaxps ymm, ymm, ymm`
- `VMINPS`:  `vminps xmm, xmm, xmm` / `vminps ymm, ymm, ymm`
- `VANDPS`:  `vandps xmm, xmm, xmm` / `vandps ymm, ymm, ymm`
- `VANDNPS`: `vandnps xmm, xmm, xmm` / `vandnps ymm, ymm, ymm`
- `VORPS`:   `vorps xmm, xmm, xmm` / `vorps ymm, ymm, ymm`
- `VXORPS`:  `vxorps xmm, xmm, xmm` / `vxorps ymm, ymm, ymm`
- `VHADDPS`: `vhaddps xmm, xmm, xmm` / `vhaddps ymm, ymm, ymm`
- `VHSUBPS`: `vhsubps xmm, xmm, xmm` / `vhsubps ymm, ymm, ymm`
- `VDPPS`:   `vdpps xmm, xmm, xmm, input_mask, output_mask`
- `VDPPS`:   `vdpps ymm, ymm, ymm, input_mask, output_mask`
- `VRCPPS`:  `vrcpps xmm, xmm` / `vrcpps ymm, ymm`
- `VRSQRTPS`: `vrsqrtps xmm, xmm` / `vrsqrtps ymm, ymm`
- `VROUNDPS`: `vroundps xmm, xmm` / `vroundps ymm, ymm`  // round to nearest, ties to even
- `VFLOORPS`: `vfloorps xmm, xmm` / `vfloorps ymm, ymm`
- `VCEILPS`:  `vceilps xmm, xmm` / `vceilps ymm, ymm`
- `VTRUNCPS`: `vtruncps xmm, xmm` / `vtruncps ymm, ymm`
- `VCMPPS`: `vcmpps xmm, xmm, xmm, predicate` / `vcmpps ymm, ymm, ymm, predicate`
- `VCMPPS` helpers: `veqps/vltps/vleps/vunordps/vneps/vnltps/vnleps/vordps/vgtps/vgeps`
- `VBLENDPS`: `vblendps xmm, xmm, xmm, mask` / `vblendps ymm, ymm, ymm, mask`
- `VPTEST`: `vptest xmm, xmm` / `vptest ymm, ymm`  // set ZF and CF from packed bitwise tests
- `VZEROUPPER`: `vzeroupper`  // clear bits 128–255 of all YMM registers

#### Comparisons and branches

Floating-point comparisons select AVX when available and otherwise use SSE.

- `CMP`:    `cmp r64, r64` / `cmp r64, simm32`
- `UCOMISS`: `ucomiss xmm, xmm`
- `UCOMISD`: `ucomisd xmm, xmm`
- `JCC`:    `jcc cond, rel32`
- `Jcc` helpers (each takes `rel32`):
  - Equality: `je/jeq/jz`, `jne/jnz`
  - Unsigned: `ja/jnbe/jgtu`, `jae/jnb/jnc/jgeu`, `jb/jnae/jc/jltu`, `jbe/jna/jleu`
  - Signed: `jg/jnle/jgt`, `jge/jnl`, `jl/jnge/jlt`, `jle/jng`
  - Flags: `jo/jno` (overflow), `js/jns` (sign), `jp/jpe/jnp/jpo` (parity)
- `SETCC`:  `setcc cond, r8`
- `CMOVcc`: `cmoveq/cmovne/cmovgt/cmovge/cmovlt/cmovle r64, r64`
- `CMOVcc`: `cmovgtu/cmovgeu/cmovltu/cmovleu/cmovp r64, r64`
- `BRANCH`: `branch cond, r64, r64, rel32`
- `BRANCH`: `branch cond, r64, simm32, rel32`
- `BRANCHS`: `branchs cond, xmm, xmm, rel32`  // `beqs/bnes/bgts/bges/blts/bles`
- `BRANCHD`: `branchd cond, xmm, xmm, rel32`  // `beqd/bned/bgtd/bged/bltd/bled`
- `CSET`:   `cset cond, r64, r64, r8`
- `CSET`:   `cset cond, r64, simm32, r8`
- `CSETS`:  `csets cond, xmm, xmm, r8`        // `seteqs/setnes/setgts/setges/setlts/setles`
- `CSETD`:  `csetd cond, xmm, xmm, r8`        // `seteqd/setned/setgtd/setged/setltd/setled`

#### Stack, calls, and control flow

- `PUSH`:   `push r64`                        // pseudo-instruction
- `POP`:    `pop r64`                         // pseudo-instruction
- `BEGIN`:  `begin`                           // `PUSH RBP` + `MOV RBP, RSP`
- `END`:    `end`                             // `MOV RSP, RBP` + `POP RBP` + `RET`
- `CALL`:   `call rel32`
- `CALL`:   `call r64`
- `JMP`:    `jmp rel32`
- `JMP`:    `jmp r64`
- `RET`:    `ret`

#### System

- `CPUID`:  `cpuid`                           // query processor identification and features
