# Copyright (c) 2026, Mistivia <i@mistivia.com>
# Distributed under the terms of the GPLv3

import ctypes
import struct
from jitasm.system import memory_map, unmap, set_mem_rx, get_page_size
from enum import Enum

class CpuFeatures:
    def __init__(self, avx, avx2, fma):
        assert type(avx)  is bool
        assert type(avx2) is bool
        assert type(fma)  is bool
        self.avx  = avx
        self.avx2 = avx2
        self.fma  = fma

cpu_features = CpuFeatures(False, False, False)

class VexPP(Enum):
    NONE = 0b00
    P66  = 0b01
    PF3  = 0b10
    PF2  = 0b11

class VexMap(Enum):
    MAP_0F   = 0b00001
    MAP_0F38 = 0b00010
    MAP_0F3A = 0b00011

class VexL(Enum):
    L128 = 0
    L256 = 1

class VexW(Enum):
    W0 = 0
    W1 = 1

class WordSize(Enum):
    BYTE  = 8
    WORD  = 16
    DWORD = 32
    QWORD = 64
    M128  = 128
    M256  = 256

BYTE = WordSize.BYTE
WORD = WordSize.WORD
DWORD = WordSize.DWORD
QWORD = WordSize.QWORD
M128  = WordSize.M128
M256  = WordSize.M256

class CondCode(Enum):
    EQ = 'eq'
    NE = 'ne'
    GT = 'gt'
    GE = 'ge'
    LT = 'lt'
    LE = 'le'
    GTU = 'gtu'
    GEU = 'geu'
    LTU = 'ltu'
    LEU = 'leu'
    P  = 'p'
    NP = 'np'
    O  = 'o'
    NO = 'no'
    S  = 's'
    NS = 'ns'

EQ = CondCode.EQ
NE = CondCode.NE
GT = CondCode.GT
GE = CondCode.GE
LT = CondCode.LT
LE = CondCode.LE
GTU = CondCode.GTU
GEU = CondCode.GEU
LTU = CondCode.LTU
LEU = CondCode.LEU
P  = CondCode.P
NP = CondCode.NP
O  = CondCode.O
NO = CondCode.NO
S  = CondCode.S
NS = CondCode.NS

COND_CODE_IDS = {
    CondCode.EQ: 0x4,
    CondCode.NE: 0x5,
    CondCode.GT: 0xF,
    CondCode.GE: 0xD,
    CondCode.LT: 0xC,
    CondCode.LE: 0xE,
    CondCode.GTU: 0x7,
    CondCode.GEU: 0x3,
    CondCode.LTU: 0x2,
    CondCode.LEU: 0x6,
    CondCode.P:  0xA,
    CondCode.NP: 0xB,
    CondCode.O:  0x0,
    CondCode.NO: 0x1,
    CondCode.S:  0x8,
    CondCode.NS: 0x9,
}

def xmm_cond_code(cond):
    assert type(cond) is CondCode
    if cond is CondCode.GT:
        return CondCode.GTU
    elif cond is CondCode.GE:
        return CondCode.GEU
    elif cond is CondCode.LT:
        return CondCode.LTU
    elif cond is CondCode.LE:
        return CondCode.LEU
    elif cond in (CondCode.EQ, CondCode.NE, CondCode.P, CondCode.NP):
        return cond
    elif cond in (CondCode.GTU, CondCode.GEU, CondCode.LTU, CondCode.LEU):
        raise EmitterError('unsigned condition code cannot be used with xmm operands')
    elif cond in (CondCode.O, CondCode.NO, CondCode.S, CondCode.NS):
        raise EmitterError('overflow and sign conditions cannot be used with xmm operands')
    else:
        assert False

class RegName(Enum):
    RAX = 'rax'
    RBX = 'rbx'
    RCX = 'rcx'
    RDX = 'rdx'
    RDI = 'rdi'
    RSI = 'rsi'
    RBP = 'rbp'
    RSP = 'rsp'
    RIP = 'rip'
    R8  = 'r8'
    R9  = 'r9'
    R10 = 'r10'
    R11 = 'r11'
    R12 = 'r12'
    R13 = 'r13'
    R14 = 'r14'
    R15 = 'r15'

class Reg:
    def __init__(self, name, size):
        assert type(name) is RegName
        assert type(size) is WordSize
        self.name = name
        self.size = size

    def __mul__(self, scale):
        assert type(scale) is int
        if self.name == RegName.RIP:
            raise EmitterError('rip can only be added to a label')
        return Sib(index=self, scale=scale)

    def __rmul__(self, scale):
        assert type(scale) is int
        return self * scale

    def __add__(self, other):
        assert type(other) in (Reg, Sib, int, str)
        if self.name == RegName.RIP:
            if type(other) is str:
                return Rel(other)
            raise EmitterError('rip can only be added to a label')
        if type(other) is Reg:
            index = other
            if index.name == RegName.RIP:
                raise EmitterError('rip can only be added to a label')
            return Sib(self, index)
        elif type(other) is Sib:
            return other.__radd__(self)
        elif type(other) is int:
            return Sib(self, offset=other)
        else:
            raise EmitterError('invalid register address expression')

    def __radd__(self, other):
        assert type(other) is int
        if self.name == RegName.RIP:
            raise EmitterError('rip can only be added to a label')
        if type(other) is int:
            return Sib(self, offset=other)
        else:
            assert False

    def __sub__(self, other):
        assert type(other) is int
        return self + -other

class EmitterError(RuntimeError):
    pass

def require_avx():
    if not cpu_features.avx:
        raise EmitterError('cannot encode VEX instruction without AVX support')

def require_avx2():
    if not cpu_features.avx2:
        raise EmitterError('cannot encode AVX2 instruction without AVX2 support')

def require_fma():
    if not cpu_features.fma:
        raise EmitterError('cannot encode FMA instruction without FMA support')

RAX = Reg(RegName.RAX, QWORD)
RBX = Reg(RegName.RBX, QWORD)
RCX = Reg(RegName.RCX, QWORD)
RDX = Reg(RegName.RDX, QWORD)
RDI = Reg(RegName.RDI, QWORD)
RSI = Reg(RegName.RSI, QWORD)
RBP = Reg(RegName.RBP, QWORD)
RSP = Reg(RegName.RSP, QWORD)
RIP = Reg(RegName.RIP, QWORD)
R8  = Reg(RegName.R8,  QWORD)
R9  = Reg(RegName.R9,  QWORD)
R10 = Reg(RegName.R10, QWORD)
R11 = Reg(RegName.R11, QWORD)
R12 = Reg(RegName.R12, QWORD)
R13 = Reg(RegName.R13, QWORD)
R14 = Reg(RegName.R14, QWORD)
R15 = Reg(RegName.R15, QWORD)

rax = Reg(RegName.RAX, QWORD)
rbx = Reg(RegName.RBX, QWORD)
rcx = Reg(RegName.RCX, QWORD)
rdx = Reg(RegName.RDX, QWORD)
rdi = Reg(RegName.RDI, QWORD)
rsi = Reg(RegName.RSI, QWORD)
rbp = Reg(RegName.RBP, QWORD)
rsp = Reg(RegName.RSP, QWORD)
rip = Reg(RegName.RIP, QWORD)
r8  = Reg(RegName.R8,  QWORD)
r9  = Reg(RegName.R9,  QWORD)
r10 = Reg(RegName.R10, QWORD)
r11 = Reg(RegName.R11, QWORD)
r12 = Reg(RegName.R12, QWORD)
r13 = Reg(RegName.R13, QWORD)
r14 = Reg(RegName.R14, QWORD)
r15 = Reg(RegName.R15, QWORD)

EAX  = Reg(RegName.RAX, DWORD)
EBX  = Reg(RegName.RBX, DWORD)
ECX  = Reg(RegName.RCX, DWORD)
EDX  = Reg(RegName.RDX, DWORD)
EDI  = Reg(RegName.RDI, DWORD)
ESI  = Reg(RegName.RSI, DWORD)
EBP  = Reg(RegName.RBP, DWORD)
ESP  = Reg(RegName.RSP, DWORD)
R8D  = Reg(RegName.R8,  DWORD)
R9D  = Reg(RegName.R9,  DWORD)
R10D = Reg(RegName.R10, DWORD)
R11D = Reg(RegName.R11, DWORD)
R12D = Reg(RegName.R12, DWORD)
R13D = Reg(RegName.R13, DWORD)
R14D = Reg(RegName.R14, DWORD)
R15D = Reg(RegName.R15, DWORD)

eax = Reg(RegName.RAX, DWORD)
ebx = Reg(RegName.RBX, DWORD)
ecx = Reg(RegName.RCX, DWORD)
edx = Reg(RegName.RDX, DWORD)
edi = Reg(RegName.RDI, DWORD)
esi = Reg(RegName.RSI, DWORD)
ebp = Reg(RegName.RBP, DWORD)
esp = Reg(RegName.RSP, DWORD)
r8d  = Reg(RegName.R8,  DWORD)
r9d  = Reg(RegName.R9,  DWORD)
r10d = Reg(RegName.R10, DWORD)
r11d = Reg(RegName.R11, DWORD)
r12d = Reg(RegName.R12, DWORD)
r13d = Reg(RegName.R13, DWORD)
r14d = Reg(RegName.R14, DWORD)
r15d = Reg(RegName.R15, DWORD)

AX   = Reg(RegName.RAX, WORD)
BX   = Reg(RegName.RBX, WORD)
CX   = Reg(RegName.RCX, WORD)
DX   = Reg(RegName.RDX, WORD)
DI   = Reg(RegName.RDI, WORD)
SI   = Reg(RegName.RSI, WORD)
BP   = Reg(RegName.RBP, WORD)
SP   = Reg(RegName.RSP, WORD)
R8W  = Reg(RegName.R8,  WORD)
R9W  = Reg(RegName.R9,  WORD)
R10W = Reg(RegName.R10, WORD)
R11W = Reg(RegName.R11, WORD)
R12W = Reg(RegName.R12, WORD)
R13W = Reg(RegName.R13, WORD)
R14W = Reg(RegName.R14, WORD)
R15W = Reg(RegName.R15, WORD)

ax = Reg(RegName.RAX, WORD)
bx = Reg(RegName.RBX, WORD)
cx = Reg(RegName.RCX, WORD)
dx = Reg(RegName.RDX, WORD)
di = Reg(RegName.RDI, WORD)
si = Reg(RegName.RSI, WORD)
bp = Reg(RegName.RBP, WORD)
sp = Reg(RegName.RSP, WORD)
r8w  = Reg(RegName.R8,  WORD)
r9w  = Reg(RegName.R9,  WORD)
r10w = Reg(RegName.R10, WORD)
r11w = Reg(RegName.R11, WORD)
r12w = Reg(RegName.R12, WORD)
r13w = Reg(RegName.R13, WORD)
r14w = Reg(RegName.R14, WORD)
r15w = Reg(RegName.R15, WORD)

AL   = Reg(RegName.RAX, BYTE)
BL   = Reg(RegName.RBX, BYTE)
CL   = Reg(RegName.RCX, BYTE)
DL   = Reg(RegName.RDX, BYTE)
DIL  = Reg(RegName.RDI, BYTE)
SIL  = Reg(RegName.RSI, BYTE)
BPL  = Reg(RegName.RBP, BYTE)
SPL  = Reg(RegName.RSP, BYTE)
R8B  = Reg(RegName.R8,  BYTE)
R9B  = Reg(RegName.R9,  BYTE)
R10B = Reg(RegName.R10, BYTE)
R11B = Reg(RegName.R11, BYTE)
R12B = Reg(RegName.R12, BYTE)
R13B = Reg(RegName.R13, BYTE)
R14B = Reg(RegName.R14, BYTE)
R15B = Reg(RegName.R15, BYTE)

al = Reg(RegName.RAX, BYTE)
bl = Reg(RegName.RBX, BYTE)
cl = Reg(RegName.RCX, BYTE)
dl = Reg(RegName.RDX, BYTE)
dil = Reg(RegName.RDI, BYTE)
sil = Reg(RegName.RSI, BYTE)
bpl = Reg(RegName.RBP, BYTE)
spl = Reg(RegName.RSP, BYTE)
r8b  = Reg(RegName.R8,  BYTE)
r9b  = Reg(RegName.R9,  BYTE)
r10b = Reg(RegName.R10, BYTE)
r11b = Reg(RegName.R11, BYTE)
r12b = Reg(RegName.R12, BYTE)
r13b = Reg(RegName.R13, BYTE)
r14b = Reg(RegName.R14, BYTE)
r15b = Reg(RegName.R15, BYTE)

REG_IDS = {
    RegName.RAX: 0,  RegName.RCX: 1,  RegName.RDX: 2,  RegName.RBX: 3,
    RegName.RSP: 4,  RegName.RBP: 5,  RegName.RSI: 6,  RegName.RDI: 7,
    RegName.R8:  8,  RegName.R9:  9,  RegName.R10: 10, RegName.R11: 11,
    RegName.R12: 12, RegName.R13: 13, RegName.R14: 14, RegName.R15: 15,
}

def reg_id(reg):
    assert type(reg) is Reg
    return REG_IDS[reg.name]

def signed_bytes(value, size):
    assert type(value) is int
    assert type(size) is int
    min_value = -(1 << (size * 8 - 1))
    max_value = (1 << (size * 8 - 1)) - 1
    if value < min_value or value > max_value:
        return None
    return value.to_bytes(size, 'little', signed=True)

class Xmm:
    def __init__(self, id):
        assert type(id) is int
        self.id = id

XMM0  = Xmm(0)
XMM1  = Xmm(1)
XMM2  = Xmm(2)
XMM3  = Xmm(3)
XMM4  = Xmm(4)
XMM5  = Xmm(5)
XMM6  = Xmm(6)
XMM7  = Xmm(7)
XMM8  = Xmm(8)
XMM9  = Xmm(9)
XMM10 = Xmm(10)
XMM11 = Xmm(11)
XMM12 = Xmm(12)
XMM13 = Xmm(13)
XMM14 = Xmm(14)
XMM15 = Xmm(15)

xmm0  = Xmm(0)
xmm1  = Xmm(1)
xmm2  = Xmm(2)
xmm3  = Xmm(3)
xmm4  = Xmm(4)
xmm5  = Xmm(5)
xmm6  = Xmm(6)
xmm7  = Xmm(7)
xmm8  = Xmm(8)
xmm9  = Xmm(9)
xmm10 = Xmm(10)
xmm11 = Xmm(11)
xmm12 = Xmm(12)
xmm13 = Xmm(13)
xmm14 = Xmm(14)
xmm15 = Xmm(15)

class Ymm:
    def __init__(self, id):
        assert type(id) is int
        self.id = id

YMM0  = Ymm(0)
YMM1  = Ymm(1)
YMM2  = Ymm(2)
YMM3  = Ymm(3)
YMM4  = Ymm(4)
YMM5  = Ymm(5)
YMM6  = Ymm(6)
YMM7  = Ymm(7)
YMM8  = Ymm(8)
YMM9  = Ymm(9)
YMM10 = Ymm(10)
YMM11 = Ymm(11)
YMM12 = Ymm(12)
YMM13 = Ymm(13)
YMM14 = Ymm(14)
YMM15 = Ymm(15)

ymm0  = Ymm(0)
ymm1  = Ymm(1)
ymm2  = Ymm(2)
ymm3  = Ymm(3)
ymm4  = Ymm(4)
ymm5  = Ymm(5)
ymm6  = Ymm(6)
ymm7  = Ymm(7)
ymm8  = Ymm(8)
ymm9  = Ymm(9)
ymm10 = Ymm(10)
ymm11 = Ymm(11)
ymm12 = Ymm(12)
ymm13 = Ymm(13)
ymm14 = Ymm(14)
ymm15 = Ymm(15)

def encode_vex(dst, src1, src2, opcode, vex_map, pp, w, imm = None):
    assert type(dst)     in (Xmm, Ymm)
    assert type(src1)    in (Xmm, Ymm, type(None))
    assert type(src2)    in (Xmm, Ymm)
    assert type(opcode)  is int
    assert type(vex_map) is VexMap
    assert type(pp)      is VexPP
    assert type(w)       is VexW
    assert type(imm)     in (int, type(None))
    if dst.id < 0 or dst.id > 15:
        raise EmitterError('invalid VEX register')
    if src1 is not None and (src1.id < 0 or src1.id > 15):
        raise EmitterError('invalid VEX register')
    if src2.id < 0 or src2.id > 15:
        raise EmitterError('invalid VEX register')
    if opcode < 0 or opcode > 0xFF:
        raise EmitterError('VEX opcode must fit in one byte')

    if type(dst) is Ymm:
        l = VexL.L256
    else:
        l = VexL.L128
    byte2 = ((~(dst.id >> 3) & 1) << 7) | (1 << 6) | ((~(src2.id >> 3) & 1) << 5) | vex_map.value
    if src1 is None:
        vvvv = 0b1111
    else:
        vvvv = ~src1.id & 0b1111
    byte3 = (w.value << 7) | (vvvv << 3) | (l.value << 2) | pp.value
    mod_rm = (0b11 << 6) | ((dst.id & 0b111) << 3) | (src2.id & 0b111)
    if imm is not None and (imm > 0xFF or imm < 0):
        raise EmitterError('VEX: invalid immediate number')
    if imm is None:
        result = bytes((0xC4, byte2, byte3, opcode, mod_rm))
    else:
        result = bytes((0xC4, byte2, byte3, opcode, mod_rm, imm))
    assert type(result) is bytes
    return result

def encode_vex_rm(dst, src, l, opcode, vex_map, pp, w):
    assert type(dst)     in (Mem, int)
    assert type(src)     in (int, Mem)
    assert type(l)       is VexL
    assert type(opcode)  is int
    assert type(vex_map) is VexMap
    assert type(pp)      is VexPP
    assert type(w)       is VexW
    if opcode < 0 or opcode > 0xFF:
        raise EmitterError('VEX opcode must fit in one byte')
    if type(dst) is int and type(src) is Mem:
        reg = dst
        mem = src
    elif type(dst) is Mem and type(src) is int:
        mem = dst
        reg = src
    else:
        raise EmitterError('VEX r/m encoding requires one register and one memory operand')
    if reg < 0 or reg > 15:
        raise EmitterError('invalid VEX register')
    encoded = EncodedRegMemOp(mem, reg)
    byte2 = ((~(reg >> 3) & 1) << 7) | ((~(encoded.rex >> 1) & 1) << 6) | ((~encoded.rex & 1) << 5) | vex_map.value
    byte3 = (w.value << 7) | (0b1111 << 3) | (l.value << 2) | pp.value
    result = bytes((0xC4, byte2, byte3, opcode, encoded.mod_rm)) + encoded.suffix
    assert type(result) is bytes
    return result

class Sib: # r64 + r64 * scale + offset
    def __init__(self, base = None, index = None, scale = 1, offset = 0):
        assert type(base)   in (Reg, type(None))
        assert type(index)  in (Reg, type(None))
        assert type(scale)  is int
        assert type(offset) is int
        self.base = base
        self.index = index
        self.scale = scale
        self.offset = offset
    
    def __eq__(self, other):
        if type(other) is not Sib:
            return False
        return self.base == other.base and self.index == other.index \
            and self.scale == other.scale and self.offset == other.offset

    def __add__(self, other):
        assert type(other) in (Reg, Sib, int)
        if type(other) is int:
            offset = other
            result = Sib(self.base, self.index, self.scale, self.offset + offset)
        elif type(other) is Reg:
            reg = other
            if reg.name == RegName.RIP:
                raise EmitterError('rip can only be added to a label')
            if self.base is None:
                result = Sib(reg, self.index, self.scale, self.offset)
            elif self.index is None:
                result = Sib(self.base, reg, 1, self.offset)
            else:
                raise EmitterError('address expression already has a base and index')
        elif type(other) is Sib:
            sib = other
            if self.base is not None and sib.base is not None:
                raise EmitterError('both address expressions have a base')
            if self.index is not None and sib.index is not None:
                raise EmitterError('both address expressions have an index')
            if self.base is not None:
                base = self.base
            else:
                base = sib.base
            if self.index is not None:
                index = self.index
                scale = self.scale
            else:
                index = sib.index
                scale = sib.scale
            result = Sib(base, index, scale, self.offset + sib.offset)
        else:
            assert False
        assert type(result) is Sib
        return result

    def __radd__(self, other): # returns Sib
        assert type(other) in (Reg, Sib, int)
        return self + other

    def __sub__(self, other): # returns Sib
        assert type(other) is int
        return self + -other

def validate_sib(sib):
    assert type(sib) is Sib
    if sib.scale not in (1, 2, 4, 8):
        raise EmitterError('invalid SIB scale')
    if signed_bytes(sib.offset, 4) is None:
        raise EmitterError('invalid SIB displacement')
    if sib.base is None and sib.index is None:
        raise EmitterError('SIB must have a base or index register')
    if sib.base is not None and (sib.base.size != QWORD or sib.base.name == RegName.RIP):
        raise EmitterError('invalid SIB base register')
    if sib.index is not None and (sib.index.size != QWORD or sib.index.name == RegName.RIP):
        raise EmitterError('invalid SIB index register')
    if sib.index is not None and sib.index.name == RegName.RSP:
        raise EmitterError('rsp cannot be used as a SIB index register')
    if sib.index is None and sib.scale != 1:
        raise EmitterError('SIB scale requires an index register')

class Rel: # relative to rip
    def __init__(self, label):
        assert type(label) is str
        self.label = label

    def __eq__(self, other):
        return type(other) == Rel and self.label == other.label

class Mem:
    def __init__(self, size, addr):
        assert type(size) is WordSize
        assert type(addr) in (Reg, Sib, Rel, Xmm, Ymm)
        self.size = size
        self.addr = addr

    def __eq__(self, other):
        return type(other) == Mem and self.size == other.size and self.addr == other.addr

def byte_ptr(addr):
    assert type(addr) in (Reg, Sib, Rel)
    if addr == RIP:
        raise EmitterError('rip requires a relative label')
    return Mem(BYTE, addr)

def word_ptr(addr):
    assert type(addr) in (Reg, Sib, Rel)
    if addr == RIP:
        raise EmitterError('rip requires a relative label')
    return Mem(WORD, addr)

def dword_ptr(addr):
    assert type(addr) in (Reg, Sib, Rel)
    if addr == RIP:
        raise EmitterError('rip requires a relative label')
    return Mem(DWORD, addr)

def qword_ptr(addr):
    assert type(addr) in (Reg, Sib, Rel)
    if addr == RIP:
        raise EmitterError('rip requires a relative label')
    return Mem(QWORD, addr)

def m128_ptr(addr):
    assert type(addr) in (Reg, Sib, Rel)
    if addr == RIP:
        raise EmitterError('rip requires a relative label')
    return Mem(M128, addr)

def m256_ptr(addr):
    assert type(addr) in (Reg, Sib, Rel)
    if addr == RIP:
        raise EmitterError('rip requires a relative label')
    return Mem(M256, addr)

class MemMap:
    def __init__(self, ptr, size):
        assert type(ptr) is int
        assert type(size) is int
        self.ptr = ptr
        self.size = size

def close_mem_map(mapping: MemMap):
    unmap(mapping.ptr, mapping.size)

class EncodedRegMemOp:
    def __init__(self, mem, reg_id):
        assert type(mem)    is Mem
        assert type(reg_id) is int
        
        rex = 0
        suffix = bytearray()

        addr = mem.addr
        if type(addr) is Reg:
            base = addr
            if base == RIP or base.size != QWORD:
                raise EmitterError('invalid base register type')
            base_id = REG_IDS[base.name]
            rex |= base_id >> 3
            rm = base_id & 7
            if rm == 4:
                mod_rm = ((reg_id & 7) << 3) | 4
                suffix.append(0x20 | rm)
            elif rm == 5:
                mod_rm = 0x40 | ((reg_id & 7) << 3) | rm
                suffix.append(0)
            else:
                mod_rm = ((reg_id & 7) << 3) | rm
        elif type(addr) is Rel:
            mod_rm = ((reg_id & 7) << 3) | 5
            suffix.extend(b'\x00\x00\x00\x00')
        elif type(addr) is Sib:
            validate_sib(addr)
            if addr.index is None:
                index_bits = 4
            else:
                index_id = REG_IDS[addr.index.name]
                index_bits = index_id & 7
                rex |= (index_id >> 3) << 1

            scale_bits = {1: 0, 2: 1, 4: 2, 8: 3}[addr.scale]
            if addr.base is None:
                displacement = signed_bytes(addr.offset, 4)
                if displacement is None:
                    raise EmitterError('invalid displacement')
                mod = 0
                base_bits = 5
            else:
                base_id = REG_IDS[addr.base.name]
                base_bits = base_id & 7
                rex |= base_id >> 3
                if addr.offset == 0 and base_bits != 5:
                    mod = 0
                    displacement = b''
                else:
                    displacement = signed_bytes(addr.offset, 1)
                    if displacement is not None:
                        mod = 1
                    else:
                        displacement = signed_bytes(addr.offset, 4)
                        if displacement is None:
                            raise EmitterError('invalid displacement')
                        mod = 2

            mod_rm = (mod << 6) | ((reg_id & 7) << 3) | 4
            suffix.append((scale_bits << 6) | (index_bits << 3) | base_bits)
            suffix.extend(displacement)
        else:
            raise EmitterError("cannnot use xmm or ymm registers in memreg op")

        self.rex = rex
        self.mod_rm = mod_rm
        self.suffix = bytes(suffix)

class Section(Enum):
    TEXT  = 'text'
    DATA  = 'data'

class RipDelta:
    def __init__(self, rip):
        assert type(rip) is int
        self.rip = rip

class LabelDelta:
    def __init__(self, base_label):
        assert type(base_label) is str
        self.base_label = base_label

class LabelRef:
    def __init__(self, position, delta):
        assert type(position) is int
        assert type(delta) in (RipDelta, LabelDelta)
        self.position = position
        self.delta = delta

class Emitter:
    def __init__(self):
        self.text = bytearray(b'')
        self.data = bytearray(b'')
        self.section = Section.TEXT
        self.labels = {}
        self.label_refs ={}
        self.mapping= None
        self.symbols = None

    def add_label_ref(self, name: str, pos: int, delta: RipDelta | LabelDelta):
        self.label_refs.setdefault(name, []).append(LabelRef(pos, delta))

    def symbol(self, s: str):
        if self.symbols is None or s not in self.symbols:
            raise EmitterError('symbol not found')
        result = self.symbols[s]
        assert type(result) is int
        return result

    def emit_bytes(self, b):
        assert type(b) is bytes
        if self.section == Section.TEXT:
            self.text.extend(b)
        if self.section == Section.DATA:
            self.data.extend(b)

    def require_text_section(self, name: str):
        if self.section == Section.DATA:
            raise EmitterError('%s: cannot emit code at data section' % name)

    def finalize(self):
        page_size = get_page_size()
        text_size = max(page_size, (len(self.text) + page_size - 1) // page_size * page_size)
        data_size = max(page_size, (len(self.data) + page_size - 1) // page_size * page_size)
        ptr = memory_map(text_size + data_size)
        mapping = MemMap(ptr, text_size + data_size)
        mapping_address = ptr
        text_address = mapping_address
        data_address = mapping_address + text_size

        patches = []
        for name, references in self.label_refs.items():
            assert type(name)       is str
            assert type(references) is list
            label = self.labels.get(name)
            if label is None:
                close_mem_map(mapping)
                raise EmitterError('link error: label not found')
            label_section, label_offset = label
            if label_section == Section.TEXT:
                label_address = text_address + label_offset
            else:
                label_address = data_address + label_offset
            for ref in references:
                if type(ref.delta) is RipDelta:
                    displacement = label_address - (text_address + ref.delta.rip)
                    encoded = signed_bytes(displacement, 4)
                    if encoded is None or ref.position < 0 or ref.position + 4 > len(self.text):
                        close_mem_map(mapping)
                        raise EmitterError('link error: offset out of range')
                    patches.append((ref.position, encoded))
                elif type(ref.delta) is LabelDelta:
                    base = self.labels.get(ref.delta.base_label)
                    if base is None:
                        close_mem_map(mapping)
                        raise EmitterError('link error: label not found')
                    base_section, base_offset = base
                    base_address = (
                        text_address + base_offset
                        if base_section == Section.TEXT
                        else data_address + base_offset
                    )
                    encoded = signed_bytes(label_address - base_address, 4)
                    if encoded is None or ref.position < 0 or ref.position + 4 > len(self.data):
                        close_mem_map(mapping)
                        raise EmitterError('link error: offset out of range')
                    self.data[ref.position:ref.position + 4] = encoded
                else:
                    assert False

        for reference_offset, encoded in patches:
            self.text[reference_offset:reference_offset + 4] = encoded

        buffer = (ctypes.c_ubyte * mapping.size).from_address(mapping.ptr)
        view = memoryview(buffer).cast('B')
        view[:len(self.text)] = self.text
        view[text_size:text_size + len(self.data)] = self.data

        if set_mem_rx(mapping.ptr, mapping.size) != 0:
            close_mem_map(mapping)
            raise EmitterError('link error: mprotect failed')

        if self.mapping is not None:
            close_mem_map(self.mapping)
        self.mapping = mapping

        symbols = {}
        for name, (section, offset) in self.labels.items():
            assert type(name)    is str
            assert type(section) is Section
            assert type(offset)  is int
            if not name.startswith('.'):
                if section == Section.TEXT:
                    base_address = text_address
                else:
                    base_address = data_address
                symbols[name] = base_address + offset
        self.symbols = symbols
    
    def unmap(self):
        if self.mapping is not None:
            close_mem_map(self.mapping)
            self.mapping = None

    def align(self, numbytes):
        assert type(numbytes) is int
        if self.section != Section.DATA:
            raise EmitterError('align: must be emitted at data section')
        if numbytes <= 0:
            raise EmitterError('align: alignment must be positive')
        padding = -len(self.data) % numbytes
        self.emit_bytes(b'\x00' * padding)

    def db(self, *values):
        if self.section != Section.DATA:
            raise EmitterError('db: must be emitted at data section')
        for value in values:
            assert type(value) is int
            if value < -(1 << 7) or value >= (1 << 8):
                raise EmitterError('db: value must fit in 8 bits')
            self.emit_bytes(bytes((value & 0xFF,)))

    def dw(self, *values):
        if self.section != Section.DATA:
            raise EmitterError('dw: must be emitted at data section')
        for value in values:
            assert type(value) is int
            if value < -(1 << 15) or value >= (1 << 16):
                raise EmitterError('dw: value must fit in 16 bits')
            self.emit_bytes((value & 0xFFFF).to_bytes(2, 'little'))

    def dd(self, *values: int | float | tuple[str, str]):
        if self.section != Section.DATA:
            raise EmitterError('dd: must be emitted at data section')
        for value in values:
            if type(value) is float:
                try:
                    self.emit_bytes(struct.pack('<f', value))
                except OverflowError:
                    raise EmitterError('dd: float must fit in 32 bits') from None
            elif type(value) is tuple and len(value) == 2 and type(value[0]) is str and type(value[1]) is str:
                target_label = value[0]
                base_label = value[1]
                self.add_label_ref(target_label, len(self.data), LabelDelta(base_label))
                self.emit_bytes(b'\x00\x00\x00\x00')
            elif type(value) is int:
                if value < -(1 << 31) or value >= (1 << 32):
                    raise EmitterError('dd: value must fit in 32 bits')
                else:
                    self.emit_bytes((value & 0xFFFFFFFF).to_bytes(4, 'little'))
            else:
                assert False

    def dq(self, *values):
        if self.section != Section.DATA:
            raise EmitterError('dq: must be emitted at data section')
        for value in values:
            if type(value) is float:
                self.emit_bytes(struct.pack('<d', value))
            elif type(value) is int:
                if value < -(1 << 63) or value >= (1 << 64):
                    raise EmitterError('dq: value must fit in 64 bits')
                else:
                    self.emit_bytes((value & 0xFFFFFFFFFFFFFFFF).to_bytes(8, 'little'))
            else:
                assert False

    def ascii(self, value):
        assert type(value) is str
        if self.section != Section.DATA:
            raise EmitterError('ascii: must be emitted at data section')
        try:
            encoded = value.encode('ascii')
        except UnicodeEncodeError:
            raise EmitterError('ascii: value must contain only ASCII characters') from None
        self.emit_bytes(encoded)

    def asciz(self, value):
        assert type(value) is str
        if self.section != Section.DATA:
            raise EmitterError('asciz: must be emitted at data section')
        try:
            encoded = value.encode('ascii')
        except UnicodeEncodeError:
            raise EmitterError('asciz: value must contain only ASCII characters') from None
        self.emit_bytes(encoded + b'\x00')

    def label(self, name):
        assert type(name) is str
        if name in self.labels:
            raise EmitterError('label already defined')
        if self.section == Section.TEXT:
            self.labels[name] = (Section.TEXT, self.section_offset())
        elif self.section == Section.DATA:
            self.labels[name] = (Section.DATA, self.section_offset())
        else:
            raise EmitterError('invalid section')

    def set_section(self, s):
        assert type(s) is Section
        self.section = s

    def section_offset(self):
        if self.section == Section.TEXT:
            return len(self.text)
        return len(self.data)

    def emit_mem_op(self, reg, mem, opcode, rex_w, legacy_prefix = b''):
        assert type(reg) is Reg
        assert type(mem) is Mem
        assert type(opcode) is bytes
        assert type(rex_w) is bool
        assert type(legacy_prefix) is bytes
        reg_index = reg_id(reg)
        if rex_w:
            rex = 0x48
        else:
            rex = 0x40
        rex |= (reg_index >> 3) << 2
        encoded = EncodedRegMemOp(mem, reg_index)
        rex |= encoded.rex
        emit_rex = rex != 0x40 or (mem.size == BYTE and reg_index >= 4)
        if emit_rex:
            rex_prefix = bytes((rex,))
        else:
            rex_prefix = b''
        instruction_start = self.section_offset()
        self.emit_bytes(legacy_prefix + rex_prefix + opcode + bytes((encoded.mod_rm,)) + encoded.suffix)
        if type(mem.addr) is Rel:
            disp_pos = instruction_start + len(legacy_prefix) + len(rex_prefix) + len(opcode) + 1
            self.add_label_ref(mem.addr.label, disp_pos, RipDelta(len(self.text)))

    def emit_mov_mem(self, op1, op2):
        if type(op1) is Reg and type(op2) is Mem:
            reg = op1
            mem = op2
            opcode = b'\x8b'
        elif type(op1) is Mem and type(op2) is Reg:
            mem = op1
            reg = op2
            if mem.size == BYTE:
                opcode = b'\x88'
            else:
                opcode = b'\x89'
        else:
            raise EmitterError("op type error in emit_mov_mem")

        if mem.size == WORD:
            legacy_prefix = b'\x66'
        else:
            legacy_prefix = b''
        self.emit_mem_op(reg, mem, opcode, mem.size == QWORD, legacy_prefix)

    def mov(self, op1, op2):
        self.require_text_section('mov')
        if type(op1) is Reg and type(op2) is Reg:
            if op1 == RIP or op2 == RIP or op1.size != QWORD or op2.size != QWORD:
                raise EmitterError('mov: register must be qword and cannot be rip')
            dst = reg_id(op1)
            src = reg_id(op2)
            rex = 0x48 | ((src >> 3) << 2) | (dst >> 3)
            mod_rm = 0xC0 | ((src & 7) << 3) | (dst & 7)
            self.emit_bytes(bytes((rex, 0x89, mod_rm)))
        elif type(op1) is Reg and type(op2) is int:
            if op1 == RIP or op1.size != QWORD or not -(1 << 63) <= op2 < (1 << 64):
                raise EmitterError('mov: register must be qword and cannot be rip, imm must be 64 bit number')
            if op2 == 0:
                self.xor(op1, op1)
                return
            dst = reg_id(op1)
            rex = 0x48 | (dst >> 3)
            immediate = (op2 & ((1 << 64) - 1)).to_bytes(8, 'little')
            self.emit_bytes(bytes((rex, 0xB8 | (dst & 7))) + immediate)
        elif type(op1) is Reg and type(op2) is Mem:
            if op1 == RIP or op1.size != QWORD or op2.size != QWORD:
                raise EmitterError('mov: register must be qword and cannot be rip')
            self.emit_mov_mem(op1, op2)
        elif type(op1) is Mem and type(op2) is Reg:
            if op2 == RIP or op2.size != QWORD:
                raise EmitterError('mov: register must be qword and cannot be rip')
            self.emit_mov_mem(op1, op2)
        else:
            raise EmitterError('mov: invalid form')

    def movzx(self, op1, op2):
        assert type(op1) in (Reg, Mem)
        assert type(op2) in (Reg, Mem)
        self.require_text_section('movzx')
        if type(op1) is not Reg or op1 == RIP or op1.size != QWORD:
            raise EmitterError('movzx: destination must be a qword register')

        dst = reg_id(op1)
        if type(op2) is Reg:
            src = op2
            if src == RIP or src.size not in (BYTE, WORD, DWORD):
                raise EmitterError('movzx: source must be a byte, word, or dword register')
            src_id = reg_id(src)
            if src.size == DWORD:
                rex = 0x40 | ((dst >> 3) << 2) | (src_id >> 3)
                if rex != 0x40:
                    rex_prefix = bytes((rex,))
                else:
                    rex_prefix = b''
                mod_rm = 0xC0 | ((dst & 7) << 3) | (src_id & 7)
                self.emit_bytes(rex_prefix + bytes((0x8B, mod_rm)))
            else:
                if src.size == BYTE:
                    opcode = 0xB6
                else:
                    opcode = 0xB7
                rex = 0x48 | ((dst >> 3) << 2) | (src_id >> 3)
                mod_rm = 0xC0 | ((dst & 7) << 3) | (src_id & 7)
                self.emit_bytes(bytes((rex, 0x0F, opcode, mod_rm)))
        elif type(op2) is Mem:
            mem = op2
            if mem.size not in (BYTE, WORD, DWORD):
                raise EmitterError('movzx: source must be byte, word, or dword memory')
            if mem.size == DWORD:
                self.emit_mov_mem(op1, mem)
                return
            if mem.size == BYTE:
                opcode = 0xB6
            else:
                opcode = 0xB7
            self.emit_mem_op(op1, mem, bytes((0x0F, opcode)), True)
        else:
            raise EmitterError('movzx: invalid form')

    def movsx(self, op1, op2):
        assert type(op1) in (Reg, Mem)
        assert type(op2) in (Reg, Mem)
        self.require_text_section('movsx')
        if type(op1) is not Reg or op1 == RIP or op1.size != QWORD:
            raise EmitterError('movsx: destination must be a qword register')

        dst = reg_id(op1)
        if type(op2) is Reg:
            src = op2
            if src == RIP or src.size not in (BYTE, WORD, DWORD):
                raise EmitterError('movsx: source must be a byte, word, or dword register')
            src_id = reg_id(src)
            rex = 0x48 | ((dst >> 3) << 2) | (src_id >> 3)
            mod_rm = 0xC0 | ((dst & 7) << 3) | (src_id & 7)
            if src.size == DWORD:
                self.emit_bytes(bytes((rex, 0x63, mod_rm)))
            else:
                if src.size == BYTE:
                    opcode = 0xBE
                else:
                    opcode = 0xBF
                self.emit_bytes(bytes((rex, 0x0F, opcode, mod_rm)))
        elif type(op2) is Mem:
            mem = op2
            if mem.size not in (BYTE, WORD, DWORD):
                raise EmitterError('movsx: source must be byte, word, or dword memory')
            if mem.size == DWORD:
                opcode = b'\x63'
            else:
                if mem.size == BYTE:
                    opcode = b'\x0f\xbe'
                else:
                    opcode = b'\x0f\xbf'
            self.emit_mem_op(op1, mem, opcode, True)
        else:
            raise EmitterError('movsx: invalid form')

    def lea(self, op1, op2):
        self.require_text_section('lea')
        if type(op1) is Reg and type(op2) is Mem:
            dst = op1
            mem = op2
            if dst == RIP or dst.size != QWORD:
                raise EmitterError('lea: destination must be a qword register')
            self.emit_mem_op(dst, mem, b'\x8d', True)
        else:
            raise EmitterError('lea: invalid form')

    def emit_cmov(self, opcode, op1, op2):
        assert type(opcode) is int
        assert type(op1) is Reg
        assert type(op2) is Reg
        self.require_text_section('cmov')
        if op1 == RIP or op1.size != QWORD:
            raise EmitterError('cmov: destination must be a qword register')
        if op2 == RIP or op2.size != QWORD:
            raise EmitterError('cmov: source must be a qword register')
        if opcode < 0x40 or opcode > 0x4F:
            raise EmitterError('cmov: invalid opcode')

        dst = reg_id(op1)
        src = reg_id(op2)
        rex = 0x48 | ((dst >> 3) << 2) | (src >> 3)
        mod_rm = 0xC0 | ((dst & 7) << 3) | (src & 7)
        self.emit_bytes(bytes((rex, 0x0F, opcode, mod_rm)))

    def cmoveq(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Reg
        self.emit_cmov(0x40 | COND_CODE_IDS[EQ], op1, op2)

    def cmovne(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Reg
        self.emit_cmov(0x40 | COND_CODE_IDS[NE], op1, op2)

    def cmovgt(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Reg
        self.emit_cmov(0x40 | COND_CODE_IDS[GT], op1, op2)

    def cmovge(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Reg
        self.emit_cmov(0x40 | COND_CODE_IDS[GE], op1, op2)

    def cmovlt(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Reg
        self.emit_cmov(0x40 | COND_CODE_IDS[LT], op1, op2)

    def cmovle(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Reg
        self.emit_cmov(0x40 | COND_CODE_IDS[LE], op1, op2)

    def cmovgtu(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Reg
        self.emit_cmov(0x40 | COND_CODE_IDS[GTU], op1, op2)

    def cmovgeu(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Reg
        self.emit_cmov(0x40 | COND_CODE_IDS[GEU], op1, op2)

    def cmovltu(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Reg
        self.emit_cmov(0x40 | COND_CODE_IDS[LTU], op1, op2)

    def cmovleu(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Reg
        self.emit_cmov(0x40 | COND_CODE_IDS[LEU], op1, op2)

    def cmovp(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Reg
        self.emit_cmov(0x40 | COND_CODE_IDS[P], op1, op2)

    def emit_mov_scalar_mem(self, xmm, mem, opcode, prefix):
        assert type(xmm) is Xmm
        assert type(mem) is Mem
        assert type(opcode) is int
        assert type(prefix) is bytes
        rex = 0x40 | ((xmm.id >> 3) << 2)
        encoded = EncodedRegMemOp(mem, xmm.id)
        rex |= encoded.rex
        if rex != 0x40:
            rex_prefix = bytes((rex,))
        else:
            rex_prefix = b''
        instruction_start = self.section_offset()
        self.emit_bytes(
            prefix + rex_prefix + bytes((0x0F, opcode, encoded.mod_rm)) + encoded.suffix
        )
        if type(mem.addr) is Rel:
            disp_pos = instruction_start + len(prefix) + len(rex_prefix) + 3
            self.add_label_ref(mem.addr.label, disp_pos, RipDelta(len(self.text)))

    def emit_mov_scalar(self, op1, op2, size, prefix, name):
        assert type(size) is WordSize
        assert type(prefix) is bytes
        assert type(name) is str
        self.require_text_section(name)
        if type(op1) is Xmm and type(op2) is Xmm:
            dst = op1
            src = op2
            if dst.id < 0 or dst.id > 15 or src.id < 0 or src.id > 15:
                raise EmitterError('%s: invalid xmm register' % name)
            rex = 0x40 | ((dst.id >> 3) << 2) | (src.id >> 3)
            if rex != 0x40:
                rex_prefix = bytes((rex,))
            else:
                rex_prefix = b''
            mod_rm = 0xC0 | ((dst.id & 7) << 3) | (src.id & 7)
            self.emit_bytes(prefix + rex_prefix + bytes((0x0F, 0x10, mod_rm)))
        elif type(op1) is Xmm and type(op2) is Mem:
            dst = op1
            mem = op2
            if dst.id < 0 or dst.id > 15 or mem.size != size:
                raise EmitterError('%s: operands have incompatible sizes' % name)
            self.emit_mov_scalar_mem(dst, mem, 0x10, prefix)
        elif type(op1) is Mem and type(op2) is Xmm:
            mem = op1
            src = op2
            if src.id < 0 or src.id > 15 or mem.size != size:
                raise EmitterError('%s: operands have incompatible sizes' % name)
            self.emit_mov_scalar_mem(src, mem, 0x11, prefix)
        else:
            raise EmitterError('%s: invalid form' % name)

    def movss_sse(self, op1, op2):
        assert type(op1) in (Xmm, Mem)
        assert type(op2) in (Xmm, Mem)
        self.emit_mov_scalar(op1, op2, DWORD, b'\xf3', 'movss')

    def emit_mov_scalar_avx(self, op1, op2, size, pp, name):
        assert type(size) is WordSize
        assert type(pp) is VexPP
        assert type(name) is str
        self.require_text_section(name)
        if type(op1) is Xmm and type(op2) is Xmm:
            dst = op1
            src = op2
            self.emit_bytes(encode_vex(
                dst, dst, src, 0x10, VexMap.MAP_0F, pp, VexW.W0,
            ))
        elif type(op1) is Xmm and type(op2) is Mem:
            dst = op1
            mem = op2
            if dst.id < 0 or dst.id > 15 or mem.size != size:
                raise EmitterError('%s: operands have incompatible sizes' % name)
            instruction_start = self.section_offset()
            self.emit_bytes(encode_vex_rm(
                dst.id, mem, VexL.L128, 0x10,
                VexMap.MAP_0F, pp, VexW.W0,
            ))
            if type(mem.addr) is Rel:
                self.add_label_ref(mem.addr.label, instruction_start + 5, RipDelta(len(self.text)))
        elif type(op1) is Mem and type(op2) is Xmm:
            mem = op1
            src = op2
            if src.id < 0 or src.id > 15 or mem.size != size:
                raise EmitterError('%s: operands have incompatible sizes' % name)
            instruction_start = self.section_offset()
            self.emit_bytes(encode_vex_rm(mem, src.id, VexL.L128, 0x11,VexMap.MAP_0F, pp, VexW.W0))
            if type(mem.addr) is Rel:
                self.add_label_ref(mem.addr.label, instruction_start + 5, RipDelta(len(self.text)))
        else:
            raise EmitterError('%s: invalid form' % name)

    def movss_avx(self, op1, op2):
        assert type(op1) in (Xmm, Mem)
        assert type(op2) in (Xmm, Mem)
        require_avx()
        self.emit_mov_scalar_avx(op1, op2, DWORD, VexPP.PF3, 'movss')

    def movsd_sse(self, op1, op2):
        assert type(op1) in (Xmm, Mem)
        assert type(op2) in (Xmm, Mem)
        self.emit_mov_scalar(op1, op2, QWORD, b'\xf2', 'movsd')

    def movsd_avx(self, op1: Operand, op2: Operand):
        assert type(op1) in (Xmm, Mem)
        assert type(op2) in (Xmm, Mem)
        require_avx()
        self.emit_mov_scalar_avx(op1, op2, QWORD, VexPP.PF2, 'movsd')

    def emit_scalar_arith(self, op1, op2, opcode, prefix, name):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        assert type(opcode) is int
        assert type(prefix) is bytes
        assert type(name) is str
        if op1.id < 0 or op1.id > 15 or op2.id < 0 or op2.id > 15:
            raise EmitterError('%s: invalid xmm register' % name)
        rex = 0x40 | ((op1.id >> 3) << 2) | (op2.id >> 3)
        if rex != 0x40:
            rex_prefix = bytes((rex,))
        else:
            rex_prefix = b''
        mod_rm = 0xC0 | ((op1.id & 7) << 3) | (op2.id & 7)
        self.emit_bytes(prefix + rex_prefix + bytes((0x0F, opcode, mod_rm)))

    def addss_sse(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        self.require_text_section('addss')
        self.emit_scalar_arith(op1, op2, 0x58, b'\xf3', 'addss')

    def subss_sse(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        self.require_text_section('subss')
        self.emit_scalar_arith(op1, op2, 0x5C, b'\xf3', 'subss')

    def mulss_sse(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        self.require_text_section('mulss')
        self.emit_scalar_arith(op1, op2, 0x59, b'\xf3', 'mulss')

    def divss_sse(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        self.require_text_section('divss')
        self.emit_scalar_arith(op1, op2, 0x5E, b'\xf3', 'divss')

    def addsd_sse(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        self.require_text_section('addsd')
        self.emit_scalar_arith(op1, op2, 0x58, b'\xf2', 'addsd')

    def subsd_sse(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        self.require_text_section('subsd')
        self.emit_scalar_arith(op1, op2, 0x5C, b'\xf2', 'subsd')

    def mulsd_sse(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        self.require_text_section('mulsd')
        self.emit_scalar_arith(op1, op2, 0x59, b'\xf2', 'mulsd')

    def divsd_sse(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        self.require_text_section('divsd')
        self.emit_scalar_arith(op1, op2, 0x5E, b'\xf2', 'divsd')

    def emit_scalar_arith_avx(self, op1, op2, opcode, pp, name):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        assert type(opcode) is int
        assert type(pp) is VexPP
        assert type(name) is str
        self.require_text_section(name)
        self.emit_bytes(encode_vex(op1, op1, op2, opcode, VexMap.MAP_0F, pp, VexW.W0))

    def addss_avx(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        require_avx()
        self.emit_scalar_arith_avx(op1, op2, 0x58, VexPP.PF3, 'addss')

    def subss_avx(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        require_avx()
        self.emit_scalar_arith_avx(op1, op2, 0x5C, VexPP.PF3, 'subss')

    def mulss_avx(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        require_avx()
        self.emit_scalar_arith_avx(op1, op2, 0x59, VexPP.PF3, 'mulss')

    def divss_avx(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        require_avx()
        self.emit_scalar_arith_avx(op1, op2, 0x5E, VexPP.PF3, 'divss')

    def addsd_avx(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        require_avx()
        self.emit_scalar_arith_avx(op1, op2, 0x58, VexPP.PF2, 'addsd')

    def subsd_avx(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        require_avx()
        self.emit_scalar_arith_avx(op1, op2, 0x5C, VexPP.PF2, 'subsd')

    def mulsd_avx(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        require_avx()
        self.emit_scalar_arith_avx(op1, op2, 0x59, VexPP.PF2, 'mulsd')

    def divsd_avx(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        require_avx()
        self.emit_scalar_arith_avx(op1, op2, 0x5E, VexPP.PF2, 'divsd')

    def movss(self, op1, op2):
        assert type(op1) in (Xmm, Mem)
        assert type(op2) in (Xmm, Mem)
        if cpu_features.avx:
            self.movss_avx(op1, op2)
        else:
            self.movss_sse(op1, op2)

    def addss(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        if cpu_features.avx:
            self.addss_avx(op1, op2)
        else:
            self.addss_sse(op1, op2)

    def subss(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        if cpu_features.avx:
            self.subss_avx(op1, op2)
        else:
            self.subss_sse(op1, op2)

    def mulss(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        if cpu_features.avx:
            self.mulss_avx(op1, op2)
        else:
            self.mulss_sse(op1, op2)

    def divss(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        if cpu_features.avx:
            self.divss_avx(op1, op2)
        else:
            self.divss_sse(op1, op2)

    def movsd(self, op1, op2):
        assert type(op1) in (Xmm, Mem)
        assert type(op2) in (Xmm, Mem)
        if cpu_features.avx:
            self.movsd_avx(op1, op2)
        else:
            self.movsd_sse(op1, op2)

    def addsd(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        if cpu_features.avx:
            self.addsd_avx(op1, op2)
        else:
            self.addsd_sse(op1, op2)

    def subsd(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        if cpu_features.avx:
            self.subsd_avx(op1, op2)
        else:
            self.subsd_sse(op1, op2)

    def mulsd(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        if cpu_features.avx:
            self.mulsd_avx(op1, op2)
        else:
            self.mulsd_sse(op1, op2)

    def divsd(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        if cpu_features.avx:
            self.divsd_avx(op1, op2)
        else:
            self.divsd_sse(op1, op2)

    def emit_cvtsi2s(self, op1, op2, prefix, name):
        assert type(op1) is Xmm
        assert type(op2) is Reg
        assert type(prefix) is bytes
        assert type(name) is str
        self.require_text_section(name)
        if op1.id < 0 or op1.id > 15:
            raise EmitterError('%s: invalid xmm register' % name)
        if op2 == RIP or op2.size != QWORD:
            raise EmitterError('%s: source must be a qword register' % name)
        src = reg_id(op2)
        rex = 0x48 | ((op1.id >> 3) << 2) | (src >> 3)
        mod_rm = 0xC0 | ((op1.id & 7) << 3) | (src & 7)
        self.emit_bytes(prefix + bytes((rex, 0x0F, 0x2A, mod_rm)))

    def cvtsi2ss_sse(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Reg
        self.emit_cvtsi2s(op1, op2, b'\xf3', 'cvtsi2ss')

    def cvtsi2sd_sse(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Reg
        self.emit_cvtsi2s(op1, op2, b'\xf2', 'cvtsi2sd')

    def cvtsi2ss(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Reg
        self.cvtsi2ss_sse(op1, op2)

    def cvtsi2sd(self, op1, op2):
        assert type(op1) is Xmm
        assert type(op2) is Reg
        self.cvtsi2sd_sse(op1, op2)

    def emit_cvtts2si(self, op1, op2, prefix, name):
        assert type(op1) is Reg
        assert type(op2) is Xmm
        assert type(prefix) is bytes
        assert type(name) is str
        self.require_text_section(name)
        if op1 == RIP or op1.size != QWORD:
            raise EmitterError('%s: destination must be a qword register' % name)
        if op2.id < 0 or op2.id > 15:
            raise EmitterError('%s: invalid xmm register' % name)
        dst = reg_id(op1)
        rex = 0x48 | ((dst >> 3) << 2) | (op2.id >> 3)
        mod_rm = 0xC0 | ((dst & 7) << 3) | (op2.id & 7)
        self.emit_bytes(prefix + bytes((rex, 0x0F, 0x2C, mod_rm)))

    def cvttss2si_sse(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Xmm
        self.emit_cvtts2si(op1, op2, b'\xf3', 'cvttss2si')

    def cvttsd2si_sse(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Xmm
        self.emit_cvtts2si(op1, op2, b'\xf2', 'cvttsd2si')

    def cvttss2si(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Xmm
        self.cvttss2si_sse(op1, op2)

    def cvttsd2si(self, op1, op2):
        assert type(op1) is Reg
        assert type(op2) is Xmm
        self.cvttsd2si_sse(op1, op2)

    def emit_round_scalar(self, op1, op2, mode, opcode, name):
        assert type(op1) is Xmm
        assert type(op2) is Xmm
        assert type(mode) is int
        assert type(opcode) is int
        assert type(name) is str
        if op1.id < 0 or op1.id > 15 or op2.id < 0 or op2.id > 15:
            raise EmitterError('%s: invalid xmm register' % name)
        rex = 0x40 | ((op1.id >> 3) << 2) | (op2.id >> 3)
        if rex != 0x40:
            rex_prefix = bytes((rex,))
        else:
            rex_prefix = b''
        mod_rm = 0xC0 | ((op1.id & 7) << 3) | (op2.id & 7)
        self.emit_bytes(
            b'\x66' + rex_prefix + bytes((0x0F, 0x3A, opcode, mod_rm, mode))
        )

    def rounds_sse(self, op1: Xmm, op2: Xmm):
        self.require_text_section('rounds')
        self.emit_round_scalar(op1, op2, 0, 0x0A, 'rounds')

    def floors_sse(self, op1: Xmm, op2: Xmm):
        self.require_text_section('floors')
        self.emit_round_scalar(op1, op2, 1, 0x0A, 'floors')

    def ceils_sse(self, op1: Xmm, op2: Xmm):
        self.require_text_section('ceils')
        self.emit_round_scalar(op1, op2, 2, 0x0A, 'ceils')

    def truncs_sse(self, op1: Xmm, op2: Xmm):
        self.require_text_section('truncs')
        self.emit_round_scalar(op1, op2, 3, 0x0A, 'truncs')

    def roundd_sse(self, op1: Xmm, op2: Xmm):
        self.require_text_section('roundd')
        self.emit_round_scalar(op1, op2, 0, 0x0B, 'roundd')

    def floord_sse(self, op1: Xmm, op2: Xmm):
        self.require_text_section('floord')
        self.emit_round_scalar(op1, op2, 1, 0x0B, 'floord')

    def ceild_sse(self, op1: Xmm, op2: Xmm):
        self.require_text_section('ceild')
        self.emit_round_scalar(op1, op2, 2, 0x0B, 'ceild')

    def truncd_sse(self, op1: Xmm, op2: Xmm):
        self.require_text_section('truncd')
        self.emit_round_scalar(op1, op2, 3, 0x0B, 'truncd')

    def emit_round_scalar_avx(
        self,
        op1: Xmm,
        op2: Xmm,
        mode: int,
        opcode: int,
        name: str,
    ):
        self.require_text_section(name)
        self.emit_bytes(encode_vex(
            op1, op1, op2, opcode, VexMap.MAP_0F3A, VexPP.P66, VexW.W0, mode,
        ))

    def rounds_avx(self, op1: Xmm, op2: Xmm):
        require_avx()
        self.emit_round_scalar_avx(op1, op2, 0, 0x0A, 'rounds')

    def floors_avx(self, op1: Xmm, op2: Xmm):
        require_avx()
        self.emit_round_scalar_avx(op1, op2, 1, 0x0A, 'floors')

    def ceils_avx(self, op1: Xmm, op2: Xmm):
        require_avx()
        self.emit_round_scalar_avx(op1, op2, 2, 0x0A, 'ceils')

    def truncs_avx(self, op1: Xmm, op2: Xmm):
        require_avx()
        self.emit_round_scalar_avx(op1, op2, 3, 0x0A, 'truncs')

    def roundd_avx(self, op1: Xmm, op2: Xmm):
        require_avx()
        self.emit_round_scalar_avx(op1, op2, 0, 0x0B, 'roundd')

    def floord_avx(self, op1: Xmm, op2: Xmm):
        require_avx()
        self.emit_round_scalar_avx(op1, op2, 1, 0x0B, 'floord')

    def ceild_avx(self, op1: Xmm, op2: Xmm):
        require_avx()
        self.emit_round_scalar_avx(op1, op2, 2, 0x0B, 'ceild')

    def truncd_avx(self, op1: Xmm, op2: Xmm):
        require_avx()
        self.emit_round_scalar_avx(op1, op2, 3, 0x0B, 'truncd')

    def rounds(self, op1: Xmm, op2: Xmm):
        if cpu_features.avx:
            self.rounds_avx(op1, op2)
        else:
            self.rounds_sse(op1, op2)

    def floors(self, op1: Xmm, op2: Xmm):
        if cpu_features.avx:
            self.floors_avx(op1, op2)
        else:
            self.floors_sse(op1, op2)

    def ceils(self, op1: Xmm, op2: Xmm):
        if cpu_features.avx:
            self.ceils_avx(op1, op2)
        else:
            self.ceils_sse(op1, op2)

    def truncs(self, op1: Xmm, op2: Xmm):
        if cpu_features.avx:
            self.truncs_avx(op1, op2)
        else:
            self.truncs_sse(op1, op2)

    def roundd(self, op1: Xmm, op2: Xmm):
        if cpu_features.avx:
            self.roundd_avx(op1, op2)
        else:
            self.roundd_sse(op1, op2)

    def floord(self, op1: Xmm, op2: Xmm):
        if cpu_features.avx:
            self.floord_avx(op1, op2)
        else:
            self.floord_sse(op1, op2)

    def ceild(self, op1: Xmm, op2: Xmm):
        if cpu_features.avx:
            self.ceild_avx(op1, op2)
        else:
            self.ceild_sse(op1, op2)

    def truncd(self, op1: Xmm, op2: Xmm):
        if cpu_features.avx:
            self.truncd_avx(op1, op2)
        else:
            self.truncd_sse(op1, op2)

    def emit_binary_op(
        self,
        op1: Reg,
        op2: Reg | int,
        opcode: int,
        imm_id: int,
    ):
        if op1 == RIP or op1.size != QWORD:
            raise EmitterError('binary op: first operand must be a qword register')
        dst = reg_id(op1)
        if type(op2) is Reg:
            src = op2
            if src == RIP or src.size != QWORD:
                raise EmitterError('binary op: second operand must be a qword register')
            src_id = reg_id(src)
            rex = 0x48 | ((src_id >> 3) << 2) | (dst >> 3)
            mod_rm = 0xC0 | ((src_id & 7) << 3) | (dst & 7)
            self.emit_bytes(bytes((rex, opcode, mod_rm)))
        elif type(op2) is int:
            immediate = op2
            encoded = signed_bytes(immediate, 4)
            if encoded is None:
                raise EmitterError('binary op: immediate must fit in signed 32 bits')
            rex = 0x48 | (dst >> 3)
            mod_rm = 0xC0 | (imm_id << 3) | (dst & 7)
            self.emit_bytes(bytes((rex, 0x81, mod_rm)) + encoded)
        else:
            assert False

    def add(self, op1: Reg, op2: Reg | int):
        self.require_text_section('add')
        self.emit_binary_op(op1, op2, 0x01, 0)

    def sub(self, op1: Reg, op2: Reg | int):
        self.require_text_section('sub')
        self.emit_binary_op(op1, op2, 0x29, 5)

    def bitand(self, op1: Reg, op2: Reg | int):
        self.require_text_section('bitand')
        self.emit_binary_op(op1, op2, 0x21, 4)

    def bitor(self, op1: Reg, op2: Reg | int):
        self.require_text_section('bitor')
        self.emit_binary_op(op1, op2, 0x09, 1)

    def xor(self, op1: Reg, op2: Reg | int):
        self.require_text_section('xor')
        self.emit_binary_op(op1, op2, 0x31, 6)

    def bitnot(self, op: Reg):
        self.require_text_section('bitnot')
        self.xor(op, -1)

    def neg(self, op: Reg):
        self.require_text_section('neg')
        if op == RIP or op.size != QWORD:
            raise EmitterError('neg: operand must be a qword register')
        dst = reg_id(op)
        rex = 0x48 | (dst >> 3)
        mod_rm = 0xD8 | (dst & 7)
        self.emit_bytes(bytes((rex, 0xF7, mod_rm)))

    def imul(self, op1: Reg, op2: Reg | int):
        self.require_text_section('imul')
        if op1 == RIP or op1.size != QWORD:
            raise EmitterError('imul: first operand must be a qword register')
        dst = reg_id(op1)
        if type(op2) is Reg:
            src = op2
            if src == RIP or src.size != QWORD:
                raise EmitterError('imul: second operand must be a qword register')
            src_id = reg_id(src)
            rex = 0x48 | ((dst >> 3) << 2) | (src_id >> 3)
            mod_rm = 0xC0 | ((dst & 7) << 3) | (src_id & 7)
            self.emit_bytes(bytes((rex, 0x0F, 0xAF, mod_rm)))
        elif type(op2) is int:
            immediate = op2
            encoded = signed_bytes(immediate, 4)
            if encoded is None:
                raise EmitterError('imul: immediate must fit in signed 32 bits')
            rex = 0x48 | ((dst >> 3) << 2) | (dst >> 3)
            mod_rm = 0xC0 | ((dst & 7) << 3) | (dst & 7)
            self.emit_bytes(bytes((rex, 0x69, mod_rm)) + encoded)
        else:
            assert False

    def emit_xchg(self, op1: Reg, op2: Reg):
        dst = reg_id(op1)
        src = reg_id(op2)
        rex = 0x48 | ((src >> 3) << 2) | (dst >> 3)
        mod_rm = 0xC0 | ((src & 7) << 3) | (dst & 7)
        self.emit_bytes(bytes((rex, 0x87, mod_rm)))

    def emit_div(self, op1: Reg, op2: Reg, signed: bool):
        if op1 == RIP or op1.size != QWORD:
            raise EmitterError('div: first operand must be a qword register')
        if op2 == RIP or op2.size != QWORD:
            raise EmitterError('div: second operand must be a qword register')
        if op1 == op2:
            raise EmitterError('div: operands must be different registers')
        if (op1 == RAX and op2 == RDX) or (op1 == RDX and op2 == RAX):
            raise EmitterError('div: rax and rdx cannot be used together')

        divisor = op2
        if op2 == RAX:
            self.emit_xchg(op1, RAX)
            divisor = op1
        elif op2 == RDX:
            self.mov(RAX, op1)
            self.emit_xchg(op1, RDX)
            divisor = op1
        elif op1 != RAX:
            self.mov(RAX, op1)

        if signed:
            self.emit_bytes(b'\x48\x99')
            imm_id = 7
        else:
            self.xor(RDX, RDX)
            imm_id = 6

        divisor_id = reg_id(divisor)
        rex = 0x48 | (divisor_id >> 3)
        mod_rm = 0xC0 | (imm_id << 3) | (divisor_id & 7)
        self.emit_bytes(bytes((rex, 0xF7, mod_rm)))

        if op1 == RDX:
            self.mov(op2, RDX)
            self.mov(RDX, RAX)
        elif op2 == RAX:
            self.mov(op1, RAX)
            self.mov(RAX, RDX)
        else:
            if op1 != RAX:
                self.mov(op1, RAX)
            if op2 != RDX:
                self.mov(op2, RDX)

    def idiv(self, op1, op2): # returns None
        assert type(op1) is Reg
        assert type(op2) is Reg
        self.require_text_section('idiv')
        self.emit_div(op1, op2, True)

    def div(self, op1: Reg, op2: Reg):
        self.require_text_section('div')
        self.emit_div(op1, op2, False)

    def emit_shift(self, op1: Reg, op2: Reg | int, imm_id: int):
        if op1 == RIP or op1.size != QWORD:
            raise EmitterError('shift: first operand must be a qword register')
        dst = reg_id(op1)
        rex = 0x48 | (dst >> 3)
        mod_rm = 0xC0 | (imm_id << 3) | (dst & 7)
        if type(op2) is Reg:
            src = op2
            if src == RIP or src.size != QWORD:
                raise EmitterError('shift: second operand must be a qword register')
            self.mov(RCX, src)
            self.emit_bytes(bytes((rex, 0xD3, mod_rm)))
        elif type(op2) is int:
            immediate = op2
            if immediate < 0 or immediate >= (1 << 8):
                raise EmitterError('shift: immediate must fit in unsigned 8 bits')
            self.emit_bytes(bytes((rex, 0xC1, mod_rm, immediate)))
        else:
            assert False

    def shl(self, op1: Reg, op2: Reg | int):
        self.require_text_section('shl')
        self.emit_shift(op1, op2, 4)

    def sar(self, op1: Reg, op2: Reg | int):
        self.require_text_section('sar')
        self.emit_shift(op1, op2, 7)

    def shr(self, op1: Reg, op2: Reg | int):
        self.require_text_section('shr')
        self.emit_shift(op1, op2, 5)

    def ror(self, op1: Reg, op2: Reg | int):
        self.require_text_section('ror')
        self.emit_shift(op1, op2, 1)

    def rol(self, op1: Reg, op2: Reg | int):
        self.require_text_section('rol')
        self.emit_shift(op1, op2, 0)

    def push(self, r: Reg):
        self.require_text_section('push')
        if r == RIP or r.size != QWORD:
            raise EmitterError('push: operand must be a qword register')
        if r == RSP:
            self.mov(qword_ptr(RSP - 8), RSP)
            self.sub(RSP, 8)
            return
        self.sub(RSP, 8)
        self.mov(qword_ptr(RSP), r)

    def pop(self, r: Reg):
        self.require_text_section('pop')
        if r == RIP or r.size != QWORD:
            raise EmitterError('pop: operand must be a qword register')
        if r == RSP:
            self.add(RSP, 8)
            self.mov(RSP, qword_ptr(RSP - 8))
            return
        self.mov(r, qword_ptr(RSP))
        self.add(RSP, 8)

    def begin(self):
        self.push(RBP)
        self.mov(RBP, RSP)

    def end(self):
        self.mov(RSP, RBP)
        self.pop(RBP)
        self.ret()

    def call(self, target: str | Reg):
        self.require_text_section('call')
        if type(target) is str:
            label = target
            instruction_start = self.section_offset()
            self.emit_bytes(b'\xe8\x00\x00\x00\x00')
            self.add_label_ref(label, instruction_start + 1, RipDelta(len(self.text)))
        elif type(target) is Reg:
            reg = target
            if reg == RIP or reg.size != QWORD:
                raise EmitterError('call: target must be a qword register')
            target_id = reg_id(reg)
            if target_id >= 8:
                rex_prefix = bytes((0x40 | (target_id >> 3),))
            else:
                rex_prefix = b''
            mod_rm = 0xD0 | (target_id & 7)
            self.emit_bytes(rex_prefix + bytes((0xFF, mod_rm)))
        else:
            assert False

    def jmp(self, target: str | Reg):
        self.require_text_section('jmp')
        if type(target) is str:
            label = target
            instruction_start = self.section_offset()
            self.emit_bytes(b'\xe9\x00\x00\x00\x00')
            self.add_label_ref(label, instruction_start + 1, RipDelta(len(self.text)))
        elif type(target) is Reg:
            reg = target
            if reg == RIP or reg.size != QWORD:
                raise EmitterError('jmp: target must be a qword register')
            target_id = reg_id(reg)
            if target_id >= 8:
                rex_prefix = bytes((0x40 | (target_id >> 3),))
            else:
                rex_prefix = b''
            mod_rm = 0xE0 | (target_id & 7)
            self.emit_bytes(rex_prefix + bytes((0xFF, mod_rm)))
        else:
            assert False

    def cmp(self, op1: Reg, op2: Reg | int):
        self.require_text_section('cmp')
        if op1 == RIP or op1.size != QWORD:
            raise EmitterError('cmp: first operand must be a qword register')

        dst = reg_id(op1)
        if type(op2) is Reg:
            src = op2
            if src == RIP or src.size != QWORD:
                raise EmitterError('cmp: second operand must be a qword register')
            src_id = reg_id(src)
            rex = 0x48 | ((src_id >> 3) << 2) | (dst >> 3)
            mod_rm = 0xC0 | ((src_id & 7) << 3) | (dst & 7)
            self.emit_bytes(bytes((rex, 0x39, mod_rm)))
        elif type(op2) is int:
            immediate = op2
            encoded = signed_bytes(immediate, 4)
            if encoded is None:
                raise EmitterError('cmp: immediate must fit in signed 32 bits')
            rex = 0x48 | (dst >> 3)
            mod_rm = 0xF8 | (dst & 7)
            self.emit_bytes(bytes((rex, 0x81, mod_rm)) + encoded)
        else:
            assert False

    def emit_ucomis(self, x1: Xmm, x2: Xmm, prefix: bytes, name: str):
        self.require_text_section(name)
        if x1.id < 0 or x1.id > 15 or x2.id < 0 or x2.id > 15:
            raise EmitterError('%s: invalid xmm register' % name)
        rex = 0x40 | ((x1.id >> 3) << 2) | (x2.id >> 3)
        if rex != 0x40:
            rex_prefix = bytes((rex,))
        else:
            rex_prefix = b''
        mod_rm = 0xC0 | ((x1.id & 7) << 3) | (x2.id & 7)
        self.emit_bytes(prefix + rex_prefix + bytes((0x0F, 0x2E, mod_rm)))

    def ucomiss_sse(self, x1: Xmm, x2: Xmm):
        self.emit_ucomis(x1, x2, b'', 'ucomiss')

    def ucomisd_sse(self, x1: Xmm, x2: Xmm):
        self.emit_ucomis(x1, x2, b'\x66', 'ucomisd')

    def emit_ucomis_avx(self, x1: Xmm, x2: Xmm, pp: VexPP, name: str):
        self.require_text_section(name)
        self.emit_bytes(encode_vex(x1, None, x2, 0x2E, VexMap.MAP_0F, pp, VexW.W0))

    def ucomiss_avx(self, x1: Xmm, x2: Xmm):
        require_avx()
        self.emit_ucomis_avx(x1, x2, VexPP.NONE, 'ucomiss')

    def ucomisd_avx(self, x1: Xmm, x2: Xmm):
        require_avx()
        self.emit_ucomis_avx(x1, x2, VexPP.P66, 'ucomisd')

    def ucomiss(self, x1: Xmm, x2: Xmm):
        if cpu_features.avx:
            self.ucomiss_avx(x1, x2)
        else:
            self.ucomiss_sse(x1, x2)

    def ucomisd(self, x1: Xmm, x2: Xmm):
        if cpu_features.avx:
            self.ucomisd_avx(x1, x2)
        else:
            self.ucomisd_sse(x1, x2)

    def jcc(self, cond: CondCode, label: str):
        self.require_text_section('jcc')
        instruction_start = self.section_offset()
        self.emit_bytes(bytes((0x0F, 0x80 | COND_CODE_IDS[cond])) + b'\x00\x00\x00\x00')
        self.add_label_ref(label, instruction_start + 2, RipDelta(len(self.text)))

    def ja(self, label: str):
        self.jcc(GTU, label)

    def jae(self, label: str):
        self.jcc(GEU, label)

    def jb(self, label: str):
        self.jcc(LTU, label)

    def jbe(self, label: str):
        self.jcc(LEU, label)

    def jc(self, label: str):
        self.jcc(LTU, label)

    def jnc(self, label: str):
        self.jcc(GEU, label)

    def je(self, label: str):
        self.jcc(EQ, label)

    def jne(self, label: str):
        self.jcc(NE, label)

    def jz(self, label: str):
        self.jcc(EQ, label)

    def jnz(self, label: str):
        self.jcc(NE, label)

    def jg(self, label: str):
        self.jcc(GT, label)

    def jge(self, label: str):
        self.jcc(GE, label)

    def jl(self, label: str):
        self.jcc(LT, label)

    def jle(self, label: str):
        self.jcc(LE, label)

    def jna(self, label: str):
        self.jcc(LEU, label)

    def jnae(self, label: str):
        self.jcc(LTU, label)

    def jnb(self, label: str):
        self.jcc(GEU, label)

    def jnbe(self, label: str):
        self.jcc(GTU, label)

    def jng(self, label: str):
        self.jcc(LE, label)

    def jnge(self, label: str):
        self.jcc(LT, label)

    def jnl(self, label: str):
        self.jcc(GE, label)

    def jnle(self, label: str):
        self.jcc(GT, label)

    def jo(self, label: str):
        self.jcc(O, label)

    def jno(self, label: str):
        self.jcc(NO, label)

    def js(self, label: str):
        self.jcc(S, label)

    def jns(self, label: str):
        self.jcc(NS, label)

    def jp(self, label: str):
        self.jcc(P, label)

    def jpe(self, label: str):
        self.jcc(P, label)

    def jnp(self, label: str):
        self.jcc(NP, label)

    def jpo(self, label: str):
        self.jcc(NP, label)

    def jeq(self, label: str):
        self.jcc(EQ, label)

    def jgt(self, label: str):
        self.jcc(GT, label)

    def jlt(self, label: str):
        self.jcc(LT, label)

    def jgtu(self, label: str):
        self.jcc(GTU, label)

    def jgeu(self, label: str):
        self.jcc(GEU, label)

    def jltu(self, label: str):
        self.jcc(LTU, label)

    def jleu(self, label: str):
        self.jcc(LEU, label)

    def setcc(self, cond: CondCode, r: Reg):
        self.require_text_section('setcc')
        if r.name == RegName.RIP or r.size != BYTE:
            raise EmitterError('setcc: destination must be a byte register')
        dst = reg_id(r)
        rex = 0x40 | (dst >> 3)
        if rex != 0x40 or dst >= 4:
            rex_prefix = bytes((rex,))
        else:
            rex_prefix = b''
        mod_rm = 0xC0 | (dst & 7)
        self.emit_bytes(rex_prefix + bytes((0x0F, 0x90 | COND_CODE_IDS[cond], mod_rm)))

    def branch(self, cond: CondCode, op1: Reg, op2: Reg | int, label: str):
        self.cmp(op1, op2)
        self.jcc(cond, label)

    def branchs(self, cond: CondCode, op1: Xmm, op2: Xmm, label: str):
        cond = xmm_cond_code(cond)
        self.ucomiss(op1, op2)
        self.jcc(cond, label)

    def branchd(self, cond: CondCode, op1: Xmm, op2: Xmm, label: str):
        cond = xmm_cond_code(cond)
        self.ucomisd(op1, op2)
        self.jcc(cond, label)

    def beq(self, op1: Reg, op2: Reg | int, label: str):
        self.branch(EQ, op1, op2, label)

    def bne(self, op1: Reg, op2: Reg | int, label: str):
        self.branch(NE, op1, op2, label)

    def bgt(self, op1: Reg, op2: Reg | int, label: str):
        self.branch(GT, op1, op2, label)

    def blt(self, op1: Reg, op2: Reg | int, label: str):
        self.branch(LT, op1, op2, label)

    def bge(self, op1: Reg, op2: Reg | int, label: str):
        self.branch(GE, op1, op2, label)

    def ble(self, op1: Reg, op2: Reg | int, label: str):
        self.branch(LE, op1, op2, label)

    def beqs(self, op1: Xmm, op2: Xmm, label: str):
        self.branchs(EQ, op1, op2, label)

    def beqd(self, op1: Xmm, op2: Xmm, label: str):
        self.branchd(EQ, op1, op2, label)

    def bnes(self, op1: Xmm, op2: Xmm, label: str):
        self.branchs(NE, op1, op2, label)

    def bned(self, op1: Xmm, op2: Xmm, label: str):
        self.branchd(NE, op1, op2, label)

    def bgts(self, op1: Xmm, op2: Xmm, label: str):
        self.branchs(GT, op1, op2, label)

    def bgtd(self, op1: Xmm, op2: Xmm, label: str):
        self.branchd(GT, op1, op2, label)

    def blts(self, op1: Xmm, op2: Xmm, label: str):
        self.branchs(LT, op1, op2, label)

    def bltd(self, op1: Xmm, op2: Xmm, label: str):
        self.branchd(LT, op1, op2, label)

    def bges(self, op1: Xmm, op2: Xmm, label: str):
        self.branchs(GE, op1, op2, label)

    def bged(self, op1: Xmm, op2: Xmm, label: str):
        self.branchd(GE, op1, op2, label)

    def bles(self, op1: Xmm, op2: Xmm, label: str):
        self.branchs(LE, op1, op2, label)

    def bled(self, op1: Xmm, op2: Xmm, label: str):
        self.branchd(LE, op1, op2, label)

    def bgtu(self, op1: Reg, op2: Reg | int, label: str):
        self.branch(GTU, op1, op2, label)

    def bltu(self, op1: Reg, op2: Reg | int, label: str):
        self.branch(LTU, op1, op2, label)

    def bgeu(self, op1: Reg, op2: Reg | int, label: str):
        self.branch(GEU, op1, op2, label)

    def bleu(self, op1: Reg, op2: Reg | int, label: str):
        self.branch(LEU, op1, op2, label)

    def cset(self, cond: CondCode, op1: Reg, op2: Reg | int, r: Reg):
        if r.name == RegName.RIP or r.size != BYTE:
            raise EmitterError('cset: destination must be a byte register')
        self.cmp(op1, op2)
        self.setcc(cond, r)

    def csets(self, cond: CondCode, op1: Xmm, op2: Xmm, r: Reg):
        if r.name == RegName.RIP or r.size != BYTE:
            raise EmitterError('csets: destination must be a byte register')
        cond = xmm_cond_code(cond)
        self.ucomiss(op1, op2)
        self.setcc(cond, r)

    def csetd(self, cond: CondCode, op1: Xmm, op2: Xmm, r: Reg):
        if r.name == RegName.RIP or r.size != BYTE:
            raise EmitterError('csetd: destination must be a byte register')
        cond = xmm_cond_code(cond)
        self.ucomisd(op1, op2)
        self.setcc(cond, r)

    def seteq(self, op1: Reg, op2: Reg | int, r: Reg):
        self.cset(EQ, op1, op2, r)

    def setne(self, op1: Reg, op2: Reg | int, r: Reg):
        self.cset(NE, op1, op2, r)

    def setgt(self, op1: Reg, op2: Reg | int, r: Reg):
        self.cset(GT, op1, op2, r)

    def setlt(self, op1: Reg, op2: Reg | int, r: Reg):
        self.cset(LT, op1, op2, r)

    def setge(self, op1: Reg, op2: Reg | int, r: Reg):
        self.cset(GE, op1, op2, r)

    def setle(self, op1: Reg, op2: Reg | int, r: Reg):
        self.cset(LE, op1, op2, r)

    def seteqs(self, op1: Xmm, op2: Xmm, r: Reg):
        self.csets(EQ, op1, op2, r)

    def seteqd(self, op1: Xmm, op2: Xmm, r: Reg):
        self.csetd(EQ, op1, op2, r)

    def setnes(self, op1: Xmm, op2: Xmm, r: Reg):
        self.csets(NE, op1, op2, r)

    def setned(self, op1: Xmm, op2: Xmm, r: Reg):
        self.csetd(NE, op1, op2, r)

    def setgts(self, op1: Xmm, op2: Xmm, r: Reg):
        self.csets(GT, op1, op2, r)

    def setgtd(self, op1: Xmm, op2: Xmm, r: Reg):
        self.csetd(GT, op1, op2, r)

    def setlts(self, op1: Xmm, op2: Xmm, r: Reg):
        self.csets(LT, op1, op2, r)

    def setltd(self, op1: Xmm, op2: Xmm, r: Reg):
        self.csetd(LT, op1, op2, r)

    def setges(self, op1: Xmm, op2: Xmm, r: Reg):
        self.csets(GE, op1, op2, r)

    def setged(self, op1: Xmm, op2: Xmm, r: Reg):
        self.csetd(GE, op1, op2, r)

    def setles(self, op1: Xmm, op2: Xmm, r: Reg):
        self.csets(LE, op1, op2, r)

    def setled(self, op1: Xmm, op2: Xmm, r: Reg):
        self.csetd(LE, op1, op2, r)

    def setgtu(self, op1: Reg, op2: Reg | int, r: Reg):
        self.cset(GTU, op1, op2, r)

    def setltu(self, op1: Reg, op2: Reg | int, r: Reg):
        self.cset(LTU, op1, op2, r)

    def setgeu(self, op1: Reg, op2: Reg | int, r: Reg):
        self.cset(GEU, op1, op2, r)

    def setleu(self, op1: Reg, op2: Reg | int, r: Reg):
        self.cset(LEU, op1, op2, r)

    def ret(self):
        self.require_text_section('ret')
        self.emit_bytes(b'\xc3')

    def cpuid(self):
        self.require_text_section('cpuid')
        self.emit_bytes(b'\x0f\xa2')

    def emit_vmov(self, op1, op2, load_opcode, store_opcode, name):
        assert type(op1) in (Xmm, Ymm, Mem)
        assert type(op2) in (Xmm, Ymm, Mem)
        assert type(load_opcode) is int
        assert type(store_opcode) is int
        assert type(name) is str
        self.require_text_section(name)
        if type(op1) is Xmm and type(op2) is Xmm:
            dst = op1
            src = op2
            self.emit_bytes(encode_vex(dst, None, src, load_opcode, VexMap.MAP_0F, VexPP.NONE, VexW.W0))
            return
        elif type(op1) is Ymm and type(op2) is Ymm:
            dst = op1
            src = op2
            self.emit_bytes(encode_vex(dst, None, src, load_opcode, VexMap.MAP_0F, VexPP.NONE, VexW.W0))
            return
        elif type(op1) is Xmm and type(op2) is Mem:
            reg = op1
            mem = op2
            l = VexL.L128
            size = M128
            opcode = load_opcode
        elif type(op1) is Ymm and type(op2) is Mem:
            reg = op1
            mem = op2
            l = VexL.L256
            size = M256
            opcode = load_opcode
        elif type(op1) is Mem and type(op2) is Xmm:
            mem = op1
            reg = op2
            l = VexL.L128
            size = M128
            opcode = store_opcode
        elif type(op1) is Mem and type(op2) is Ymm:
            mem = op1
            reg = op2
            l = VexL.L256
            size = M256
            opcode = store_opcode
        else:
            raise EmitterError('%s: invalid form' % name)
        if mem.size != size:
            raise EmitterError('%s: operands have incompatible sizes' % name)
        instruction_start = self.section_offset()
        self.emit_bytes(encode_vex_rm(reg.id, mem, l, opcode, VexMap.MAP_0F, VexPP.NONE, VexW.W0))
        if type(mem.addr) is Rel:
            self.add_label_ref(mem.addr.label, instruction_start + 5, RipDelta(len(self.text)))

    def vmovaps(self, op1, op2):
        require_avx()
        self.emit_vmov(op1, op2, 0x28, 0x29, 'vmovaps')

    def vmovups(self, op1, op2):
        require_avx()
        self.emit_vmov(op1, op2, 0x10, 0x11, 'vmovups')

    def emit_v_arith_ps(self, dst, src1, src2, opcode, name):
        assert type(dst) in (Xmm, Ymm)
        assert type(src1) == type(dst)
        assert type(src2) == type(dst)
        assert type(opcode) is int
        assert type(name) is str
        self.require_text_section(name)
        if type(dst) is Xmm and type(src1) is Xmm and type(src2) is Xmm:
            self.emit_bytes(encode_vex(dst, src1, src2, opcode, VexMap.MAP_0F, VexPP.NONE, VexW.W0))
        elif type(dst) is Ymm and type(src1) is Ymm and type(src2) is Ymm:
            self.emit_bytes(encode_vex(dst, src1, src2, opcode, VexMap.MAP_0F, VexPP.NONE, VexW.W0))
        else:
            raise EmitterError('%s: invalid form' % name)

    def vaddps(self, dst: T, src1: T, src2: T):
        require_avx()
        self.emit_v_arith_ps(dst, src1, src2, 0x58, 'vaddps')

    def vsubps(self, dst: T, src1: T, src2: T):
        require_avx()
        self.emit_v_arith_ps(dst, src1, src2, 0x5C, 'vsubps')

    def vmulps(self, dst: T, src1: T, src2: T):
        require_avx()
        self.emit_v_arith_ps(dst, src1, src2, 0x59, 'vmulps')

    def vdivps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.emit_v_arith_ps(dst, src1, src2, 0x5E, 'vdivps')

    def vaddsubps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        self.require_text_section('vaddsubps')
        require_avx()
        self.emit_bytes(encode_vex(dst, src1, src2, 0xD0, VexMap.MAP_0F, VexPP.PF2, VexW.W0))

    def vsqrtps(self, dst, src):
        assert type(dst) in (Xmm, Ymm)
        assert type(src) == type(dst)
        self.require_text_section('vsqrtps')
        require_avx()
        if type(dst) is Xmm and type(src) is Xmm:
            self.emit_bytes(encode_vex(dst, None, src, 0x51, VexMap.MAP_0F, VexPP.NONE, VexW.W0))
        elif type(dst) is Ymm and type(src) is Ymm:
            self.emit_bytes(encode_vex(dst, None, src, 0x51, VexMap.MAP_0F, VexPP.NONE, VexW.W0))
        else:
            raise EmitterError('vsqrtps: invalid form')

    def vmaxps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.emit_v_arith_ps(dst, src1, src2, 0x5F, 'vmaxps')

    def vminps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.emit_v_arith_ps(dst, src1, src2, 0x5D, 'vminps')

    def vandps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.emit_v_arith_ps(dst, src1, src2, 0x54, 'vandps')

    def vandnps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        self.require_text_section('vandnps')
        require_avx()
        self.emit_bytes(encode_vex(dst, src1, src2, 0x55, VexMap.MAP_0F, VexPP.NONE, VexW.W0))

    def vorps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.emit_v_arith_ps(dst, src1, src2, 0x56, 'vorps')

    def vxorps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        self.require_text_section('vxorps')
        require_avx()
        self.emit_bytes(encode_vex(dst, src1, src2, 0x57, VexMap.MAP_0F, VexPP.NONE, VexW.W0))

    def emit_vroundps[T: (Xmm, Ymm)](self, dst: T, src: T, mode: int):
        self.require_text_section('vroundps')
        if mode < 0 or mode > 0x0F:
            raise EmitterError('vroundps: mode must fit in 4 bits')
        if type(dst) is Xmm and type(src) is Xmm:
            self.emit_bytes(encode_vex(
                dst, None, src, 0x08, VexMap.MAP_0F3A, VexPP.P66, VexW.W0, mode,
            ))
        elif type(dst) is Ymm and type(src) is Ymm:
            self.emit_bytes(encode_vex(
                dst, None, src, 0x08, VexMap.MAP_0F3A, VexPP.P66, VexW.W0, mode,
            ))
        else:
            raise EmitterError('vroundps: invalid form')

    def vroundps[T: (Xmm, Ymm)](self, dst: T, src: T):
        require_avx()
        self.emit_vroundps(dst, src, 0)

    def vfloorps[T: (Xmm, Ymm)](self, dst: T, src: T):
        require_avx()
        self.emit_vroundps(dst, src, 1)

    def vceilps[T: (Xmm, Ymm)](self, dst: T, src: T):
        require_avx()
        self.emit_vroundps(dst, src, 2)

    def vtruncps[T: (Xmm, Ymm)](self, dst: T, src: T):
        require_avx()
        self.emit_vroundps(dst, src, 3)

    def vcmpps[T: (Xmm, Ymm)](
        self,
        dst: T,
        src1: T,
        src2: T,
        predicate: int,
    ):
        self.require_text_section('vcmpps')
        require_avx()
        if predicate < 0 or predicate > 7:
            raise EmitterError('vcmpps: predicate must be between 0 and 7')
        if type(dst) is Xmm and type(src1) is Xmm and type(src2) is Xmm:
            self.emit_bytes(encode_vex(
                dst, src1, src2, 0xC2, VexMap.MAP_0F, VexPP.NONE, VexW.W0, predicate,
            ))
        elif type(dst) is Ymm and type(src1) is Ymm and type(src2) is Ymm:
            self.emit_bytes(encode_vex(
                dst, src1, src2, 0xC2, VexMap.MAP_0F, VexPP.NONE, VexW.W0, predicate,
            ))
        else:
            raise EmitterError('vcmpps: invalid form')

    def veqps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.vcmpps(dst, src1, src2, 0)

    def vltps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.vcmpps(dst, src1, src2, 1)

    def vleps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.vcmpps(dst, src1, src2, 2)

    def vunordps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.vcmpps(dst, src1, src2, 3)

    def vneps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.vcmpps(dst, src1, src2, 4)

    def vnltps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.vcmpps(dst, src1, src2, 5)

    def vnleps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.vcmpps(dst, src1, src2, 6)

    def vordps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.vcmpps(dst, src1, src2, 7)

    def vgtps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.vcmpps(dst, src2, src1, 1)

    def vgeps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        require_avx()
        self.vcmpps(dst, src2, src1, 2)
    
    def vhaddps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        self.require_text_section('vhaddps')
        require_avx()
        self.emit_bytes(encode_vex(dst, src1, src2, 0x7C, VexMap.MAP_0F, VexPP.PF2, VexW.W0))

    def vhsubps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T):
        self.require_text_section('vhsubps')
        require_avx()
        self.emit_bytes(encode_vex(dst, src1, src2, 0x7D, VexMap.MAP_0F, VexPP.PF2, VexW.W0))

    def vdpps(self, dst, src1, src2, input_mask, output_mask):
        assert type(dst)         in (Xmm, Ymm)
        assert type(src1)        in (Xmm, Ymm)
        assert type(src2)        in (Xmm, Ymm)
        assert type(input_mask)  is list
        assert type(output_mask) is list
        self.require_text_section('vdpps')
        require_avx()
        if len(input_mask) != 4:
            raise EmitterError('vdpps: input mask must contain four booleans')
        if len(output_mask) != 4:
            raise EmitterError('vdpps: output mask must contain four booleans')
        imm8 = 0
        for i, value in enumerate(input_mask):
            assert type(value) is bool
            imm8 += int(value) << (i + 4)
        output_imm = 0
        for i, value in enumerate(output_mask):
            assert type(value) is bool
            output_imm += int(value) << i
        imm8 |= output_imm
        self.emit_bytes(encode_vex(dst, src1, src2, 0x40, VexMap.MAP_0F3A, VexPP.P66, VexW.W0, imm8))

    def vrcpps[T: (Xmm, Ymm)](self, dst: T, src: T):
        self.require_text_section('vrcpps')
        require_avx()
        self.emit_bytes(encode_vex(dst, None, src, 0x53, VexMap.MAP_0F, VexPP.NONE, VexW.W0))

    def vrsqrtps[T: (Xmm, Ymm)](self, dst: T, src: T):
        self.require_text_section('vrsqrtps')
        require_avx()
        self.emit_bytes(encode_vex(dst, None, src, 0x52, VexMap.MAP_0F, VexPP.NONE, VexW.W0))

    def vzeroupper(self):
        self.require_text_section('vzeroupper')
        require_avx()
        self.emit_bytes(b'\xc5\xf8\x77')

    def vptest(self, op1, op2):
        assert type(op1) in (Xmm, Ymm)
        assert type(op1) == type(op2)
        self.require_text_section('vptest')
        require_avx()
        self.emit_bytes(encode_vex(op1, None, op2, 0x17, VexMap.MAP_0F38, VexPP.P66, VexW.W0))

    def vblendps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T, mask: list[int]):
        self.require_text_section('vblendps')
        require_avx()
        if type(dst) is Xmm:
            size = 4
        else:
            size = 8
        if len(mask) != size or any(value not in (1, 2) for value in mask):
            raise EmitterError('vblendps: mask must contain %d integers, each 1 or 2' % size)
        imm8 = sum((value - 1) << i for i, value in enumerate(mask))
        self.emit_bytes(encode_vex(dst, src1, src2, 0x0C, VexMap.MAP_0F3A, VexPP.P66, VexW.W0, imm8))

    def vshufps[T: (Xmm, Ymm)](self, dst: T, src1: T, src2: T, imm: list[int]):
        self.require_text_section('vshufps')
        require_avx()
        if len(imm) != 4 or any(value < 0 or value > 3 for value in imm):
            raise EmitterError('vshufps: imm must contain four integers between 0 and 3')
        imm8 = sum(value << (2 * i) for i, value in enumerate(imm))
        self.emit_bytes(encode_vex(dst, src1, src2, 0xC6, VexMap.MAP_0F, VexPP.NONE, VexW.W0, imm8))

    def vpermilps(self, dst, src1, src2):
        assert type(dst)  in (Xmm, Ymm)
        assert type(src1) in (Xmm, Ymm)
        assert type(src2) in (Xmm, Ymm, list)
        if type(src2) is not list:
            assert type(dst) == type(src1) == type(src2)
        else:
            assert type(dst) == type(src1)
        self.require_text_section('vpermilps')
        require_avx()
        if type(src2) is list:
            for value in src2: assert type(value) is int
            if len(src2) != 4 or any(value < 0 or value > 3 for value in src2):
                raise EmitterError('vpermilps: imm must contain four integers between 0 and 3')
            imm8 = sum(value << (2 * i) for i, value in enumerate(src2))
            self.emit_bytes(encode_vex(dst, None, src1, 0x04, VexMap.MAP_0F3A, VexPP.P66, VexW.W0, imm8))
        else:
            self.emit_bytes(encode_vex(dst, src1, src2, 0x0C, VexMap.MAP_0F38, VexPP.P66, VexW.W0))

    # zero_mask: 1 zeros the corresponding element; 0 keeps its value after insertion.
    def vinsertps(self, dst: Xmm, src1: Xmm, src2: Xmm, count_dst: int, count_src: int, zero_mask: list[int]):
        self.require_text_section('vinsertps')
        require_avx()
        if count_dst < 0 or count_dst > 3:
            raise EmitterError('vinsertps: count_dst must be between 0 and 3')
        if count_src < 0 or count_src > 3:
            raise EmitterError('vinsertps: count_src must be between 0 and 3')
        if len(zero_mask) != 4 or any(value not in (0, 1) for value in zero_mask):
            raise EmitterError('vinsertps: zero_mask must contain four integers, each 0 or 1')
        imm8 = (count_src << 6) | (count_dst << 4)
        imm8 |= sum(value << i for i, value in enumerate(zero_mask))
        self.emit_bytes(encode_vex(dst, src1, src2, 0x21, VexMap.MAP_0F3A, VexPP.P66, VexW.W0, imm8))

    def vbroadcastss(self, dst: Xmm | Ymm, src: Xmm | Mem):
        self.require_text_section('vbroadcastss')
        require_avx()
        if type(src) is Xmm:
            require_avx2()
            if type(dst) is Ymm:
                # The source field only encodes the register number; the opcode fixes its width to XMM.
                self.emit_bytes(encode_vex(dst, None, Ymm(src.id), 0x18, VexMap.MAP_0F38, VexPP.P66, VexW.W0))
            else:
                self.emit_bytes(encode_vex(dst, None, src, 0x18, VexMap.MAP_0F38, VexPP.P66, VexW.W0))
            return
        if src.size != DWORD:
            raise EmitterError('vbroadcastss: source must be dword memory')
        l = VexL.L128
        if type(dst) is Ymm:
            l = VexL.L256
        instruction_start = self.section_offset()
        self.emit_bytes(encode_vex_rm(dst.id, src, l, 0x18, VexMap.MAP_0F38, VexPP.P66, VexW.W0))
        if type(src.addr) is Rel:
            self.add_label_ref(src.addr.label, instruction_start + 5, RipDelta(len(self.text)))

def init_cpu_features():
    global cpu_features
    e = Emitter()

    e.label('max_basic_leaf')
    e.push(RBX)
    e.mov(RAX, 0)
    e.mov(RCX, 0)
    e.cpuid()
    e.pop(RBX)
    e.ret()

    e.label('leaf1_ecx')
    e.push(RBX)
    e.mov(RAX, 1)
    e.mov(RCX, 0)
    e.cpuid()
    e.mov(RAX, RCX)
    e.pop(RBX)
    e.ret()

    e.label('leaf7_ebx')
    e.push(RBX)
    e.mov(RAX, 7)
    e.mov(RCX, 0)
    e.cpuid()
    e.mov(RAX, RBX)
    e.pop(RBX)
    e.ret()

    e.finalize()
    query = ctypes.CFUNCTYPE(ctypes.c_uint64)
    try:
        max_basic_leaf = query(e.symbol('max_basic_leaf'))()
        if max_basic_leaf >= 1:
            leaf1_ecx = query(e.symbol('leaf1_ecx'))()
        else:
            leaf1_ecx = 0
        if max_basic_leaf >= 7:
            leaf7_ebx = query(e.symbol('leaf7_ebx'))()
        else:
            leaf7_ebx = 0
    finally:
        e.unmap()

    cpu_features = CpuFeatures(
        avx=bool(leaf1_ecx & (1 << 28)),
        avx2=bool(leaf7_ebx & (1 << 5)),
        fma=bool(leaf1_ecx & (1 << 12)),
    )

init_cpu_features()
