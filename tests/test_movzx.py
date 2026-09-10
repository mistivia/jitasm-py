# Copyright (c) 2026, Mistivia <i@mistivia.com>
# Distributed under the terms of the GPLv3

import ctypes

from jitasm.utils import ccall
from jitasm.x86_64 import *

def test_movzx() -> None:
    e = Emitter()
    e.label('f')
    e.mov(RAX, 0xFEDCBA98765432FF)
    e.movzx(RAX, AL)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == 0xFF

    e = Emitter()
    e.label('f')
    e.mov(R8, 0xFEDCBA987654ABCD)
    e.movzx(RAX, R8W)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == 0xABCD

    e = Emitter()
    e.label('f')
    e.mov(R10, 0xFEDCBA9887654321)
    e.movzx(R9, R10D)
    e.mov(RAX, R9)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == 0x87654321

    e = Emitter()
    e.label('f')
    e.movzx(RAX, Mem(BYTE, RDI))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    value8 = ctypes.c_uint8(0x80)
    assert ccall(f, ctypes.addressof(value8)) == 0x80

    e = Emitter()
    e.label('f')
    e.movzx(RAX, Mem(WORD, RDI))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    value16 = ctypes.c_uint16(0x8001)
    assert ccall(f, ctypes.addressof(value16)) == 0x8001

    e = Emitter()
    e.label('f')
    e.movzx(RAX, Mem(DWORD, RDI))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    value32 = ctypes.c_uint32(0x80000001)
    assert ccall(f, ctypes.addressof(value32)) == 0x80000001

    values = (ctypes.c_uint16 * 4)(0x8000, 0x8001, 0x8002, 0x8003)
    e = Emitter()
    e.label('f')
    e.mov(R8, RDI)
    e.mov(R9, RSI)
    e.movzx(R10, Mem(WORD, Sib(R8, R9, 2, 2)))
    e.mov(RAX, R10)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(values), 1) == 0x8002

    value8 = ctypes.c_uint8(0xFE)
    e = Emitter()
    e.label('f')
    e.movzx(RAX, Mem(BYTE, Sib(None, RDI, 1, 0)))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(value8)) == 0xFE

    data = (ctypes.c_uint8 * 512)()
    data[128] = 0x34
    data[129] = 0xF2
    e = Emitter()
    e.label('f')
    e.movzx(RAX, Mem(WORD, Sib(RDI, None, 1, 128)))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(data)) == 0xF234

    value32 = ctypes.c_uint32(0xF2345678)
    e = Emitter()
    e.label('f')
    e.mov(R11, R13)
    e.mov(R13, RDI)
    e.movzx(RAX, Mem(DWORD, Sib(R13)))
    e.mov(R13, R11)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(value32)) == 0xF2345678

    values = (ctypes.c_uint16 * 4)(0x1000, 0x2000, 0x3000, 0x4000)
    e = Emitter()
    e.label('f')
    e.mov(R11, R12)
    e.mov(R12, RSI)
    e.movzx(RAX, Mem(WORD, Sib(RDI, R12, 2, 0)))
    e.mov(R12, R11)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(values), 3) == 0x4000

    e = Emitter()
    e.label('f')
    e.movzx(RAX, Mem(BYTE, Rel('value')))
    e.ret()
    e.set_section(Section.DATA)
    e.label('value')
    e._emit_bytes(b'\x80')
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == 0x80
