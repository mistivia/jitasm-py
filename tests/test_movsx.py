# Copyright (c) 2026, Mistivia <i@mistivia.com>
# Distributed under the terms of the GPLv3

import ctypes

from jitasm.utils import ccall
from jitasm.x86_64 import *

def test_movsx() -> None:
    e = Emitter()
    e.label('f')
    e.mov(RAX, 0x80)
    e.movsx(RAX, AL)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == -128

    e = Emitter()
    e.label('f')
    e.mov(R8, 0x8001)
    e.movsx(RAX, R8W)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == -32767

    e = Emitter()
    e.label('f')
    e.mov(R10, 0x80000001)
    e.movsx(R9, R10D)
    e.mov(RAX, R9)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == -2147483647

    value8 = ctypes.c_uint8(0x80)
    e = Emitter()
    e.label('f')
    e.movsx(RAX, Mem(BYTE, RDI))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(value8)) == -128

    value16 = ctypes.c_uint16(0x8001)
    e = Emitter()
    e.label('f')
    e.movsx(RAX, Mem(WORD, RDI))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(value16)) == -32767

    value32 = ctypes.c_uint32(0x80000001)
    e = Emitter()
    e.label('f')
    e.movsx(RAX, Mem(DWORD, RDI))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(value32)) == -2147483647

    values = (ctypes.c_uint16 * 4)(0x8000, 0x8001, 0x8002, 0x8003)
    e = Emitter()
    e.label('f')
    e.mov(R8, RDI)
    e.mov(R9, RSI)
    e.movsx(R10, Mem(WORD, Sib(R8, R9, 2, 2)))
    e.mov(RAX, R10)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(values), 1) == -32766

    value8 = ctypes.c_uint8(0xFE)
    e = Emitter()
    e.label('f')
    e.movsx(RAX, Mem(BYTE, Sib(None, RDI, 1, 0)))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(value8)) == -2

    data = (ctypes.c_uint8 * 512)()
    data[129] = 0x80
    e = Emitter()
    e.label('f')
    e.movsx(RAX, Mem(BYTE, Sib(RDI, None, 1, -129)))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(data) + 258) == -128

    value32 = ctypes.c_uint32(0xF2345678)
    e = Emitter()
    e.label('f')
    e.mov(R11, R13)
    e.mov(R13, RDI)
    e.movsx(RAX, Mem(DWORD, Sib(R13)))
    e.mov(R13, R11)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(value32)) == -231451016

    values = (ctypes.c_uint8 * 32)()
    values[24] = 0x81
    e = Emitter()
    e.label('f')
    e.mov(R11, R12)
    e.mov(R12, RSI)
    e.movsx(RAX, Mem(BYTE, Sib(RDI, R12, 8, 0)))
    e.mov(R12, R11)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(values), 3) == -127

    e = Emitter()
    e.label('f')
    e.movsx(RAX, Mem(BYTE, Rel('value')))
    e.ret()
    e.set_section(Section.DATA)
    e.label('value')
    e._emit_bytes(b'\x80')
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == -128
