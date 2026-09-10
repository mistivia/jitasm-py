# Copyright (c) 2026, Mistivia <i@mistivia.com>
# Distributed under the terms of the GPLv3

import ctypes

from jitasm.utils import ccall

from jitasm.x86_64 import *


def test_mov() -> None:
    e = Emitter()
    e.mov(RAX, 0)
    e.mov(R8, 0)
    assert e.text == b'\x48\x31\xc0\x4d\x31\xc0'

    e = Emitter()
    e.label('f')
    e.mov(RAX, 0)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == 0

    e = Emitter()
    e.label('f')
    e.mov(R8, RDI)
    e.mov(R8, 0)
    e.mov(RAX, R8)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, 0xFEDCBA9876543210) == 0

    e = Emitter()
    e.label('f')
    e.mov(R8, 0xFEDCBA9876543210)
    e.mov(RAX, R8)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == -0x0123456789ABCDF0

    e = Emitter()
    e.label('f')
    e.mov(RAX, (1 << 64) - 1)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == -1

    e = Emitter()
    failed = False
    try:
        e.mov(RAX, 1 << 64)
    except EmitterError:
        failed = True
    assert failed
    assert e.text == b''

    e = Emitter()
    e.label('f')
    e.mov(RAX, 0xFEDCBA9876543210)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == -0x0123456789ABCDF0

    value = ctypes.c_uint64(0xFEDCBA9876543210)
    e = Emitter()
    e.label('f')
    e.mov(RAX, Mem(QWORD, RDI))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(value)) == -0x0123456789ABCDF0

    value = ctypes.c_uint64(0)
    e = Emitter()
    e.label('f')
    e.mov(Mem(QWORD, RDI), RSI)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    _ = ccall(f, ctypes.addressof(value), 0xFEDCBA9876543210)
    assert value.value == 0xFEDCBA9876543210

    value = ctypes.c_uint64(0xFFFFFFFFFFFFFFFF)
    e = Emitter()
    e.label('f')
    e.mov(Mem(DWORD, RDI), RSI)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    _ = ccall(f, ctypes.addressof(value), 0x1234567887654321)
    assert value.value == 0xFFFFFFFF87654321

    value = ctypes.c_uint64(0xFFFFFFFFFFFFFFFF)
    e = Emitter()
    e.label('f')
    e.mov(Mem(WORD, RDI), RSI)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    _ = ccall(f, ctypes.addressof(value), 0x1234567887654321)
    assert value.value == 0xFFFFFFFFFFFF4321

    value = ctypes.c_uint64(0xFFFFFFFFFFFFFFFF)
    e = Emitter()
    e.label('f')
    e.mov(Mem(BYTE, RDI), RSI)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    _ = ccall(f, ctypes.addressof(value), 0x1234567887654321)
    assert value.value == 0xFFFFFFFFFFFFFF21

    e = Emitter()
    e.label('f')
    e.mov(RAX, Mem(QWORD, Rel('value')))
    e.ret()
    e.set_section(Section.DATA)
    e.label('value')
    e._emit_bytes((0xFEDCBA9876543210).to_bytes(8, 'little'))
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == -0x0123456789ABCDF0
