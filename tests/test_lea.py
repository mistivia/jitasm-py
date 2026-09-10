# Copyright (c) 2026, Mistivia <i@mistivia.com>
# Distributed under the terms of the GPLv3


from jitasm.utils import ccall

from jitasm.x86_64 import *


def test_lea() -> None:
    e = Emitter()
    e.label('f')
    e.lea(RAX, Mem(QWORD, RDI))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, 0x123456789ABCDEF0) == 0x123456789ABCDEF0

    e = Emitter()
    e.label('f')
    e.lea(RAX, Mem(QWORD, Sib(RDI, RSI, 8, 16)))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, 1000, 3) == 1040

    e = Emitter()
    e.label('f')
    e.lea(R8, Mem(QWORD, Sib(RDI, RSI, 4, -8)))
    e.mov(RAX, R8)
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, 1000, 3) == 1004

    e = Emitter()
    e.label('f')
    e.lea(RAX, Mem(QWORD, Rel('value')))
    e.ret()
    e.set_section(Section.DATA)
    e.label('value')
    e._emit_bytes(b'\x00')
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == e.symbol('value')
