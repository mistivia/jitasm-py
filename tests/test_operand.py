# Copyright (c) 2026, Mistivia <i@mistivia.com>
# Distributed under the terms of the GPLv3

import ctypes

import jitasm.x86_64 as x86
from jitasm.utils import ccall

from jitasm.x86_64 import *


def test_operand() -> None:
    if x86.cpu_features.avx:
        assert encode_vex(
            XMM1, XMM2, XMM3, 0x58, VexMap.MAP_0F, VexPP.NONE, VexW.W0
        ) == b'\xc4\xe1\x68\x58\xcb'
        assert encode_vex(
            YMM1, YMM2, YMM3, 0x58, VexMap.MAP_0F, VexPP.NONE, VexW.W0
        ) == b'\xc4\xe1\x6c\x58\xcb'
        assert encode_vex(
            XMM9, XMM10, XMM11, 0x58, VexMap.MAP_0F, VexPP.NONE, VexW.W0
        ) == b'\xc4\x41\x28\x58\xcb'
        assert encode_vex_rm(
            1, dword_ptr(R8), VexL.L128, 0x10,
            VexMap.MAP_0F, VexPP.PF3, VexW.W0,
        ) == b'\xc4\xc1\x7a\x10\x08'
        assert encode_vex_rm(
            dword_ptr(R8 + R9 * 4 + 16), 10, VexL.L128, 0x11,
            VexMap.MAP_0F, VexPP.PF3, VexW.W0,
        ) == b'\xc4\x01\x7a\x11\x54\x88\x10'
        assert encode_vex_rm(
            2, dword_ptr(RIP + 'value'), VexL.L128, 0x10,
            VexMap.MAP_0F, VexPP.PF3, VexW.W0,
        ) == b'\xc4\xe1\x7a\x10\x15\x00\x00\x00\x00'

        failed = False
        try:
            _ = encode_vex(
                XMM1, XMM2, XMM3, 0x58, VexMap.MAP_0F, VexPP.NONE, VexW.W0, 256,
            )
        except EmitterError:
            failed = True
        assert failed

        failed = False
        try:
            _ = encode_vex_rm(
                16, dword_ptr(RAX), VexL.L128, 0x10,
                VexMap.MAP_0F, VexPP.PF3, VexW.W0,
            )
        except EmitterError:
            failed = True
        assert failed

    assert RAX * 4 == Sib(index=RAX, scale=4)
    assert 4 * RAX == Sib(index=RAX, scale=4)
    assert RAX + RCX == Sib(RAX, RCX)
    assert RAX + RCX * 4 + 8 == Sib(RAX, RCX, 4, 8)
    assert RCX * 4 + RAX + 8 == Sib(RAX, RCX, 4, 8)
    assert 8 + RAX + RCX * 2 == Sib(RAX, RCX, 2, 8)
    assert RAX + 4 + 12 == Sib(RAX, offset=16)
    assert RAX - 8 == Sib(RAX, offset=-8)
    assert RAX + RCX * 4 - 8 == Sib(RAX, RCX, 4, -8)
    assert RAX + 20 - 8 == Sib(RAX, offset=12)
    assert 4 + RAX + 12 + 20 == Sib(RAX, offset=36)
    assert RAX + RCX * 4 + 8 + 16 == Sib(RAX, RCX, 4, 24)
    assert Sib(RAX) + RCX + 16 == Sib(RAX, RCX, 1, 16)
    assert Sib(RAX, offset=8) + Sib(index=RCX, scale=4, offset=16) == Sib(RAX, RCX, 4, 24)
    assert RIP + 'value' == Rel('value')
    assert byte_ptr(RAX) == Mem(BYTE, RAX)
    assert word_ptr(RAX + 2) == Mem(WORD, Sib(RAX, offset=2))
    assert dword_ptr(RAX + RCX * 4) == Mem(DWORD, Sib(RAX, RCX, 4))
    assert qword_ptr(RIP + 'value') == Mem(QWORD, Rel('value'))

    values = (ctypes.c_uint64 * 4)(11, 22, 33, 44)
    e = Emitter()
    e.label('f')
    e.mov(RAX, qword_ptr(RDI + RSI * 8 + 8))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(values), 1) == 33

    e = Emitter()
    e.label('f')
    e.mov(RAX, qword_ptr(RDI + 4 + 12))
    e.ret()
    e.finalize()
    f = e.symbol('f')
    assert ccall(f, ctypes.addressof(values)) == 33

    e = Emitter()
    e.label('f')
    e.movzx(RAX, byte_ptr(RIP + 'value'))
    e.ret()
    e.set_section(Section.DATA)
    e.label('value')
    e._emit_bytes(b'\x80')
    e.finalize()
    f = e.symbol('f')
    assert ccall(f) == 0x80
