# Copyright (c) 2026, Mistivia <i@mistivia.com>
# Distributed under the terms of the GPLv3

from jitasm.utils import ccall
from jitasm.x86_64 import *


def test_jcc() -> None:
    e = Emitter()
    e.label('ja')
    e.cmp(RDI, RSI)
    e.ja('.ja_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.ja_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jae')
    e.cmp(RDI, RSI)
    e.jae('.jae_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jae_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jb')
    e.cmp(RDI, RSI)
    e.jb('.jb_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jb_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jbe')
    e.cmp(RDI, RSI)
    e.jbe('.jbe_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jbe_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jc')
    e.cmp(RDI, RSI)
    e.jc('.jc_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jc_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jnc')
    e.cmp(RDI, RSI)
    e.jnc('.jnc_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jnc_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('je')
    e.cmp(RDI, RSI)
    e.je('.je_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.je_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jne')
    e.cmp(RDI, RSI)
    e.jne('.jne_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jne_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jz')
    e.cmp(RDI, RSI)
    e.jz('.jz_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jz_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jnz')
    e.cmp(RDI, RSI)
    e.jnz('.jnz_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jnz_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jg')
    e.cmp(RDI, RSI)
    e.jg('.jg_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jg_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jge')
    e.cmp(RDI, RSI)
    e.jge('.jge_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jge_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jl')
    e.cmp(RDI, RSI)
    e.jl('.jl_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jl_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jle')
    e.cmp(RDI, RSI)
    e.jle('.jle_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jle_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jna')
    e.cmp(RDI, RSI)
    e.jna('.jna_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jna_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jnae')
    e.cmp(RDI, RSI)
    e.jnae('.jnae_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jnae_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jnb')
    e.cmp(RDI, RSI)
    e.jnb('.jnb_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jnb_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jnbe')
    e.cmp(RDI, RSI)
    e.jnbe('.jnbe_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jnbe_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jng')
    e.cmp(RDI, RSI)
    e.jng('.jng_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jng_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jnge')
    e.cmp(RDI, RSI)
    e.jnge('.jnge_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jnge_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jnl')
    e.cmp(RDI, RSI)
    e.jnl('.jnl_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jnl_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jnle')
    e.cmp(RDI, RSI)
    e.jnle('.jnle_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jnle_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jo')
    e.cmp(RDI, RSI)
    e.jo('.jo_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jo_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jno')
    e.cmp(RDI, RSI)
    e.jno('.jno_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jno_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('js')
    e.cmp(RDI, RSI)
    e.js('.js_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.js_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jns')
    e.cmp(RDI, RSI)
    e.jns('.jns_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jns_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jp')
    e.cmp(RDI, RSI)
    e.jp('.jp_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jp_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jpe')
    e.cmp(RDI, RSI)
    e.jpe('.jpe_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jpe_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jnp')
    e.cmp(RDI, RSI)
    e.jnp('.jnp_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jnp_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jpo')
    e.cmp(RDI, RSI)
    e.jpo('.jpo_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jpo_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jeq')
    e.cmp(RDI, RSI)
    e.jeq('.jeq_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jeq_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jgt')
    e.cmp(RDI, RSI)
    e.jgt('.jgt_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jgt_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jlt')
    e.cmp(RDI, RSI)
    e.jlt('.jlt_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jlt_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jgtu')
    e.cmp(RDI, RSI)
    e.jgtu('.jgtu_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jgtu_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jgeu')
    e.cmp(RDI, RSI)
    e.jgeu('.jgeu_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jgeu_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jltu')
    e.cmp(RDI, RSI)
    e.jltu('.jltu_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jltu_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('jleu')
    e.cmp(RDI, RSI)
    e.jleu('.jleu_taken')
    e.mov(RAX, 0)
    e.ret()
    e.label('.jleu_taken')
    e.mov(RAX, 1)
    e.ret()

    e.label('backward')
    e.mov(RAX, 3)
    e.label('.loop')
    e.sub(RAX, 1)
    e.jnz('.loop')
    e.ret()

    e.finalize()
    try:
        assert ccall(e.symbol('ja'), -1, 1) == 1
        assert ccall(e.symbol('ja'), 3, 3) == 0
        assert ccall(e.symbol('ja'), 0, 1) == 0

        assert ccall(e.symbol('jae'), -1, 1) == 1
        assert ccall(e.symbol('jae'), 3, 3) == 1
        assert ccall(e.symbol('jae'), 0, 1) == 0

        assert ccall(e.symbol('jb'), 0, 1) == 1
        assert ccall(e.symbol('jb'), 3, 3) == 0
        assert ccall(e.symbol('jb'), -1, 1) == 0

        assert ccall(e.symbol('jbe'), 0, 1) == 1
        assert ccall(e.symbol('jbe'), 3, 3) == 1
        assert ccall(e.symbol('jbe'), -1, 1) == 0

        assert ccall(e.symbol('jc'), 0, 1) == 1
        assert ccall(e.symbol('jc'), 3, 3) == 0
        assert ccall(e.symbol('jc'), -1, 1) == 0

        assert ccall(e.symbol('jnc'), -1, 1) == 1
        assert ccall(e.symbol('jnc'), 3, 3) == 1
        assert ccall(e.symbol('jnc'), 0, 1) == 0

        assert ccall(e.symbol('je'), 3, 3) == 1
        assert ccall(e.symbol('je'), 2, 3) == 0
        assert ccall(e.symbol('je'), 4, 3) == 0

        assert ccall(e.symbol('jne'), 3, 3) == 0
        assert ccall(e.symbol('jne'), 2, 3) == 1
        assert ccall(e.symbol('jne'), 4, 3) == 1

        assert ccall(e.symbol('jz'), 3, 3) == 1
        assert ccall(e.symbol('jz'), 2, 3) == 0
        assert ccall(e.symbol('jz'), 4, 3) == 0

        assert ccall(e.symbol('jnz'), 3, 3) == 0
        assert ccall(e.symbol('jnz'), 2, 3) == 1
        assert ccall(e.symbol('jnz'), 4, 3) == 1

        assert ccall(e.symbol('jg'), 4, 3) == 1
        assert ccall(e.symbol('jg'), 3, 3) == 0
        assert ccall(e.symbol('jg'), -1, 1) == 0

        assert ccall(e.symbol('jge'), 4, 3) == 1
        assert ccall(e.symbol('jge'), 3, 3) == 1
        assert ccall(e.symbol('jge'), -1, 1) == 0

        assert ccall(e.symbol('jl'), -1, 1) == 1
        assert ccall(e.symbol('jl'), 3, 3) == 0
        assert ccall(e.symbol('jl'), 4, 3) == 0

        assert ccall(e.symbol('jle'), -1, 1) == 1
        assert ccall(e.symbol('jle'), 3, 3) == 1
        assert ccall(e.symbol('jle'), 4, 3) == 0

        assert ccall(e.symbol('jna'), 0, 1) == 1
        assert ccall(e.symbol('jna'), 3, 3) == 1
        assert ccall(e.symbol('jna'), -1, 1) == 0

        assert ccall(e.symbol('jnae'), 0, 1) == 1
        assert ccall(e.symbol('jnae'), 3, 3) == 0
        assert ccall(e.symbol('jnae'), -1, 1) == 0

        assert ccall(e.symbol('jnb'), -1, 1) == 1
        assert ccall(e.symbol('jnb'), 3, 3) == 1
        assert ccall(e.symbol('jnb'), 0, 1) == 0

        assert ccall(e.symbol('jnbe'), -1, 1) == 1
        assert ccall(e.symbol('jnbe'), 3, 3) == 0
        assert ccall(e.symbol('jnbe'), 0, 1) == 0

        assert ccall(e.symbol('jng'), -1, 1) == 1
        assert ccall(e.symbol('jng'), 3, 3) == 1
        assert ccall(e.symbol('jng'), 4, 3) == 0

        assert ccall(e.symbol('jnge'), -1, 1) == 1
        assert ccall(e.symbol('jnge'), 3, 3) == 0
        assert ccall(e.symbol('jnge'), 4, 3) == 0

        assert ccall(e.symbol('jnl'), 4, 3) == 1
        assert ccall(e.symbol('jnl'), 3, 3) == 1
        assert ccall(e.symbol('jnl'), -1, 1) == 0

        assert ccall(e.symbol('jnle'), 4, 3) == 1
        assert ccall(e.symbol('jnle'), 3, 3) == 0
        assert ccall(e.symbol('jnle'), -1, 1) == 0

        assert ccall(e.symbol('jo'), -9223372036854775808, 1) == 1
        assert ccall(e.symbol('jo'), 3, 1) == 0

        assert ccall(e.symbol('jno'), -9223372036854775808, 1) == 0
        assert ccall(e.symbol('jno'), 3, 1) == 1

        assert ccall(e.symbol('js'), 0, 1) == 1
        assert ccall(e.symbol('js'), 3, 1) == 0

        assert ccall(e.symbol('jns'), 0, 1) == 0
        assert ccall(e.symbol('jns'), 3, 1) == 1

        assert ccall(e.symbol('jp'), 3, 0) == 1
        assert ccall(e.symbol('jp'), 1, 0) == 0

        assert ccall(e.symbol('jpe'), 3, 0) == 1
        assert ccall(e.symbol('jpe'), 1, 0) == 0

        assert ccall(e.symbol('jnp'), 3, 0) == 0
        assert ccall(e.symbol('jnp'), 1, 0) == 1

        assert ccall(e.symbol('jpo'), 3, 0) == 0
        assert ccall(e.symbol('jpo'), 1, 0) == 1

        assert ccall(e.symbol('jeq'), 3, 3) == 1
        assert ccall(e.symbol('jeq'), 2, 3) == 0
        assert ccall(e.symbol('jeq'), 4, 3) == 0

        assert ccall(e.symbol('jgt'), 4, 3) == 1
        assert ccall(e.symbol('jgt'), 3, 3) == 0
        assert ccall(e.symbol('jgt'), -1, 1) == 0

        assert ccall(e.symbol('jlt'), -1, 1) == 1
        assert ccall(e.symbol('jlt'), 3, 3) == 0
        assert ccall(e.symbol('jlt'), 4, 3) == 0

        assert ccall(e.symbol('jgtu'), -1, 1) == 1
        assert ccall(e.symbol('jgtu'), 3, 3) == 0
        assert ccall(e.symbol('jgtu'), 0, 1) == 0

        assert ccall(e.symbol('jgeu'), -1, 1) == 1
        assert ccall(e.symbol('jgeu'), 3, 3) == 1
        assert ccall(e.symbol('jgeu'), 0, 1) == 0

        assert ccall(e.symbol('jltu'), 0, 1) == 1
        assert ccall(e.symbol('jltu'), 3, 3) == 0
        assert ccall(e.symbol('jltu'), -1, 1) == 0

        assert ccall(e.symbol('jleu'), 0, 1) == 1
        assert ccall(e.symbol('jleu'), 3, 3) == 1
        assert ccall(e.symbol('jleu'), -1, 1) == 0

        assert ccall(e.symbol('backward')) == 0
    finally:
        e.unmap()
