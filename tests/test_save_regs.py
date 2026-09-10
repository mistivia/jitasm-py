import ctypes
import random

from jitasm.utils import ccall
from jitasm.x86_64 import *

def test_save_regs():
    e = Emitter()
    e.label('entry')
    e.begin()
    e.mov(r8, 12)
    e.mov(r9, 13)
    with e.save_regs(r8, r9):
        e.mov(r8, 22)
        e.mov(r9, 23)
    e.mov(rax, 0)
    e.add(rax, r8)
    e.add(rax, r9)
    e.add(rax, r8)
    e.end()
    e.finalize()
    assert ccall(e.symbol('entry')) == 37


