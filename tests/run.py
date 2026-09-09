# Copyright (c) 2026, Mistivia <i@mistivia.com>
# Distributed under the terms of the GPLv3

import os

if os.name == 'nt':
    from test_windows import test_windows

    test_windows()
elif os.name == 'posix':
    import jitasm.x86_64 as x86

    from test_arith import test_arith
    from test_basic import test_function_returning_42
    from test_bitwise import test_bitwise
    from test_call import test_call
    from test_cmov import test_cmov
    from test_cond import test_cond
    from test_cpu_features import test_cpu_features
    from test_data import test_data
    from test_div import test_div
    from test_float import test_float
    from test_float32 import test_float32
    from test_label import test_label
    from test_jmp import test_jmp
    from test_jcc import test_jcc
    from test_lea import test_lea
    from test_mov import test_mov
    from test_movsd import test_movsd
    from test_movsx import test_movsx
    from test_movzx import test_movzx
    from test_mul import test_mul
    from test_operand import test_operand
    from test_qsort import test_qsort
    from test_sib import test_sib
    from test_shift import test_shift
    from test_simd import test_simd
    from test_unmap import test_unmap
    from test_utils import test_utils

    def test_floats() -> None:
        test_cond()
        test_float()
        test_float32()
        test_movsd()

    test_arith()
    test_function_returning_42()
    test_bitwise()
    test_call()
    test_cmov()
    test_cpu_features()
    test_data()
    test_div()
    cpu_features = x86.cpu_features
    x86.cpu_features = x86.CpuFeatures(False, False, False)
    try:
        test_floats()
    finally:
        x86.cpu_features = cpu_features
    if cpu_features.avx:
        test_floats()
        test_simd()
    test_label()
    test_jmp()
    test_jcc()
    test_mov()
    test_movzx()
    test_mul()
    test_movsx()
    test_sib()
    test_shift()
    test_lea()
    test_operand()
    test_qsort()
    test_unmap()
    test_utils()
else:
    raise RuntimeError('unsupported operating system: %s' % os.name)
