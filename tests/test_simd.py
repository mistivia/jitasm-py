# Copyright (c) 2026, Mistivia <i@mistivia.com>
# Distributed under the terms of the GPLv3

import ctypes
import math

from jitasm.utils import ccall
from jitasm.x86_64 import *


def simd_move() -> None:
    source_storage = (ctypes.c_ubyte * 95)()
    result_storage = (ctypes.c_ubyte * 95)()
    source_address = (ctypes.addressof(source_storage) + 31) & ~31
    result_address = (ctypes.addressof(result_storage) + 31) & ~31
    assert source_address % 32 == 0
    assert result_address % 32 == 0
    source = (ctypes.c_float * 16).from_address(source_address)
    result = (ctypes.c_float * 16).from_address(result_address)
    source[:] = range(16)

    e = Emitter()
    e.label('f')
    e.vmovaps(ymm0, m256_ptr(rdi))
    e.vmovaps(ymm1, ymm0)
    e.vmovaps(m256_ptr(rsi), ymm1)
    e.vmovaps(xmm2, m128_ptr(rdi))
    e.vmovaps(xmm3, xmm2)
    e.vmovaps(m128_ptr(rsi), xmm3)
    e.vmovaps(ymm4, m256_ptr(rdi + rdx * 8))
    e.vmovaps(m256_ptr(rsi + rcx * 8), ymm4)
    e.ret()
    e.finalize()
    _ = ccall(e.symbol('f'), source_address, result_address, 4, 4)
    assert list(result) == list(source)

    result[:] = [0.0] * 16
    e = Emitter()
    e.label('f')
    e.vmovaps(ymm0, m256_ptr(RIP + 'value'))
    e.vmovaps(m256_ptr(rdi), ymm0)
    e.vmovaps(xmm1, m128_ptr(RIP + 'value128'))
    e.vmovaps(m128_ptr(rdi + 32), xmm1)
    e.ret()
    e.set_section(Section.DATA)
    e.align(32)
    e.label('value')
    e.dd(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    e.label('value128')
    e.dd(8.0, 9.0, 10.0, 11.0)
    e.finalize()
    _ = ccall(e.symbol('f'), result_address)
    assert list(result[:12]) == [float(i) for i in range(12)]


def simd_arithmetic() -> None:
    left = (ctypes.c_float * 8)(*range(1, 9))
    right = (ctypes.c_float * 8)(*range(9, 17))
    add_result = (ctypes.c_float * 8)()
    sub_result = (ctypes.c_float * 8)()
    mul_result = (ctypes.c_float * 8)()
    div_result = (ctypes.c_float * 8)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm0, m256_ptr(rdi))
    e.vmovups(ymm1, m256_ptr(rsi))
    e.vaddps(ymm2, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx), ymm2)
    e.vsubps(ymm3, ymm1, ymm0)
    e.vmovups(m256_ptr(rcx), ymm3)
    e.vmulps(ymm4, ymm0, ymm1)
    e.vmovups(m256_ptr(r8), ymm4)
    e.vdivps(ymm5, ymm1, ymm0)
    e.vmovups(m256_ptr(r9), ymm5)
    e.ret()
    e.finalize()
    _ = ccall(
        e.symbol('f'), ctypes.addressof(left), ctypes.addressof(right),
        ctypes.addressof(add_result), ctypes.addressof(sub_result),
        ctypes.addressof(mul_result), ctypes.addressof(div_result),
    )
    assert list(add_result) == [float(i + i + 10) for i in range(8)]
    assert list(sub_result) == [8.0] * 8
    assert list(mul_result) == [float((i + 1) * (i + 9)) for i in range(8)]
    assert list(div_result) == [ctypes.c_float((i + 9) / (i + 1)).value for i in range(8)]

    xmm_result = (ctypes.c_float * 16)()
    e = Emitter()

    e.label('f')
    (
        e.vmovups(xmm0, m128_ptr(rdi)),
        e.vmovups(xmm1, m128_ptr(rsi)),
        e.vaddps(xmm2, xmm0, xmm1),
        e.vmovups(m128_ptr(rdx), xmm2),
        e.vsubps(xmm3, xmm1, xmm0),
        e.vmovups(m128_ptr(rdx + 16), xmm3),
        e.vmulps(xmm4, xmm0, xmm1),
        e.vmovups(m128_ptr(rdx + 32), xmm4),
        e.vdivps(xmm5, xmm1, xmm0),
        e.vmovups(m128_ptr(rdx + 48), xmm5),
        e.ret(),
    )

    e.finalize()
    _ = ccall(
        e.symbol('f'), ctypes.addressof(left), ctypes.addressof(right),
        ctypes.addressof(xmm_result),
    )
    assert list(xmm_result[:4]) == [float(i + i + 10) for i in range(4)]
    assert list(xmm_result[4:8]) == [8.0] * 4
    assert list(xmm_result[8:12]) == [float((i + 1) * (i + 9)) for i in range(4)]
    assert list(xmm_result[12:]) == [ctypes.c_float((i + 9) / (i + 1)).value for i in range(4)]


def simd_addsub() -> None:
    left = (ctypes.c_float * 8)(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    right = (ctypes.c_float * 8)(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
    result = (ctypes.c_float * 12)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm8, m256_ptr(rdi))
    e.vmovups(ymm9, m256_ptr(rsi))
    e.vaddsubps(ymm10, ymm8, ymm9)
    e.vmovups(m256_ptr(rdx), ymm10)
    e.vaddsubps(xmm11, xmm8, xmm9)
    e.vmovups(m128_ptr(rdx + 32), xmm11)
    e.ret()
    e.finalize()
    _ = ccall(
        e.symbol('f'), ctypes.addressof(left), ctypes.addressof(right), ctypes.addressof(result),
    )

    assert list(result[:8]) == [-8.0, 12.0, -8.0, 16.0, -8.0, 20.0, -8.0, 24.0]
    assert list(result[8:12]) == [-8.0, 12.0, -8.0, 16.0]


def simd_min_max_sqrt() -> None:
    packed_left = (ctypes.c_float * 8)(1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0)
    packed_right = (ctypes.c_float * 8)(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
    packed_result = (ctypes.c_float * 36)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm0, m256_ptr(rdi))
    e.vmovups(ymm1, m256_ptr(rsi))
    e.vsqrtps(ymm2, ymm0)
    e.vmovups(m256_ptr(rdx), ymm2)
    e.vmaxps(ymm3, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx + 32), ymm3)
    e.vminps(ymm4, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx + 64), ymm4)
    e.vsqrtps(xmm2, xmm0)
    e.vmovups(m128_ptr(rdx + 96), xmm2)
    e.vmaxps(xmm3, xmm0, xmm1)
    e.vmovups(m128_ptr(rdx + 112), xmm3)
    e.vminps(xmm4, xmm0, xmm1)
    e.vmovups(m128_ptr(rdx + 128), xmm4)
    e.ret()
    e.finalize()
    _ = ccall(
        e.symbol('f'), ctypes.addressof(packed_left), ctypes.addressof(packed_right),
        ctypes.addressof(packed_result),
    )

    expected_sqrt = [float(i) for i in range(1, 9)]
    expected_max = [max(left, right) for left, right in zip(packed_left, packed_right)]
    expected_min = [min(left, right) for left, right in zip(packed_left, packed_right)]
    assert list(packed_result[:8]) == expected_sqrt
    assert list(packed_result[8:16]) == expected_max
    assert list(packed_result[16:24]) == expected_min
    assert list(packed_result[24:28]) == expected_sqrt[:4]
    assert list(packed_result[28:32]) == expected_max[:4]
    assert list(packed_result[32:36]) == expected_min[:4]


def simd_bitwise() -> None:
    bitwise_left = (ctypes.c_uint32 * 8)(
        0xffffffff, 0x00000000, 0xaaaaaaaa, 0x55555555,
        0x12345678, 0x87654321, 0x7f800000, 0x80000000,
    )
    bitwise_right = (ctypes.c_uint32 * 8)(
        0x0f0f0f0f, 0xf0f0f0f0, 0x55555555, 0xaaaaaaaa,
        0xffffffff, 0x00000000, 0x007fffff, 0x7fffffff,
    )
    bitwise_result = (ctypes.c_uint32 * 48)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm0, m256_ptr(rdi))
    e.vmovups(ymm1, m256_ptr(rsi))
    e.vandps(ymm2, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx), ymm2)
    e.vorps(ymm3, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx + 32), ymm3)
    e.vandps(xmm2, xmm0, xmm1)
    e.vmovups(m128_ptr(rdx + 64), xmm2)
    e.vorps(xmm3, xmm0, xmm1)
    e.vmovups(m128_ptr(rdx + 80), xmm3)
    e.vandnps(ymm4, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx + 96), ymm4)
    e.vxorps(ymm5, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx + 128), ymm5)
    e.vandnps(xmm6, xmm0, xmm1)
    e.vmovups(m128_ptr(rdx + 160), xmm6)
    e.vxorps(xmm7, xmm0, xmm1)
    e.vmovups(m128_ptr(rdx + 176), xmm7)
    e.ret()
    e.finalize()
    _ = ccall(
        e.symbol('f'), ctypes.addressof(bitwise_left), ctypes.addressof(bitwise_right),
        ctypes.addressof(bitwise_result),
    )

    expected_and = [left & right for left, right in zip(bitwise_left, bitwise_right)]
    expected_or = [left | right for left, right in zip(bitwise_left, bitwise_right)]
    assert list(bitwise_result[:8]) == expected_and
    assert list(bitwise_result[8:16]) == expected_or
    assert list(bitwise_result[16:20]) == expected_and[:4]
    assert list(bitwise_result[20:24]) == expected_or[:4]
    assert list(bitwise_result[24:32]) == [
        0x00000000, 0xf0f0f0f0, 0x55555555, 0xaaaaaaaa,
        0xedcba987, 0x00000000, 0x007fffff, 0x7fffffff,
    ]
    assert list(bitwise_result[32:40]) == [
        0xf0f0f0f0, 0xf0f0f0f0, 0xffffffff, 0xffffffff,
        0xedcba987, 0x87654321, 0x7fffffff, 0xffffffff,
    ]
    assert list(bitwise_result[40:44]) == [0x00000000, 0xf0f0f0f0, 0x55555555, 0xaaaaaaaa]
    assert list(bitwise_result[44:48]) == [0xf0f0f0f0, 0xf0f0f0f0, 0xffffffff, 0xffffffff]


def simd_round() -> None:
    values = (ctypes.c_float * 8)(1.5, 2.5, -1.5, -2.5, 1.1, -1.1, 2.9, -2.9)
    result = (ctypes.c_float * 36)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm0, m256_ptr(rdi))
    e.vroundps(ymm1, ymm0)
    e.vmovups(m256_ptr(rsi), ymm1)
    e.vfloorps(ymm2, ymm0)
    e.vmovups(m256_ptr(rsi + 32), ymm2)
    e.vceilps(ymm3, ymm0)
    e.vmovups(m256_ptr(rsi + 64), ymm3)
    e.vtruncps(ymm4, ymm0)
    e.vmovups(m256_ptr(rsi + 96), ymm4)
    e.vroundps(xmm5, xmm0)
    e.vmovups(m128_ptr(rsi + 128), xmm5)
    e.ret()
    e.finalize()
    _ = ccall(e.symbol('f'), ctypes.addressof(values), ctypes.addressof(result))

    assert list(result[:8]) == [2.0, 2.0, -2.0, -2.0, 1.0, -1.0, 3.0, -3.0]
    assert list(result[8:16]) == [1.0, 2.0, -2.0, -3.0, 1.0, -2.0, 2.0, -3.0]
    assert list(result[16:24]) == [2.0, 3.0, -1.0, -2.0, 2.0, -1.0, 3.0, -2.0]
    assert list(result[24:32]) == [1.0, 2.0, -1.0, -2.0, 1.0, -1.0, 2.0, -2.0]
    assert list(result[32:36]) == [2.0, 2.0, -2.0, -2.0]


def simd_compare() -> None:
    left = (ctypes.c_float * 8)(1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0)
    right = (ctypes.c_float * 8)(1.0, 3.0, 2.0, 4.0, 0.0, -3.0, -3.0, -5.0)
    result = (ctypes.c_uint32 * 20)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm0, m256_ptr(rdi))
    e.vmovups(ymm1, m256_ptr(rsi))
    e.vcmpps(ymm2, ymm0, ymm1, 0)
    e.vmovups(m256_ptr(rdx), ymm2)
    e.vcmpps(ymm3, ymm0, ymm1, 1)
    e.vmovups(m256_ptr(rdx + 32), ymm3)
    e.vcmpps(xmm4, xmm0, xmm1, 2)
    e.vmovups(m128_ptr(rdx + 64), xmm4)
    e.ret()
    e.finalize()
    _ = ccall(
        e.symbol('f'), ctypes.addressof(left), ctypes.addressof(right),
        ctypes.addressof(result),
    )

    assert list(result[:8]) == [
        0xffffffff, 0, 0, 0xffffffff, 0, 0, 0xffffffff, 0,
    ]
    assert list(result[8:16]) == [
        0, 0xffffffff, 0, 0, 0xffffffff, 0, 0, 0,
    ]
    assert list(result[16:20]) == [0xffffffff, 0xffffffff, 0, 0xffffffff]


def simd_compare_pseudo() -> None:
    left = (ctypes.c_float * 8)(1.0, 2.0, 3.0, 4.0, float('nan'), -2.0, -3.0, -4.0)
    right = (ctypes.c_float * 8)(1.0, 3.0, 2.0, 4.0, 0.0, -3.0, -3.0, -5.0)
    result = (ctypes.c_uint32 * 80)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm0, m256_ptr(rdi))
    e.vmovups(ymm1, m256_ptr(rsi))
    e.veqps(ymm2, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx), ymm2)
    e.vltps(ymm2, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx + 32), ymm2)
    e.vleps(ymm2, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx + 64), ymm2)
    e.vunordps(ymm2, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx + 96), ymm2)
    e.vneps(ymm2, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx + 128), ymm2)
    e.vnltps(ymm2, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx + 160), ymm2)
    e.vnleps(ymm2, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx + 192), ymm2)
    e.vordps(ymm2, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx + 224), ymm2)
    e.vgtps(ymm2, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx + 256), ymm2)
    e.vgeps(ymm2, ymm0, ymm1)
    e.vmovups(m256_ptr(rdx + 288), ymm2)
    e.ret()
    e.finalize()
    _ = ccall(
        e.symbol('f'), ctypes.addressof(left), ctypes.addressof(right),
        ctypes.addressof(result),
    )

    assert list(result[:8]) == [
        0xffffffff, 0, 0, 0xffffffff, 0, 0, 0xffffffff, 0,
    ]
    assert list(result[8:16]) == [
        0, 0xffffffff, 0, 0, 0, 0, 0, 0,
    ]
    assert list(result[16:24]) == [
        0xffffffff, 0xffffffff, 0, 0xffffffff, 0, 0, 0xffffffff, 0,
    ]
    assert list(result[24:32]) == [
        0, 0, 0, 0, 0xffffffff, 0, 0, 0,
    ]
    assert list(result[32:40]) == [
        0, 0xffffffff, 0xffffffff, 0, 0xffffffff, 0xffffffff, 0, 0xffffffff,
    ]
    assert list(result[40:48]) == [
        0xffffffff, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff,
    ]
    assert list(result[48:56]) == [
        0, 0, 0xffffffff, 0, 0xffffffff, 0xffffffff, 0, 0xffffffff,
    ]
    assert list(result[56:64]) == [
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0, 0xffffffff,
        0xffffffff, 0xffffffff,
    ]
    assert list(result[64:72]) == [
        0, 0, 0xffffffff, 0, 0, 0xffffffff, 0, 0xffffffff,
    ]
    assert list(result[72:80]) == [
        0xffffffff, 0, 0xffffffff, 0xffffffff, 0, 0xffffffff, 0xffffffff,
        0xffffffff,
    ]


def simd_horizontal() -> None:
    left = (ctypes.c_float * 8)(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
    right = (ctypes.c_float * 8)(3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0)
    result = (ctypes.c_float * 24)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm8, m256_ptr(rdi))
    e.vmovups(ymm9, m256_ptr(rsi))
    e.vhaddps(ymm10, ymm8, ymm9)
    e.vmovups(m256_ptr(rdx), ymm10)
    e.vhsubps(ymm11, ymm8, ymm9)
    e.vmovups(m256_ptr(rdx + 32), ymm11)
    e.vhaddps(xmm12, xmm8, xmm9)
    e.vmovups(m128_ptr(rdx + 64), xmm12)
    e.vhsubps(xmm13, xmm8, xmm9)
    e.vmovups(m128_ptr(rdx + 80), xmm13)
    e.ret()
    e.finalize()
    _ = ccall(
        e.symbol('f'), ctypes.addressof(left), ctypes.addressof(right),
        ctypes.addressof(result),
    )

    assert list(result[:8]) == [3.0, 12.0, 8.0, 18.0, 48.0, 192.0, 30.0, 42.0]
    assert list(result[8:16]) == [-1.0, -4.0, -2.0, -4.0, -16.0, -64.0, -4.0, -4.0]
    assert list(result[16:20]) == [3.0, 12.0, 8.0, 18.0]
    assert list(result[20:24]) == [-1.0, -4.0, -2.0, -4.0]


def simd_dot_product() -> None:
    left = (ctypes.c_float * 8)(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
    right = (ctypes.c_float * 8)(3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0)
    result = (ctypes.c_float * 12)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm8, m256_ptr(rdi))
    e.vmovups(ymm9, m256_ptr(rsi))
    e.vdpps(ymm10, ymm8, ymm9, [True, True, False, True], [True, False, True, False])
    e.vmovups(m256_ptr(rdx), ymm10)
    e.vdpps(xmm11, xmm8, xmm9, [False, True, True, False], [False, True, False, True])
    e.vmovups(m128_ptr(rdx + 32), xmm11)
    e.ret()
    e.finalize()
    _ = ccall(
        e.symbol('f'), ctypes.addressof(left), ctypes.addressof(right),
        ctypes.addressof(result),
    )

    assert list(result[:8]) == [101.0, 0.0, 101.0, 0.0, 3696.0, 0.0, 3696.0, 0.0]
    assert list(result[8:12]) == [0.0, 38.0, 0.0, 38.0]

    failed = False
    try:
        e.vdpps(xmm0, xmm1, xmm2, [True] * 3, [False] * 4)
    except EmitterError:
        failed = True
    assert failed

    failed = False
    try:
        e.vdpps(ymm0, ymm1, ymm2, [False] * 4, [True] * 5)
    except EmitterError:
        failed = True
    assert failed


def simd_reciprocal() -> None:
    values = (ctypes.c_float * 8)(1.0, 4.0, 16.0, 64.0, 0.25, 0.0625, 256.0, 1024.0)
    result = (ctypes.c_float * 24)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm8, m256_ptr(rdi))
    e.vrcpps(ymm9, ymm8)
    e.vmovups(m256_ptr(rsi), ymm9)
    e.vrsqrtps(ymm10, ymm8)
    e.vmovups(m256_ptr(rsi + 32), ymm10)
    e.vrcpps(xmm11, xmm8)
    e.vmovups(m128_ptr(rsi + 64), xmm11)
    e.vrsqrtps(xmm12, xmm8)
    e.vmovups(m128_ptr(rsi + 80), xmm12)
    e.ret()
    e.finalize()
    _ = ccall(e.symbol('f'), ctypes.addressof(values), ctypes.addressof(result))

    assert math.isclose(result[0], 1.0, rel_tol=0.001)
    assert math.isclose(result[1], 0.25, rel_tol=0.001)
    assert math.isclose(result[2], 0.0625, rel_tol=0.001)
    assert math.isclose(result[3], 0.015625, rel_tol=0.001)
    assert math.isclose(result[4], 4.0, rel_tol=0.001)
    assert math.isclose(result[5], 16.0, rel_tol=0.001)
    assert math.isclose(result[6], 0.00390625, rel_tol=0.001)
    assert math.isclose(result[7], 0.0009765625, rel_tol=0.001)
    assert math.isclose(result[8], 1.0, rel_tol=0.001)
    assert math.isclose(result[9], 0.5, rel_tol=0.001)
    assert math.isclose(result[10], 0.25, rel_tol=0.001)
    assert math.isclose(result[11], 0.125, rel_tol=0.001)
    assert math.isclose(result[12], 2.0, rel_tol=0.001)
    assert math.isclose(result[13], 4.0, rel_tol=0.001)
    assert math.isclose(result[14], 0.0625, rel_tol=0.001)
    assert math.isclose(result[15], 0.03125, rel_tol=0.001)
    assert math.isclose(result[16], 1.0, rel_tol=0.001)
    assert math.isclose(result[17], 0.25, rel_tol=0.001)
    assert math.isclose(result[18], 0.0625, rel_tol=0.001)
    assert math.isclose(result[19], 0.015625, rel_tol=0.001)
    assert math.isclose(result[20], 1.0, rel_tol=0.001)
    assert math.isclose(result[21], 0.5, rel_tol=0.001)
    assert math.isclose(result[22], 0.25, rel_tol=0.001)
    assert math.isclose(result[23], 0.125, rel_tol=0.001)


def simd_blend() -> None:
    left = (ctypes.c_float * 8)(1, 2, 3, 4, 5, 6, 7, 8)
    right = (ctypes.c_float * 8)(11, 12, 13, 14, 15, 16, 17, 18)
    result = (ctypes.c_float * 36)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm8, m256_ptr(rdi))
    e.vmovups(ymm9, m256_ptr(rsi))
    e.vblendps(ymm12, ymm8, ymm9, [1, 2, 1, 2, 2, 1, 2, 1])
    e.vmovups(m256_ptr(rdx), ymm12)
    e.vmovups(xmm10, m128_ptr(rdi))
    e.vblendps(xmm11, xmm10, xmm9, [2, 1, 1, 2])
    e.vmovups(m128_ptr(rdx + 32), xmm11)
    e.vblendps(xmm11, xmm11, xmm9, [1, 1, 1, 1])
    e.vmovups(m128_ptr(rdx + 48), xmm11)
    e.vblendps(ymm9, ymm8, ymm9, [2, 2, 2, 2, 2, 2, 2, 2])
    e.vmovups(m256_ptr(rdx + 64), ymm9)
    e.vmovups(m256_ptr(rdx + 96), ymm8)
    e.vmovups(m128_ptr(rdx + 128), xmm10)
    e.vzeroupper()
    e.ret()
    e.finalize()
    try:
        _ = ccall(e.symbol('f'), ctypes.addressof(left), ctypes.addressof(right), ctypes.addressof(result))
        assert list(result[:8]) == [1, 12, 3, 14, 15, 6, 17, 8]
        assert list(result[8:12]) == [11, 2, 3, 14]
        assert list(result[12:16]) == [11, 2, 3, 14]
        assert list(result[16:24]) == [11, 12, 13, 14, 15, 16, 17, 18]
        assert list(result[24:32]) == [1, 2, 3, 4, 5, 6, 7, 8]
        assert list(result[32:36]) == [1, 2, 3, 4]
    finally:
        e.unmap()

    failed = False
    try:
        e.vblendps(xmm0, xmm1, xmm2, [1, 2, 1])
    except EmitterError:
        failed = True
    assert failed

    failed = False
    try:
        e.vblendps(ymm0, ymm1, ymm2, [1, 2, 1, 2])
    except EmitterError:
        failed = True
    assert failed

    failed = False
    try:
        e.vblendps(xmm0, xmm1, xmm2, [1, 2, 0, 1])
    except EmitterError:
        failed = True
    assert failed

    failed = False
    try:
        e.vblendps(ymm0, ymm1, ymm2, [1, 2, 1, 2, 1, 2, 3, 1])
    except EmitterError:
        failed = True
    assert failed


def simd_ptest() -> None:
    left = (ctypes.c_uint32 * 8)(1, 0, 0, 0, 2, 0, 0, 0)
    right = (ctypes.c_uint32 * 8)()
    e = Emitter()
    e.label('xmm_flags')
    e.vmovups(xmm8, m128_ptr(rdi))
    e.vmovups(xmm9, m128_ptr(rsi))
    e.vptest(xmm8, xmm9)
    e.setcc(EQ, al)
    e.setcc(LTU, dl)
    e.movzx(rax, al)
    e.movzx(rdx, dl)
    e.shl(rdx, 1)
    e.bitor(rax, rdx)
    e.ret()
    e.label('ymm_flags')
    e.vmovups(ymm8, m256_ptr(rdi))
    e.vmovups(ymm9, m256_ptr(rsi))
    e.vptest(ymm8, ymm9)
    e.setcc(EQ, al)
    e.setcc(LTU, dl)
    e.movzx(rax, al)
    e.movzx(rdx, dl)
    e.shl(rdx, 1)
    e.bitor(rax, rdx)
    e.vzeroupper()
    e.ret()
    e.finalize()
    try:
        assert ccall(e.symbol('xmm_flags'), ctypes.addressof(left), ctypes.addressof(right)) == 3
        right[0] = 1
        assert ccall(e.symbol('xmm_flags'), ctypes.addressof(left), ctypes.addressof(right)) == 2
        right[0] = 2
        assert ccall(e.symbol('xmm_flags'), ctypes.addressof(left), ctypes.addressof(right)) == 1
        right[0] = 3
        assert ccall(e.symbol('xmm_flags'), ctypes.addressof(left), ctypes.addressof(right)) == 0
        right[0] = 0
        assert ccall(e.symbol('ymm_flags'), ctypes.addressof(left), ctypes.addressof(right)) == 3
        right[4] = 2
        assert ccall(e.symbol('ymm_flags'), ctypes.addressof(left), ctypes.addressof(right)) == 2
        right[4] = 4
        assert ccall(e.symbol('ymm_flags'), ctypes.addressof(left), ctypes.addressof(right)) == 1
        right[4] = 6
        assert ccall(e.symbol('ymm_flags'), ctypes.addressof(left), ctypes.addressof(right)) == 0
    finally:
        e.unmap()


def simd_zero_upper() -> None:
    source = (ctypes.c_float * 8)(1, 2, 3, 4, 5, 6, 7, 8)
    result = (ctypes.c_float * 16)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm0, m256_ptr(rdi))
    e.vmovups(ymm15, m256_ptr(rdi))
    e.vzeroupper()
    e.vmovups(m256_ptr(rsi), ymm0)
    e.vmovups(m256_ptr(rsi + 32), ymm15)
    e.ret()
    e.finalize()
    try:
        _ = ccall(e.symbol('f'), ctypes.addressof(source), ctypes.addressof(result))
        assert list(result[:8]) == [1, 2, 3, 4, 0, 0, 0, 0]
        assert list(result[8:]) == [1, 2, 3, 4, 0, 0, 0, 0]
    finally:
        e.unmap()


def simd_shuffle() -> None:
    left = (ctypes.c_float * 8)(1, 2, 3, 4, 5, 6, 7, 8)
    right = (ctypes.c_float * 8)(9, 10, 11, 12, 13, 14, 15, 16)
    result = (ctypes.c_float * 12)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm8, m256_ptr(rdi))
    e.vmovups(ymm9, m256_ptr(rsi))
    e.vshufps(ymm10, ymm8, ymm9, [3, 0, 2, 1])
    e.vmovups(m256_ptr(rdx), ymm10)
    e.vshufps(xmm8, xmm8, xmm9, [0, 1, 2, 1])
    e.vmovups(m128_ptr(rdx + 32), xmm8)
    e.vzeroupper()
    e.ret()
    e.finalize()
    try:
        _ = ccall(e.symbol('f'), ctypes.addressof(left), ctypes.addressof(right), ctypes.addressof(result))
        assert list(result[:8]) == [4, 1, 11, 10, 8, 5, 15, 14]
        assert list(result[8:]) == [1, 2, 11, 10]
    finally:
        e.unmap()


def simd_permute_immediate() -> None:
    source = (ctypes.c_float * 8)(1, 2, 3, 4, 5, 6, 7, 8)
    result = (ctypes.c_float * 12)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm14, m256_ptr(rdi))
    e.vpermilps(ymm15, ymm14, [3, 2, 1, 0])
    e.vmovups(m256_ptr(rsi), ymm15)
    e.vpermilps(xmm14, xmm14, [2, 0, 2, 1])
    e.vmovups(m128_ptr(rsi + 32), xmm14)
    e.vzeroupper()
    e.ret()
    e.finalize()
    try:
        _ = ccall(e.symbol('f'), ctypes.addressof(source), ctypes.addressof(result))
        assert list(result[:8]) == [4, 3, 2, 1, 8, 7, 6, 5]
        assert list(result[8:]) == [3, 1, 3, 2]
    finally:
        e.unmap()


def simd_permute_variable() -> None:
    source = (ctypes.c_float * 8)(1, 2, 3, 4, 5, 6, 7, 8)
    control = (ctypes.c_uint32 * 8)(3, 0, 6, 0xffffffff, 1, 3, 0, 2)
    result = (ctypes.c_float * 12)()
    e = Emitter()
    e.label('f')
    e.vmovups(ymm8, m256_ptr(rdi))
    e.vmovups(ymm9, m256_ptr(rsi))
    e.vpermilps(ymm10, ymm8, ymm9)
    e.vmovups(m256_ptr(rdx), ymm10)
    e.vpermilps(xmm9, xmm8, xmm9)
    e.vmovups(m128_ptr(rdx + 32), xmm9)
    e.vzeroupper()
    e.ret()
    e.finalize()
    try:
        _ = ccall(e.symbol('f'), ctypes.addressof(source), ctypes.addressof(control), ctypes.addressof(result))
        assert list(result[:8]) == [4, 1, 3, 4, 6, 8, 5, 7]
        assert list(result[8:]) == [4, 1, 3, 4]
    finally:
        e.unmap()


def simd_shuffle_invalid() -> None:
    e = Emitter()
    failed = False
    try:
        e.vshufps(xmm0, xmm1, xmm2, [0, 1, 2])
    except EmitterError:
        failed = True
    assert failed

    failed = False
    try:
        e.vshufps(ymm0, ymm1, ymm2, [0, 1, 2, 4])
    except EmitterError:
        failed = True
    assert failed

    failed = False
    try:
        e.vshufps(xmm0, xmm1, xmm2, [-1, 0, 1, 2])
    except EmitterError:
        failed = True
    assert failed

    failed = False
    try:
        e.vpermilps(xmm0, xmm1, [0, 1, 2, 3, 0])
    except EmitterError:
        failed = True
    assert failed

    failed = False
    try:
        e.vpermilps(ymm0, ymm1, [0, 1, 2, 4])
    except EmitterError:
        failed = True
    assert failed

    failed = False
    try:
        e.vpermilps(xmm0, xmm1, [-1, 0, 1, 2])
    except EmitterError:
        failed = True
    assert failed

    e.set_section(Section.DATA)
    failed = False
    try:
        e.vshufps(xmm0, xmm1, xmm2, [0, 1, 2, 3])
    except EmitterError:
        failed = True
    assert failed

    failed = False
    try:
        e.vpermilps(xmm0, xmm1, [0, 1, 2, 3])
    except EmitterError:
        failed = True
    assert failed

    failed = False
    try:
        e.vpermilps(ymm0, ymm1, ymm2)
    except EmitterError:
        failed = True
    assert failed


def test_simd() -> None:
    simd_move()
    simd_arithmetic()
    simd_addsub()
    simd_min_max_sqrt()
    simd_bitwise()
    simd_round()
    simd_compare()
    simd_compare_pseudo()
    simd_horizontal()
    simd_dot_product()
    simd_reciprocal()
    simd_blend()
    simd_ptest()
    simd_zero_upper()
    simd_shuffle()
    simd_permute_immediate()
    simd_permute_variable()
    simd_shuffle_invalid()
