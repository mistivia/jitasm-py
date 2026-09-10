# Copyright (c) 2026, Mistivia <i@mistivia.com>
# Distributed under the terms of the GPLv3

import jitasm.x86_64 as x86
from jitasm.x86_64 import Emitter, Section


def test_cpu_features() -> None:
    e = Emitter()
    e.cpuid()
    assert e.text == b'\x0f\xa2'

    e.set_section(Section.DATA)
    failed = False
    try:
        e.cpuid()
    except x86.EmitterError:
        failed = True
    assert failed

    x86._init_cpu_features()
    assert type(x86.cpu_features.avx) is bool
    assert type(x86.cpu_features.avx2) is bool
    assert type(x86.cpu_features.fma) is bool
