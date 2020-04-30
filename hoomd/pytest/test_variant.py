# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

import numpy
import hoomd
import hoomd.variant
import pytest


def test_constant():
    """ Test construction and properties of variant.constant
    """

    a = hoomd.variant.Constant(10.0)

    assert a.value == 10.0

    a.value = 0.125


def test_constant_eval():
    a = hoomd.variant.Constant(10.0)

    for i in range(10000):
        assert a(i) == 10.0

    for i in range(10000000000, 10000010000):
        assert a(i) == 10.0


def test_custom():
    class CustomVariant(hoomd.variant.Variant):
        def __init__(self):
            hoomd.variant.Variant.__init__(self)

        def __call__(self, timestep):
            return (float(timestep)**(1 / 2))

        def _min(self):
            return 0.0

        def _max(self):
            return 1.0

    c = CustomVariant()

    # test that the custom variant can be called from c++

    for i in range(10000):
        assert hoomd._hoomd._test_variant_call(c, i) == float(i)**(1 / 2)

    for i in range(10000000000, 10000010000):
        assert hoomd._hoomd._test_variant_call(c, i) == float(i)**(1 / 2)

    assert hoomd._hoomd._test_variant_min(c) == 0.0
    assert hoomd._hoomd._test_variant_max(c) == 1.0


def test_ramp():
    a = hoomd.variant.Ramp(1.0, 11.0, 100, 10000)

    for i in range(100):
        assert a(i) == 1.0

    assert a(100) == 1.0
    numpy.testing.assert_allclose(a(2600), 3.5)
    numpy.testing.assert_allclose(a(5100), 6.0)
    numpy.testing.assert_allclose(a(7600), 8.5)
    assert a(10100) == 11.0

    for i in range(10000000000, 10000010000):
        assert a(i) == 11.0


def test_ramp_properties():
    a = hoomd.variant.Ramp(1.0, 11.0, 100, 10000)
    assert a.A == 1.0
    assert a.B == 11.0
    assert a.t_start == 100
    assert a.t_ramp == 10000

    a.A = 100
    assert a.A == 100.0
    assert a(0) == 100.0

    a.B = -25
    assert a.B == -25.0
    assert a(10100) == -25.0

    a.t_start = 1000
    assert a.t_start == 1000
    assert a(1000) == 100.0
    assert a(1001) < 100.0

    a.t_ramp = int(1e6)
    assert a.t_ramp == int(1e6)
    assert a(1001000) == -25.0
    assert a(1000999) > -25.0

    with pytest.raises(ValueError):
        a.t_ramp = int(2**53 + 1)


def test_cycle():
    A = 0.0
    B = 10.0
    t_start = 100
    t_A = 200
    t_AB = 400
    t_B = 300
    t_BA = 100

    a = hoomd.variant.Cycle(A=A,
                            B=B,
                            t_start=t_start,
                            t_A=t_A,
                            t_AB=t_AB,
                            t_B=t_B,
                            t_BA=t_BA)

    # t_start
    for i in range(t_start):
        assert a(i) == A

    period = t_A + t_AB + t_B + t_BA
    for cycle in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 10000000000]:
        offset = period * cycle + t_start

        # t_A
        for i in range(offset, offset + t_A):
            assert a(i) == A

        # t_AB ramp
        numpy.testing.assert_allclose(a(offset + t_A + t_AB // 2), (A + B) / 2)

        # t_B
        for i in range(offset + t_A + t_AB, offset + t_A + t_AB + t_B):
            assert a(i) == B

        # t_BA ramp
        numpy.testing.assert_allclose(a(offset + t_A + t_AB + t_B + t_BA // 2),
                                      (A + B) / 2)


def test_cycle_properties():
    a = hoomd.variant.Cycle(A=1.0,
                            B=12.0,
                            t_start=100,
                            t_A=200,
                            t_AB=300,
                            t_B=400,
                            t_BA=500)
    assert a.A == 1.0
    assert a.B == 12.0
    assert a.t_start == 100
    assert a.t_A == 200
    assert a.t_AB == 300
    assert a.t_B == 400
    assert a.t_BA == 500

    a.A = 100
    assert a.A == 100.0
    assert a(0) == 100.0

    a.B = -25
    assert a.B == -25.0
    assert a(600) == -25.0

    a.t_start = 1000
    assert a.t_start == 1000
    assert a(1000) == 100.0
    assert a(1201) < 100.0

    a.t_A = 400
    assert a.t_A == 400
    assert a(1400) == 100.0
    assert a(1401) < 100.0

    a.t_AB = 600
    assert a.t_AB == int(600)
    assert a(2000) == -25.0
    assert a(1999) > -25.0

    a.t_B = 1000
    assert a.t_B == 1000
    assert a(3000) == -25.0
    assert a(3001) > -25.0

    a.t_BA = 10000
    assert a.t_BA == 10000
    assert a(13000) == 100.0
    assert a(12999) < 100.0

    with pytest.raises(ValueError):
        a.t_AB = int(2**53 + 1)

    with pytest.raises(ValueError):
        a.t_BA = int(2**53 + 1)
