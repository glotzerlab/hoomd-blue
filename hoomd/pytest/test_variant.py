# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

import numpy
import hoomd
import hoomd.variant


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

    c = CustomVariant()

    # test that the custom variant can be called from c++

    for i in range(10000):
        assert hoomd._hoomd._test_variant_call(c, i) == float(i)**(1 / 2)

    for i in range(10000000000, 10000010000):
        assert hoomd._hoomd._test_variant_call(c, i) == float(i)**(1 / 2)


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

