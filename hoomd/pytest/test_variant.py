# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

import hoomd
import hoomd.variant


def test_constant():
    """ Test construction and properties of variant.constant
    """

    a = hoomd.variant.ConstantVariant(10.0)

    assert a.value == 10.0

    a.value = 0.125


def test_constant_eval():
    a = hoomd.variant.ConstantVariant(10.0)

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
