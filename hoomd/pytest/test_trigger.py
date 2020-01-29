# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

import hoomd
import hoomd.triggers


def test_periodic_properties():
    """ Test construction and properties of PeriodicTrigger
    """

    a = hoomd.triggers.PeriodicTrigger(123)

    assert a.period == 123
    assert a.phase == 0

    a.period = 10000000000

    assert a.period == 10000000000
    assert a.phase == 0

    a.phase = 6000000000

    assert a.period == 10000000000
    assert a.phase == 6000000000

    b = hoomd.triggers.PeriodicTrigger(phase=3, period=456)

    assert b.period == 456
    assert b.phase == 3


def test_periodic_eval():
    a = hoomd.triggers.PeriodicTrigger(period=456, phase=18)

    for i in range(10000):
        assert a(i) == ((i - 18) % 456 == 0)

    for i in range(10000000000, 10000010000):
        assert a(i) == ((i - 18) % 456 == 0)

    b = hoomd.triggers.PeriodicTrigger(period=10000000000, phase=6000000000)

    assert b(6000000000)
    assert not b(6000000001)
    assert b(16000000000)
    assert not b(16000000001)


def test_custom():
    class CustomTrigger(hoomd.triggers.Trigger):
        def __init__(self):
            hoomd.triggers.Trigger.__init__(self)

        def __call__(self, timestep):
            return (timestep**(1 / 2)).is_integer()

    c = CustomTrigger()

    # test that the custom trigger can be called from c++
    assert hoomd._hoomd._test_trigger_call(c, 0)
    assert hoomd._hoomd._test_trigger_call(c, 1)
    assert not hoomd._hoomd._test_trigger_call(c, 2)
    assert not hoomd._hoomd._test_trigger_call(c, 3)
    assert hoomd._hoomd._test_trigger_call(c, 4)
    assert not hoomd._hoomd._test_trigger_call(c, 5)
    assert not hoomd._hoomd._test_trigger_call(c, 6)
    assert not hoomd._hoomd._test_trigger_call(c, 7)
    assert not hoomd._hoomd._test_trigger_call(c, 8)
    assert hoomd._hoomd._test_trigger_call(c, 9)
    assert hoomd._hoomd._test_trigger_call(c, 250000000000)
    assert not hoomd._hoomd._test_trigger_call(c, 250000000001)
