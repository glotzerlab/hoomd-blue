# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test the Trigger classes."""

import hoomd
import hoomd.trigger


def test_periodic_properties():
    """Test construction and properties of Periodic."""
    a = hoomd.trigger.Periodic(123)

    assert a.period == 123
    assert a.phase == 0

    a.period = 10000000000

    assert a.period == 10000000000
    assert a.phase == 0

    a.phase = 6000000000

    assert a.period == 10000000000
    assert a.phase == 6000000000

    b = hoomd.trigger.Periodic(phase=3, period=456)

    assert b.period == 456
    assert b.phase == 3


def test_periodic_str():
    """Test the Periodic __str__ method."""
    b = hoomd.trigger.Periodic(phase=456, period=3)
    assert str(b) == "hoomd.trigger.Periodic(period=3, phase=456)"


def test_periodic_eval():
    """Test the Periodic trigger evaluation."""
    a = hoomd.trigger.Periodic(period=456, phase=18)

    for i in range(10000):
        assert a(i) == ((i - 18) % 456 == 0)

    # Test values greater than 2^32
    for i in range(10000000000, 10000010000):
        assert a(i) == ((i - 18) % 456 == 0)

    # Test trigger with values greater than 2^32
    b = hoomd.trigger.Periodic(period=10000000000, phase=6000000000)

    assert b(6000000000)
    assert not b(6000000001)
    assert b(16000000000)
    assert not b(16000000001)


def test_before_str():
    """Test the Before __str__ method."""
    a = hoomd.trigger.Before(1000)
    assert str(a) == "hoomd.trigger.Before(timestep=1000)"


def test_before_eval():
    """Test the Before trigger."""
    a = hoomd.trigger.Before(1000)

    assert all(a(i) for i in range(1000))
    assert not any(a(i) for i in range(1000, 10000))

    # tests for values greater than 2^32
    assert not any(a(i) for i in range(10000000000, 10000010000))
    a = hoomd.trigger.Before(10000000000)
    assert all(a(i) for i in range(9999990000, 10000000000))
    assert not any(a(i) for i in range(10000000000, 10000010000))


def test_after_str():
    """Test the After __str__ method."""
    a = hoomd.trigger.After(1000)
    assert str(a) == "hoomd.trigger.After(timestep=1000)"


def test_after_eval():
    """Test the After trigger."""
    a = hoomd.trigger.After(1000)

    assert not any(a(i) for i in range(1001))
    assert all(a(i) for i in range(1001, 10000))

    # tests for values greater than 2^32
    assert all(a(i) for i in range(10000000000, 10000010000))
    a = hoomd.trigger.After(10000000000)
    assert not any(a(i) for i in range(9999990000, 10000000001))
    assert all(a(i) for i in range(10000000001, 10000010000))


def test_on_str():
    """Test the On __str__ method."""
    a = hoomd.trigger.On(1000)
    assert str(a) == "hoomd.trigger.On(timestep=1000)"


def test_on_eval():
    """Test the On trigger."""
    a = hoomd.trigger.On(1000)

    assert not any(a(i) for i in range(1000))
    assert a(1000)
    assert not any(a(i) for i in range(1001, 10000))

    # tests for values greater than 2^32
    assert not any(a(i) for i in range(10000000000, 10000010000))
    a = hoomd.trigger.On(10000000000)
    assert not any(a(i) for i in range(9999990000, 10000000000))
    assert a(10000000000)
    assert not any(a(i) for i in range(10000000001, 10000010000))


def test_not_str():
    """Test the Not __str__ method."""
    a = hoomd.trigger.Not(hoomd.trigger.After(1000))
    assert str(a).startswith("hoomd.trigger.Not(")


def test_not_eval():
    """Test the Not Trigger."""
    a = hoomd.trigger.Not(hoomd.trigger.After(1000))
    assert str(a).startswith("hoomd.trigger.Not(")

    assert all(a(i) for i in range(1001))
    assert not any(a(i) for i in range(1001, 10000))

    # tests for values greater than 2^32
    assert not any(a(i) for i in range(10000000000, 10000010000))
    a = hoomd.trigger.Not(hoomd.trigger.After(10000000000))
    assert all(a(i) for i in range(9999990000, 10000000001))
    assert not any(a(i) for i in range(10000000001, 10000010000))


def test_and_str():
    """Test the And __str__ method."""
    a = hoomd.trigger.And(
        [hoomd.trigger.Before(1000),
         hoomd.trigger.After(1000)])
    assert str(a).startswith("hoomd.trigger.And(")


def test_and_eval():
    """Test the And trigger."""
    a = hoomd.trigger.And(
        [hoomd.trigger.Before(1000),
         hoomd.trigger.After(1000)])

    assert not any(a(i) for i in range(1000))
    assert not any(a(i) for i in range(1000, 10000))

    # tests for values greater than 2^32
    assert not any(a(i) for i in range(10000000000, 10000010000))
    a = hoomd.trigger.And(
        [hoomd.trigger.Before(10000000000),
         hoomd.trigger.After(10000000000)])
    assert not any(a(i) for i in range(9999990000, 10000010000))


def test_or_str():
    """Test the Or __str__ method."""
    a = hoomd.trigger.Or([
        hoomd.trigger.Before(1000),
        hoomd.trigger.On(1000),
        hoomd.trigger.After(1000)
    ])
    assert str(a).startswith("hoomd.trigger.Or(")


def test_or_eval():
    """Test the Or trigger."""
    a = hoomd.trigger.Or([
        hoomd.trigger.Before(1000),
        hoomd.trigger.On(1000),
        hoomd.trigger.After(1000)
    ])

    assert all(a(i) for i in range(10000))

    # tests for values greater than 2^32
    assert all(a(i) for i in range(10000000000, 10000010000))
    a = hoomd.trigger.Or([
        hoomd.trigger.Before(10000000000),
        hoomd.trigger.On(10000000000),
        hoomd.trigger.After(10000000000)
    ])
    assert all(a(i) for i in range(9999990000, 10000010000))


def test_custom():
    """Test CustomTrigger."""
    class CustomTrigger(hoomd.trigger.Trigger):

        def __init__(self):
            hoomd.trigger.Trigger.__init__(self)

        def compute(self, timestep):
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
