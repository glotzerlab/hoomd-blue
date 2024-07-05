# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test the Trigger classes."""
import itertools
from inspect import isclass
import pickle

import pytest

import hoomd
import hoomd.trigger


class CustomTrigger(hoomd.trigger.Trigger):

    def __init__(self):
        hoomd.trigger.Trigger.__init__(self)

    def compute(self, timestep):
        return (timestep**(1 / 2)).is_integer()

    def __str__(self):
        return "CustomTrigger()"

    def __eq__(self, other):
        return isinstance(other, CustomTrigger)


# List of trigger classes
_classes = [
    hoomd.trigger.Periodic, hoomd.trigger.Before, hoomd.trigger.After,
    hoomd.trigger.On, hoomd.trigger.Not, hoomd.trigger.And, hoomd.trigger.Or,
    CustomTrigger
]

# List of kwargs for the class constructors
_kwargs = [{
    'period': (456, 10000000000),
    'phase': (18, 60000000000)
}, {
    'timestep': (100, 10000000000)
}, {
    'timestep': (100, 10000000000)
}, {
    'timestep': (100, 10000000000)
}, {
    'trigger': (hoomd.trigger.Periodic(10, 1), hoomd.trigger.Before(100))
}, {
    'triggers': ((hoomd.trigger.Periodic(10, 1), hoomd.trigger.Before(100)),
                 (hoomd.trigger.After(100), hoomd.trigger.On(101)))
}, {
    'triggers': ((hoomd.trigger.Periodic(10, 1), hoomd.trigger.Before(100)),
                 (hoomd.trigger.After(100), hoomd.trigger.On(101)))
}, {}]


def _cartesian(grid):
    for values in itertools.product(*grid.values()):
        yield dict(zip(grid.keys(), values))


def _test_name(arg):
    if not isinstance(arg, hoomd.trigger.Trigger):
        return None
    else:
        if isclass(arg):
            return arg.__name__
        else:
            return arg.__class__.__name__


# Go over all class and constructor pairs
@pytest.mark.parametrize('cls, kwargs',
                         ((cls, kwarg)
                          for cls, kwargs in zip(_classes, _kwargs)
                          for kwarg in _cartesian(kwargs)),
                         ids=_test_name)
def test_properties(cls, kwargs):
    instance = cls(**kwargs)
    for key, value in kwargs.items():
        assert getattr(instance, key) == value


_strings_beginning = ("hoomd.trigger.Periodic(", "hoomd.trigger.Before(",
                      "hoomd.trigger.After(", "hoomd.trigger.On(",
                      "hoomd.trigger.Not(", "hoomd.trigger.And(",
                      "hoomd.trigger.Or(", "CustomTrigger()")


# Trigger instanace for the first arguments in _kwargs
def triggers():
    _single_kwargs = [next(_cartesian(kwargs)) for kwargs in _kwargs]
    return (cls(**kwargs) for cls, kwargs in zip(_classes, _single_kwargs))


@pytest.mark.parametrize('trigger, instance_string',
                         zip(triggers(), _strings_beginning),
                         ids=_test_name)
def test_str(trigger, instance_string):
    assert str(trigger).startswith(instance_string)


_eval_funcs = [
    lambda x: (x - 18) % 456 == 0,  # periodic
    lambda x: x < 100,  # before
    lambda x: x > 100,  # after
    lambda x: x == 100,  # on
    lambda x: not (x - 1) % 10 == 0,  # not
    lambda x: (x - 1) % 10 == 0 and x < 100,  # and
    lambda x: (x - 1) % 10 == 0 or x < 100,  # or
    lambda x: (x**(1 / 2)).is_integer()
]


@pytest.mark.parametrize('trigger, eval_func',
                         zip(triggers(), _eval_funcs),
                         ids=_test_name)
def test_eval(trigger, eval_func):
    for i in range(10000):
        assert trigger(i) == eval_func(i)
    # Test values greater than 2^32
    for i in range(10000000000, 10000010000):
        assert trigger(i) == eval_func(i)


@pytest.mark.parametrize('trigger', triggers(), ids=_test_name)
def test_pickling(trigger):
    pkled_trigger = pickle.loads(pickle.dumps(trigger))
    assert trigger == pkled_trigger


def test_custom():
    c = CustomTrigger()

    # test that the custom trigger can be called from c++
    assert hoomd._hoomd._test_trigger_call(c, 0)
    assert not hoomd._hoomd._test_trigger_call(c, 250000000001)
