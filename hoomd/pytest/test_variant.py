# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from inspect import isclass
import itertools
import pickle

import numpy as np
from hoomd.conftest import pickling_check
import hoomd
import hoomd.variant
import pytest

_classes = [
    hoomd.variant.Constant, hoomd.variant.Ramp, hoomd.variant.Cycle,
    hoomd.variant.Power
]

_test_kwargs = [
    # Constant: first args value=1
    {
        'value': np.linspace(1, 10, 3)
    },
    # Ramp: first args A=1, B=3, t_start=0, t_ramp=10
    {
        'A': np.linspace(1, 10, 3),
        'B': np.linspace(3, 10, 3),
        't_start': (0, 10, 10000000000),
        't_ramp': (10, 20, 2000000000000)
    },
    # Cycle: first args A=2, B=5, t_start=0, t_A=10, t_AB=15, t_B=10, t_BA_20
    {
        'A': np.linspace(2, 10, 3),
        'B': np.linspace(5, 10, 3),
        't_start': (0, 10, 10000000000),
        't_A': (10, 20, 2000000000000),
        't_AB': (15, 30, 40000000000),
        't_B': (10, 20, 2000000000000),
        't_BA': (20, 40, 560000000000)
    },
    # Power: first args A=1, B=10, t_start=0, t_ramp=10
    {
        'A': np.linspace(1, 10, 3),
        'B': np.linspace(10, 100, 3),
        'power': np.linspace(2, 5, 3),
        't_start': (0, 10, 10000000000),
        't_ramp': (10, 20, 2000000000000)
    }
]


def _to_kwargs(specs):
    """Take a dictionary of iterables into a generator of dicts."""
    for value in zip(*specs.values()):
        yield dict(zip(specs.keys(), value))


def _test_id(value):
    if isinstance(value, hoomd.variant.Variant):
        if isclass(value):
            return value.__name__
        else:
            return value.__class__.__name__
    else:
        return None


@pytest.mark.parametrize('cls, kwargs',
                         ((cls, kwarg)
                          for cls, kwargs in zip(_classes, _test_kwargs)
                          for kwarg in _to_kwargs(kwargs)),
                         ids=_test_id)
def test_construction(cls, kwargs):
    variant = cls(**kwargs)
    for key, value in kwargs.items():
        assert getattr(variant, key) == value


_expected_min_max = [
    (1., 1.),
    (1., 3.),
    (2., 5.),
    (1., 10.),
]

_single_kwargs = [next(_to_kwargs(kwargs)) for kwargs in _test_kwargs]


def variants():
    return (cls(**kwargs) for cls, kwargs in zip(_classes, _single_kwargs))


@pytest.mark.parametrize('variant, expected_min_max',
                         zip(variants(), _expected_min_max),
                         ids=_test_id)
def test_min_max(variant, expected_min_max):
    assert np.isclose(variant.min, expected_min_max[0])
    assert np.isclose(variant.max, expected_min_max[1])
    assert np.allclose(variant.range, expected_min_max)


@pytest.mark.parametrize('variant, attrs',
                         ((variant, kwarg)
                          for variant, kwargs in zip(variants(), _test_kwargs)
                          for kwarg in _to_kwargs(kwargs)),
                         ids=_test_id)
def test_setattr(variant, attrs):
    for attr, value in attrs.items():
        setattr(variant, attr, value)
        assert getattr(variant, attr) == value


def constant_eval(value):

    def expected_value(timestep):
        return value

    return expected_value


def power_eval(A, B, power, t_start, t_ramp):

    def expected_value(timestep):
        if timestep < t_start:
            return A
        elif timestep < t_start + t_ramp:
            inv_a, inv_b = (A**(1 / power)), (B**(1 / power))
            frac = (timestep - t_start) / t_ramp
            return ((inv_b * frac) + ((1 - frac) * inv_a))**power
        else:
            return B

    return expected_value


def ramp_eval(A, B, t_start, t_ramp):

    def expected_value(timestep):
        if timestep < t_start:
            return A
        elif timestep < t_start + t_ramp:
            frac = (timestep - t_start) / t_ramp
            return (B * frac) + ((1 - frac) * A)
        else:
            return B

    return expected_value


def cycle_eval(A, B, t_start, t_A, t_AB, t_BA, t_B):
    period = t_A + t_B + t_AB + t_BA

    def expected_value(timestep):
        delta = (timestep - t_start) % period
        if timestep < t_start or delta < t_A:
            return A
        elif delta < t_A + t_AB:
            scale = (delta - t_A) / t_AB
            return (scale * B) + ((1 - scale) * A)
        elif delta < t_A + t_AB + t_B:
            return B
        else:
            scale = (delta - (t_A + t_AB + t_B)) / t_BA
            return (scale * A) + ((1 - scale) * B)

    return expected_value


_eval_constructors = [constant_eval, ramp_eval, cycle_eval, power_eval]


@pytest.mark.parametrize('variant, evaluator, kwargs',
                         ((variant, evaluator, kwarg)
                          for variant, evaluator, kwargs in zip(
                              variants(), _eval_constructors, _test_kwargs)
                          for kwarg in _to_kwargs(kwargs)),
                         ids=_test_id)
def test_evaulation(variant, evaluator, kwargs):
    for attr, value in kwargs.items():
        setattr(variant, attr, value)
    eval_func = evaluator(**kwargs)
    for i in range(1000):
        assert np.isclose(eval_func(i), variant(i))
    for i in range(1000000000000, 1000000001000):
        assert np.isclose(eval_func(i), variant(i))


class CustomVariant(hoomd.variant.Variant):

    def __init__(self):
        hoomd.variant.Variant.__init__(self)
        self._a = 1

    def __call__(self, timestep):
        return (float(timestep)**(1 / 2))

    def _min(self):
        return 0.0

    def _max(self):
        return 1.0

    def __eq__(self, other):
        return isinstance(other, type(self))


@pytest.mark.parametrize(
    'variant',
    (variant for variant in itertools.chain(variants(), (CustomVariant(),))),
    ids=_test_id)
def test_pickling(variant):
    # This also tests equality of objects with the same attributes
    pickling_check(variant)


def test_custom():
    c = CustomVariant()

    # test that the custom variant can be called from c++

    for i in range(10000):
        assert hoomd._hoomd._test_variant_call(c, i) == float(i)**(1 / 2)

    for i in range(10000000000, 10000010000):
        assert hoomd._hoomd._test_variant_call(c, i) == float(i)**(1 / 2)

    assert hoomd._hoomd._test_variant_min(c) == 0.0
    assert hoomd._hoomd._test_variant_max(c) == 1.0

    pkled_variant = pickle.loads(pickle.dumps(c))
    for i in range(0, 10000, 100):
        assert (hoomd._hoomd._test_variant_call(pkled_variant,
                                                i) == float(i)**(1 / 2))
