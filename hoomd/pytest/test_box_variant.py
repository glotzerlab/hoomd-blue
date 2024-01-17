# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import copy
from inspect import isclass
import itertools
import pickle

import numpy as np
import numpy.testing as npt
from hoomd.conftest import pickling_check
import hoomd
import hoomd.variant
import pytest


box_to_array = lambda box: np.array([box.Lx, box.Ly, box.Lz, box.xy, box.xz, box.yz])
_test_box1 = hoomd.Box(10, 50, 20, 0.2, 0.4, 0.6)
_test_box2 = hoomd.Box(16, 25, 36, 0.1, 0.2, 0.3)
_test_scalar_variant1 = hoomd.variant.Ramp(0, 1, 100, 200)
_test_scalar_variant2 = hoomd.variant.Ramp(0, 1, 10, 30)

valid_constructors = [
    (hoomd.variant.box.Constant, {'box': _test_box1}),
    (hoomd.variant.box.Interpolate, {'initial_box': _test_box1, 'final_box': _test_box2, 'variant': _test_scalar_variant1}),
    (hoomd.variant.box.InverseVolumeRamp, {'initial_box': _test_box2, 'final_volume': 1000, 't_start': 10, 't_ramp': 50}),
]

# variant: dict(attr: [val1, val2,...])
valid_attrs = [
    (hoomd.variant.box.Constant, {'box': [_test_box1, _test_box2]}),
    (hoomd.variant.box.Interpolate, {'initial_box': [_test_box1, _test_box2], 'final_box': [_test_box2, _test_box1], 'variant': [_test_scalar_variant1, _test_scalar_variant2]}),
    (hoomd.variant.box.InverseVolumeRamp, {'initial_box': [_test_box1, _test_box2], 'final_volume': [1000, 300], 't_start': [0, 10], 't_ramp': [10, 50, 100]}),
]

@pytest.mark.parametrize('cls, kwargs', valid_constructors)
def test_construction(cls, kwargs):
    variant = cls(**kwargs)
    for key, value in kwargs.items():
        assert getattr(variant, key) == value

@pytest.mark.parametrize('cls, attrs', valid_attrs)
def test_setattr(cls, attrs):
    kwargs = {k: v[0] for k, v in attrs.items()}
    variant = cls(**kwargs)
    new_attrs = [(k, v[1]) for k, v in attrs.items()]
    for attr, value in new_attrs:
        setattr(variant, attr, value)
        assert getattr(variant, attr) == value

class VolumeRampBoxVariant(hoomd.variant.box.BoxVariant):

    def __init__(self, box1, final_volume, t_start, t_ramp):
        self._initial_volume = box1.volume
        self._box1 = box1
        self._volume_variant = hoomd.variant.Ramp(box1.volume, final_volume, t_start, t_ramp)
        hoomd.variant.box.BoxVariant.__init__(self)

    def __call__(self, timestep):
        current_volume = self._volume_variant(timestep)
        scale_L = (current_volume / self._initial_volume)**(1/3)
        return np.concatenate((self._box1.L * scale_L, self._box1.tilts))

    def __eq__(self, other):
        return isinstance(other, type(self))


def test_custom():
    # test that the custom variant can be called from c++ and that it returns
    # the expected values

    final_volume = _test_box1.volume * 2
    test_box = hoomd.Box(_test_box1.Lx, _test_box1.Ly, _test_box1.Lz, _test_box1.xy, _test_box1.xz, _test_box1.yz)
    custom_variant = VolumeRampBoxVariant(_test_box1, final_volume, 100, 100)

    box_t = lambda t: hoomd._hoomd._test_vector_variant_box_call(custom_variant, t)
    for _t, _f in (
        (0, 0), (42, 0), (100, 0), (101, 0.01), (150, 0.5), (175, 0.75),
        (199, 0.99), (200, 1.0), (250, 1.0), (123456789, 1.0)):
        test_box.volume = (1 - _f) * _test_box1.volume + _f * final_volume
        npt.assert_allclose(box_t(_t), box_to_array(test_box))

def test_interpolate_evaluation():
    t_start = 50
    t_ramp = 100
    scalar_variant = hoomd.variant.Ramp(0, 1, t_start, t_ramp)
    box_variant = hoomd.variant.box.Interpolate(_test_box1, _test_box2, scalar_variant)
    npt.assert_allclose(box_variant(0), box_to_array(_test_box1))
    npt.assert_allclose(box_variant(25), box_to_array(_test_box1))
    npt.assert_allclose(box_variant(t_start), box_to_array(_test_box1))

    npt.assert_allclose(
        box_variant(51),
        0.99 * box_to_array(_test_box1) + 0.01 * box_to_array(_test_box2))
    npt.assert_allclose(
        box_variant(75),
        0.75 * box_to_array(_test_box1) + 0.25 * box_to_array(_test_box2))
    npt.assert_allclose(
        box_variant(100),
        0.5 * box_to_array(_test_box1) + 0.5 * box_to_array(_test_box2))
    npt.assert_allclose(
        box_variant(125),
        0.25 * box_to_array(_test_box1) + 0.75 * box_to_array(_test_box2))
    npt.assert_allclose(
        box_variant(149),
        0.01 * box_to_array(_test_box1) + 0.99 * box_to_array(_test_box2))

    npt.assert_allclose(box_variant(t_start+t_ramp), box_to_array(_test_box2))
    npt.assert_allclose(box_variant(t_start+t_ramp+100), box_to_array(_test_box2))
    npt.assert_allclose(box_variant(t_start+t_ramp+1000000), box_to_array(_test_box2))

def test_inverse_volume_ramp_evaluation():
    box1 = hoomd.Box(10, 10, 10, 0.1, 0.2, 0.3)
    final_volume = 500
    t_start = 10
    t_ramp = 100
    variant = hoomd.variant.box.InverseVolumeRamp(box1, final_volume, t_start, t_ramp)
    volume = lambda variant, timestep: hoomd.Box(*variant(timestep)).volume
    assert volume(variant, 0) == box1.volume
    assert volume(variant, 5) == box1.volume
    assert volume(variant, 10) == box1.volume
    assert volume(variant, 11) != box1.volume
    npt.assert_allclose(volume(variant, 35), 1 / (0.75 / box1.volume + 0.25 / final_volume))
    npt.assert_allclose(volume(variant, 60), 1 / (0.5 * (1 / box1.volume + 1 / final_volume)))
    npt.assert_allclose(volume(variant, 85), 1 / (0.25 / box1.volume + 0.75 / final_volume))
    npt.assert_allclose(volume(variant, 110), final_volume)
    npt.assert_allclose(volume(variant, 1010), final_volume)
    # make sure tilts don't change
    npt.assert_allclose(box1.tilts, variant(0)[3:])
    npt.assert_allclose(box1.tilts, variant(5)[3:])
    npt.assert_allclose(box1.tilts, variant(25)[3:])
    npt.assert_allclose(box1.tilts, variant(125)[3:])
    npt.assert_allclose(box1.tilts, variant(625)[3:])
