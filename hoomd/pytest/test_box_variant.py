# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import numpy.testing as npt
import hoomd
import hoomd.variant
import pytest


def box_to_array(box):
    return np.array([box.Lx, box.Ly, box.Lz, box.xy, box.xz, box.yz])


test_box1 = hoomd.Box(10, 50, 20, 0.2, 0.4, 0.6)
test_box2 = hoomd.Box(16, 25, 36, 0.1, 0.2, 0.3)
scalar_variant1 = hoomd.variant.Ramp(0, 1, 100, 200)
scalar_variant2 = hoomd.variant.Ramp(0, 1, 10, 30)

valid_constructors = [
    (hoomd.variant.box.Constant, {
        'box': test_box1
    }),
    (hoomd.variant.box.Interpolate, {
        'initial_box': test_box1,
        'final_box': test_box2,
        'variant': scalar_variant1
    }),
    (hoomd.variant.box.InverseVolumeRamp, {
        'initial_box': test_box2,
        'final_volume': 1000,
        't_start': 10,
        't_ramp': 50
    }),
]

# variant: dict(attr: [val1, val2,...])
valid_attrs = [
    (hoomd.variant.box.Constant, {
        'box': [test_box1, test_box2]
    }),
    (hoomd.variant.box.Interpolate, {
        'initial_box': [test_box1, test_box2],
        'final_box': [test_box2, test_box1],
        'variant': [scalar_variant1, scalar_variant2]
    }),
    (hoomd.variant.box.InverseVolumeRamp, {
        'initial_box': [test_box1, test_box2],
        'final_volume': [1000, 300],
        't_start': [0, 10],
        't_ramp': [10, 50, 100]
    }),
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
        self._volume_variant = hoomd.variant.Ramp(box1.volume, final_volume,
                                                  t_start, t_ramp)
        hoomd.variant.box.BoxVariant.__init__(self)

    def __call__(self, timestep):
        current_volume = self._volume_variant(timestep)
        scale_L = (current_volume / self._initial_volume)**(1 / 3)
        return np.concatenate((self._box1.L * scale_L, self._box1.tilts))

    def __eq__(self, other):
        return isinstance(other, type(self))


def test_custom():
    # test that the custom variant can be called from c++ and that it returns
    # the expected values

    final_volume = test_box1.volume * 2
    test_box = hoomd.Box(test_box1.Lx, test_box1.Ly, test_box1.Lz, test_box1.xy,
                         test_box1.xz, test_box1.yz)
    custom_variant = VolumeRampBoxVariant(test_box1, final_volume, 100, 100)

    def box_t(custom_variant, timestep):
        return hoomd._hoomd._test_vector_variant_box_call(
            custom_variant, timestep)

    for t, f in ((0, 0), (42, 0), (100, 0), (101, 0.01), (150, 0.5),
                 (175, 0.75), (199, 0.99), (200, 1.0), (250, 1.0), (123456789,
                                                                    1.0)):
        test_box.volume = (1 - f) * test_box1.volume + f * final_volume
        npt.assert_allclose(box_t(custom_variant, t), box_to_array(test_box))


def test_interpolate_evaluation():
    t_start = 50
    t_ramp = 100
    scalar_variant = hoomd.variant.Ramp(0, 1, t_start, t_ramp)
    box_variant = hoomd.variant.box.Interpolate(test_box1, test_box2,
                                                scalar_variant)
    npt.assert_allclose(box_variant(0), box_to_array(test_box1))
    npt.assert_allclose(box_variant(25), box_to_array(test_box1))
    npt.assert_allclose(box_variant(t_start), box_to_array(test_box1))

    npt.assert_allclose(
        box_variant(51),
        0.99 * box_to_array(test_box1) + 0.01 * box_to_array(test_box2))
    npt.assert_allclose(
        box_variant(75),
        0.75 * box_to_array(test_box1) + 0.25 * box_to_array(test_box2))
    npt.assert_allclose(
        box_variant(100),
        0.5 * box_to_array(test_box1) + 0.5 * box_to_array(test_box2))
    npt.assert_allclose(
        box_variant(125),
        0.25 * box_to_array(test_box1) + 0.75 * box_to_array(test_box2))
    npt.assert_allclose(
        box_variant(149),
        0.01 * box_to_array(test_box1) + 0.99 * box_to_array(test_box2))

    npt.assert_allclose(box_variant(t_start + t_ramp), box_to_array(test_box2))
    npt.assert_allclose(box_variant(t_start + t_ramp + 100),
                        box_to_array(test_box2))
    npt.assert_allclose(box_variant(t_start + t_ramp + 1000000),
                        box_to_array(test_box2))


def test_inverse_volume_ramp_evaluation():
    box1 = hoomd.Box(10, 10, 10, 0.1, 0.2, 0.3)
    final_volume = 500
    t_start = 10
    t_ramp = 100
    variant = hoomd.variant.box.InverseVolumeRamp(box1, final_volume, t_start,
                                                  t_ramp)

    def get_volume(variant, timestep):
        return hoomd.Box(*variant(timestep)).volume

    assert get_volume(variant, 0) == box1.volume
    assert get_volume(variant, 5) == box1.volume
    assert get_volume(variant, 10) == box1.volume
    assert get_volume(variant, 11) != box1.volume
    npt.assert_allclose(get_volume(variant, 35),
                        (0.75 / box1.volume + 0.25 / final_volume)**-1)
    npt.assert_allclose(get_volume(variant, 60),
                        (0.5 / box1.volume + 0.5 / final_volume)**-1)
    npt.assert_allclose(get_volume(variant, 85),
                        (0.25 / box1.volume + 0.75 / final_volume)**-1)
    npt.assert_allclose(get_volume(variant, 110), final_volume)
    npt.assert_allclose(get_volume(variant, 1010), final_volume)
    # make sure tilts don't change
    for step in (0, 5, 252, 125, 625):
        npt.assert_allclose(box1.tilts, variant(step)[3:])
