# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test the C++ internal wall data structures."""

import numpy as np
import pytest

from hoomd.md import _md


class _TestCounter:

    def __init__(self):
        self.previous_cls = None
        self.count = 0

    def __call__(self, arg):
        # Return empty string for non class arguments
        if isinstance(arg, dict):
            return ""
        if arg == self.previous_cls:
            self.count += 1
        self.count = 1
        self.previous_cls = arg
        return f"{arg.__name__}-{self.count}"


_test_args = (
    (_md.SphereWall, ({
        "radius": 4.0,
        "origin": (1.0, 0, 0),
        "inside": True,
        "open": False
    }, {
        "radius": 1.0,
        "origin": (0.0, 0.0, 0.0),
        "inside": False,
        "open": False
    }, {
        "radius": 3.1415,
        "origin": (-1.0, -2.0, 2.0),
        "inside": False,
        "open": True
    })),
    (_md.CylinderWall, ({
        "radius": 4.0,
        "origin": (1.0, 0, 0),
        "axis": (1.0, 0.0, 0),
        "inside": True,
        "open": True
    }, {
        "radius": 1.0,
        "origin": (0.0, 0.0, 0.0),
        "axis": (0.0, 1.0, 0.0),
        "inside": False,
        "open": False
    }, {
        "radius": 3.1415,
        "origin": (-1.0, -2.0, 1.0),
        "axis": (0.0, 0.0, 1.0),
        "inside": False,
        "open": True
    })),
    (
        _md.PlaneWall,
        (
            # The normals have to be unit vectors for the equality checks to
            # hold.  The C++ class currently normalizes any input vector.
            {
                "origin": (1.0, 0, 0),
                "normal": (1.0, 0.0, 0),
                "open": False
            },
            {
                "origin": (0.0, 0.0, 0.0),
                "normal": (0.0, 1.0, 0.0),
                "open": True
            },
            {
                "origin": (-1.0, -2.0, 1.0),
                "normal": (0.0, 0.0, 1.0),
                "open": False
            })))


@pytest.mark.parametrize("cls, constructor_kwargs",
                         ((cls, constructor_kwargs)
                          for cls, arg_list in _test_args
                          for constructor_kwargs in arg_list),
                         ids=_TestCounter())
def test_valid_construction(cls, constructor_kwargs):
    obj = cls(**constructor_kwargs)
    for key, value in constructor_kwargs.items():
        assert np.allclose(getattr(obj, key), value)


@pytest.mark.parametrize("cls, constructor_kwargs",
                         ((cls, arg_list[0]) for cls, arg_list in _test_args),
                         ids=_TestCounter())
def test_immutability(cls, constructor_kwargs):
    obj = cls(**constructor_kwargs)
    for key, value in constructor_kwargs.items():
        with pytest.raises(AttributeError):
            setattr(obj, key, value)
