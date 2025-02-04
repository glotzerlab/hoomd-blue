# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test the internal wall data lists for C++.

This test file also tests the ``array_view`` from hoomd/ArrayView.h, and the
Python wrapper `hoomd.data.array_view._ArrayViewWrapper` which need a concrete
example for testing.
"""

import numpy as np
import pytest

from hoomd import conftest
from hoomd.data.array_view import _ArrayViewWrapper
from hoomd.md import _md


class TestArrayViewPython(conftest.BaseListTest):
    _cpp_view = False

    @pytest.fixture(params=(True, False))
    def cpp_view(self, request):
        self._cpp_view = request.param
        return self._cpp_view

    @pytest.fixture(params=("sphere", "cylinder", "plane"))
    def mode(self, request):
        self._mode = request.param
        return self._mode

    @pytest.fixture
    def generate_plain_collection(self):
        if self._mode == "sphere":
            generation_func = self._generate_sphere
        elif self._mode == "cylinder":
            generation_func = self._generate_cylinder
        else:
            generation_func = self._generate_plane

        def generate(n):
            return [generation_func() for _ in range(n)]

        return generate

    def _generate_sphere(self):
        args = (float, (float,) * 3, bool, bool)
        return _md.SphereWall(*self.generator(args))

    def _generate_cylinder(self):
        args = (float, (float,) * 3, (float,) * 3, bool, bool)
        return _md.CylinderWall(*self.generator(args))

    def _generate_plane(self):
        args = ((float,) * 3, (float,) * 3, bool)
        return _md.PlaneWall(*self.generator(args))

    def test_contains(self, populated_collection, generate_plain_collection):
        """Contains does not work for array views."""
        pass

    def test_remove(self, populated_collection):
        """Remove does not work for array views."""
        pass

    @pytest.fixture
    def empty_collection(self, mode):
        self._wall_collection = _md.WallCollection._unsafe_create()

        get_array_view = getattr(self._wall_collection, f"get_{mode}_list")
        if self._cpp_view:
            return get_array_view()
        return _ArrayViewWrapper(get_array_view)

    def is_equal(self, a, b):
        # logic is necessary as C++ walls do not define equality
        attrs = {
            "SphereWall": ("radius", "origin", "inside"),
            "CylinderWall": ("radius", "origin", "axis", "inside"),
            "PlaneWall": ("origin", "normal")
        }[type(a).__name__]
        return all(
            np.allclose(getattr(a, attr), getattr(b, attr)) for attr in attrs)

    def get_collection_size(self):
        return getattr(self._wall_collection, f"num_{self._mode}s")

    def final_check(self, test_collection):
        """Check if an array view is equivalent to the original buffer."""
        # cannot use enumerate since array_view does not currently support
        # iteration.

        assert len(test_collection) == self.get_collection_size()

        index_func = getattr(self._wall_collection, f"get_{self._mode}")
        for i, item in enumerate(test_collection):
            assert self.is_equal(item, index_func(i))


# We need separate classes since the C++ ArrayView class uses size_t index
# values and cannot accept negative indices. If this changes in the future then
# this can be removed.
class TestArrayViewCpp(TestArrayViewPython):
    _negative_indexing = False
    _allow_slices = False
    _cpp_view = True
