# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test the internal wall data lists for C++.

This test file also tests the ``array_view`` from hoomd/ArrayView.h, and the
Python wrapper `hoomd.data.array_view._ArrayViewWrapper` which need a concrete
example for testing.
"""

import numpy as np
import pytest

from hoomd.data.array_view import _ArrayViewWrapper
from hoomd.md import _md


@pytest.fixture(scope="function")
def wall_collection():
    return _md.WallCollection._unsafe_create()


@pytest.fixture(scope="session", params=("sphere", "cylinder", "plane"))
def wall_type_lists(request):
    """Returns a list of wall data objects and their type name."""
    return {
        "sphere": ("sphere", (_md.SphereWall(4.0, (1.0, 0.0, 0.0), False, True),
                              _md.SphereWall(6.5, (0.0, 1.0, 0.0), True, False),
                              _md.SphereWall(0.5, (1.0, 2.0, 3.0), True,
                                             True))),
        "cylinder":
            ("cylinder", (_md.CylinderWall(4.0, (1.0, 0.0, 0.0),
                                           (1.0, 0.0, 0.0), False, True),
                          _md.CylinderWall(6.5, (0.0, 1.0, 0.0),
                                           (0.0, 1.0, 0.0), True, False),
                          _md.CylinderWall(0.5, (1.0, 2.0, 3.0),
                                           (3.0, 2.5, 3.0), True, False))),
        "plane": ("plane", (_md.PlaneWall(
            (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            False), _md.PlaneWall(
                (0.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                True), _md.PlaneWall((1.0, 2.0, 3.0), (3.0, 2.5, 3.0), True)))
    }[request.param]


def wall_equality(a, b):
    """Define the equality relation between objects of the same wall data type.

    Necessary since these objects do not define equality currently. If they do
    in the future, this can be removed.
    """
    attrs = {
        "SphereWall": ("radius", "origin", "inside"),
        "CylinderWall": ("radius", "origin", "axis", "inside"),
        "PlaneWall": ("origin", "normal")
    }[type(a).__name__]
    return all(
        np.allclose(getattr(a, attr), getattr(b, attr)) for attr in attrs)


def check_equivalent(array_view, collection, type_str):
    """Check if an array view is equivalent to the original buffer."""
    # cannot use enumerate since array_view does not currently support
    # iteration.

    assert len(array_view) == get_collection_size(collection, type_str)

    index_func = getattr(collection, get_index_func_name(type_str))
    for i, item in enumerate(array_view):
        assert wall_equality(item, index_func(i))


def get_index_func_name(type_str):
    return f"get_{type_str}"


def get_collection_size(wall_collection, type_str):
    return getattr(wall_collection, f"num_{type_str}s")


@pytest.fixture(scope="function", params=(False, True))
def array_view_factory(request):
    """Fixture to test the pybind11 array_view and Python wrapper."""

    def factory(wall_collection, type_str):
        get_array_view = getattr(wall_collection, f"get_{type_str}_list")
        if request.param:
            return get_array_view()
        return _ArrayViewWrapper(get_array_view)

    return factory


def test_append(array_view_factory, wall_collection, wall_type_lists):
    type_str, walls_list = wall_type_lists
    array_view = array_view_factory(wall_collection, type_str)
    for i, wall in enumerate(walls_list):
        array_view.append(wall)
        assert len(array_view) == i + 1
        assert get_collection_size(wall_collection, type_str) == i + 1
        assert wall_equality(array_view[i], wall)
    check_equivalent(array_view, wall_collection, type_str)


def test_getitem(array_view_factory, wall_collection, wall_type_lists):
    type_str, walls_list = wall_type_lists
    array_view = array_view_factory(wall_collection, type_str)
    for wall in walls_list:
        array_view.append(wall)
    for i in range(len(array_view)):
        wall_equality(array_view[i], walls_list[i])
    with pytest.raises(IndexError):
        array_view[len(array_view)]
    # This is outside the bounds of the C++ buffer size
    with pytest.raises(IndexError):
        array_view[1000]


@pytest.mark.parametrize("insert_index", (0, 1, 2, 3))
def test_insert(array_view_factory, wall_collection, wall_type_lists,
                insert_index):
    type_str, walls_list = wall_type_lists
    array_view = array_view_factory(wall_collection, type_str)
    array_view.extend(walls_list[:-1])
    for i in range(len(array_view) - 1):
        wall_equality(array_view[i], walls_list[i])

    array_view.insert(insert_index, walls_list[-1])
    assert len(array_view) == len(walls_list)
    assert get_collection_size(wall_collection, type_str) == len(walls_list)
    assert wall_equality(array_view[min(insert_index,
                                        len(array_view) - 1)], walls_list[-1])
    check_equivalent(array_view, wall_collection, type_str)


@pytest.mark.parametrize("delete_index", (0, 1, 2, 3))
def test_delitem(array_view_factory, wall_collection, wall_type_lists,
                 delete_index):
    type_str, walls_list = wall_type_lists
    array_view = array_view_factory(wall_collection, type_str)
    array_view.extend(walls_list)

    if delete_index >= len(array_view):
        with pytest.raises(IndexError):
            del array_view[delete_index]
        return

    del array_view[delete_index]
    assert len(array_view) == len(walls_list) - 1
    assert get_collection_size(wall_collection, type_str) == len(walls_list) - 1
    if delete_index >= len(array_view):
        array_view_index = len(array_view) - 1
        wall_lists_index = array_view_index
    else:
        array_view_index = delete_index
        wall_lists_index = delete_index + 1
    assert wall_equality(array_view[array_view_index],
                         walls_list[wall_lists_index])
    check_equivalent(array_view, wall_collection, type_str)


def test_len(array_view_factory, wall_collection, wall_type_lists):
    type_str, walls_list = wall_type_lists
    array_view = array_view_factory(wall_collection, type_str)
    for i, wall in enumerate(walls_list):
        assert len(array_view) == i
        assert get_collection_size(wall_collection, type_str) == i
        array_view.append(wall)
    n_walls = len(walls_list)
    assert len(array_view) == n_walls
    assert get_collection_size(wall_collection, type_str) == n_walls
    for n in range(n_walls - 1, -1, -1):
        del array_view[0]
        assert len(array_view) == n
        assert get_collection_size(wall_collection, type_str) == n


@pytest.mark.parametrize("wall_slice", (slice(0), slice(1, 3), slice(2)))
def test_extend(array_view_factory, wall_collection, wall_type_lists,
                wall_slice):
    type_str, walls_list = wall_type_lists
    array_view = array_view_factory(wall_collection, type_str)

    use_slice = walls_list[wall_slice]
    array_view.extend(use_slice)

    assert len(array_view) == len(use_slice)
    assert get_collection_size(wall_collection, type_str) == len(use_slice)
    check_equivalent(array_view, wall_collection, type_str)


def test_clear(array_view_factory, wall_collection, wall_type_lists):
    type_str, walls_list = wall_type_lists
    array_view = array_view_factory(wall_collection, type_str)
    array_view.extend(walls_list)
    array_view.clear()
    assert len(array_view) == 0
    assert get_collection_size(wall_collection, type_str) == 0


@pytest.mark.parametrize("set_index", (0, 1, 2, 20, 200))
def test_setitem(array_view_factory, wall_collection, wall_type_lists,
                 set_index):
    type_str, walls_list = wall_type_lists
    array_view = array_view_factory(wall_collection, type_str)
    array_view.extend(walls_list)
    # Ensure that we set to a new wall.
    from_index = (set_index + 1) % len(array_view)
    if set_index >= len(array_view):
        with pytest.raises(IndexError):
            array_view[set_index] = walls_list[from_index]
        return

    array_view[set_index] = walls_list[from_index]
    assert wall_equality(array_view[set_index], walls_list[from_index])
    check_equivalent(array_view, wall_collection, type_str)
    assert len(array_view) == len(walls_list)
    assert get_collection_size(wall_collection, type_str) == len(walls_list)


@pytest.mark.parametrize("pop_index", (0, 1, 2, 20, 200))
def test_pop(array_view_factory, wall_collection, wall_type_lists, pop_index):
    type_str, walls_list = wall_type_lists
    array_view = array_view_factory(wall_collection, type_str)
    array_view.extend(walls_list)
    if pop_index >= len(array_view):
        with pytest.raises(IndexError):
            array_view.pop(pop_index)
        return

    wall = array_view.pop(pop_index)
    assert wall_equality(wall, walls_list[pop_index])
    check_equivalent(array_view, wall_collection, type_str)
    assert len(array_view) == len(walls_list) - 1
    assert get_collection_size(wall_collection, type_str) == len(walls_list) - 1


@pytest.mark.parametrize("slice_", (slice(0), slice(1, 3), slice(2)))
def test_slices_for_python_wrapper(wall_collection, wall_type_lists, slice_):
    """Test __getitem__ and __delitem__ for _ArrayViewWrapper."""
    type_str, walls_list = wall_type_lists
    array_view = _ArrayViewWrapper(
        getattr(wall_collection, f"get_{type_str}_list"))
    array_view.extend(walls_list)

    # Check getitem with slices
    assert len(array_view[slice_]) == len(walls_list[slice_])
    assert all(
        wall_equality(x, y)
        for x, y in zip(array_view[slice_], walls_list[slice_]))

    # Check delitem with slices must convert walls_list to list to make mutable
    # and prevent alteration of the underlying data.
    walls_list = list(walls_list)
    del walls_list[slice_]
    del array_view[slice_]
    assert len(array_view) == len(walls_list)
    print(len(array_view))
    assert all(wall_equality(x, y) for x, y in zip(array_view, walls_list))
    check_equivalent(array_view, wall_collection, type_str)
