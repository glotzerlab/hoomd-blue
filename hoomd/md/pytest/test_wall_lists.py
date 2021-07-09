"""Test the internal wall data lists for C++.

This test file also tests the array_view from hoomd/ArrayView.h which needs a
concrete example for testing.
"""

import numpy as np
import pytest

from hoomd.md import _md


@pytest.fixture(scope="function")
def wall_collection():
    return _md.WallCollection()


@pytest.fixture(scope="session", params=("sphere", "cylinder", "plane"))
def wall_type_lists(request):
    """Returns a list of wall data objects and their type name."""
    return {
        "sphere": ("sphere", (_md.SphereWall(4.0, (1.0, 0.0, 0.0), False),
                              _md.SphereWall(6.5, (0.0, 1.0, 0.0), True),
                              _md.SphereWall(0.5, (1.0, 2.0, 3.0), True))),
        "cylinder": ("cylinder", (_md.CylinderWall(4.0, (1.0, 0.0, 0.0),
                                                   (1.0, 0.0, 0.0), False),
                                  _md.CylinderWall(6.5, (0.0, 1.0, 0.0),
                                                   (0.0, 1.0, 0.0), True),
                                  _md.CylinderWall(0.5, (1.0, 2.0, 3.0),
                                                   (3.0, 2.5, 3.0), True))),
        "plane": ("plane", (_md.PlaneWall(
            (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            False), _md.PlaneWall(
                (0.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                True), _md.PlaneWall((1.0, 2.0, 3.0), (3.0, 2.5, 3.0), True)))
    }[request.param]


def check_equivalent(array_view, collection, index_func, item_eq=None):
    """Check if an array view is equivalent to the original buffer."""
    # cannot use enumerate since array_view does not currently support
    # iteration.
    if item_eq is None:

        def item_eq(a, b):
            return a == b

    for i in range(len(array_view)):
        assert item_eq(array_view[i], getattr(collection, index_func)(i))


class WallDataEquivalence:
    """Define the equality relation between objects of the same wall data type.

    Necessary since these objects do not define equality currently. If they do
    in the future, this can be removed.
    """

    def __init__(self, type_str):
        self.attrs = {
            "sphere": ("radius", "origin", "inside"),
            "cylinder": ("radius", "origin", "axis", "inside"),
            "plane": ("origin", "normal", "inside")
        }[type_str]

    def __call__(self, a, b):
        return all(
            np.allclose(getattr(a, attr), getattr(b, attr))
            for attr in self.attrs)


def get_index_func_name(type_str):
    return f"get_{type_str}"


def get_collection_size(wall_collection, type_str):
    return getattr(wall_collection, f"num_{type_str}s")


def get_array_view(wall_collection, type_str):
    return getattr(wall_collection, f"get_{type_str}_list")()


def test_append(wall_collection, wall_type_lists):
    type_str, wall_lists = wall_type_lists
    walls_eq_func = WallDataEquivalence(type_str)
    array_view = get_array_view(wall_collection, type_str)
    for i, wall in enumerate(wall_lists):
        array_view.append(wall)
        assert len(array_view) == i + 1
        assert get_collection_size(wall_collection, type_str) == i + 1
        assert walls_eq_func(array_view[i], wall)
    check_equivalent(array_view, wall_collection, get_index_func_name(type_str),
                     walls_eq_func)


def test_getitem(wall_collection, wall_type_lists):
    type_str, wall_lists = wall_type_lists
    walls_eq_func = WallDataEquivalence(type_str)
    array_view = get_array_view(wall_collection, type_str)
    for wall in wall_lists:
        array_view.append(wall)
    for i in range(len(array_view)):
        walls_eq_func(array_view[i], wall_lists[i])
    with pytest.raises(RuntimeError):
        array_view[len(array_view)]
    # This is outside the bounds of the C++ buffer size
    with pytest.raises(RuntimeError):
        array_view[1000]


@pytest.mark.parametrize("insert_index", (0, 1, 2, 3))
def test_insert(wall_collection, wall_type_lists, insert_index):
    type_str, wall_lists = wall_type_lists
    walls_eq_func = WallDataEquivalence(type_str)
    array_view = get_array_view(wall_collection, type_str)
    array_view.extend(wall_lists[:-1])
    for i in range(len(array_view) - 1):
        walls_eq_func(array_view[i], wall_lists[i])

    if insert_index > len(array_view):
        with pytest.raises(RuntimeError):
            array_view.insert(insert_index, wall_lists[-1])
        return

    array_view.insert(insert_index, wall_lists[-1])
    assert len(array_view) == len(wall_lists)
    assert get_collection_size(wall_collection, type_str) == len(wall_lists)
    assert walls_eq_func(array_view[insert_index], wall_lists[-1])
    check_equivalent(array_view, wall_collection, get_index_func_name(type_str),
                     walls_eq_func)


@pytest.mark.parametrize("delete_index", (0, 1, 2, 3))
def test_delitem(wall_collection, wall_type_lists, delete_index):
    type_str, wall_lists = wall_type_lists
    walls_eq_func = WallDataEquivalence(type_str)
    array_view = get_array_view(wall_collection, type_str)
    array_view.extend(wall_lists)

    if delete_index >= len(array_view):
        with pytest.raises(RuntimeError):
            del array_view[delete_index]
        return

    del array_view[delete_index]
    assert len(array_view) == len(wall_lists) - 1
    assert get_collection_size(wall_collection, type_str) == len(wall_lists) - 1
    if delete_index >= len(array_view):
        array_view_index = len(array_view) - 1
        wall_lists_index = array_view_index
    else:
        array_view_index = delete_index
        wall_lists_index = delete_index + 1
    assert walls_eq_func(array_view[array_view_index],
                         wall_lists[wall_lists_index])
    check_equivalent(array_view, wall_collection, get_index_func_name(type_str),
                     walls_eq_func)


def test_len(wall_collection, wall_type_lists):
    type_str, wall_lists = wall_type_lists
    array_view = get_array_view(wall_collection, type_str)
    for i, wall in enumerate(wall_lists):
        print(i)
        assert len(array_view) == i
        assert get_collection_size(wall_collection, type_str) == i
        array_view.append(wall)
    n_walls = len(wall_lists)
    assert len(array_view) == n_walls
    assert get_collection_size(wall_collection, type_str) == n_walls
    for n in range(n_walls - 1, -1, -1):
        del array_view[0]
        assert len(array_view) == n
        assert get_collection_size(wall_collection, type_str) == n


@pytest.mark.parametrize("wall_slice", (slice(0), slice(1, 3), slice(2)))
def test_extend(wall_collection, wall_type_lists, wall_slice):
    type_str, wall_lists = wall_type_lists
    walls_eq_func = WallDataEquivalence(type_str)
    array_view = get_array_view(wall_collection, type_str)

    use_slice = wall_lists[wall_slice]
    array_view.extend(use_slice)

    assert len(array_view) == len(use_slice)
    assert get_collection_size(wall_collection, type_str) == len(use_slice)
    check_equivalent(array_view, wall_collection, get_index_func_name(type_str),
                     walls_eq_func)


def test_clear(wall_collection, wall_type_lists):
    type_str, wall_lists = wall_type_lists
    array_view = get_array_view(wall_collection, type_str)
    array_view.extend(wall_lists)
    array_view.clear()
    assert len(array_view) == 0
    assert get_collection_size(wall_collection, type_str) == 0


@pytest.mark.parametrize("set_index", (0, 1, 2, 20, 200))
def test_setitem(wall_collection, wall_type_lists, set_index):
    type_str, wall_lists = wall_type_lists
    walls_eq_func = WallDataEquivalence(type_str)
    array_view = get_array_view(wall_collection, type_str)
    array_view.extend(wall_lists)
    from_index = (set_index + 1) % len(array_view)
    if set_index >= len(array_view):
        with pytest.raises(RuntimeError):
            array_view[set_index] = wall_lists[from_index]
        return

    array_view[set_index] = wall_lists[from_index]
    assert walls_eq_func(array_view[set_index], wall_lists[from_index])
    check_equivalent(array_view, wall_collection, get_index_func_name(type_str),
                     walls_eq_func)
    assert len(array_view) == len(wall_lists)
    assert get_collection_size(wall_collection, type_str) == len(wall_lists)


@pytest.mark.parametrize("pop_index", (0, 1, 2, 20, 200))
def test_pop(wall_collection, wall_type_lists, pop_index):
    type_str, wall_lists = wall_type_lists
    walls_eq_func = WallDataEquivalence(type_str)
    array_view = get_array_view(wall_collection, type_str)
    array_view.extend(wall_lists)
    if pop_index >= len(array_view):
        with pytest.raises(RuntimeError):
            array_view.pop(pop_index)
        return

    wall = array_view.pop(pop_index)
    assert walls_eq_func(wall, wall_lists[pop_index])
    check_equivalent(array_view, wall_collection, get_index_func_name(type_str),
                     walls_eq_func)
    assert len(array_view) == len(wall_lists) - 1
    assert get_collection_size(wall_collection, type_str) == len(wall_lists) - 1
