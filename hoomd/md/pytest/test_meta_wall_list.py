# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from collections import defaultdict

import numpy as np
import pytest

import hoomd


class WallGenerator:
    rng = np.random.default_rng(1264556)
    scale = 1e3

    @classmethod
    def generate(cls, types=("Sphere", "Cylinder", "Plane")):
        type = cls.rng.choice(types)
        origin = (cls.float(), cls.float(), cls.float())
        inside = cls.rng.choice((True, False))
        if type == "Sphere":
            return hoomd.wall.Sphere(radius=cls.float(),
                                     origin=origin,
                                     inside=inside)
        if type == "Cylinder":
            return hoomd.wall.Cylinder(radius=cls.float(),
                                       origin=origin,
                                       axis=(cls.float(), cls.float(),
                                             cls.float()),
                                       inside=inside)
        if type == "Plane":
            normal = np.array((cls.float(), cls.float(), cls.float()))
            vector_norm = np.linalg.norm(normal)
            if vector_norm == 0:
                assert "Generated invalid normal."
            normal /= vector_norm
            return hoomd.wall.Plane(origin=origin, normal=normal)

    @classmethod
    def float(cls):
        return cls.rng.random() * cls.scale

    @classmethod
    def generate_n(cls, N):
        for _ in range(N):
            yield cls.generate()


@pytest.fixture
def blank_meta_list():
    return hoomd.wall._WallsMetaList()


@pytest.fixture
def meta_list():
    return hoomd.wall._WallsMetaList(
        walls=[w for w in WallGenerator.generate_n(20)])


def check_backend(wall_list):
    frontend = wall_list._walls
    type_counter = defaultdict(lambda: 0)
    for item, backend_index in zip(frontend, wall_list._backend_list_index):
        item_type = type(item)
        assert item_type is backend_index.type
        assert type_counter[item_type] == backend_index.index
        assert wall_list._backend_lists[item_type][backend_index.index] is item
        type_counter[item_type] += 1


def test_construction():
    walls = [wall for wall in WallGenerator.generate_n(10)]
    wall_list = hoomd.wall._WallsMetaList(walls)
    assert all(a == b for a, b in zip(walls, wall_list))
    check_backend(wall_list)


@pytest.mark.parametrize("N", (1, 10, 15, 20))
def test_len(N):
    walls = [wall for wall in WallGenerator.generate_n(N)]
    wall_list = hoomd.wall._WallsMetaList(walls)
    assert len(wall_list) == N


def test_append(blank_meta_list):
    for i, wall in enumerate(WallGenerator.generate_n(10), start=1):
        blank_meta_list.append(wall)
        assert len(blank_meta_list) == i
        check_backend(blank_meta_list)


def test_getitem(meta_list):
    walls = meta_list._walls
    for i in range(len(walls)):
        assert walls[i], meta_list[i]
    with pytest.raises(IndexError):
        meta_list[len(meta_list)]


@pytest.mark.parametrize("insert_index", (0, 1, 2, 3, 12, 1000))
def test_insert(meta_list, insert_index):
    wall = WallGenerator.generate()
    meta_list.insert(insert_index, wall)
    assert meta_list[min(insert_index, len(meta_list) - 1)] is wall
    check_backend(meta_list)


@pytest.mark.parametrize("delete_index", (0, 1, 2, 3, 12, 1000))
def test_delitem(meta_list, delete_index):
    original_len = len(meta_list)
    if delete_index >= len(meta_list):
        with pytest.raises(IndexError):
            del meta_list[delete_index]
        return

    del meta_list[delete_index]
    assert len(meta_list) == original_len - 1
    check_backend(meta_list)


@pytest.mark.parametrize("wall_slice", (slice(0), slice(1, 3), slice(2)))
def test_extend(blank_meta_list, wall_slice):
    walls = [w for w in WallGenerator.generate_n(10)]
    use_slice = walls[wall_slice]
    blank_meta_list.extend(use_slice)

    assert len(blank_meta_list) == len(use_slice)
    assert all(a is b for a, b in zip(use_slice, blank_meta_list))
    check_backend(blank_meta_list)


def test_clear(meta_list):
    meta_list.clear()
    assert len(meta_list) == 0
    check_backend(meta_list)


@pytest.mark.parametrize("set_index", (0, 1, 2, 20, 200))
def test_setitem(meta_list, set_index):
    wall = WallGenerator.generate()
    if set_index >= len(meta_list):
        with pytest.raises(IndexError):
            meta_list[set_index] = wall
        return

    meta_list[set_index] = wall
    assert meta_list[set_index] is wall
    check_backend(meta_list)


@pytest.mark.parametrize("pop_index", (0, 1, 2, 20, 200))
def test_pop(meta_list, pop_index):
    original_len = len(meta_list)
    if pop_index >= len(meta_list):
        with pytest.raises(IndexError):
            meta_list.pop(pop_index)
        return

    meta_list.pop(pop_index)
    assert len(meta_list) == original_len - 1
    check_backend(meta_list)
