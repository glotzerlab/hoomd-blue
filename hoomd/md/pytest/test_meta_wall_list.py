# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from collections import defaultdict

import numpy as np
import pytest

import hoomd
from hoomd import conftest


class TestWallMetaList(conftest.BaseListTest):

    @pytest.fixture
    def generate_plain_collection(self):

        def generate(n):
            return [self.generate_wall() for _ in range(n)]

        return generate

    @pytest.fixture
    def empty_collection(self):
        return hoomd.wall._WallsMetaList()

    def generate_wall(self):
        kwargs = {"origin": (float, float, float), "open": bool}
        random_type = self.generator.int(3)
        if random_type == 0:
            kwargs.update({"radius": float, "inside": bool})
            return hoomd.wall.Sphere(**self.generator(kwargs))
        elif random_type == 1:
            kwargs.update({
                "radius": float,
                "axis": (float,) * 3,
                "inside": bool
            })
            return hoomd.wall.Cylinder(**self.generator(kwargs))
        normal = self.generator.ndarray((3,))
        vector_norm = np.linalg.norm(normal)
        if vector_norm == 0:
            assert "Generated invalid normal."
        normal /= vector_norm
        return hoomd.wall.Plane(normal=normal, **self.generator(kwargs))

    def is_equal(self, a, b):
        return a == b

    def final_check(self, test_list):
        frontend = test_list._walls
        type_counter = defaultdict(lambda: 0)
        for item, backend_index in zip(frontend, test_list._backend_list_index):
            item_type = type(item)
            assert item_type is backend_index.type
            assert type_counter[item_type] == backend_index.index
            assert test_list._backend_lists[item_type][
                backend_index.index] is item
            type_counter[item_type] += 1

    def test_construction(self):
        walls = [self.generate_wall() for _ in range(10)]
        wall_list = hoomd.wall._WallsMetaList(walls)
        assert all(a == b for a, b in zip(walls, wall_list))
        self.final_check(wall_list)
