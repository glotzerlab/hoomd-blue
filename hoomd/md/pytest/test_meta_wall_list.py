# Copyright (c) 2009-2023 The Regents of the University of Michigan.
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
        origin = (self.float(), self.float(), self.float())
        open = self.bool()
        random_type = self.int(3)
        if random_type == 0:
            return hoomd.wall.Sphere(radius=self.float(),
                                     origin=origin,
                                     open=open,
                                     inside=self.bool())
        elif random_type == 1:
            return hoomd.wall.Cylinder(radius=self.float(),
                                       origin=origin,
                                       axis=(self.float(), self.float(),
                                             self.float()),
                                       inside=self.bool(),
                                       open=open)
        normal = np.array((self.float(), self.float(), self.float()))
        vector_norm = np.linalg.norm(normal)
        if vector_norm == 0:
            assert "Generated invalid normal."
        normal /= vector_norm
        return hoomd.wall.Plane(origin=origin, normal=normal, open=open)

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
