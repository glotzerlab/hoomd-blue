# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.external.wall."""

import hoomd
import itertools
import pytest

wall_types = [
    hoomd.wall.Cylinder(1.0, (0, 0, 1)),
    hoomd.wall.Plane((0, 0, 0), (1, 1, 1)),
    hoomd.wall.Sphere(1.0)
]
valid_wall_lists = []
for r in 1, 2, 3:
    walls_ = list(itertools.combinations(wall_types, r))
    valid_wall_lists.extend(walls_)


@pytest.mark.cpu
@pytest.mark.parametrize("wall_list", valid_wall_lists)
def test_valid_construction(device, wall_list):
    """Test that WallPotential can be constructed with valid arguments."""
    walls = hoomd.hpmc.external.wall.WallPotential(wall_list)

    # validate the params were set properly
    for wall_input, wall_in_object in itertools.zip_longest(
            wall_list, walls.walls):
        assert wall_input == wall_in_object
