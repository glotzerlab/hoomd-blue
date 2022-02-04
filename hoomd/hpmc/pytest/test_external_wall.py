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


@pytest.fixture(scope="module")
def add_default_integrator():

    def add(simulation):
        mc = hoomd.hpmc.integrate.Sphere()
        mc.shape['A'] = dict(diameter=0)
        wall_list = [hoomd.wall.Sphere(1.0)]
        walls = hoomd.hpmc.external.wall.WallPotential(wall_list)
        mc.external_potential = walls
        simulation.operations.integrator = mc
        return mc, walls

    return add


# TODO: parameterize over all shapes and wall geometries
@pytest.mark.cpu
def test_attaching(simulation_factory, two_particle_snapshot_factory,
                   add_default_integrator):
    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    mc, walls = add_default_integrator(sim)

    # create C++ mirror classes and set parameters
    sim.run(0)
    assert mc._attached
    assert walls._attached


# TODO: parameterize over all shapes and wall geometries
@pytest.mark.cpu
def test_detaching(simulation_factory, two_particle_snapshot_factory,
                   add_default_integrator):
    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    mc, walls = add_default_integrator(sim)

    # create C++ mirror classes and set parameters
    sim.run(0)

    # make sure objecst are attached
    sim.operations.remove(mc)
    assert not mc._attached
    assert not walls._attached
