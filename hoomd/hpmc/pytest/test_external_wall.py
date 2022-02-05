# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.external.wall."""

import hoomd
import itertools
import pytest
from hoomd.hpmc.pytest.conftest import _valid_args

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


wall_args = {
    hoomd.wall.Sphere: (1.0,),
    hoomd.wall.Cylinder: (1.0, (0, 0, 1)),
    hoomd.wall.Plane: ((0, 0, 0), (1, 1, 1))
}


@pytest.fixture(scope="module")
def add_default_integrator():

    def add(simulation, integrator_class, wall_types):
        mc = integrator_class()
        mc.shape['A'] = mc_params[integrator_class]
        wall_list = [wt(*wall_args[wt]) for wt in wall_types]
        walls = hoomd.hpmc.external.wall.WallPotential(wall_list)
        mc.external_potential = walls
        simulation.operations.integrator = mc
        return mc, walls

    return add


mc_params = {}
for integrator_class, args, _ in _valid_args:
    if type(integrator_class) is tuple:
        base_shape, integrator_class = integrator_class
    mc_params.setdefault(integrator_class, args)

shape_wall_compatibilities = [
    (hoomd.hpmc.integrate.ConvexPolygon, {
        hoomd.wall.Sphere: False,
        hoomd.wall.Cylinder: False,
        hoomd.wall.Plane: False
    }),
    (hoomd.hpmc.integrate.ConvexPolyhedron, {
        hoomd.wall.Sphere: True,
        hoomd.wall.Cylinder: True,
        hoomd.wall.Plane: True
    }),
    (hoomd.hpmc.integrate.ConvexSpheropolygon, {
        hoomd.wall.Sphere: False,
        hoomd.wall.Cylinder: False,
        hoomd.wall.Plane: False
    }),
    (hoomd.hpmc.integrate.ConvexSpheropolyhedron, {
        hoomd.wall.Sphere: True,
        hoomd.wall.Cylinder: False,
        hoomd.wall.Plane: True
    }),
    (hoomd.hpmc.integrate.ConvexSpheropolyhedronUnion, {
        hoomd.wall.Sphere: False,
        hoomd.wall.Cylinder: False,
        hoomd.wall.Plane: False
    }),
    (hoomd.hpmc.integrate.Ellipsoid, {
        hoomd.wall.Sphere: False,
        hoomd.wall.Cylinder: False,
        hoomd.wall.Plane: False
    }),
    (hoomd.hpmc.integrate.FacetedEllipsoid, {
        hoomd.wall.Sphere: False,
        hoomd.wall.Cylinder: False,
        hoomd.wall.Plane: False
    }),
    (hoomd.hpmc.integrate.FacetedEllipsoidUnion, {
        hoomd.wall.Sphere: False,
        hoomd.wall.Cylinder: False,
        hoomd.wall.Plane: False
    }),
    (hoomd.hpmc.integrate.Polyhedron, {
        hoomd.wall.Sphere: False,
        hoomd.wall.Cylinder: False,
        hoomd.wall.Plane: False
    }),
    (hoomd.hpmc.integrate.SimplePolygon, {
        hoomd.wall.Sphere: False,
        hoomd.wall.Cylinder: False,
        hoomd.wall.Plane: False
    }),
    (hoomd.hpmc.integrate.Sphere, {
        hoomd.wall.Sphere: True,
        hoomd.wall.Cylinder: True,
        hoomd.wall.Plane: True
    }),
    (hoomd.hpmc.integrate.SphereUnion, {
        hoomd.wall.Sphere: False,
        hoomd.wall.Cylinder: False,
        hoomd.wall.Plane: False
    }),
    (hoomd.hpmc.integrate.Sphinx, {
        hoomd.wall.Sphere: False,
        hoomd.wall.Cylinder: False,
        hoomd.wall.Plane: False
    }),
]
valid_flattened_shape_wall_combos = []
invalid_flattened_shape_wall_combos = []
for shape, wall_info in shape_wall_compatibilities:
    valid_flattened_shape_wall_combos.extend(
        [shape, wall] for wall, v in wall_info.items() if v)
    invalid_flattened_shape_wall_combos.extend(
        [shape, wall] for wall, v in wall_info.items() if not v)


@pytest.mark.cpu
@pytest.mark.parametrize("shapewall", invalid_flattened_shape_wall_combos)
def test_attaching_invalid_combos(simulation_factory,
                                  two_particle_snapshot_factory,
                                  add_default_integrator, shapewall):
    integrator_class, wall = shapewall
    sim = simulation_factory(two_particle_snapshot_factory())
    mc, walls = add_default_integrator(sim, integrator_class, [
        wall,
    ])
    with pytest.raises(NotImplementedError):
        sim.run(0)


@pytest.mark.cpu
@pytest.mark.parametrize("shapewall", valid_flattened_shape_wall_combos)
def test_attaching_valid_combos(simulation_factory,
                                two_particle_snapshot_factory,
                                add_default_integrator, shapewall):
    integrator_class, wall = shapewall
    sim = simulation_factory(two_particle_snapshot_factory())
    mc, walls = add_default_integrator(sim, integrator_class, [
        wall,
    ])
    sim.run(0)
    assert mc._attached
    assert walls._attached


@pytest.mark.cpu
@pytest.mark.parametrize("shapewall", valid_flattened_shape_wall_combos)
def test_detaching(simulation_factory, two_particle_snapshot_factory,
                   add_default_integrator, shapewall):
    sim = simulation_factory(two_particle_snapshot_factory())
    integrator_class, wall = shapewall
    mc, walls = add_default_integrator(sim, integrator_class, [
        wall,
    ])
    sim.run(0)
    sim.operations.remove(mc)
    assert not mc._attached
    assert not walls._attached


shape_multiwall_combos = []
for shape, wall_info in shape_wall_compatibilities:
    if any((v for v in wall_info.values())):
        shape_multiwall_combos.append(
            (shape, [(w, v) for w, v in wall_info.items()]))


@pytest.mark.cpu
@pytest.mark.parametrize("shapewalls", shape_multiwall_combos)
def test_multiple_wall_geometries(simulation_factory,
                                  two_particle_snapshot_factory,
                                  add_default_integrator, shapewalls):
    sim = simulation_factory(two_particle_snapshot_factory())
    integrator_class, walls_valid = shapewalls
    walls = [x[0] for x in walls_valid]
    is_valids = [x[1] for x in walls_valid]
    mc, walls = add_default_integrator(sim, integrator_class, walls)
    if all(is_valids):
        sim.run(0)
    else:
        with pytest.raises(NotImplementedError):
            sim.run(0)


@pytest.mark.cpu
def test_replace_with_invalid(simulation_factory, two_particle_snapshot_factory,
                              add_default_integrator):
    sim = simulation_factory(two_particle_snapshot_factory())
    integrator_class = hoomd.hpmc.integrate.ConvexSpheropolyhedron
    walls = [hoomd.wall.Sphere, hoomd.wall.Plane]
    mc, walls = add_default_integrator(sim, integrator_class, walls)
    sim.run(0)
    with pytest.raises(NotImplementedError):
        mc.external_potential.walls = [hoomd.wall.Cylinder(1.2345, (0, 0, 0))]


@pytest.mark.cpu
def test_replace_with_valid(simulation_factory, two_particle_snapshot_factory,
                            add_default_integrator):
    sim = simulation_factory(two_particle_snapshot_factory())
    integrator_class = hoomd.hpmc.integrate.ConvexSpheropolyhedron
    walls = [hoomd.wall.Plane]
    mc, walls = add_default_integrator(sim, integrator_class, walls)
    sim.run(0)
    mc.external_potential.walls = [hoomd.wall.Sphere(1.0)]
    sim.run(0)
