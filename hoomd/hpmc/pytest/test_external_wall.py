# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.external.wall."""

from collections import defaultdict

import hoomd
from hoomd.hpmc.pytest.conftest import _valid_args
import itertools
import numpy as np
import pytest

wall_instances = [
    hoomd.wall.Cylinder(1.0, (0, 0, 1)),
    hoomd.wall.Plane((0, 0, 0), (1, 1, 1)),
    hoomd.wall.Sphere(1.0)
]
valid_wall_lists = []
for r in 1, 2, 3:
    walls_ = list(itertools.combinations(wall_instances, r))
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


default_wall_args = {
    hoomd.wall.Sphere: (1.0,),
    hoomd.wall.Cylinder: (1.0, (0, 0, 1)),
    hoomd.wall.Plane: ((0, 0, 0), (1, 1, 1))
}


@pytest.fixture(scope="module")
def add_default_integrator():

    def add(simulation,
            integrator_class,
            wall_types,
            use_default_wall_args=True):
        mc = integrator_class()
        mc.shape['A'] = mc_params[integrator_class]
        if use_default_wall_args:
            wall_list = [wt(*default_wall_args[wt]) for wt in wall_types]
        else:
            wall_list = wall_types
        walls = hoomd.hpmc.external.wall.WallPotential(wall_list)
        mc.external_potential = walls
        simulation.operations.integrator = mc
        return mc, walls

    return add


_integrator_classes = []
mc_params = {}
for integrator_class, args, _ in _valid_args:
    if isinstance(integrator_class, tuple):
        _integrator_classes.append(integrator_class[0])
        base_shape, integrator_class = integrator_class
    else:
        _integrator_classes.append(integrator_class)
    mc_params.setdefault(integrator_class, args)


def default_wall_compatibility():
    return {
        hoomd.wall.Sphere: False,
        hoomd.wall.Cylinder: False,
        hoomd.wall.Plane: False
    }


shape_wall_compatibilities = defaultdict(default_wall_compatibility)
shape_wall_compatibilities.update({
    hoomd.hpmc.integrate.ConvexPolyhedron: {
        hoomd.wall.Sphere: True,
        hoomd.wall.Cylinder: True,
        hoomd.wall.Plane: True
    },
    hoomd.hpmc.integrate.ConvexSpheropolyhedron: {
        hoomd.wall.Sphere: True,
        hoomd.wall.Cylinder: False,
        hoomd.wall.Plane: True
    },
    hoomd.hpmc.integrate.Sphere: {
        hoomd.wall.Sphere: True,
        hoomd.wall.Cylinder: True,
        hoomd.wall.Plane: True
    },
})

valid_flattened_shape_wall_combos = []
invalid_flattened_shape_wall_combos = []
for shape in _integrator_classes:
    wall_info = shape_wall_compatibilities[shape]
    valid_flattened_shape_wall_combos.extend(
        [shape, wall] for wall, v in wall_info.items() if v)
    invalid_flattened_shape_wall_combos.extend(
        [shape, wall] for wall, v in wall_info.items() if not v)


@pytest.mark.cpu
@pytest.mark.parametrize("shape_cls, wall", invalid_flattened_shape_wall_combos)
def test_attaching_invalid_combos(simulation_factory,
                                  two_particle_snapshot_factory,
                                  add_default_integrator, shape_cls, wall):
    sim = simulation_factory(two_particle_snapshot_factory())
    mc, walls = add_default_integrator(sim, shape_cls, [wall])
    with pytest.raises(NotImplementedError):
        sim.run(0)


@pytest.mark.cpu
@pytest.mark.parametrize("shape_cls, wall", valid_flattened_shape_wall_combos)
def test_attaching_valid_combos(simulation_factory,
                                two_particle_snapshot_factory,
                                add_default_integrator, shape_cls, wall):
    sim = simulation_factory(two_particle_snapshot_factory())
    mc, walls = add_default_integrator(sim, shape_cls, [wall])
    sim.run(0)
    assert mc._attached
    assert walls._attached


@pytest.mark.cpu
@pytest.mark.parametrize("shape_cls, wall", valid_flattened_shape_wall_combos)
def test_detaching(simulation_factory, two_particle_snapshot_factory,
                   add_default_integrator, shape_cls, wall):
    sim = simulation_factory(two_particle_snapshot_factory())
    mc, walls = add_default_integrator(sim, shape_cls, [wall])
    sim.run(0)
    sim.operations.remove(mc)
    assert not mc._attached
    assert not walls._attached


shape_multiwall_combos = []
for shape in _integrator_classes:
    wall_info = shape_wall_compatibilities[shape]
    if any((v for v in wall_info.values())):
        shape_multiwall_combos.append(
            (shape, [(w, v) for w, v in wall_info.items()]))


@pytest.mark.cpu
@pytest.mark.parametrize("shape_cls, wall_validity", shape_multiwall_combos)
def test_multiple_wall_geometries(simulation_factory, shape_cls, wall_validity,
                                  two_particle_snapshot_factory,
                                  add_default_integrator):
    sim = simulation_factory(two_particle_snapshot_factory())
    walls = [x[0] for x in wall_validity]
    is_valids = [x[1] for x in wall_validity]
    mc, walls = add_default_integrator(sim, shape_cls, walls)
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
    wall_types = [hoomd.wall.Sphere, hoomd.wall.Plane]
    mc, walls = add_default_integrator(sim, integrator_class, wall_types)
    sim.run(0)
    with pytest.raises(NotImplementedError):
        mc.external_potential.walls = [hoomd.wall.Cylinder(1.2345, (0, 0, 0))]


@pytest.mark.cpu
def test_replace_with_invalid_by_append(simulation_factory,
                                        two_particle_snapshot_factory,
                                        add_default_integrator):
    sim = simulation_factory(two_particle_snapshot_factory())
    integrator_class = hoomd.hpmc.integrate.ConvexSpheropolyhedron
    wall_types = [hoomd.wall.Sphere, hoomd.wall.Plane]
    mc, walls = add_default_integrator(sim, integrator_class, wall_types)
    sim.run(0)
    with pytest.raises(NotImplementedError):
        new_wall = hoomd.wall.Cylinder(1.2345, (0, 0, 0))
        mc.external_potential.walls.append(new_wall)


@pytest.mark.cpu
def test_replace_with_invalid_by_extend(simulation_factory,
                                        two_particle_snapshot_factory,
                                        add_default_integrator):
    sim = simulation_factory(two_particle_snapshot_factory())
    integrator_class = hoomd.hpmc.integrate.ConvexSpheropolyhedron
    wall_types = [hoomd.wall.Sphere, hoomd.wall.Plane]
    mc, walls = add_default_integrator(sim, integrator_class, wall_types)
    sim.run(0)
    with pytest.raises(NotImplementedError):
        new_walls = [hoomd.wall.Cylinder(1.2345, (0, 0, 0))]
        mc.external_potential.walls.extend(new_walls)


@pytest.mark.cpu
def test_replace_with_valid(simulation_factory, two_particle_snapshot_factory,
                            add_default_integrator):
    sim = simulation_factory(two_particle_snapshot_factory())
    integrator_class = hoomd.hpmc.integrate.ConvexSpheropolyhedron
    wall_types = [hoomd.wall.Plane]
    mc, walls = add_default_integrator(sim, integrator_class, wall_types)
    sim.run(0)
    mc.external_potential.walls = [hoomd.wall.Sphere(1.0)]


@pytest.mark.cpu
def test_replace_with_valid_by_append(simulation_factory,
                                      two_particle_snapshot_factory,
                                      add_default_integrator):
    sim = simulation_factory(two_particle_snapshot_factory())
    integrator_class = hoomd.hpmc.integrate.ConvexSpheropolyhedron
    wall_types = [hoomd.wall.Plane]
    mc, walls = add_default_integrator(sim, integrator_class, wall_types)
    sim.run(0)
    mc.external_potential.walls.append(hoomd.wall.Sphere(1.0))


@pytest.mark.cpu
def test_replace_with_valid_by_extend(simulation_factory,
                                      two_particle_snapshot_factory,
                                      add_default_integrator):
    sim = simulation_factory(two_particle_snapshot_factory())
    integrator_class = hoomd.hpmc.integrate.ConvexSpheropolyhedron
    wall_types = [hoomd.wall.Plane]
    mc, walls = add_default_integrator(sim, integrator_class, wall_types)
    sim.run(0)
    mc.external_potential.walls.extend([hoomd.wall.Sphere(1.0)])


L_cube = 1.0
cube_vertices = np.array(
    list(itertools.product((-L_cube / 2, L_cube / 2), repeat=3)))
cube_rc = max(np.linalg.norm(cube_vertices, axis=1))  # circumsphere radius
cube_face_rc = np.sqrt(2) / 2
cube_r_s = 0.1  # sweep radius for spherocube
rot_x_45deg = [0.92387953, 0.38268343, 0, 0]
cube_def = dict(vertices=cube_vertices)
spherocube_def = dict(vertices=cube_vertices, sweep_radius=cube_r_s)
overlap_test_info = [
    # sphere completely inside spherical cavity
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.Sphere,
        [hoomd.wall.Sphere(1.0, (0, 0, 0), inside=True)],
        dict(diameter=1.0),
        False,
    ),
    # big sphere in small spherical cavity
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.Sphere,
        [hoomd.wall.Sphere(1.0, (0, 0, 0), inside=True)],
        dict(diameter=10.0),
        True,
    ),
    # sphere inside spherical forbidden region
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.Sphere,
        [hoomd.wall.Sphere(1.0, (0, 0, 0), inside=False)],
        dict(diameter=1.0),
        True,
    ),
    # sphere outside spherical forbidden region (i.e., no overlaps)
    (
        (5, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.Sphere,
        [hoomd.wall.Sphere(1.0, (0, 0, 0), inside=False)],
        dict(diameter=1.0),
        False,
    ),
    # big sphere outside spherical forbidden region but extends into it
    (
        (2, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.Sphere,
        [hoomd.wall.Sphere(1.0, (0, 0, 0), inside=False)],
        dict(diameter=5.0),
        True,
    ),
    # cube safely nestled in the center of a spherical cavity
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.ConvexPolyhedron,
        [hoomd.wall.Sphere(1.1 * cube_rc, (0, 0, 0), inside=True)],
        cube_def,
        False,
    ),
    # cube inside a too-small spherical cavity
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.ConvexPolyhedron,
        [hoomd.wall.Sphere(0.9 * cube_rc, (0, 0, 0), inside=True)],
        cube_def,
        True,
    ),
    # cube safely outside a spherical forbidden region
    (
        (0, 1.51, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.ConvexPolyhedron,
        [hoomd.wall.Sphere(1.0, (0, 0, 0), inside=False)],
        cube_def,
        False,
    ),
    # cube outside a spherical forbidden region but rotated to overlap
    (
        (0, 1.51, 0),
        rot_x_45deg,
        hoomd.hpmc.integrate.ConvexPolyhedron,
        [hoomd.wall.Sphere(1.0, (0, 0, 0), inside=False)],
        cube_def,
        True,
    ),
    # spherocube safely nestled in the center of a spherical cavity
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.ConvexSpheropolyhedron,
        [hoomd.wall.Sphere(1.1 * (cube_rc + cube_r_s), (0, 0, 0), inside=True)],
        spherocube_def,
        False,
    ),
    # spherocube inside a too-small spherical cavity
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.ConvexSpheropolyhedron,
        [hoomd.wall.Sphere(cube_rc, (0, 0, 0), inside=True)],
        spherocube_def,
        True,
    ),
    # spherocube safely outside a spherical forbidden region
    (
        (0, 1 + 0.5 + cube_r_s + 0.01, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.ConvexSpheropolyhedron,
        [hoomd.wall.Sphere(1.0, (0, 0, 0), inside=False)],
        spherocube_def,
        False,
    ),
    # spherocube outside a spherical forbidden region but rotated to overlap
    (
        (0, 1 + 0.5 + cube_r_s + 0.01, 0),
        rot_x_45deg,
        hoomd.hpmc.integrate.ConvexSpheropolyhedron,
        [hoomd.wall.Sphere(1.0, (0, 0, 0), inside=False)],
        spherocube_def,
        True,
    ),
    # sphere on allowed side of plane wall
    (
        (1.29, 1.29, 1.29),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.Sphere,
        [hoomd.wall.Plane((1, 1, 1), (1, 1, 1))],
        dict(diameter=1.0),
        False,
    ),
    # sphere on allowed side of plane wall but overlapping with wall
    (
        (1.28, 1.28, 1.28),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.Sphere,
        [hoomd.wall.Plane((1, 1, 1), (1, 1, 1))],
        dict(diameter=1.0),
        True,
    ),
    # sphere on disallowed side of plane wall
    (
        (1.29, 1.29, 1.29),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.Sphere,
        [hoomd.wall.Plane((1, 1, 1), (-1, -1, -1))],
        dict(diameter=1.0),
        True,
    ),
    # cube with face parallel to wall and barely on allowed side
    (
        (0, 1.51, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.ConvexPolyhedron,
        [hoomd.wall.Plane((0, 1, 0), (0, 1, 0))],
        cube_def,
        False,
    ),
    # cube barely on allowed side but rotated to intersect wall
    (
        (0, 1.51, 0),
        rot_x_45deg,
        hoomd.hpmc.integrate.ConvexPolyhedron,
        [hoomd.wall.Plane((0, 1, 0), (0, 1, 0))],
        cube_def,
        True,
    ),
    # cube all the way on forbidden side of planar wall
    (
        (0, 1.51, 0),
        rot_x_45deg,
        hoomd.hpmc.integrate.ConvexPolyhedron,
        [hoomd.wall.Plane((0, -1, 0), (0, -1, 0))],
        cube_def,
        True,
    ),
    # spherocube with face parallel to wall and barely on overlapping
    (
        (0, 1.51, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.ConvexSpheropolyhedron,
        [hoomd.wall.Plane((0, 1, 0), (0, 1, 0))],
        spherocube_def,
        True,
    ),
    # spherocube with face parallel to wall and barely on allowed side
    (
        (0, 1.61, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.ConvexSpheropolyhedron,
        [hoomd.wall.Plane((0, 1, 0), (0, 1, 0))],
        spherocube_def,
        False,
    ),
    # spherocube barely on allowed side but rotated to intersect wall
    (
        (0, 1.61, 0),
        rot_x_45deg,
        hoomd.hpmc.integrate.ConvexSpheropolyhedron,
        [hoomd.wall.Plane((0, 1, 0), (0, 1, 0))],
        spherocube_def,
        True,
    ),
    # spherocube all the way on forbidden side of planar wall
    (
        (0, 1.61, 0),
        rot_x_45deg,
        hoomd.hpmc.integrate.ConvexSpheropolyhedron,
        [hoomd.wall.Plane((0, -1, 0), (0, -1, 0))],
        spherocube_def,
        True,
    ),
    # sphere in middle of cylindrical pore with larger radius than sphere
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.Sphere,
        [hoomd.wall.Cylinder(1.0, (1, 0, 0))],
        dict(diameter=1.0),
        False,
    ),
    # make sure translating cylinder wall along normal does not affect overlaps
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.Sphere,
        [hoomd.wall.Cylinder(1.0, (1, 0, 0), origin=(2, 0, 0))],
        dict(diameter=1.0),
        False,
    ),
    # sphere in middle of cylindrical pore with smaller radius than sphere
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.Sphere,
        [hoomd.wall.Cylinder(0.4, (1, 0, 0))],
        dict(diameter=1.0),
        True,
    ),
    # sphere in forbidden inverse cylinder space
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.Sphere,
        [hoomd.wall.Cylinder(1.0, (1, 0, 0), origin=(0, 2, 0))],
        dict(diameter=1.0),
        True,
    ),
    # sphere in forbidden cylinder space
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.Sphere,
        [hoomd.wall.Cylinder(1.0, (1, 0, 0), inside=False)],
        dict(diameter=1.0),
        True,
    ),
    # sphere in allowed inverse cylinder space
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.Sphere,
        [hoomd.wall.Cylinder(1.0, (1, 0, 0), origin=(0, 3, 0), inside=False)],
        dict(diameter=1.0),
        False,
    ),

    # cube in middle of cylindrical pore with larger radius than the
    # circumsphere radius of the cube projected onto the circular cross-section
    # of the cylinder (i.e., the square face of the cube)
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.ConvexPolyhedron,
        [hoomd.wall.Cylinder(cube_face_rc * 1.01, (1, 0, 0))],
        cube_def,
        False,
    ),
    # cube in middle of cylindrical pore with smaller radius than the
    # circumsphere radius of the cube projected onto the circular cross-section
    # of the cylinder (i.e., the square face of the cube)
    (
        (0, 0, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.ConvexPolyhedron,
        [hoomd.wall.Cylinder(0.99 * cube_face_rc, (1, 0, 0))],
        cube_def,
        True,
    ),
    # cube in allowed inverse cylinder space
    (
        (0, 3, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.ConvexPolyhedron,
        [hoomd.wall.Cylinder(1.0, (1, 0, 0), inside=False)],
        cube_def,
        False,
    ),
    # cube center in allowed inverse cylinder space but vertices extend to
    # forbidden region
    (
        (0, 0.6, 0),
        (1, 0, 0, 0),
        hoomd.hpmc.integrate.ConvexPolyhedron,
        [hoomd.wall.Cylinder(1.0, (1, 0, 0), inside=False)],
        cube_def,
        True,
    ),
]


@pytest.mark.cpu
@pytest.mark.parametrize(
    "pos, orientation, shape, wall_list, shapedef, expecting_overlap",
    overlap_test_info)
def test_overlaps(simulation_factory, one_particle_snapshot_factory,
                  add_default_integrator, pos, orientation, shape, wall_list,
                  shapedef, expecting_overlap):
    sim = simulation_factory(
        one_particle_snapshot_factory(position=pos,
                                      orientation=orientation,
                                      L=100))
    mc, walls = add_default_integrator(sim,
                                       shape,
                                       wall_list,
                                       use_default_wall_args=False)
    mc.shape['A'] = shapedef
    sim.run(0)
    assert (mc.external_potential.overlaps > 0) == expecting_overlap
