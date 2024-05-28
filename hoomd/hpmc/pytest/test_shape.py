# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from collections.abc import Sequence

import hoomd
from hoomd.conftest import (operation_pickling_check, logging_check,
                            autotuned_kernel_parameter_check)
from hoomd.error import DataAccessError
import hoomd.hpmc
import numpy as np
import pytest
import hoomd.hpmc.pytest.conftest
from hoomd.logging import LoggerCategories
from copy import deepcopy


def check_dict(shape_dict, args):
    """Check that two dictionaries are equivalent.

    Arguments: shape_dict and args - dictionaries to test

    Ex: mc = hoomd.hpmc.integrate.Sphere()
        mc.shape["A"] = {"diameter": 1}
        check_dict(mc.shape["A"], {"diameter": 1})

    Useful for more complex nested dictionaries (like the shape key in unions)
    Used to test that the dictionary passed in is what gets passed out
    """
    for key, val in args.items():
        if isinstance(shape_dict[key], list) and len(shape_dict[key]) > 0 \
           and key != 'shapes':
            np.testing.assert_allclose(shape_dict[key], val)
        elif key == 'shapes':
            for i in range(len(shape_dict[key])):
                shape_args = shape_dict[key][i]
                val_args = val[i]
                for shape_key in shape_args:
                    if isinstance(shape_args[shape_key], Sequence):
                        np.testing.assert_allclose(val_args[shape_key],
                                                   shape_args[shape_key])
                    else:
                        assert shape_args[shape_key] == val_args[shape_key]
        else:
            np.testing.assert_almost_equal(shape_dict[key], val)


def test_dict_conversion(cpp_args):
    shape_params = cpp_args[0]
    args = cpp_args[1]
    test_shape = shape_params(args)
    test_dict = test_shape.asDict()
    check_dict(test_dict, args)


def test_valid_shape_params(valid_args):
    integrator = valid_args[0]
    args = valid_args[1]
    # Need to unpack union integrators
    if isinstance(integrator, tuple):
        inner_integrator = integrator[0]
        integrator = integrator[1]
        inner_mc = inner_integrator()
        for i in range(len(args["shapes"])):
            # This will fill in default values for the inner shape objects
            inner_mc.shape["A"] = args["shapes"][i]
            args["shapes"][i] = inner_mc.shape["A"].to_base()
    mc = integrator()
    mc.shape["A"] = args
    check_dict(mc.shape["A"], args)


def test_invalid_shape_params(invalid_args):
    integrator = invalid_args[0]
    if isinstance(integrator, tuple):
        integrator = integrator[1]
    args = invalid_args[1]
    mc = integrator()
    with pytest.raises(hoomd.error.TypeConversionError):
        mc.shape["A"] = args


@pytest.mark.parametrize(
    "cpp_shape",
    [hoomd.hpmc._hpmc.EllipsoidParams, hoomd.hpmc._hpmc.FacetedEllipsoidParams])
@pytest.mark.parametrize("c", [0.0, -0.5])
def test_semimajor_axis_validity(cpp_shape, c):
    args = {
        'a': 0.125,
        'b': 0.375,
        'c': c,
        # These properties are only read for the FacetedEllipsoid
        'normals': [],
        'offsets': [],
        'vertices': [],
        'origin': []
    }
    with pytest.raises(ValueError) as err:
        cpp_shape({"ignore_statistics": False} | args)

    assert ("All semimajor axes must be nonzero!" in str(err))


def test_shape_attached(simulation_factory, two_particle_snapshot_factory,
                        valid_args):
    integrator = valid_args[0]
    args = valid_args[1]
    n_dimensions = valid_args[2]
    if isinstance(integrator, tuple):
        inner_integrator = integrator[0]
        integrator = integrator[1]
        inner_mc = inner_integrator()
        for i in range(len(args["shapes"])):
            # This will fill in default values for the inner shape objects
            inner_mc.shape["A"] = args["shapes"][i]
            args["shapes"][i] = inner_mc.shape["A"].to_base()
    mc = integrator()
    mc.shape["A"] = args
    sim = simulation_factory(
        two_particle_snapshot_factory(dimensions=n_dimensions))
    assert sim.operations.integrator is None
    sim.operations.add(mc)
    sim.operations._schedule()
    check_dict(mc.shape["A"], args)


def test_moves(simulation_factory, lattice_snapshot_factory, test_moves_args):
    integrator = test_moves_args[0]
    args = test_moves_args[1]
    n_dimensions = test_moves_args[2]
    mc = integrator()
    mc.shape['A'] = args

    sim = simulation_factory(lattice_snapshot_factory(dimensions=n_dimensions))
    sim.operations.add(mc)

    with pytest.raises(DataAccessError):
        sim.operations.integrator.translate_moves
    with pytest.raises(DataAccessError):
        sim.operations.integrator.rotate_moves
    sim.operations._schedule()

    assert sum(sim.operations.integrator.translate_moves) == 0
    assert sum(sim.operations.integrator.rotate_moves) == 0

    sim.run(10)
    accepted_rejected_trans = sum(sim.operations.integrator.translate_moves)
    assert accepted_rejected_trans > 0
    if 'sphere' not in str(integrator).lower():
        accepted_rejected_rot = sum(sim.operations.integrator.rotate_moves)
        assert accepted_rejected_rot > 0


def test_kernel_parameters(simulation_factory, lattice_snapshot_factory,
                           test_moves_args):
    integrator = test_moves_args[0]

    if integrator == hoomd.hpmc.integrate.Sphinx:
        pytest.skip("Sphinx does not build on the GPU by default.")

    args = test_moves_args[1]
    n_dimensions = test_moves_args[2]
    mc = integrator()
    mc.shape['A'] = args

    sim = simulation_factory(lattice_snapshot_factory(dimensions=n_dimensions))
    sim.operations.add(mc)
    sim.run(0)

    autotuned_kernel_parameter_check(instance=mc, activate=lambda: sim.run(1))


# An ellipsoid with a = b = c should be a sphere
# A spheropolyhedron with a single vertex should be a sphere
# A sphinx where the indenting sphere is negligible should also be a sphere
_sphere_shapes = [({
    'diameter': 1
}, hoomd.hpmc.integrate.Sphere),
                  ({
                      'a': 0.5,
                      'b': 0.5,
                      'c': 0.5
                  }, hoomd.hpmc.integrate.Ellipsoid),
                  ({
                      'vertices': [(0, 0, 0)],
                      'sweep_radius': 0.5
                  }, hoomd.hpmc.integrate.ConvexSpheropolyhedron),
                  ({
                      'diameters': [1, -0.0001],
                      'centers': [(0, 0, 0), (0, 0, 0.5)]
                  }, hoomd.hpmc.integrate.Sphinx)]


@pytest.fixture(scope="function", params=_sphere_shapes)
def sphere_overlap_args(request):
    return deepcopy(request.param)


def test_overlaps_sphere(device, sphere_overlap_args, simulation_factory,
                         two_particle_snapshot_factory):
    integrator_args = sphere_overlap_args[0]
    integrator = sphere_overlap_args[1]

    mc = integrator()
    mc.shape["A"] = integrator_args
    diameter = 1

    # Should overlap when spheres are less than one diameter apart
    sim = simulation_factory(
        two_particle_snapshot_factory(dimensions=3, d=diameter * 0.9))
    sim.operations.add(mc)
    sim.operations._schedule()
    assert mc.overlaps > 0

    # Should not overlap when spheres are larger than one diameter apart
    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, diameter * 1.1, 0)
    sim.state.set_snapshot(s)
    assert mc.overlaps == 0

    # Should barely overlap when spheres are exactly than one diameter apart
    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, diameter * 0.9999, 0)
    sim.state.set_snapshot(s)
    assert mc.overlaps == 1


def test_overlaps_ellipsoid(device, simulation_factory,
                            two_particle_snapshot_factory):
    a = 1 / 4
    b = 1 / 2
    c = 1
    mc = hoomd.hpmc.integrate.Ellipsoid()
    mc.shape["A"] = {'a': a, 'b': b, 'c': c}

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=2))
    sim.operations.add(mc)
    sim.operations._schedule()
    assert mc.overlaps == 0

    abc_list = [(0, 0, c), (0, b, 0), (a, 0, 0)]
    for abc in abc_list:
        # Should barely overlap when ellipsoids are exactly than one diameter
        # apart
        s = sim.state.get_snapshot()
        if s.communicator.rank == 0:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = (abc[0] * 0.9 * 2, abc[1] * 0.9 * 2,
                                       abc[2] * 0.9 * 2)
        sim.state.set_snapshot(s)
        assert mc.overlaps == 1

        # Should not overlap when ellipsoids are larger than one diameter apart
        s = sim.state.get_snapshot()
        if s.communicator.rank == 0:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = (abc[0] * 1.15 * 2, abc[1] * 1.15 * 2,
                                       abc[2] * 1.15 * 2)
        sim.state.set_snapshot(s)
        assert mc.overlaps == 0

    # Line up ellipsoids where they aren't overlapped, and then rotate one so
    # they overlap
    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (a * 1.1 * 2, 0, 0)
        s.particles.orientation[1] = tuple(
            np.array([1, 0, 0.45, 0]) / (1.2025**0.5))
    sim.state.set_snapshot(s)
    assert mc.overlaps > 0


_triangle = {
    'vertices': [(0, (0.75**0.5) / 2), (-0.5, -(0.75**0.5) / 2),
                 (0.5, -(0.75**0.5) / 2)]
}
_square = {"vertices": np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)]) / 2}
# Args should work for ConvexPolygon, SimplePolygon, and ConvexSpheropolygon
_polygon_shapes = [(_triangle, hoomd.hpmc.integrate.ConvexPolygon),
                   (_triangle, hoomd.hpmc.integrate.SimplePolygon),
                   (_triangle, hoomd.hpmc.integrate.ConvexSpheropolygon),
                   (_square, hoomd.hpmc.integrate.ConvexPolygon),
                   (_square, hoomd.hpmc.integrate.SimplePolygon),
                   (_square, hoomd.hpmc.integrate.ConvexSpheropolygon)]


@pytest.fixture(scope="function", params=_polygon_shapes)
def polygon_overlap_args(request):
    return deepcopy(request.param)


def test_overlaps_polygons(device, polygon_overlap_args, simulation_factory,
                           two_particle_snapshot_factory):
    integrator_args = polygon_overlap_args[0]
    integrator = polygon_overlap_args[1]
    mc = integrator()
    mc.shape['A'] = integrator_args

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=2))
    sim.operations.add(mc)
    sim.operations._schedule()
    assert mc.overlaps == 0

    # Place center of shape 2 on each of shape 1's vertices
    for vert in mc.shape["A"]["vertices"]:
        s = sim.state.get_snapshot()
        if s.communicator.rank == 0:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = (vert[0], vert[1], 0)
        sim.state.set_snapshot(s)
        assert mc.overlaps > 0

    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, 1.05, 0)
    sim.state.set_snapshot(s)
    assert mc.overlaps == 0

    # Rotate one of the shapes so they will overlap
    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.orientation[1] = tuple(
            np.array([1, 0, 0, 0.45]) / (1.2025**0.5))
    sim.state.set_snapshot(s)
    assert mc.overlaps > 0


_tetrahedron_verts = np.array([(1, 1, 1), (-1, -1, 1), (1, -1, -1),
                               (-1, 1, -1)]) / 2

_tetrahedron_faces = [[1, 3, 2], [3, 0, 2], [1, 0, 3], [1, 2, 0]]

_cube_verts = [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5),
               (-0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5),
               (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)]

_cube_faces = [[0, 2, 6], [6, 4, 0], [5, 0, 4], [5, 1, 0], [5, 4, 6], [5, 6, 7],
               [3, 2, 0], [3, 0, 1], [3, 6, 2], [3, 7, 6], [3, 1, 5], [3, 5, 7]]

# Test args with ConvexPolyhedron, ConvexSpheropolyhedron, and Polyhedron
_polyhedron_shapes = [({
    "vertices": _tetrahedron_verts
}, hoomd.hpmc.integrate.ConvexPolyhedron),
                      ({
                          "vertices": _tetrahedron_verts
                      }, hoomd.hpmc.integrate.ConvexSpheropolyhedron),
                      ({
                          "vertices": _tetrahedron_verts,
                          "faces": _tetrahedron_faces
                      }, hoomd.hpmc.integrate.Polyhedron),
                      ({
                          "vertices": _cube_verts
                      }, hoomd.hpmc.integrate.ConvexPolyhedron),
                      ({
                          "vertices": _cube_verts
                      }, hoomd.hpmc.integrate.ConvexSpheropolyhedron),
                      ({
                          "vertices": _cube_verts,
                          "faces": _cube_faces
                      }, hoomd.hpmc.integrate.Polyhedron)]


@pytest.fixture(scope="function", params=_polyhedron_shapes)
def polyhedron_overlap_args(request):
    return deepcopy(request.param)


def test_overlaps_polyhedra(device, polyhedron_overlap_args, simulation_factory,
                            two_particle_snapshot_factory):
    integrator_args = polyhedron_overlap_args[0]
    integrator = polyhedron_overlap_args[1]

    mc = integrator()
    mc.shape['A'] = integrator_args
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=2))
    sim.operations.add(mc)
    sim.operations._schedule()
    assert mc.overlaps == 0

    # Place center of shape 2 on each of shape 1's vertices
    for vert in mc.shape["A"]["vertices"]:
        s = sim.state.get_snapshot()
        if s.communicator.rank == 0:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = vert
        sim.state.set_snapshot(s)
        assert mc.overlaps > 0

    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, 0.9, 0)
    sim.state.set_snapshot(s)
    assert mc.overlaps > 0

    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, 1.1, 0)
    sim.state.set_snapshot(s)
    assert mc.overlaps == 0

    # Rotate one of the polyhedra so they will overlap
    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.orientation[1] = tuple(np.array([1, 1, 1, 0]) / (3**0.5))
    sim.state.set_snapshot(s)
    assert mc.overlaps > 0


_spheropolygon_shapes = [{
    'vertices': _triangle['vertices'],
    'sweep_radius': 0.2
}, {
    'vertices': _square['vertices'],
    'sweep_radius': 0.1
}]


@pytest.fixture(scope="function", params=_spheropolygon_shapes)
def spheropolygon_overlap_args(request):
    return deepcopy(request.param)


def test_overlaps_spheropolygon(device, spheropolygon_overlap_args,
                                simulation_factory,
                                two_particle_snapshot_factory):
    mc = hoomd.hpmc.integrate.ConvexSpheropolygon()
    mc.shape['A'] = spheropolygon_overlap_args

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=2, d=2))
    sim.operations.add(mc)
    sim.operations._schedule()
    assert mc.overlaps == 0

    # Place center of shape 2 on each of shape 1's vertices
    for vert in mc.shape["A"]["vertices"]:
        s = sim.state.get_snapshot()
        if s.communicator.rank == 0:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = (vert[0], vert[1], 0)
        sim.state.set_snapshot(s)
        assert mc.overlaps > 0

    # Place shapes where they wouldn't overlap w/o sweep radius
    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, 1.2, 0)
    sim.state.set_snapshot(s)
    assert mc.overlaps > 0

    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, 1.3, 0)
    sim.state.set_snapshot(s)
    assert mc.overlaps == 0

    # Rotate one of the shapes so they will overlap
    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.orientation[1] = tuple(
            np.array([1, 0, 0, 0.45]) / (1.2025**0.5))
    sim.state.set_snapshot(s)
    assert mc.overlaps > 0


_spheropolyhedron_shapes = [{
    'vertices': _tetrahedron_verts,
    'sweep_radius': 0.2
}, {
    'vertices': _cube_verts,
    'sweep_radius': 0.2
}]


@pytest.fixture(scope="function", params=_spheropolyhedron_shapes)
def spheropolyhedron_overlap_args(request):
    return deepcopy(request.param)


def test_overlaps_spheropolyhedron(device, spheropolyhedron_overlap_args,
                                   simulation_factory,
                                   two_particle_snapshot_factory):
    mc = hoomd.hpmc.integrate.ConvexSpheropolyhedron()
    mc.shape['A'] = spheropolyhedron_overlap_args

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=2))
    sim.operations.add(mc)
    sim.operations._schedule()
    assert mc.overlaps == 0

    # Place center of shape 2 on each of shape 1's vertices
    for vert in mc.shape["A"]["vertices"]:
        s = sim.state.get_snapshot()
        if s.communicator.rank == 0:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = vert
        sim.state.set_snapshot(s)
        assert mc.overlaps > 0

    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, 1.2, 0)
    sim.state.set_snapshot(s)
    assert mc.overlaps > 0

    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, 1.5, 0)
    sim.state.set_snapshot(s)
    assert mc.overlaps == 0

    # Rotate one of the polyhedra so they will overlap
    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.orientation[1] = tuple(np.array([1, 1, 1, 0]) / (3**0.5))
    sim.state.set_snapshot(s)
    assert mc.overlaps > 0


_union_shapes = [
    ({
        'diameter': 1
    }, hoomd.hpmc.integrate.SphereUnion),
    ({
        "vertices": _tetrahedron_verts
    }, hoomd.hpmc.integrate.ConvexSpheropolyhedronUnion),
    ({
        "normals": [(0, 0, 1)],
        "a": 0.5,
        "b": 0.5,
        "c": 1,
        "vertices": [],
        "origin": (0, 0, 0),
        "offsets": [0]
    }, hoomd.hpmc.integrate.FacetedEllipsoidUnion),
]


@pytest.fixture(scope="function", params=_union_shapes)
def union_overlap_args(request):
    return deepcopy(request.param)


def test_overlaps_union(device, union_overlap_args, simulation_factory,
                        two_particle_snapshot_factory):
    inner_args = union_overlap_args[0]
    integrator = union_overlap_args[1]

    union_args = {
        'shapes': [inner_args, inner_args],
        'positions': [(0, 0, 0), (0, 0, 1)]
    }
    mc = integrator()
    mc.shape['A'] = union_args

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=2))
    sim.operations.add(mc)
    sim.operations._schedule()

    assert mc.overlaps == 0
    test_positions = [(1.1, 0, 0), (0, 1.1, 0)]
    test_orientations = np.array([[1, 0, -0.06, 0], [1, 0.06, 0, 0]])
    test_orientations = test_orientations.T / np.linalg.norm(test_orientations,
                                                             axis=1)
    test_orientations = test_orientations.T
    # Shapes are stacked in z direction
    for i in range(len(test_positions)):
        s = sim.state.get_snapshot()
        if s.communicator.rank == 0:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = test_positions[i]
            s.particles.orientation[1] = (1, 0, 0, 0)
        sim.state.set_snapshot(s)
        assert mc.overlaps == 0

        # Slightly rotate union about x or y axis so they overlap
        if s.communicator.rank == 0:
            s.particles.orientation[1] = test_orientations[i]
        sim.state.set_snapshot(s)

        assert mc.overlaps > 0

    for pos in [(0.9, 0, 0), (0, 0.9, 0), (0, 0, 1.1)]:
        s = sim.state.get_snapshot()
        if s.communicator.rank == 0:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = pos
        sim.state.set_snapshot(s)
        assert mc.overlaps > 0


def test_overlaps_faceted_ellipsoid(device, simulation_factory,
                                    two_particle_snapshot_factory):
    a = 1 / 2
    b = 1 / 2
    c = 1
    mc = hoomd.hpmc.integrate.FacetedEllipsoid()
    mc.shape['A'] = {
        "normals": [(0, 0, 1)],
        "a": 0.5,
        "b": 0.5,
        "c": 1,
        "vertices": [],
        "origin": (0, 0, 0),
        "offsets": [0]
    }

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=2))
    sim.operations.add(mc)
    sim.operations._schedule()
    assert mc.overlaps == 0

    abc_list = [(0, 0, c / 2), (0, b, 0), (a, 0, 0)]
    for abc in abc_list:
        # Should barely overlap when ellipsoids are exactly than one diameter
        # apart
        s = sim.state.get_snapshot()
        if s.communicator.rank == 0:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = (abc[0] * 0.9 * 2, abc[1] * 0.9 * 2,
                                       abc[2] * 0.9 * 2)
        sim.state.set_snapshot(s)
        assert mc.overlaps == 1

        # Should not overlap when ellipsoids are larger than one diameter apart
        s = sim.state.get_snapshot()
        if s.communicator.rank == 0:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = (abc[0] * 1.15 * 2, abc[1] * 1.15 * 2,
                                       abc[2] * 1.15 * 2)
        sim.state.set_snapshot(s)
        assert mc.overlaps == 0

    # Line up ellipsoids where they aren't overlapped, and then rotate one so
    # they overlap
    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (a * 1.1 * 2, 0, 0)
        s.particles.orientation[1] = tuple(
            np.array([1, 0, 0.45, 0]) / (1.2025**0.5))
    sim.state.set_snapshot(s)
    assert mc.overlaps > 0


def test_overlaps_sphinx(device, simulation_factory,
                         two_particle_snapshot_factory):

    mc = hoomd.hpmc.integrate.Sphinx()
    mc.shape["A"] = {'diameters': [1, -1], 'centers': [(0, 0, 0), (0.75, 0, 0)]}

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=2))
    sim.operations.add(mc)
    sim.operations._schedule()
    assert mc.overlaps == 0

    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0.74, 0, 0)
    sim.state.set_snapshot(s)
    assert mc.overlaps > 0

    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0.76, 0, 0)
    sim.state.set_snapshot(s)
    assert mc.overlaps == 0


def test_pickling(valid_args, simulation_factory,
                  two_particle_snapshot_factory):
    integrator = valid_args[0]
    args = valid_args[1]
    n_dimensions = valid_args[2]
    # Need to unpack union integrators
    if isinstance(integrator, tuple):
        inner_integrator = integrator[0]
        integrator = integrator[1]
        inner_mc = inner_integrator()
        for i in range(len(args["shapes"])):
            # This will fill in default values for the inner shape objects
            inner_mc.shape["A"] = args["shapes"][i]
            args["shapes"][i] = inner_mc.shape["A"].to_base()
    mc = integrator()
    mc.shape["A"] = args
    # L needs to be ridiculously large as to not be too small for the domain
    # decomposition of some of the shapes definitions in valid_args which have
    # shapes with large extent in at least one dimension.
    sim = simulation_factory(
        two_particle_snapshot_factory(L=1000, dimensions=n_dimensions))
    operation_pickling_check(mc, sim)


def test_logging():
    logging_check(
        hoomd.hpmc.integrate.HPMCIntegrator, ('hpmc', 'integrate'), {
            'map_overlaps': {
                'category': LoggerCategories.sequence,
                'default': True
            },
            'mps': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'overlaps': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'rotate_moves': {
                'category': LoggerCategories.sequence,
                'default': True
            },
            'translate_moves': {
                'category': LoggerCategories.sequence,
                'default': True
            }
        })

    integrators = (hoomd.hpmc.integrate.Sphere,
                   hoomd.hpmc.integrate.ConvexPolygon,
                   hoomd.hpmc.integrate.ConvexSpheropolygon,
                   hoomd.hpmc.integrate.Polyhedron,
                   hoomd.hpmc.integrate.ConvexPolyhedron,
                   hoomd.hpmc.integrate.ConvexSpheropolyhedron,
                   hoomd.hpmc.integrate.Ellipsoid,
                   hoomd.hpmc.integrate.SphereUnion)

    type_shapes_check = {
        'type_shapes': {
            'category': LoggerCategories.object,
            'default': True
        }
    }

    for integrator in integrators:
        logging_check(integrator, ('hpmc', 'integrate'), type_shapes_check)


# test_fugacity fails on the GPU for unknown reasons - not fixing as the
# implicit depletant code is slated for removal.
@pytest.mark.cpu
def test_fugacity(simulation_factory, two_particle_snapshot_factory,
                  test_moves_args):
    integrator = test_moves_args[0]
    args = test_moves_args[1]
    mc = integrator()
    mc.shape['A'] = args
    mc.depletant_fugacity["A"] = 0.1
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.run(2)
