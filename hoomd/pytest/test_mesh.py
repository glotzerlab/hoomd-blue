# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.mesh import Mesh
import numpy
import pytest
from hoomd.error import DataAccessError, MutabilityError


@pytest.fixture(scope='session')
def mesh_snapshot_factory(device):

    def make_snapshot(d=1.0, phi_deg=45, particle_types=['A'], L=20):
        phi_rad = phi_deg * (numpy.pi / 180)
        # the central particles are along the x-axis, so phi is determined from
        # the angle in the yz plane.

        s = hoomd.Snapshot(device.communicator)
        N = 4
        if s.communicator.rank == 0:
            box = [L, L, L, 0, 0, 0]
            s.configuration.box = box
            s.particles.N = N
            s.particles.types = particle_types
            # shift particle positions slightly in z so MPI tests pass
            s.particles.position[:] = [[
                0.0, d * numpy.cos(phi_rad / 2),
                d * numpy.sin(phi_rad / 2) + 0.1
            ], [0.0, 0.0, 0.1], [d, 0.0, 0.1],
                                       [
                                           d, d * numpy.cos(phi_rad / 2),
                                           -d * numpy.sin(phi_rad / 2) + 0.1
                                       ]]

        return s

    return make_snapshot


def test_empty_mesh(simulation_factory, two_particle_snapshot_factory):

    sim = simulation_factory(two_particle_snapshot_factory(d=2.0))
    mesh = Mesh()

    assert mesh.size == 0
    assert mesh.types[0] == "mesh"
    assert len(mesh.triangulation["triangles"]) == 0
    assert len(mesh.triangulation["type_ids"]) == 0
    with pytest.raises(DataAccessError):
        mesh.bonds
    with pytest.raises(DataAccessError):
        mesh.triangles
    with pytest.raises(DataAccessError):
        mesh.type_ids

    mesh._attach(sim)

    assert mesh.size == 0
    assert mesh.types[0] == "mesh"
    assert len(mesh.triangulation["triangles"]) == 0
    assert len(mesh.triangulation["type_ids"]) == 0
    assert len(mesh.triangles) == 0
    assert len(mesh.bonds) == 0
    assert len(mesh.type_ids) == 0


def test_mesh_setter():
    mesh = Mesh()

    mesh.types = ["vesicle", "patch"]
    assert mesh.types == ["vesicle", "patch"]

    mesh_type_ids = numpy.array([0, 1])

    mesh_triangles = numpy.array([[0, 1, 2], [1, 2, 3]])

    with pytest.raises(ValueError):
        mesh.triangulation = dict(type_ids=[0], triangles=mesh_triangles)

    mesh.triangulation = dict(type_ids=mesh_type_ids, triangles=mesh_triangles)

    assert mesh.size == 2
    assert numpy.array_equal(mesh.triangulation["triangles"], mesh_triangles)
    assert numpy.array_equal(mesh.triangulation["type_ids"], mesh_type_ids)


def test_mesh_setter_attached(simulation_factory, mesh_snapshot_factory):
    sim = simulation_factory(mesh_snapshot_factory(d=0.969, L=5))
    mesh = Mesh()

    mesh._attach(sim)

    with pytest.raises(MutabilityError):
        mesh.types = ["vesicle"]
    with pytest.raises(AttributeError):
        mesh.size = 3

    mesh_type_ids = numpy.array([0, 0])

    mesh_triangles = numpy.array([[0, 1, 2], [1, 2, 3]])

    with pytest.raises(ValueError):
        mesh.triangulation = dict(type_ids=[0], triangles=mesh_triangles)

    mesh.triangulation = dict(type_ids=mesh_type_ids, triangles=mesh_triangles)

    assert mesh.size == 2
    assert numpy.array_equal(mesh.triangulation["triangles"], mesh_triangles)
    assert numpy.array_equal(mesh.triangulation["type_ids"], mesh_type_ids)
    assert numpy.array_equal(mesh.triangles, mesh_triangles)
    assert numpy.array_equal(mesh.type_ids, mesh_type_ids)
    assert numpy.array_equal(
        mesh.bonds, numpy.array([[0, 1], [1, 2], [0, 2], [2, 3], [1, 3]]))
