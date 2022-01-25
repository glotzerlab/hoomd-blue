# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.mesh import Mesh
import numpy
import pytest
from hoomd.error import DataAccessError, MutabilityError, TypeConversionError


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
    if sim.device.communicator.rank == 0:
        assert mesh.size == 0
        assert mesh.types == ["mesh"]
        assert len(mesh.triangles) == 0
        with pytest.raises(DataAccessError):
            mesh.bonds == 0

        mesh._add(sim)
        mesh._attach()

        assert mesh.size == 0
        assert mesh.types == ["mesh"]
        assert len(mesh.triangles) == 0
        assert len(mesh.bonds) == 0


def test_mesh_setter():
    mesh = Mesh(name=["vesicle"])

    mesh.size = 1
    assert mesh.size == 1

    with pytest.raises(TypeConversionError):
        mesh.types = "mesh"
    with pytest.raises(TypeConversionError):
        mesh.types = ["me", "sh"]
    mesh.types = ["mesh"]
    assert mesh.types == ["mesh"]

    mesh.triangles = numpy.array([[0, 1, 2], [1, 2, 3]])

    assert mesh.size == 2
    assert numpy.array_equal(mesh.triangles, numpy.array([[0, 1, 2], [1, 2,
                                                                      3]]))


def test_mesh_setter_attached(simulation_factory, mesh_snapshot_factory):
    sim = simulation_factory(mesh_snapshot_factory(d=0.969, L=5))
    mesh = Mesh()

    mesh._add(sim)
    mesh._attach()

    if sim.device.communicator.rank == 0:
        with pytest.raises(MutabilityError):
            mesh.types = ["vesicle"]
        with pytest.raises(MutabilityError):
            mesh.size = 3

        mesh.triangles = numpy.array([[0, 1, 2], [1, 2, 3]])

        assert mesh.size == 2
        assert numpy.array_equal(mesh.triangles,
                                 numpy.array([[0, 1, 2], [1, 2, 3]]))
        assert numpy.array_equal(
            mesh.bonds, numpy.array([[0, 1], [1, 2], [2, 0], [2, 3], [3, 1]]))
