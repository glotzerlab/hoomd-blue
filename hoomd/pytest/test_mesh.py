# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from hoomd.mesh import Mesh
import numpy
import pytest
from hoomd.error import DataAccessError, MutabilityError


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
        mesh.bonds, numpy.array([[0, 1], [1, 2], [0, 2], [2, 3], [1, 3]])
    )
