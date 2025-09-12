# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.mesh3D import Mesh3D
import numpy
import pytest
from hoomd.snapshot import Snapshot
from hoomd import Simulation
from hoomd.error import DataAccessError, MutabilityError


def test_mesh_setter(device, simulation_factory):
    reference_positions = numpy.array([[0, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0],
                            [0, 0, 1],
                             [1, 1, 1]])
    tetrahedra = numpy.array([[0, 1, 2, 3],
                              [1, 2, 3, 4]])
    types = ["A", "B"]
    type_ids = [0, 1]
    reference_tags = [0,1,2,3,4]

    mesh = Mesh3D(reference_positions, tetrahedra, types, type_ids, reference_tags)
    snapshot = Snapshot(device.communicator)
    if snapshot.communicator.rank == 0:
        snapshot.particles.N = len(reference_positions)
        snapshot.particles.position[:] = reference_positions
        snapshot.particles.types = ["A"]
        snapshot.configuration.box = [10,10,10,0,0,0]

    sim = simulation_factory(snapshot)
    mesh._attach(sim)
