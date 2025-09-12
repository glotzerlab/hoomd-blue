import hoomd
from hoomd.mesh3D import Mesh3D
import pytest
import numpy
from hoomd.snapshot import Snapshot
from hoomd import Simulation

def test_elastic(device, simulation_factory):
    reference_positions = numpy.array([[0, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1],
                             [1, 1, 1]])
    positions = reference_positions.copy()
    tetrahedra = numpy.array([[0, 1, 2, 3],
                              [1, 2, 3, 4]])
    types = ["A", "B"]
    type_ids = [0, 1]
    reference_tags = [0,1,2,3,4]

    mesh = Mesh3D(reference_positions, tetrahedra, types, type_ids)

    snapshot = Snapshot(device.communicator)
    if snapshot.communicator.rank == 0:
        snapshot.particles.N = len(positions)
        snapshot.particles.position[:] = positions
        snapshot.particles.types = ["A"]
        snapshot.configuration.box = [10,10,10,0,0,0]
    sim = simulation_factory(snapshot)

    elastic = hoomd.md.mesh.Elastic(mesh, reference_tags)
    elastic.params["A"] = {"C_xxxx": 1.0, "C_xxyy": 1.0, "C_xyxy": 1.0}
    elastic.params["B"] = {"C_xxxx": 2.0, "C_xxyy": 2.0, "C_xyxy": 2.0}

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(elastic)

    sim.operations.integrator = integrator
    sim.run(0)

    assert elastic.energy == pytest.approx(0, abs = 1e-4)
