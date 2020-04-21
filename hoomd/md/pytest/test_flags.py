import hoomd
import pytest
import numpy
import itertools


def test_per_particle_virial(simulation_factory, lattice_snapshot_factory):
    cell = hoomd.md.nlist.Cell()
    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[('A', 'A')] = dict(sigma=1.0, epsilon=1.0)
    lj.r_cut[('A', 'A')] = 2.5

    a = 2**(1.0 / 6.0)
    sim = simulation_factory(lattice_snapshot_factory(n=20, a=a, r=a * 0.01))

    assert sim.always_compute_pressure == False

    # TODO: allow forces to be added directly
    # sim.operations.add(lj)

    # For now, add to an integrator
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005)
    sim.operations.integrator.forces.append(lj)

    sim.operations.schedule()

    assert sim.always_compute_pressure == False

    # virials should be None at first
    virials = lj.virials

    if sim.device.comm.rank == 0:
        assert virials is None

    # virials should be non-zero after setting flags
    sim.always_compute_pressure = True

    virials = lj.virials

    if sim.device.comm.rank == 0:
        assert numpy.sum(virials * virials) > 0.0


# TODO: test compute thermo once it is implemented
