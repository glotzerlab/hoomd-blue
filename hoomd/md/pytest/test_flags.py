# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import numpy


def test_per_particle_virial(simulation_factory, lattice_snapshot_factory):
    cell = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[('A', 'A')] = dict(sigma=1.0, epsilon=1.0)
    lj.r_cut[('A', 'A')] = 2.5

    a = 2**(1.0 / 6.0)
    sim = simulation_factory(lattice_snapshot_factory(n=20, a=a, r=a * 0.01))

    assert not sim.always_compute_pressure

    # TODO: allow forces to be added directly
    # sim.operations.add(lj)

    # For now, add to an integrator
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005)
    sim.operations.integrator.forces.append(lj)

    sim.operations._schedule()

    assert not sim.always_compute_pressure

    # virials should be None at first
    virials = lj.virials

    if sim.device.communicator.rank == 0:
        assert virials is None

    # virials should be non-zero after setting flags
    sim.always_compute_pressure = True

    virials = lj.virials

    if sim.device.communicator.rank == 0:
        assert numpy.sum(virials * virials) > 0.0


# TODO: test compute thermo once it is implemented
