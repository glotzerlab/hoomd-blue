# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.conftest import pickling_check, autotuned_kernel_parameter_check
import numpy
import pytest


@pytest.fixture(scope='session')
def polymer_snapshot_factory(device):
    """Make a snapshot with polymers and distance constraints."""

    def make_snapshot(polymer_length=10,
                      N_polymers=10,
                      polymer_spacing=1.2,
                      bead_spacing=1.1):
        """Make the snapshot.

        Args:
            polymer_length: Number of particles in each polymer
            N_polymers: Number of polymers to place
            polymer_spacing: distance between the polymers
            bead_spacing: distance between the beads in the polymer

        Place N_polymers polymers in a 2D simulation with distance constraints
        between beads in each polymer.
        """
        s = hoomd.Snapshot(device.communicator)

        if s.communicator.rank == 0:
            s.configuration.box = [
                polymer_spacing * N_polymers, bead_spacing * polymer_length, 0,
                0, 0, 0
            ]
            s.particles.N = polymer_length * N_polymers
            s.particles.types = ['A']
            x_coords = numpy.linspace(-polymer_spacing * N_polymers / 2,
                                      polymer_spacing * N_polymers / 2,
                                      num=N_polymers,
                                      endpoint=False) + polymer_spacing / 2
            y_coords = numpy.linspace(-bead_spacing * polymer_length / 2,
                                      bead_spacing * polymer_length / 2,
                                      num=N_polymers,
                                      endpoint=False) + bead_spacing / 2

            position = []
            constraint_values = []
            constraint_groups = []

            for x in x_coords:
                for i, y in enumerate(y_coords):
                    position.append([x, y, 0])
                    if i & 1:
                        constraint_values.append(bead_spacing)
                        tag = len(position) - 1
                        constraint_groups.append([tag, tag - 1])

            s.particles.position[:] = position
            s.constraints.N = len(constraint_values)
            s.constraints.value[:] = constraint_values
            s.constraints.group[:] = constraint_groups

        return s

    return make_snapshot


def test_attach_detach(simulation_factory, polymer_snapshot_factory):
    """Ensure that md.constrain.Distance can be attached.

    Also test that parameters can be set.
    """
    # detached
    d = hoomd.md.constrain.Distance(tolerance=1e-5)

    assert d.tolerance == 1e-5
    d.tolerance = 1e-3
    assert d.tolerance == 1e-3

    # attached
    sim = simulation_factory(polymer_snapshot_factory())
    integrator = hoomd.md.Integrator(dt=0.005)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator.methods.append(nve)
    integrator.constraints.append(d)

    sim.run(0)

    assert d.tolerance == 1e-3
    d.tolerance = 1e-5
    assert d.tolerance == 1e-5


def test_pickling(simulation_factory, polymer_snapshot_factory):
    """Test that md.constrain.Distance can be pickled and unpickled."""
    # detached
    d = hoomd.md.constrain.Distance(tolerance=1e-5)
    pickling_check(d)

    # attached
    sim = simulation_factory(polymer_snapshot_factory())
    integrator = hoomd.md.Integrator(dt=0.005)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator.methods.append(nve)
    integrator.constraints.append(d)

    sim.run(0)
    pickling_check(d)


def test_basic_simulation(simulation_factory, polymer_snapshot_factory):
    """Ensure that distances are constrained in a basic simulation."""
    d = hoomd.md.constrain.Distance()

    sim = simulation_factory(polymer_snapshot_factory())
    integrator = hoomd.md.Integrator(dt=0.005)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator.methods.append(nve)
    integrator.constraints.append(d)

    cell = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
    lj.r_cut[('A', 'A')] = 2**(1 / 6)

    integrator.forces.append(lj)
    sim.operations.integrator = integrator

    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
    sim.run(10)

    snap = sim.state.get_snapshot()

    if snap.communicator.rank == 0:
        # compute bond lengths in unwrapped particle coordinates
        box_lengths = snap.configuration.box[0:3]
        r = snap.particles.position + snap.particles.image * box_lengths
        constraints = snap.constraints.group

        delta_r = r[constraints[:, 1]] - r[constraints[:, 0]]
        bond_lengths = numpy.sqrt(numpy.sum(delta_r * delta_r, axis=1))

        numpy.testing.assert_allclose(bond_lengths,
                                      snap.constraints.value,
                                      rtol=1e-5)

    autotuned_kernel_parameter_check(instance=d, activate=lambda: sim.run(1))
