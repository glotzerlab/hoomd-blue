# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.conftest import pickling_check, autotuned_kernel_parameter_check
import pytest
import numpy


@pytest.fixture(scope='session')
def two_charged_particle_snapshot_factory(two_particle_snapshot_factory):
    """Make a snapshot with two charged particles."""

    def make_snapshot(particle_types=['A'], dimensions=3, d=1, L=20, q=1):
        """Make the snapshot.

        Args:
            particle_types: List of particle type names
            dimensions: Number of dimensions (2 or 3)
            d: Distance apart to place particles
            L: Box length
            q: Particle charge
        """
        s = two_particle_snapshot_factory(particle_types=particle_types,
                                          dimensions=dimensions,
                                          d=d,
                                          L=L)

        if s.communicator.rank == 0:
            s.particles.charge[0] = -q
            s.particles.charge[1] = q
        return s

    return make_snapshot


def test_attach_detach(simulation_factory,
                       two_charged_particle_snapshot_factory):
    """Ensure that md.long_range.pppm.Coulomb can be attached.

    Also test that parameters can be set.
    """
    # detached
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    ewald, coulomb = hoomd.md.long_range.pppm.make_pppm_coulomb_forces(
        nlist=nlist, resolution=(64, 64, 64), order=6, r_cut=3.0, alpha=0)

    assert ewald.nlist is nlist
    assert coulomb.nlist is nlist
    assert coulomb.resolution == (64, 64, 64)
    assert coulomb.order == 6
    assert coulomb.r_cut == 3.0
    assert coulomb.alpha == 0

    nlist2 = hoomd.md.nlist.Tree(buffer=0.4)
    coulomb.nlist = nlist2
    assert coulomb.nlist is nlist2
    assert ewald.nlist is nlist2

    coulomb.resolution = (16, 32, 128)
    assert coulomb.resolution == (16, 32, 128)

    coulomb.order = 4
    assert coulomb.order == 4

    coulomb.r_cut = 2.5
    assert coulomb.r_cut == 2.5

    coulomb.alpha = 1.5
    assert coulomb.alpha == 1.5

    # attached
    sim = simulation_factory(two_charged_particle_snapshot_factory())
    integrator = hoomd.md.Integrator(dt=0.005)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator.methods.append(nve)
    integrator.forces.extend([ewald, coulomb])
    sim.operations.integrator = integrator

    sim.run(0)

    assert ewald._attached
    assert coulomb._attached

    assert coulomb.resolution == (16, 32, 128)
    assert coulomb.order == 4
    assert coulomb.r_cut == 2.5
    assert coulomb.alpha == 1.5

    assert ewald.params[('A', 'A')]['alpha'] == 1.5

    with pytest.raises(AttributeError):
        coulomb.resolution = (32, 32, 32)
    with pytest.raises(AttributeError):
        coulomb.order = 5
    with pytest.raises(AttributeError):
        coulomb.r_cut = 4.5
    with pytest.raises(AttributeError):
        coulomb.alpha = 3.0


def test_kernel_parameters(simulation_factory,
                           two_charged_particle_snapshot_factory):
    """Test that md.long_range.pppm.Coulomb can be pickled and unpickled."""
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    ewald, coulomb = hoomd.md.long_range.pppm.make_pppm_coulomb_forces(
        nlist=nlist, resolution=(64, 64, 64), order=6, r_cut=3.0, alpha=0)

    sim = simulation_factory(two_charged_particle_snapshot_factory())
    integrator = hoomd.md.Integrator(dt=0.005)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator.methods.append(nve)
    integrator.forces.extend([ewald, coulomb])
    sim.operations.integrator = integrator

    sim.run(0)

    autotuned_kernel_parameter_check(instance=coulomb,
                                     activate=lambda: sim.run(1))


def test_pickling(simulation_factory, two_charged_particle_snapshot_factory):
    """Test that md.long_range.pppm.Coulomb can be pickled and unpickled."""
    # detached
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    ewald, coulomb = hoomd.md.long_range.pppm.make_pppm_coulomb_forces(
        nlist=nlist, resolution=(64, 64, 64), order=6, r_cut=3.0, alpha=0)
    pickling_check(coulomb)

    # attached
    sim = simulation_factory(two_charged_particle_snapshot_factory())
    integrator = hoomd.md.Integrator(dt=0.005)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator.methods.append(nve)
    integrator.forces.extend([ewald, coulomb])
    sim.operations.integrator = integrator

    sim.run(0)

    assert ewald._attached
    assert coulomb._attached

    pickling_check(coulomb)


def test_pppm_energy(simulation_factory, two_charged_particle_snapshot_factory):
    """Test that md.long_range.pppm.Coulomb computes the correct energy."""
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    ewald, coulomb = hoomd.md.long_range.pppm.make_pppm_coulomb_forces(
        nlist=nlist, resolution=(64, 64, 64), order=6, r_cut=3.0, alpha=0)

    sim = simulation_factory(two_charged_particle_snapshot_factory())
    integrator = hoomd.md.Integrator(dt=0.005)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator.methods.append(nve)
    integrator.forces.extend([ewald, coulomb])
    sim.operations.integrator = integrator

    sim.run(0)

    energy = ewald.energy + coulomb.energy

    # The reference energy is from a LAMMPS simulation. The tolerance is large
    # as the PPPM parameters do not directly map between the two codes
    numpy.testing.assert_allclose(energy, -1.0021254, rtol=1e-2)
