import hoomd
import pytest
import numpy as np

_k = [30.0, 25.0, 20.0]
_r0 = [1.6, 1.7, 1.8]
_epsilon = [0.9, 1.0, 1.1]
_sigma = [1.1, 1.0, 0.9]


def get_bond_params():
    return zip(_k, _r0, _epsilon, _sigma)


def get_bond_params_and_forces_and_energies():
    forces = [282.296, 146.288, 88.8238]
    energies = [70.5638, 49.2476, 35.3135]
    return zip(_k, _r0, _epsilon, _sigma, forces, energies)


@pytest.mark.parametrize("bond_params_tuple", get_bond_params())
def test_before_attaching(bond_params_tuple):
    k, r0, epsilon, sigma = bond_params_tuple
    bond_params = dict(k=k, r0=r0, epsilon=epsilon, sigma=sigma)
    bond_potential = hoomd.md.bond.FENE()
    bond_potential.params['bond'] = bond_params

    for key in bond_params.keys():
        np.testing.assert_allclose(bond_potential.params['bond'][key],
                                   bond_params[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("bond_params_tuple", get_bond_params())
def test_after_attaching(two_particle_snapshot_factory, simulation_factory,
                         bond_params_tuple):
    snap = two_particle_snapshot_factory(d=0.969, L=5)
    if snap.communicator.rank == 0:
        snap.bonds.N = 1
        snap.bonds.types = ['bond']
        snap.bonds.typeid[0] = 0
        snap.bonds.group[0] = (0, 1)
    sim = simulation_factory(snap)

    k, r0, epsilon, sigma = bond_params_tuple
    bond_params = dict(k=k, r0=r0, epsilon=epsilon, sigma=sigma)
    bond_potential = hoomd.md.bond.FENE()
    bond_potential.params['bond'] = bond_params

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(bond_potential)

    nvt = hoomd.md.methods.Langevin(kT=1, filter=hoomd.filter.All(), alpha=0.1)
    integrator.methods.append(nvt)
    sim.operations.integrator = integrator

    sim.run(0)
    for key in bond_params.keys():
        np.testing.assert_allclose(bond_potential.params['bond'][key],
                                   bond_params[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("bond_params_and_force_and_energy",
                         get_bond_params_and_forces_and_energies())
def test_forces_and_energies(two_particle_snapshot_factory, simulation_factory,
                             bond_params_and_force_and_energy):
    snap = two_particle_snapshot_factory(d=0.969, L=5)
    if snap.communicator.rank == 0:
        snap.bonds.N = 1
        snap.bonds.types = ['bond']
        snap.bonds.typeid[0] = 0
        snap.bonds.group[0] = (0, 1)
        snap.particles.diameter[0] = 0.5
        snap.particles.diameter[1] = 0.5
    sim = simulation_factory(snap)

    k, r0, epsilon, sigma, force, energy = bond_params_and_force_and_energy
    bond_params = dict(k=k, r0=r0, epsilon=epsilon, sigma=sigma)
    bond_potential = hoomd.md.bond.FENE()
    bond_potential.params['bond'] = bond_params

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(bond_potential)

    nvt = hoomd.md.methods.Langevin(kT=1, filter=hoomd.filter.All(), alpha=0.1)
    integrator.methods.append(nvt)
    sim.operations.integrator = integrator

    sim.run(0)

    sim_energies = sim.operations.integrator.forces[0].energies
    sim_forces = sim.operations.integrator.forces[0].forces
    if sim.device.communicator.rank == 0:
        np.testing.assert_allclose(sum(sim_energies),
                                   energy,
                                   rtol=1e-2,
                                   atol=1e-5)
        np.testing.assert_allclose(sim_forces[0], [force, 0.0, 0.0],
                                   rtol=1e-2,
                                   atol=1e-5)
        np.testing.assert_allclose(sim_forces[1], [-1 * force, 0.0, 0.0],
                                   rtol=1e-2,
                                   atol=1e-5)
