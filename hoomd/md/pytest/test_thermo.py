# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.conftest import operation_pickling_check, logging_check
from hoomd.error import DataAccessError
from hoomd.logging import LoggerCategories
import pytest
import numpy as np
""" Each entry is a quantity and its type """
_thermo_qtys = [
    ('kinetic_temperature', float),
    ('pressure', float),
    ('pressure_tensor', list),
    ('kinetic_energy', float),
    ('translational_kinetic_energy', float),
    ('rotational_kinetic_energy', float),
    ('potential_energy', float),
    ('degrees_of_freedom', float),
    ('translational_degrees_of_freedom', float),
    ('rotational_degrees_of_freedom', float),
    ('num_particles', int),
]


def test_attach_detach(simulation_factory, two_particle_snapshot_factory):
    # test before attaching to simulation
    group = hoomd.filter.All()
    thermo = hoomd.md.compute.ThermodynamicQuantities(group)
    for qty, typ in _thermo_qtys:
        with pytest.raises(DataAccessError):
            getattr(thermo, qty)

    # make simulation and test state of operations
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.add(thermo)
    assert len(sim.operations.computes) == 1
    sim.operations._schedule()

    # make sure quantities are computable without failure
    for qty, typ in _thermo_qtys:
        calc_qty = getattr(thermo, qty)
        assert type(calc_qty) is typ

    # detach from simulation and test properties again
    sim.operations.remove(thermo)
    assert len(sim.operations.computes) == 0
    for qty, typ in _thermo_qtys:
        with pytest.raises(DataAccessError):
            getattr(thermo, qty)


def _assert_thermo_properties(thermo, npart, rdof, tdof, pe, rke, tke, ke, p,
                              pt, volume):

    assert thermo.num_particles == npart
    assert thermo.rotational_degrees_of_freedom == rdof
    assert thermo.translational_degrees_of_freedom == tdof
    assert thermo.degrees_of_freedom == (thermo.translational_degrees_of_freedom
                                         + thermo.rotational_degrees_of_freedom)

    np.testing.assert_allclose(thermo.potential_energy, pe)
    np.testing.assert_allclose(thermo.rotational_kinetic_energy, rke, rtol=1e-4)
    np.testing.assert_allclose(thermo.translational_kinetic_energy,
                               tke,
                               rtol=1e-4)
    np.testing.assert_allclose(thermo.kinetic_energy, ke, rtol=1e-4)
    np.testing.assert_allclose(thermo.kinetic_temperature,
                               2 * thermo.kinetic_energy
                               / thermo.degrees_of_freedom,
                               rtol=1e-4)
    np.testing.assert_allclose(thermo.pressure, p, rtol=1e-4)
    np.testing.assert_allclose(thermo.pressure_tensor, pt, rtol=1e-4, atol=5e-5)
    np.testing.assert_allclose(thermo.volume, volume, rtol=1e-7, atol=1e-7)


def test_basic_system_3d(simulation_factory, two_particle_snapshot_factory):
    filt = hoomd.filter.All()
    thermo = hoomd.md.compute.ThermodynamicQuantities(filt)
    snap = two_particle_snapshot_factory()
    if snap.communicator.rank == 0:
        snap.particles.velocity[:] = [[-2, 0, 0], [2, 0, 0]]
    sim = simulation_factory(snap)
    sim.always_compute_pressure = True
    sim.operations.add(thermo)

    integrator = hoomd.md.Integrator(dt=0.0001)
    thermostat = hoomd.md.methods.thermostats.MTTK(kT=1.0, tau=1.0)
    integrator.methods.append(hoomd.md.methods.ConstantVolume(filt, thermostat))
    sim.operations.integrator = integrator

    sim.run(1)

    volume = (snap.configuration.box[0] * snap.configuration.box[1]
              * snap.configuration.box[2])

    _assert_thermo_properties(thermo, 2, 0, 3, 0.0, 0.0, 4.0, 4.0,
                              2.0 / 3 * thermo.kinetic_energy / 20**3,
                              [8.0 / 20.0**3, 0., 0., 0., 0., 0.], volume)


def test_basic_system_2d(simulation_factory, lattice_snapshot_factory):
    filterA = hoomd.filter.Type(['A'])
    filterB = hoomd.filter.Type(['B'])
    thermoA = hoomd.md.compute.ThermodynamicQuantities(filterA)
    thermoB = hoomd.md.compute.ThermodynamicQuantities(filterB)
    snap = lattice_snapshot_factory(particle_types=['A', 'B'],
                                    dimensions=2,
                                    n=2)
    if snap.communicator.rank == 0:
        snap.particles.velocity[:] = [[-1, 0, 0], [2, 0, 0]] * 2
        snap.particles.typeid[:] = [0, 1, 0, 1]
    sim = simulation_factory(snap)
    sim.always_compute_pressure = True
    sim.operations.add(thermoA)
    sim.operations.add(thermoB)

    integrator = hoomd.md.Integrator(dt=0.0001)
    thermostat = hoomd.md.methods.thermostats.MTTK(kT=1.0, tau=1.0)
    integrator.methods.append(
        hoomd.md.methods.ConstantVolume(filterA, thermostat))
    integrator.methods.append(
        hoomd.md.methods.Langevin(filterB, kT=1, default_gamma=0.00001))
    sim.operations.integrator = integrator

    sim.run(1)

    volume = snap.configuration.box[0] * snap.configuration.box[1]

    # tests for group A
    _assert_thermo_properties(thermoA, 2, 0, 4, 0.0, 0.0, 1.0, 1.0,
                              thermoA.kinetic_energy / 2.0**2,
                              (2.0 / 2.0**2, 0., 0., 0., 0., 0.), volume)

    # tests for group B
    _assert_thermo_properties(thermoB, 2, 0, 4, 0.0, 0.0, 4.0, 4.0,
                              thermoB.kinetic_energy / 2.0**2,
                              (8.0 / 2.0**2, 0., 0., 0., 0., 0.), volume)


def test_system_rotational_dof(simulation_factory, device):

    snap = hoomd.Snapshot(device.communicator)
    if snap.communicator.rank == 0:
        box = [10, 10, 10, 0, 0, 0]
        snap.configuration.box = box
        snap.particles.N = 3
        snap.particles.position[:] = [[0, 1, 0], [-1, 1, 0], [1, 1, 0]]
        snap.particles.velocity[:] = [[0, 0, 0], [0, -1, 0], [0, 1, 0]]
        snap.particles.moment_inertia[:] = [[2.0, 0, 0], [1, 1, 1], [1, 1, 1]]
        snap.particles.angmom[:] = [[0, 2, 4, 6]] * 3
        snap.particles.types = ['A']

    filt = hoomd.filter.All()
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=filt)
    sim = simulation_factory(snap)
    sim.always_compute_pressure = True
    sim.operations.add(thermo)

    integrator = hoomd.md.Integrator(dt=0.0001, integrate_rotational_dof=True)
    thermostat = hoomd.md.methods.thermostats.MTTK(kT=1.0, tau=1.0)
    integrator.methods.append(hoomd.md.methods.ConstantVolume(filt, thermostat))
    sim.operations.integrator = integrator

    sim.run(0)

    _assert_thermo_properties(thermo,
                              3,
                              7,
                              6,
                              0.0,
                              57 / 4.,
                              1.0,
                              61 / 4.,
                              2. / 3 * thermo.translational_kinetic_energy
                              / 10.0**3, (0., 0., 0., 2. / 10**3, 0., 0.),
                              volume=1000)

    integrator.integrate_rotational_dof = False
    sim.run(0)
    assert thermo.rotational_degrees_of_freedom == 0


def test_pickling(simulation_factory, two_particle_snapshot_factory):
    filter_ = hoomd.filter.All()
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter_)
    sim = simulation_factory(two_particle_snapshot_factory())
    operation_pickling_check(thermo, sim)


def test_logging():
    logging_check(
        hoomd.md.compute.ThermodynamicQuantities, ('md', 'compute'), {
            'kinetic_temperature': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'pressure': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'pressure_tensor': {
                'category': LoggerCategories.sequence,
                'default': True
            },
            'kinetic_energy': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'translational_kinetic_energy': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'rotational_kinetic_energy': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'potential_energy': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'degrees_of_freedom': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'translational_degrees_of_freedom': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'rotational_degrees_of_freedom': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'num_particles': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'volume': {
                'category': LoggerCategories.scalar,
                'default': True
            }
        })
