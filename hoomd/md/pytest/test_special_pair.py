# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd import md
from hoomd.conftest import expected_loggable_params
from hoomd.conftest import (logging_check, pickling_check,
                            autotuned_kernel_parameter_check)
import pytest
import numpy

import itertools

R = 0.9
CHARGE = [-2.5, 2.5]

# test parameters include the class, special pair params, r_cut, force, and
# energy
special_pair_test_parameters = [
    (
        hoomd.md.special_pair.LJ,
        dict(epsilon=1.5, sigma=0.5),
        2.5,
        24 * 0.5**6 * 1.5 * (R**6 - 2 * 0.5**6) / R**13,
        4 * 1.5 * ((0.5 / R)**12 - (0.5 / R)**6),
    ),
    (
        hoomd.md.special_pair.Coulomb,
        dict(alpha=1.5),
        2.5,
        -1.5 * CHARGE[0] * CHARGE[1] / R**2,
        1.5 * CHARGE[0] * CHARGE[1] / R,
    ),
]


@pytest.mark.parametrize("special_pair_cls, params, r_cut, force, energy",
                         special_pair_test_parameters)
def test_before_attaching(special_pair_cls, params, r_cut, force, energy):
    potential = special_pair_cls()
    potential.params['A-A'] = params
    potential.r_cut['A-A'] = r_cut
    assert potential.r_cut['A-A'] == r_cut
    for key in params:
        assert potential.params['A-A'][key] == pytest.approx(params[key])


@pytest.fixture(scope='session')
def snapshot_factory(two_particle_snapshot_factory):

    def make_snapshot():
        snapshot = two_particle_snapshot_factory(d=R, L=R * 10)
        if snapshot.communicator.rank == 0:
            snapshot.particles.charge[:] = CHARGE
            snapshot.pairs.N = 1
            snapshot.pairs.types = ['A-A']
            snapshot.pairs.typeid[0] = 0
            snapshot.pairs.group[0] = (0, 1)

        return snapshot

    return make_snapshot


@pytest.mark.parametrize("special_pair_cls, params, r_cut, force, energy",
                         special_pair_test_parameters)
def test_after_attaching(snapshot_factory, simulation_factory, special_pair_cls,
                         params, r_cut, force, energy):
    snapshot = snapshot_factory()
    sim = simulation_factory(snapshot)

    potential = special_pair_cls()
    potential.params['A-A'] = params
    potential.r_cut['A-A'] = r_cut

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    forces=[potential])

    sim.run(0)
    potential.r_cut['A-A'] = r_cut
    for key in params:
        assert potential.params['A-A'][key] == pytest.approx(params[key])


@pytest.mark.parametrize("special_pair_cls, params, r_cut, force, energy",
                         special_pair_test_parameters)
def test_forces_and_energies(snapshot_factory, simulation_factory,
                             special_pair_cls, params, r_cut, force, energy):
    snapshot = snapshot_factory()
    sim = simulation_factory(snapshot)

    potential = special_pair_cls()
    potential.params['A-A'] = params
    potential.r_cut['A-A'] = r_cut

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    forces=[potential])

    sim.run(0)

    sim_energies = potential.energies
    sim_forces = potential.forces
    if sim.device.communicator.rank == 0:
        assert sum(sim_energies) == pytest.approx(energy)
        numpy.testing.assert_allclose(sim_forces[0], [force, 0.0, 0.0],
                                      rtol=1e-6,
                                      atol=1e-5)
        numpy.testing.assert_allclose(sim_forces[1], [-1 * force, 0.0, 0.0],
                                      rtol=1e-6,
                                      atol=1e-5)


# Test Logging
@pytest.mark.parametrize(
    'cls, expected_namespace, expected_loggables',
    zip((md.special_pair.SpecialPair, md.special_pair.LJ,
         md.special_pair.Coulomb), itertools.repeat(('md', 'special_pair')),
        itertools.repeat(expected_loggable_params)))
def test_logging(cls, expected_namespace, expected_loggables):
    logging_check(cls, expected_namespace, expected_loggables)


@pytest.mark.parametrize("special_pair_cls, params, r_cut, force, energy",
                         special_pair_test_parameters)
def test_kernel_parameters(simulation_factory, snapshot_factory,
                           special_pair_cls, params, r_cut, force, energy):
    snapshot = snapshot_factory()
    sim = simulation_factory(snapshot)

    potential = special_pair_cls()
    potential.params['A-A'] = params
    potential.r_cut['A-A'] = r_cut

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    forces=[potential])

    sim.run(0)

    autotuned_kernel_parameter_check(instance=potential,
                                     activate=lambda: sim.run(1))


# Test Pickling
@pytest.mark.parametrize("special_pair_cls, params, r_cut, force, energy",
                         special_pair_test_parameters)
def test_pickling(simulation_factory, snapshot_factory, special_pair_cls,
                  params, r_cut, force, energy):
    snapshot = snapshot_factory()
    sim = simulation_factory(snapshot)

    potential = special_pair_cls()
    potential.params['A-A'] = params
    potential.r_cut['A-A'] = r_cut

    pickling_check(potential)

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    forces=[potential])

    sim.run(0)
    pickling_check(potential)
