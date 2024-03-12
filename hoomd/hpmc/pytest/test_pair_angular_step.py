# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.pair.AngularStep."""

import copy
import pytest
import numpy as np
import rowan

from hoomd import hpmc
from hoomd.error import TypeConversionError, IncompleteSpecificationError


class LJ:
    pass


@pytest.fixture(scope="function")
def pair_potential():
    lj = hpmc.pair.LennardJones()
    lj.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0, r_cut=2.0)
    return lj


def test_contruction(pair_potential):
    """Test hpmc angular step only works with hpmc pair potentials."""
    # this should work
    hpmc.pair.AngularStep(pair_potential)

    # this should not
    with pytest.raises(TypeError):
        hpmc.pair.AngularStep(LJ())


@pytest.fixture(scope='function')
def angular_step_potential(pair_potential):
    return hpmc.pair.AngularStep(pair_potential)


def _valid_particle_dicts():
    valid_dicts = [
        # numpy arrays
        dict(directors=[np.array([1.0, 0, 0]),
                        np.array([0, 1.0, 0])],
             deltas=[0.1, 0.2]),
        # lists
        dict(directors=[[1.0, 0, 0], [0, 1.0, 0]], deltas=[0.1, 0.2]),
        # tuples
        dict(directors=[(1.0, 0, 0), (0, 1.0, 0)], deltas=[0.1, 0.2]),
    ]
    return valid_dicts


@pytest.fixture(scope='module', params=_valid_particle_dicts())
def valid_particle_dict(request):
    return copy.deepcopy(request.param)


@pytest.fixture(scope='module')
def pair_angular_step_simulation_factory(simulation_factory,
                                         two_particle_snapshot_factory):
    """Make two particle sphere simulations with an angular step potential."""

    def make_angular_step_sim(d=1, theta_0=0, theta_1=0):
        snapshot = two_particle_snapshot_factory(d=d)
        if snapshot.communicator.rank == 0:
            snapshot.particles.orientation[0] = rowan.from_axis_angle((0, 0, 1),
                                                                      theta_0)
            snapshot.particles.orientation[1] = rowan.from_axis_angle((0, 0, 1),
                                                                      theta_1)
        sim = simulation_factory(snapshot)

        sphere = hpmc.integrate.Sphere()
        sphere.shape["A"] = dict(diameter=0.0)
        sim.operations.integrator = sphere
        return sim

    return make_angular_step_sim


@pytest.mark.cpu
def test_valid_particle_params(pair_angular_step_simulation_factory,
                               angular_step_potential, valid_particle_dict):
    """Test we can set and attach with valid particle params."""
    angular_step_potential.mask["A"] = valid_particle_dict
    sim = pair_angular_step_simulation_factory()
    sim.operations.integrator.pair_potentials = [angular_step_potential]
    sim.run(0)


def _invalid_particle_dicts():
    invalid_dicts = [
        # missing directors
        dict(deltas=[0.1, 0.2]),
        # missing deltas
        dict(directors=[(1.0, 0, 0), (0, 1.0, 0)]),
        # directors list too short
        dict(directors=[(1.0, 0, 0)], deltas=[0.1, 0.2]),
        # deltas list too short
        dict(directors=[(1.0, 0, 0), (0, 1.0, 0)], deltas=[0.1]),
        # one of the directors tuples is 2 elements
        dict(directors=[(1.0, 0, 0), (0, 1.0)], deltas=[0.1, 0.2]),
        # set one of the values set to the wrong type
        dict(directors=[(1.0, 0, 0), (0, 1.0, 0)], deltas='invalid'),
        # include an unexpected key
        dict(directors=[(1.0, 0, 0), (0, 1.0, 0)],
             deltas=[0.1, 0.2],
             key='invalid'),
    ]
    return invalid_dicts


@pytest.fixture(scope='module', params=_invalid_particle_dicts())
def invalid_particle_dict(request):
    return copy.deepcopy(request.param)


@pytest.mark.cpu
def test_invalid_particle_params(pair_angular_step_simulation_factory,
                                 angular_step_potential, invalid_particle_dict):
    """Test that invalid parameter combinations result in errors."""
    with pytest.raises((IncompleteSpecificationError, TypeConversionError,
                        KeyError, RuntimeError)):
        angular_step_potential.mask["A"] = invalid_particle_dict
        sim = pair_angular_step_simulation_factory()
        sim.operations.integrator.pair_potentials = [angular_step_potential]
        sim.run(0)


@pytest.mark.cpu
def test_get_set_patch_params(pair_angular_step_simulation_factory,
                              angular_step_potential):
    """Testing getting/setting in multiple ways, before and after attaching."""
    # before attaching, setting as dict
    particle_dict = dict(directors=[(1.0, 0, 0)], deltas=[0.1])
    angular_step_potential.mask["A"] = particle_dict
    assert angular_step_potential.mask["A"]["directors"] == particle_dict[
        "directors"]
    assert angular_step_potential.mask["A"]["deltas"] == particle_dict["deltas"]

    # after attaching, setting as dict
    sim = pair_angular_step_simulation_factory()
    sim.operations.integrator.pair_potentials = [angular_step_potential]
    sim.run(0)
    new_particle_dict = dict(directors=[(0, 1, 0)], deltas=[0.2])
    angular_step_potential.mask["A"] = new_particle_dict
    assert angular_step_potential.mask["A"]["directors"] == new_particle_dict[
        "directors"]
    assert angular_step_potential.mask["A"]["deltas"] == new_particle_dict[
        "deltas"]

    # after attaching, change the director value
    angular_step_potential.mask["A"]["directors"] = [(0, 0, 1.0)]
    assert angular_step_potential.mask["A"]["directors"] == [(0, 0, 1.0)]
    assert angular_step_potential.mask["A"]["deltas"] == [0.2]

    # after attaching, change the delta value
    angular_step_potential.mask["A"]["deltas"] = [0.3]
    assert angular_step_potential.mask["A"]["directors"] == [(0, 0, 1.0)]
    assert angular_step_potential.mask["A"]["deltas"] == pytest.approx([0.3])


@pytest.mark.cpu
def test_detach(pair_angular_step_simulation_factory, angular_step_potential):
    particle_dict = dict(directors=[(1.0, 0, 0), (0, 1.0, 0)],
                         deltas=[0.1, 0.2])
    angular_step_potential.mask["A"] = particle_dict
    sim = pair_angular_step_simulation_factory()
    sim.operations.integrator.pair_potentials = [angular_step_potential]
    sim.run(0)

    # detach from simulation
    sim.operations.integrator.pair_potentials.remove(angular_step_potential)

    assert not angular_step_potential._attached
    assert not angular_step_potential.isotropic_potential._attached


def lj(r, r_cut, epsilon, sigma):
    """Compute the lj energy."""
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)


# Test 1 particle type

#  params
#  theta_0, # rotation of the first particle
#  theta_1,
#  distance between particles,
#  expected energy
angular_step_test_parameters_one_type = [
    (  # rotate pi (overlap), > rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        0,
        np.pi,
        5.0,
        0.0,
    ),
    (  # no rotation (no overlap), > rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        0,
        0,
        5.0,
        0.0,
    ),
    (  # rotate pi (overlap), < rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        0,
        np.pi,
        2.0,
        lj(2.0, 4.0, 1, 1),
    ),
    (  # no rotation (no overlap), < rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        0,
        0,
        2.0,
        0.0,
    ),
    (  # rotate pi-0.5 (no overlap), > rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        0,
        np.pi - 0.5,
        5.0,
        0.0,
    ),
    (  # rotate pi-0.5 (no overlap), < rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        0,
        np.pi - 0.5,
        2.0,  # < rcut
        0.0,
    ),
    (  # rotate pi-0.099 (just overlapped), > rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        0,
        np.pi - 0.099,
        5.0,
        0.0,
    ),
    (  # rotate pi-0.099 (just overlapped), < rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        0,
        np.pi - 0.099,
        2.0,
        lj(2.0, 4.0, 1, 1),
    ),
    (  # rotate pi-0.05 (overlap), > rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        0,
        np.pi - 0.05,
        5.0,
        0.0,
    ),
    (  # rotate pi-0.05 (overlap), < rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        0,
        np.pi - 0.05,
        2.0,
        lj(2.0, 4.0, 1, 1),
    ),
]


@pytest.mark.parametrize('params, theta_0, theta_1, d, expected_energy',
                         angular_step_test_parameters_one_type)
@pytest.mark.cpu
def test_energy(pair_angular_step_simulation_factory, params, theta_0, theta_1,
                d, expected_energy):
    """Test that LennardJones computes the correct energies for 1 pair."""
    lennard_jones = hpmc.pair.LennardJones(mode='none')
    lennard_jones.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0, r_cut=4.0)
    angular_step = hpmc.pair.AngularStep(isotropic_potential=lennard_jones)
    angular_step.mask['A'] = params

    simulation = pair_angular_step_simulation_factory(d=d,
                                                      theta_0=theta_0,
                                                      theta_1=theta_1)
    simulation.operations.integrator.pair_potentials = [angular_step]
    simulation.run(0)

    assert angular_step.energy == pytest.approx(expected=expected_energy,
                                                rel=1e-5)


# Test 2 particle types

#  params_0
#  params_1
#  theta_0, # rotation of the first particle
#  theta_1,
#  distance between particles,
#  expected energy
angular_step_test_parameters_two_types = [
    (  # 2 types, rotate pi (overlap), > rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        dict(directors=[(1.0, 0, 0), (0.0, 1.0, 0.0)], deltas=[0.1, 0.2]),
        0,
        np.pi,
        5.0,
        0.0,
    ),
    (  # 2 types, no rotation (no overlap), > rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        dict(directors=[(1.0, 0, 0), (0.0, 1.0, 0.0)], deltas=[0.1, 0.2]),
        0,
        0,
        5.0,
        0.0,
    ),
    (  # 2 types, rotate pi (overlap), < rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        dict(directors=[(1.0, 0, 0), (0.0, 1.0, 0.0)], deltas=[0.1, 0.2]),
        0,
        np.pi,
        2.0,
        lj(2.0, 4.0, 1, 1),
    ),
    (  # 2 types, no rotation (no overlap), < rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        dict(directors=[(1.0, 0, 0), (0.0, 1.0, 0.0)], deltas=[0.1, 0.2]),
        0,
        0,
        2.0,
        0.0,
    ),
    (  # 2 types, rotate pi-0.5 (no overlap), > rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        dict(directors=[(1.0, 0, 0), (0.0, 1.0, 0.0)], deltas=[0.1, 0.2]),
        0,
        np.pi - 0.5,
        5.0,
        0.0,
    ),
    (  # 2 types, rotate pi-0.5 (no overlap), < rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        dict(directors=[(1.0, 0, 0), (0.0, 1.0, 0.0)], deltas=[0.1, 0.2]),
        0,
        np.pi - 0.5,
        2.0,  # < rcut
        0.0,
    ),
    (  # 2 types, rotate pi-0.099 (just overlapped), > rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        dict(directors=[(1.0, 0, 0), (0.0, 1.0, 0.0)], deltas=[0.1, 0.2]),
        0,
        np.pi - 0.099,
        5.0,
        0.0,
    ),
    (  # 2 types, rotate pi-0.099 (just overlapped), < rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        dict(directors=[(1.0, 0, 0), (0.0, 1.0, 0.0)], deltas=[0.1, 0.2]),
        0,
        np.pi - 0.099,
        2.0,
        lj(2.0, 4.0, 1, 1),
    ),
    (  # 2 types, rotate pi-0.05 (overlap), > rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        dict(directors=[(1.0, 0, 0), (0.0, 1.0, 0.0)], deltas=[0.1, 0.2]),
        0,
        np.pi - 0.05,
        5.0,
        0.0,
    ),
    (  # 2 types, rotate pi-0.05 (overlap), < rcut
        dict(directors=[(1.0, 0, 0)], deltas=[0.1]),
        dict(directors=[(1.0, 0, 0), (0.0, 1.0, 0.0)], deltas=[0.1, 0.2]),
        0,
        np.pi - 0.05,
        2.0,
        lj(2.0, 4.0, 1, 1),
    ),
]


@pytest.mark.parametrize(
    'params_0, params_1, theta_0, theta_1, d,'
    'expected_energy', angular_step_test_parameters_two_types)
@pytest.mark.cpu
def test_energy_two_types(pair_angular_step_simulation_factory, params_0,
                          params_1, theta_0, theta_1, d, expected_energy):
    """Test that LennardJones computes the correct energies for 1 pair."""
    lennard_jones = hpmc.pair.LennardJones(mode='none')
    lennard_jones.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0, r_cut=4.0)
    lennard_jones.params[('A', 'B')] = dict(epsilon=2.0, sigma=1.0, r_cut=4.0)
    lennard_jones.params[('B', 'B')] = dict(epsilon=3.0, sigma=1.0, r_cut=4.0)
    angular_step = hpmc.pair.AngularStep(isotropic_potential=lennard_jones)
    angular_step.mask['A'] = params_0
    angular_step.mask['B'] = params_1

    simulation = pair_angular_step_simulation_factory(d=d,
                                                      theta_0=theta_0,
                                                      theta_1=theta_1)
    simulation.operations.integrator.pair_potentials = [angular_step]
    simulation.run(0)

    assert angular_step.energy == pytest.approx(expected=expected_energy,
                                                rel=1e-5)
