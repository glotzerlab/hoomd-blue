# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.pair.AngularStep."""

import copy
import pytest
import numpy as np

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
        dict(directors=np.array([[1.0, 0, 0], [0, 1.0, 0]]),
             deltas=[0.1, 0.2]),
        # tuples
        dict(directors=[(1.0, 0, 0), (0, 1.0, 0)],
             deltas=[0.1, 0.2]),    
    ]
    return valid_dicts

@pytest.fixture(scope='module', params=_valid_particle_dicts())
def valid_particle_dict(request):
    return copy.deepcopy(request.param)

@pytest.fixture(scope='module')
def pair_angular_step_simulation_factory(simulation_factory,
                                  two_particle_snapshot_factory):
    """Make two particle sphere simulations with an angular step potential."""

    def make_angular_step_sim(angular_step_potential, 
                              particle_types=['A'], d=1, L=20):
        sim = simulation_factory(
            two_particle_snapshot_factory(particle_types, d=d, L=L))
        sphere = hpmc.integrate.Sphere()
        sphere.shape["A"] = dict(diameter=2.0)
        sphere.pair_potentials = [angular_step_potential]
        sim.operations.integrator = sphere
        return sim

    return make_angular_step_sim

@pytest.mark.cpu
def test_valid_particle_params(pair_angular_step_simulation_factory, 
                               angular_step_potential,
                               valid_particle_dict):
    """Test we can set and attach with valid particle params."""
    angular_step_potential.patch["A"] = valid_particle_dict
    sim = pair_angular_step_simulation_factory(angular_step_potential)
    sim.run(0)

def _invalid_particle_dicts():
    invalid_dicts = [
        # missing directors
        dict(deltas=[0.1, 0.2]),
        # missing deltas
        dict(directors=[(1.0, 0, 0), (0, 1.0, 0)]),
        # directors list too short
        dict(directors=[(1.0, 0, 0)],
             deltas=[0.1, 0.2]),
        # deltas list too short
        dict(directors=[(1.0, 0, 0), (0, 1.0, 0)],
             deltas=[0.1]),
        # one of the directors tuples is 2 elements
        dict(directors=[(1.0, 0, 0), (0, 1.0)],
             deltas=[0.1, 0.2]),
        # set one of the values set to the wrong type
        dict(directors=[(1.0, 0, 0), (0, 1.0, 0)],
             deltas='invalid'),
    ]
    return invalid_dicts

@pytest.fixture(scope='module', params=_invalid_particle_dicts())
def invalid_particle_dict(request):
    return copy.deepcopy(request.param)


@pytest.mark.cpu
def test_invalid_particle_params(pair_angular_step_simulation_factory, 
                                 angular_step_potential,
                                 invalid_particle_dict):
    """Test that invalid parameter combinations result in errors."""
    with pytest.raises((IncompleteSpecificationError, TypeConversionError,
                        KeyError, RuntimeError)):
        angular_step_potential.patch["A"] = invalid_particle_dict
        sim = pair_angular_step_simulation_factory(angular_step_potential)
        sim.run(0)

@pytest.mark.cpu
def test_default_particle_params(pair_angular_step_simulation_factory, 
                                 angular_step_potential):
    """Test default values for directors and deltas."""
    angular_step_potential.patch["A"] = dict()
    sim = pair_angular_step_simulation_factory(angular_step_potential)
    sim.run(0)

    particle_dict = angular_step_potential.patch["A"]
    assert (particle_dict["directors"], None)
    assert (particle_dict["deltas"], None)

@pytest.mark.cpu
def test_get_set_patch_params(pair_angular_step_simulation_factory, 
                              angular_step_potential):
    """Testing getting/setting in multiple ways, before and after attaching."""
    # before attaching, setting as dict
    particle_dict = dict(directors=[(0, 0, 1)], deltas=[0.1])
    angular_step_potential.patch["A"] = particle_dict
    assert angular_step_potential.patch["A"]["directors"] 
           == particle_dict["directors"]
    assert angular_step_potential.patch["A"]["deltas"] 
           == particle_dict["deltas"]

    # after attaching, setting as dict
    sim = pair_angular_step_simulation_factory(angular_step_potential)
    sim.run(0)
    new_particle_dict = dict(directors=[(0, 1, 0)], deltas=[0.2])
    angular_step_potential.patch["A"] = new_particle_dict
    assert angular_step_potential.patch["A"]["directors"] 
           == new_particle_dict["directors"]
    assert angular_step_potential.patch["A"]["deltas"] 
           == new_particle_dict["deltas"]

    # after attaching, setting directors only
    angular_step_potential.patch["A"]["directors"] = [(0, 1, 0)]
    assert angular_step_potential.patch["A"]["directors"] == [(0, 1, 0)]
    assert angular_step_potential.patch["A"]["deltas"] == None

    # after attaching, setting deltas only
    angular_step_potential.patch["A"]["deltas"] = [0.1, 0.2]
    assert angular_step_potential.patch["A"]["directors"] == None
    assert angular_step_potential.patch["A"]["deltas"] == [0.1, 0.2]

@pytest.mark.cpu
def test_detach(pair_angular_step_simulation_factory, angular_step_potential):
    particle_dict = dict(directors=[(1.0, 0, 0), (0, 1.0, 0)],
             deltas=[0.1, 0.2])
    angular_step_potential.patch["A"] = particle_dict
    sim = pair_angular_step_simulation_factory(angular_step_potential)
    sim.run(0)

    # detach from simulation
    sim.operations.integrator.pair_potentials.remove(angular_step_potential)

    assert not angular_step_potential._attached
    assert not angular_step_potential.isotropic_potential._attached

@pytest.mark.cpu
def test_energy(pair_angular_step_simulation_factory, 
                angular_step_potential):
    """Test the energy is being calculated right in a small system."""
    # set up system

    lj = angular_step_potential.isotropic_potential
    lj.params[("B", "B")] = dict(epsilon=3.0, sigma=1.0, r_cut=4.0)
    lj.params[("A", "B")] = dict(epsilon=2.0, sigma=1.0, r_cut=4.0)
    lj.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0, r_cut=4.0)
    angular_step_potential.patch["A"] = dict(
                                directors=[(1.0, 0, 0), (0, 1.0, 0)], 
                                deltas=[0.1, 0.2])
    angular_step_potential.patch["B"] = dict(directors=[(1.0, 0, 0), 
                                                  (0, 1.0, 0), (0, 0, 1.0)], 
                                     deltas=[0.1, 0.2, 0.3])
    sim = pair_angular_step_simulation_factory(angular_step_potential,
                                        particle_types=['A', 'B'],
                                        d=3,
                                        L=30)
    sim.operations.integrator.shape["B"] = dict(diameter=2)
    sim.run(0)

    def lj_energy(epsilon, sigma, distance):
        sdivd = sigma / distance
        return 4 * epsilon * (sdivd**12 - sdivd**6)

    system_energy = lj_energy(1.0, 1.0, 3.0) + lj_energy(
        3.0, 1.0, 3.0) + lj_energy(2.0, 1.0, 1.0)
    npt.assert_allclose(system_energy, angular_step_potential.energy)



