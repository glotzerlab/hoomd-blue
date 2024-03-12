# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.pair.Union."""

import copy
import pytest
import rowan
import numpy as np
import numpy.testing as npt

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
    """Test hpmc union only works with hpmc pair potentials."""
    # this should work
    hpmc.pair.Union(pair_potential)

    # this should not
    with pytest.raises(TypeError):
        hpmc.pair.Union(LJ())


@pytest.fixture(scope='function')
def union_potential(pair_potential):
    return hpmc.pair.Union(pair_potential)


def _valid_body_dicts():
    valid_dicts = [
        # numpy arrays
        dict(types=["A", "A"],
             positions=np.array([[0, 0, 1.0], [0, 0, -1.0]]),
             orientations=rowan.random.rand(2),
             charges=[-0.5, 0.5]),
        # tuples
        dict(types=["A", "A"],
             positions=[(0, 0, 1.0), (0, 0, -1.0)],
             orientations=[(0.5, 0.5, -0.5, -0.5), (-0.5, 0.5, -0.5, 0.5)],
             charges=[-0.5, 0.5]),
        # orientations and charges should have defaults
        dict(types=["A", "A"], positions=[(0, 0, 1.0), (0, 0, -1.0)]),
        # No constituents
        None,
    ]
    return valid_dicts


@pytest.fixture(scope='module', params=_valid_body_dicts())
def valid_body_dict(request):
    return copy.deepcopy(request.param)


@pytest.fixture(scope='module')
def pair_union_simulation_factory(simulation_factory,
                                  two_particle_snapshot_factory):
    """Make two particle sphere simulations with a union potential."""

    def make_union_sim(union_potential, particle_types=['A'], d=1, L=20):
        sim = simulation_factory(
            two_particle_snapshot_factory(particle_types, d=d, L=L))
        sphere = hpmc.integrate.Sphere()
        sphere.shape["A"] = dict(diameter=2.0)
        sphere.pair_potentials = [union_potential]
        sim.operations.integrator = sphere
        return sim

    return make_union_sim


@pytest.mark.cpu
def test_valid_body_params(pair_union_simulation_factory, union_potential,
                           valid_body_dict):
    """Test we can set and attach with valid body params."""
    union_potential.body["A"] = valid_body_dict
    sim = pair_union_simulation_factory(union_potential)
    sim.run(0)


def _invalid_body_dicts():
    invalid_dicts = [
        # missing types
        dict(positions=[(0, 0, 1.0), (0, 0, -1.0)],
             orientations=[(0.5, 0.5, -0.5, -0.5), (-0.5, 0.5, -0.5, 0.5)],
             charges=[-0.5, 0.5]),
        # missing positions
        dict(types=["A", "A"],
             orientations=[(0.5, 0.5, -0.5, -0.5), (-0.5, 0.5, -0.5, 0.5)],
             charges=[-0.5, 0.5]),
        # positions list too short
        dict(types=["A", "A"],
             positions=[(0, 0, 1.0)],
             orientations=[(0.5, 0.5, -0.5, -0.5), (-0.5, 0.5, -0.5, 0.5)],
             charges=[-0.5, 0.5]),
        # one of the orientations tuples is 3 elements
        dict(types=["A", "A"],
             positions=[(0, 0, 1.0), (0, 0, -1.0)],
             orientations=[(0.5, 0.5, -0.5, -0.5), (-0.5, 0.5, -0.5)],
             charges=[-0.5, 0.5]),
        # one of the positions tuples is 2 elements
        dict(types=["A", "A"],
             positions=[(0, 0, 1.0), (0, -1.0)],
             orientations=[(0.5, 0.5, -0.5, -0.5), (-0.5, 0.5, -0.5, 0.5)],
             charges=[-0.5, 0.5]),
        # set one of the values set to the wrong type
        dict(types=["A", "A"],
             positions=[(0, 0, 1.0), (0, 0, -1.0)],
             orientations=[(0.5, 0.5, -0.5, -0.5), (-0.5, 0.5, -0.5, 0.5)],
             charges='invalid'),
    ]
    return invalid_dicts


@pytest.fixture(scope='module', params=_invalid_body_dicts())
def invalid_body_dict(request):
    return copy.deepcopy(request.param)


@pytest.mark.cpu
def test_invalid_body_params(pair_union_simulation_factory, union_potential,
                             invalid_body_dict):
    """Test that invalid parameter combinations result in errors."""
    with pytest.raises((IncompleteSpecificationError, TypeConversionError,
                        KeyError, RuntimeError)):
        union_potential.body["A"] = invalid_body_dict
        sim = pair_union_simulation_factory(union_potential)
        sim.run(0)


@pytest.mark.cpu
def test_default_body_params(pair_union_simulation_factory, union_potential):
    """Test default values for charges and orientations."""
    union_potential.body["A"] = dict(types=["A", "A"],
                                     positions=[(0, 0, 1.0), (0, 0, -1.0)])
    sim = pair_union_simulation_factory(union_potential)
    sim.run(0)

    body_dict = union_potential.body["A"]
    npt.assert_allclose(body_dict["charges"], 0.0)
    npt.assert_allclose(body_dict["orientations"], [[1.0, 0.0, 0.0, 0.0]] * 2)


@pytest.mark.cpu
def test_get_set_body_params(pair_union_simulation_factory, union_potential):
    """Testing getting/setting in multiple ways, before and after attaching."""
    # before attaching, setting as dict
    body_dict = dict(types=['A'], positions=[(0, 0, 1)])
    union_potential.body["A"] = body_dict
    assert union_potential.body["A"]["positions"] == body_dict["positions"]
    assert union_potential.body["A"]["types"] == body_dict["types"]
    assert 'orientations' not in union_potential.body["A"]
    assert 'charges' not in union_potential.body["A"]

    # after attaching, setting as dict
    sim = pair_union_simulation_factory(union_potential)
    sim.run(0)
    new_body_dict = dict(types=['A'], positions=[(0, 1, 0)])
    union_potential.body["A"] = new_body_dict
    assert union_potential.body["A"]["positions"] == new_body_dict["positions"]
    assert union_potential.body["A"]["types"] == new_body_dict["types"]
    assert union_potential.body["A"]["orientations"] == [(1.0, 0.0, 0.0, 0.0)]
    assert union_potential.body["A"]["charges"] == [0]

    # after attaching, setting orientations only
    union_potential.body["A"]["orientations"] = [(0.5, 0.5, 0.5, 0.5)]
    assert union_potential.body["A"]["positions"] == new_body_dict["positions"]
    assert union_potential.body["A"]["types"] == new_body_dict["types"]
    assert union_potential.body["A"]["orientations"] == [(0.5, 0.5, 0.5, 0.5)]
    assert union_potential.body["A"]["charges"] == [0]

    # after attaching, setting positions only
    union_potential.body["A"]["positions"] = [(1.0, 0.0, 0.0)]
    assert union_potential.body["A"]["positions"] == [(1.0, 0.0, 0.0)]
    assert union_potential.body["A"]["types"] == new_body_dict["types"]
    assert union_potential.body["A"]["orientations"] == [(0.5, 0.5, 0.5, 0.5)]
    assert union_potential.body["A"]["charges"] == [0]


@pytest.mark.cpu
def test_get_set_properties(pair_union_simulation_factory, union_potential):
    """Test getting/setting leaf capacity and constituent potential."""
    # assert values are right on construction
    assert union_potential.leaf_capacity == 0
    lj = union_potential.constituent_potential
    assert lj.params[('A', 'A')] == dict(epsilon=1.0,
                                         sigma=1.0,
                                         r_cut=2.0,
                                         r_on=0.0)

    # try to set params
    lj2 = hpmc.pair.LennardJones()
    lj2.params[('A', 'A')] = dict(epsilon=0.5, sigma=2.0, r_cut=3.0)
    with pytest.raises(AttributeError):
        union_potential.constituent_potential = lj2
    union_potential.leaf_capacity = 3
    assert union_potential.leaf_capacity == 3

    # attach
    union_potential.body["A"] = dict(types=["A"], positions=[(0, 0, 1)])
    sim = pair_union_simulation_factory(union_potential)
    sim.run(0)

    # set after attaching
    union_potential.leaf_capacity = 5
    assert union_potential.leaf_capacity == 5


@pytest.mark.cpu
def test_detach(pair_union_simulation_factory, union_potential):
    body_dict = dict(types=['A'], positions=[(0, 0, 1)])
    union_potential.body["A"] = body_dict
    sim = pair_union_simulation_factory(union_potential)
    sim.run(0)

    # detach from simulation
    sim.operations.integrator.pair_potentials.remove(union_potential)

    assert not union_potential._attached
    assert not union_potential.constituent_potential._attached


@pytest.mark.cpu
@pytest.mark.parametrize("leaf_capacity", (0, 4))
def test_energy(pair_union_simulation_factory, union_potential, leaf_capacity):
    """Test the energy is being calculated right in a small system."""
    # set up system
    union_potential.leaf_capacity = leaf_capacity

    lj = union_potential.constituent_potential
    lj.params[("B", "B")] = dict(epsilon=3.0, sigma=1.0, r_cut=4.0)
    lj.params[("A", "B")] = dict(epsilon=2.0, sigma=1.0, r_cut=4.0)
    lj.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0, r_cut=4.0)
    union_potential.body["A"] = dict(types=['A', 'B'],
                                     positions=[(-1, 0, 0), (1, 0, 0)])
    union_potential.body["B"] = dict(types=['A', 'B'],
                                     positions=[(-1, 0, 0), (1, 0, 0)])
    sim = pair_union_simulation_factory(union_potential,
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
    npt.assert_allclose(system_energy, union_potential.energy)
