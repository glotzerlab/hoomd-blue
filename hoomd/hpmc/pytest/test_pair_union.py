"""Test hoomd.hpmc.pair.Union."""

import copy
import hoomd
import pytest
import rowan
import numpy as np
import numpy.testing as npt

from hoomd import hpmc
from hoomd.error import TypeConversionError


@pytest.fixture(scope="function")
def pair_potential():
    lj = hpmc.pair.LennardJones()
    lj.params[("A", "A")] = dict(epsilon=1.0, sigma=1.0, r_cut=2.0)
    return lj


def test_contruction(pair_potential):
    """Test hpmc union only works with hpmc pair potentials."""
    # this should work
    patch = hpmc.pair.Union(pair_potential)

    # this should not
    with pytest.raises(TypeConversionError):
        patch = hpmc.pair.Union(hoomd.md.pair.LJ(hoomd.md.nlist.Cell))


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
        dict(types=["A", "A"],
             positions=[(0, 0, 1.0), (0, 0, -1.0)]),
    ]
    return valid_dicts


@pytest.fixture(scope='module', params=_valid_body_dicts())
def valid_body_dict(request):
    return copy.deepcopy(request.param)


@pytest.fixture(scope='module')
def pair_union_simulation_factory(simulation_factory, two_particle_snapshot_factory):
    """Make two particle sphere simulations with a union potential."""
    def make_union_sim(union_potential):
        sim = simulation_factory(two_particle_snapshot_factory())
        sphere = hpmc.integrate.Sphere()
        sphere.shape["A"] = dict(diameter=2.0)
        sphere.pair_potentials = [union_potential]
        sim.operations.integrator = sphere
        return sim
    return make_union_sim


@pytest.mark.cpu
def test_valid_body_params(pair_union_simulation_factory, union_potential, valid_body_dict):
    """Test we can set, attach, and run with valid body params."""
    union_potential.body["A"] = valid_body_dict
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


def test_multiple_body_types():
    pass


def test_invalid_body_params():
    pass


def test_get_set_body_params():
    pass


def test_detach():
    pass


def test_energy():
    pass
