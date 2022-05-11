# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import copy as cp
import hoomd
from hoomd.logging import LoggerCategories
import numpy as np
import pytest
import random
from hoomd.md.nlist import Cell, Stencil, Tree
from hoomd.conftest import logging_check, pickling_check


def _nlist_params():
    """Each entry in the lsit is a tuple (class_obj, dict(required_args))."""
    nlists = []
    nlists.append((Cell, {}))
    nlists.append((Tree, {}))
    nlists.append((Stencil, dict(cell_width=0.5)))
    return nlists


@pytest.fixture(scope="function",
                params=_nlist_params(),
                ids=(lambda x: x[0].__name__))
def nlist_params(request):
    return cp.deepcopy(request.param)


def _assert_nlist_params(nlist, param_dict):
    """Assert the params of the nlist are the same as in the dictionary."""
    for param, item in param_dict.items():
        if isinstance(item, (tuple, list)):
            assert all(
                a == b
                for a, b in zip(getattr(nlist, param), param_dict[param]))
        else:
            assert getattr(nlist, param) == param_dict[param]


def test_common_params(nlist_params):
    nlist_cls, required_args = nlist_params
    nlist = nlist_cls(**required_args, buffer=0.4)
    default_params_dict = {
        "buffer": 0.4,
        "exclusions": ('bond',),
        "rebuild_check_delay": 1,
        "check_dist": True,
    }
    _assert_nlist_params(nlist, default_params_dict)
    new_params_dict = {
        "buffer":
            np.random.uniform(5.0),
        "exclusions":
            random.sample([
                'bond', '1-4', 'angle', 'dihedral', 'special_pair', 'body',
                '1-3', 'constraint', 'meshbond'
            ], np.random.randint(9)),
        "rebuild_check_delay":
            np.random.randint(8),
        "check_dist":
            False,
    }
    for param in new_params_dict.keys():
        setattr(nlist, param, new_params_dict[param])
    _assert_nlist_params(nlist, new_params_dict)


def test_cell_specific_params():
    nlist = Cell(buffer=0.4)
    _assert_nlist_params(nlist, dict(deterministic=False))
    nlist.deterministic = True
    _assert_nlist_params(nlist, dict(deterministic=True))


def test_stencil_specific_params():
    cell_width = np.random.uniform(12.1)
    nlist = Stencil(cell_width=cell_width, buffer=0.4)
    _assert_nlist_params(nlist, dict(deterministic=False,
                                     cell_width=cell_width))
    nlist.deterministic = True
    x = np.random.uniform(25.5)
    nlist.cell_width = x
    _assert_nlist_params(nlist, dict(deterministic=True, cell_width=x))


def test_simple_simulation(nlist_params, simulation_factory,
                           lattice_snapshot_factory):
    nlist_cls, required_args = nlist_params
    nlist = nlist_cls(**required_args, buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist, default_r_cut=1.1)
    lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
    lj.params[('A', 'B')] = dict(epsilon=1, sigma=1)
    lj.params[('B', 'B')] = dict(epsilon=1, sigma=1)
    integrator = hoomd.md.Integrator(0.005)
    integrator.forces.append(lj)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1))

    sim = simulation_factory(lattice_snapshot_factory(n=10))
    sim.operations.integrator = integrator
    sim.run(2)


def test_auto_detach_simulation(simulation_factory,
                                two_particle_snapshot_factory):
    nlist = Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist, default_r_cut=1.1)
    lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
    lj.params[('A', 'B')] = dict(epsilon=1, sigma=1)
    lj.params[('B', 'B')] = dict(epsilon=1, sigma=1)
    lj_2 = cp.deepcopy(lj)
    lj_2.nlist = nlist
    integrator = hoomd.md.Integrator(0.005, forces=[lj, lj_2])
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1))

    sim = simulation_factory(
        two_particle_snapshot_factory(particle_types=["A", "B"], d=2.0))
    sim.operations.integrator = integrator
    sim.run(0)
    del integrator.forces[1]
    assert nlist._attached
    assert hasattr(nlist, "_cpp_obj")
    del integrator.forces[0]
    assert not nlist._attached
    assert nlist._cpp_obj is None


def test_pickling(simulation_factory, two_particle_snapshot_factory):
    nlist = Cell(0.4)
    pickling_check(nlist)
    lj = hoomd.md.pair.LJ(nlist, default_r_cut=1.1)
    lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
    lj.params[('A', 'B')] = dict(epsilon=1, sigma=1)
    lj.params[('B', 'B')] = dict(epsilon=1, sigma=1)
    integrator = hoomd.md.Integrator(0.005, forces=[lj])
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1))

    sim = simulation_factory(
        two_particle_snapshot_factory(particle_types=["A", "B"], d=2.0))
    sim.operations.integrator = integrator
    sim.run(0)
    pickling_check(nlist)


def test_cell_properties(simulation_factory, lattice_snapshot_factory):
    nlist = hoomd.md.nlist.Cell(buffer=0)
    lj = hoomd.md.pair.LJ(nlist, default_r_cut=1.1)
    lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
    lj.params[('A', 'B')] = dict(epsilon=1, sigma=1)
    lj.params[('B', 'B')] = dict(epsilon=1, sigma=1)
    integrator = hoomd.md.Integrator(0.005)
    integrator.forces.append(lj)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1))

    sim = simulation_factory(lattice_snapshot_factory(n=10))
    sim.operations.integrator = integrator

    sim.run(10)

    assert nlist.num_builds == 10
    assert nlist.shortest_rebuild == 1
    dim = nlist.dimensions
    assert len(dim) == 3
    assert dim >= (1, 1, 1)
    assert nlist.allocated_particles_per_cell >= 1


def test_logging():
    logging_check(
        hoomd.md.nlist.NeighborList, ('md', 'nlist'), {
            'shortest_rebuild': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'num_builds': {
                'category': LoggerCategories.scalar,
                'default': False
            },
        })

    logging_check(
        hoomd.md.nlist.Cell, ('md', 'nlist'), {
            'shortest_rebuild': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'num_builds': {
                'category': LoggerCategories.scalar,
                'default': False
            },
            'dimensions': {
                'category': LoggerCategories.sequence,
                'default': False
            },
            'allocated_particles_per_cell': {
                'category': LoggerCategories.scalar,
                'default': False
            },
        })
