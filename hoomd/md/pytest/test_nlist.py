# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import copy as cp
import hoomd
from hoomd.logging import LoggerCategories
import numpy as np
import pytest
import random
from hoomd.md.nlist import Cell, Stencil, Tree
from hoomd.conftest import (logging_check, pickling_check,
                            autotuned_kernel_parameter_check)


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

    # Force nlist to update every step to ensure autotuning occurs.
    nlist.check_dist = False
    nlist.rebuild_check_delay = 1
    autotuned_kernel_parameter_check(instance=nlist,
                                     activate=lambda: sim.run(1))


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
    base_loggables = {
        'shortest_rebuild': {
            'category': LoggerCategories.scalar,
            'default': True
        },
        'num_builds': {
            'category': LoggerCategories.scalar,
            'default': False
        }
    }
    logging_check(hoomd.md.nlist.NeighborList, ('md', 'nlist'), base_loggables)

    logging_check(
        hoomd.md.nlist.Cell, ('md', 'nlist'), {
            **base_loggables,
            'dimensions': {
                'category': LoggerCategories.sequence,
                'default': False
            },
            'allocated_particles_per_cell': {
                'category': LoggerCategories.scalar,
                'default': False
            },
        })


TRUE_PAIR_LIST = [[0, 3], [0, 6], [0, 9], [0, 18], [0, 1], [0, 2],
                  [1, 2], [1, 4], [1, 7], [1, 10], [1, 19], [2, 5], [2, 8],
                  [2, 11], [2, 20], [3, 6], [3, 12], [3, 21], [3, 4], [3, 5],
                  [4, 5], [4, 7], [4, 13], [4, 22], [5, 8], [5, 14], [5, 23],
                  [6, 15], [6, 24], [6, 7], [6, 8], [7, 8], [7, 16], [7, 25],
                  [8, 17], [8, 26], [9, 18], [9, 12], [9, 15], [9, 10], [9, 11],
                  [10, 11], [10, 19], [10, 13], [10, 16], [11, 20], [11, 14],
                  [11, 17], [12, 15], [12, 21], [12, 13], [12, 14], [13, 14],
                  [13, 16], [13, 22], [14, 17], [14, 23], [15, 24], [15, 16],
                  [15, 17], [16, 17], [16, 25], [17, 26], [18, 21], [18, 24],
                  [18, 19], [18, 20], [19, 20], [19, 22], [19, 25], [20, 23],
                  [20, 26], [21, 24], [21, 22], [21, 23], [22, 23], [22, 25],
                  [23, 26], [24, 25], [24, 26], [25, 26]]

TRUE_PAIR_LIST = set([frozenset(pair) for pair in TRUE_PAIR_LIST])


def test_global_pair_list(simulation_factory, lattice_snapshot_factory):

    nlist = hoomd.md.nlist.Cell(buffer=0, default_r_cut=1.1)
    print(nlist)

    sim = simulation_factory(lattice_snapshot_factory(n=3))
    sim.operations.computes.append(nlist)

    sim.run(0)

    pair_list = nlist.pair_list
    if sim.device.communicator.rank == 0:
        pair_list = set([frozenset(pair) for pair in pair_list])
        assert pair_list == TRUE_PAIR_LIST


def test_rank_local_pair_list(simulation_factory, lattice_snapshot_factory):

    nlist = hoomd.md.nlist.Cell(buffer=0, default_r_cut=1.1)

    sim = simulation_factory(lattice_snapshot_factory(n=3))
    sim.operations.computes.append(nlist)

    sim.run(0)

    local_pair_list = nlist.local_pair_list
    pair_list = set()
    with sim.state.cpu_local_snapshot as data:
        rtags = data.particles.rtag
        for pair in local_pair_list:
            pair_list.add(frozenset([rtags[pair[0]], rtags[pair[1]]]))
        assert pair_list == TRUE_PAIR_LIST


def test_rank_local_nlist_arrays(simulation_factory, lattice_snapshot_factory):

    nlist = hoomd.md.nlist.Cell(buffer=0, default_r_cut=1.1)

    sim = simulation_factory(lattice_snapshot_factory(n=3))
    sim.operations.computes.append(nlist)

    sim.run(0)

    local_pair_list = nlist.local_pair_list

    with nlist.cpu_local_nlist_arrays as data:
        k = 0
        for i, (head, nn) in enumerate(zip(data.head_list, data.n_neigh)):
            for j_idx in range(head, head + nn):
                j = data.nlist[j_idx]
                pair = local_pair_list[k]
                assert i == pair[0]
                assert j == pair[1]
                k += 1
