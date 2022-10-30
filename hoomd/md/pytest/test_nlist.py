# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import copy as cp
import hoomd
from hoomd.logging import LoggerCategories
import numpy as np
import pytest
import json
import random
from pathlib import Path
from hoomd.md.nlist import Cell, Stencil, Tree
from hoomd.conftest import (logging_check, pickling_check,
                            autotuned_kernel_parameter_check)

try:
    from mpi4py import MPI
    MPI4PY_IMPORTED = True
except ImportError:
    MPI4PY_IMPORTED = False


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


path = Path(__file__).parent / "true_pair_list.json"
_TRUE_PAIR_LIST = json.load(path.open())
TRUE_PAIR_LIST = set([frozenset(pair) for pair in _TRUE_PAIR_LIST])


def test_global_pair_list(simulation_factory, lattice_snapshot_factory):

    # start off with only the neighborlist, stand alone compute test
    nlist = hoomd.md.nlist.Tree(buffer=0, default_r_cut=0.0)
    sim: hoomd.Simulation = simulation_factory(lattice_snapshot_factory())
    sim.operations.computes.append(nlist)

    sim.run(0)
    assert nlist._attached
    pair_list = nlist.pair_list
    if sim.device.communicator.rank == 0:
        pair_list = set([frozenset(pair) for pair in pair_list])
        assert pair_list == set()

    # set r_cut
    nlist.r_cut[('A', 'A')] = 1.1
    # BUG: with MPI and domain decomposition
    # it appears an explicit `run` is required for the ghost layer to be updated
    sim.run(0)
    pair_list = nlist.pair_list
    if sim.device.communicator.rank == 0:
        pair_list = set([frozenset(pair) for pair in pair_list])
        assert pair_list == TRUE_PAIR_LIST

    sim.operations.computes.clear()
    assert not nlist._attached
    with pytest.raises(hoomd.error.DataAccessError):
        nlist.pair_list

    # remove as stand alone compute and add force compute
    sim.operations.computes.clear()
    integrator = hoomd.md.Integrator(0.005)
    lj = hoomd.md.pair.LJ(nlist, default_r_cut=0.0)
    lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
    integrator.forces.append(lj)
    integrator.methods.append(hoomd.md.methods.NVE(hoomd.filter.All()))
    sim.operations.integrator = integrator

    pair_list = nlist.pair_list
    if sim.device.communicator.rank == 0:
        pair_list = set([frozenset(pair) for pair in pair_list])
        assert pair_list == TRUE_PAIR_LIST

    # now relax nlist.r_cut
    nlist.r_cut[('A', 'A')] = 0.0
    pair_list = nlist.pair_list
    if sim.device.communicator.rank == 0:
        pair_list = set([frozenset(pair) for pair in pair_list])
        assert pair_list == set()

    # now set lj.r_cut
    lj.r_cut[('A', 'A')] = 1.1
    pair_list = nlist.pair_list
    if sim.device.communicator.rank == 0:
        pair_list = set([frozenset(pair) for pair in pair_list])
        assert pair_list == TRUE_PAIR_LIST

    # zero lj.r_cut again
    lj.r_cut[('A', 'A')] = 0.0
    pair_list = nlist.pair_list
    if sim.device.communicator.rank == 0:
        pair_list = set([frozenset(pair) for pair in pair_list])
        assert pair_list == set()

    # set nlist.r_cut again
    nlist.r_cut[('A', 'A')] = 1.1
    pair_list = nlist.pair_list
    if sim.device.communicator.rank == 0:
        pair_list = set([frozenset(pair) for pair in pair_list])
        assert pair_list == TRUE_PAIR_LIST

    # re-add nlist and remove force compute
    nlist.r_cut[('A', 'A')] = 0.0
    sim.operations.computes.append(nlist)
    sim.operations.integrator.forces.clear()
    pair_list = nlist.pair_list
    if sim.device.communicator.rank == 0:
        pair_list = set([frozenset(pair) for pair in pair_list])
        assert pair_list == set()

    # setting lj.r_cut should not change the pair list since it is not attached
    lj.r_cut[('A', 'A')] = 1.1
    pair_list = nlist.pair_list
    if sim.device.communicator.rank == 0:
        pair_list = set([frozenset(pair) for pair in pair_list])
        assert pair_list == set()

    # remove stand alone nlist compute, nlist should no longer be connected to
    # the simulation
    sim.operations.computes.clear()
    assert not nlist._attached
    with pytest.raises(hoomd.error.DataAccessError):
        nlist.pair_list

    # add back nlist compute and set nlist.r_cut
    nlist.r_cut[('A', 'A')] = 1.1
    sim.operations.computes.append(nlist)
    pair_list = nlist.pair_list
    if sim.device.communicator.rank == 0:
        pair_list = set([frozenset(pair) for pair in pair_list])
        assert pair_list == TRUE_PAIR_LIST


def test_rank_local_pair_list(simulation_factory, lattice_snapshot_factory):

    nlist = hoomd.md.nlist.Tree(buffer=0.0, default_r_cut=1.1)
    sim: hoomd.Simulation = simulation_factory(lattice_snapshot_factory())
    sim.operations.computes.append(nlist)

    sim.run(0)

    local_pair_list = nlist.local_pair_list
    tag_pair_list = []
    with sim.state.cpu_local_snapshot as data:
        tags = data.particles.tag_with_ghost
        for pair in local_pair_list:
            tag_pair_list.append([tags[pair[0]], tags[pair[1]]])

    set_tag_pair_list = set([frozenset(p) for p in tag_pair_list])
    assert set_tag_pair_list.issubset(TRUE_PAIR_LIST)

    if MPI4PY_IMPORTED:

        tag_pair_list = np.array(tag_pair_list, dtype=np.int32)

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        local_count = np.int32(len(tag_pair_list) * 2)
        counts = None
        if rank == 0:
            counts = np.empty(size, dtype=np.int32)
        comm.Gather(local_count, counts, root=0)

        local_pairs = tag_pair_list
        global_pairs_buffer = None

        if rank == 0:
            total_counts = np.sum(counts)
            displacements = np.cumsum(counts) - counts[0]
            global_pairs_buffer = [
                np.empty((total_counts // 2, 2), dtype=np.int32), counts,
                displacements, MPI.UINT32_T
            ]

        comm.Gatherv(local_pairs, global_pairs_buffer, root=0)

        if rank == 0:
            global_pairs = global_pairs_buffer[0]
            global_pair_set = set([frozenset(list(p)) for p in global_pairs])

            assert global_pair_set == TRUE_PAIR_LIST


def test_rank_local_nlist_arrays(simulation_factory, lattice_snapshot_factory):

    nlist = hoomd.md.nlist.Cell(buffer=0.0, default_r_cut=1.1)
    sim = simulation_factory(lattice_snapshot_factory())
    sim.operations.computes.append(nlist)

    sim.run(0)

    with nlist.cpu_local_nlist_arrays as data:
        with sim.state.cpu_local_snapshot as snap_data:
            tags = snap_data.particles.tag_with_ghost
            for i, (head, nn) in enumerate(zip(data.head_list, data.n_neigh)):
                for j_idx in range(head, head + nn):
                    j = data.nlist[j_idx]
                    assert frozenset(tags[i], tags[j]) in TRUE_PAIR_LIST
