# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import copy as cp
import hoomd
from hoomd.logging import LoggerCategories
import numpy as np
import pytest
import json
import random
import collections
from pathlib import Path
from hoomd.md.nlist import Cell, Stencil, Tree
from hoomd.conftest import (logging_check, pickling_check,
                            autotuned_kernel_parameter_check)

try:
    from mpi4py import MPI
    MPI4PY_IMPORTED = True
except ImportError:
    MPI4PY_IMPORTED = False

try:
    import cupy
    CUPY_IMPORTED = True
except ImportError:
    CUPY_IMPORTED = False


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


_path = Path(__file__).parent / "true_pair_list.json"
TRUE_PAIR_LIST = set([frozenset(pair) for pair in json.load(_path.open())])


def _setup_standard_rcut(sim_factory, snap_factory):
    nlist = hoomd.md.nlist.Cell(buffer=0.0, default_r_cut=1.1)
    sim: hoomd.Simulation = sim_factory(snap_factory())
    sim.operations.computes.append(nlist)
    sim.run(0)

    return sim, nlist, True


def _setup_no_rcut(sim_factory, snap_factory):
    nlist = hoomd.md.nlist.Tree(buffer=0.0, default_r_cut=0.0)
    sim: hoomd.Simulation = sim_factory(snap_factory())
    sim.operations.computes.append(nlist)
    sim.run(0)

    return sim, nlist, False


def _setup_set_rcut_later(sim_factory, snap_factory):
    nlist = hoomd.md.nlist.Cell(buffer=0.0, default_r_cut=0.0)
    sim: hoomd.Simulation = sim_factory(snap_factory())
    sim.operations.computes.append(nlist)
    sim.run(0)

    nlist.r_cut[('A', 'A')] = 1.1

    return sim, nlist, True


def _setup_with_force_no_rcut(sim_factory, snap_factory):
    nlist = hoomd.md.nlist.Tree(buffer=0.0, default_r_cut=0.0)
    sim: hoomd.Simulation = sim_factory(snap_factory())
    sim.operations.computes.append(nlist)

    integrator = hoomd.md.Integrator(0.005)
    lj = hoomd.md.pair.LJ(nlist, default_r_cut=0.0)
    lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
    integrator.forces.append(lj)
    integrator.methods.append(
        hoomd.md.methods.ConstantVolume(hoomd.filter.All()))
    sim.operations.integrator = integrator
    sim.run(0)

    return sim, nlist, False


def _setup_with_force_rcut_later(sim_factory, snap_factory):
    nlist = hoomd.md.nlist.Tree(buffer=0.0, default_r_cut=0.0)
    sim: hoomd.Simulation = sim_factory(snap_factory())
    sim.operations.computes.append(nlist)

    integrator = hoomd.md.Integrator(0.005)
    lj = hoomd.md.pair.LJ(nlist, default_r_cut=0.0)
    lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
    integrator.forces.append(lj)
    integrator.methods.append(
        hoomd.md.methods.ConstantVolume(hoomd.filter.All()))
    sim.operations.integrator = integrator
    sim.run(0)

    lj.r_cut[('A', 'A')] = 1.1

    return sim, nlist, True


def _setup_with_force_rcut_on_nlist(sim_factory, snap_factory):
    nlist = hoomd.md.nlist.Cell(buffer=0.0, default_r_cut=1.1)
    sim: hoomd.Simulation = sim_factory(snap_factory())
    sim.operations.computes.append(nlist)

    integrator = hoomd.md.Integrator(0.005)
    lj = hoomd.md.pair.LJ(nlist, default_r_cut=0.0)
    lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
    integrator.forces.append(lj)
    integrator.methods.append(
        hoomd.md.methods.ConstantVolume(hoomd.filter.All()))
    sim.operations.integrator = integrator
    sim.run(0)

    return sim, nlist, True


def _setup_with_force_drop_nlist(sim_factory, snap_factory):
    nlist = hoomd.md.nlist.Cell(buffer=0.0, default_r_cut=1.1)
    sim: hoomd.Simulation = sim_factory(snap_factory())
    sim.operations.computes.append(nlist)

    integrator = hoomd.md.Integrator(0.005)
    lj = hoomd.md.pair.LJ(nlist, default_r_cut=1.1)
    lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
    integrator.forces.append(lj)
    integrator.methods.append(
        hoomd.md.methods.ConstantVolume(hoomd.filter.All()))
    sim.operations.integrator = integrator
    sim.run(0)

    sim.operations.computes.clear()

    return sim, nlist, True


def _setup_with_force_drop_force(sim_factory, snap_factory):
    nlist = hoomd.md.nlist.Cell(buffer=0.0, default_r_cut=1.1)
    sim: hoomd.Simulation = sim_factory(snap_factory())
    sim.operations.computes.append(nlist)

    integrator = hoomd.md.Integrator(0.005)
    lj = hoomd.md.pair.LJ(nlist, default_r_cut=1.1)
    lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
    integrator.forces.append(lj)
    integrator.methods.append(
        hoomd.md.methods.ConstantVolume(hoomd.filter.All()))
    sim.operations.integrator = integrator
    sim.run(0)

    integrator.forces.clear()

    return sim, nlist, True


pair_setup_funcs = [
    _setup_standard_rcut, _setup_no_rcut, _setup_set_rcut_later,
    _setup_with_force_no_rcut, _setup_with_force_rcut_later,
    _setup_with_force_rcut_on_nlist, _setup_with_force_drop_nlist,
    _setup_with_force_drop_force
]


def _check_pair_set(sim, nlist, truth_set):
    """Asserts that the pair list is correct."""
    pair_list = nlist.pair_list
    if sim.device.communicator.rank == 0:
        assert len(pair_list) == len(truth_set)  # ensures no duplicates
        pair_list = set([frozenset(pair) for pair in pair_list])
        assert pair_list == truth_set


@pytest.mark.parametrize("setup", pair_setup_funcs)
def test_global_pair_list(simulation_factory, lattice_snapshot_factory, setup):

    sim, nlist, full = setup(simulation_factory, lattice_snapshot_factory)

    if full:
        truth_set = TRUE_PAIR_LIST
    else:
        truth_set = set()

    _check_pair_set(sim, nlist, truth_set)


def _check_local_pairs_with_mpi(tag_pair_list, broadcast=False):

    tag_pair_list = np.array(tag_pair_list, dtype=np.int32)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    local_count = np.uint32(tag_pair_list.size)
    counts = None
    if rank == 0:
        counts = np.empty(size, dtype=np.uint32)
    comm.Gather(local_count, counts, root=0)

    local_pairs = tag_pair_list
    global_pairs_buffer = None
    global_pairs = None

    if rank == 0:
        total_counts = int(np.sum(counts))
        displacements = np.zeros(size, dtype=np.uint32)
        displacements[1:] = displacements[:-1] + counts[:-1]
        array = np.empty((total_counts // 2, 2), dtype=np.uint32)
        global_pairs_buffer = [array, counts, displacements, MPI.UINT32_T]

    comm.Gatherv(local_pairs, global_pairs_buffer, root=0)

    if rank == 0:
        global_pairs = global_pairs_buffer[0]
        global_pair_set = set([frozenset(list(p)) for p in global_pairs])

        assert global_pair_set == TRUE_PAIR_LIST

    if broadcast:
        return comm.bcast(global_pairs, root=0)


def _check_local_pair_counts(sim, global_pairs, half_nlist=True):

    if half_nlist:
        local_count = 1
    else:
        local_count = 2

    local_counts = collections.Counter()
    ghost_counts = collections.Counter()
    with sim.state.cpu_local_snapshot as data:
        tags = data.particles.tag
        ghost_tags = data.particles.ghost_tag
        for pair in global_pairs:
            if pair[0] in tags and pair[1] in tags:
                local_counts[frozenset(pair)] += 1
            elif pair[0] in ghost_tags and pair[1] in tags:
                ghost_counts[frozenset(pair)] += 1
            elif pair[0] in tags and pair[1] in ghost_tags:
                ghost_counts[frozenset(pair)] += 1

    assert all([count == local_count for count in local_counts.values()])
    assert all([count == 2 for count in ghost_counts.values()])


@pytest.mark.parametrize("setup", pair_setup_funcs)
def test_rank_local_pair_list(simulation_factory, lattice_snapshot_factory,
                              setup):

    sim, nlist, full = setup(simulation_factory, lattice_snapshot_factory)

    if full:
        truth_set = TRUE_PAIR_LIST
    else:
        truth_set = set()

    local_pair_list = nlist.local_pair_list
    tag_pair_list = []
    with sim.state.cpu_local_snapshot as data:
        tags = data.particles.tag_with_ghost
        for pair in local_pair_list:
            tag_pair_list.append([tags[pair[0]], tags[pair[1]]])

    set_tag_pair_list = set([frozenset(p) for p in tag_pair_list])
    assert set_tag_pair_list.issubset(truth_set)

    if full and MPI4PY_IMPORTED:
        global_pairs = _check_local_pairs_with_mpi(tag_pair_list,
                                                   broadcast=True)

        _check_local_pair_counts(sim, global_pairs)


@pytest.mark.parametrize("setup", pair_setup_funcs)
def test_cpu_local_nlist_arrays(simulation_factory, lattice_snapshot_factory,
                                setup):

    sim, nlist, full = setup(simulation_factory, lattice_snapshot_factory)

    if full:
        truth_set = TRUE_PAIR_LIST
    else:
        truth_set = set()

    tag_pair_list = []
    with nlist.cpu_local_nlist_arrays as data:
        with sim.state.cpu_local_snapshot as snap_data:

            half_nlist = data.half_nlist

            tags = snap_data.particles.tag_with_ghost
            for i, (head, nn) in enumerate(zip(data.head_list, data.n_neigh)):
                for j_idx in range(head, head + nn):
                    j = data.nlist[j_idx]
                    tag_pair_list.append([tags[i], tags[j]])

    set_tag_pair_list = set([frozenset(p) for p in tag_pair_list])
    assert set_tag_pair_list.issubset(truth_set)

    if full and MPI4PY_IMPORTED:
        global_pairs = _check_local_pairs_with_mpi(tag_pair_list,
                                                   broadcast=True)

        _check_local_pair_counts(sim, global_pairs, half_nlist)


@pytest.mark.parametrize("setup", pair_setup_funcs)
def test_gpu_local_nlist_arrays(simulation_factory, lattice_snapshot_factory,
                                setup):

    sim, nlist, full = setup(simulation_factory, lattice_snapshot_factory)

    if full:
        truth_set = TRUE_PAIR_LIST
    else:
        truth_set = set()

    if isinstance(sim.device, hoomd.device.CPU):
        with pytest.raises(RuntimeError):
            with nlist.gpu_local_nlist_arrays as data:
                pass
        return

    if not CUPY_IMPORTED:
        pytest.skip("Cupy is not installed")

    get_local_pairs = cupy.RawKernel(
        r'''
extern "C" __global__
void get_local_pairs(
        const unsigned int N,
        const unsigned long* heads,
        const unsigned int* nns,
        const unsigned int* nlist,
        const unsigned int* tags,
        const unsigned long* offsets,
        unsigned long* pairs) {
    unsigned int i = (unsigned int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (i >= N)
        return;
    uint2* pair = (uint2*)pairs;
    unsigned long head = heads[i];
    unsigned int nn = nns[i];
    unsigned long offset = offsets[i];
    unsigned int tag_i = tags[i];
    for (unsigned int idx = 0; idx < nn; idx++) {
        unsigned int j = nlist[head + idx];
        pair[offset + idx] = make_uint2(tag_i, tags[j]);
    }
}
''', 'get_local_pairs')

    with nlist.gpu_local_nlist_arrays as data:
        with sim.state.gpu_local_snapshot as snap_data:
            tags = snap_data.particles.tag_with_ghost._coerce_to_ndarray()

            half_nlist = data.half_nlist

            head_list = data.head_list._coerce_to_ndarray()
            n_neigh = data.n_neigh._coerce_to_ndarray()
            raw_nlist = data.nlist._coerce_to_ndarray()

            N = int(head_list.size)
            n_pairs = int(cupy.sum(n_neigh))
            offsets = cupy.cumsum(n_neigh.astype(cupy.uint64)) \
                - n_neigh[0]
            device_local_pairs = cupy.zeros((n_pairs, 2), dtype=cupy.uint32)

            block = 256
            n_grid = (N + 255) // 256
            get_local_pairs((n_grid,), (block,),
                            (N, head_list, n_neigh, raw_nlist, tags, offsets,
                             device_local_pairs))

    local_pairs = cupy.asnumpy(device_local_pairs)

    set_tag_pair_list = set([frozenset(p) for p in local_pairs])
    assert set_tag_pair_list.issubset(truth_set)

    if full and MPI4PY_IMPORTED:
        global_pairs = _check_local_pairs_with_mpi(local_pairs, broadcast=True)

        _check_local_pair_counts(sim, global_pairs, half_nlist)
