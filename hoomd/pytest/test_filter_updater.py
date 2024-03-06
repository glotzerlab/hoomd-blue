# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

import hoomd
import hoomd.conftest


@pytest.fixture
def filter_list():

    class NewFilter(hoomd.filter.CustomFilter):

        def __call__(self, state):
            return np.array([])

        def __hash__(self):
            return hash(self.__class__.__name__)

        def __eq__(self, other):
            return isinstance(self, other.__class__)

    return [
        hoomd.filter.All(),
        hoomd.filter.Tags([1, 2, 3]),
        hoomd.filter.Type(["A"]),
        NewFilter()
    ]


def test_initialization_setting(filter_list):
    filter_updater = hoomd.update.FilterUpdater(1, [])
    assert filter_updater.trigger == hoomd.trigger.Periodic(1)
    assert filter_updater.filters == []
    filter_updater.filters.extend(filter_list)
    assert len(filter_updater.filters) == 4
    assert filter_list == filter_updater.filters

    filter_updater = hoomd.update.FilterUpdater(5, filter_list)
    assert filter_updater.trigger == hoomd.trigger.Periodic(5)
    assert len(filter_updater.filters) == 4
    assert filter_list == filter_updater.filters
    filter_updater.trigger = hoomd.trigger.After(100)
    assert filter_updater.trigger == hoomd.trigger.After(100)


@pytest.fixture
def filter_updater(filter_list):
    return hoomd.update.FilterUpdater(1, filter_list)


@pytest.fixture(scope="function")
def simulation(lattice_snapshot_factory, simulation_factory, filter_list):
    sim = simulation_factory(
        lattice_snapshot_factory(particle_types=["A", "B"]))
    # place filters in state list manually to enable updating the particle
    # groups.
    for filter_ in filter_list:
        sim.state._get_group(filter_)
    return sim


def test_attaching(simulation, filter_updater):
    simulation.operations += filter_updater
    trigger = filter_updater.trigger
    filters = filter_updater.filters
    simulation.run(0)
    assert trigger == filter_updater.trigger
    assert filters == filter_updater.filters
    assert filter_updater._cpp_obj is not None
    assert filter_updater._attached


def assert_group_match(filter_, state, mpi=False):
    filter_tags = set(filter_(state))
    group_tags = set(state._get_group(filter_).member_tags)
    # On MPI simulations, the group tags won't exactly match since they include
    # particles from every rank, so two checks are necessary. One that no
    # particles in the filters tags are not in the groups tags (below), and that
    # all local tags in group tags are in filter tags (2nd check).
    assert filter_tags - group_tags == set()
    if not mpi:
        return
    NOT_LOCAL = 4294967295
    with state.cpu_local_snapshot as snapshot:
        np.all(snapshot.particles.rtag[group_tags - filter_tags] == NOT_LOCAL)


def test_updating(simulation, filter_updater, filter_list):
    simulation.operations += filter_updater
    simulation.run(0)
    rng = np.random.default_rng(43)

    def modify_typeid(state):
        with state.cpu_local_snapshot as snapshot:
            Np = len(snapshot.particles.typeid)
            indices = rng.choice(Np, max(1, int(Np * 0.1)), replace=False)
            values = rng.choice([0, 1], len(indices))
            snapshot.particles.typeid[indices] = values

    for _ in range(4):
        modify_typeid(simulation.state)
        simulation.run(1)
        for filter_ in filter_list:
            assert_group_match(filter_, simulation.state)


def test_pickling(simulation):
    # Don't use filter_list fixture since NewFilter is not picklable.
    filters = [
        hoomd.filter.All(),
        hoomd.filter.Tags([1, 2, 3]),
        hoomd.filter.Type(["A"]),
    ]
    hoomd.conftest.operation_pickling_check(
        hoomd.update.FilterUpdater(1, filters), simulation)
