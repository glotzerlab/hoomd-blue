import numpy as np
import pytest

import hoomd
import hoomd.conftest


@pytest.fixture
def filter_list():
    return [
        hoomd.filter.All(),
        hoomd.filter.Tags([1, 2, 3]),
        hoomd.filter.Type(["A"])
    ]


def test_initialization_setting(filter_list):
    filter_updater = hoomd.update.FilterUpdater(1, [])
    assert filter_updater.trigger == hoomd.trigger.Periodic(1)
    assert filter_updater.filters == []
    filter_updater.filters.extend(filter_list)
    assert len(filter_updater.filters) == 3
    assert filter_list == filter_updater.filters

    filter_updater = hoomd.update.FilterUpdater(5, filter_list)
    assert filter_updater.trigger == hoomd.trigger.Periodic(5)
    assert len(filter_updater.filters) == 3
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
    assert hasattr(filter_updater, "_cpp_obj")
    assert filter_updater._attached


def assert_group_match(filter_, state):
    filter_tags = filter_(state)
    group_tags = state._get_group(filter_).member_tags
    assert set(filter_tags) == set(group_tags)


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


def test_pickling(simulation, filter_updater):
    hoomd.conftest.operation_pickling_check(filter_updater, simulation)
