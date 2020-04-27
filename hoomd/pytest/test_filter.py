import pytest
from hoomd.filter.type_ import Type
from hoomd.filter.tags import Tags
from hoomd.filter.set_ import SetDifference, Union, Intersection
from hoomd.snapshot import Snapshot
from copy import deepcopy
from itertools import combinations
import numpy


@pytest.fixture(scope="function")
def make_filter_snapshot(device):
    def filter_snapshot(n=10, particle_types=['A']):
        s = Snapshot(device.comm)
        if s.exists:
            s.configuration.box = [20, 20, 20, 0, 0, 0]
            s.particles.N = n
            s.particles.position[:] = numpy.random.uniform(-10, 10, size=(n, 3))
            s.particles.types = particle_types
        return s
    return filter_snapshot


def set_types(s, inds, particle_types, particle_type):
    for i in inds:
        s.particles.typeid[i] = particle_types.index(particle_type)

_type_indices = [([0, 3, 4, 8], [1, 2, 5, 6, 7, 9]),
                 ([2, 3, 5, 6, 7, 8, 9], [0, 1, 4]),
                 ([3, 7], [0, 1, 2, 4, 5, 6, 8, 9])]


@pytest.fixture(scope="function", params=_type_indices)
def type_indices(request):
    return deepcopy(request.param)


def test_type_filter(make_filter_snapshot, simulation_factory, type_indices):
    particle_types = ['A', 'B']
    N = 10
    filter_snapshot = make_filter_snapshot(n=N, particle_types=particle_types)
    sim = simulation_factory(filter_snapshot)

    A_filter = Type(["A"])
    B_filter = Type(["B"])
    AB_filter = Type(["A", "B"])
    assert A_filter(sim.state) == list(range(N))
    assert B_filter(sim.state) == []

    s = sim.state.snapshot
    if s.exists:
        set_types(s, range(N), particle_types, "B")
    sim.state.snapshot = s
    assert A_filter(sim.state) == []
    assert B_filter(sim.state) == list(range(N))

    A_inds, B_inds = type_indices
    s = sim.state.snapshot
    if s.exists:
        set_types(s, A_inds, particle_types, "A")
        set_types(s, B_inds, particle_types, "B")
    sim.state.snapshot = s
    assert A_filter(sim.state) == A_inds
    assert B_filter(sim.state) == B_inds
    assert AB_filter(sim.state) == list(range(N))

_tag_indices = [[0, 3, 4, 8],
                [1, 2, 5, 6, 7, 9],
                [2, 3, 5, 6, 7, 8, 9],
                [0, 1, 4],
                [3, 7],
                [0, 1, 2, 4, 5, 6, 8, 9]]


@pytest.fixture(scope="function", params=_tag_indices)
def tag_indices(request):
    return deepcopy(request.param)


def test_tags_filter(make_filter_snapshot, simulation_factory, tag_indices):
    particle_types = ['A']
    N = 10
    filter_snapshot = make_filter_snapshot(n=N, particle_types=particle_types)
    sim = simulation_factory(filter_snapshot)
    inds = tag_indices
    tag_filter = Tags(inds)
    assert tag_filter(sim.state) == inds

_set_indices = [([0, 3, 8], [1, 6, 7, 9], [2, 4, 5]),
                ([2, 3, 5, 7, 8], [0, 1, 4], [6, 9]),
                ([3], [0, 7, 8], [1, 2, 4, 5, 6, 9])]


@pytest.fixture(scope="function", params=_set_indices)
def set_indices(request):
    return deepcopy(request.param)


def type_not_in_combo(combo, particle_types):
    for particle_type in particle_types:
        if particle_type not in combo:
            return particle_type


def test_intersection(make_filter_snapshot, simulation_factory, set_indices):
    particle_types = ['A', 'B', 'C']
    N = 10
    filter_snapshot = make_filter_snapshot(n=N, particle_types=particle_types)
    sim = simulation_factory(filter_snapshot)
    A_inds, B_inds, C_inds = set_indices
    s = sim.state.snapshot
    if s.exists:
        set_types(s, A_inds, particle_types, "A")
        set_types(s, B_inds, particle_types, "B")
        set_types(s, C_inds, particle_types, "C")
    sim.state.snapshot = s

    for type_combo in combinations(particle_types, 2):
        combo_filter = Type(type_combo)
        for particle_type in type_combo:
            type_filter = Type([particle_type])
            intersection_filter = Intersection(combo_filter, type_filter)
            assert intersection_filter(sim.state) == type_filter(sim.state)
        remaining_type = type_not_in_combo(type_combo, particle_types)
        remaining_filter = Type([remaining_type])
        intersection_filter = Intersection(combo_filter, remaining_filter)
        assert intersection_filter(sim.state) == []


def test_union(make_filter_snapshot, simulation_factory, set_indices):
    particle_types = ['A', 'B', 'C']
    N = 10
    filter_snapshot = make_filter_snapshot(n=N, particle_types=particle_types)
    sim = simulation_factory(filter_snapshot)
    A_inds, B_inds, C_inds = set_indices
    s = sim.state.snapshot
    if s.exists:
        set_types(s, A_inds, particle_types, "A")
        set_types(s, B_inds, particle_types, "B")
        set_types(s, C_inds, particle_types, "C")
    sim.state.snapshot = s

    for type_combo in combinations(particle_types, 2):
        filter1 = Type([type_combo[0]])
        filter2 = Type([type_combo[1]])
        combo_filter = Type(type_combo)
        union_filter = Union(filter1, filter2)
        assert union_filter(sim.state) == combo_filter(sim.state)


def test_difference(make_filter_snapshot, simulation_factory, set_indices):
    particle_types = ['A', 'B', 'C']
    N = 10
    filter_snapshot = make_filter_snapshot(n=N, particle_types=particle_types)
    sim = simulation_factory(filter_snapshot)
    A_inds, B_inds, C_inds = set_indices
    s = sim.state.snapshot
    if s.exists:
        set_types(s, A_inds, particle_types, "A")
        set_types(s, B_inds, particle_types, "B")
        set_types(s, C_inds, particle_types, "C")
    sim.state.snapshot = s

    for type_combo in combinations(particle_types, 2):
        combo_filter = Type(type_combo)
        remaining_type = type_not_in_combo(type_combo, particle_types)
        remaining_filter = Type([remaining_type])
        for i in [0, 1]:
            type_filter1 = Type([type_combo[i]])
            type_filter2 = Type([type_combo[i - 1]])
            difference_filter = SetDifference(combo_filter, type_filter1)
            assert difference_filter(sim.state) == type_filter2(sim.state)
            difference_filter = SetDifference(combo_filter, type_filter2)
            assert difference_filter(sim.state) == type_filter1(sim.state)
        difference_filter = SetDifference(combo_filter, remaining_filter)
        assert difference_filter(sim.state) == combo_filter(sim.state)
