import pytest
from hoomd.filter.type_ import Type
from hoomd.filter.tags import Tags
from hoomd.filter.set_ import SetDifference, Union, Intersection
from copy import deepcopy
from itertools import combinations


def set_types(s, inds, particle_types, particle_type):
    for i in inds:
        s.particles.typeid[i] = particle_types.index(particle_type)

_type_indices = [([0, 3, 4, 8], [1, 2, 5, 6, 7, 9]),
                 ([2, 3, 5, 6, 7, 8, 9], [0, 1, 4]),
                 ([3, 7], [0, 1, 2, 4, 5, 6, 8, 9])]


@pytest.fixture(scope="function", params=_type_indices)
def type_indices(request):
    return deepcopy(request.param)


def test_type_filter(dummy_simulation_factory, type_indices):
    particle_types = ['A', 'B']
    N = 10
    sim = dummy_simulation_factory(particle_types=particle_types, n=N)

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


def test_tags_filter(dummy_simulation_factory, tag_indices):
    particle_types = ['A']
    N = 10
    sim = dummy_simulation_factory(particle_types=particle_types, n=N)
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


def test_intersection(dummy_simulation_factory, set_indices):
    particle_types = ['A', 'B', 'C']
    N = 10
    sim = dummy_simulation_factory(particle_types=particle_types, n=N)
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
