# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest
from hoomd.filter import (Type, Tags, SetDifference, Union, Intersection, All,
                          Null, Rigid)
from hoomd.snapshot import Snapshot
from copy import deepcopy
from itertools import combinations
import pickle
import numpy as np


@pytest.fixture(scope="function")
def make_filter_snapshot(device):

    def filter_snapshot(n=10, particle_types=['A']):
        s = Snapshot(device.communicator)
        if s.communicator.rank == 0:
            s.configuration.box = [20, 20, 20, 0, 0, 0]
            s.particles.N = n
            s.particles.position[:] = np.random.uniform(-10, 10, size=(n, 3))
            s.particles.types = particle_types
        return s

    return filter_snapshot


@pytest.mark.serial
def test_all_filter(make_filter_snapshot, simulation_factory):
    particle_types = ['A']
    N = 10
    filter_snapshot = make_filter_snapshot(n=N, particle_types=particle_types)
    sim = simulation_factory(filter_snapshot)
    all_filter = All()
    assert all_filter(sim.state) == list(range(N))


def test_null_filter(make_filter_snapshot, simulation_factory):
    particle_types = ['A']
    N = 10
    filter_snapshot = make_filter_snapshot(n=N, particle_types=particle_types)
    sim = simulation_factory(filter_snapshot)
    null_filter = Null()
    assert null_filter(sim.state) == []


def set_types(s, indices, particle_types, particle_type):
    for i in indices:
        s.particles.typeid[i] = particle_types.index(particle_type)


_type_indices = [([0, 3, 4, 8], [1, 2, 5, 6, 7, 9]),
                 ([2, 3, 5, 6, 7, 8, 9], [0, 1, 4]),
                 ([3, 7], [0, 1, 2, 4, 5, 6, 8, 9])]


@pytest.fixture(scope="function", params=_type_indices)
def type_indices(request):
    return deepcopy(request.param)


@pytest.mark.serial
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

    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        set_types(s, range(N), particle_types, "B")
    sim.state.set_snapshot(s)
    assert A_filter(sim.state) == []
    assert B_filter(sim.state) == list(range(N))

    A_indices, B_indices = type_indices
    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        set_types(s, A_indices, particle_types, "A")
        set_types(s, B_indices, particle_types, "B")
    sim.state.set_snapshot(s)
    assert A_filter(sim.state) == A_indices
    assert B_filter(sim.state) == B_indices
    assert AB_filter(sim.state) == list(range(N))


_tag_indices = [[0, 3, 4, 8], [1, 2, 5, 6, 7, 9], [2, 3, 5, 6, 7, 8, 9],
                [0, 1, 4], [3, 7], [0, 1, 2, 4, 5, 6, 8, 9]]


@pytest.fixture(scope="function", params=_tag_indices)
def tag_indices(request):
    return deepcopy(request.param)


def test_tags_filter(make_filter_snapshot, simulation_factory, tag_indices):
    particle_types = ['A']
    N = 10
    filter_snapshot = make_filter_snapshot(n=N, particle_types=particle_types)
    sim = simulation_factory(filter_snapshot)
    indices = tag_indices
    tag_filter = Tags(indices)
    assert tag_filter(sim.state) == indices


_set_indices = [([0, 3, 8], [1, 6, 7, 9], [2, 4, 5]),
                ([2, 3, 5, 7, 8], [0, 1, 4], [6, 9]),
                ([3], [0, 7, 8], [1, 2, 4, 5, 6, 9])]


@pytest.fixture(scope="function", params=_set_indices)
def set_indices(request):
    return deepcopy(request.param)


def test_tag_filter_equality():
    filter_a = Tags(tags=[0, 1, 2, 3])
    filter_b = Tags(tags=[2, 3, 4, 5, 6])
    filter_c = Tags(tags=[0, 1, 2, 3, 4])
    filter_d = Tags(tags=[0, 1, 2, 3])
    assert filter_a != filter_b
    assert filter_b != filter_c
    assert filter_d == filter_a


def type_not_in_combo(combo, particle_types):
    for particle_type in particle_types:
        if particle_type not in combo:
            return particle_type


def test_intersection(make_filter_snapshot, simulation_factory, set_indices):
    particle_types = ['A', 'B', 'C']
    N = 10
    filter_snapshot = make_filter_snapshot(n=N, particle_types=particle_types)
    sim = simulation_factory(filter_snapshot)
    A_indices, B_indices, C_indices = set_indices
    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        set_types(s, A_indices, particle_types, "A")
        set_types(s, B_indices, particle_types, "B")
        set_types(s, C_indices, particle_types, "C")
    sim.state.set_snapshot(s)

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
    A_indices, B_indices, C_indices = set_indices
    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        set_types(s, A_indices, particle_types, "A")
        set_types(s, B_indices, particle_types, "B")
        set_types(s, C_indices, particle_types, "C")
    sim.state.set_snapshot(s)

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
    A_indices, B_indices, C_indices = set_indices
    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        set_types(s, A_indices, particle_types, "A")
        set_types(s, B_indices, particle_types, "B")
        set_types(s, C_indices, particle_types, "C")
    sim.state.set_snapshot(s)

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


_filter_classes = [
    All,
    Tags,
    Type,
    Rigid,
    SetDifference,
    Union,
    Intersection,
]

_constructor_args = [
    (),
    ([1, 2, 3],),
    ({'a', 'b'},),
    (('center', 'free'),),
    (Tags([1, 4, 5]), Type({'a'})),
    (Tags([1, 4, 5]), Type({'a'})),
    (Tags([1, 4, 5]), Type({'a'})),
]


@pytest.mark.parametrize('constructor, args',
                         zip(_filter_classes, _constructor_args),
                         ids=lambda x: None
                         if isinstance(x, tuple) else x.__name__)
def test_pickling(constructor, args):
    filter_ = constructor(*args)
    pickled_filter = pickle.loads(pickle.dumps(filter_))
    assert pickled_filter == filter_
    assert hash(pickled_filter) == hash(filter_)
