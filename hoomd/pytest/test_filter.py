import pytest
from hoomd.filter import (Type, Tags, SetDifference, Union, Intersection, All,
                          Null, CustomFilter, Rigid)
from hoomd.snapshot import Snapshot
import hoomd.md as md
from copy import deepcopy
from itertools import combinations
import pickle
import numpy as np


@pytest.fixture(scope="function")
def make_filter_snapshot(device):

    def filter_snapshot(n=10, particle_types=['A']):
        s = Snapshot(device.communicator)
        if s.exists:
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


def set_types(s, inds, particle_types, particle_type):
    for i in inds:
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
    inds = tag_indices
    tag_filter = Tags(inds)
    assert tag_filter(sim.state) == inds


def test_rigid_filter(make_filter_snapshot, simulation_factory):
    rigid = md.constrain.Rigid()
    rigid.body["A"] = {
        "constituent_types": ["B", "B", "B", "B"],
        "positions": [
            [1, 0, -1 / (2**(1. / 2.))],
            [-1, 0, -1 / (2**(1. / 2.))],
            [0, -1, 1 / (2**(1. / 2.))],
            [0, 1, 1 / (2**(1. / 2.))],
        ],
        "orientations": [(1.0, 0.0, 0.0, 0.0)] * 4,
        "charges": [0.0, 1.0, 2.0, 3.5],
        "diameters": [1.0, 1.5, 0.5, 1.0]
    }

    snapshot = make_filter_snapshot(n=100, particle_types=["A", "B", "C"])
    snapshot.particles.typeid[50:100] = 2
    snapshot.particles.body[50:100] = -1
    sim = simulation_factory(snapshot)
    rigid.create_bodies(sim.state)

    only_centers = Rigid()
    assert np.array_equal(np.sort(only_centers(sim.state)), np.arange(50))

    only_free = Rigid(('free',))
    assert np.array_equal(np.sort(only_free(sim.state)), np.arange(50, 100))

    only_constituent = Rigid(('constituent',))
    assert np.array_equal(np.sort(only_constituent(sim.state)),
                          np.arange(100, 300))

    free_and_centers = Rigid(('free', 'center'))
    assert np.array_equal(np.sort(free_and_centers(sim.state)),
                          np.arange(0, 100))

    constituent_and_centers = Rigid(('constituent', 'center'))
    assert np.array_equal(
        np.sort(constituent_and_centers(sim.state)),
        np.concatenate((np.arange(0, 50), np.arange(100, 300))))

    constituent_and_free = Rigid(('free', 'constituent'))
    assert np.array_equal(np.sort(constituent_and_free(sim.state)),
                          np.arange(50, 300))

    all_ = Rigid(('free', 'constituent', 'center'))
    assert np.array_equal(np.sort(all_(sim.state)), np.arange(0, 300))


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
    pkled_filter = pickle.loads(pickle.dumps(filter_))
    assert pkled_filter == filter_
    assert hash(pkled_filter) == hash(filter_)


def test_custom_filter(make_filter_snapshot, simulation_factory):
    """Tests that custom particle filters work on simulations.

    Specifically we test that using the Langevin integrator method, that only
    particles selected by the custom filter move. Since the Langevin method uses
    random movements we don't need to initialize velocities or have any forces
    to test this.
    """

    class NegativeCharge(CustomFilter):
        """Grab all particles with a negative charge."""

        def __call__(self, state):
            with state.cpu_local_snapshot as snap:
                return snap.particles.tag[snap.particles.charge < 0]

        def __hash__(self):
            return hash(self.__class__.__name__)

        def __eq__(self, other):
            return isinstance(other, self.__class__)

    charge_filter = NegativeCharge()
    sim = simulation_factory(make_filter_snapshot())
    # grabs tags on individual MPI ranks
    with sim.state.cpu_local_snapshot as snap:
        # Grab half of all particles on an MPI rank, 1 particle, or no particles
        # depending on how many particles are local to the MPI ranks.
        local_Np = snap.particles.charge.shape[0]
        N_negative_charge = max(0, max(1, int(local_Np * 0.5)))
        negative_charge_ind = np.random.choice(local_Np,
                                               N_negative_charge,
                                               replace=False)
        # Get the expected tags returned by the custom filter and the positions
        # that should vary and remain static for testing after running.
        snap.particles.charge[negative_charge_ind] = -1.0
        expected_tags = snap.particles.tag[negative_charge_ind]
        positive_charge_tags = snap.particles.tag[snap.particles.charge > 0]
        positive_charge_ind = snap.particles.rtag[positive_charge_tags]
        original_positions = snap.particles.position[negative_charge_ind]
        static_positions = snap.particles.position[positive_charge_ind]

    # Test that the filter merely works as expected and that tags are correctly
    # grabbed on local MPI ranks
    assert all(np.sort(charge_filter(sim.state)) == np.sort(expected_tags))

    # Test that the filter works when used in a simulation
    langevin = md.methods.Langevin(charge_filter, 1.0)
    sim.operations += md.Integrator(0.005, methods=[langevin])
    sim.run(100)
    snap = sim.state.snapshot
    if snap.exists:
        assert not np.allclose(snap.particles.position[negative_charge_ind],
                               original_positions)
        assert np.allclose(snap.particles.position[positive_charge_tags],
                           static_positions)
