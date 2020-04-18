import pytest
from hoomd.filter.type_ import Type
from copy import deepcopy


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
