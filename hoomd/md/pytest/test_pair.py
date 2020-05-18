import hoomd
import pytest
from copy import deepcopy

_pairs = [
    ["Gauss",
     hoomd.md.pair.Gauss(hoomd.md.nlist.Cell()),
     {'epsilon': 0.05, "sigma": 0.02}],
    ["LJ",
     hoomd.md.pair.LJ(hoomd.md.nlist.Cell()),
     {"epsilon": 0.0005, "sigma": 1}],
    ["Yukawa",
     hoomd.md.pair.Yukawa(hoomd.md.nlist.Cell()),
     {"epsilon": 0.0005, "kappa": 1}],
]

@pytest.fixture(scope='function', params=_pairs, ids=(lambda x: x[0]))
def pair_and_params(request):
    return deepcopy(request.param[1:])

def test_attach(simulation_factory, two_particle_snapshot_factory, pair_and_params):
    pair = pair_and_params[0]
    params = pair_and_params[1]

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=.5))
    integrator = hoomd.md.Integrator(dt=0.5)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1, seed=1))
    pair.params[('A', 'A')] = params
    pair.r_cut[('A', 'A')] = .02
    integrator.forces.append(pair)
    sim.operations.integrator = integrator
    sim.operations.schedule()
    sim.run(10)
