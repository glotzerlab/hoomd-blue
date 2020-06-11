import hoomd
import pytest
import numpy
import itertools
from copy import deepcopy

_method_data = [
    ("Langevin", hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1, seed=1)),
    ("NVT", hoomd.md.methods.NVT(hoomd.filter.All(), kT=1, tau=1)),
]

@pytest.fixture(scope='function', params=_method_data, ids=(lambda x: x[0]))
def int_method(request):
    return deepcopy(request.param[1])

def test_attach(simulation_factory, two_particle_snapshot_factory, int_method):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=.9))
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(int_method)
    sim.operations.integrator = integrator
    sim.operations.schedule()
    sim.run(10)

