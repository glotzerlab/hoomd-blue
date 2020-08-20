import hoomd
import pytest
import numpy
import itertools
from copy import deepcopy

_method_data = [
    ("Langevin", { "method": hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1, seed=1, tally_reservoir_energy=False), "kT": 1, "seed": 1, "alpha": None, "tally_reservoir_energy": False}),
    ("NVT", { "method": hoomd.md.methods.NVT(hoomd.filter.All(), kT=1, tau=1), "kT": 1, "tau": 1}),
    ("NPT", { "method": hoomd.md.methods.NPT(hoomd.filter.All(), kT=1, tau=1, S=1, tauS=1, box_dof=[True,True,True,False,False,False], couple="xyz"), "kT": 1, "S": [1,1,1,0,0,0], "box_dof": [True,True,True,False,False,False], "tau":1, "tauS":1, "couple": "xyz", "rescale_all": False, "gamma": 0.0}),
]

@pytest.fixture(scope='function', params=_method_data, ids=(lambda x: x[0]))
def int_method(request):
    return deepcopy(request.param[1])

def test_attach(simulation_factory, two_particle_snapshot_factory, int_method):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=.9))
    integrator = hoomd.md.Integrator(.05)
    int_meth = int_method["method"]
    integrator.methods.append(int_meth)
    sim.operations.integrator = integrator
    sim.operations.schedule()
    
    sim.run(10)

    for k in int_meth._param_dict:
        if k != "filter":
           param = int_meth._getattr_param(k)
           param2 = int_method[k]
           if isinstance(param, list):
               for p in range(len(param)):
                   pa = param[p]
                   pa2 = param2[p]
                   if isinstance(pa, hoomd.variant.Constant):
                       pa = pa(0)
                   assert pa == pa2
           else:
              if isinstance(param, hoomd.variant.Constant):
                 param = param(0)
              assert param == param2


