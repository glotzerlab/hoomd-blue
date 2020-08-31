import hoomd
import pytest
import numpy
import itertools
from copy import deepcopy

def test_attach_Langevin(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=.9))
    integrator = hoomd.md.Integrator(.05)
    int_meth =  hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1, seed=1, tally_reservoir_energy=False)
    integrator.methods.append(int_meth)
    sim.operations.integrator = integrator
    sim.operations.schedule()
    int_method = { "kT": 1, "seed": 1, "alpha": None, "tally_reservoir_energy": False}
    
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

def test_attach_NVT(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=.9))
    integrator = hoomd.md.Integrator(.05)
    int_meth = hoomd.md.methods.NVT(hoomd.filter.All(), kT=1, tau=1) 
    integrator.methods.append(int_meth)
    sim.operations.integrator = integrator
    sim.operations.schedule()
    int_method = { "kT": 1, "tau": 1}
    
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

def test_attach_NPT(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=.9))
    integrator = hoomd.md.Integrator(.05)
    int_meth = hoomd.md.methods.NPT(hoomd.filter.All(), kT=1, tau=1, S=1, tauS=1, box_dof=[True,True,True,False,False,False], couple="all") 
    integrator.methods.append(int_meth)
    sim.operations.integrator = integrator
    sim.operations.schedule()
    int_method = { "kT": 1, "S": [1,1,1,0,0,0], "box_dof": [True,True,True,False,False,False], "tau": 1, "tauS": 1, "couple": "all", "rescale_all": False, "gamma": 0.0 } 
    
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
