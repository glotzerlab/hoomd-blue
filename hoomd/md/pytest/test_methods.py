import hoomd
import pytest
import numpy
import itertools

def attach_and_sim(simulation_factory, two_particle_snapshot_factory, method):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=.9))
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(method)
    sim.operations.integrator = integrator
    sim.operations.schedule()
    sim.run(10)

def test_attach_nvt(simulation_factory, two_particle_snapshot_factory):
    nvt = hoomd.md.methods.NVT(hoomd.filter.All(), kT=1, tau=1)
    attach_and_sim(simulation_factory, two_particle_snapshot_factory, nvt)

def test_attach_langevin(simulation_factory, two_particle_snapshot_factory):
    lang = hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1, seed=1)
    attach_and_sim(simulation_factory, two_particle_snapshot_factory, lang)
