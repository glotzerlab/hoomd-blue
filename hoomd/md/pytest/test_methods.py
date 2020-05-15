import hoomd
import pytest
import numpy
import itertools

def test_attach_nvt(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=.9))
    f = hoomd.filter.All()
    nvt = hoomd.md.methods.NVT(f, kT=1, tau=1)
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(nvt)
    sim.operations.integrator = integrator
    sim.operations.schedule()
    sim.run(10)

