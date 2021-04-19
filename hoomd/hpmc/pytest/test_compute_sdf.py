import hoomd
import pytest
import numpy as np


def test_before_attaching():
    sdf = hoomd.hpmc.compute.SDF(xmax=0.02, dx=1e-4)

    assert sdf.xmax == 0.02
    assert sdf.dx == 1e-4
    assert sdf.sdf is None


def test_after_attaching(simulation_factory, lattice_snapshot_factory):
    snap = lattice_snapshot_factory(particle_types=['A'])
    sim = simulation_factory(snap)
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape["A"] = {'diameter': 1.0}
    sim.operations.add(mc)

    sdf = hoomd.hpmc.compute.SDF(xmax=0.02, dx=1e-4)

    sim.operations.add(sdf)
    assert len(sim.operations.computes) == 1
    sim.run(0)

    assert sdf.xmax == 0.02
    assert sdf.dx == 1e-4

    sim.run(10)
    assert isinstance(sdf.sdf, list)
    assert len(sdf.sdf) > 0
