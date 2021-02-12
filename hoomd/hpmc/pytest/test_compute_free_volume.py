import hoomd
import pytest


def test_before_attaching():
    mc = hoomd.hpmc.integrate.Sphere(23456)
    mc.shape["A"] = {'diameter': 1.0}
    mc.shape["B"] = {'diameter': 0.2}
    mc.depletant_fugacity["B"] = 1.5
    free_volume = hoomd.hpmc.compute.FreeVolume(mc, 23456, test_type='B', nsample=100)

    assert free_volume.mc == mc
    assert free_volume.seed == 23456
    assert free_volume.test_particle_type == 'B'
    assert free_volume.num_samples == 100
    assert free_volume.free_volume is None


def test_after_attaching(simulation_factory, lattice_snapshot_factory):
    snap = lattice_snapshot_factory(particle_types=['A', 'B'])
    for i in range(snap.particles.N):
        if i % 2 == 0:
            snap.particles.typeid[i] = 1
    sim = simulation_factory(snap)
    mc = hoomd.hpmc.integrate.Sphere(23456)
    mc.shape["A"] = {'diameter': 1.0}
    mc.shape["B"] = {'diameter': 0.2}
    mc.depletant_fugacity["B"] = 1.5
    sim.operations.add(mc)

    sim.operations._schedule()
    free_volume = hoomd.hpmc.compute.FreeVolume(mc, 23456, test_type='B', nsample=100)

    sim.operations.add(free_volume)
    assert len(sim.operations.computes) == 1
    sim.run(0)

    assert free_volume.test_particle_type == 1
    assert free_volume.num_samples == 100

    sim.run(10)
    assert isinstance(free_volume.free_volume, float)
