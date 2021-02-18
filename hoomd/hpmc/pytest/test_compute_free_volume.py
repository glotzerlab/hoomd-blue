import hoomd
import pytest
import numpy as np


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
    if snap.exists:
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


_params1 = dict(particle_types=['A', 'B'], a=2.0, n=7)
_free_volume1 = (7 * 2)**3 - (7**3) * (4 / 3) * np.pi * 0.25**3

_params2 = dict(particle_types=['A', 'B'], a=1.5, n=7)
_free_volume2 = (7 * 1.5)**3 - (7**3) * (4 / 3) * np.pi * 0.25**3

_params3 = dict(particle_types=['A', 'B'], a=0.1, n=7)
_free_volume3 = 0.0

_validation_systems = [(_params1, _free_volume1),
                       (_params2, _free_volume2),
                       (_params3, _free_volume3)]

@pytest.mark.parametrize("snapshot_params, free_volume", _validation_systems)
def test_validation_systems(simulation_factory,
                            two_particle_snapshot_factory,
                            lattice_snapshot_factory,
                            snapshot_params,
                            free_volume):
    sim = simulation_factory(lattice_snapshot_factory(**snapshot_params))
    mc = hoomd.hpmc.integrate.Sphere(23456)
    mc.shape["A"] = {'diameter': 0.5}
    mc.shape["B"] = {'diameter': 0.1}
    mc.depletant_fugacity["B"] = 1.5
    sim.operations.add(mc)

    sim.operations._schedule()
    free_volume_compute = hoomd.hpmc.compute.FreeVolume(mc, 23456, test_type='B', nsample=1000)
    sim.operations.add(free_volume_compute)
    sim.run(0)
    np.testing.assert_allclose(free_volume, free_volume_compute.free_volume, rtol=1e-2)
