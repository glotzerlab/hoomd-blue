import hoomd
from hoomd.conftest import pickling_check
from hoomd.md.pair.alch import LJGauss, NVT


def test_before_attaching(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=1))
    ljg = LJGauss(hoomd.md.nlist.Cell(), default_r_cut=3.0)
    ljg.params[('A', 'A')] = dict(epsilon=1., sigma2=0.02, r0=1.8)
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(ljg)
    sim.operations.integrator = integrator
    sim.run(0)

    ar0 = ljg.alchemical_particles[('A', 'A'), 'r0']

    filt = hoomd.filter.All()
    kT = hoomd.variant.Constant(1)
    time_factor = 10

    anvt = NVT(filter=filt, kT=kT, time_factor=time_factor, alchemical_particles=[ar0])

    assert anvt.filter is filt
    assert anvt.kT == kT
    assert anvt.time_factor == time_factor

    kT = hoomd.variant.Constant(0.5)
    time_factor = 5
    anvt.kT = kT
    anvt.time_factor = time_factor
    assert anvt.kT == kT
    assert anvt.time_factor == time_factor

    assert len(anvt.alchemical_particles) == 1
    assert anvt.alchemical_particles[0] == ar0

    anvt.alchemical_particles.remove(ar0)
    assert len(anvt.alchemical_particles) == 0

    anvt.alchemical_particles.append(ar0)
    assert anvt.alchemical_particles[0] == ar0


def test_after_attaching(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=1))
    ljg = LJGauss(hoomd.md.nlist.Cell(), default_r_cut=3.0)
    ljg.params[('A', 'A')] = dict(epsilon=1., sigma2=0.02, r0=1.8)
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(ljg)
    sim.operations.integrator = integrator
    sim.run(0)

    ar0 = ljg.alchemical_particles[('A', 'A'), 'r0']

    filt = hoomd.filter.All()
    kT = hoomd.variant.Constant(1)
    time_factor = 10

    anvt = NVT(filter=filt, kT=kT, time_factor=time_factor, alchemical_particles=[ar0])
    sim.operations.integrator.methods.insert(0, anvt)
    sim.run(0)

    assert anvt.filter is filt
    assert anvt.kT == kT
    assert anvt.time_factor == time_factor

    kT = hoomd.variant.Constant(0.5)
    time_factor = 5
    anvt.kT = kT
    anvt.time_factor = time_factor
    assert anvt.kT == kT
    assert anvt.time_factor == time_factor

    assert len(anvt.alchemical_particles) == 1
    assert anvt.alchemical_particles[0] == ar0

    anvt.alchemical_particles.remove(ar0)
    assert len(anvt.alchemical_particles) == 0

    anvt.alchemical_particles.append(ar0)
    assert anvt.alchemical_particles[0] == ar0

    sim.run(10)


def test_pickling(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=1))
    ljg = LJGauss(hoomd.md.nlist.Cell(), default_r_cut=3.0)
    ljg.params[('A', 'A')] = dict(epsilon=1., sigma2=0.02, r0=1.8)
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(ljg)
    sim.operations.integrator = integrator
    sim.run(0)

    ar0 = ljg.alchemical_particles[('A', 'A'), 'r0']

    filt = hoomd.filter.All()
    kT = hoomd.variant.Constant(1)
    time_factor = 10

    anvt = NVT(filter=filt, kT=kT, time_factor=time_factor, alchemical_particles=[ar0])
    pickling_check(anvt)
    sim.operations.integrator.methods.insert(0, anvt)
    sim.run(0)
    pickling_check(anvt)
