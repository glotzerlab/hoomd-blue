import hoomd
from hoomd.conftest import pickling_check
from hoomd.md.pair.alch import LJGauss, NVT, NVE
import pytest

_NVE_args = (NVE, {}, {})

_NVT_args = (NVT, {
    'kT': hoomd.variant.Constant(1)
}, {
    'kT': hoomd.variant.Constant(0.5)
})


def get_alchemostat():
    return [_NVE_args, _NVT_args]


@pytest.mark.parametrize(
    "alchemostat_cls, extra_property_1st_value, extra_property_2nd_value",
    get_alchemostat())
def test_before_attaching(simulation_factory, two_particle_snapshot_factory,
                          alchemostat_cls, extra_property_1st_value,
                          extra_property_2nd_value):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=1))
    ljg = LJGauss(hoomd.md.nlist.Cell(), default_r_cut=3.0)
    ljg.params[('A', 'A')] = dict(epsilon=1., sigma2=0.02, r0=1.8)
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(ljg)
    sim.operations.integrator = integrator
    sim.run(0)

    ar0 = ljg.alchemical_particles[('A', 'A'), 'r0']

    filt = hoomd.filter.All()
    time_factor = 10

    alchemostat = alchemostat_cls(filter=filt,
                                  time_factor=time_factor,
                                  alchemical_particles=[ar0],
                                  **extra_property_1st_value)

    assert alchemostat.filter is filt
    assert alchemostat.time_factor == time_factor
    time_factor = 5
    alchemostat.time_factor = time_factor
    assert alchemostat.time_factor == time_factor

    assert len(alchemostat.alchemical_particles) == 1
    assert alchemostat.alchemical_particles[0] == ar0

    alchemostat.alchemical_particles.remove(ar0)
    assert len(alchemostat.alchemical_particles) == 0

    alchemostat.alchemical_particles.append(ar0)
    assert alchemostat.alchemical_particles[0] == ar0

    for name in extra_property_1st_value.keys():
        assert getattr(alchemostat, name) == extra_property_1st_value[name]
        setattr(alchemostat, name, extra_property_2nd_value[name])
        assert getattr(alchemostat, name) == extra_property_2nd_value[name]


@pytest.mark.parametrize(
    "alchemostat_cls, extra_property_1st_value, extra_property_2nd_value",
    get_alchemostat())
def test_after_attaching(simulation_factory, two_particle_snapshot_factory,
                         alchemostat_cls, extra_property_1st_value,
                         extra_property_2nd_value):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=1))
    ljg = LJGauss(hoomd.md.nlist.Cell(), default_r_cut=3.0)
    ljg.params[('A', 'A')] = dict(epsilon=1., sigma2=0.02, r0=1.8)
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(ljg)
    sim.operations.integrator = integrator
    sim.run(0)

    ar0 = ljg.alchemical_particles[('A', 'A'), 'r0']

    filt = hoomd.filter.All()
    time_factor = 10

    alchemostat = alchemostat_cls(filter=filt,
                                  time_factor=time_factor,
                                  alchemical_particles=[ar0],
                                  **extra_property_1st_value)
    sim.operations.integrator.methods.insert(0, alchemostat)
    sim.run(0)

    assert alchemostat.filter is filt
    assert alchemostat.time_factor == time_factor
    time_factor = 5
    alchemostat.time_factor = time_factor
    assert alchemostat.time_factor == time_factor

    assert len(alchemostat.alchemical_particles) == 1
    assert alchemostat.alchemical_particles[0] == ar0

    alchemostat.alchemical_particles.remove(ar0)
    assert len(alchemostat.alchemical_particles) == 0

    alchemostat.alchemical_particles.append(ar0)
    assert alchemostat.alchemical_particles[0] == ar0

    for name in extra_property_1st_value.keys():
        assert getattr(alchemostat, name) == extra_property_1st_value[name]
        setattr(alchemostat, name, extra_property_2nd_value[name])
        assert getattr(alchemostat, name) == extra_property_2nd_value[name]

    sim.run(10)


@pytest.mark.parametrize(
    "alchemostat_cls, extra_property_1st_value, extra_property_2nd_value",
    get_alchemostat())
def test_pickling(simulation_factory, two_particle_snapshot_factory,
                  alchemostat_cls, extra_property_1st_value,
                  extra_property_2nd_value):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=1))
    ljg = LJGauss(hoomd.md.nlist.Cell(), default_r_cut=3.0)
    ljg.params[('A', 'A')] = dict(epsilon=1., sigma2=0.02, r0=1.8)
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(ljg)
    sim.operations.integrator = integrator
    sim.run(0)

    ar0 = ljg.alchemical_particles[('A', 'A'), 'r0']

    filt = hoomd.filter.All()
    time_factor = 10

    alchemostat = alchemostat_cls(filter=filt,
                                  time_factor=time_factor,
                                  alchemical_particles=[ar0],
                                  **extra_property_1st_value)
    pickling_check(alchemostat)
    sim.operations.integrator.methods.insert(0, alchemostat)
    sim.run(0)
    pickling_check(alchemostat)
