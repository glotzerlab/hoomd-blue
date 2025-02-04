# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.conftest import pickling_check
import hoomd.md.alchemy
import pytest

_NVT_args = (hoomd.md.alchemy.methods.NVT, {
    'alchemical_kT': hoomd.variant.Constant(1)
}, {
    'alchemical_kT': hoomd.variant.Constant(0.5)
})


def get_alchemostat():
    return [_NVT_args]


@pytest.mark.parametrize(
    "alchemostat_cls, extra_property_1st_value, extra_property_2nd_value",
    get_alchemostat())
@pytest.mark.cpu
@pytest.mark.serial
def test_before_attaching(simulation_factory, two_particle_snapshot_factory,
                          alchemostat_cls, extra_property_1st_value,
                          extra_property_2nd_value):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=1))
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    ljg = hoomd.md.alchemy.pair.LJGauss(nlist, default_r_cut=3.0)
    ljg.params[('A', 'A')] = dict(epsilon=1., sigma=0.02, r0=1.8)
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(ljg)
    sim.operations.integrator = integrator
    r0_alchemical_dof = ljg.r0[('A', 'A')]
    period = 10
    alchemostat = alchemostat_cls(period=period,
                                  alchemical_dof=[r0_alchemical_dof],
                                  **extra_property_1st_value)

    assert alchemostat.period == period
    period = 5
    alchemostat.period = period
    assert alchemostat.period == period

    assert len(alchemostat.alchemical_dof) == 1
    assert alchemostat.alchemical_dof[0] == r0_alchemical_dof

    alchemostat.alchemical_dof.remove(r0_alchemical_dof)
    assert len(alchemostat.alchemical_dof) == 0

    alchemostat.alchemical_dof.append(r0_alchemical_dof)
    assert alchemostat.alchemical_dof[0] == r0_alchemical_dof

    for name in extra_property_1st_value.keys():
        assert getattr(alchemostat, name) == extra_property_1st_value[name]
        setattr(alchemostat, name, extra_property_2nd_value[name])
        assert getattr(alchemostat, name) == extra_property_2nd_value[name]


@pytest.mark.parametrize(
    "alchemostat_cls, extra_property_1st_value, extra_property_2nd_value",
    get_alchemostat())
@pytest.mark.cpu
@pytest.mark.serial
def test_after_attaching(simulation_factory, two_particle_snapshot_factory,
                         alchemostat_cls, extra_property_1st_value,
                         extra_property_2nd_value):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=1))
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    ljg = hoomd.md.alchemy.pair.LJGauss(nlist, default_r_cut=3.0)
    ljg.params[('A', 'A')] = dict(epsilon=1., sigma=0.02, r0=1.8)
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(ljg)
    sim.operations.integrator = integrator
    r0_alchemical_dof = ljg.r0[('A', 'A')]
    period = 10
    alchemostat = alchemostat_cls(period=period,
                                  alchemical_dof=[r0_alchemical_dof],
                                  **extra_property_1st_value)
    sim.operations.integrator.methods.insert(0, alchemostat)
    assert alchemostat.period == period
    assert len(alchemostat.alchemical_dof) == 1
    assert alchemostat.alchemical_dof[0] == r0_alchemical_dof

    alchemostat.alchemical_dof.remove(r0_alchemical_dof)
    assert len(alchemostat.alchemical_dof) == 0

    alchemostat.alchemical_dof.append(r0_alchemical_dof)
    assert alchemostat.alchemical_dof[0] == r0_alchemical_dof

    for name in extra_property_1st_value.keys():
        assert getattr(alchemostat, name) == extra_property_1st_value[name]
        setattr(alchemostat, name, extra_property_2nd_value[name])
        assert getattr(alchemostat, name) == extra_property_2nd_value[name]

    sim.run(10)


@pytest.mark.cpu
@pytest.mark.serial
@pytest.mark.parametrize("alchemical_potential",
                         [hoomd.md.alchemy.pair.LJGauss])
def test_pickling_potential(simulation_factory, two_particle_snapshot_factory,
                            alchemical_potential):
    """Test that md.constrain.Distance can be pickled and unpickled."""
    # detached
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    ljg = alchemical_potential(nlist, default_r_cut=3.0)
    ljg.params[('A', 'A')] = dict(epsilon=1., sigma=0.02, r0=1.8)
    pickling_check(ljg)

    # attached
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=1))
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(ljg)
    sim.operations.integrator = integrator
    sim.run(0)
    pickling_check(ljg)
