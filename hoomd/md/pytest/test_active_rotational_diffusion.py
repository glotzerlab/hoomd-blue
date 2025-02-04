# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

import hoomd
import hoomd.conftest


@pytest.fixture(
    params=[
        (hoomd.md.force.Active, {
            "filter": hoomd.filter.All()
        }),
        (
            hoomd.md.force.ActiveOnManifold,
            {
                "filter": hoomd.filter.All(),
                # this is the shift used by two_particle_snapshot_factory
                "manifold_constraint": hoomd.md.manifold.Plane(shift=0.1)
            })
    ],
    ids=lambda x: x[0].__name__)
def active_force(request):
    cls, kwargs = request.param
    yield cls(**kwargs)


def test_construction(active_force):
    rd_updater = hoomd.md.update.ActiveRotationalDiffusion(
        10, active_force, 0.1)

    # We want to test identity for active force since the two are linked.
    assert rd_updater.active_force is active_force
    assert rd_updater.trigger == hoomd.trigger.Periodic(10)
    assert rd_updater.rotational_diffusion == hoomd.variant.Constant(0.1)

    after_trigger = hoomd.trigger.After(100)
    ramp_variant = hoomd.variant.Ramp(0.1, 1., 100, 1_000)
    rd_updater = hoomd.md.update.ActiveRotationalDiffusion(
        after_trigger, active_force, ramp_variant)

    assert rd_updater.active_force is active_force
    assert rd_updater.trigger == after_trigger
    assert rd_updater.rotational_diffusion == ramp_variant


def check_setting(active_force, rd_updater):
    with pytest.raises(ValueError):
        rd_updater.active_force = active_force

    rd_updater.trigger = 100
    assert rd_updater.trigger == hoomd.trigger.Periodic(100)
    on_trigger = hoomd.trigger.On(100)
    rd_updater.trigger = on_trigger
    assert rd_updater.trigger == on_trigger

    rd_updater.rotational_diffusion = 1.0
    assert rd_updater.rotational_diffusion == hoomd.variant.Constant(1.0)
    power_variant = hoomd.variant.Power(0.1, 1.0, 3, 100, 1_000)
    rd_updater.rotational_diffusion = power_variant
    assert rd_updater.rotational_diffusion == power_variant


def test_setting(active_force):
    rd_updater = hoomd.md.update.ActiveRotationalDiffusion(
        10, active_force, 0.1)
    check_setting(active_force, rd_updater)


@pytest.fixture(scope="function")
def local_simulation_factory(simulation_factory, two_particle_snapshot_factory):
    """Creates simulation with state initialized.

    Note:
        The positions of the particles correspond correctly to the xy plane
        manifold used for tests. Do not change positions unless sure that the
        manifold tests with running simulation will not error.
    """

    def sim_constructor(active_force=None, rd_updater=None):
        sim = simulation_factory(two_particle_snapshot_factory())
        if isinstance(active_force, hoomd.md.force.Active):
            method = hoomd.md.methods.ConstantVolume(hoomd.filter.All())
        else:
            method = hoomd.md.methods.rattle.NVE(hoomd.filter.All(),
                                                 hoomd.md.manifold.Plane(0.1))
        sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                        methods=[method])
        if active_force is not None:
            sim.operations.integrator.forces.append(active_force)
        if rd_updater is not None:
            sim.operations.updaters.append(rd_updater)

        return sim

    return sim_constructor


def test_attaching(active_force, local_simulation_factory):
    rd_updater = hoomd.md.update.ActiveRotationalDiffusion(
        10, active_force, 0.1)
    sim = local_simulation_factory(active_force, rd_updater)
    sim.run(0)
    check_setting(active_force, rd_updater)

    sim.operations.integrator.forces.remove(active_force)
    assert not any(up is rd_updater for up in sim.operations.updaters)
    sim.operations.integrator.forces.clear()

    # Reset simulation to test for variouos error conditions
    sim.operations._unschedule()

    # ActiveRotationalDiffusion should error when active force is not attached
    sim.operations += rd_updater
    with pytest.raises(hoomd.error.SimulationDefinitionError):
        sim.run(0)

    # Add the active force to another simulation and ensure proper erroring.
    second_sim = local_simulation_factory(active_force)
    second_sim.run(0)
    sim.operations -= rd_updater
    second_sim.operations += rd_updater
    second_sim.run(0)


def test_update(active_force, local_simulation_factory):
    active_force.active_force.default = (1., 0., 0.)
    # Set torque to zero so no angular momentum exists to change orientations.
    active_force.active_torque.default = (0., 0., 0.)
    rd_updater = hoomd.md.update.ActiveRotationalDiffusion(1, active_force, 0.1)
    sim = local_simulation_factory(active_force, rd_updater)
    snapshot = sim.state.get_snapshot()
    if sim.device.communicator.rank == 0:
        old_orientations = snapshot.particles.orientation

    sim.run(10)
    snapshot = sim.state.get_snapshot()
    if sim.device.communicator.rank == 0:
        new_orientations = snapshot.particles.orientation
        assert not np.allclose(old_orientations, new_orientations)


def test_pickling(active_force, local_simulation_factory):
    # don't add the rd_updater since operation_pickling_check will deal with
    # that.
    sim = local_simulation_factory(active_force)
    rd_updater = hoomd.md.update.ActiveRotationalDiffusion(1, active_force, 0.1)
    hoomd.conftest.operation_pickling_check(rd_updater, sim)
