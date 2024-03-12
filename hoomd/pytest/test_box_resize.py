# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import numpy.testing as npt
import pytest
import hoomd
from hoomd.conftest import operation_pickling_check
from hoomd.error import MutabilityError

_n_points = 10


@pytest.fixture(scope="function")
def fractional_coordinates(n=_n_points):
    """Return fractional coordinates for testing.

    Args:
        n: number of particles

    Returns: absolute fractional coordinates
    """
    return np.random.uniform(-0.5, 0.5, size=(n, 3))


_box = (
    [
        [1., 2., 1., 1., 0., 3.],  # Initial box, 3D
        [10., 12., 20., 0., 1., 2.]
    ],  # Final box, 3D
    [
        [1., 2., 0., 1., 0., 0.],  # Initial box, 2D
        [10., 12., 0., 0., 0., 0.]
    ])  # Final box, 2D


@pytest.fixture(scope="function", params=_box, ids=['sys_3d', 'sys_2d'])
def sys(request, fractional_coordinates):
    """System box sizes and particle positions.

    Args:
        request: Fixture request information.
        fractional_coordinates: Array of fractional coordinates

    Returns: HOOMD box object and points for the initial and final system.
             Function to generate system at halfway point of the simulation.
    """
    box_start = request.param[0]
    box_end = request.param[1]

    return (make_system(fractional_coordinates,
                        box_start), lambda power: make_sys_halfway(
                            fractional_coordinates, box_start, box_end, power),
            make_system(fractional_coordinates, box_end))


def make_system(fractional_coordinates, box):
    hoomd_box = hoomd.Box.from_box(box)
    points = fractional_coordinates @ hoomd_box.to_matrix().T
    return (hoomd_box, points)


_t_start = 1
_t_ramp = 4
_t_mid = _t_start + _t_ramp // 2


@pytest.fixture(scope='function')
def trigger():
    return hoomd.trigger.After(_t_mid - 1)


def make_sys_halfway(fractional_coordinates, box_start, box_end, power):
    box_start = np.array(box_start)
    box_end = np.array(box_end)

    intermediate_t = (_t_mid - _t_start) / _t_ramp  # set to halfway, 0.5
    box_mid = hoomd.Box.from_box(box_start + (box_end - box_start)
                                 * intermediate_t**power)
    return make_system(fractional_coordinates, box_mid)


@pytest.fixture(scope='function')
def get_snapshot(sys, device):

    def make_shapshot():
        box1, points1 = sys[0]
        s = hoomd.snapshot.Snapshot(device.communicator)
        if s.communicator.rank == 0:
            s.configuration.box = box1
            s.particles.N = points1.shape[0]
            s.particles.typeid[:] = [0] * points1.shape[0]
            s.particles.types = ['A']
            s.particles.position[:] = points1
        return s

    return make_shapshot


_power = 2


@pytest.fixture(scope='function')
def variant():
    return hoomd.variant.Power(0., 1., _power, _t_start, _t_ramp)


@pytest.fixture(scope='function')
def box_resize(sys, trigger, variant):
    sys1, _, sys2 = sys
    b = hoomd.update.BoxResize(box1=sys1[0],
                               box2=sys2[0],
                               variant=variant,
                               trigger=trigger)
    return b


def assert_positions(sim, reference_points, filter=None):
    with sim.state.cpu_local_snapshot as data:
        if filter is not None:
            filter_tags = np.copy(filter(sim.state)).astype(int)
            is_particle_local = np.isin(data.particles.tag,
                                        filter_tags,
                                        assume_unique=True)
            reference_point = reference_points[
                data.particles.tag[is_particle_local]]
            pos = data.particles.position[is_particle_local]
        else:
            pos = data.particles.position[data.particles.rtag[
                data.particles.tag]]
            reference_point = reference_points[data.particles.tag]
        npt.assert_allclose(pos, reference_point)


def test_trigger(box_resize, trigger):
    assert trigger.timestep == box_resize.trigger.timestep
    for timestep in range(_t_start + _t_ramp):
        assert trigger.compute(timestep) == box_resize.trigger.compute(timestep)


def test_variant(box_resize, variant):
    for timestep in range(_t_start + _t_ramp):
        assert variant(timestep) == box_resize.variant(timestep)


def test_get_box(simulation_factory, get_snapshot, sys, box_resize):
    sys1, make_sys_halfway, sys2 = sys
    sys_halfway = make_sys_halfway(_power)

    sim = simulation_factory(get_snapshot())

    sim.operations.updaters.append(box_resize)
    sim.run(0)
    assert box_resize.get_box(0) == sys1[0]
    assert box_resize.get_box(_t_mid) == sys_halfway[0]
    assert box_resize.get_box(_t_start + _t_ramp) == sys2[0]


def test_update(simulation_factory, get_snapshot, sys):
    sys1, _, sys2 = sys
    sim = simulation_factory(get_snapshot())
    hoomd.update.BoxResize.update(sim.state, sys2[0])

    assert sim.state.box == sys2[0]
    assert_positions(sim, sys2[1])


_filter = ([[hoomd.filter.All(), hoomd.filter.Null()],
            [hoomd.filter.Null(), hoomd.filter.All()],
            [
                hoomd.filter.Tags([0, 5]),
                hoomd.filter.SetDifference(hoomd.filter.Tags([0]),
                                           hoomd.filter.All())
            ]])


@pytest.fixture(scope="function", params=_filter, ids=["All", "None", "Tags"])
def filters(request):
    return request.param


def test_position_scale(device, get_snapshot, sys, variant, trigger, filters,
                        simulation_factory):
    filter_scale, filter_noscale = filters
    sys1, make_sys_halfway, sys2 = sys
    sys_halfway = make_sys_halfway(_power)

    box_resize = hoomd.update.BoxResize(box1=sys1[0],
                                        box2=sys2[0],
                                        variant=variant,
                                        trigger=trigger,
                                        filter=filter_scale)

    sim = simulation_factory(get_snapshot())
    sim.operations.updaters.append(box_resize)

    # Run up to halfway point
    sim.run(_t_mid + 1)
    assert sim.state.box == sys_halfway[0]
    assert_positions(sim, sys1[1], filter_noscale)
    assert_positions(sim, sys_halfway[1], filter_scale)

    # Finish run
    sim.run(_t_mid)
    assert sim.state.box == sys2[0]
    assert_positions(sim, sys1[1], filter_noscale)
    assert_positions(sim, sys2[1], filter_scale)


def test_get_filter(device, get_snapshot, sys, variant, trigger, filters):
    filter_scale, _ = filters
    sys1, _, sys2 = sys
    box_resize = hoomd.update.BoxResize(box1=sys1[0],
                                        box2=sys2[0],
                                        variant=variant,
                                        trigger=trigger,
                                        filter=filter_scale)

    assert box_resize.filter == filter_scale


def test_update_filters(device, get_snapshot, sys, filters):
    sys1, _, sys2 = sys
    filter_scale, _ = filters
    sim = hoomd.Simulation(device)
    sim.create_state_from_snapshot(get_snapshot())
    hoomd.update.BoxResize.update(sim.state, sys2[0], filter=filter_scale)


def test_pickling(simulation_factory, two_particle_snapshot_factory,
                  box_resize):
    sim = simulation_factory(two_particle_snapshot_factory())
    operation_pickling_check(box_resize, sim)


def test_mutability_error(simulation_factory, two_particle_snapshot_factory):
    filt = hoomd.filter.All()
    sim = simulation_factory(two_particle_snapshot_factory())
    trig = hoomd.trigger.Periodic(1)
    box1 = hoomd.Box.cube(L=10)
    box2 = hoomd.Box.cube(L=12)
    var = hoomd.variant.Ramp(10, 12, 0, 100)
    box_op = hoomd.update.BoxResize(box1=box1,
                                    box2=box2,
                                    variant=var,
                                    trigger=trig)
    sim.operations.add(box_op)
    assert len(sim.operations.updaters) == 1
    sim.run(0)

    with pytest.raises(MutabilityError):
        box_op.filter = filt


def test_new_api(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    inverse_volume_ramp = hoomd.variant.box.InverseVolumeRamp(
        initial_box=hoomd.Box.cube(6),
        final_volume=100,
        t_start=1_000,
        t_ramp=21_000)

    box_resize = hoomd.update.BoxResize(trigger=hoomd.trigger.Periodic(1),
                                        filter=hoomd.filter.All(),
                                        box=inverse_volume_ramp)
    sim.operations.updaters.append(box_resize)

    sim.run(0)

    assert box_resize.box is inverse_volume_ramp


def test_invalid_api(variant, trigger):
    box_variant = hoomd.variant.box.InverseVolumeRamp(
        initial_box=hoomd.Box.cube(6),
        final_volume=100,
        t_start=1_000,
        t_ramp=21_000)

    box1 = hoomd.Box.cube(10)
    box2 = hoomd.Box.cube(20)

    with pytest.raises(ValueError):
        hoomd.update.BoxResize(trigger=trigger)
    with pytest.raises(ValueError):
        hoomd.update.BoxResize(trigger=trigger, box1=box1)
    with pytest.raises(ValueError):
        hoomd.update.BoxResize(trigger=trigger, box1=box1, box2=box2)
    with pytest.raises(ValueError):
        hoomd.update.BoxResize(trigger=trigger, box2=box2, variant=variant)
    with pytest.raises(ValueError):
        hoomd.update.BoxResize(trigger=trigger,
                               box1=box1,
                               box2=box2,
                               variant=variant,
                               box=box_variant)
    with pytest.raises(ValueError):
        hoomd.update.BoxResize(trigger=trigger, box1=box1, box=box_variant)
    with pytest.raises(ValueError):
        hoomd.update.BoxResize(trigger=trigger, box2=box2, box=box_variant)
    with pytest.raises(ValueError):
        hoomd.update.BoxResize(trigger=trigger,
                               variant=variant,
                               box=box_variant)

    box_resize = hoomd.update.BoxResize(trigger=trigger, box=box_variant)
    with pytest.raises(RuntimeError):
        box_resize.box1
    with pytest.raises(RuntimeError):
        box_resize.box2
    with pytest.raises(RuntimeError):
        box_resize.variant
    with pytest.raises(RuntimeError):
        box_resize.box1 = box1
    with pytest.raises(RuntimeError):
        box_resize.box2 = box2
    with pytest.raises(RuntimeError):
        box_resize.variant = variant
