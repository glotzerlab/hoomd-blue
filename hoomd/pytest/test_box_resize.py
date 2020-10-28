from math import isclose
import numpy as np
import numpy.testing as npt
import pytest
import hoomd


@pytest.fixture(scope="function")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="function")
def fractional_coordinates(n=10):
    """
    TODO: Does `numpy_random_seed()` in conftest.py run for this function?
    Args:
        n: number of particles

    Returns: absolute fractional coordinates

    """
    return np.random.uniform(-0.5, 0.5, size=(n, 3))


BOX1 = np.array([1., 2., 3., 1., 2., 3.])


@pytest.fixture(scope="function")
def sys1(fractional_coordinates):
    """Initial system box size and particle positions.
    Args:
        fractional_coordinates: Array of fractional coordinates

    Returns: hoomd box object and points for the initial system

    """
    hoomd_box = hoomd.Box.from_box(BOX1)
    points = fractional_coordinates @ hoomd_box.matrix.T
    return hoomd_box, points


BOX2 = np.array([10., 1., 6., 0., 5., 7.])


@pytest.fixture(scope='function')
def sys2(fractional_coordinates):
    """Final system box size and particle positions.
    Args:
        fractional_coordinates: Array of fractional coordinates

    Returns: hoomd box object and points for the final system

    """
    hoomd_box = hoomd.Box.from_box(BOX2)
    points = fractional_coordinates @ hoomd_box.matrix.T
    return hoomd_box, points


# BOX_HALF = BOX1 + (BOX2 - BOX1)/2
# BOX_HALF = BOX1 + (BOX2 - BOX1)/2
POWER = 0.2
# POWER = 1/(19**(1/3))


@pytest.fixture(scope='function')
def sys_halfway(fractional_coordinates):
    """Intermediate system box size and particle positions.
    Args:
        fractional_coordinates: Array of fractional coordinates

    Returns: hoomd box object and points for the final system

    """
    box_half = BOX1 + (BOX2 - BOX1) * 0.5 ** POWER
    hoomd_box = hoomd.Box.from_box(box_half)
    points = fractional_coordinates @ hoomd_box.matrix.T
    return hoomd_box, points


@pytest.fixture(scope='function')
def get_snapshot(sys1, device):
    def make_shapshot():
        box1, points1 = sys1
        s = hoomd.Snapshot()
        s.configuration.box = box1
        s.particles.N = points1.shape[0]
        s.particles.typeid[:] = [0] * points1.shape[0]
        s.particles.types = ['A']
        s.particles.position[:] = points1
        return s
    return make_shapshot


_t_start = 2
_variants = [
    hoomd.variant.Power(0., 1., POWER, _t_start, _t_start * 2)
]


@pytest.fixture(scope='function', params=_variants)
def variant(request):
    return request.param


def test_get_box(device, simulation_factory, get_snapshot,
                 variant, sys1, sys2, sys_halfway):
    sim = hoomd.Simulation(device)
    sim.create_state_from_snapshot(get_snapshot())

    trigger = hoomd.trigger.After(variant.t_start)

    box_resize = hoomd.update.BoxResize(
        box1=sys1[0], box2=sys2[0],
        variant=variant, trigger=trigger)
    sim.operations.updaters.append(box_resize)
    sim.run(variant.t_start*3 + 1)

    assert box_resize.get_box(0) == sys1[0]
    assert box_resize.get_box(variant.t_start*2) == sys_halfway[0]
    assert box_resize.get_box(variant.t_start*3 + 1) == sys2[0]


def test_user_specified_variant(device, simulation_factory, get_snapshot,
                                variant, sys1, sys2, sys_halfway,
                                scale_particles=True):
    sim = hoomd.Simulation(device)
    sim.create_state_from_snapshot(get_snapshot())

    trigger = hoomd.trigger.After(variant.t_start)

    box_resize = hoomd.update.BoxResize(
        box1=sys1[0], box2=sys2[0],
        variant=variant, trigger=trigger, scale_particles=scale_particles)
    sim.operations.updaters.append(box_resize)

    # Run up to halfway point
    sim.run(variant.t_start*2 + 1)

    assert sim.state.box == sys_halfway[0]
    if scale_particles:
        npt.assert_allclose(sys_halfway[1], sim.state.snapshot.particles.position)
    else:
        npt.assert_allclose(sys1[1], sim.state.snapshot.particles.position)

    # Finish run
    sim.run(variant.t_start)

    assert sim.state.box == sys2[0]
    if scale_particles:
        npt.assert_allclose(sys2[1], sim.state.snapshot.particles.position)
    else:
        npt.assert_allclose(sys1[1], sim.state.snapshot.particles.position)


def test_linear_volume(device, simulation_factory, get_snapshot,
                       variant, sys1, sys2,
                       scale_particles=True):
    sim = hoomd.Simulation(device)
    sim.create_state_from_snapshot(get_snapshot())

    trigger = hoomd.trigger.After(variant.t_start)

    box_resize = hoomd.update.BoxResize.linear_volume(
        box1=sys1[0], box2=sys2[0], t_start=variant.t_start, t_size=variant.t_start*2 + 1,
        trigger=trigger, scale_particles=scale_particles)
    sim.operations.updaters.append(box_resize)

    # Run up to halfway point
    sim.run(variant.t_start*2 + 1)
    halfway_volume = sys1[0].volume + (sys2[0].volume - sys1[0].volume)*0.5**(1/3)
    npt.assert_allclose(sim.state.box.volume, halfway_volume, rtol=5e-3)

    # Finish run
    sim.run(variant.t_start + 1)

    assert sim.state.box == sys2[0]
    if scale_particles:
        npt.assert_allclose(sys2[1], sim.state.snapshot.particles.position)
    else:
        npt.assert_allclose(sys1[1], sim.state.snapshot.particles.position)
