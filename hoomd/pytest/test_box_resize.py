from math import isclose
import numpy as np
import numpy.testing as npt
import pytest
import hoomd


@pytest.fixture(scope="module")
def fractional_coordinates(n=10):
    """
    TODO: Does `numpy_random_seed()` in conftest.py run for this function?
    Args:
        n: number of particles

    Returns: absolute fractional coordinates

    """
    return np.random.uniform(-0.5, 0.5, size=(n, 3))


@pytest.fixture(scope="module")
def sys1(fractional_coordinates):
    """Initial system box size and particle positions.
    Args:
        fractional_coordinates: Array of fractional coordinates

    Returns: hoomd box object and points for the initial system

    """
    hoomd_box = hoomd.Box.from_box([1., 2., 3., 1., 2., 3.])
    points = fractional_coordinates @ hoomd_box.matrix.T
    return hoomd_box, points


_box2 = [[0.5, 2., 3., 1., 2., 3.],  # Only change Lx
         [1., 2., 3., 0., 2., 3.],   # Only change xy
         [10, 2., 3., 0., 2., 3.],   # Change Lx and xy
         [10., 1., 6., 0., 5., 7.]]  # Change all


@pytest.fixture(scope='module', params=_box2)
def sys2(request, fractional_coordinates):
    hoomd_box = hoomd.Box.from_box(request.param)
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
    hoomd.variant.Power(0., 1., 0.1, _t_start, _t_start*2),
    hoomd.variant.Ramp(0., 1., _t_start, _t_start*2)
             ]


@pytest.fixture(scope='function', params=_variants)
def variant(request):
    return request.param


def test_user_specified_variant(device, simulation_factory, get_snapshot,
                                variant, sys1, sys2, scale_particles=True):
    sim = hoomd.Simulation(device)
    sim.create_state_from_snapshot(get_snapshot())

    trigger = hoomd.trigger.After(variant.t_start)

    box_resize = hoomd.update.BoxResize(
        box1=sys1[0], box2=sys2[0],
        variant=variant, trigger=trigger, scale_particles=scale_particles)
    sim.operations.updaters.append(box_resize)
    sim.run(variant.t_start*3 + 1)

    assert box_resize.get_box(0) == sys1[0]
    assert box_resize.get_box(variant.t_start*3) == sys2[0]
    assert sim.state.box == sys2[0]
    npt.assert_allclose(sys2[1], sim.state.snapshot.particles.position)


def test_variant_linear(device, simulation_factory, get_snapshot,
                         sys1, sys2, scale_particles=True):
    t_start = 2
    sim = hoomd.Simulation(device)
    sim.create_state_from_snapshot(get_snapshot())

    trigger = hoomd.trigger.After(t_start)

    box_resize = hoomd.update.BoxResize.linear_volume(
        box1=sys1[0], box2=sys2[0], t_start=t_start, t_size=t_start*2,
        trigger=trigger, scale_particles=scale_particles)
    sim.operations.updaters.append(box_resize)
    sim.run(t_start*3 + 1)

    assert box_resize.get_box(0) == sys1[0]
    assert box_resize.get_box(t_start*3) == sys2[0]
    assert sim.state.box == sys2[0]
    npt.assert_allclose(sys2[1], sim.state.snapshot.particles.position)
