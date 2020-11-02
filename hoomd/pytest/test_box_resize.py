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


_box = ([[1., 2., 3., 1., 2., 3.],    # Initial box, 3D
         [10., 1., 6., 0., 5., 7.]],  # Final box, 3D
        [[1., 2., 0., 1., 0., 0.],    # Initial box, 2D
         [10., 1., 0., 0., 0., 0.]])  # Final box, 2D


@pytest.fixture(scope="function", params=_box)
def sys(request, fractional_coordinates):
    """Initial, halfway, and final system box size and particle positions.
    Halfway system box size and points are not applicable for linear volume resizing
    Args:
        fractional_coordinates: Array of fractional coordinates

    Returns: hoomd box object and points for the initial, halfway, and final system

    """
    box_start = request.param[0]
    box_end = request.param[1]

    return (make_system(fractional_coordinates, box_start),
            make_system(fractional_coordinates, box_end))


def make_system(fractional_coordinates, box):
    hoomd_box = hoomd.Box.from_box(box)
    points = fractional_coordinates @ hoomd_box.matrix.T
    return (hoomd_box, points)


@pytest.fixture(scope='function')
def get_snapshot(sys, device):
    def make_shapshot():
        box1, points1 = sys[0]
        s = hoomd.Snapshot()
        s.configuration.box = box1
        s.particles.N = points1.shape[0]
        s.particles.typeid[:] = [0] * points1.shape[0]
        s.particles.types = ['A']
        s.particles.position[:] = points1
        return s

    return make_shapshot


_t_start = 2
_t_ramp = 4
_t_mid = _t_start + _t_ramp//2
_power = 2


class TestBoxResize:

    def make_halfway(self, box_start, box_end):
        box_halfway = box_start + (box_end - box_start) * 0.5 ** _power
        return make_system(fractional_coordinates, box_halfway)

    @pytest.fixture(scope='function')
    def box_resize(self, sys):
        sys1, sys2 = sys
        variant = hoomd.variant.Power(0., 1., _power, _t_start, _t_ramp)
        trigger = hoomd.trigger.After(variant.t_start)
        return hoomd.update.BoxResize(
            box1=sys1[0], box2=sys2[0],
            variant=variant, trigger=trigger)

    def test_get_box(self, device, simulation_factory, get_snapshot,
                     sys, box_resize):
        sys1, sys2 = sys
        sys_halfway = self.make_halfway(sys1[0], sys2[0])

        sim = hoomd.Simulation(device)
        sim.create_state_from_snapshot(get_snapshot())

        sim.operations.updaters.append(box_resize)
        sim.run(_t_start + _t_ramp)

        assert box_resize.get_box(0) == sys1[0]
        assert box_resize.get_box(_t_mid) == self.sys_halfway[0]
        assert box_resize.get_box(_t_start + _t_ramp) == sys2[0]


    # def test_box_dimensions(self, device, simulation_factory, get_snapshot,
    #                         sys, sys_halfway, box_resize):
    #     _, sys2 = sys
    #
    #     sim = hoomd.Simulation(device)
    #     sim.create_state_from_snapshot(get_snapshot())
    #
    #     # Run up to halfway point
    #     sim.run(_t_start*2 + 1)
    #     assert sim.state.box == sys_halfway[0]
    #
    #     # Finish run
    #     sim.run(_t_start)
    #     assert sim.state.box == sys2[0]

    # def test_particle_scale(self, device, simulation_factory, get_snapshot,
    #                         sys, sys_halfway, box_resize):
    #     _ , sys2 = sys
    #     sim = hoomd.Simulation(device)
    #     sim.create_state_from_snapshot(get_snapshot())
    #     sim.operations.updaters.append(box_resize)
    #
    #     # # Run up to halfway point
    #     # sim.run(_t_start*2 + 1)
    #     # npt.assert_allclose(sim.state.snapshot.particles.position, sys_halfway[1])
    #
    #     # Finish run
    #     sim.run(_t_start + _t_ramp)
    #     npt.assert_allclose(sim.state.snapshot.particles.position, sys2[1])


# class TestLinearVolume:
#
#     @pytest.fixture(scope='function', params=_box)
#     def sys_halfway(self, request, fractional_coordinates):
#         box_start = np.array(request.param[0])
#         box_end = np.array(request.param[1])
#         box_halfway = box_start + (box_end - box_start) * 0.5 ** _power
#         return make_system(fractional_coordinates, box_halfway)
#
#     @pytest.fixture(scope='function')
#     def box_resize(self, sys):
#         sys1, sys2 = sys
#         trigger = hoomd.trigger.After(_t_start)
#         variant = hoomd.variant.Power(0., 1., 1., _t_start, _t_ramp)
#         return hoomd.update.BoxResize(
#             box1=sys1[0], box2=sys2[0],
#             variant=variant, trigger=trigger)
#
#     def test_get_box(self, device, simulation_factory, box_resize,
#                      get_snapshot, sys, sys_halfway):
#         sys1, sys2 = sys
#
#         sim = hoomd.Simulation(device)
#         sim.create_state_from_snapshot(get_snapshot())
#
#         sim.operations.updaters.append(box_resize)
#         sim.run(_t_start * 3 + 1)
#
#         assert box_resize.get_box(0) == sys1[0]
#         # assert box_resize.get_box(variant.t_start * 2) == sys_halfway[0]
#         # assert box_resize.get_box(variant.t_start * 3 + 1) == sys2[0]
#
#     def test_linear_volume(self, device, simulation_factory, get_snapshot,
#                            box_resize, sys, sys_halfway,
#                            scale_particles=True):
#         sys1, sys2 = sys
#
#         sim = hoomd.Simulation(device)
#         sim.create_state_from_snapshot(get_snapshot())
#
#         sim.operations.updaters.append(box_resize)
#
#         # Run up to halfway point
#         sim.run(_t_start*2 + 1)
#         npt.assert_allclose(sim.state.box.volume, sys_halfway[0].volume, rtol=5e-3)
#
#         # Finish run
#         sim.run(_t_start + 1)
#
#         assert sim.state.box == sys2[0]
#         if scale_particles:
#             npt.assert_allclose(sys2[1], sim.state.snapshot.particles.position)
#         else:
#             npt.assert_allclose(sys1[1], sim.state.snapshot.particles.position)
