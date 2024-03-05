# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.conftest import pickling_check
import numpy as np
import pytest


@pytest.fixture
def snap():
    snap_ = hoomd.Snapshot()
    if snap_.communicator.rank == 0:
        snap_.configuration.box = [10, 10, 10, 0, 0, 0]
        snap_.particles.N = 0
        snap_.particles.types = ["A"]
        snap_.mpcd.N = 1
        snap_.mpcd.types = ["A"]
    return snap_


@pytest.mark.parametrize(
    "cls, init_args",
    [
        (hoomd.mpcd.stream.Bulk, {}),
        (
            hoomd.mpcd.stream.BounceBack,
            {
                "geometry":
                    hoomd.mpcd.geometry.ParallelPlates(
                        H=4.0, V=0.0, no_slip=True),
            },
        ),
        (
            hoomd.mpcd.stream.BounceBack,
            {
                "geometry":
                    hoomd.mpcd.geometry.PlanarPore(H=4.0, L=3.0, no_slip=True)
            },
        ),
    ],
    ids=["Bulk", "ParallelPlates", "PlanarPore"],
)
class TestStreamingMethod:

    def test_create(self, simulation_factory, snap, cls, init_args):
        sim = simulation_factory(snap)
        sm = cls(period=5, **init_args)
        ig = hoomd.mpcd.Integrator(dt=0.02, streaming_method=sm)
        sim.operations.integrator = ig

        assert ig.streaming_method is sm
        assert sm.period == 5
        sim.run(0)
        assert ig.streaming_method is sm
        assert sm.period == 5

    def test_pickling(self, simulation_factory, snap, cls, init_args):
        sm = cls(period=5, **init_args)
        pickling_check(sm)

        sim = simulation_factory(snap)
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02,
                                                          streaming_method=sm)
        sim.run(0)
        pickling_check(sm)

    @pytest.mark.parametrize(
        "force",
        [
            None,
            hoomd.mpcd.force.BlockForce(
                force=2.0, half_separation=3.0, half_width=0.5),
            hoomd.mpcd.force.ConstantForce(force=(1, -2, 3)),
            hoomd.mpcd.force.SineForce(amplitude=2.0, wavenumber=1),
        ],
        ids=["NoForce", "BlockForce", "ConstantForce", "SineForce"],
    )
    def test_force_attach(self, simulation_factory, snap, cls, init_args,
                          force):
        """Test that force can be attached with various forces."""
        sm = cls(period=5, **init_args, solvent_force=force)
        assert sm.solvent_force is force
        pickling_check(sm)

        sim = simulation_factory(snap)
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02,
                                                          streaming_method=sm)
        sim.run(0)

        assert sm.solvent_force is force
        pickling_check(sm)

    def test_forced_step(self, simulation_factory, snap, cls, init_args):
        """Test a step with particle starting in the middle, constant force in +x and -z.

        This test should be skipped or adapted if geometries are added for which
        this point is / will be out of bounds, but is legal for all the ones we
        have now.
        """
        if snap.communicator.rank == 0:
            snap.mpcd.position[0] = [0, -1, 1]
            snap.mpcd.velocity[0] = [1, -2, 3]

        sim = simulation_factory(snap)
        sm = cls(period=1,
                 **init_args,
                 solvent_force=hoomd.mpcd.force.ConstantForce((1, 0, -1)))
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        # take 1 step and check updated velocity and position
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(snap.mpcd.velocity,
                                                 [[1.1, -2.0, 2.9]])
            np.testing.assert_array_almost_equal(snap.mpcd.position,
                                                 [[0.105, -1.2, 1.295]])


class TestBulk:

    def test_bulk_step(self, simulation_factory, snap):
        if snap.communicator.rank == 0:
            snap.mpcd.N = 2
            snap.mpcd.position[:] = [[1.0, 4.85, 3.0], [-3.0, -4.75, -1.0]]
            snap.mpcd.velocity[:] = [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]

        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.Bulk(period=1)
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        # take one step
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[1.1, 4.95, 3.1], [-3.1, -4.85, -1.1]])

        # take another step, wrapping the first particle through the boundary
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[1.2, -4.95, 3.2], [-3.2, -4.95, -1.2]])

        # take another step, wrapping the second particle through the boundary
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[1.3, -4.85, 3.3], [-3.3, 4.95, -1.3]])

        # change streaming method to use a different period, and change integrator step
        # running again should not move the particles since we haven't hit next period
        ig.dt = 0.05
        ig.streaming_method = hoomd.mpcd.stream.Bulk(period=4)
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[1.3, -4.85, 3.3], [-3.3, 4.95, -1.3]])

        # but running one more should move them
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[1.5, -4.65, 3.5], [-3.5, 4.75, -1.5]])

        # then 3 more should do nothing
        sim.run(3)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[1.5, -4.65, 3.5], [-3.5, 4.75, -1.5]])


class TestParallelPlates:

    def test_step_noslip(self, simulation_factory, snap):
        """Test step with no-slip boundary conditions."""
        if snap.communicator.rank == 0:
            snap.mpcd.N = 2
            snap.mpcd.position[:] = [[4.95, -4.95, 3.85], [0.0, 0.0, -3.8]]
            snap.mpcd.velocity[:] = [[1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]]
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1, geometry=hoomd.mpcd.geometry.ParallelPlates(H=4))
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        # take one step
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[-4.95, 4.95, 3.95], [-0.1, -0.1, -3.9]])
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity, [[1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]])

        # take another step where one particle will now hit the wall
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[-4.95, 4.95, 3.95], [-0.2, -0.2, -4.0]])
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity, [[-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]])

        # take another step, reflecting the second particle
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[4.95, -4.95, 3.85], [-0.1, -0.1, -3.9]])
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity, [[-1.0, 1.0, -1.0], [1.0, 1.0, 1.0]])

    def test_step_slip(self, simulation_factory, snap):
        """Test step with slip boundary conditions."""
        if snap.communicator.rank == 0:
            snap.mpcd.N = 2
            snap.mpcd.position[:] = [[4.95, -4.95, 3.85], [0.0, 0.0, -3.8]]
            snap.mpcd.velocity[:] = [[1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]]
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.ParallelPlates(H=4, no_slip=False))
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        # take one step
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[-4.95, 4.95, 3.95], [-0.1, -0.1, -3.9]])
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity, [[1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]])

        # take another step where one particle will now hit the wall
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[-4.85, 4.85, 3.95], [-0.2, -0.2, -4.0]])
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity, [[1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])

        # take another step, reflecting the perpendicular motion of second particle
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[-4.75, 4.75, 3.85], [-0.3, -0.3, -3.9]])
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity, [[1.0, -1.0, -1.0], [-1.0, -1.0, 1.0]])

    def test_step_moving_wall(self, simulation_factory, snap):
        """Test step with moving wall.

        The first particle is matched exactly to the wall speed, and so it will
        translate at same velocity along +x for 0.3 tau. It will bounce back in
        y and z to where it started. (vx stays the same, and vy and vz flip.)

        The second particle has y and z velocities flip again, and since it
        started closer, it moves relative to original position.
        """
        if snap.communicator.rank == 0:
            snap.mpcd.N = 2
            snap.mpcd.position[:] = [[4.95, -4.95, 3.85], [0.0, 0.0, -3.8]]
            snap.mpcd.velocity[:] = [[1.0, -1.0, 1.0], [-2.0, -1.0, -1.0]]
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.ParallelPlates(H=4, V=1, no_slip=True),
        )
        ig = hoomd.mpcd.Integrator(dt=0.3, streaming_method=sm)
        sim.operations.integrator = ig

        # run one step and check bounce back of particles
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[-4.75, -4.95, 3.85], [-0.4, -0.1, -3.9]])
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity, [[1.0, 1.0, -1.0], [0.0, 1.0, 1.0]])

    def test_validate_box(self, simulation_factory, snap):
        """Test box validation raises an error on run."""
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1, geometry=hoomd.mpcd.geometry.ParallelPlates(H=10))
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        with pytest.raises(RuntimeError):
            sim.run(1)

    def test_test_of_bounds(self, simulation_factory, snap):
        """Test box validation raises an error on run."""
        if snap.communicator.rank == 0:
            snap.mpcd.position[0] = [4.95, -4.95, 3.85]
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1, geometry=hoomd.mpcd.geometry.ParallelPlates(H=3.8))
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        with pytest.raises(RuntimeError):
            sim.run(1)


class TestPlanarPore:

    def _make_particles(self, snap):
        if snap.communicator.rank == 0:
            snap.mpcd.N = 8
            snap.mpcd.position[:] = [
                [-3.05, -4, -4.11],
                [3.05, 4, 4.11],
                [-3.05, -2, 4.11],
                [3.05, 2, -4.11],
                [0, 0, 3.95],
                [0, 0, -3.95],
                [3.03, 0, -3.98],
                [3.02, 0, -3.97],
            ]
            snap.mpcd.velocity[:] = [
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [1.0, 0.0, -1.0],
                [-1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, -1.0],
                [-1.0, 0.0, -1.0],
            ]
        return snap

    def test_step_noslip(self, simulation_factory, snap):
        snap = self._make_particles(snap)
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.PlanarPore(H=4, L=3, no_slip=True))
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        # take one step, and everything should collide and bounce back
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(snap.mpcd.position[0],
                                                 [-3.05, -4, -4.11])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[0],
                                                 [-1.0, 1.0, -1.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[1],
                                                 [3.05, 4, 4.11])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[1],
                                                 [1.0, -1.0, 1.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[2],
                                                 [-3.05, -2, 4.11])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[2],
                                                 [-1.0, 0.0, 1.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[3],
                                                 [3.05, 2, -4.11])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[3],
                                                 [1.0, 0.0, -1.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[4],
                                                 [0, 0, 3.95])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[4],
                                                 [0, 0, -1.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[5],
                                                 [0, 0, -3.95])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[5],
                                                 [0, 0, 1.0])
            # hits z = -4 after 0.02, then reverses. x is 3.01, so reverses to 3.09
            np.testing.assert_array_almost_equal(snap.mpcd.position[6],
                                                 [3.09, 0, -3.92])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[6],
                                                 [1, 0, 1])
            # hits x = 3 after 0.02, then reverses. z is -3.99, so reverses to -3.91
            np.testing.assert_array_almost_equal(snap.mpcd.position[7],
                                                 [3.08, 0, -3.91])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[7],
                                                 [1, 0, 1])

        # take another step where nothing hits now
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(snap.mpcd.position[0],
                                                 [-3.15, -3.9, -4.21])
            np.testing.assert_array_almost_equal(snap.mpcd.position[1],
                                                 [3.15, 3.9, 4.21])
            np.testing.assert_array_almost_equal(snap.mpcd.position[2],
                                                 [-3.15, -2, 4.21])
            np.testing.assert_array_almost_equal(snap.mpcd.position[3],
                                                 [3.15, 2, -4.21])
            np.testing.assert_array_almost_equal(snap.mpcd.position[4],
                                                 [0, 0, 3.85])
            np.testing.assert_array_almost_equal(snap.mpcd.position[5],
                                                 [0, 0, -3.85])
            np.testing.assert_array_almost_equal(snap.mpcd.position[6],
                                                 [3.19, 0, -3.82])
            np.testing.assert_array_almost_equal(snap.mpcd.position[7],
                                                 [3.18, 0, -3.81])

    def test_step_slip(self, simulation_factory, snap):
        snap = self._make_particles(snap)
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.PlanarPore(H=4, L=3, no_slip=False))
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        # take one step, and everything should collide and bounce back
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(snap.mpcd.position[0],
                                                 [-3.05, -4.1, -4.01])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[0],
                                                 [-1.0, -1.0, 1.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[1],
                                                 [3.05, 4.1, 4.01])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[1],
                                                 [1.0, 1.0, -1.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[2],
                                                 [-3.05, -2, 4.01])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[2],
                                                 [-1.0, 0.0, -1.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[3],
                                                 [3.05, 2, -4.01])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[3],
                                                 [1.0, 0.0, 1.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[4],
                                                 [0, 0, 3.95])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[4],
                                                 [0, 0, -1.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[5],
                                                 [0, 0, -3.95])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[5],
                                                 [0, 0, 1.0])
            # hits z = -4 after 0.02, then reverses. x is not touched because slip
            np.testing.assert_array_almost_equal(snap.mpcd.position[6],
                                                 [2.93, 0, -3.92])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[6],
                                                 [-1, 0, 1])
            # hits x = 3 after 0.02, then reverses. z is not touched because slip
            np.testing.assert_array_almost_equal(snap.mpcd.position[7],
                                                 [3.08, 0, -4.07])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[7],
                                                 [1, 0, -1])

        # take another step where nothing hits now
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(snap.mpcd.position[0],
                                                 [-3.15, -4.2, -3.91])
            np.testing.assert_array_almost_equal(snap.mpcd.position[1],
                                                 [3.15, 4.2, 3.91])
            np.testing.assert_array_almost_equal(snap.mpcd.position[2],
                                                 [-3.15, -2, 3.91])
            np.testing.assert_array_almost_equal(snap.mpcd.position[3],
                                                 [3.15, 2, -3.91])
            np.testing.assert_array_almost_equal(snap.mpcd.position[4],
                                                 [0, 0, 3.85])
            np.testing.assert_array_almost_equal(snap.mpcd.position[5],
                                                 [0, 0, -3.85])
            np.testing.assert_array_almost_equal(snap.mpcd.position[6],
                                                 [2.83, 0, -3.82])
            np.testing.assert_array_almost_equal(snap.mpcd.position[7],
                                                 [3.18, 0, -4.17])

    def test_validate_box(self, simulation_factory, snap):
        """Test box validation raises an error on run."""
        sim = simulation_factory(snap)
        ig = hoomd.mpcd.Integrator(dt=0.1)
        sim.operations.integrator = ig

        ig.streaming_method = hoomd.mpcd.stream.BounceBack(
            period=1, geometry=hoomd.mpcd.geometry.PlanarPore(H=10, L=2))
        with pytest.raises(RuntimeError):
            sim.run(1)

        ig.streaming_method = hoomd.mpcd.stream.BounceBack(
            period=1, geometry=hoomd.mpcd.geometry.PlanarPore(H=4, L=10))
        with pytest.raises(RuntimeError):
            sim.run(1)

    def test_test_of_bounds(self, simulation_factory, snap):
        """Test box validation raises an error on run."""
        snap = self._make_particles(snap)
        sim = simulation_factory(snap)
        ig = hoomd.mpcd.Integrator(dt=0.1)
        sim.operations.integrator = ig

        ig.streaming_method = hoomd.mpcd.stream.BounceBack(
            period=1, geometry=hoomd.mpcd.geometry.PlanarPore(H=3.8, L=3))
        with pytest.raises(RuntimeError):
            sim.run(1)

        ig.streaming_method = hoomd.mpcd.stream.BounceBack(
            period=1, geometry=hoomd.mpcd.geometry.PlanarPore(H=4, L=3.5))
        with pytest.raises(RuntimeError):
            sim.run(1)
