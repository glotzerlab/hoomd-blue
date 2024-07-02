# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

import hoomd
from hoomd.conftest import pickling_check


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
                        separation=8.0, speed=0.0, no_slip=True),
            },
        ),
        (
            hoomd.mpcd.stream.BounceBack,
            {
                "geometry":
                    hoomd.mpcd.geometry.PlanarPore(
                        separation=8.0, length=6.0, no_slip=True)
            },
        ),
        (
            hoomd.mpcd.stream.BounceBack,
            {
                "geometry": hoomd.mpcd.geometry.Sphere(radius=4.0, no_slip=True)
            },
        ),
    ],
    ids=["Bulk", "ParallelPlates", "PlanarPore", "Sphere"],
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
            hoomd.mpcd.force.BlockForce(force=2.0, separation=3.0, width=0.5),
            hoomd.mpcd.force.ConstantForce(force=(1, -2, 3)),
            hoomd.mpcd.force.SineForce(amplitude=2.0, wavenumber=1),
        ],
        ids=["NoForce", "BlockForce", "ConstantForce", "SineForce"],
    )
    def test_force_attach(self, simulation_factory, snap, cls, init_args,
                          force):
        """Test that force can be attached with various forces."""
        sm = cls(period=5, **init_args, mpcd_particle_force=force)
        assert sm.mpcd_particle_force is force
        pickling_check(sm)

        sim = simulation_factory(snap)
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02,
                                                          streaming_method=sm)
        sim.run(0)

        assert sm.mpcd_particle_force is force
        pickling_check(sm)

    def test_forced_step(self, simulation_factory, snap, cls, init_args):
        """Test a forced step.

        The particle starts in the middle, and there is a constant force in +x
        and -z.

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
                 mpcd_particle_force=hoomd.mpcd.force.ConstantForce((1, 0, -1)))
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

        # change streaming method to use a different period, and change
        # integrator step running again should not move the particles since we
        # haven't hit next period
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
            snap.mpcd.position[:] = [[4.95, 3.85, -4.95], [0.0, -3.8, 0.0]]
            snap.mpcd.velocity[:] = [[1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]]
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.ParallelPlates(separation=8.0))
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        # take one step
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[-4.95, 3.95, 4.95], [-0.1, -3.9, -0.1]])
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity, [[1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]])

        # take another step where one particle will now hit the wall
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[-4.95, 3.95, 4.95], [-0.2, -4.0, -0.2]])
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity, [[-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]])

        # take another step, reflecting the second particle
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[4.95, 3.85, -4.95], [-0.1, -3.9, -0.1]])
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity, [[-1.0, -1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_step_slip(self, simulation_factory, snap):
        """Test step with slip boundary conditions."""
        if snap.communicator.rank == 0:
            snap.mpcd.N = 2
            snap.mpcd.position[:] = [[4.95, 3.85, -4.95], [0.0, -3.8, 0.0]]
            snap.mpcd.velocity[:] = [[1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]]
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.ParallelPlates(separation=8.0,
                                                        no_slip=False),
        )
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        # take one step
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[-4.95, 3.95, 4.95], [-0.1, -3.9, -0.1]])
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity, [[1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]])

        # take another step where one particle will now hit the wall
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[-4.85, 3.95, 4.85], [-0.2, -4.0, -0.2]])
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity, [[1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])

        # take another step, reflecting perpendicular motion of second particle
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[-4.75, 3.85, 4.75], [-0.3, -3.9, -0.3]])
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity, [[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]])

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
            snap.mpcd.position[:] = [[4.95, 3.85, -4.95], [0.0, -3.8, 0.0]]
            snap.mpcd.velocity[:] = [[1.0, 1.0, -1.0], [-2.0, -1.0, -1.0]]
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.ParallelPlates(separation=8.0,
                                                        speed=1,
                                                        no_slip=True),
        )
        ig = hoomd.mpcd.Integrator(dt=0.3, streaming_method=sm)
        sim.operations.integrator = ig

        # run one step and check bounce back of particles
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position, [[-4.75, 3.85, -4.95], [-0.4, -3.9, -0.1]])
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity, [[1.0, -1.0, 1.0], [0.0, 1.0, 1.0]])

    @pytest.mark.parametrize("H,expected_result", [(4.0, True), (3.8, False)])
    def test_check_mpcd_particles(self, simulation_factory, snap, H,
                                  expected_result):
        if snap.communicator.rank == 0:
            snap.mpcd.position[0] = [0, 3.85, 0]
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.ParallelPlates(separation=2 * H))
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        sim.run(0)
        assert sm.check_mpcd_particles() is expected_result


class TestPlanarPore:

    def _make_particles(self, snap):
        if snap.communicator.rank == 0:
            snap.mpcd.N = 8
            snap.mpcd.position[:] = [
                [-3.05, -4.11, -4],
                [3.05, 4.11, 4],
                [-3.05, 4.11, -2],
                [3.05, -4.11, 2],
                [0, 3.95, 0],
                [0, -3.95, 0],
                [3.03, -3.98, 0],
                [3.02, -3.97, 0],
            ]
            snap.mpcd.velocity[:] = [
                [1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [-1.0, -1.0, 0.0],
                [-1.0, -1.0, 0.0],
            ]
        return snap

    def test_step_noslip(self, simulation_factory, snap):
        snap = self._make_particles(snap)
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.PlanarPore(separation=8.0,
                                                    length=6.0,
                                                    no_slip=True),
        )
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        # take one step, and everything should collide and bounce back
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(snap.mpcd.position[0],
                                                 [-3.05, -4.11, -4])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[0],
                                                 [-1.0, -1.0, 1.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[1],
                                                 [3.05, 4.11, 4])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[1],
                                                 [1.0, 1.0, -1.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[2],
                                                 [-3.05, 4.11, -2])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[2],
                                                 [-1.0, 1.0, 0.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[3],
                                                 [3.05, -4.11, 2])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[3],
                                                 [1.0, -1.0, 0.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[4],
                                                 [0, 3.95, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[4],
                                                 [0, -1.0, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[5],
                                                 [0, -3.95, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[5],
                                                 [0, 1.0, 0])
            # hits y = -4 after 0.02, then reverses.
            # x is 3.01, so reverses to 3.09
            np.testing.assert_array_almost_equal(snap.mpcd.position[6],
                                                 [3.09, -3.92, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[6],
                                                 [1, 1, 0])
            # hits x = 3 after 0.02, then reverses.
            # y is -3.99, so reverses to -3.91
            np.testing.assert_array_almost_equal(snap.mpcd.position[7],
                                                 [3.08, -3.91, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[7],
                                                 [1, 1, 0])

        # take another step where nothing hits now
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(snap.mpcd.position[0],
                                                 [-3.15, -4.21, -3.9])
            np.testing.assert_array_almost_equal(snap.mpcd.position[1],
                                                 [3.15, 4.21, 3.9])
            np.testing.assert_array_almost_equal(snap.mpcd.position[2],
                                                 [-3.15, 4.21, -2])
            np.testing.assert_array_almost_equal(snap.mpcd.position[3],
                                                 [3.15, -4.21, 2])
            np.testing.assert_array_almost_equal(snap.mpcd.position[4],
                                                 [0, 3.85, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[5],
                                                 [0, -3.85, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[6],
                                                 [3.19, -3.82, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[7],
                                                 [3.18, -3.81, 0])

    def test_step_slip(self, simulation_factory, snap):
        snap = self._make_particles(snap)
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.PlanarPore(separation=8.0,
                                                    length=6.0,
                                                    no_slip=False),
        )
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        # take one step, and everything should collide and bounce back
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(snap.mpcd.position[0],
                                                 [-3.05, -4.01, -4.1])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[0],
                                                 [-1.0, 1.0, -1.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[1],
                                                 [3.05, 4.01, 4.1])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[1],
                                                 [1.0, -1.0, 1.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[2],
                                                 [-3.05, 4.01, -2])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[2],
                                                 [-1.0, -1.0, 0.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[3],
                                                 [3.05, -4.01, 2])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[3],
                                                 [1.0, 1.0, 0.0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[4],
                                                 [0, 3.95, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[4],
                                                 [0, -1.0, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[5],
                                                 [0, -3.95, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[5],
                                                 [0, 1.0, 0])
            # hits y = -4 after 0.02, then reverses.
            # x is not touched because slip
            np.testing.assert_array_almost_equal(snap.mpcd.position[6],
                                                 [2.93, -3.92, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[6],
                                                 [-1, 1, 0])
            # hits x = 3 after 0.02, then reverses.
            # y is not touched because slip
            np.testing.assert_array_almost_equal(snap.mpcd.position[7],
                                                 [3.08, -4.07, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[7],
                                                 [1, -1, 0])

        # take another step where nothing hits now
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(snap.mpcd.position[0],
                                                 [-3.15, -3.91, -4.2])
            np.testing.assert_array_almost_equal(snap.mpcd.position[1],
                                                 [3.15, 3.91, 4.2])
            np.testing.assert_array_almost_equal(snap.mpcd.position[2],
                                                 [-3.15, 3.91, -2])
            np.testing.assert_array_almost_equal(snap.mpcd.position[3],
                                                 [3.15, -3.91, 2])
            np.testing.assert_array_almost_equal(snap.mpcd.position[4],
                                                 [0, 3.85, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[5],
                                                 [0, -3.85, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[6],
                                                 [2.83, -3.82, 0])
            np.testing.assert_array_almost_equal(snap.mpcd.position[7],
                                                 [3.18, -4.17, 0])

    def test_check_mpcd_particles(self, simulation_factory, snap):
        """Test box validation raises an error on run."""
        snap = self._make_particles(snap)
        sim = simulation_factory(snap)
        ig = hoomd.mpcd.Integrator(dt=0.1)
        sim.operations.integrator = ig

        ig.streaming_method = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.PlanarPore(separation=8.0, length=6.0),
        )
        sim.run(0)
        assert ig.streaming_method.check_mpcd_particles()

        ig.streaming_method = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.PlanarPore(separation=7.6, length=6.0),
        )
        sim.run(0)
        assert not ig.streaming_method.check_mpcd_particles()

        ig.streaming_method = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.PlanarPore(separation=8.0, length=7.0),
        )
        assert not ig.streaming_method.check_mpcd_particles()


class TestSphere:

    def _make_particles(self, snap):
        if snap.communicator.rank == 0:
            snap.mpcd.N = 4

            # particle 1: Hits the wall in the second streaming step, gets
            # reflected accordingly
            snap.mpcd.position[0] = [2.85, 0.895, np.sqrt(6) + 0.075]
            snap.mpcd.velocity[0] = [1., 0.7, -0.5]

            # particle 2: Always inside the sphere, so no reflection by the BC
            snap.mpcd.position[1] = [0., 0., 0.]
            snap.mpcd.velocity[1] = [-1., -1., -1.]

            # particle 3: Hits the wall normally and gets reflected back.
            snap.mpcd.position[2] = 0.965 * np.array([-1., -2., np.sqrt(11)])
            snap.mpcd.velocity[2] = 0.25 * np.array([-1., -2., np.sqrt(11)])

            # particle 4: Lands almost exactly on the sphere surface and needs
            # to be backtracked one complete step
            snap.mpcd.position[3] = [1.92, -1.96, -np.sqrt(8) + 0.05]
            snap.mpcd.velocity[3] = [0.8, -0.4, -0.5]
        return snap

    def test_step_noslip(self, simulation_factory, snap):
        """Test step with no-slip boundary conditions."""
        snap = self._make_particles(snap)
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1, geometry=hoomd.mpcd.geometry.Sphere(radius=4.0))
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position[0],
                [2.95, 0.965, np.sqrt(6) + 0.025])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[0],
                                                 [1., 0.7, -0.5])
            np.testing.assert_array_almost_equal(snap.mpcd.position[1],
                                                 [-0.1, -0.1, -0.1])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[1],
                                                 [-1., -1., -1.])
            np.testing.assert_array_almost_equal(
                snap.mpcd.position[2],
                0.99 * np.array([-1., -2., np.sqrt(11)]))
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity[2],
                0.25 * np.array([-1., -2., np.sqrt(11)]))
            np.testing.assert_array_almost_equal(snap.mpcd.position[3],
                                                 [2., -2., -np.sqrt(8)])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[3],
                                                 [0.8, -0.4, -0.5])

        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position[0],
                [2.95, 0.965, np.sqrt(6) + 0.025])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[0],
                                                 [-1., -0.7, 0.5])
            np.testing.assert_array_almost_equal(snap.mpcd.position[1],
                                                 [-0.2, -0.2, -0.2])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[1],
                                                 [-1., -1., -1.])
            np.testing.assert_array_almost_equal(
                snap.mpcd.position[2],
                0.985 * np.array([-1., -2., np.sqrt(11)]))
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity[2],
                -0.25 * np.array([-1., -2., np.sqrt(11)]))
            np.testing.assert_array_almost_equal(
                snap.mpcd.position[3], [1.92, -1.96, -np.sqrt(8) + 0.05])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[3],
                                                 [-0.8, 0.4, 0.5])

        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position[0],
                [2.85, 0.895, np.sqrt(6) + 0.075])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[0],
                                                 [-1., -0.7, 0.5])
            np.testing.assert_array_almost_equal(snap.mpcd.position[1],
                                                 [-0.3, -0.3, -0.3])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[1],
                                                 [-1., -1., -1.])
            np.testing.assert_array_almost_equal(
                snap.mpcd.position[2],
                0.96 * np.array([-1., -2., np.sqrt(11)]))
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity[2],
                -0.25 * np.array([-1., -2., np.sqrt(11)]))
            np.testing.assert_array_almost_equal(
                snap.mpcd.position[3], [1.84, -1.92, -np.sqrt(8) + 0.1])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[3],
                                                 [-0.8, 0.4, 0.5])

    def test_step_slip(self, simulation_factory, snap):
        """Test step with slip boundary conditions."""
        snap = self._make_particles(snap)
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.Sphere(radius=4.0, no_slip=False),
        )
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.mpcd.position[0],
                [2.95, 0.965, np.sqrt(6) + 0.025])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[0],
                                                 [1., 0.7, -0.5])
            np.testing.assert_array_almost_equal(snap.mpcd.position[1],
                                                 [-0.1, -0.1, -0.1])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[1],
                                                 [-1., -1., -1.])
            np.testing.assert_array_almost_equal(
                snap.mpcd.position[2],
                0.99 * np.array([-1., -2., np.sqrt(11)]))
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity[2],
                0.25 * np.array([-1., -2., np.sqrt(11)]))
            np.testing.assert_array_almost_equal(snap.mpcd.position[3],
                                                 [2., -2., -np.sqrt(8)])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[3],
                                                 [0.8, -0.4, -0.5])

        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            # point of contact
            r1_before = np.array([3., 1., np.sqrt(6)])
            # velocity before reflection
            v1_before = np.array([1., 0.7, -0.5])
            # velocity after reflection
            v1_after = v1_before - 1 / 8. * np.dot(v1_before,
                                                   r1_before) * r1_before
            # position after reflection
            r1_after = r1_before + v1_after * 0.05
            np.testing.assert_array_almost_equal(snap.mpcd.position[0],
                                                 r1_after)
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[0],
                                                 v1_after)
            np.testing.assert_array_almost_equal(snap.mpcd.position[1],
                                                 [-0.2, -0.2, -0.2])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[1],
                                                 [-1., -1., -1.])
            np.testing.assert_array_almost_equal(
                snap.mpcd.position[2],
                0.985 * np.array([-1., -2., np.sqrt(11)]))
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity[2],
                -0.25 * np.array([-1., -2., np.sqrt(11)]))
            r2_before = np.array([2., -2., -np.sqrt(8)])
            v2_before = np.array([0.8, -0.4, -0.5])
            v2_after = v2_before - 1. / 8. * np.dot(v2_before,
                                                    r2_before) * r2_before
            r2_after = r2_before + v2_after * 0.1
            np.testing.assert_array_almost_equal(snap.mpcd.position[3],
                                                 r2_after)
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[3],
                                                 v2_after)

        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            # one step streaming
            r1_after += v1_after * 0.1
            np.testing.assert_array_almost_equal(snap.mpcd.position[0],
                                                 r1_after)
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[0],
                                                 v1_after)
            np.testing.assert_array_almost_equal(snap.mpcd.position[1],
                                                 [-0.3, -0.3, -0.3])
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[1],
                                                 [-1., -1., -1.])
            np.testing.assert_array_almost_equal(
                snap.mpcd.position[2],
                0.96 * np.array([-1., -2., np.sqrt(11)]))
            np.testing.assert_array_almost_equal(
                snap.mpcd.velocity[2],
                -0.25 * np.array([-1., -2., np.sqrt(11)]))
            r2_after += v2_after * 0.1
            np.testing.assert_array_almost_equal(snap.mpcd.position[3],
                                                 r2_after)
            np.testing.assert_array_almost_equal(snap.mpcd.velocity[3],
                                                 v2_after)

    @pytest.mark.parametrize("R,expected_result", [(4.0, True), (3.8, False)])
    def test_check_mpcd_particles(self, simulation_factory, snap, R,
                                  expected_result):
        if snap.communicator.rank == 0:
            snap.mpcd.position[0] = [0, 3.85, 0]
        sim = simulation_factory(snap)
        sm = hoomd.mpcd.stream.BounceBack(
            period=1, geometry=hoomd.mpcd.geometry.Sphere(radius=R))
        ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=sm)
        sim.operations.integrator = ig

        sim.run(0)
        assert sm.check_mpcd_particles() is expected_result
