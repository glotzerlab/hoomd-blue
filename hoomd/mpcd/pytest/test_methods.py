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
        snap_.particles.N = 2
        snap_.particles.types = ["A"]
        snap_.particles.position[:] = [[4.95, 3.85, -4.95], [0.0, -3.8, 0.0]]
        snap_.particles.velocity[:] = [[1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]]
        snap_.particles.mass[:] = [1.0, 2.0]
    return snap_


@pytest.fixture
def integrator():
    bb = hoomd.mpcd.methods.BounceBack(
        filter=hoomd.filter.All(),
        geometry=hoomd.mpcd.geometry.ParallelPlates(separation=8.0))
    ig = hoomd.mpcd.Integrator(dt=0.1, methods=[bb])
    return ig


class TestBounceBack:

    def test_pickling(self, simulation_factory, snap, integrator):
        pickling_check(integrator.methods[0])

        sim = simulation_factory(snap)
        sim.operations.integrator = integrator
        sim.run(0)
        pickling_check(integrator.methods[0])

    def test_step_noslip(self, simulation_factory, snap, integrator):
        """Test step with no-slip boundary conditions."""
        sim = simulation_factory(snap)
        sim.operations.integrator = integrator

        # take one step
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.particles.position,
                [[-4.95, 3.95, 4.95], [-0.1, -3.9, -0.1]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity, [[1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]])

        # take another step where one particle will now hit the wall
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.particles.position,
                [[-4.95, 3.95, 4.95], [-0.2, -4.0, -0.2]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity,
                [[-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]])

        # take another step, reflecting the second particle
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.particles.position,
                [[4.95, 3.85, -4.95], [-0.1, -3.9, -0.1]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity, [[-1.0, -1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_step_slip(self, simulation_factory, snap, integrator):
        """Test step with slip boundary conditions."""
        integrator.methods[0].geometry.no_slip = False

        sim = simulation_factory(snap)
        sim.operations.integrator = integrator

        # take one step
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.particles.position,
                [[-4.95, 3.95, 4.95], [-0.1, -3.9, -0.1]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity, [[1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]])

        # take another step where one particle will now hit the wall
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.particles.position,
                [[-4.85, 3.95, 4.85], [-0.2, -4.0, -0.2]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity,
                [[1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])

        # take another step, reflecting perpendicular motion of second particle
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.particles.position,
                [[-4.75, 3.85, 4.75], [-0.3, -3.9, -0.3]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity, [[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]])

    def test_step_moving_wall(self, simulation_factory, snap, integrator):
        integrator.dt = 0.3
        integrator.methods[0].geometry.speed = 1.0

        if snap.communicator.rank == 0:
            snap.particles.velocity[1] = [-2.0, -1.0, -1.0]
        sim = simulation_factory(snap)
        sim.operations.integrator = integrator

        # run one step and check bounce back of particles
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.particles.position,
                [[-4.75, 3.85, -4.95], [-0.4, -3.9, -0.1]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity, [[1.0, -1.0, 1.0], [0.0, 1.0, 1.0]])

    def test_accel(self, simulation_factory, snap, integrator):
        force = hoomd.md.force.Constant(filter=hoomd.filter.All())
        force.constant_force["A"] = (2, 4, -2)
        integrator.forces.append(force)

        if snap.communicator.rank == 0:
            snap.particles.position[:] = [[0, 0, 0], [0, 0, 0]]
        sim = simulation_factory(snap)
        sim.operations.integrator = integrator

        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.particles.position,
                [[0.11, 0.12, -0.11], [-0.095, -0.09, -0.105]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity, [[1.2, 1.4, -1.2], [-0.9, -0.8, -1.1]])

    @pytest.mark.parametrize("H,expected_result", [(4.0, True), (3.8, False)])
    def test_check_particles(self, simulation_factory, snap, integrator, H,
                             expected_result):
        """Test box validation raises an error on run."""
        integrator.methods[0].geometry.separation = 2 * H

        sim = simulation_factory(snap)
        sim.operations.integrator = integrator

        sim.run(0)
        assert integrator.methods[0].check_particles() is expected_result

    def test_md_integrator(self, simulation_factory, snap):
        """Test we can also attach to a normal MD integrator."""
        bb = hoomd.mpcd.methods.BounceBack(
            filter=hoomd.filter.All(),
            geometry=hoomd.mpcd.geometry.ParallelPlates(separation=8.0))
        integrator = hoomd.md.Integrator(dt=0.1, methods=[bb])

        sim = simulation_factory(snap)
        sim.operations.integrator = integrator

        # verify one step works right
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.particles.position,
                [[-4.95, 3.95, 4.95], [-0.1, -3.9, -0.1]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity, [[1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]])
