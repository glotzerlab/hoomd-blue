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
        snap_.particles.N = 2
        snap_.particles.types = ["A"]
        snap_.particles.position[:] = [[4.95, -4.95, 3.85], [0.0, 0.0, -3.8]]
        snap_.particles.velocity[:] = [[1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]]
        snap_.particles.mass[:] = [1.0, 2.0]
    return snap_


@pytest.fixture
def integrator():
    bb = hoomd.mpcd.methods.BounceBack(
        filter=hoomd.filter.All(),
        geometry=hoomd.mpcd.geometry.ParallelPlates(H=4))
    ig = hoomd.mpcd.Integrator(dt=0.1, methods=[bb])
    return ig


class TestBounceBack:

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
                [[-4.95, 4.95, 3.95], [-0.1, -0.1, -3.9]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity, [[1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]])

        # take another step where one particle will now hit the wall
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.particles.position,
                [[-4.95, 4.95, 3.95], [-0.2, -0.2, -4.0]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity,
                [[-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]])

        # take another step, reflecting the second particle
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.particles.position,
                [[4.95, -4.95, 3.85], [-0.1, -0.1, -3.9]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity, [[-1.0, 1.0, -1.0], [1.0, 1.0, 1.0]])

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
                [[-4.95, 4.95, 3.95], [-0.1, -0.1, -3.9]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity, [[1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]])

        # take another step where one particle will now hit the wall
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.particles.position,
                [[-4.85, 4.85, 3.95], [-0.2, -0.2, -4.0]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity,
                [[1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])

        # take another step, reflecting the perpendicular motion of second particle
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            np.testing.assert_array_almost_equal(
                snap.particles.position,
                [[-4.75, 4.75, 3.85], [-0.3, -0.3, -3.9]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity, [[1.0, -1.0, -1.0], [-1.0, -1.0, 1.0]])

    def test_step_moving_wall(self, simulation_factory, snap, integrator):
        integrator.dt = 0.3
        integrator.methods[0].geometry.V = 1.0

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
                [[-4.75, -4.95, 3.85], [-0.4, -0.1, -3.9]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity, [[1.0, 1.0, -1.0], [0.0, 1.0, 1.0]])

    def test_accel(self, simulation_factory, snap, integrator):
        force = hoomd.md.force.Constant(filter=hoomd.filter.All())
        force.constant_force["A"] = (2, -2, 4)
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
                [[0.11, -0.11, 0.12], [-0.095, -0.105, -0.09]])
            np.testing.assert_array_almost_equal(
                snap.particles.velocity, [[1.2, -1.2, 1.4], [-0.9, -1.1, -0.8]])

    def test_validate_box(self, simulation_factory, snap, integrator):
        """Test box validation raises an error on run."""
        integrator.methods[0].geometry.H = 10

        sim = simulation_factory(snap)
        sim.operations.integrator = integrator

        with pytest.raises(RuntimeError):
            sim.run(1)

    def test_test_of_bounds(self, simulation_factory, snap, integrator):
        """Test box validation raises an error on run."""
        integrator.methods[0].geometry.H = 3.8

        sim = simulation_factory(snap)
        sim.operations.integrator = integrator

        with pytest.raises(RuntimeError):
            sim.run(1)
