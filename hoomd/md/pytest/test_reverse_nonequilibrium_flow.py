# Copyright (c) 2009-2026 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest
import hoomd
import numpy as np

from hoomd.conftest import pickling_check
from hoomd.logging import LoggerCategories
from hoomd.conftest import logging_check


@pytest.fixture
def snap():
    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [20, 20, 20, 0, 0, 0]
        snap.particles.N = 0
        snap.particles.types = ["A"]
    return snap


@pytest.fixture
def flow():
    flow = hoomd.md.update.ReverseNonequilibriumShearFlow(
        trigger=1, filter=hoomd.filter.All(), num_swaps=1, slab_width=1.0
    )
    return flow


decompositions = [None]
if hoomd.communicator.Communicator().num_ranks == 2:
    decompositions += [(1, 2, 1), (2, 1, 1)]


class TestReverseNonequilibriumShearFlow:
    def test_create(self, simulation_factory, snap, flow):
        # before attachment
        flow.num_swaps = 5
        flow.target_momentum = 2.5
        trigger = hoomd.trigger.Periodic(1)
        flow.trigger = trigger

        assert flow.trigger is trigger
        assert flow.slab_width == 1.0

        with pytest.raises(hoomd.error.DataAccessError):
            assert flow.summed_exchanged_momentum == 0.0

        sim = simulation_factory(snap)
        sim.operations.integrator = hoomd.md.Integrator(dt=0.09)
        sim.operations.updaters.append(flow)
        sim.run(0)

        # after attachment
        assert flow.num_swaps == 5
        assert flow.target_momentum == 2.5
        assert flow.trigger is trigger
        assert flow.slab_width == 1.0

        assert flow.summed_exchanged_momentum == 0.0

    def test_pickling(self, simulation_factory, snap, flow):
        pickling_check(flow)

        flow.trigger = hoomd.trigger.Periodic(1)
        flow.num_swaps = 5
        pickling_check(flow)

        sim = simulation_factory(snap)
        sim.operations.integrator = hoomd.md.Integrator(dt=0.09)
        sim.operations.updaters.append(flow)
        sim.run(0)
        pickling_check(flow)

    # particle locations in/out of the slabs; and at the boundaries
    @pytest.mark.parametrize("decomposition", decompositions)
    @pytest.mark.parametrize(
        "v0, v1, y0, y1, swap",
        [
            [-2, 2, 0.5, -9.5, True],
            [2, -2, 0.5, -9.5, False],
            [-2, -2, 0.5, -9.5, False],
            [2, 2, 0.5, -9.5, False],
            [-2, 2, 0.5, -7.0, False],
            [-2, 2, 5.0, -9.5, False],
            [-2, 2, 5.0, -7.0, False],
            [-2, 2, 0.0, -10.0, True],
            [-2, 2, 0.0, -9.0, False],
            [-2, 2, 1.0, -9.0, False],
            [-2, 2, 1.0, -10.0, False],
        ],
    )
    def test_particle_swapping(
        self, simulation_factory, snap, flow, v0, v1, y0, y1, swap, decomposition
    ):
        if snap.communicator.rank == 0:
            snap.particles.N = 2
            snap.particles.position[0] = [1.0, y0, 2.0]
            snap.particles.position[1] = [1.0, y1, 2.0]
            snap.particles.velocity[0] = [v0, 4.0, 5.0]
            snap.particles.velocity[1] = [v1, 1.0, 3.0]

        flow.target_momentum = 2.0
        sim = simulation_factory(snap, domain_decomposition=decomposition)
        sim.operations.integrator = hoomd.md.Integrator(dt=0.09)
        sim.operations.updaters.append(flow)
        sim.run(1)

        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            assert np.allclose(
                snap.particles.velocity[0], [v1 if swap else v0, 4.0, 5.0]
            )
            assert np.allclose(
                snap.particles.velocity[1], [v0 if swap else v1, 1.0, 3.0]
            )

        sim.run(1)

        # confirm that no further swaps happen after the previous run
        snapshot = sim.state.get_snapshot()
        if snapshot.communicator.rank == 0:
            assert np.allclose(
                snapshot.particles.velocity[0], snap.particles.velocity[0]
            )
            assert np.allclose(
                snapshot.particles.velocity[1], snap.particles.velocity[1]
            )

    def test_swapping_at_different_timesteps(self, simulation_factory, snap, flow):
        flow.target_momentum = 2.0
        # set initial positions and velocities
        if snap.communicator.rank == 0:
            snap.particles.N = 4
            snap.particles.position[0] = [1.0, 0.0, 2.0]
            snap.particles.position[1] = [2.0, 0.3, 1.0]
            snap.particles.position[2] = [1.0, -9.7, 3.0]
            snap.particles.position[3] = [3.0, -10.0, 1.0]

            snap.particles.velocity[0] = [-1.0, 4.0, 5.0]
            snap.particles.velocity[1] = [-9.0, 1.0, 3.0]
            snap.particles.velocity[2] = [1.0, -2.0, 4.0]
            snap.particles.velocity[3] = [5.0, -4.0, -1.0]

        # set up simulation
        flow.num_swaps = 1
        sim = simulation_factory(snap)
        sim.operations.integrator = hoomd.md.Integrator(dt=0.09)
        sim.operations.updaters.append(flow)
        sim.run(1)

        assert flow.summed_exchanged_momentum == 2.0

        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            # check that swap only happens between particle ids 0,2
            assert np.allclose(snap.particles.velocity[0], [1.0, 4.0, 5.0])
            assert np.allclose(snap.particles.velocity[1], [-9.0, 1.0, 3.0])
            assert np.allclose(snap.particles.velocity[2], [-1.0, -2.0, 4.0])
            assert np.allclose(snap.particles.velocity[3], [5.0, -4.0, -1.0])

        sim.run(1)

        assert flow.summed_exchanged_momentum == 2.0 + 14.0

        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            # check that swapping is now between particle ids 1,3
            assert np.allclose(snap.particles.velocity[0], [1.0, 4.0, 5.0])
            assert np.allclose(snap.particles.velocity[1], [5.0, 1.0, 3.0])
            assert np.allclose(snap.particles.velocity[2], [-1.0, -2.0, 4.0])
            assert np.allclose(snap.particles.velocity[3], [-9.0, -4.0, -1.0])

    def test_default_target_momentum(self, simulation_factory, snap, flow):
        if snap.communicator.rank == 0:
            snap.particles.N = 4
            # set initial positions and velocities
            snap.particles.position[0] = [1.0, 0.5, 2.0]
            snap.particles.position[1] = [2.0, 0.3, 1.0]
            snap.particles.position[2] = [1.0, -9.7, 3.0]
            snap.particles.position[3] = [3.0, -9.2, 1.0]

            snap.particles.velocity[0] = [-9.0, 4.0, 5.0]
            snap.particles.velocity[1] = [-1.0, 1.0, 3.0]
            snap.particles.velocity[2] = [1.0, -2.0, 4.0]
            snap.particles.velocity[3] = [3.0, -4.0, -1.0]

        # set up simulation
        sim = simulation_factory(snap)
        sim.operations.integrator = hoomd.md.Integrator(dt=0.09)
        sim.operations.updaters.append(flow)
        sim.run(1)

        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            # check that swap only happens between fastest and slowest particles
            assert np.allclose(snap.particles.velocity[0], [3.0, 4.0, 5.0])
            assert np.allclose(snap.particles.velocity[1], [-1.0, 1.0, 3.0])
            assert np.allclose(snap.particles.velocity[2], [1.0, -2.0, 4.0])
            assert np.allclose(snap.particles.velocity[3], [-9.0, -4.0, -1.0])

    def test_particle_group(self, simulation_factory, snap):
        if snap.communicator.rank == 0:
            snap.particles.N = 6
            snap.particles.types = ["A", "B"]
            snap.particles.typeid[:] = [1, 1, 0, 0, 0, 0]

            # the particle positions of type B is chosen to be in upper and lower slab
            snap.particles.position[0] = [1.5, 0.2, 1.0]
            snap.particles.position[1] = [2.5, -9.5, 2.0]
            snap.particles.position[2] = [1.0, 0.5, 2.0]
            snap.particles.position[3] = [2.0, 0.3, 1.0]
            snap.particles.position[4] = [1.0, -9.7, 3.0]
            snap.particles.position[5] = [3.0, -9.2, 1.0]

            # the velocities of type B are chosen to be the same as target momentum
            snap.particles.velocity[0] = [-2.0, 2.0, 1.0]
            snap.particles.velocity[1] = [2.0, -1.0, 2.0]
            snap.particles.velocity[2] = [-9.0, 4.0, 5.0]
            snap.particles.velocity[3] = [-1.0, 1.0, 3.0]
            snap.particles.velocity[4] = [1.0, -2.0, 4.0]
            snap.particles.velocity[5] = [3.0, -4.0, -1.0]

            # set up simulation
            flow = hoomd.md.update.ReverseNonequilibriumShearFlow(
                trigger=1,
                filter=hoomd.filter.Type(["A"]),
                num_swaps=1,
                slab_width=1.0,
                target_momentum=2.0,
            )
            sim = simulation_factory(snap)
            sim.operations.integrator = hoomd.md.Integrator(dt=0.09)
            sim.operations.updaters.append(flow)
            sim.run(1)

            snap = sim.state.get_snapshot()
            if snap.communicator.rank == 0:
                # if particles of type A are filtered, B won't be swapped
                assert np.allclose(snap.particles.velocity[0], [-2.0, 2.0, 1.0])
                assert np.allclose(snap.particles.velocity[1], [2.0, -1.0, 2.0])
                assert np.allclose(snap.particles.velocity[2], [-9.0, 4.0, 5.0])
                assert np.allclose(snap.particles.velocity[3], [1.0, 1.0, 3.0])
                assert np.allclose(snap.particles.velocity[4], [-1.0, -2.0, 4.0])
                assert np.allclose(snap.particles.velocity[5], [3, -4.0, -1.0])

    def test_logging(self):
        logging_check(
            hoomd.md.update.ReverseNonequilibriumShearFlow,
            ("md", "update"),
            {
                "summed_exchanged_momentum": {
                    "category": LoggerCategories.scalar,
                    "default": True,
                }
            },
        )
