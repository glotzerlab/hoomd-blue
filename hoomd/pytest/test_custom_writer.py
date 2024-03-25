# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
from hoomd import conftest


class WriteTimestep(hoomd.custom.Action):
    test_attr = True
    trigger = None  # should not be passed through

    def __init__(self):
        self.timesteps_run = []

    def act(self, timestep):
        self.timesteps_run.append(timestep)

    @hoomd.logging.log
    def fourty_two(self):
        return 42


class TestCustomWriter:
    """Serves as tests for the CustomWriter and custom.Action classes.

    This tests the custom action class in a simulation context which requires
    wrapping the class in a custom operation instance.
    """

    def test_attach_detach(self, simulation_factory,
                           two_particle_snapshot_factory):
        sim = simulation_factory(two_particle_snapshot_factory())
        writer = hoomd.write.CustomWriter(2, WriteTimestep())
        sim.operations += writer
        sim.run(0)
        assert writer.action._state is sim.state
        assert writer.action._attached
        assert writer._attached
        sim.operations.writers.pop()
        assert writer.action._state is None
        assert not writer.action._attached
        assert not writer._attached

    @pytest.mark.skipif(not hoomd.version.md_built,
                        reason="BUILD_MD=on required")
    def test_flags(self, simulation_factory, two_particle_snapshot_factory):
        sim = simulation_factory(two_particle_snapshot_factory())
        action = WriteTimestep()
        action.flags = [hoomd.custom.Action.Flags.PRESSURE_TENSOR]
        sim.operations += hoomd.write.CustomWriter(2, action)
        gauss = hoomd.md.pair.Gaussian(hoomd.md.nlist.Cell(0.5))
        gauss.params[("A", "A")] = {"sigma": 1.0, "epsilon": 1.0}
        gauss.r_cut[("A", "A")] = 2.0
        sim.operations += hoomd.md.Integrator(
            0.005,
            methods=[hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1.0)],
            forces=[gauss])
        # WriteTimestep is not run so pressure is not available
        sim.run(1)
        virials = gauss.virials
        assert virials is None
        # WriteTimestep is run so pressure is available
        sim.run(1)
        virials = gauss.virials
        if sim.device.communicator.rank == 0:
            assert any(virials.ravel() != 0)

    def test_logging(self):
        expected_namespace = ("pytest", "test_custom_writer")
        conftest.logging_check(
            WriteTimestep, ("pytest", "test_custom_writer"), {
                "fourty_two": {
                    "category": hoomd.logging.LoggerCategories.scalar,
                    "default": True
                }
            })
        writer = hoomd.write.CustomWriter(2, WriteTimestep())
        # Check namespace
        log_quantity = writer._export_dict["fourty_two"]
        assert log_quantity.namespace == expected_namespace + (
            WriteTimestep.__name__,)
        assert log_quantity.default
        assert log_quantity.category == hoomd.logging.LoggerCategories.scalar

    def test_passthrough(self):
        writer = hoomd.update.CustomUpdater(1, WriteTimestep())
        assert writer.test_attr
        assert writer.trigger is not None
        assert writer.trigger == hoomd.trigger.Periodic(1)

    def test_act(self, simulation_factory, two_particle_snapshot_factory):
        sim = simulation_factory(two_particle_snapshot_factory())
        writer = hoomd.write.CustomWriter(2, WriteTimestep())
        sim.operations += writer
        sim.run(10)
        assert writer.timesteps_run == [2, 4, 6, 8, 10]
