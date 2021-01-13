import hoomd
import pytest
import numpy as np


def test_attach(simulation_factory, two_particle_snapshot_factory, tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    filename = d / "temporary_test_file.dcd"
    sim = simulation_factory(two_particle_snapshot_factory())
    dcd_dump = hoomd.write.DCD(filename, hoomd.trigger.Periodic(1))
    sim.operations.add(dcd_dump)
    sim.operations._schedule()
    sim.run(10)
