import hoomd
import pytest
import numpy as np


def test_attach(simulation_factory, two_particle_snapshot_factory, tmp_path):
    filename = tmp_path / "temporary_test_file.dcd"
    sim = simulation_factory(two_particle_snapshot_factory())
    dcd_dump = hoomd.write.DCD(filename, hoomd.trigger.Periodic(1))
    sim.operations.add(dcd_dump)
    sim.run(10)


def test_set_period(tmp_path):
    filename = tmp_path / "temporary_test_file.dcd"
    dcd_dump = hoomd.write.DCD(filename, hoomd.trigger.Periodic(1))
    with pytest.raises(RuntimeError):
        dcd_dump.set_period(1)


def test_enabled(tmp_path):
    filename = tmp_path / "temporary_test_file.dcd"
    dcd_dump = hoomd.write.DCD(filename, hoomd.trigger.Periodic(1))
    dcd_dump.enabled = False
    with pytest.raises(RuntimeError):
        dcd_dump.enable()


def test_write(simulation_factory, two_particle_snapshot_factory, tmp_path):
    garnett = pytest.importorskip("garnett")
    dcd_reader = garnett.reader.DCDFileReader()
    filename = tmp_path / "temporary_test_file.dcd"
    sim = simulation_factory(two_particle_snapshot_factory())
    dcd_dump = hoomd.write.DCD(filename, hoomd.trigger.Periodic(1))
    sim.operations.add(dcd_dump)
    snap = sim.state.snapshot
    positions = []
    for i in range(10):
        snap = sim.state.snapshot
        position1 = np.asarray(snap.particles.position[0])
        position2 = np.asarray(snap.particles.position[1])
        position1 += 0.1 * i * (-1)**i
        position2 += 0.1 * (i + 1) * (-1)**(i - 1)
        if snap.exists:
            snap.particles.position[0] = position1
            snap.particles.position[1] = position2
        sim.state.snapshot = snap
        sim.run(1)
        positions.append([list(position1), list(position2)])
    with open(filename, 'rb') as dcdfile:
        traj = dcd_reader.read(dcdfile)
        traj.load()
    for i in range(len(traj)):
        for j in [0, 1]:
            np.testing.assert_allclose(traj[i].position[j], positions[i][j])
