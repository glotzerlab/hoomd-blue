import hoomd
from hoomd.conftest import operation_pickling_check
import pytest
import numpy as np


def test_attach(simulation_factory, two_particle_snapshot_factory, tmp_path):
    filename = tmp_path / "temporary_test_file.dcd"
    sim = simulation_factory(two_particle_snapshot_factory())
    dcd_dump = hoomd.write.DCD(filename, hoomd.trigger.Periodic(1))
    sim.operations.add(dcd_dump)
    sim.run(10)


# pip installing garnett does not use Cythonized code, so this warning will
# always be raised unless garnett is built locally.
@pytest.mark.filterwarnings("ignore:Failed to import dcdreader library")
def test_write(simulation_factory, two_particle_snapshot_factory, tmp_path):
    garnett = pytest.importorskip("garnett")
    dcd_reader = garnett.reader.DCDFileReader()
    filename = tmp_path / "temporary_test_file.dcd"
    sim = simulation_factory(two_particle_snapshot_factory())
    dcd_dump = hoomd.write.DCD(filename, hoomd.trigger.Periodic(1))
    sim.operations.add(dcd_dump)
    positions = []

    snap = sim.state.snapshot
    if snap.exists:
        position1 = np.asarray(snap.particles.position[0])
        position2 = np.asarray(snap.particles.position[1])
        positions.append([list(position1), list(position2)])

    sim.run(1)

    if sim.device.communicator.rank == 0:
        with open(filename, 'rb') as dcdfile:
            traj = dcd_reader.read(dcdfile)
            traj.load()
        for i in range(len(traj)):
            for j in [0, 1]:
                np.testing.assert_allclose(traj[i].position[j], positions[i][j])


def test_pickling(simulation_factory, two_particle_snapshot_factory, tmp_path):
    filename = tmp_path / "temporary_test_file.dcd"
    sim = simulation_factory(two_particle_snapshot_factory())
    dcd_dump = hoomd.write.DCD(filename, hoomd.trigger.Periodic(1))
    operation_pickling_check(dcd_dump, sim)
