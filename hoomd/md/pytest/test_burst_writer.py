# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from pathlib import Path

import hoomd
import numpy as np
import pytest
try:
    import gsd.hoomd
except ImportError:
    pytest.skip("gsd not available", allow_module_level=True)

from hoomd.pytest.test_snapshot import assert_equivalent_snapshots

N_RUN_STEPS = 3


@pytest.fixture(scope='function')
def hoomd_snapshot(lattice_snapshot_factory):
    snap = lattice_snapshot_factory(particle_types=['t1', 't2'], n=10, a=2.0)
    if snap.communicator.rank == 0:
        Np = snap.particles.N
        snap.particles.typeid[:] = np.repeat([0, 1], int(Np / 2))
        snap.particles.velocity[:] = np.linspace(1, 2, 3)
        snap.particles.acceleration[:] = np.linspace(1, 2, 3)
        snap.particles.mass[:] = 1.1
        snap.particles.charge[:] = 2.0
        snap.particles.angmom[:] = np.array([0, 0, 0, 1])

        # bonds
        snap.bonds.types = ['b1', 'b2']
        snap.bonds.N = 2
        snap.bonds.typeid[:] = [0, 1]
        snap.bonds.group[0] = [0, 1]
        snap.bonds.group[1] = [2, 3]

        # angles
        snap.angles.types = ['a1', 'a2']
        snap.angles.N = 2
        snap.angles.typeid[:] = [1, 0]
        snap.angles.group[0] = [0, 1, 2]
        snap.angles.group[1] = [2, 3, 0]

        # dihedrals
        snap.dihedrals.types = ['d1']
        snap.dihedrals.N = 1
        snap.dihedrals.typeid[:] = [0]
        snap.dihedrals.group[0] = [0, 1, 2, 3]

        # impropers
        snap.impropers.types = ['i1']
        snap.impropers.N = 1
        snap.impropers.typeid[:] = [0]
        snap.impropers.group[0] = [3, 2, 1, 0]

        # constraints
        snap.constraints.N = 1
        snap.constraints.group[0] = [0, 1]
        snap.constraints.value[0] = 2.5

        # special pairs
        snap.pairs.types = ['p1', 'p2']
        snap.pairs.N = 2
        snap.pairs.typeid[:] = [0, 1]
        snap.pairs.group[0] = [0, 1]
        snap.pairs.group[1] = [2, 3]

    return snap


def lj_integrator():
    integrator = hoomd.md.Integrator(dt=0.005)
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4),
                          default_r_cut=2.5)
    lj.params.default = {'sigma': 1, 'epsilon': 1}
    integrator.forces.append(lj)
    langevin = hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1)
    integrator.methods.append(langevin)

    return integrator


@pytest.fixture(scope='function')
def sim(simulation_factory, hoomd_snapshot):
    sim = simulation_factory(hoomd_snapshot)
    sim.operations.integrator = lj_integrator()
    return sim


def check_write(sim: hoomd.Simulation, filename: str, trigger_period: int):
    snaps = []
    for _ in range(N_RUN_STEPS):
        sim.run(trigger_period)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            snaps.append(snap)
    sim.operations.writers[0].dump()
    sim.operations.writers[0].flush()
    if sim.device.communicator.rank == 0:
        with gsd.hoomd.open(name=filename, mode='r') as traj:
            # have to skip first frame which is from the first call.
            for snap, gsd_snap in zip(snaps, traj[1:]):
                assert_equivalent_snapshots(gsd_snap, snap)


def test_write_on_start(sim, tmp_path):
    filename = tmp_path / "temporary_test_file.gsd"
    burst_writer = hoomd.write.Burst(trigger=1,
                                     filename=filename,
                                     mode='wb',
                                     dynamic=['property', 'momentum'],
                                     max_burst_size=3)
    sim.operations.writers.append(burst_writer)
    # Errors when file does not exist
    with pytest.raises(RuntimeError):
        # still creates file before erroring.
        sim.run(0)
    sim.operations.writers.clear()
    burst_writer = hoomd.write.Burst(trigger=1,
                                     filename=filename,
                                     mode='wb',
                                     dynamic=['property', 'momentum'],
                                     max_burst_size=3)
    sim.operations.writers.append(burst_writer)
    # Errors when file exists without frame
    with pytest.raises(RuntimeError):
        sim.run(0)


def test_len(sim, tmp_path):
    filename = tmp_path / "temporary_test_file.gsd"

    burst_trigger = hoomd.trigger.Periodic(period=2, phase=1)
    burst_writer = hoomd.write.Burst(trigger=burst_trigger,
                                     filename=filename,
                                     mode='wb',
                                     dynamic=['property', 'momentum'],
                                     max_burst_size=3,
                                     write_at_start=True)
    sim.operations.writers.append(burst_writer)
    sim.run(8)
    assert len(burst_writer) == 3
    burst_writer.dump()
    assert len(burst_writer) == 0


def test_burst_dump(sim, tmp_path):
    filename = tmp_path / "temporary_test_file.gsd"

    burst_trigger = hoomd.trigger.Periodic(period=2, phase=1)
    burst_writer = hoomd.write.Burst(trigger=burst_trigger,
                                     filename=filename,
                                     mode='wb',
                                     dynamic=['property', 'momentum'],
                                     max_burst_size=3,
                                     write_at_start=True)
    sim.operations.writers.append(burst_writer)
    sim.run(8)
    burst_writer.flush()
    if sim.device.communicator.rank == 0:
        assert Path(filename).exists()
        with gsd.hoomd.open(filename, "r") as traj:
            # First frame is always written
            assert len(traj) == 1

    burst_writer.dump()
    burst_writer.flush()
    if sim.device.communicator.rank == 0:
        with gsd.hoomd.open(name=filename, mode='r') as traj:
            assert [frame.configuration.step for frame in traj] == [0, 3, 5, 7]


def test_burst_max_size(sim, tmp_path):
    filename = Path(tmp_path / "temporary_test_file.gsd")
    burst_writer = hoomd.write.Burst(filename=str(filename),
                                     trigger=hoomd.trigger.Periodic(1),
                                     mode='wb',
                                     dynamic=['property', 'momentum'],
                                     max_burst_size=N_RUN_STEPS,
                                     write_at_start=True)
    sim.operations.writers.append(burst_writer)
    # Run 1 extra step to fill the burst which does not include the first frame
    sim.run(N_RUN_STEPS + 1)
    # Should write the last N_RUN_STEPS not any of the former.
    check_write(sim, filename, 1)


def test_burst_mode_xb(sim, tmp_path):
    filename = tmp_path / "temporary_test_file.gsd"
    if sim.device.communicator.rank == 0:
        Path(filename).touch()
    burst_writer = hoomd.write.Burst(filename=filename,
                                     trigger=hoomd.trigger.Periodic(1),
                                     mode='xb',
                                     dynamic=['property', 'momentum'],
                                     write_at_start=True)
    sim.operations.writers.append(burst_writer)
    if sim.device.communicator.rank == 0:
        with pytest.raises(RuntimeError):
            sim.run(0)

    sim.operations.remove(burst_writer)
    # test mode=xb creates a new file
    filename_xb = tmp_path / "new_temporary_test_file.gsd"

    burst_writer = hoomd.write.Burst(filename=filename_xb,
                                     trigger=hoomd.trigger.Periodic(1),
                                     mode='xb',
                                     dynamic=['property', 'momentum'],
                                     write_at_start=True)
    sim.operations.writers.append(burst_writer)
    check_write(sim, filename_xb, 1)


def test_write_burst_log(sim, tmp_path):

    filename = tmp_path / "temporary_test_file.gsd"

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    logger = hoomd.logging.Logger()
    logger.add(thermo)

    burst_writer = hoomd.write.Burst(filename=filename,
                                     trigger=hoomd.trigger.Periodic(1),
                                     filter=hoomd.filter.Null(),
                                     mode='wb',
                                     logger=logger,
                                     write_at_start=True)
    sim.operations.writers.append(burst_writer)

    kinetic_energies = []
    for _ in range(N_RUN_STEPS):
        sim.run(1)
        kinetic_energies.append(thermo.kinetic_energy)
    burst_writer.dump()
    burst_writer.flush()
    if sim.device.communicator.rank == 0:
        key = "md/compute/ThermodynamicQuantities/kinetic_energy"
        with gsd.hoomd.open(name=filename, mode='r') as traj:
            for frame, sim_ke in zip(traj[1:], kinetic_energies):
                assert frame.log[key] == sim_ke
