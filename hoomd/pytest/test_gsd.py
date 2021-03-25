import hoomd
import gsd.hoomd
import numpy as np
import pytest
from hoomd.pytest.test_snapshot import assert_equivalent_snapshots


@pytest.fixture(scope='function')
def hoomd_snapshot(lattice_snapshot_factory):
    snap = lattice_snapshot_factory(particle_types=['p1', 'p2'],
                                    n=10, a=2.0)

    typeid_list = [0] * int(snap.particles.N / 2)
    typeid_list.extend([1] * int(snap.particles.N / 2))
    snap.particles.typeid[:] = typeid_list[:]
    snap.particles.velocity[:] = np.tile(np.linspace(1, 2, 3),
                                         (snap.particles.N, 1))
    snap.particles.acceleration[:] = np.tile(np.linspace(1, 2, 3),
                                             (snap.particles.N, 1))
    snap.particles.mass[:] = snap.particles.N * [1]
    snap.particles.charge[:] = snap.particles.N * [2]
    snap.particles.diameter[:] = snap.particles.N * [0.5]

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

@pytest.fixture(scope='function')
def create_md_sim(simulation_factory, device, hoomd_snapshot):
    sim = simulation_factory(hoomd_snapshot)
    integrator = hoomd.md.Integrator(dt=0.005)
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(), r_cut=2.5)
    lj.params.default = {'sigma': 1, 'epsilon': 1}
    integrator.forces.append(lj)
    langevin = hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1, seed=1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    return sim


def test_write(simulation_factory, hoomd_snapshot, tmp_path):

    filename = tmp_path / "temporary_test_file.gsd"

    sim = simulation_factory(hoomd_snapshot)
    hoomd.write.GSD.write(state=sim.state, mode='wb',
                          filename=str(filename))

    with gsd.hoomd.open(name=filename, mode='wb') as traj:
        assert_equivalent_snapshots(traj, hoomd_snapshot)


def test_write_gsd_trigger(create_md_sim, tmp_path):

    filename = tmp_path / "temporary_test_file.gsd"

    sim = create_md_sim
    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='wb', dynamic=['momentum'])
    sim.operations.writers.append(gsd_writer)

    snapshot_list = []
    for _ in range(5):
        sim.run(1)
        snapshot_list.append(sim.state.snapshot)

    with gsd.hoomd.open(name=filename, mode='rb') as traj:
        assert len(traj) == len(snapshot_list)
        for gsd_snap, hoomd_snap in zip(traj, snapshot_list):
            assert_equivalent_snapshots(gsd_snap, hoomd_snap)


def test_write_gsd_mode(create_md_sim, hoomd_snapshot, tmp_path, simulation_factory):

    filename = tmp_path / "temporary_test_file.gsd"

    with open(filename, 'w') as traj:
        traj.write('test')

    sim = create_md_sim
    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='wb', dynamic=['momentum'])
    sim.operations.writers.append(gsd_writer)

    # run 5 steps and create a gsd file for testing mode=ab
    snapshot_list = []
    for _ in range(5):
        sim.run(1)
        snapshot_list.append(sim.state.snapshot)

    # test mode=ab
    sim.operations.writers.clear()

    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='wb', dynamic=['momentum'])
    sim.operations.writers.append(gsd_writer)

    snap_list = []
    for _ in range(5):
        sim.run(1)
        snap_list.append(sim.state.snapshot)

    with gsd.hoomd.open(name=filename, mode='rb') as traj:
        for gsd_snap, hoomd_snap in zip(traj[5:], snap_list):
            assert_equivalent_snapshots(gsd_snap, hoomd_snap)

    # test mode=xb raises an exception when the file exists
    sim.operations.writers.clear()

    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='xb', dynamic=['momentum'])
    sim.operations.writers.append(gsd_writer)
    with pytest.raises(Exception):
        sim.run(1)

    # test mode=xb creating file
    filename_xb = tmp_path / "new_temporary_test_file.gsd"

    sim = simulation_factory(hoomd_snapshot)
    integrator = hoomd.md.Integrator(dt=0.005)
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(), r_cut=2.5)
    lj.params.default = {'sigma': 1, 'epsilon': 1}
    integrator.forces.append(lj)
    langevin = hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1, seed=1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    gsd_writer = hoomd.write.GSD(filename=filename_xb,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='xb', dynamic=['momentum'])
    sim.operations.writers.append(gsd_writer)

    sim.run(5)

    with gsd.hoomd.open(name=filename_xb, mode='rb') as traj:
        assert len(traj) == len(snapshot_list)
        for gsd_snap, hoomd_snap in zip(traj, snapshot_list):
            assert_equivalent_snapshots(gsd_snap, hoomd_snap)


def test_write_gsd_filter(create_md_sim, hoomd_snapshot, tmp_path, simulation_factory):

    # test Null filter
    filename = tmp_path / "temporary_test_file.gsd"

    sim = create_md_sim
    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 filter=hoomd.filter.Null(),
                                 mode='wb', dynamic=['momentum'])
    sim.operations.writers.append(gsd_writer)

    sim.run(3)

    with gsd.hoomd.open(name=filename, mode='rb') as traj:
        for frame in traj:
            assert frame.particles.N == 0

    # test using Type filter
    # filename = tmp_path / "new_temporary_test_file.gsd"

    sim = simulation_factory(hoomd_snapshot)
    integrator = hoomd.md.Integrator(dt=0.005)
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(), r_cut=2.5)
    lj.params.default = {'sigma': 1, 'epsilon': 1}
    integrator.forces.append(lj)
    langevin = hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1, seed=1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 filter=hoomd.filter.Type(['p2']),
                                 mode='wb')
    sim.operations.writers.append(gsd_writer)

    sim.run(5)
    snapshot = sim.state.snapshot

    with gsd.hoomd.open(name=filename, mode='rb') as traj:
        for frame in traj:
            np.testing.assert_array_equal(frame.particles.typeid,
                                          [1] * int(snapshot.particles.N / 2))


def test_write_gsd_truncate(create_md_sim, tmp_path):

    filename = tmp_path / "temporary_test_file.gsd"

    sim = create_md_sim

    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 truncate=True,
                                 mode='wb')
    sim.operations.writers.append(gsd_writer)

    sim.run(2)
    snapshot = sim.state.snapshot

    with gsd.hoomd.open(name=filename, mode='rb') as traj:
        for gsd_snap in traj:
            assert_equivalent_snapshots(gsd_snap, snapshot)


def test_write_gsd_dynamic(create_md_sim, tmp_path):
    # To do:test everything for dynamic
    filename = tmp_path / "temporary_test_file.gsd"

    sim = create_md_sim

    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='wb', dynamic=['momentum'])
    sim.operations.writers.append(gsd_writer)

    velocity_list = []
    for _ in range(5):
        sim.run(1)
        snap = sim.state.snapshot
        velocity_list.append(snap.particles.velocity)

    with gsd.hoomd.open(name=filename, mode='rb') as traj:
        for step in range(5):
            np.testing.assert_allclose(traj[step].particles.velocity,
                                       velocity_list[step],
                                       rtol=1e-07, atol=1.5e-07)


def test_write_gsd_log(create_md_sim, tmp_path):

    filename = tmp_path / "temporary_test_file.gsd"

    sim = create_md_sim

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    logger = hoomd.logging.Logger()
    logger.add(thermo)

    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 filter=hoomd.filter.Null(),
                                 mode='wb',
                                 log=logger)
    sim.operations.writers.append(gsd_writer)

    kinetic_energy_list = []
    for _ in range(5):
        sim.run(1)
        kinetic_energy_list.append(thermo.kinetic_energy)

    with gsd.hoomd.open(name=filename, mode='rb') as traj:
        for s in range(5):
            e = traj[s].log['md/compute/ThermodynamicQuantities/kinetic_energy']
            assert e == kinetic_energy_list[s]
