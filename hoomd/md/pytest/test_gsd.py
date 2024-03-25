# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import numpy as np
import pytest
try:
    import gsd.hoomd
except ImportError:
    pytest.skip("gsd not available", allow_module_level=True)

from hoomd.pytest.test_snapshot import assert_equivalent_snapshots


@pytest.fixture(scope='function')
def hoomd_snapshot(lattice_snapshot_factory):
    snap = lattice_snapshot_factory(particle_types=['t1', 't2'], n=10, a=2.0)
    if snap.communicator.rank == 0:
        typeid_list = [0] * int(snap.particles.N / 2)
        typeid_list.extend([1] * int(snap.particles.N / 2))
        snap.particles.typeid[:] = typeid_list[:]
        snap.particles.velocity[:] = np.tile(np.linspace(1, 2, 3),
                                             (snap.particles.N, 1))
        snap.particles.acceleration[:] = np.tile(np.linspace(1, 2, 3),
                                                 (snap.particles.N, 1))
        snap.particles.mass[:] = snap.particles.N * [1]
        snap.particles.charge[:] = snap.particles.N * [2]
        snap.particles.angmom[:] = snap.particles.N * [[0, 0, 0, 1]]

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
def create_md_sim(simulation_factory, device, hoomd_snapshot):
    sim = simulation_factory(hoomd_snapshot)
    sim.operations.integrator = lj_integrator()

    return sim


def test_write(simulation_factory, hoomd_snapshot, tmp_path):
    filename = tmp_path / "temporary_test_file.gsd"
    sim = simulation_factory(hoomd_snapshot)
    hoomd.write.GSD.write(state=sim.state, mode='wb', filename=str(filename))
    if hoomd_snapshot.communicator.rank == 0:
        with gsd.hoomd.open(name=filename, mode='r') as traj:
            assert len(traj) == 1
            assert_equivalent_snapshots(traj[0], hoomd_snapshot)


def test_write_gsd_trigger(create_md_sim, tmp_path):

    filename = tmp_path / "temporary_test_file.gsd"

    sim = create_md_sim
    gsd_trigger = hoomd.trigger.Periodic(period=10, phase=5)
    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=gsd_trigger,
                                 mode='wb',
                                 dynamic=['property', 'momentum'])
    sim.operations.writers.append(gsd_writer)

    sim.run(30)
    gsd_writer.flush()

    if sim.device.communicator.rank == 0:
        with gsd.hoomd.open(name=filename, mode='r') as traj:
            assert [frame.configuration.step for frame in traj] == [5, 15, 25]


def test_write_gsd_mode(create_md_sim, hoomd_snapshot, tmp_path,
                        simulation_factory):

    filename = tmp_path / "temporary_test_file.gsd"

    sim = create_md_sim
    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='wb',
                                 dynamic=['property', 'momentum'])
    sim.operations.writers.append(gsd_writer)

    # run 5 steps and create a gsd file for testing mode=ab
    sim.run(5)

    # test mode=ab
    sim.operations.writers.clear()

    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='ab',
                                 dynamic=['property', 'momentum'])
    sim.operations.writers.append(gsd_writer)

    snap_list = []
    for _ in range(5):
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            snap_list.append(snap)
    if sim.device.communicator.rank == 0:
        with gsd.hoomd.open(name=filename, mode='r') as traj:
            for gsd_snap, hoomd_snap in zip(traj[5:], snap_list):
                assert_equivalent_snapshots(gsd_snap, hoomd_snap)

    # test mode=xb raises an exception when the file exists
    sim.operations.writers.clear()

    if sim.device.communicator.num_ranks == 1:
        gsd_writer = hoomd.write.GSD(filename=filename,
                                     trigger=hoomd.trigger.Periodic(1),
                                     mode='xb',
                                     dynamic=['property', 'momentum'])
        with pytest.raises(Exception, match='.*File exists.*'):
            sim.operations.writers.append(gsd_writer)
            sim.run(1)

    # test mode=xb creates a new file
    filename_xb = tmp_path / "new_temporary_test_file.gsd"
    sim = simulation_factory(hoomd_snapshot)
    sim.operations.integrator = lj_integrator()

    gsd_writer = hoomd.write.GSD(filename=filename_xb,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='xb',
                                 dynamic=['property', 'momentum'])
    sim.operations.writers.append(gsd_writer)

    snapshot_list = []
    for _ in range(5):
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            snapshot_list.append(snap)

    gsd_writer.flush()

    if sim.device.communicator.rank == 0:
        with gsd.hoomd.open(name=filename_xb, mode='r') as traj:
            assert len(traj) == len(snapshot_list)
            for gsd_snap, hoomd_snap in zip(traj, snapshot_list):
                assert_equivalent_snapshots(gsd_snap, hoomd_snap)


def test_write_gsd_filter(create_md_sim, tmp_path):

    # test Null filter
    filename = tmp_path / "temporary_test_file.gsd"

    sim = create_md_sim
    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 filter=hoomd.filter.Null(),
                                 mode='wb',
                                 dynamic=['property', 'momentum'])
    sim.operations.writers.append(gsd_writer)

    sim.run(3)

    if sim.device.communicator.rank == 0:
        with gsd.hoomd.open(name=filename, mode='r') as traj:
            for frame in traj:
                assert frame.particles.N == 0


def test_write_gsd_truncate(create_md_sim, tmp_path):

    filename = tmp_path / "temporary_test_file.gsd"

    sim = create_md_sim

    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 truncate=True,
                                 mode='wb')
    sim.operations.writers.append(gsd_writer)

    sim.run(2)
    snapshot = sim.state.get_snapshot()

    if snapshot.communicator.rank == 0:
        with gsd.hoomd.open(name=filename, mode='r') as traj:
            for gsd_snap in traj:
                assert_equivalent_snapshots(gsd_snap, snapshot)


def test_write_gsd_dynamic(simulation_factory, create_md_sim, tmp_path):

    filename = tmp_path / "temporary_test_file.gsd"

    sim = create_md_sim

    # test default dynamic=['property']
    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='wb')
    sim.operations.writers.append(gsd_writer)
    velocity_list = []
    position_list = []
    for _ in range(5):
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            position_list.append(snap.particles.position)
            velocity_list.append(snap.particles.velocity)
    N_particles = sim.state.N_particles

    gsd_writer.flush()

    if sim.device.communicator.rank == 0:
        with gsd.hoomd.open(name=filename, mode='r') as traj:
            for step in range(5):
                np.testing.assert_allclose(traj[step].particles.position,
                                           position_list[step],
                                           rtol=1e-07,
                                           atol=1.5e-07)
                np.testing.assert_allclose(traj[step].particles.orientation,
                                           N_particles * [[1, 0, 0, 0]],
                                           rtol=1e-07,
                                           atol=1.5e-07)
                np.testing.assert_allclose(traj[step].particles.velocity,
                                           velocity_list[0],
                                           rtol=1e-07,
                                           atol=1.5e-07)

    # test dynamic=['property', 'momentum']
    sim.operations.writers.clear()
    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='wb',
                                 dynamic=['property', 'momentum'])
    sim.operations.writers.append(gsd_writer)

    velocity_list = []
    angmom_list = []
    for _ in range(5):
        sim.run(1)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            velocity_list.append(snap.particles.velocity)
            angmom_list.append(snap.particles.angmom)

    gsd_writer.flush()

    if sim.device.communicator.rank == 0:
        with gsd.hoomd.open(name=filename, mode='r') as traj:
            for step in range(5):
                np.testing.assert_allclose(traj[step].particles.velocity,
                                           velocity_list[step],
                                           rtol=1e-07,
                                           atol=1.5e-07)
                np.testing.assert_allclose(traj[step].particles.angmom,
                                           angmom_list[step],
                                           rtol=1e-07,
                                           atol=1.5e-07)
                np.testing.assert_allclose(traj[step].particles.image,
                                           N_particles * [[0, 0, 0]],
                                           rtol=1e-07,
                                           atol=1.5e-07)

    # test dynamic=['property', 'attribute']
    if snap.communicator.rank == 0:
        snap.particles.typeid[:] = N_particles * [1]
        snap.particles.mass[:] = N_particles * [0.8]
        snap.particles.charge[:] = N_particles * [0]

    sim.state.set_snapshot(snap)

    sim.operations.writers.clear()
    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='ab',
                                 dynamic=['property', 'attribute'])
    sim.operations.writers.append(gsd_writer)
    sim.run(5)

    gsd_writer.flush()

    if sim.device.communicator.rank == 0:
        with gsd.hoomd.open(name=filename, mode='r') as traj:
            for step in range(5, 10):
                np.testing.assert_allclose(traj[step].particles.mass,
                                           N_particles * [0.8],
                                           rtol=1e-07,
                                           atol=1.5e-07)
                np.testing.assert_allclose(traj[step].particles.charge,
                                           N_particles * [0],
                                           rtol=1e-07,
                                           atol=1.5e-07)
                np.testing.assert_allclose(traj[step].particles.body,
                                           N_particles * [-1],
                                           rtol=1e-07,
                                           atol=1.5e-07)
                np.testing.assert_allclose(traj[step].particles.moment_inertia,
                                           N_particles * [[0, 0, 0]],
                                           rtol=1e-07,
                                           atol=1.5e-07)

    # test dynamic=['property', 'topology']
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        snap.bonds.N = 3
        snap.bonds.typeid[2] = 0
        snap.bonds.group[2] = [10, 11]

    sim.state.set_snapshot(snap)

    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='ab',
                                 dynamic=['property', 'topology'])
    sim.operations.writers.append(gsd_writer)
    sim.run(1)

    gsd_writer.flush()

    if sim.device.communicator.rank == 0:
        with gsd.hoomd.open(name=filename, mode='r') as traj:
            assert traj[-1].bonds.N == 3


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
                                 logger=logger)
    sim.operations.writers.append(gsd_writer)

    kinetic_energy_list = []
    for _ in range(5):
        sim.run(1)
        kinetic_energy_list.append(thermo.kinetic_energy)

    gsd_writer.flush()

    if sim.device.communicator.rank == 0:
        with gsd.hoomd.open(name=filename, mode='r') as traj:
            for s in range(5):
                e = traj[s].log[
                    'md/compute/ThermodynamicQuantities/kinetic_energy']
                assert e == kinetic_energy_list[s]


dynamic_fields = [
    'particles/position',
    'particles/orientation',
    'particles/velocity',
    'particles/angmom',
    'particles/image',
    'particles/typeid',
    'particles/mass',
    'particles/charge',
    'particles/diameter',
    'particles/body',
    'particles/moment_inertia',
]


@pytest.mark.parametrize('dynamic_field', dynamic_fields)
def test_write_gsd_finegrained_dynamic(simulation_factory, hoomd_snapshot,
                                       tmp_path, dynamic_field):

    filename = tmp_path / "test_finegrained_dynamic.gsd"

    # make all fields in snapshot non-default
    if hoomd_snapshot.communicator.rank == 0:
        hoomd_snapshot.particles.orientation[:] = np.tile(
            [0.707, 0, 0, 0.707], (hoomd_snapshot.particles.N, 1))
        hoomd_snapshot.particles.image[:] = np.tile(
            [0, 1, 2], (hoomd_snapshot.particles.N, 1))
        hoomd_snapshot.particles.types = ['A', 'B']
        hoomd_snapshot.particles.typeid[:] = np.tile(1,
                                                     hoomd_snapshot.particles.N)
        hoomd_snapshot.particles.mass[:] = np.tile(2,
                                                   hoomd_snapshot.particles.N)
        hoomd_snapshot.particles.diameter[:] = np.tile(
            4, hoomd_snapshot.particles.N)
        hoomd_snapshot.particles.body[:] = np.tile(4,
                                                   hoomd_snapshot.particles.N)
        hoomd_snapshot.particles.moment_inertia[:] = np.tile(
            [1, 2, 3], [hoomd_snapshot.particles.N, 1])

    sim = simulation_factory(hoomd_snapshot)

    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='wb',
                                 dynamic=[dynamic_field])
    gsd_writer.write_diameter = True
    sim.operations.writers.append(gsd_writer)

    sim.run(2)

    gsd_writer.flush()

    if sim.device.communicator.rank == 0:
        with gsd.fl.open(name=filename, mode='r') as f:
            for field in dynamic_fields:
                if field == dynamic_field:
                    assert f.chunk_exists(frame=1, name=field)
                else:
                    assert not f.chunk_exists(frame=1, name=field)

            data = f.read_chunk(frame=1, name=dynamic_field)
            np.testing.assert_allclose(data,
                                       getattr(hoomd_snapshot.particles,
                                               dynamic_field[10:]),
                                       rtol=1e-07,
                                       atol=1.5e-07)


@pytest.mark.parametrize('dynamic_field', dynamic_fields)
def test_write_gsd_finegrained_dynamic_alldefault(simulation_factory,
                                                  hoomd_snapshot, tmp_path,
                                                  dynamic_field):

    filename = tmp_path / "test_finegrained_dynamic.gsd"

    # make all fields in snapshot default
    if hoomd_snapshot.communicator.rank == 0:
        hoomd_snapshot.particles.position[:] = np.tile(
            [0, 0, 0], [hoomd_snapshot.particles.N, 1])
        hoomd_snapshot.particles.velocity[:] = np.tile(
            [0, 0, 0], [hoomd_snapshot.particles.N, 1])
        hoomd_snapshot.particles.angmom[:] = np.tile(
            [0, 0, 0, 0], [hoomd_snapshot.particles.N, 1])
        hoomd_snapshot.particles.typeid[:] = np.tile(0,
                                                     hoomd_snapshot.particles.N)
        hoomd_snapshot.particles.charge[:] = np.tile(0,
                                                     hoomd_snapshot.particles.N)

    sim = simulation_factory(hoomd_snapshot)

    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='wb',
                                 dynamic=[dynamic_field])
    gsd_writer.write_diameter = True
    sim.operations.writers.append(gsd_writer)

    sim.run(2)

    gsd_writer.flush()

    if sim.device.communicator.rank == 0:
        with gsd.fl.open(name=filename, mode='r') as f:
            assert f.nframes == 2

            for field in dynamic_fields:
                assert not f.chunk_exists(frame=1, name=field)


def test_write_gsd_no_dynamic(simulation_factory, hoomd_snapshot, tmp_path):
    """Ensure that GSD files with no dynamic properties wite expected chunks."""
    filename = tmp_path / "test_no_dynamic.gsd"

    sim = simulation_factory(hoomd_snapshot)

    gsd_writer = hoomd.write.GSD(filename=filename,
                                 trigger=hoomd.trigger.Periodic(1),
                                 mode='wb',
                                 dynamic=[])
    sim.operations.writers.append(gsd_writer)

    sim.run(2)

    gsd_writer.flush()

    if sim.device.communicator.rank == 0:
        with gsd.fl.open(name=filename, mode='r') as f:
            assert f.nframes == 2

            assert f.chunk_exists(frame=0, name='configuration/step')
            assert f.chunk_exists(frame=0, name='configuration/box')
            assert f.chunk_exists(frame=0, name='particles/N')
            # particles/positions is not default, so it is written to frame 0.
            assert f.chunk_exists(frame=0, name='particles/position')

            assert f.chunk_exists(frame=1, name='configuration/step')
            assert not f.chunk_exists(frame=1, name='configuration/box')
            assert not f.chunk_exists(frame=1, name='particles/N')
            assert not f.chunk_exists(frame=1, name='particles/position')
