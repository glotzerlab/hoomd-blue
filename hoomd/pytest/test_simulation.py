# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import time

import hoomd
import numpy as np
import pytest
from copy import deepcopy
from hoomd.error import MutabilityError
from hoomd.logging import LoggerCategories
from hoomd.conftest import logging_check, ListWriter
try:
    import gsd.hoomd
    skip_gsd = False
except ImportError:
    skip_gsd = True

skip_gsd = pytest.mark.skipif(skip_gsd,
                              reason="gsd Python package was not found.")


class SleepUpdater(hoomd.custom.Action):

    def act(self, timestep):
        time.sleep(1e-6 * timestep)

    @classmethod
    def wrapped(cls):
        return hoomd.update.CustomUpdater(1, cls())


def make_gsd_frame(hoomd_snapshot):
    s = gsd.hoomd.Frame()
    for attr in dir(hoomd_snapshot):
        if attr[0] != '_' and attr not in [
                'exists', 'replicate', 'communicator', 'mpcd'
        ]:
            if hoomd_snapshot.communicator.rank == 0:
                for prop in dir(getattr(hoomd_snapshot, attr)):
                    if prop[0] != '_':
                        # s.attr.prop = hoomd_snapshot.attr.prop
                        setattr(getattr(s, attr), prop,
                                getattr(getattr(hoomd_snapshot, attr), prop))
    return s


def set_types(s, inds, particle_types, particle_type):
    if s.communicator.rank == 0:
        for i in inds:
            s.particles.typeid[i] = particle_types.index(particle_type)


def update_positions(snap):
    if snap.communicator.rank == 0:
        noise = 0.01
        rs = np.random.RandomState(0)
        mean = [0] * 3
        var = noise * noise
        cov = np.diag([var, var, var])
        shape = snap.particles.position.shape
        snap.particles.position[:] += rs.multivariate_normal(mean,
                                                             cov,
                                                             size=shape[:-1])
    return snap


def assert_equivalent_snapshots(gsd_snap, hoomd_snap):
    if hoomd_snap.communicator.rank == 0:

        for attr in dir(hoomd_snap):
            if attr[0] == '_' or attr in [
                    'exists', 'replicate', 'communicator', 'mpcd'
            ]:
                continue
            for prop in dir(getattr(hoomd_snap, attr)):
                if prop[0] == '_':
                    continue
                elif prop == 'types':
                    assert getattr(getattr(gsd_snap, attr), prop) == \
                        getattr(getattr(hoomd_snap, attr), prop)
                else:
                    np.testing.assert_allclose(
                        getattr(getattr(gsd_snap, attr), prop),
                        getattr(getattr(hoomd_snap, attr), prop))


def random_inds(n):
    return np.random.choice(np.arange(n),
                            size=int(n * np.random.rand()),
                            replace=False)


def test_initialization(simulation_factory):
    with pytest.raises(TypeError):
        sim = hoomd.Simulation()

    sim = simulation_factory()  # noqa


def test_device_property(device):
    sim = hoomd.Simulation(device)
    assert sim.device is device

    with pytest.raises(ValueError):
        sim.device = device


def test_allows_compute_pressure(simulation_factory, lattice_snapshot_factory):
    sim = simulation_factory()
    assert not sim.always_compute_pressure
    with pytest.raises(RuntimeError):
        sim.always_compute_pressure = True
    sim.create_state_from_snapshot(lattice_snapshot_factory())
    sim.always_compute_pressure = True
    assert sim.always_compute_pressure is True


def test_run(simulation_factory, lattice_snapshot_factory):
    sim = simulation_factory()
    with pytest.raises(RuntimeError):
        sim.run(1)  # Before setting state

    sim = simulation_factory(lattice_snapshot_factory())
    sim.run(1)

    assert sim.operations._scheduled


def test_tps(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory()
    assert sim.tps is None

    sim = simulation_factory(two_particle_snapshot_factory())
    assert sim.tps == 0

    list_writer = ListWriter(sim, "tps")
    sim.operations.writers.append(
        hoomd.write.CustomWriter(action=list_writer,
                                 trigger=hoomd.trigger.Periodic(1)))
    sim.operations += SleepUpdater.wrapped()
    sim.run(10)
    tps = list_writer.data
    assert len(np.unique(tps)) > 1


def test_walltime(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory()
    assert sim.walltime == 0

    sim = simulation_factory(two_particle_snapshot_factory())
    assert sim.walltime == 0

    list_writer = ListWriter(sim, "walltime")
    sim.operations.writers.append(
        hoomd.write.CustomWriter(action=list_writer,
                                 trigger=hoomd.trigger.Periodic(1)))
    sim.operations += SleepUpdater.wrapped()
    sim.run(10)
    walltime = list_writer.data
    assert all(a >= b for a, b in zip(walltime[1:], walltime[:-1]))


def test_timestep(simulation_factory, lattice_snapshot_factory):
    sim = simulation_factory()
    assert sim.timestep is None

    initial_steps = 10
    sim.timestep = initial_steps
    assert sim.timestep == initial_steps
    sim.create_state_from_snapshot(lattice_snapshot_factory())
    assert sim.timestep == initial_steps

    with pytest.raises(RuntimeError):
        sim.timestep = 20


def test_run_with_timestep(simulation_factory, lattice_snapshot_factory):
    sim = simulation_factory(lattice_snapshot_factory())

    steps = 0
    n_step_list = [1, 10, 100]
    for n_steps in n_step_list:
        steps += n_steps
        sim.run(n_steps)
        assert sim.timestep == steps

    assert sim.timestep == sum(n_step_list)


_state_args = [((10, ['A']), 10), ((5, ['A', 'B']), 20),
               ((8, ['A', 'B', 'C']), 4)]


@pytest.fixture(scope="function", params=_state_args)
def state_args(request):
    return deepcopy(request.param)


@skip_gsd
def test_state_from_gsd(device, simulation_factory, lattice_snapshot_factory,
                        state_args, tmp_path):
    snap_params, nsteps = state_args

    d = tmp_path / "sub"
    d.mkdir()
    filename = d / "temporary_test_file.gsd"
    if device.communicator.rank == 0:
        f = gsd.hoomd.open(name=filename, mode='w')

    sim = simulation_factory(
        lattice_snapshot_factory(n=snap_params[0],
                                 particle_types=snap_params[1]))
    snap = sim.state.get_snapshot()
    snapshot_dict = {}
    snapshot_dict[0] = snap

    if device.communicator.rank == 0:
        f.append(make_gsd_frame(snap))

    box = sim.state.box
    for step in range(1, nsteps):
        particle_type = np.random.choice(snap_params[1])
        snap = update_positions(sim.state.get_snapshot())
        set_types(snap, random_inds(snap_params[0]), snap_params[1],
                  particle_type)

        if device.communicator.rank == 0:
            f.append(make_gsd_frame(snap))
            snapshot_dict[step] = snap
        else:
            snapshot_dict[step] = None

    if device.communicator.rank == 0:
        f.close()

    for step, snap in snapshot_dict.items():
        sim = simulation_factory()
        sim.create_state_from_gsd(filename, frame=step)
        assert box == sim.state.box

        assert_equivalent_snapshots(snap, sim.state.get_snapshot())


@skip_gsd
def test_state_from_gsd_box_dims(device, simulation_factory,
                                 lattice_snapshot_factory, tmp_path):

    def modify_gsd_snap(gsd_snap):
        """Add nonzero z values to gsd box for testing."""
        new_box = list(gsd_snap.configuration.box)
        new_box[2] = 1e2 * (np.random.random() - 0.5)
        new_box[4] = 1e2 * (np.random.random() - 0.5)
        new_box[5] = 1e2 * (np.random.random() - 0.5)
        gsd_snap.configuration.box = new_box
        return gsd_snap

    d = tmp_path / "sub"
    d.mkdir()
    filename = d / "temporary_test_file.gsd"
    if device.communicator.rank == 0:
        f = gsd.hoomd.open(name=filename, mode='w')

    sim = simulation_factory(
        lattice_snapshot_factory(n=10, particle_types=["A", "B"], dimensions=2))
    snap = sim.state.get_snapshot()

    checks = range(3)
    if device.communicator.rank == 0:
        for step in checks:
            f.append(modify_gsd_snap(make_gsd_frame(snap)))
        f.close()

    for step in checks:
        sim = simulation_factory()
        sim.create_state_from_gsd(filename, frame=step)
        assert sim.state.box.dimensions == 2
        assert sim.state.box.Lz == 0.0
        assert sim.state.box.xz == 0.0
        assert sim.state.box.yz == 0.0


@skip_gsd
def test_state_from_gsd_frame(simulation_factory, lattice_snapshot_factory,
                              device, state_args, tmp_path):
    snap_params, nsteps = state_args

    sim = simulation_factory(
        lattice_snapshot_factory(n=snap_params[0],
                                 particle_types=snap_params[1]))
    snap = sim.state.get_snapshot()
    snap = make_gsd_frame(snap)
    gsd_snapshot_list = [snap]
    box = sim.state.box
    for _ in range(1, nsteps):
        particle_type = np.random.choice(snap_params[1])
        snap = update_positions(sim.state.get_snapshot())
        set_types(snap, random_inds(snap_params[0]), snap_params[1],
                  particle_type)
        snap = make_gsd_frame(snap)
        gsd_snapshot_list.append(snap)

    for snap in gsd_snapshot_list:
        sim = hoomd.Simulation(device)
        sim.create_state_from_snapshot(snap)
        assert box == sim.state.box
        assert_equivalent_snapshots(snap, sim.state.get_snapshot())


def test_writer_order(simulation_factory, two_particle_snapshot_factory):
    """Ensure that writers run at the end of the loop step."""

    class StepRecorder(hoomd.custom.Action):

        def __init__(self):
            self.steps = []

        def act(self, timestep):
            self.steps.append(timestep)

    record = StepRecorder()
    periodic = hoomd.trigger.Periodic(period=100, phase=0)
    writer = hoomd.write.CustomWriter(action=record, trigger=periodic)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.writers.append(writer)

    sim.run(500)
    assert record.steps == [100, 200, 300, 400, 500]
    sim.run(500)
    assert record.steps == [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def test_writer_order_initial(simulation_factory,
                              two_particle_snapshot_factory):
    """Ensure that writers optionally run at the beginning of the loop."""

    class StepRecorder(hoomd.custom.Action):

        def __init__(self):
            self.steps = []

        def act(self, timestep):
            self.steps.append(timestep)

    record = StepRecorder()
    periodic = hoomd.trigger.Periodic(period=100, phase=0)
    writer = hoomd.write.CustomWriter(action=record, trigger=periodic)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.writers.append(writer)

    sim.run(500, write_at_start=True)
    assert record.steps == [0, 100, 200, 300, 400, 500]
    sim.run(500, write_at_start=True)
    assert record.steps == [
        0,
        100,
        200,
        300,
        400,
        500,
        500,
        600,
        700,
        800,
        900,
        1000,
    ]


def test_large_timestep(simulation_factory, lattice_snapshot_factory):
    """Test that simluations suport large timestep values."""
    sim = simulation_factory()
    sim.timestep = 2**64 - 100
    sim.create_state_from_snapshot(lattice_snapshot_factory())

    sim.run(99)
    assert sim.timestep == 2**64 - 1


def test_timestep_wrap(simulation_factory, lattice_snapshot_factory):
    """Test that time steps wrap around."""
    sim = simulation_factory()
    sim.timestep = 2**64 - 100
    sim.create_state_from_snapshot(lattice_snapshot_factory())

    sim.run(200)
    assert sim.timestep == 100


def test_initial_timestep_range(simulation_factory):
    """Test that the initial timestep cannot be set out of range."""
    sim = simulation_factory()

    with pytest.raises(ValueError):
        sim.timestep = -1

    with pytest.raises(ValueError):
        sim.timestep = 2**64

    sim.timestep = 2**64 - 1


def test_run_limit(simulation_factory, lattice_snapshot_factory):
    """Test limits of Simulation.run()."""
    sim = simulation_factory(lattice_snapshot_factory())

    with pytest.raises(ValueError):
        sim.run(2**64 - 1)

    with pytest.raises(ValueError):
        sim.run(-1)


def test_seed(device, lattice_snapshot_factory):

    sim = hoomd.Simulation(device)
    assert sim.seed is None

    sim.seed = 42
    assert sim.seed == 42

    sim.seed = 0x123456789abcdef
    assert sim.seed == 0xcdef

    sim.create_state_from_snapshot(lattice_snapshot_factory())
    assert sim.seed == 0xcdef

    sim.seed = 20
    assert sim.seed == 20


def test_seed_constructor_out_of_range(device, lattice_snapshot_factory):
    sim = hoomd.Simulation(device, seed=0x123456789abcdef)

    sim.create_state_from_snapshot(lattice_snapshot_factory())
    assert sim.seed == 0xcdef


def test_operations_setting(tmp_path, simulation_factory,
                            lattice_snapshot_factory):
    sim = simulation_factory()
    sim.create_state_from_snapshot(lattice_snapshot_factory())

    def check_operation_setting(sim, old_operations, new_operations):
        assert sim.operations is not new_operations
        assert new_operations._simulation is None
        assert old_operations._simulation is sim
        scheduled = old_operations._scheduled
        sim.operations = new_operations
        assert sim.operations is new_operations
        assert new_operations._simulation is sim
        assert old_operations._simulation is None
        if scheduled:
            assert new_operations._scheduled
            assert not old_operations._scheduled

    operations = hoomd.Operations()
    # Add some operations to test the setting
    operations += hoomd.update.BoxResize(trigger=40,
                                         box=hoomd.variant.box.Interpolate(hoomd.Box.cube(10),
                                         hoomd.Box.cube(20),
                                         hoomd.variant.Ramp(
                                            0, 1, 0, 100)))
    operations += hoomd.write.GSD(filename=tmp_path / "foo.gsd", trigger=10)
    operations += hoomd.write.Table(10, logger=hoomd.logging.Logger(['scalar']))
    operations.tuners.clear()
    # Check setting before scheduling
    check_operation_setting(sim, sim.operations, operations)

    sim.run(0)
    # Check setting after scheduling
    new_operations = hoomd.Operations()
    new_operations += hoomd.update.BoxResize(trigger=80,
                                             box=hoomd.variant.box.Interpolate(hoomd.Box.cube(300),
                                             hoomd.Box.cube(20),
                                             hoomd.variant.Ramp(
                                                 0, 1, 0, 100)))
    new_operations += hoomd.write.GSD(filename=tmp_path / "bar.gsd", trigger=20)
    new_operations += hoomd.write.Table(20,
                                        logger=hoomd.logging.Logger(['scalar']))
    check_operation_setting(sim, sim.operations, new_operations)


def test_mutability_error(simulation_factory, two_particle_snapshot_factory,
                          tmp_path):
    filt = hoomd.filter.All()
    sim = simulation_factory(two_particle_snapshot_factory())
    trig = hoomd.trigger.Periodic(1)

    filename = tmp_path / "temporary_test_file.gsd"
    GSD_dump = hoomd.write.GSD(filename=filename, trigger=trig)
    sim.operations.add(GSD_dump)
    sim.run(0)

    with pytest.raises(MutabilityError):
        GSD_dump.filter = filt


def test_logging():
    logging_check(
        hoomd.Simulation, (), {
            'final_timestep': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'seed': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'timestep': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'tps': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'walltime': {
                'category': LoggerCategories.scalar,
                'default': True
            }
        })
