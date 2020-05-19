import hoomd
import numpy as np
import pytest
from copy import deepcopy
try:
    import gsd.hoomd
    skip_gsd = False
except ImportError:
    skip_gsd = True

skip_gsd = pytest.mark.skipif(
    skip_gsd, reason="gsd Python package was not found.")


@pytest.fixture(scope="function")
def get_snapshot(device):
    def make_snapshot(n=10, particle_types=['A']):
        s = hoomd.snapshot.Snapshot(device.comm)
        if s.exists:
            s.configuration.box = [20, 20, 20, 0, 0, 0]
            s.particles.N = n
            s.particles.position[:] = np.random.uniform(-10, 10, size=(n, 3))
            s.particles.types = particle_types
        return s
    return make_snapshot


def make_gsd_snapshot(hoomd_snapshot):
    s = gsd.hoomd.Snapshot()
    for attr in dir(hoomd_snapshot):
        if attr[0] != '_' and attr not in ['exists', 'replicate']:
            for prop in dir(getattr(hoomd_snapshot, attr)):
                if prop[0] != '_':
                    # s.attr.prop = hoomd_snapshot.attr.prop
                    setattr(getattr(s, attr), prop,
                            getattr(getattr(hoomd_snapshot, attr), prop))
    return s


def set_types(s, inds, particle_types, particle_type):
    for i in inds:
        s.particles.typeid[i] = particle_types.index(particle_type)


def update_positions(snap):
    if snap.exists:
        noise = 0.01
        rs = np.random.RandomState(0)
        mean = [0] * 3
        var = noise * noise
        cov = np.diag([var, var, var])
        shape = snap.particles.position.shape
        snap.particles.position[:] += rs.multivariate_normal(mean, cov,
                                                             size=shape[:-1])
    return snap


def assert_equivalent_snapshots(gsd_snap, hoomd_snap):
    for attr in dir(hoomd_snap):
        if attr[0] == '_' or attr in ['exists', 'replicate']:
            continue
        for prop in dir(getattr(hoomd_snap, attr)):
            if prop[0] == '_':
                continue
            elif prop == 'types':
                if hoomd_snap.exists:
                    assert getattr(getattr(gsd_snap, attr), prop) == \
                        getattr(getattr(hoomd_snap, attr), prop)
            else:
                if hoomd_snap.exists:
                    np.testing.assert_allclose(
                        getattr(getattr(gsd_snap, attr), prop),
                        getattr(getattr(hoomd_snap, attr), prop)
                    )


def assert_equivalent_boxes(box1, box2):
    assert box1.Lx == box2.Lx
    assert box1.Ly == box2.Ly
    assert box1.Lz == box2.Lz
    assert box1.xy == box2.xy
    assert box1.xz == box2.xz
    assert box1.yz == box2.yz


def random_inds(n):
    return np.random.choice(np.arange(n),
                            size=int(n * np.random.rand()),
                            replace=False)


def test_initialization(device, simulation_factory, get_snapshot):
    with pytest.raises(TypeError):
        sim = hoomd.Simulation()

    sim = hoomd.Simulation(device)


def test_device_property(device):
    sim = hoomd.Simulation(device)
    assert sim.device is device

    with pytest.raises(ValueError):
        sim.device = device


def test_allows_compute_pressure(device, get_snapshot):
    sim = hoomd.Simulation(device)
    assert sim.always_compute_pressure is False
    with pytest.raises(RuntimeError):
        sim.always_compute_pressure = True
    sim.create_state_from_snapshot(get_snapshot())
    sim.always_compute_pressure = True
    assert sim.always_compute_pressure is True


def test_run(simulation_factory, get_snapshot, device):
    sim = hoomd.Simulation(device)
    with pytest.raises(RuntimeError):
        sim.run(1)  # Before setting state

    sim = simulation_factory(get_snapshot())
    sim.run(1)

    assert sim.operations.scheduled


def test_timestep(simulation_factory, get_snapshot, device):
    sim = hoomd.Simulation(device)
    assert sim.timestep is None

    initial_steps = 10
    sim.timestep = initial_steps
    assert sim.timestep == initial_steps
    sim.create_state_from_snapshot(get_snapshot())
    assert sim.timestep == initial_steps

    with pytest.raises(RuntimeError):
        sim.timestep = 20


def test_run_with_timestep(simulation_factory, get_snapshot, device):
    sim = hoomd.Simulation(device)
    sim.create_state_from_snapshot(get_snapshot())
    sim.operations.schedule()

    steps = 0
    n_step_list = [1, 10, 100]
    for n_steps in n_step_list:
        steps += n_steps
        sim.run(n_steps)
        assert sim.timestep == steps

    assert sim.timestep == sum(n_step_list)


_state_args = [((10, ['A']), 10),
               ((5, ['A', 'B']), 20),
               ((50, ['A', 'B', 'C']), 4),
               ((100, ['A', 'B']), 30)]


@pytest.fixture(scope="function", params=_state_args)
def state_args(request):
    return deepcopy(request.param)


@skip_gsd
def test_state_from_gsd(simulation_factory, get_snapshot,
                        device, state_args, tmp_path):
    snap_params, nsteps = state_args

    d = tmp_path / "sub"
    d.mkdir()
    filename = d / "temporary_test_file.gsd"
    with gsd.hoomd.open(name=filename, mode='wb+') as file:
        sim = simulation_factory(get_snapshot(n=snap_params[0],
                                              particle_types=snap_params[1]))
        snap = sim.state.snapshot
        snapshot_dict = {}
        snapshot_dict[0] = snap
        file.append(make_gsd_snapshot(snap))
        box = sim.state.box
        for step in range(1, nsteps):
            particle_type = np.random.choice(snap_params[1])
            snap = update_positions(sim.state.snapshot)
            set_types(snap, random_inds(snap.particles.N),
                      snap_params[1], particle_type)
            file.append(make_gsd_snapshot(snap))
            snapshot_dict[step] = snap

    for step, snap in snapshot_dict.items():
        sim = hoomd.Simulation(device)
        sim.create_state_from_gsd(filename, frame=step)
        assert_equivalent_boxes(box, sim.state.box)
        assert_equivalent_snapshots(snap, sim.state.snapshot)
