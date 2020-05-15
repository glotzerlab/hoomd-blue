import hoomd
import numpy as np
import pytest
from copy import deepcopy
try:
    import gsd.hoomd
    skip_gsd = False
except ImportError:
    skip_gsd = True
skip_gsd = pytest.mark.skipif(skip_gsd, reason="gsd Python package was not found.")

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


def assert_equivalent_snapshots(snap1, snap2):
    for attr in dir(snap2):
        if attr[0] != '_' and attr not in ['exists', 'replicate']:
            for prop in dir(getattr(snap2, attr)):
                if prop[0] != '_':
                    if prop == 'types':
                        assert getattr(getattr(snap1, attr), prop) == \
                            getattr(getattr(snap2, attr), prop)
                    else:
                        np.testing.assert_allclose(getattr(getattr(snap1, attr),
                                                           prop),
                                                   getattr(getattr(snap2, attr),
                                                           prop))


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
        sim = hoomd.simulation.Simulation()

    sim = hoomd.simulation.Simulation(device)
    with pytest.raises(RuntimeError):
        sim.run(1)  # Before setting state

    sim = simulation_factory(get_snapshot())
    with pytest.raises(RuntimeError):
        sim.run(1)  # Before scheduling operations

    sim.operations.schedule()
    sim.run(1)


def test_run(simulation_factory, get_snapshot):
    sim = simulation_factory(get_snapshot())
    sim.operations.schedule()
    assert sim.timestep == 0
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
        box = sim.state.box

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
        sim = hoomd.simulation.Simulation(device)
        sim.create_state_from_gsd(filename, frame=step)
        assert_equivalent_boxes(box, sim.state.box)
        assert_equivalent_snapshots(snap, sim.state.snapshot)
