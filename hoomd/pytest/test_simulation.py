import hoomd
import numpy as np
import pytest
import os
from copy import deepcopy


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


def assert_equivalent_snapshots(snap1, snap2):
    particles1 = snap1.particles
    particles2 = snap2.particles
    particle_data = [(particles1.N, particles2.N),
                     (particles1.typeid, particles2.typeid),
                     (particles1.mass, particles2.mass),
                     (particles1.diameter, particles2.diameter),
                     (particles1.charge, particles2.charge),
                     (particles1.position, particles2.position),
                     (particles1.orientation, particles2.orientation),
                     (particles1.velocity, particles2.velocity),
                     (particles1.acceleration, particles2.acceleration),
                     (particles1.image, particles2.image),
                     (particles1.body, particles2.body),
                     (particles1.moment_inertia, particles2.moment_inertia),
                     (particles1.angmom, particles2.angmom)]

    assert particles1.types == particles2.types
    for snap_data1, snap_data2 in particle_data:
        np.testing.assert_allclose(snap_data1, snap_data2)


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

_tetrahedron_verts = [(0.5, 0.5, 0.5), (-0.5, -0.5, 0.5),
                      (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5)]

_state_args = [((10, ['A']),
                hoomd.hpmc.integrate.Sphere,
                {'diameter': 1}, [10]),
               ((100, ['A']),
                hoomd.hpmc.integrate.ConvexPolyhedron,
                {'vertices': _tetrahedron_verts}, [10, 3]),
               ((5, ['A']),
                hoomd.hpmc.integrate.Ellipsoid,
                {'a': 0.2, 'b': 0.25, 'c': 0.5}, [1, 3, 4]),
               ((50, ['A']), hoomd.hpmc.integrate.ConvexSpheropolyhedron,
                {'vertices': _tetrahedron_verts, 'sweep_radius': 0.1}, [10])]


@pytest.fixture(scope="function", params=_state_args)
def state_args(request):
    return deepcopy(request.param)


class TemporaryFileContext():
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        os.remove(self.filename)


def test_state_from_gsd(simulation_factory, get_snapshot, device, state_args):
    filename = 'temporary_test_file.gsd'
    snap_params, integrator, shape_dict, run_sequence = state_args

    with TemporaryFileContext(filename) as file:
        sim = simulation_factory(get_snapshot(n=snap_params[0],
                                              particle_types=snap_params[1]))
        mc = integrator(2345)
        mc.shape['A'] = shape_dict

        sim.operations.add(mc)
        gsd_dumper = hoomd.dump.GSD(filename=file.filename,
                                    trigger=1,
                                    overwrite=True)
        gsd_logger = hoomd.logger.Logger()
        gsd_logger += mc
        gsd_dumper.log = gsd_logger
        sim.operations.add(gsd_dumper)
        sim.operations.schedule()
        snapshot_dict = {}
        initial_snap = sim.state.snapshot

        count = 0
        for nsteps in run_sequence:
            sim.run(nsteps)
            count += nsteps
            snapshot_dict[count] = sim.state.snapshot

        final_snap = sim.state.snapshot
        sim.run(1)

        initial_snap_sim = hoomd.simulation.Simulation(device)
        initial_snap_sim.create_state_from_gsd(file.filename, frame=0)
        assert_equivalent_snapshots(initial_snap,
                                    initial_snap_sim.state.snapshot)

        final_snap_sim = hoomd.simulation.Simulation(device)
        final_snap_sim.create_state_from_gsd(file.filename)
        assert_equivalent_snapshots(final_snap, final_snap_sim.state.snapshot)

        for nsteps, snap in snapshot_dict.items():
            sim = hoomd.simulation.Simulation(device)
            sim.create_state_from_gsd(file.filename, frame=nsteps)
            snap = sim.state.snapshot
            assert_equivalent_snapshots(snap, sim.state.snapshot)
