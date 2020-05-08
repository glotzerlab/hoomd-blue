import hoomd
import numpy
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
            s.particles.position[:] = numpy.random.uniform(-10, 10, size=(n, 3))
            s.particles.types = particle_types
        return s
    return make_snapshot


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


class FileContext():
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        os.remove(self.filename)


def test_state_from_gsd(simulation_factory, get_snapshot, device, state_args):
    filename = 'test_file.gsd'

    with FileContext(filename) as file:
        snap_params, integrator, shape_dict, run_sequence = state_args
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
        pos_dict = {}
        initial_pos = sim.state.snapshot.particles.position
        count = 0
        for nsteps in run_sequence:
            sim.run(nsteps)
            count += nsteps
            pos_dict[count] = sim.state.snapshot.particles.position

        final_pos = sim.state.snapshot.particles.position
        sim.run(1)

        initial_pos_sim = hoomd.simulation.Simulation(device)
        initial_pos_sim.create_state_from_gsd(file.filename, frame=0)
        initial_pos_snap = initial_pos_sim.state.snapshot
        numpy.testing.assert_allclose(initial_pos,
                                      initial_pos_snap.particles.position)

        final_pos_sim = hoomd.simulation.Simulation(device)
        final_pos_sim.create_state_from_gsd(file.filename)
        final_pos_snap = final_pos_sim.state.snapshot
        numpy.testing.assert_allclose(final_pos,
                                      final_pos_snap.particles.position)

        for nsteps, positions in pos_dict.items():
            sim = hoomd.simulation.Simulation(device)
            sim.create_state_from_gsd(file.filename, frame=nsteps)
            snap = sim.state.snapshot
            numpy.testing.assert_allclose(positions, snap.particles.position)
