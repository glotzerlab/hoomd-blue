import hoomd
import numpy
import pytest
import os


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


def test_state_from_gsd(simulation_factory, get_snapshot, device):
    filename = 'test_file.gsd'
    sim = simulation_factory(get_snapshot())
    mc = hoomd.hpmc.integrate.Sphere(2345)
    mc.shape['A'] = dict(diameter=1)

    sim.operations.add(mc)
    gsd_dumper = hoomd.dump.GSD(filename=filename, trigger=1, overwrite=True)
    gsd_logger = hoomd.logger.Logger()
    gsd_logger += mc
    gsd_dumper.log = gsd_logger
    sim.operations.add(gsd_dumper)
    sim.operations.schedule()

    initial_pos = sim.state.snapshot.particles.position
    sim.run(10)
    final_pos = sim.state.snapshot.particles.position
    sim.run(1)

    initial_pos_sim = hoomd.simulation.Simulation(device)
    initial_pos_sim.create_state_from_gsd(filename, frame=0)

    final_pos_sim = hoomd.simulation.Simulation(device)
    final_pos_sim.create_state_from_gsd(filename)

    initial_pos_snap = initial_pos_sim.state.snapshot
    final_pos_snap = final_pos_sim.state.snapshot
    numpy.testing.assert_allclose(initial_pos,
                                  initial_pos_snap.particles.position)
    numpy.testing.assert_allclose(final_pos,
                                  final_pos_snap.particles.position)

    os.remove(filename)
