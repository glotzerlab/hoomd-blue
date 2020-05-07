from hoomd.snapshot import Snapshot
from hoomd.simulation import Simulation
import numpy
import pytest


@pytest.fixture(scope="function")
def get_snapshot(device):
    def make_snapshot(n=10, particle_types=['A']):
        s = Snapshot(device.comm)
        if s.exists:
            s.configuration.box = [20, 20, 20, 0, 0, 0]
            s.particles.N = n
            s.particles.position[:] = numpy.random.uniform(-10, 10, size=(n, 3))
            s.particles.types = particle_types
        return s
    return make_snapshot


def test_initialization(device, simulation_factory, get_snapshot):
    with pytest.raises(TypeError):
        sim = Simulation()

    sim = Simulation(device)
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
