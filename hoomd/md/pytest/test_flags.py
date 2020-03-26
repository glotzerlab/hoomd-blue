import hoomd
import pytest
import numpy
import itertools


@pytest.fixture(scope='session')
def md_simulation_factory(device):
    def make_simulation(particle_types=['A']):
        s = hoomd.Snapshot(device.comm)

        # initialize particles on a simple cubic lattice and perturb them
        # slightly off the lattice positions
        # TODO: When hpmc_integrator_api is merged, port this functionality
        # over to the fixture in conftest.py
        l = 20
        a = 2**(1.0 / 6.0)
        r = a * 0.01

        if s.exists:
            s.configuration.box = [l * a, l * a, l * a, 0, 0, 0]

            N = l**3
            s.particles.N = N
            pos = numpy.array(list(itertools.product(range(-l // 2, l // 2),
                                                        repeat=3)))
            s.particles.position[:] = pos
            s.particles.position[:] += [a / 2, a / 2, a / 2]
            s.particles.position[:] += numpy.random.uniform(-r, r, size=(N, 3))
            s.particles.types = particle_types

        sim = hoomd.Simulation(device)

        # reduce sorter grid to avoid Hilbert curve overhead in unit tests
        for tuner in sim.operations.tuners:
            if isinstance(tuner, hoomd.tuner.ParticleSorter):
                tuner.grid = 8

        sim.create_state_from_snapshot(s)
        return sim

    return make_simulation


def test_per_particle_virial(md_simulation_factory):
    cell = hoomd.md.nlist.Cell()
    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[('A', 'A')] = dict(sigma=1.0, epsilon=1.0)
    lj.r_cut[('A', 'A')] = 2.5

    sim = md_simulation_factory()

    assert sim.always_compute_pressure == False

    # TODO: allow forces to be added directly
    # sim.operations.add(lj)

    # For now, add to an integrator
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005)
    sim.operations.integrator.forces.append(lj)

    sim.operations.schedule()

    assert sim.always_compute_pressure == False

    # virials should be None at first
    virials = lj.virials

    if sim.device.comm.rank == 0:
        assert virials is None

    # virials should be non-zero after setting flags
    sim.always_compute_pressure = True

    virials = lj.virials

    if sim.device.comm.rank == 0:
        assert numpy.sum(virials * virials) > 0.0


# TODO: test compute thermo once it is implemented
