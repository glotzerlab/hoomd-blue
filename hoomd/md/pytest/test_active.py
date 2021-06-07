import hoomd
from hoomd.conftest import pickling_check


def test_attach(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(
        hoomd.md.force.Active(filter=hoomd.filter.All(), rotation_diff=0.01))
    sim.operations.integrator = integrator
    sim.operations._schedule()
    sim.run(10)


def test_pickling(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    active = hoomd.md.force.Active(filter=hoomd.filter.All(),
                                   rotation_diff=0.01)
    pickling_check(active)
    integrator = hoomd.md.Integrator(
        .05,
        methods=[hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0)],
        forces=[active])
    sim.operations.integrator = integrator
    sim.run(0)
    pickling_check(active)
