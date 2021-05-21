import hoomd
from hoomd.conftest import pickling_check


def test_attributes():
    all_ = hoomd.filter.All()
    active = hoomd.md.force.Active(filter=all_, rotation_diff=0.01)

    assert active.filter is all_
    assert active.rotation_diff==0.01
    assert active.active_force['A'] == (1.0,0.0,0.0)
    assert active.active_torque['A'] == (0.0,0.0,0.0)
    assert active.manifold_constraint is None

    active.rotation_diff= 0.1
    assert active.rotation_diff==0.1
    active.active_force['A'] = (0.5,0.0,0.0)
    assert active.active_force['A'] == (0.5,0.0,0.0)

def test_attach(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(hoomd.md.force.Active(filter=hoomd.filter.All(), rotation_diff=0.01))
    sim.operations.integrator = integrator
    sim.run(0)


def test_pickling(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    active = hoomd.md.force.Active(
        filter=hoomd.filter.All(), rotation_diff=0.01)
    pickling_check(active)
    integrator = hoomd.md.Integrator(
        .05,
        methods=[hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0)],
        forces=[active]
    )
    sim.operations.integrator = integrator
    sim.run(0)
    pickling_check(active)
