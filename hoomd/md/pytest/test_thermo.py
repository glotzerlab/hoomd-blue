import hoomd
import numpy as np

""" Each entry is a quantity and its type """
_thermo_qtys = [
    ('kinetic_temperature', float),
    ('pressure', float),
    ('pressure_tensor', list),
    ('kinetic_energy', float),
    ('translational_kinetic_energy', float),
    ('rotational_kinetic_energy', float),
    ('potential_energy', float),
    ('degrees_of_freedom', int),
    ('translational_degrees_of_freedom', int),
    ('rotational_degrees_of_freedom', int),
    ('num_particles', int),
]

def test_attach_detach(simulation_factory, two_particle_snapshot_factory):
    # test before attaching to simulation
    group = hoomd.filter.All()
    thermo = hoomd.md.compute.ThermodynamicQuantities(group)
    for qty, typ in _thermo_qtys:
        assert getattr(thermo, qty) == None

    # make simulation and test state of operations
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.add(thermo)
    assert len(sim.operations.computes) == 1
    sim.operations.schedule()

    # make sure quantities are computable without failure
    for qty, typ in _thermo_qtys:
        calc_qty = getattr(thermo, qty)
        print(qty, typ)
        assert type(calc_qty) == typ

    # detach from simulation and test properties again
    sim.operations.remove(thermo)
    assert len(sim.operations.computes) == 0
    for qty, typ in _thermo_qtys:
        assert getattr(thermo, qty) == None

def test_basic_system_3d(simulation_factory, two_particle_snapshot_factory):
    group = hoomd.filter.All()
    thermo = hoomd.md.compute.ThermodynamicQuantities(group)
    snap = two_particle_snapshot_factory()
    if snap.exists:
        snap.particles.velocity[:] = [[-2, 0, 0], [2, 0, 0]]
    sim = simulation_factory(snap)
    sim.operations.add(thermo)

    integrator = hoomd.md.Integrator(dt=0.001)
    integrator.methods.append(hoomd.md.methods.NVT(group, tau=1, kT=1))
    sim.operations.integrator = integrator

    sim.operations.schedule()
    sim.run(1)

    assert thermo.num_particles == 2
    assert thermo.rotational_degrees_of_freedom == 0
    assert thermo.translational_degrees_of_freedom == 3
    assert thermo.degrees_of_freedom == 3

    np.testing.assert_allclose(thermo.potential_energy, 0.0)
    np.testing.assert_allclose(thermo.translational_kinetic_energy, 4.0, rtol=1e-5)
    np.testing.assert_allclose(thermo.kinetic_energy, 4.0, rtol=1e-5)
    np.testing.assert_allclose(thermo.kinetic_temperature, 2/3*4.0, rtol=1e-5)


def test_basic_system_2d(simulation_factory, lattice_snapshot_factory):
    group = hoomd.filter.Type('A')
    thermo = hoomd.md.compute.ThermodynamicQuantities(group)
    snap = lattice_snapshot_factory(particle_types=['A', 'B']*2, dimensions=2, n=2)
    if snap.exists:
        snap.particles.velocity[:] = [[-1, 0, 0], [2, 0, 0]]*2
    sim = simulation_factory(snap)
    sim.operations.add(thermo)

    integrator = hoomd.md.Integrator(dt=0.001)
    integrator.methods.append(hoomd.md.methods.NVT(group, tau=1, kT=1))
    sim.operations.integrator = integrator

    sim.operations.schedule()
    sim.run(1)

    assert thermo.num_particles == 2
    assert thermo.rotational_degrees_of_freedom == 0
    assert thermo.translational_degrees_of_freedom == 4
    assert thermo.degrees_of_freedom == 4

    np.testing.assert_allclose(thermo.potential_energy, 0.0)
    np.testing.assert_allclose(thermo.translational_kinetic_energy, 1.0, rtol=1e-5)
    np.testing.assert_allclose(thermo.kinetic_energy, 1.0, rtol=1e-5)
    np.testing.assert_allclose(thermo.kinetic_temperature, 2/4*1.0, rtol=1e-5)

