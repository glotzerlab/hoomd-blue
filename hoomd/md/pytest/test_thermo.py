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
        print(type(calc_qty), typ)
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

    integrator = hoomd.md.Integrator(dt=0.0001)
    integrator.methods.append(hoomd.md.methods.NVT(group, tau=1, kT=1))
    sim.operations.integrator = integrator

    sim.operations.schedule()
    sim.run(1)

    assert thermo.num_particles == 2
    assert thermo.rotational_degrees_of_freedom == 0
    assert thermo.translational_degrees_of_freedom == 3
    assert thermo.degrees_of_freedom == 3

    np.testing.assert_allclose(thermo.potential_energy, 0.0)
    np.testing.assert_allclose(thermo.rotational_kinetic_energy, 0.0, rtol=1e-5)
    np.testing.assert_allclose(thermo.translational_kinetic_energy, 4.0, rtol=1e-5)
    np.testing.assert_allclose(thermo.kinetic_energy, 4.0, rtol=1e-5)
    np.testing.assert_allclose(thermo.kinetic_temperature, 2*thermo.kinetic_energy/thermo.degrees_of_freedom, rtol=1e-5)
    np.testing.assert_allclose(thermo.pressure, thermo.kinetic_energy/20**3, rtol=1e-5)
    (pxx, pxy, pxz, pyy, pyz, pzz) = thermo.pressure_tensor
    np.testing.assert_allclose(pxx, 8.0, rtol=1e-5)
    np.testing.assert_allclose(pxy, 0.0, rtol=1e-5)
    np.testing.assert_allclose(pxz, 0.0, rtol=1e-5)
    np.testing.assert_allclose(pyy, 0.0, rtol=1e-5)
    np.testing.assert_allclose(pyz, 0.0, rtol=1e-5)
    np.testing.assert_allclose(pzz, 0.0, rtol=1e-5)



def test_basic_system_2d(simulation_factory, lattice_snapshot_factory):
    group = hoomd.filter.Type(['A'])
    groupB = hoomd.filter.Type(['B'])
    thermo = hoomd.md.compute.ThermodynamicQuantities(group)
    thermoB = hoomd.md.compute.ThermodynamicQuantities(groupB)
    snap = lattice_snapshot_factory(particle_types=['A', 'B']*2, dimensions=2, n=2)
    if snap.exists:
        snap.particles.velocity[:] = [[-1, 0, 0], [2, 0, 0]]*2
    sim = simulation_factory(snap)
    sim.operations.add(thermo)
    sim.operations.add(thermoB)

    integrator = hoomd.md.Integrator(dt=0.0001)
    integrator.methods.append(hoomd.md.methods.NVT(group, tau=1, kT=1))
    integrator.methods.append(hoomd.md.methods.Langevin(groupB, kT=1, seed=3))
    sim.operations.integrator = integrator

    sim.operations.schedule()
    sim.run(1)

    # tests for group A
    assert thermo.num_particles == 2
    assert thermo.rotational_degrees_of_freedom == 0
    assert thermo.translational_degrees_of_freedom == 2
    assert thermo.degrees_of_freedom == 2
    np.testing.assert_allclose(thermo.potential_energy, 0.0)
    np.testing.assert_allclose(thermo.rotational_kinetic_energy, 0.0, rtol=1e-5)
    np.testing.assert_allclose(thermo.translational_kinetic_energy, 1.0, rtol=1e-5)
    np.testing.assert_allclose(thermo.kinetic_energy, 1.0, rtol=1e-5)
    np.testing.assert_allclose(thermo.kinetic_temperature, 2*thermo.kinetic_energy/thermo.degrees_of_freedom, rtol=1e-5)
    np.testing.assert_allclose(thermo.pressure, thermo.kinetic_energy/2.0**3, rtol=1e-5)
    (pxx, pxy, pxz, pyy, pyz, pzz) = thermo.pressure_tensor
    np.testing.assert_allclose(pxx, 2.0, rtol=1e-5)
    np.testing.assert_allclose(pxy, 0.0, rtol=1e-5)
    np.testing.assert_allclose(pxz, 0.0, rtol=1e-5)
    np.testing.assert_allclose(pyy, 0.0, rtol=1e-5)
    np.testing.assert_allclose(pyz, 0.0, rtol=1e-5)
    np.testing.assert_allclose(pzz, 0.0, rtol=1e-5)

    # tests for group B
    assert thermoB.num_particles == 2
    assert thermoB.rotational_degrees_of_freedom == 0
    assert thermoB.translational_degrees_of_freedom == 4
    assert thermoB.degrees_of_freedom == 4
    np.testing.assert_allclose(thermoB.potential_energy, 0.0)
    np.testing.assert_allclose(thermoB.rotational_kinetic_energy, 0.0, rtol=1e-5)
    np.testing.assert_allclose(thermoB.translational_kinetic_energy, 4.0, rtol=1e-5)
    np.testing.assert_allclose(thermoB.kinetic_energy, 4.0, rtol=1e-5)
    np.testing.assert_allclose(thermoB.kinetic_temperature, 2*thermoB.kinetic_energy/thermoB.degrees_of_freedom, rtol=1e-5)
    np.testing.assert_allclose(thermoB.pressure, thermoB.kinetic_energy/2.0**3, rtol=1e-5)
    (pxx, pxy, pxz, pyy, pyz, pzz) = thermoB.pressure_tensor
    np.testing.assert_allclose(pxx, 2.0, rtol=1e-5)
    np.testing.assert_allclose(pxy, 0.0, rtol=1e-5)
    np.testing.assert_allclose(pxz, 0.0, rtol=1e-5)
    np.testing.assert_allclose(pyy, 0.0, rtol=1e-5)
    np.testing.assert_allclose(pyz, 0.0, rtol=1e-5)
    np.testing.assert_allclose(pzz, 0.0, rtol=1e-5)

