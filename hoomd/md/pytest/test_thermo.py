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
        assert getattr(thermo, qty) is None

    # make simulation and test state of operations
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.add(thermo)
    assert len(sim.operations.computes) == 1
    sim.operations.schedule()

    # make sure quantities are computable without failure
    for qty, typ in _thermo_qtys:
        calc_qty = getattr(thermo, qty)
        assert type(calc_qty) == typ

    # detach from simulation and test properties again
    sim.operations.remove(thermo)
    assert len(sim.operations.computes) == 0
    for qty, typ in _thermo_qtys:
        assert getattr(thermo, qty) is None


def _assert_thermo_properties(thermo, npart, rdof, tdof, pe, rke, tke, ke, p, pt):

    assert thermo.num_particles == npart
    assert thermo.rotational_degrees_of_freedom == rdof
    assert thermo.translational_degrees_of_freedom == tdof
    assert thermo.degrees_of_freedom == (thermo.translational_degrees_of_freedom +
                                        thermo.rotational_degrees_of_freedom)

    np.testing.assert_allclose(thermo.potential_energy, pe)
    np.testing.assert_allclose(thermo.rotational_kinetic_energy, rke, rtol=1e-5)
    np.testing.assert_allclose(thermo.translational_kinetic_energy, tke, rtol=1e-5)
    np.testing.assert_allclose(thermo.kinetic_energy, ke, rtol=1e-5)
    np.testing.assert_allclose(thermo.kinetic_temperature, 2*thermo.kinetic_energy/thermo.degrees_of_freedom, rtol=1e-5)
    np.testing.assert_allclose(thermo.pressure, p, rtol=1e-5)
    np.testing.assert_allclose(thermo.pressure_tensor, pt, rtol=1e-5, atol=5e-5)


def test_basic_system_3d(simulation_factory, two_particle_snapshot_factory):
    filt = hoomd.filter.All()
    thermo = hoomd.md.compute.ThermodynamicQuantities(filt)
    snap = two_particle_snapshot_factory()
    if snap.exists:
        snap.particles.velocity[:] = [[-2, 0, 0], [2, 0, 0]]
    sim = simulation_factory(snap)
    sim.always_compute_pressure = True
    sim.operations.add(thermo)

    integrator = hoomd.md.Integrator(dt=0.0001)
    integrator.methods.append(hoomd.md.methods.NVT(filt, tau=1, kT=1))
    sim.operations.integrator = integrator

    sim.run(1)

    _assert_thermo_properties(thermo, 2, 0, 3, 0.0, 0.0, 4.0, 4.0,
                              2.0/3*thermo.kinetic_energy/20**3,
                              [8.0/20.0**3, 0., 0., 0., 0., 0.])


def test_basic_system_2d(simulation_factory, lattice_snapshot_factory):
    filterA = hoomd.filter.Type(['A'])
    filterB = hoomd.filter.Type(['B'])
    thermoA = hoomd.md.compute.ThermodynamicQuantities(filterA)
    thermoB = hoomd.md.compute.ThermodynamicQuantities(filterB)
    snap = lattice_snapshot_factory(particle_types=['A', 'B'], dimensions=2, n=2)
    if snap.exists:
        snap.particles.velocity[:] = [[-1, 0, 0], [2, 0, 0]]*2
        snap.particles.typeid[:] = [0, 1, 0, 1]
    sim = simulation_factory(snap)
    sim.always_compute_pressure = True
    sim.operations.add(thermoA)
    sim.operations.add(thermoB)

    integrator = hoomd.md.Integrator(dt=0.0001)
    integrator.methods.append(hoomd.md.methods.NVT(filterA, tau=1, kT=1))
    integrator.methods.append(hoomd.md.methods.Langevin(filterB, kT=1, seed=3, alpha=0.00001))
    sim.operations.integrator = integrator

    sim.run(1)

    # tests for group A
    _assert_thermo_properties(thermoA, 2, 0, 4, 0.0, 0.0, 1.0, 1.0,
                              thermoA.kinetic_energy/2.0**2,
                              (2.0/2.0**2, 0., 0., 0., 0., 0.))

    # tests for group B
    _assert_thermo_properties(thermoB, 2, 0, 4, 0.0, 0.0, 4.0, 4.0,
                              thermoB.kinetic_energy/2.0**2,
                              (8.0/2.0**2, 0., 0., 0., 0., 0.))


def test_system_rotational_dof(simulation_factory, device):

    snap = hoomd.Snapshot(device.communicator)
    if snap.exists:
        box= [10, 10, 10, 0, 0, 0]
        snap.configuration.box = box
        snap.configuration.dimensions = 3
        snap.particles.N = 3
        snap.particles.position[:] = [[0, 1, 0], [-1, 1, 0], [1, 1, 0]]
        snap.particles.velocity[:] = [[0, 0, 0], [0, -1, 0], [0, 1, 0]]
        snap.particles.moment_inertia[:] = [[2.0, 0, 0], [1, 1, 1], [1, 1, 1]]
        snap.particles.angmom[:] = [[0, 2, 4, 6]] * 3
        snap.particles.types = ['A']

    filt = hoomd.filter.All()
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=filt)
    sim = simulation_factory(snap)
    sim.always_compute_pressure = True
    sim.operations.add(thermo)

    integrator = hoomd.md.Integrator(dt=0.0001)
    integrator.aniso = True
    integrator.methods.append(hoomd.md.methods.NVT(filt, tau=1, kT=1))
    sim.operations.integrator = integrator

    sim.run(1)

    _assert_thermo_properties(thermo, 3, 7, 6, 0.0, 57/4., 1.0, 61/4.,
                              2./3*thermo.translational_kinetic_energy/10.0**3,
                              (0., 0., 0., 2./10**3, 0., 0.))

