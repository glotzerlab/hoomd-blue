# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.external.field."""

import hoomd
import pytest
import numpy as np

valid_constructor_args = [
    dict(reference_positions=[[0, 0, 0]],
         reference_orientations=[[1, 0, 0, 0]],
         k_translational=hoomd.variant.Cycle(1, 5, 0, 10, 20, 10, 15),
         k_rotational=hoomd.variant.Cycle(1, 5, 0, 10, 20, 10, 15),
         symmetries=[[1, 0, 0, 0]]),
    dict(reference_positions=[[0, 0, 0]],
         reference_orientations=[[1, 0, 0, 0]],
         k_translational=hoomd.variant.Power(1, 5, 3, 0, 100),
         k_rotational=hoomd.variant.Power(1, 5, 3, 0, 100),
         symmetries=[[1, 0, 0, 0]]),
    dict(reference_positions=[[0, 0, 0]],
         reference_orientations=[[1, 0, 0, 0]],
         k_translational=hoomd.variant.Ramp(100, 0, 0, 1000),
         k_rotational=hoomd.variant.Ramp(10, 0, 0, 500),
         symmetries=[[1, 0, 0, 0]]),
    dict(reference_positions=[[0, 0, 0]],
         reference_orientations=[[1, 0, 0, 0]],
         k_translational=hoomd.variant.Constant(10),
         k_rotational=hoomd.variant.Constant(0),
         symmetries=[[1, 0, 0, 0]]),
]


@pytest.mark.cpu
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_harmonicfield(device, constructor_args):
    """Test that HarmonicField can be constructed with valid arguments."""
    field = hoomd.hpmc.external.field.Harmonic(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert np.all(getattr(field, attr) == value)


@pytest.fixture(scope="module")
def add_default_integrator():

    def add(simulation):
        mc = hoomd.hpmc.integrate.Sphere()
        mc.shape['A'] = dict(diameter=0)
        snapshot = simulation.state.get_snapshot()
        if simulation.device.communicator.rank == 0:
            reference_positions = snapshot.particles.position
            reference_orientations = snapshot.particles.orientation
        else:
            reference_positions = [[0, 0, 0], [0, 0, 0]]
            reference_orientations = [[1, 0, 0, 0], [1, 0, 0, 0]]
        lattice = hoomd.hpmc.external.field.Harmonic(
            reference_positions=reference_positions,
            reference_orientations=reference_orientations,
            k_translational=1.0,
            k_rotational=1.0,
            symmetries=[[1, 0, 0, 0]])
        mc.external_potential = lattice
        simulation.operations.integrator = mc
        return mc, lattice

    return add


@pytest.mark.cpu
def test_attaching(simulation_factory, two_particle_snapshot_factory,
                   add_default_integrator):
    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    mc, lattice = add_default_integrator(sim)

    # create C++ mirror classes and set parameters
    sim.run(0)

    # make sure objecst are attached
    assert mc._attached
    assert lattice._attached


@pytest.mark.cpu
def test_detaching(simulation_factory, two_particle_snapshot_factory,
                   add_default_integrator):
    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    mc, lattice = add_default_integrator(sim)

    # create C++ mirror classes and set parameters
    sim.run(0)

    # make sure objecst are attached
    sim.operations.remove(mc)
    assert not mc._attached
    assert not lattice._attached


@pytest.mark.cpu
def test_harmonic_displacement_energy(simulation_factory,
                                      two_particle_snapshot_factory,
                                      add_default_integrator):
    """Ensure harmonic displacements result in expected energy."""
    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    mc, lattice = add_default_integrator(sim)
    mc.shape['A'] = dict(diameter=0, orientable=True)

    dx = 0.01
    disp = np.array([dx, 0, 0])
    lattice.reference_positions = lattice.reference_positions + disp

    # run and check energy
    sim.run(0)
    k_translational = lattice.k_translational(sim.timestep)
    assert np.allclose(lattice.energy,
                       0.5 * dx**2 * k_translational * sim.state.N_particles)

    # make some moves and make sure the different energies are not zero
    sim.run(10)
    assert lattice.energy_rotational != 0.0
    assert lattice.energy_translational != 0.0
    assert np.allclose(lattice.energy,
                       lattice.energy_translational + lattice.energy_rotational)

    # set k_rotational to zero and ensure the rotational energy is zero
    lattice.k_rotational = 0
    assert lattice.energy_rotational == 0.0

    # set k_translational to zero and ensure the translational energy is zero
    lattice.k_translational = 0
    assert lattice.energy_translational == 0.0


@pytest.mark.cpu
def test_harmonic_displacement(simulation_factory,
                               two_particle_snapshot_factory,
                               add_default_integrator):
    """Ensure particles remain close to reference positions."""
    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    mc, lattice = add_default_integrator(sim)
    particle_diameter = 0.5
    mc.shape['A'] = dict(diameter=particle_diameter)
    k_trans = 100.0
    lattice.k_translational = k_trans

    # run and check that particles haven't moved farther than half a diameter
    sim.run(100)
    snapshot = sim.state.get_snapshot()
    if snapshot.communicator.rank == 0:
        new_positions = snapshot.particles.position
        dx = np.linalg.norm(new_positions - lattice.reference_positions, axis=1)
        assert np.all(np.less(dx, particle_diameter / 2))
