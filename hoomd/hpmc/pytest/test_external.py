# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test hoomd.hpmc.field."""

import hoomd
import pytest
import numpy as np

valid_constructor_args = [
    dict(reference_positions=[[0, 0, 0]],
         reference_orientations=[[1, 0, 0, 0]],
         k_translational=1.0,
         k_rotational=1.0,
         symmetries=[[1, 0, 0, 0]]),
]


@pytest.mark.cpu
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_harmonicfield(device, constructor_args):
    """Test that HarmonicField can be constructed with valid arguments."""
    field = hoomd.hpmc.field.Harmonic(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert np.all(getattr(field, attr) == value)


@pytest.mark.cpu
def test_attaching(device, simulation_factory, two_particle_snapshot_factory):
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # create field
    snapshot = sim.state.get_snapshot()
    if device.communicator.rank == 0:
        reference_positions = snapshot.particles.position
        reference_orientations = snapshot.particles.orientation
    else:
        reference_positions = [[0, 0, 0], [0, 0, 0]]
        reference_orientations = [[1, 0, 0, 0], [1, 0, 0, 0]]
    lattice = hoomd.hpmc.field.Harmonic(
        reference_positions=reference_positions,
        reference_orientations=reference_orientations,
        k_translational=1.0,
        k_rotational=1.0,
        symmetries=[[1, 0, 0, 0]])
    mc.external_potential = lattice

    # create C++ mirror classes and set parameters
    sim.run(0)

    # make sure objecst are attached
    assert mc._attached
    assert lattice._attached


@pytest.mark.cpu
def test_detaching(device, simulation_factory, two_particle_snapshot_factory):
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # create harmonic field
    snapshot = sim.state.get_snapshot()
    if device.communicator.rank == 0:
        reference_positions = snapshot.particles.position
        reference_orientations = snapshot.particles.orientation
    else:
        reference_positions = [[0, 0, 0], [0, 0, 0]]
        reference_orientations = [[1, 0, 0, 0], [1, 0, 0, 0]]
    lattice = hoomd.hpmc.field.Harmonic(
        reference_positions=reference_positions,
        reference_orientations=reference_orientations,
        k_translational=1.0,
        k_rotational=1.0,
        symmetries=[[1, 0, 0, 0]])
    mc.external_potential = lattice

    # create C++ mirror classes and set parameters
    sim.run(0)

    # make sure objecst are attached
    sim.operations.remove(mc)
    assert not mc._attached
    assert not lattice._attached


@pytest.mark.cpu
def test_harmonic_displacement_energy(device, simulation_factory,
                                      two_particle_snapshot_factory):
    """Ensure harmonic displacements result in expected energy."""
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # create harmonic field
    k_trans = 1.0
    snapshot = sim.state.get_snapshot()
    if device.communicator.rank == 0:
        reference_positions = snapshot.particles.position
        reference_orientations = snapshot.particles.orientation
    else:
        reference_positions = [[0, 0, 0], [0, 0, 0]]
        reference_orientations = [[1, 0, 0, 0], [1, 0, 0, 0]]
    lattice = hoomd.hpmc.field.Harmonic(
        reference_positions=reference_positions,
        reference_orientations=reference_orientations,
        k_translational=k_trans,
        k_rotational=1.0,
        symmetries=[[1, 0, 0, 0]])
    mc.external_potential = lattice

    dx = 0.01
    disp = np.array([dx, 0, 0])
    lattice.reference_positions = lattice.reference_positions + disp

    # run and check energy
    sim.run(0)
    assert np.allclose(lattice.energy,
                       0.5 * dx**2 * k_trans * sim.state.N_particles)


@pytest.mark.cpu
def test_harmonic_displacement(device, simulation_factory,
                               two_particle_snapshot_factory):
    """Ensure particles remain close to reference positions."""
    mc = hoomd.hpmc.integrate.Sphere()
    particle_diameter = 0.5
    mc.shape['A'] = dict(diameter=particle_diameter)

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # create lattice field
    k_trans = 100.0
    snapshot = sim.state.get_snapshot()
    if device.communicator.rank == 0:
        reference_positions = snapshot.particles.position
        reference_orientations = snapshot.particles.orientation
    else:
        reference_positions = [[0, 0, 0], [0, 0, 0]]
        reference_orientations = [[1, 0, 0, 0], [1, 0, 0, 0]]
    lattice = hoomd.hpmc.field.Harmonic(
        reference_positions=reference_positions,
        reference_orientations=reference_orientations,
        k_translational=k_trans,
        k_rotational=1.0,
        symmetries=[[1, 0, 0, 0]])
    mc.external_potential = lattice

    # run and check that particles haven't moved farther than half a diameter
    sim.run(100)
    snapshot = sim.state.get_snapshot()
    if device.communicator.rank == 0:
        new_positions = snapshot.particles.position
        dx = np.linalg.norm(new_positions - reference_positions, axis=1)
        assert np.all(np.less(dx, particle_diameter / 2))
