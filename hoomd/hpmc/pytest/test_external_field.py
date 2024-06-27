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

    def add(simulation, field_type):
        mc = hoomd.hpmc.integrate.Sphere()
        mc.shape['A'] = dict(diameter=0)
        snapshot = simulation.state.get_snapshot()
        if field_type == 'harmonic':
            if simulation.device.communicator.rank == 0:
                reference_positions = snapshot.particles.position
                reference_orientations = snapshot.particles.orientation
            else:
                reference_positions = [[0, 0, 0], [0, 0, 0]]
                reference_orientations = [[1, 0, 0, 0], [1, 0, 0, 0]]
            field = hoomd.hpmc.external.field.Harmonic(
                reference_positions=reference_positions,
                reference_orientations=reference_orientations,
                k_translational=1.0,
                k_rotational=1.0,
                symmetries=[[1, 0, 0, 0]])
            mc.external_potential = field
        elif field_type == 'linear':
            field = hoomd.hpmc.external.Linear(default_alpha=3.0)
            mc.external_potentials = [field]
        simulation.operations.integrator = mc
        return mc, field

    return add


@pytest.mark.cpu
class TestExternalPotentialHarmonic:

    def test_attaching(self, simulation_factory, two_particle_snapshot_factory,
                       add_default_integrator):
        # create simulation & attach objects
        sim = simulation_factory(two_particle_snapshot_factory())
        mc, lattice = add_default_integrator(sim, 'harmonic')

        # create C++ mirror classes and set parameters
        sim.run(0)

        # make sure objecst are attached
        assert mc._attached
        assert lattice._attached

    def test_detaching(self, simulation_factory, two_particle_snapshot_factory,
                       add_default_integrator):
        # create simulation & attach objects
        sim = simulation_factory(two_particle_snapshot_factory())
        mc, lattice = add_default_integrator(sim, 'harmonic')

        # create C++ mirror classes and set parameters
        sim.run(0)

        # make sure objecst are attached
        sim.operations.remove(mc)
        assert not mc._attached
        assert not lattice._attached

    def test_harmonic_displacement_energy(self, simulation_factory,
                                          two_particle_snapshot_factory,
                                          add_default_integrator):
        """Ensure harmonic displacements result in expected energy."""
        # create simulation & attach objects
        sim = simulation_factory(two_particle_snapshot_factory())
        mc, lattice = add_default_integrator(sim, 'harmonic')
        mc.shape['A'] = dict(diameter=0, orientable=True)

        dx = 0.01
        disp = np.array([dx, 0, 0])
        lattice.reference_positions = lattice.reference_positions + disp

        # run and check energy
        sim.run(0)
        k_translational = lattice.k_translational(sim.timestep)
        assert np.allclose(
            lattice.energy,
            0.5 * dx**2 * k_translational * sim.state.N_particles)

        # make some moves and make sure the different energies are not zero
        sim.run(10)
        assert lattice.energy_rotational != 0.0
        assert lattice.energy_translational != 0.0
        assert np.allclose(
            lattice.energy,
            lattice.energy_translational + lattice.energy_rotational)

        # set k_rotational to zero and ensure the rotational energy is zero
        lattice.k_rotational = 0
        assert lattice.energy_rotational == 0.0

        # set k_translational to zero and ensure the translational energy is
        # zero
        lattice.k_translational = 0
        assert lattice.energy_translational == 0.0

    def test_harmonic_displacement(self, simulation_factory,
                                   two_particle_snapshot_factory,
                                   add_default_integrator):
        """Ensure particles remain close to reference positions."""
        # create simulation & attach objects
        sim = simulation_factory(two_particle_snapshot_factory())
        mc, lattice = add_default_integrator(sim, 'harmonic')
        particle_diameter = 0.5
        mc.shape['A'] = dict(diameter=particle_diameter)
        k_trans = 100.0
        lattice.k_translational = k_trans

        # run and check that particles haven't moved farther than half a
        # diameter
        sim.run(100)
        snapshot = sim.state.get_snapshot()
        if snapshot.communicator.rank == 0:
            new_positions = snapshot.particles.position
            dx = np.linalg.norm(new_positions - lattice.reference_positions,
                                axis=1)
            assert np.all(np.less(dx, particle_diameter / 2))


@pytest.mark.cpu
class TestExternalPotentialLinear:

    def test_valid_construction_linearfield(self, device):
        """Test that Linear can be constructed with valid arguments."""
        hoomd.hpmc.external.Linear(default_alpha=1.0)

    def test_attaching(self, simulation_factory, two_particle_snapshot_factory,
                       add_default_integrator):
        # create simulation & attach objects
        sim = simulation_factory(two_particle_snapshot_factory())
        mc, field = add_default_integrator(sim, 'linear')

        # create C++ mirror classes and set parameters
        sim.run(0)

        # verify objecst are attached
        assert mc._attached
        assert field._attached

    def test_detaching(self, simulation_factory, two_particle_snapshot_factory,
                       add_default_integrator):
        # create simulation & attach objects
        sim = simulation_factory(two_particle_snapshot_factory())
        mc, lattice = add_default_integrator(sim, 'linear')

        # create C++ mirror classes and set parameters
        sim.run(0)

        # make sure objecst are attached
        sim.operations.remove(mc)
        assert not mc._attached
        assert not lattice._attached

    def test_energy(self, simulation_factory, two_particle_snapshot_factory,
                    add_default_integrator):
        """Verify energy is what it should be with known particle positions."""
        sim = simulation_factory(two_particle_snapshot_factory(d=1.0))
        mc, field = add_default_integrator(sim, 'linear')
        field.plane_normal = (1, 0, 0)
        field.plane_origin = (0.3, 0.2, 0.4)
        field.alpha['A'] = 1.23456
        mc.shape['A'] = dict(diameter=0, orientable=True)
        sim.run(0)

        # move one particle to break symmetry and give non-zero energy
        snapshot = sim.state.get_snapshot()
        if sim.device.communicator.rank == 0:
            x0 = snapshot.particles.position[0][0]
            snapshot.particles.position[1][0] = 2 * x0
        sim.state.set_snapshot(snapshot)

        if sim.device.communicator.rank == 0:
            field_reference_energy = 0.0
            for r in snapshot.particles.position:
                field_reference_energy += np.dot(
                    r - field.plane_origin,
                    field.plane_normal) * field.alpha['A']

        field_energy = field.energy
        mc_external_energy = mc.external_energy

        if sim.device.communicator.rank == 0:
            assert field_energy == pytest.approx(field_reference_energy)
            assert field_energy == pytest.approx(mc_external_energy)

        # Test that HPMCIntegrator correctly handles multiple fields.
        field2 = hoomd.hpmc.external.Linear(default_alpha=2.345,
                                            plane_origin=(0.1, 0.2, 0.3),
                                            plane_normal=(-0.2, 5, -2))
        mc.external_potentials.append(field2)
        field2_reference_energy = 0
        if sim.device.communicator.rank == 0:
            for r in snapshot.particles.position:
                field2_reference_energy += np.dot(
                    r - field2.plane_origin,
                    field2.plane_normal) * field2.alpha['A']

        field2_energy = field2.energy
        mc_external_energy = mc.external_energy

        if sim.device.communicator.rank == 0:
            assert field2_energy == pytest.approx(field2_reference_energy)
            assert field_energy + field2_energy == pytest.approx(
                mc_external_energy)

        # Test that origin shifting does not change the energy
        mc.d['A'] = 0
        sim.run(1000)
        field_energy = field.energy
        field2_energy = field2.energy

        if sim.device.communicator.rank == 0:
            assert field_energy == pytest.approx(field_reference_energy)
            assert field2_energy == pytest.approx(field2_reference_energy)

    def test_normalization_of_plane_normal(self, simulation_factory,
                                           two_particle_snapshot_factory,
                                           add_default_integrator):
        # create simulation & attach objects
        sim = simulation_factory(two_particle_snapshot_factory())
        mc, field = add_default_integrator(sim, 'linear')
        field.plane_normal = (1, 2, 3)
        magnitude = np.linalg.norm(field.plane_normal)

        # create C++ mirror classes and set parameters
        assert field.plane_normal == (1, 2, 3)
        sim.run(0)  # normalization occurs on attaching
        np.testing.assert_allclose(
            np.array(field.plane_normal) * magnitude, (1, 2, 3))

    @pytest.mark.validate
    def test_z_bias(self, device, simulation_factory, lattice_snapshot_factory):
        """Test that particles respond to a potential as expected.

        This test simulates a system of particles with a linear potential that
        biases their z-coordinates to more negative values.

        """
        sim = simulation_factory(lattice_snapshot_factory(a=1.1, n=5))
        mc = hoomd.hpmc.integrate.Sphere(default_d=0.01)
        mc.shape['A'] = dict(diameter=1)
        mc.nselect = 1

        # expand box and add external field
        old_box = sim.state.box
        new_box = hoomd.Box(Lx=3 * old_box.Lx,
                            Ly=3 * old_box.Ly,
                            Lz=3 * old_box.Lz)
        sim.state.set_box(new_box)
        ext = hoomd.hpmc.external.Linear(default_alpha=1000,
                                         plane_origin=(0, 0, 0),
                                         plane_normal=(0, 0, 1))
        mc.external_potentials = [ext]
        sim.operations.integrator = mc

        snapshot = sim.state.get_snapshot()
        sim.run(0)
        if snapshot.communicator.rank == 0:
            old_mean_z = np.mean(snapshot.particles.position[:, 2])
        old_energy = ext.energy

        for n in range(10):
            sim.run(1e3)
            snapshot = sim.state.get_snapshot()
            new_energy = ext.energy
            if snapshot.communicator.rank == 0:
                new_mean_z = np.mean(snapshot.particles.position[:, 2])
                # since the potential's origin is (0, 0, 0) and its normal is
                # (0, 0, 1), the total energy should be equal to N * alpha *
                # mean_z, where N is the number of particles, alpha is the
                # prefactor, and mean_z is the average position of the
                # particles. verify that this equality holds.
                np.testing.assert_allclose(
                    new_energy,
                    snapshot.particles.N * ext.alpha['A'] * new_mean_z)
                assert new_mean_z < old_mean_z
                assert new_energy < old_energy
                old_mean_z = new_mean_z
                old_energy = new_energy
