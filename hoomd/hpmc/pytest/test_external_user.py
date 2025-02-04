# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.external.user.CPPExternalPotential."""

import hoomd
import pytest
import numpy as np

# check if llvm_enabled
llvm_disabled = not hoomd.version.llvm_enabled

valid_constructor_args = [
    dict(code='return -1;'),
    dict(code='return -1;', param_array=[1]),
]

# setable attributes before attach for CPPExternalPotential objects
valid_attrs = [('code', 'return -1;'), ('param_array', [1])]

# attributes that cannot be set after object is attached
attr_error = [('code', 'return -1.0;'), ('param_array', [1])]

# list of tuples with (
# orientation of p1,
# orientation of p2,
# charge of both particles,
# expected result
# )
electric_field_params = [
    ([(1, 0, 0, 0), (0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0)], 1, 0),
    ([(1, 0, 0, 0), (0, np.sqrt(2) / 2, 0, np.sqrt(2) / 2)], 1, -1),
    ([(0, np.sqrt(2) / 2, 0, -np.sqrt(2) / 2),
      (0, np.sqrt(2) / 2, 0, np.sqrt(2) / 2)], 1, 0),
    ([(0, np.sqrt(2) / 2, 0, np.sqrt(2) / 2),
      (0, np.sqrt(2) / 2, 0, np.sqrt(2) / 2)], 1, -2),
    ([(0, np.sqrt(2) / 2, 0, -np.sqrt(2) / 2),
      (0, np.sqrt(2) / 2, 0, -np.sqrt(2) / 2)], 1, 2),
    ([(0, np.sqrt(2) / 2, 0, -np.sqrt(2) / 2),
      (0, np.sqrt(2) / 2, 0, -np.sqrt(2) / 2)], -1, -2),
    ([(0, np.sqrt(2) / 2, 0, -np.sqrt(2) / 2),
      (0, np.sqrt(2) / 2, 0, -np.sqrt(2) / 2)], -3, -6),
]


@pytest.mark.cpu
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_cpp_external(device, constructor_args):
    """Test that CPPExternalPotential can be constructed with valid args."""
    ext = hoomd.hpmc.external.user.CPPExternalPotential(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(ext, attr) == value


@pytest.mark.cpu
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_attaching(device, simulation_factory, two_particle_snapshot_factory,
                   constructor_args):
    ext = hoomd.hpmc.external.user.CPPExternalPotential(**constructor_args)
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.external_potential = ext

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # create C++ mirror classes and set parameters
    sim.run(0)

    # make sure objects are attached
    assert mc._attached
    assert ext._attached


@pytest.mark.cpu
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_detaching(device, simulation_factory, two_particle_snapshot_factory,
                   constructor_args):
    ext = hoomd.hpmc.external.user.CPPExternalPotential(**constructor_args)
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.external_potential = ext

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # create C++ mirror classes and set parameters
    sim.run(0)

    # make sure objects are attached
    sim.operations.remove(mc)
    assert not mc._attached
    assert not ext._attached


@pytest.mark.cpu
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_valid_construction_and_attach_cpp_external(
        device, simulation_factory, two_particle_snapshot_factory,
        constructor_args):
    """Test that CPPExternalPotential can be attached with valid arguments."""
    # create objects
    ext = hoomd.hpmc.external.user.CPPExternalPotential(**constructor_args)
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.external_potential = ext

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # create C++ mirror classes and set parameters
    sim.run(0)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(ext, attr) == value


@pytest.mark.cpu
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_cpp_external(device, attr, value):
    """Test that CPPExternalPotential can get and set attributes before \
            attached."""
    ext = hoomd.hpmc.external.user.CPPExternalPotential(code='return 0;')

    setattr(ext, attr, value)
    assert getattr(ext, attr) == value


@pytest.mark.cpu
@pytest.mark.parametrize("attr,val", attr_error)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_raise_attr_error_cpp_external(device, attr, val, simulation_factory,
                                       two_particle_snapshot_factory):
    """Test that CPPExternalPotential raises AttributeError if we try to set \
            certain attributes after attaching."""
    ext = hoomd.hpmc.external.user.CPPExternalPotential(code='return 0;')
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.external_potential = ext

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.run(0)

    # try to reset when attached
    with pytest.raises(AttributeError):
        setattr(ext, attr, val)


@pytest.mark.cpu
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_change_param_array_values(device, simulation_factory,
                                   two_particle_snapshot_factory):
    """Test that changing param_array values behaves correctly."""
    ext = hoomd.hpmc.external.user.CPPExternalPotential(
        code='return param_array[0];', param_array=[1])
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.1)
    mc.shape['A'] = dict(diameter=0.1)
    mc.d['A'] = 0.1
    mc.external_potential = ext

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    N = sim.state.N_particles
    sim.operations.integrator = mc
    sim.run(0)
    for _n in range(3):
        ext.param_array[0] = _n
        sim.run(1)
        assert ext.energy / N == _n


@pytest.mark.cpu
@pytest.mark.parametrize("orientations,charge, result", electric_field_params)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_electric_field(device, orientations, charge, result,
                        simulation_factory, two_particle_snapshot_factory):
    """Test that CPPExternalPotential computes the correct energies for static \
            point-like electric dipoles immersed in an uniform electric field.

    Here, we test the potential energy of a point dipole in an electric field
    oriented along the z-direction. Note that we 1) use charge as a proxy for
    the dipole moment and 2) ignore units for simplicity.

    """
    electric_field = """ vec3<Scalar> E(0,0,1);
                         vec3<Scalar> p = charge*rotate(q_i, vec3<Scalar>
                            (1,0,0));
                         return -dot(p,E);
                     """

    sim = simulation_factory(two_particle_snapshot_factory())

    ext = hoomd.hpmc.external.user.CPPExternalPotential(code=electric_field)
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0, orientable=True)
    mc.external_potential = ext

    sim.operations.integrator = mc
    with sim.state.cpu_local_snapshot as data:
        data.particles.charge[:] = charge
        N = len(data.particles.position)
        for global_idx in [0, 1]:
            idx = data.particles.rtag[global_idx]
            if idx < N:
                data.particles.orientation[idx, :] = orientations[global_idx]
    sim.run(0)

    energy = ext.energy
    assert np.isclose(energy, result)


@pytest.mark.cpu
@pytest.mark.validate
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_z_bias(device, simulation_factory, lattice_snapshot_factory):
    """Test that a biasing potential restrains particles to a specified region.

    This test simulates a system of particles with a harmonic potential that
    biases their z-coordinates to 0.  Note that the test is probabilistic in
    nature, but we use enough particles and a strong enough potential that the
    probability of particles moving away from z=0 in the simulation is
    vanishingly small.

    """
    sim = simulation_factory(lattice_snapshot_factory(a=1.1, n=5))
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.01)
    mc.shape['A'] = dict(diameter=1)
    mc.nselect = 1

    # expand box and add external field
    old_box = sim.state.box
    new_box = hoomd.Box(Lx=3 * old_box.Lx, Ly=3 * old_box.Ly, Lz=3 * old_box.Lz)
    sim.state.set_box(new_box)
    ext = hoomd.hpmc.external.user.CPPExternalPotential(
        code="return 1000*r_i.z*r_i.z;")
    mc.external_potential = ext
    sim.operations.integrator = mc

    snapshot = sim.state.get_snapshot()
    if snapshot.communicator.rank == 0:
        old_z_range = np.ptp(snapshot.particles.position[:, 2])
    sim.run(0)
    old_energy = ext.energy

    for n in range(10):
        sim.run(1e3)
        snapshot = sim.state.get_snapshot()
        if snapshot.communicator.rank == 0:
            new_z_range = np.ptp(snapshot.particles.position[:, 2])
            assert (new_z_range < old_z_range)
            old_z_range = new_z_range
        new_energy = ext.energy
        assert new_energy < old_energy
        old_energy = new_energy
