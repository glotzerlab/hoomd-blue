# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test hoomd.hpmc.external.user.CPPExternalField"""

import hoomd
import pytest
import numpy as np

# check if llvm_enabled
llvm_disabled = not hoomd.version.llvm_enabled

valid_constructor_args = [dict(code='return -1;', clang_exec='/usr/bin/clang')]

# setable attributes before attach for CPPExternalField objects
valid_attrs = [('code', 'return -1;'), ('clang_exec', 'clang')]

# attributes that cannot be set after object is attached
attr_error = [('code', 'return -1.0;'), ('clang_exec', '/usr/bin/clang')]

# (orientation of p1, orientation of p2, charge of both particles, expected result)
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

attr_translator = {'code': '_code', 'clang_exec': '_clang_exec'}


@pytest.mark.cpu
@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_cpp_external(device, constructor_args):
    """Test that CPPExternalField can be constructed with valid arguments."""
    ext = hoomd.hpmc.external.user.CPPExternalField(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        attr = attr_translator.get(attr, attr)
        assert getattr(ext, attr) == value


@pytest.mark.cpu
@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_valid_construction_and_attach_cpp_external(
        device, simulation_factory, two_particle_snapshot_factory,
        constructor_args):
    """Test that CPPExternalField can be attached with valid arguments."""
    # create objects
    ext = hoomd.hpmc.external.user.CPPExternalField(**constructor_args)
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.field = ext

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # create C++ mirror classes and set parameters
    sim.run(0)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        attr = attr_translator.get(attr, attr)
        assert getattr(ext, attr) == value


@pytest.mark.cpu
@pytest.mark.serial
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_cpp_external(device, attr, value):
    """Test that CPPExternalField can get and set attributes before attached."""
    ext = hoomd.hpmc.external.user.CPPExternalField(code='return 0;')

    setattr(ext, attr, value)
    assert getattr(ext, attr) == value


@pytest.mark.cpu
@pytest.mark.serial
@pytest.mark.parametrize("attr,val", attr_error)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_raise_attr_error_cpp_external(device, attr, val, simulation_factory,
                                       two_particle_snapshot_factory):
    """Test that CPPExternalField raises AttributeError if we
       try to set certain attributes after attaching.
    """

    ext = hoomd.hpmc.external.user.CPPExternalField(code='return 0;')
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.field = ext

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.run(0)

    # try to reset when attached
    with pytest.raises(AttributeError):
        setattr(ext, attr, val)


@pytest.mark.cpu
@pytest.mark.serial
@pytest.mark.parametrize("orientations,charge, result", electric_field_params)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_electric_field(device, orientations, charge, result,
                        simulation_factory, two_particle_snapshot_factory):
    """Test that CPPExternalField computes the correct energies for static
       point-like electric dipoles inmersed in an uniform electric field.
    """

    # pontential energy of a point dipole in an electric field in the z direction
    #    - use charge as a proxy for dipole moment
    #    - ignore units for simplicity
    electric_field = """ vec3<Scalar> E(0,0,1);
                         vec3<Scalar> p = charge*rotate(q_i, vec3<Scalar> (1,0,0));
                         return -dot(p,E);
                     """

    sim = simulation_factory(two_particle_snapshot_factory())

    ext = hoomd.hpmc.external.user.CPPExternalField(code=electric_field)
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0, orientable=True)
    mc.field = ext

    sim.operations.integrator = mc
    with sim.state.cpu_local_snapshot as data:
        data.particles.orientation[0, :] = orientations[0]
        data.particles.orientation[1, :] = orientations[1]
        data.particles.charge[:] = charge
    sim.run(0)

    assert np.isclose(ext.energy, result)


@pytest.mark.cpu
@pytest.mark.serial
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_gravity(device, simulation_factory, lattice_snapshot_factory):
    """ This test simulates a sedimentation experiment by using an elongated
        box in the z-dimension and adding an effective gravitational
        potential with a wall. Note that it is technically probabilistic in
        nature, but we use enough particles and a strong enough gravitational
        potential that the probability of particles rising in the simulation is
        vanishingly small.
    """

    sim = simulation_factory(lattice_snapshot_factory(a=1.1, n=5))
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.01)
    mc.shape['A'] = dict(diameter=1)

    # expand box and add gravity field
    old_box = sim.state.box
    new_box = hoomd.Box(Lx=1.5 * old_box.Lx,
                        Ly=1.5 * old_box.Ly,
                        Lz=20 * old_box.Lz)
    sim.state.set_box(new_box)
    ext = hoomd.hpmc.external.user.CPPExternalField(code="return 1000*r_i.z;")
    mc.field = ext
    sim.operations.integrator = mc

    old_avg_z = np.mean(sim.state.get_snapshot().particles.position[:, 2])
    sim.run(0)
    old_energy = ext.energy

    sim.run(6e3)

    new_avg_z = np.mean(sim.state.get_snapshot().particles.position[:, 2])
    assert new_avg_z < old_avg_z
    assert ext.energy < old_energy
