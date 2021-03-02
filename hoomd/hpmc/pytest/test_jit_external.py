# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test hoomd.jit.ext.UserExternal"""

import hoomd
from hoomd import jit
import pytest
import numpy as np


valid_constructor_args = [
    dict(code='return -1;',
         llvm_ir_file="code.ll",
         clang_exec='/usr/bin/clang')
]


# setable attributes before attach for UserExternal objects
valid_attrs = [('code', 'return -1;'),
               ('llvm_ir_file', 'code.ll'),
               ('clang_exec', 'clang')
]


# attributes that cannot be set after object is attached
attr_error =  [('code', 'return -1.0;'),
               ('llvm_ir_file', 'test.ll'),
               ('clang_exec', '/usr/bin/clang')
]


# (orientation of p1, orientation of p2, charge of both particles, expected result)
electric_field_params = [([(1, 0, 0, 0), (0, np.sqrt(2)/2, np.sqrt(2)/2, 0)], 1, 0),
                         ([(1, 0, 0, 0),(0, np.sqrt(2)/2, 0, np.sqrt(2)/2)], 1, -1),
                         ([(0, np.sqrt(2)/2, 0, -np.sqrt(2)/2),(0, np.sqrt(2)/2, 0, np.sqrt(2)/2)], 1, 0),
                         ([(0, np.sqrt(2)/2, 0, np.sqrt(2)/2),(0, np.sqrt(2)/2,0, np.sqrt(2)/2)], 1, -2),
                         ([(0, np.sqrt(2)/2, 0, -np.sqrt(2)/2), (0, np.sqrt(2)/2, 0, -np.sqrt(2)/2)], 1,  2),
                         ([(0, np.sqrt(2)/2, 0, -np.sqrt(2)/2), (0, np.sqrt(2)/2, 0, -np.sqrt(2)/2)], -1,  -2),
                         ([(0, np.sqrt(2)/2, 0, -np.sqrt(2)/2), (0, np.sqrt(2)/2, 0, -np.sqrt(2)/2)], -3,  -6),
]


@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_user_external(device, constructor_args):
    """Test that UserExternal can be constructed with valid arguments."""
    ext = jit.external.UserExternal(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(ext, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_and_attach_user_external(device, simulation_factory,
                                       two_particle_snapshot_factory,
                                       constructor_args):
    """Test that UserExternal can be attached with valid arguments."""
    # create objects
    ext = jit.external.UserExternal(**constructor_args)
    mc = hoomd.hpmc.integrate.Sphere(seed=1)
    mc.shape['A'] = dict(diameter=0)

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.operations += ext

    # create C++ mirror classes and set parameters
    sim.run(0)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(ext, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_user_external(device, attr, value):
    """Test that UserExternal can get and set attributes before attached."""
    ext = jit.external.UserExternal(code='return 0;')

    setattr(ext, attr, value)
    assert getattr(ext, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("attr,val",  attr_error)
def test_raise_attr_error_user_external(device, attr, val,
                                     simulation_factory,
                                     two_particle_snapshot_factory):
    """Test that UserExternal raises AttributeError if we
       try to set certain attributes after attaching.
    """

    ext = jit.external.UserExternal(code='return 0;')

    mc = hoomd.hpmc.integrate.Sphere(seed=1)
    mc.shape['A'] = dict(diameter=0)
    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.operations += ext
    sim.run(0)
    # try to reset when attached
    with pytest.raises(AttributeError):
        setattr(ext, attr, val)


@pytest.mark.serial
@pytest.mark.parametrize("orientations,charge, result", electric_field_params)
def test_electric_field(device, orientations, charge, result,
                        simulation_factory, two_particle_snapshot_factory):
    """Test that UserExternal computes the correct field energies."""

    # pontential energy of a point dipole in an electric field in the z direction
    #    - use charge as a proxy for dipole moment
    #    - ignore units for simplicity
    electric_field = """ vec3<Scalar> E(0,0,1);
                         vec3<Scalar> p = charge*rotate(q_i, vec3<Scalar> (1,0,0));
                         return -dot(p,E);
                     """

    sim = simulation_factory(two_particle_snapshot_factory())

    ext = jit.external.UserExternal(code=electric_field)
    mc = hoomd.hpmc.integrate.Sphere(d=0, a=0, seed=1)
    mc.shape['A'] = dict(diameter=0, orientable=True)

    sim.operations.integrator = mc
    sim.operations += ext
    with sim.state.cpu_local_snapshot as data:
        data.particles.orientation[0, :] = orientations[0]
        data.particles.orientation[1, :] = orientations[1]
        data.particles.charge[:] = charge
    sim.run(0)

    assert np.isclose(ext.energy, result)
