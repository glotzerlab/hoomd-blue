# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test hoomd.hpmc.pair.user.CPPUnionPotential."""

import hoomd
import pytest
import numpy as np

# check if llvm_enabled
llvm_disabled = not hoomd.version.llvm_enabled

valid_constructor_args = [
    dict(
        r_cut_isotropic=3,
        r_cut_constituent=2.0,
        param_array_isotropic=[0, 1],
        param_array_constituent=[2.0, 4.0],
        code_isotropic='return -1;',
        code_constituent='return -2;',
    ),
    dict(
        r_cut_isotropic=1.0,
        r_cut_constituent=3.0,
        param_array_isotropic=[1, 2, 3, 4],
        param_array_constituent=[1, 2, 3, 4],
        code_isotropic='return -1;',
        code_constituent='return 0;',
    )
]

# setable attributes before attach for CPPPotential objects
valid_attrs = [
    ('r_cut_isotropic', 1.4),
    ('r_cut_constituent', 1.0),
    ('code_isotropic', 'return -1;'),
    ('code_constituent', 'return -1;'),
]

# setable attributes after attach for CPPPotential objects
valid_attrs_after_attach = [
    ('r_cut_isotropic', 1.3),
    ('r_cut_constituent', 1.3),
]

# attributes that cannot be set after object is attached
attr_error = [
        ('code_isotropic', 'return -1.0;'),
        ('code_constituent', 'return -2.0;'),
]

positions_orientations_result = [
    ([(0, 0, 0), (1, 0, 0)], [(1, 0, 0, 0), (1, 0, 0, 0)], -1),
    ([(0, 0, 0), (1, 0, 0)], [(1, 0, 0, 0), (0, 0, 1, 0)], 1),
    ([(0, 0, 0), (0, 0, 1)], [(1, 0, 0, 0), (1, 0, 0, 0)], 1 / 2),
    ([(0, 0, 0), (0, 0, 1)], [(1, 0, 0, 0), (0, 0, 1, 0)], -1 / 2),
]


@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_cpp_union_potential(device, constructor_args):
    """Test that CPPUnionPotential can be constructed with valid arguments."""
    patch = hoomd.hpmc.pair.user.CPPUnionPotential(**constructor_args)

    # validate the params were set properly
    # note that the constituent and isotropic codes do not get stored in the
    # object's param dict, but they do get saved as a prive member in the
    # python object, so we prepend a '_' to the code attributes in this test
    for attr, value in constructor_args.items():
        if attr in ['code_constituent', 'code_isotropic']:
            attr = f'_{attr}'
        try:
            assert getattr(patch, attr) == value
        except ValueError:
            assert all(getattr(patch, attr) == value)


@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_valid_construction_and_attach_cpp_union_potential(
        device, simulation_factory, two_particle_snapshot_factory,
        constructor_args):
    """Test that CPPUnionPotential can be attached with valid arguments."""
    # create objects
    patch = hoomd.hpmc.pair.user.CPPUnionPotential(**constructor_args)
    patch.positions['A'] = [(0, 0, 0)]
    patch.orientations['A'] = [(1, 0, 0, 0)]
    patch.diameters['A'] = [1.0]
    patch.typeids['A'] = [0]
    patch.charges['A'] = [0]
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
    mc.potential = patch

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # create C++ mirror classes and set parameters
    sim.run(0)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        if attr in ['code_constituent', 'code_isotropic']:
            attr = f'_{attr}'
        try:
            assert getattr(patch, attr) == value
        except ValueError:  # array-like
            assert all(getattr(patch, attr) == value)


@pytest.mark.serial
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_cpp_union_potential(device, attr, value):
    """Test that CPPUnionPotential can get and set attributes before \
            attached."""
    patch = hoomd.hpmc.pair.user.CPPUnionPotential(
            r_cut_isotropic=1.4,
            r_cut_constituent=1.0,
            code_isotropic='return 5;',
            code_constituent='return 6;',
            )

    setattr(patch, attr, value)
    assert getattr(patch, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("attr,value", valid_attrs_after_attach)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_valid_setattr_attached_cpp_union_potential(device, attr, value,
                                              simulation_factory,
                                              two_particle_snapshot_factory):
    """Test that CPPUnionPotential can get and set attributes after attached."""
    patch = hoomd.hpmc.pair.user.CPPUnionPotential(
            r_cut_isotropic=1.4,
            r_cut_constituent=1.0,
            code_isotropic='return 5;',
            code_constituent='return 6;',
            )
    patch.positions['A'] = [(0, 0, 0)]
    patch.orientations['A'] = [(1, 0, 0, 0)]
    patch.diameters['A'] = [1.0]
    patch.typeids['A'] = [0]
    patch.charges['A'] = [0]
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.potential = patch

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # create C++ mirror classes and set parameters
    sim.run(0)

    # validate the params were set properly
    setattr(patch, attr, value)
    assert getattr(patch, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("attr,val", attr_error)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_raise_attr_error_cpp_union_potential(device,
                                              attr, 
                                              val,
                                              simulation_factory,
                                              two_particle_snapshot_factory):
    """Test that CPPUnionPotential raises AttributeError if we \
            try to set certain attributes after attaching."""
    patch = hoomd.hpmc.pair.user.CPPUnionPotential(
            r_cut_isotropic=1.4,
            r_cut_constituent=1.0,
            code_isotropic='return 5;',
            code_constituent='return 6;',
            )
    patch.positions['A'] = [(0, 0, 0)]
    patch.orientations['A'] = [(1, 0, 0, 0)]
    patch.diameters['A'] = [1.0]
    patch.typeids['A'] = [0]
    patch.charges['A'] = [0]
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.potential = patch

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.run(0)

    # make sure the AttributeError gets raised when trying to change the
    # properties that cannot be changed after attachment
    # in this case, this is the jit-compiled code, because it gets compiled
    # upon attachment, so changing after attaching would return code that does
    # not reflect what was attached
    with pytest.raises(AttributeError):
        setattr(patch, attr, val)


@pytest.mark.serial
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_param_array_union(device, simulation_factory,
                           two_particle_snapshot_factory):
    """Test passing in parameter arrays to the union patch objects.

    This test tests that changes to the parameter array are reflected on the
    energy calculation. It therefore also tests that the energies are computed
    correctly.

    """
    square_well_isotropic = """
                   float rsq = dot(r_ij, r_ij);
                   float rcut = param_array[0];
                   if (rsq < rcut*rcut)
                       return param_array[1];
                   else
                       return 0.0f;
                  """
    square_well_constituent = """
                   float rsq = dot(r_ij, r_ij);
                   float rcut = param_array_constituent[0];
                   if (rsq < rcut*rcut)
                       return param_array_constituent[1];
                   else
                       return 0.0f;
                  """

    # set up the system and patches
    sim = simulation_factory(two_particle_snapshot_factory())
    r_cut_iso = 5
    params = dict(
        code_isotropic=square_well_isotropic,
        param_array_isotropic=[2.5, 1.0],
        r_cut_isotropic=r_cut_iso,
        code_constituent=square_well_constituent,
        param_array_constituent=[1.5, 3.0],
        r_cut_constituent=1.5,
    )
    patch = hoomd.hpmc.pair.user.CPPUnionPotential(**params)
    const_particle_pos = [(0.0, -0.5, 0), (0.0, 0.5, 0)]
    patch.positions['A'] = const_particle_pos
    patch.orientations['A'] = [(1, 0, 0, 0), (1, 0, 0, 0)]
    patch.diameters['A'] = [1.0, 1.0]
    patch.typeids['A'] = [0, 0]
    patch.charges['A'] = [0, 0]
    mc = hoomd.hpmc.integrate.SphereUnion()
    sphere1 = dict(diameter=1)
    mc.shape["A"] = dict(shapes=[sphere1, sphere1],
                         positions=const_particle_pos,
                         orientations=[(1, 0, 0, 0), (1, 0, 0, 0)])
    mc.potential = patch
    sim.operations.integrator = mc

    # first test the case where r_cut_isotropic = 0, so particles only interact
    # through the constituent particles
    # there's 2 cases here, one where they interact with only 1 other
    # constituent particle, and then where the interact with all of the
    # neighboring constituent particles

    # first, only interact with nearest neighboring constituent particle
    patch.param_array_isotropic[0] = 0.0
    patch.param_array_isotropic[1] = -1.0
    patch.param_array_constituent[0] = 1.1
    patch.param_array_constituent[1] = -1.0
    sim.run(0)
    assert (np.isclose(patch.energy, -2.0))  # 2 interacting pairs

    # now extend r_cut_constituent so that all constituent particles interact
    # with all other constituent particles
    patch.param_array_constituent[0] = np.sqrt(2) + 0.1
    sim.run(0)
    assert (np.isclose(patch.energy, -4.0))  # 4 interacting pairs this time

    # now add a respulsive interaction between the union centers
    # and increase its r_cut so that the particle centers interact
    # this should increase the energy by 1 unit
    patch.param_array_isotropic[0] = 2.0
    patch.param_array_isotropic[1] = 1.0
    sim.run(0)
    assert (np.isclose(patch.energy, -3.0))

    # now make r_cut_constituent zero so that only the centers interact with
    # their repulsive square well (square shoulder?)
    patch.param_array_constituent[0] = 0
    sim.run(0)
    assert (np.isclose(patch.energy, 1.0))

    # change epsilon of center-center interaction to make it attractive
    # to make sure that change is reflected in the energy
    patch.param_array_isotropic[1] = -1.0
    sim.run(0)
    assert (np.isclose(patch.energy, -1.0))

    # set both r_cuts to zero to make sure no particles interact
    patch.param_array_isotropic[0] = 0
    sim.run(0)
    assert (np.isclose(patch.energy, 0.0))
