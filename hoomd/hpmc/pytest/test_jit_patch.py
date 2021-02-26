# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test hoomd.jit.patch.UserPatch and hoomd.jit.patch.UserUnionPatch."""

import hoomd
from hoomd import jit
import pytest
import numpy as np


valid_constructor_args = [
    dict(r_cut = 3,
         array_size=2,
         code='return -1;',
         llvm_ir_file="code.ll",
         clang_exec='/usr/bin/clang'),
    dict(r_cut = 2,
         array_size=4,
         code='return -1;'),
]


valid_constructor_args_union = [
    dict(r_cut_union = 3,
         r_cut = 3,
         array_size = 2,
         array_size_union = 2,
         code ='return -3;',
         code_union='return -1;',
         llvm_ir_file_union="code_union.ll",
         llvm_ir_file="code.ll",
         clang_exec='/usr/bin/clang'),
    dict(r_cut_union = 3,
         array_size_union = 2,
         code_union='return -1;',
         llvm_ir_file_union="code_union.ll"),
]


# setable attributes before attach for UserPatch objects
valid_attrs = [('r_cut', 1.4),
               ('code', 'return -1;'),
               ('llvm_ir_file', 'code.ll'),
               ('clang_exec', 'clang')
]


# setable attributes before attach for UserUnionPatch objects
valid_attrs_union = valid_attrs + [ ('r_cut_union', 3.6),
                                    ('code_union', 'return -2;'),
                                    ('llvm_ir_file_union', 'code_union.ll'),
                                    ('leaf_capacity', 2)
]


# setable attributes after attach for UserPatch objects
valid_attrs_after_attach = [ ('r_cut', 1.3)
]


# setable attributes after attach for UserUnionPatch objects
valid_attrs_after_attach_union = valid_attrs_after_attach + [ ('r_cut_union', 2),
                                                              ('leaf_capacity',3)
]


# attributes that cannot be set after object is attached
attr_error =  [('array_size', 1.1),
               ('code', 'return -1.0;'),
               ('llvm_ir_file', 'test.ll'),
               ('clang_exec', '/usr/bin/clang')
]


# attributes that cannot be set after object is attached
attr_error_union =  attr_error+ [('array_size_union', 2.1),
                                 ('code_union', 'return -3.0;'),
                                 ('llvm_ir_file_union', 'test.ll')
]


positions_orientations_result = [([(0,0,0),(1,0,0)], [(1,0,0,0),(1,0,0,0)],-1),
                                 ([(0,0,0),(1,0,0)], [(1,0,0,0),(0,0,1,0)], 1),
                                 ([(0,0,0),(0,0,1)], [(1,0,0,0),(1,0,0,0)], 1/2),
                                 ([(0,0,0),(0,0,1)], [(1,0,0,0),(0,0,1,0)], -1/2),
]


@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_user_patch(constructor_args):
    """Test that UserPatch can be constructed with valid arguments."""
    patch = jit.patch.UserPatch(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(patch, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args_union)
def test_valid_construction_user_union_patch(constructor_args):
    """Test that UserUnionPatch can be constructed with valid arguments."""
    patch = jit.patch.UserUnionPatch(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(patch, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_and_attach_user_patch(simulation_factory,
                                       two_particle_snapshot_factory,
                                       constructor_args):
    """Test that UserPatch can be attached with valid arguments."""
    # create objects
    patch = jit.patch.UserPatch(**constructor_args)
    mc = hoomd.hpmc.integrate.Sphere(seed=1)
    mc.shape['A'] = dict(diameter=0)

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.operations += patch

    # create C++ mirror classes and set parameters
    sim.run(0)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(patch, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args_union)
def test_valid_construction_and_attach_user_union_patch(simulation_factory,
                                       two_particle_snapshot_factory,
                                       constructor_args):
    """Test that UserUnionPatch can be attached with valid arguments."""
    # create objects
    patch = jit.patch.UserUnionPatch(**constructor_args)
    patch.positions['A'] = [(0,0,0.5), (0,0,-0.5)]
    patch.orientations['A'] = [(1,0,0,0)]*2
    patch.diameters['A'] = [0, 0]
    patch.typeids['A'] = [0, 0]
    patch.charges['A'] = [0, 0]

    mc = hoomd.hpmc.integrate.Sphere(seed=1)
    mc.shape['A'] = dict(diameter=0)

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.operations += patch

    # create C++ mirror classes and set parameters
    sim.run(0)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(patch, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_user_patch(attr, value):
    """Test that UserPatch can get and set attributes before attached."""
    patch = jit.patch.UserPatch(r_cut=2)

    setattr(patch, attr, value)
    assert getattr(patch, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("attr,value", valid_attrs_union)
def test_valid_setattr_user_union_patch(attr, value):
    """Test that UserUnionPatch can get and set attributes before attached."""

    patch = jit.patch.UserUnionPatch(r_cut_union=2)

    setattr(patch, attr, value)
    assert getattr(patch, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("attr,value", valid_attrs_after_attach)
def test_valid_setattr_attached_user_patch(attr, value,
                                           simulation_factory,
                                           two_particle_snapshot_factory):
    """Test that UserPatch can get and set attributes after attached."""

    patch = jit.patch.UserPatch(r_cut=2, code='return -1;')
    mc = hoomd.hpmc.integrate.Sphere(seed=1)
    mc.shape['A'] = dict(diameter=0)

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.operations += patch

    # create C++ mirror classes and set parameters
    sim.run(0)

    # validate the params were set properly
    setattr(patch, attr, value)
    assert getattr(patch, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("attr,value", valid_attrs_after_attach_union)
def test_valid_setattr_attached_user_union_patch(attr, value,
                                                 simulation_factory,
                                                 two_particle_snapshot_factory):
    """Test that UserUnionPatch can get and set attributes after attached."""

    patch = jit.patch.UserUnionPatch(r_cut_union=2, code_union='return 0;')
    patch.positions['A'] = [(0,0,0.5), (0,0,-0.5)]
    patch.orientations['A'] = [(1,0,0,0)]*2
    patch.diameters['A'] = [0, 0]
    patch.typeids['A'] = [0, 0]
    patch.charges['A'] = [0, 0]

    mc = hoomd.hpmc.integrate.Sphere(seed=1)
    mc.shape['A'] = dict(diameter=0)

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.operations += patch

    # create C++ mirror classes and set parameters
    sim.run(0)

    # validate the params were set properly
    setattr(patch, attr, value)
    assert getattr(patch, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("attr,val",  attr_error)
def test_raise_attr_error_user_patch(attr, val,
                                     simulation_factory,
                                     two_particle_snapshot_factory):
    """Test that UserPatch raises AttributeError if we
       try to set certain attributes after attaching.
    """

    patch = jit.patch.UserPatch(r_cut=2,
                                code='return 0;',
                                llvm_ir_file='code.ll')

    mc = hoomd.hpmc.integrate.Sphere(seed=1)
    mc.shape['A'] = dict(diameter=0)
    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.operations += patch
    sim.run(0)
    # try to reset when attached
    with pytest.raises(AttributeError):
        setattr(patch, attr, val)


@pytest.mark.serial
@pytest.mark.parametrize("attr,val",  attr_error)
def test_raise_attr_error_user_union_patch(attr, val,
                                           simulation_factory,
                                           two_particle_snapshot_factory):
    """Test that UserunionPatch raises AttributeError if we
       try to set certain attributes after attaching.
    """
    patch = jit.patch.UserUnionPatch(r_cut_union=2,
                                     code_union='return 0;',
                                     llvm_ir_file_union ='code.ll')

    patch.positions['A'] = [(0,0,0.5), (0,0,-0.5)]
    patch.orientations['A'] = [(1,0,0,0)]*2
    patch.diameters['A'] = [0, 0]
    patch.typeids['A'] = [0, 0]
    patch.charges['A'] = [0, 0]

    mc = hoomd.hpmc.integrate.Sphere(seed=1)
    mc.shape['A'] = dict(diameter=0)
    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.operations += patch
    sim.run(0)

    # try to reset when attached
    with pytest.raises(AttributeError):
        setattr(patch, attr, val)

    if attr != 'clang_exec':
        with pytest.raises(AttributeError):
            setattr(patch, attr + '_union', val)


@pytest.mark.serial
@pytest.mark.parametrize("positions,orientations,result",
                          positions_orientations_result)
def test_user_patch(positions,orientations,result,
                    simulation_factory, two_particle_snapshot_factory):
    """Test that UserPatch computes the correct."""

    # interaction between point dipoles
    dipole_dipole = """ float rsq = dot(r_ij, r_ij);
                        float r_cut = {0};
                        if (rsq < r_cut*r_cut)
                            {{
                            float lambda = 1.0;
                            float r = fast::sqrt(rsq);
                            float r3inv = 1.0 / rsq / r;
                            vec3<float> t = r_ij / r;
                            vec3<float> pi_o(1,0,0);
                            vec3<float> pj_o(1,0,0);
                            rotmat3<float> Ai(q_i);
                            rotmat3<float> Aj(q_j);
                            vec3<float> pi = Ai * pi_o;
                            vec3<float> pj = Aj * pj_o;
                            float Udd = (lambda*r3inv/2)*( dot(pi,pj)
                                         - 3 * dot(pi,t) * dot(pj,t));
                            return Udd;
                           }}
                        else
                            return 0.0f;
                    """

    sim = simulation_factory(two_particle_snapshot_factory())

    r_cut = sim.state.box.Lx/2.
    patch = jit.patch.UserPatch(r_cut=r_cut, code=dipole_dipole.format(r_cut))
    mc = hoomd.hpmc.integrate.Sphere(d=0.05, seed=1)
    mc.shape['A'] = dict(diameter=0)

    sim.operations.integrator = mc
    sim.operations += patch
    with sim.state.cpu_local_snapshot as data:
        data.particles.position[0, :] = positions[0]
        data.particles.position[1, :] = positions[1]
        data.particles.orientation[0, :] = orientations[0]
        data.particles.orientation[1, :] = orientations[1]
    sim.run(0)

    assert np.isclose(patch.energy, result)


@pytest.mark.serial
@pytest.mark.parametrize("cls", [jit.patch.UserPatch, jit.patch.UserUnionPatch])
def test_alpha_iso(cls, simulation_factory,two_particle_snapshot_factory):
    """Test that: i) changes to the alpha_iso array reflect on the energy
       caltulation, ii) that it can be accessed from UserPatch and UserUnionPatch
       objects and iii) that the energy computed from both classes agree.
    """

    lennard_jones =  """
                     float rsq = dot(r_ij, r_ij);
                     float rcut  = alpha_iso[0];
                     if (rsq <= rcut*rcut)
                         {{
                         float sigma = alpha_iso[1];
                         float eps   = alpha_iso[2];
                         float sigmasq = sigma*sigma;
                         float rsqinv = sigmasq / rsq;
                         float r6inv = rsqinv*rsqinv*rsqinv;
                         return 4.0f*eps*r6inv*(r6inv-1.0f);
                         }}
                     else
                         {{
                         return 0.0f;
                         }}
                     """

    sim = simulation_factory(two_particle_snapshot_factory())

    r_cut = 5
    params = dict(code=lennard_jones, array_size=3, r_cut=r_cut)
    if "union" in cls.__name__.lower():
        params.update({"r_cut_union": 0})
        params.update({"code_union": "return 0;"})
    patch = cls(**params)
    if "union" in cls.__name__.lower():
        patch.positions['A'] = [(0,0,0)]
        patch.orientations['A'] = [(1,0,0,0)]
        patch.diameters['A'] = [0]
        patch.typeids['A'] = [0]
        patch.charges['A'] = [0]
    mc = hoomd.hpmc.integrate.Sphere(d=0, seed=1)
    mc.shape['A'] = dict(diameter=0)

    sim.operations.integrator = mc
    sim.operations += patch

    dist = 1
    with sim.state.cpu_local_snapshot as data:
        data.particles.position[0, :] = (0,0,0)
        data.particles.position[1, :] = (dist,0,0)

    sim.run(0)
    # set alpha to sensible LJ values: [rcut, sigma, epsilon]
    patch.alpha_iso[0] = 2.5;
    patch.alpha_iso[1] = 1.2;
    patch.alpha_iso[2] = 1;
    energy_old = patch.energy

    # make sure energies are calculated properly when using alpha
    sigma_r_6 = (patch.alpha_iso[1] / dist)**6
    energy_actual = 4.0*patch.alpha_iso[2]*sigma_r_6*(sigma_r_6-1.0)
    assert np.isclose(energy_old, energy_actual)

    # double epsilon and check that energy is doubled
    patch.alpha_iso[2] = 2
    sim.run(0)
    assert np.isclose(patch.energy, 2.0*energy_old)

    # set r_cut to zero and check energy is zero
    patch.alpha_iso[0] = 0
    sim.run(0)
    assert np.isclose(patch.energy, 0.0)


@pytest.mark.serial
def test_user_union_patch(simulation_factory,two_particle_snapshot_factory):
    """Test that alpha_iso and alpha_union are accessed proberly by
       UserUnionPatch and that it computes the correct energy for
       unions of particles (dumbells).
    """

    sim = simulation_factory(two_particle_snapshot_factory())
    patch = jit.patch.UserUnionPatch(r_cut_union=1.4, r_cut=0)
    patch.code = "return alpha_iso[0];"
    patch.code_union = "return alpha_union[0];"

    # define dumbell parameters
    patch.positions['A'] = [(0,0,0.5), (0,0,-0.5)]
    patch.orientations['A'] = [(1,0,0,0)]*2
    patch.diameters['A'] = [0, 0]
    patch.typeids['A'] = [0, 0]
    patch.charges['A'] = [0, 0]

    mc = hoomd.hpmc.integrate.Sphere(seed=2, d=0, a=0)
    mc.shape["A"] = dict(diameter=0)

    # attach objects
    sim.operations.integrator = mc
    sim.operations += patch

    # place dumbells side by side (parallel along the axis joining the particles)
    with sim.state.cpu_local_snapshot as data:
        data.particles.position[0, :] = (0, 0.5, 0)
        data.particles.position[1, :] = (0, -0.5, 0)
    sim.run(0)

    patch.alpha_union[0] = 0
    assert np.isclose(patch.energy, 0)

    patch.alpha_union[0] = -1.0
    assert np.isclose(patch.energy, -2.0)

    patch.alpha_union[0] = -1.5
    assert np.isclose(patch.energy, -3.0)

    # slightly increase the range to include diagonal interactions
    patch.r_cut_union = 1.5
    assert np.isclose(patch.energy, -6.0)

    # add isotropic repulsion
    patch.r_cut = 1.5
    patch.alpha_iso[0] = 2.5
    assert np.isclose(patch.energy, -3.5)
