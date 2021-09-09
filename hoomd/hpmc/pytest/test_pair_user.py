# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test hoomd.hpmc.pair.user.CPPPotential and hoomd.hpmc.pair.user.CPPUnionPotential."""

import hoomd
import pytest
import numpy as np

# check if llvm_enabled
llvm_disabled = not hoomd.version.llvm_enabled

# this is the IR generated for a pair kernel that is just "return 0;"
return0_ir = """
; ModuleID = 'patch-code.cc'
source_filename = "patch-code.cc"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx11.0.0"

%struct.vec3 = type { float, float, float }
%struct.quat = type { float, %struct.vec3 }

@param_array = local_unnamed_addr global float* null, align 8
@alpha_union = local_unnamed_addr global float* null, align 8

; Function Attrs: norecurse nounwind readnone ssp uwtable
define float @eval(%struct.vec3* nocapture nonnull readnone align 4 dereferenceable(12) %0, i32 %1, %struct.quat* nocapture nonnull readnone align 4 dereferenceable(16) %2, float %3, float %4, i32 %5, %struct.quat* nocapture nonnull readnone align 4 dereferenceable(16) %6, float %7, float %8) local_unnamed_addr #0 {
  ret float 0.000000e+00
}

attributes #0 = { norecurse nounwind readnone ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 3]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{!"clang version 11.1.0"}
"""

valid_constructor_args = [
    dict(r_cut=3,
         param_array=[0, 1],
         code='return -1;',
         llvm_ir=return0_ir,
         clang_exec='/usr/bin/clang'),
    dict(r_cut=2, param_array=[1, 2, 3, 4], code='return -1;'),
]

# setable attributes before attach for CPPPotential objects
valid_attrs = [('r_cut', 1.4), ('code', 'return -1;'),
               ('llvm_ir', return0_ir), ('clang_exec', 'clang')]

# setable attributes after attach for CPPPotential objects
valid_attrs_after_attach = [('r_cut', 1.3)]

# attributes that cannot be set after object is attached
attr_error = [('code', 'return -1.0;')]

positions_orientations_result = [
    ([(0, 0, 0), (1, 0, 0)], [(1, 0, 0, 0), (1, 0, 0, 0)], -1),
    ([(0, 0, 0), (1, 0, 0)], [(1, 0, 0, 0), (0, 0, 1, 0)], 1),
    ([(0, 0, 0), (0, 0, 1)], [(1, 0, 0, 0), (1, 0, 0, 0)], 1 / 2),
    ([(0, 0, 0), (0, 0, 1)], [(1, 0, 0, 0), (0, 0, 1, 0)], -1 / 2),
]

attr_translator = {'code': '_code'}

@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_cpp_potential(device, constructor_args):
    """Test that CPPPotential can be constructed with valid arguments."""
    patch = hoomd.hpmc.pair.user.CPPPotential(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        attr = attr_translator.get(attr, attr)
        if attr == 'clang_exec':  # not saved as a class variable
            continue
        try:
            assert getattr(patch, attr) == value
        except ValueError:
            assert all(getattr(patch, attr) == value)


@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_valid_construction_and_attach_cpp_potential(
        device, simulation_factory, two_particle_snapshot_factory,
        constructor_args):
    """Test that CPPPotential can be attached with valid arguments."""
    # create objects
    patch = hoomd.hpmc.pair.user.CPPPotential(**constructor_args)
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
        attr = attr_translator.get(attr, attr)
        if attr == 'clang_exec':
            continue
        try:
            assert getattr(patch, attr) == value
        except ValueError:  # array-like
            assert all(getattr(patch, attr) == value)


@pytest.mark.serial
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_cpp_potential(device, attr, value):
    """Test that CPPPotential can get and set attributes before attached."""
    patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=2, code='return 0;')

    setattr(patch, attr, value)
    assert getattr(patch, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("attr,value", valid_attrs_after_attach)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_valid_setattr_attached_cpp_potential(device, attr, value,
                                              simulation_factory,
                                              two_particle_snapshot_factory):
    """Test that CPPPotential can get and set attributes after attached."""

    patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=2, code='return -1;')
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
def test_raise_attr_error_cpp_potential(device, attr, val, simulation_factory,
                                        two_particle_snapshot_factory):
    """Test that CPPPotential raises AttributeError if we
       try to set certain attributes after attaching.
    """

    patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=2,
                                              code='return 0;',
                                              llvm_ir=return0_ir)
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.potential = patch

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.run(0)

    # try to reset when attached
    with pytest.raises(AttributeError):
        setattr(patch, attr, val)


@pytest.mark.serial
@pytest.mark.parametrize("positions,orientations,result",
                         positions_orientations_result)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_cpp_potential(device, positions, orientations, result,
                       simulation_factory, two_particle_snapshot_factory):
    """Test that CPPPotential computes the correct."""

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

    sim = simulation_factory(two_particle_snapshot_factory(d=2, L=100))

    r_cut = sim.state.box.Lx / 2.
    patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=r_cut,
                                              code=dipole_dipole.format(r_cut))
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.potential = patch

    sim.operations.integrator = mc
    with sim.state.cpu_local_snapshot as data:
        data.particles.position[0, :] = positions[0]
        data.particles.position[1, :] = positions[1]
        data.particles.orientation[0, :] = orientations[0]
        data.particles.orientation[1, :] = orientations[1]
    sim.run(0)

    assert np.isclose(patch.energy, result)


@pytest.mark.serial
@pytest.mark.parametrize("cls", [hoomd.hpmc.pair.user.CPPPotential])
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_param_array(device, cls, simulation_factory,
                   two_particle_snapshot_factory):
    """Test that: i) changes to the param_array reflect on the energy
       caltulation, ii) that it can be accessed from CPPPotential and CPPUnionPotential
       objects and iii) that the energy computed from both classes agree.
    """

    lennard_jones = """
                     float rsq = dot(r_ij, r_ij);
                     float rcut  = param_array[0];
                     if (rsq <= rcut*rcut)
                         {{
                         float sigma = param_array[1];
                         float eps   = param_array[2];
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
    params = dict(code=lennard_jones, param_array=[2.5, 1.2, 1.0], r_cut=r_cut)
    if "union" in cls.__name__.lower():
        params.update({"r_cut_union": 0})
        params.update({"code_union": "return 0;"})
    patch = cls(**params)
    if "union" in cls.__name__.lower():
        patch.positions['A'] = [(0, 0, 0)]
        patch.orientations['A'] = [(1, 0, 0, 0)]
        patch.diameters['A'] = [0]
        patch.typeids['A'] = [0]
        patch.charges['A'] = [0]
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.potential = patch

    sim.operations.integrator = mc

    dist = 1
    with sim.state.cpu_local_snapshot as data:
        data.particles.position[0, :] = (0, 0, 0)
        data.particles.position[1, :] = (dist, 0, 0)

    sim.run(0)
    # set alpha to sensible LJ values: [rcut, sigma, epsilon]
    patch.param_array[0] = 2.5
    patch.param_array[1] = 1.2
    patch.param_array[2] = 1
    energy_old = patch.energy

    # make sure energies are calculated properly when using alpha
    sigma_r_6 = (patch.param_array[1] / dist)**6
    energy_actual = 4.0 * patch.param_array[2] * sigma_r_6 * (sigma_r_6 - 1.0)
    assert np.isclose(energy_old, energy_actual)

    # double epsilon and check that energy is doubled
    patch.param_array[2] = 2
    sim.run(0)
    assert np.isclose(patch.energy, 2.0 * energy_old)

    # set r_cut to zero and check energy is zero
    patch.param_array[0] = 0
    sim.run(0)
    assert np.isclose(patch.energy, 0.0)
