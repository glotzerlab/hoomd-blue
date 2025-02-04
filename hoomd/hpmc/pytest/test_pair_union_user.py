# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.pair.user.CPPPotentialUnion."""

import hoomd
import pytest
import numpy as np
from hoomd.conftest import autotuned_kernel_parameter_check

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


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_cpp_union_potential(device, constructor_args):
    """Test that CPPPotentialUnion can be constructed with valid arguments."""
    patch = hoomd.hpmc.pair.user.CPPPotentialUnion(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert np.all(getattr(patch, attr) == value)


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_valid_construction_and_attach_cpp_union_potential(
        device, simulation_factory, two_particle_snapshot_factory,
        constructor_args):
    """Test that CPPPotentialUnion can be attached with valid arguments."""
    # create objects
    patch = hoomd.hpmc.pair.user.CPPPotentialUnion(**constructor_args)
    patch.positions['A'] = [(0, 0, 0)]
    patch.orientations['A'] = [(1, 0, 0, 0)]
    patch.diameters['A'] = [1.0]
    patch.typeids['A'] = [0]
    patch.charges['A'] = [0]
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
    mc.pair_potential = patch

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # catch the error if running on the GPU
    if isinstance(device, hoomd.device.GPU):
        with pytest.raises(RuntimeError):
            sim.run(0)

    # create C++ mirror classes and set parameters
    sim.run(0)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert np.all(getattr(patch, attr) == value)


@pytest.mark.validate
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_attaching(device, simulation_factory, two_particle_snapshot_factory):
    patch = hoomd.hpmc.pair.user.CPPPotentialUnion(
        r_cut_isotropic=1.4,
        r_cut_constituent=1.0,
        code_isotropic='',
        code_constituent='return 6;',
        param_array_constituent=[],
        param_array_isotropic=[],
    )
    patch.positions['A'] = [(0, 0, 0)]
    patch.orientations['A'] = [(1, 0, 0, 0)]
    patch.diameters['A'] = [1.0]
    patch.typeids['A'] = [0]
    patch.charges['A'] = [0]
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
    mc.pair_potential = patch
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.run(0)
    assert mc._attached
    assert patch._attached


@pytest.mark.validate
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_kernel_parameters(simulation_factory, lattice_snapshot_factory):
    patch = hoomd.hpmc.pair.user.CPPPotentialUnion(
        r_cut_isotropic=1.4,
        r_cut_constituent=1.0,
        code_isotropic='',
        code_constituent='return 6;',
        param_array_constituent=[],
        param_array_isotropic=[],
    )
    patch.positions['A'] = [(0, 0, 0)]
    patch.orientations['A'] = [(1, 0, 0, 0)]
    patch.diameters['A'] = [1.0]
    patch.typeids['A'] = [0]
    patch.charges['A'] = [0]
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
    mc.pair_potential = patch
    sim = simulation_factory(lattice_snapshot_factory())
    sim.operations.integrator = mc
    sim.run(0)

    autotuned_kernel_parameter_check(instance=patch,
                                     activate=lambda: sim.run(1))


@pytest.mark.validate
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_detaching(device, simulation_factory, two_particle_snapshot_factory):
    patch = hoomd.hpmc.pair.user.CPPPotentialUnion(
        r_cut_isotropic=1.4,
        r_cut_constituent=1.0,
        code_isotropic='',
        code_constituent='return 6;',
        param_array_constituent=[],
        param_array_isotropic=[],
    )
    patch.positions['A'] = [(0, 0, 0)]
    patch.orientations['A'] = [(1, 0, 0, 0)]
    patch.diameters['A'] = [1.0]
    patch.typeids['A'] = [0]
    patch.charges['A'] = [0]
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
    mc.pair_potential = patch
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.run(0)
    sim.operations.remove(mc)
    assert not mc._attached
    assert not patch._attached


@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_cpp_union_potential(device, attr, value):
    """Test that CPPPotentialUnion can get and set attributes before \
            attached."""
    patch = hoomd.hpmc.pair.user.CPPPotentialUnion(
        r_cut_isotropic=1.4,
        r_cut_constituent=1.0,
        code_isotropic='return 5;',
        code_constituent='return 6;',
        param_array_constituent=[],
        param_array_isotropic=[],
    )

    setattr(patch, attr, value)
    assert getattr(patch, attr) == value


@pytest.mark.parametrize("attr,value", valid_attrs_after_attach)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_valid_setattr_attached_cpp_union_potential(
        device, attr, value, simulation_factory, two_particle_snapshot_factory):
    """Test that CPPPotentialUnion can get and set attributes after attached."""
    patch = hoomd.hpmc.pair.user.CPPPotentialUnion(
        r_cut_isotropic=1.4,
        r_cut_constituent=1.0,
        code_isotropic='return 5;',
        code_constituent='return 6;',
        param_array_constituent=[],
        param_array_isotropic=[],
    )
    patch.positions['A'] = [(0, 0, 0)]
    patch.orientations['A'] = [(1, 0, 0, 0)]
    patch.diameters['A'] = [1.0]
    patch.typeids['A'] = [0]
    patch.charges['A'] = [0]
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.pair_potential = patch

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # catch the error if running on the GPU
    if isinstance(device, hoomd.device.GPU):
        with pytest.raises(RuntimeError):
            sim.run(0)
        return

    # create C++ mirror classes and set parameters
    sim.run(0)

    # validate the params were set properly
    setattr(patch, attr, value)
    assert getattr(patch, attr) == value


@pytest.mark.parametrize("attr,val", attr_error)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_raise_attr_error_cpp_union_potential(device, attr, val,
                                              simulation_factory,
                                              two_particle_snapshot_factory):
    """Test that CPPPotentialUnion raises AttributeError if we \
            try to set certain attributes after attaching."""
    patch = hoomd.hpmc.pair.user.CPPPotentialUnion(
        r_cut_isotropic=1.4,
        r_cut_constituent=1.0,
        code_isotropic='return 5;',
        code_constituent='return 6;',
        param_array_constituent=[],
        param_array_isotropic=[],
    )
    patch.positions['A'] = [(0, 0, 0)]
    patch.orientations['A'] = [(1, 0, 0, 0)]
    patch.diameters['A'] = [1.0]
    patch.typeids['A'] = [0]
    patch.charges['A'] = [0]
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.pair_potential = patch

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # catch the error if running on the GPU
    if isinstance(device, hoomd.device.GPU):
        with pytest.raises(RuntimeError):
            sim.run(0)
        return

    # make sure the AttributeError gets raised when trying to change the
    # properties that cannot be changed after attachment
    # in this case, this is the jit-compiled code, because it gets compiled
    # upon attachment, so changing after attaching would return code that does
    # not reflect what was attached
    sim.run(0)
    with pytest.raises(AttributeError):
        setattr(patch, attr, val)


@pytest.mark.cpu
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_param_array_union_cpu(device, simulation_factory,
                               two_particle_snapshot_factory):
    """Test passing in parameter arrays to the union patch objects.

    This test tests that changes to the parameter array are reflected on the
    energy calculation. It therefore also tests that the energies are computed
    correctly. This test is for the CPU, where we can pass code into
    code_isotropic.

    """
    square_well_isotropic = """
                   float rsq = dot(r_ij, r_ij);
                   float rcut = param_array_isotropic[0];
                   if (rsq < rcut*rcut)
                       return param_array_isotropic[1];
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
    sim = simulation_factory(two_particle_snapshot_factory(L=40))
    r_cut_iso = 5
    patch = hoomd.hpmc.pair.user.CPPPotentialUnion(
        code_isotropic=square_well_isotropic,
        param_array_isotropic=[2.5, 1.0],
        r_cut_isotropic=r_cut_iso,
        code_constituent=square_well_constituent,
        param_array_constituent=[1.5, 3.0],
        r_cut_constituent=1.5,
    )
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
    mc.pair_potential = patch
    sim.operations.integrator = mc

    # test the case where r_cut_isotropic = 0, so particles only interact
    # through the constituent particles
    # there are 2 cases here, one where they interact with only 1 other
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


@pytest.mark.gpu
@pytest.mark.validate
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_param_array_union_gpu(device, simulation_factory,
                               two_particle_snapshot_factory):
    """Test passing in parameter arrays to the union patch objects.

    This test tests that changes to the parameter array are reflected on the
    energy calculation. It therefore also tests that the energies are computed
    correctly. We test this on the GPU where code_isotropic is unused, so we do
    not pass any code into that argument of the patch constructor.

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
    sim = simulation_factory(two_particle_snapshot_factory(L=40))
    patch = hoomd.hpmc.pair.user.CPPPotentialUnion(
        code_isotropic='',
        code_constituent=square_well_constituent,
        param_array_constituent=[1.5, 3.0],
        r_cut_constituent=1.5,
        r_cut_isotropic=0,
        param_array_isotropic=[],
    )
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
    mc.pair_potential = patch
    sim.operations.integrator = mc

    # there are 2 cases here, one where they interact with only 1 other
    # constituent particle, and then where the interact with all of the
    # neighboring constituent particles

    # first, only interact with nearest neighboring constituent particle
    patch.param_array_constituent[0] = 1.1
    patch.param_array_constituent[1] = -1.0
    sim.run(0)
    assert (np.isclose(patch.energy, -2.0))  # 2 interacting pairs

    # now extend r_cut_constituent so that all constituent particles interact
    # with all other constituent particles
    patch.param_array_constituent[0] = np.sqrt(2) + 0.1
    sim.run(0)
    assert (np.isclose(patch.energy, -4.0))  # 4 interacting pairs this time

    # change the depth of the square well to make sure it's getting read on the
    # GPU
    old_energy = patch.energy
    scale_factor = 3.0
    patch.param_array_constituent[1] *= scale_factor
    assert (np.isclose(old_energy * scale_factor, patch.energy))


@pytest.mark.validate
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_cpp_potential_union_sticky_spheres(device, simulation_factory,
                                            two_particle_snapshot_factory):
    """Validate the behavior of the CPPPotentialUnion class for sticky spheres.

    This test is analogous to the test_cpp_potential_sticky_sheres in
    test_pair_user, but instead a constituent particle is placed at the center
    of each "union" (just a union of 1 particle). The idea is that if the
    particles are sticky enough, they should never move out of each others'
    range of interaction.

    """
    max_r_interact = 1.003
    square_well = r'''float rsq = dot(r_ij, r_ij);
                    float epsilon = param_array_constituent[0];
                    float rcut = {:.16f}f;
                    if (rsq > rcut * rcut)
                        return 0.0f;
                    else
                        return -epsilon;
    '''.format(max_r_interact)

    sim = simulation_factory(two_particle_snapshot_factory(d=2, L=100))
    mc = hoomd.hpmc.integrate.SphereUnion()
    r_cut = 1.5
    patch = hoomd.hpmc.pair.user.CPPPotentialUnion(
        code_constituent=square_well,
        r_cut_constituent=r_cut,
        param_array_constituent=[100.0],
        r_cut_isotropic=0,
        code_isotropic='',
        param_array_isotropic=[],
    )

    origin = [0, 0, 0]
    patch.positions['A'] = [origin]
    patch.orientations['A'] = [(1, 0, 0, 0)]
    patch.diameters['A'] = [1.0]
    patch.typeids['A'] = [0]
    patch.charges['A'] = [0]
    sphere1 = dict(diameter=1)
    mc.shape["A"] = dict(shapes=[sphere1],
                         positions=[origin],
                         orientations=[(1, 0, 0, 0)])
    mc.pair_potential = patch

    sim.operations.integrator = mc

    # set the particle positions
    separation = 1.001
    with sim.state.cpu_local_snapshot as snapshot:
        N = len(snapshot.particles.position)
        for tag, r_x in zip([0, 1], [-separation / 2, separation / 2]):
            index = snapshot.particles.rtag[tag]
            if index < N:
                snapshot.particles.position[index, :] = [r_x, 0, 0]

    # first make sure particles remain "stuck"
    for step in range(10):
        sim.run(100)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            dist = np.linalg.norm(snap.particles.position[0]
                                  - snap.particles.position[1])
            assert dist < max_r_interact

    # now make the potential repulsive and make sure the particles move apart
    patch.param_array_constituent[0] = -100.0
    for step in range(10):
        sim.run(100)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            dist = np.linalg.norm(snap.particles.position[0]
                                  - snap.particles.position[1])
            assert dist > max_r_interact
