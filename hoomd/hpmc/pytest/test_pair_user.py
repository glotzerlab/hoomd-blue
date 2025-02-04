# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.pair.user.CPPPotential."""

import hoomd
import pytest
import numpy as np
from hoomd.conftest import autotuned_kernel_parameter_check

# check if llvm_enabled
llvm_disabled = not hoomd.version.llvm_enabled

valid_constructor_args = [
    dict(
        r_cut=3,
        param_array=[0, 1],
        code='return -1;',
    ),
    dict(
        r_cut=1.0,
        param_array=[],
        code='return 0.0f;',
    ),
]

# setable attributes before attach for CPPPotential objects
valid_attrs = [('r_cut', 1.4), ('code', 'return -1;')]

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


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_constructor_before_attach_cpp_potential(device,
                                                       constructor_args):
    """Test that CPPPotential can be constructed with valid arguments.

    This test also tests that the properties can be modified before attaching.

    """
    patch = hoomd.hpmc.pair.user.CPPPotential(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert np.all(getattr(patch, attr) == value)


@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setting_before_attach_cpp_potential(device, attr, value):
    """Test that CPPPotential can be constructed with valid arguments.

    This test also tests that the properties can be modified before attaching.

    """
    patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=3,
                                              param_array=[0, 1],
                                              code='return -1;')

    # ensure we can set properties
    setattr(patch, attr, value)
    assert getattr(patch, attr) == value


@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_attaching(device, simulation_factory, two_particle_snapshot_factory):
    patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=3,
                                              param_array=[0, 1],
                                              code='return -1;')
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
    patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=1.1,
                                              param_array=[0, 1],
                                              code='return -1;')
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
    patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=3,
                                              param_array=[0, 1],
                                              code='return -1;')
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
    mc.pair_potential = patch
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc
    sim.run(0)
    sim.operations.remove(mc)
    assert not mc._attached
    assert not patch._attached


@pytest.mark.validate
@pytest.mark.parametrize("attr_set,value_set", valid_attrs_after_attach)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_modify_after_attach_cpp_potential(device, simulation_factory,
                                           two_particle_snapshot_factory,
                                           attr_set, value_set):
    """Test that CPPPotential can be attached with valid arguments."""
    # create objects
    patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=3,
                                              param_array=[0, 1],
                                              code='return -1;')
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
    mc.pair_potential = patch

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # create C++ mirror classes and set parameters
    sim.run(0)

    # validate the params were set properly
    sim.run(0)
    setattr(patch, attr_set, value_set)
    assert getattr(patch, attr_set) == value_set


@pytest.mark.validate
@pytest.mark.parametrize("err_attr,err_val", attr_error)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_error_after_attach_cpp_potential(device, simulation_factory,
                                          two_particle_snapshot_factory,
                                          err_attr, err_val):
    """Test that CPPPotential can be attached with valid arguments."""
    # create objects
    patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=3,
                                              param_array=[0, 1],
                                              code='return -1;')
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
    mc.pair_potential = patch

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    # create C++ mirror classes and set parameters
    sim.run(0)

    # make sure we can't set properties than can't be set after attach
    sim.run(0)
    with pytest.raises(AttributeError):
        setattr(patch, err_attr, err_val)


@pytest.mark.validate
@pytest.mark.parametrize("positions,orientations,result",
                         positions_orientations_result)
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_cpp_potential(device, positions, orientations, result,
                       simulation_factory, two_particle_snapshot_factory):
    """Test that CPPPotential computes the correct energy.

    Here, we test the interaction between point dipoles and ensure the energy
    is what we expect.

    """
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

    r_cut = sim.state.box.Lx / 2. * 0.4
    patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=r_cut,
                                              code=dipole_dipole.format(r_cut),
                                              param_array=[])
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.pair_potential = patch

    sim.operations.integrator = mc

    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        snap.particles.position[0, :] = positions[0]
        snap.particles.position[1, :] = positions[1]
        snap.particles.orientation[0, :] = orientations[0]
        snap.particles.orientation[1, :] = orientations[1]
    sim.state.set_snapshot(snap)
    sim.run(0)

    assert np.isclose(patch.energy, result)


@pytest.mark.validate
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_param_array(device, simulation_factory, two_particle_snapshot_factory):
    """Test passing in parameter arrays to the patch object.

    This test tests that changes to the parameter array are reflected on the
    energy calculation.

    """
    lennard_jones = """
                     float rsq = dot(r_ij, r_ij);
                     float rcut = param_array[0];
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

    sim = simulation_factory(two_particle_snapshot_factory(L=50))

    r_cut = 5
    patch = hoomd.hpmc.pair.user.CPPPotential(code=lennard_jones,
                                              param_array=[2.5, 1.2, 1.0],
                                              r_cut=r_cut)
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=0)
    mc.pair_potential = patch

    sim.operations.integrator = mc

    dist = 1

    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        snap.particles.position[0] = [0, 0, 0]
        snap.particles.position[1] = [dist, 0, 0]
    sim.state.set_snapshot(snap)

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


@pytest.mark.validate
@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
def test_cpp_potential_sticky_spheres(device, simulation_factory,
                                      two_particle_snapshot_factory):
    """Validate the behavior of the CPPPotential class for sticky spheres.

    This test constructs a system of 2 hard spheres with a very deep, very
    short-ranged square well attraction at the surface of the sphere. Given the
    depth of the attracive well, any move that separates the particles beyond
    their range of attraction should be rejected, and the particles will always
    remain within range of the attraction.

    """
    max_r_interact = 1.003
    square_well = r'''float rsq = dot(r_ij, r_ij);
                    float epsilon = param_array[0];
                    float rcut = {:.16f}f;
                    if (rsq > rcut * rcut)
                        return 0.0f;
                    else
                        return -epsilon;
    '''.format(max_r_interact)

    sim = simulation_factory(two_particle_snapshot_factory(d=2, L=100))

    r_cut = 1.5
    patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=r_cut,
                                              code=square_well,
                                              param_array=[100.0])
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
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

    # first make sure the particles remain stuck together
    for step in range(10):
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            dist = np.linalg.norm(snap.particles.position[0]
                                  - snap.particles.position[1])
            assert dist < max_r_interact

    # now make the interaction repulsive and make sure the particles separate
    patch.param_array[0] = -100.0
    for step in range(10):
        sim.run(100)
        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            dist = np.linalg.norm(snap.particles.position[0]
                                  - snap.particles.position[1])
            assert dist > max_r_interact
