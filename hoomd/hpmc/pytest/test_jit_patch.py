# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test hoomd.jit.patch.UserPatch."""

import hoomd
from hoomd import jit
import pytest
import numpy as np

positions_orientations_result = [
                                 ([(0,0,0),(1,0,0)], [(1,0,0,0),(1,0,0,0)],-1),
                                 ([(0,0,0),(1,0,0)], [(1,0,0,0),(0,0,1,0)], 1),
                                 ([(0,0,0),(0,0,1)], [(1,0,0,0),(1,0,0,0)], 1/2),
                                 ([(0,0,0),(0,0,1)], [(1,0,0,0),(0,0,1,0)], -1/2),
]

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

@pytest.mark.parametrize("positions,orientations,result",
                          positions_orientations_result)
def test_user_patch(positions,orientations,result,
                    simulation_factory, two_particle_snapshot_factory):
    """"""

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


@pytest.mark.parametrize("cls", [jit.patch.UserPatch, jit.patch.UserUnionPatch])
def test_alpha_iso(cls, simulation_factory,two_particle_snapshot_factory):
    """"""

    sim = simulation_factory(two_particle_snapshot_factory())

    r_cut = sim.state.box.Lx/2
    params = dict(code=lennard_jones, array_size=3)
    if "union" in cls.__name__.lower():
        params.update({"r_cut_union" :r_cut})
        params.update({"code_union" : "return 0;"})
    else:
        params.update({"r_cut" :r_cut})
    patch = cls(**params)
    mc = hoomd.hpmc.integrate.Sphere(d=0, seed=1)
    mc.shape['A'] = dict(diameter=0)

    sim.operations.integrator = mc
    sim.operations += patch

    dist = 2
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
    assert np.isclose(patch.energy, energy_old)

    # double epsilon and check that energy is doubled
    patch.alpha_iso[2] = 2
    sim.run(0)
    assert np.isclose(patch.energy, 2.0*energy_old)

    # set r_cut to zero and check energy is zero
    patch.alpha_iso[0] = 0
    sim.run(0)
    assert np.isclose(patch.energy, 0.0)
