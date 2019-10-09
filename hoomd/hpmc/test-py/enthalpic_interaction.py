from __future__ import division
from __future__ import print_function

import hoomd
from hoomd import context, data, init, analyze, lattice
from hoomd import hpmc, jit

import unittest
import os
import numpy as np
import itertools
import sys

context.initialize();

class enthalpic_dipole_interaction(unittest.TestCase):
        def setUp(self):
            self.lamb = 1      # Dipolar coupling constant: mu0*m^2/ (2 * pi * d^3 * K * T)
            self.r_cut = 15;   # interaction cut-off
            self.diameter = 1; # particles diameter
            # interaction between point dipoles
            self.dipole_dipole = """ float rsq = dot(r_ij, r_ij);
                                float r_cut = {0};
                                if (rsq < r_cut*r_cut)
                                    {{
                                    float lambda = {1};
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
                            """.format(self.r_cut,self.lamb);
            self.snapshot = data.make_snapshot(N=2, box=data.boxdim(L=2*self.r_cut, dimensions=3), particle_types=['A']);

        # a) particle 1 on top of particle 2: parallel dipole orientations
        def test_head_to_tail_parallel(self):
            self.snapshot.particles.position[0,:]    = (0,0,0);
            self.snapshot.particles.position[1,:]    = (self.diameter,0,0);
            self.snapshot.particles.orientation[0,:] = (1,0,0,0);
            self.snapshot.particles.orientation[1,:] = (1,0,0,0);
            init.read_snapshot(self.snapshot);
            self.mc = hpmc.integrate.sphere(seed=10,a=0,d=0);
            self.mc.shape_param.set('A', diameter=self.diameter,orientable=True);
            self.patch = jit.patch.user(mc=self.mc,r_cut=self.r_cut, code=self.dipole_dipole);
            self.log = analyze.log(filename=None, quantities=['hpmc_patch_energy'],period=0,overwrite=True);
            hoomd.run(0, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), -self.lamb);

            # Disable patch with log = True and check logged energy is correct
            self.patch.disable(log=True);
            hoomd.run(2, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), -self.lamb);

            # Re-enable patch and check energy is correct again
            self.patch.enable();
            hoomd.run(2, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), -self.lamb);

            # Disable patch w/o log option and check energy is 0
            self.patch.disable();
            hoomd.run(2, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), 0);


        # b) particle 1 on top of particle 2: antiparallel dipole orientations
        def test_head_to_tail_antiparallel(self):
            self.snapshot.particles.position[0,:]    = (0,0,0);
            self.snapshot.particles.position[1,:]    = (self.diameter,0,0);
            self.snapshot.particles.orientation[0,:] = (1,0,0,0);
            self.snapshot.particles.orientation[1,:] = (0,0,1,0);
            init.read_snapshot(self.snapshot);
            self.mc = hpmc.integrate.sphere(seed=10,a=0,d=0);
            self.mc.shape_param.set('A', diameter=self.diameter,orientable=True);
            self.patch = jit.patch.user(mc=self.mc,r_cut=self.r_cut, code=self.dipole_dipole);
            self.log = analyze.log(filename=None, quantities=['hpmc_patch_energy'],period=0,overwrite=True)
            hoomd.run(0, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), self.lamb);

            # Disable patch with log = True and check logged energy is correct
            self.patch.disable(log=True);
            hoomd.run(2, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), self.lamb);

            # Re-enable patch and check energy is correct again
            self.patch.enable();
            hoomd.run(2, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), self.lamb);

            # Disable patch w/o log option and check energy is 0
            self.patch.disable();
            hoomd.run(2, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), 0);

        # c) particles side by side: parallel dipole orientations
        def test_side_to_side_parallel(self):
            self.snapshot.particles.position[0,:]    = (0,0,0);
            self.snapshot.particles.position[1,:]    = (0,0,self.diameter);
            self.snapshot.particles.orientation[0,:] = (1,0,0,0);
            self.snapshot.particles.orientation[1,:] = (1,0,0,0);
            init.read_snapshot(self.snapshot);
            self.mc = hpmc.integrate.sphere(seed=10,a=0,d=0);
            self.mc.shape_param.set('A', diameter=self.diameter,orientable=True);
            self.patch = jit.patch.user(mc=self.mc,r_cut=self.r_cut, code=self.dipole_dipole);
            self.log = analyze.log(filename=None, quantities=['hpmc_patch_energy'],period=0,overwrite=True)
            hoomd.run(0, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), self.lamb/2);

            # Disable patch with log = True and check logged energy is correct
            self.patch.disable(log=True);
            hoomd.run(2, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), self.lamb/2);

            # Re-enable patch and check energy is correct again
            self.patch.enable();
            hoomd.run(2, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), self.lamb/2);

            # Disable patch w/o log option and check energy is 0
            self.patch.disable();
            hoomd.run(2, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), 0);

        # d) particles side by side: antiparallel dipole orientations
        def test_side_to_side_parallel(self):
           self.snapshot.particles.position[0,:]    = (0,0,0);
           self.snapshot.particles.position[1,:]    = (0,0,self.diameter);
           self.snapshot.particles.orientation[0,:] = (1,0,0,0);
           self.snapshot.particles.orientation[1,:] = (0,0,1,0);
           init.read_snapshot(self.snapshot);
           self.mc = hpmc.integrate.sphere(seed=10,a=0,d=0);
           self.mc.shape_param.set('A', diameter=self.diameter,orientable=True);
           self.patch = jit.patch.user(mc=self.mc,r_cut=self.r_cut, code=self.dipole_dipole);
           self.log = analyze.log(filename=None, quantities=['hpmc_patch_energy'],period=0,overwrite=True)
           hoomd.run(0, quiet=True);
           self.assertEqual(self.log.query('hpmc_patch_energy'), -self.lamb/2);

           # Disable patch with log = True and check logged energy is correct
           self.patch.disable(log=True);
           hoomd.run(2, quiet=True);
           self.assertEqual(self.log.query('hpmc_patch_energy'), -self.lamb/2);

           # Re-enable patch and check energy is correct again
           self.patch.enable();
           hoomd.run(2, quiet=True);
           self.assertEqual(self.log.query('hpmc_patch_energy'), -self.lamb/2);

           # Disable patch w/o log option and check energy is 0
           self.patch.disable();
           hoomd.run(2, quiet=True);
           self.assertEqual(self.log.query('hpmc_patch_energy'), 0);

        # test the isotropic part of patch.user_union
        def test_head_to_tail_parallel_union(self):
            self.snapshot.particles.position[0,:]    = (0,0,0);
            self.snapshot.particles.position[1,:]    = (self.diameter,0,0);
            self.snapshot.particles.orientation[0,:] = (1,0,0,0);
            self.snapshot.particles.orientation[1,:] = (1,0,0,0);
            init.read_snapshot(self.snapshot);
            self.mc = hpmc.integrate.sphere(seed=10,a=0,d=0);
            self.mc.shape_param.set('A', diameter=self.diameter,orientable=True);
            self.patch = jit.patch.user_union(mc=self.mc,r_cut_iso=self.r_cut, code_iso=self.dipole_dipole,
                r_cut=0.0, code='return 0.0;');
            self.log = analyze.log(filename=None, quantities=['hpmc_patch_energy'],period=0,overwrite=True);
            hoomd.run(0, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), -self.lamb);

            # Disable patch with log = True and check logged energy is correct
            self.patch.disable(log=True);
            hoomd.run(2, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), -self.lamb);

            # Re-enable patch and check energy is correct again
            self.patch.enable();
            hoomd.run(2, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), -self.lamb);

            # Disable patch w/o log option and check energy is 0
            self.patch.disable();
            hoomd.run(2, quiet=True);
            self.assertEqual(self.log.query('hpmc_patch_energy'), 0);

        def tearDown(self):
            del self.mc;
            del self.patch;
            del self.snapshot;
            del self.log;
            context.initialize();

class patch_alpha_user(unittest.TestCase):

    def setUp(self):
        lennard_jones = """
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
        self.dist = 2.0; # distance between test particles
        snapshot = data.make_snapshot(N=2, box=data.boxdim(L=10, dimensions=3), particle_types=['A']);
        snapshot.particles.position[0,:] = (0,0,0);
        snapshot.particles.position[1,:] = (self.dist,0,0);
        system = init.read_snapshot(snapshot);
        mc = hpmc.integrate.sphere(seed=1,d=0);
        mc.shape_param.set('A',diameter=0);
        self.patch = jit.patch.user(mc=mc, r_cut=2.5, array_size=3, code=lennard_jones);
        self.logger = analyze.log(filename=None, quantities=["hpmc_patch_energy"], period=1);

    def test_alphas(self):

        # raise error if array is larger than allocated memory
        with self.assertRaises(ValueError):
            self.patch.alpha_iso[:] = [1]*4;

        self.assertAlmostEqual(len(self.patch.alpha_iso), 3)

        # check alpha array is set properly
        self.patch.alpha_iso[:] = [-1., 2.7, 10];
        np.testing.assert_allclose(self.patch.alpha_iso, [-1., 2.7, 10]);

        # set alpha to sensible LJ values: [rcut, sigma, epsilon]
        self.patch.alpha_iso[0] = 2.5;
        self.patch.alpha_iso[1] = 1.2;
        self.patch.alpha_iso[2] = 1;
        np.testing.assert_allclose(self.patch.alpha_iso, [2.5, 1.2, 1]);

        # get energy for previus LJ params
        hoomd.run(0, quiet=True);
        energy_old = self.logger.query("hpmc_patch_energy");
        # make sure energies are calculated properly when using alpha
        sigma_r_6 = (self.patch.alpha_iso[1] / self.dist)**6;
        energy_actual = 4.0*self.patch.alpha_iso[2]*sigma_r_6*(sigma_r_6-1.0);
        self.assertAlmostEqual(energy_old, energy_actual);

        # double epsilon
        self.patch.alpha_iso[2] = 2;
        hoomd.run(1);
        # make sure energy is doubled
        energy_new = self.logger.query("hpmc_patch_energy");
        self.assertAlmostEqual(energy_new, 2.0*energy_old);

        # set r_cut to zero and check energy is zero
        self.patch.alpha_iso[0] = 0;
        hoomd.run(1);
        self.assertAlmostEqual(self.logger.query("hpmc_patch_energy"), 0);

    def tearDown(self):
        del self.logger
        del self.patch
        context.initialize();

class patch_alpha_user_union(unittest.TestCase):

    def setUp(self):
        # square well attraction on constituent spheres
        square_well = """float rsq = dot(r_ij, r_ij);
                              float r_cut = alpha_union[0];
                              if (rsq < r_cut*r_cut)
                                  return alpha_union[1];
                              else
                                  return 0.0f;
                           """

        # soft repulsion between centers of unions
        soft_repulsion = """float rsq = dot(r_ij, r_ij);
                                  float r_cut = alpha_iso[0];
                                  if (rsq < r_cut*r_cut)
                                    return alpha_iso[1];
                                  else
                                    return 0.0f;
                         """
        diameter = 1.0;
        snapshot = data.make_snapshot(N=2, box=data.boxdim(L=10, dimensions=3), particle_types=['A']);
        snapshot.particles.position[0,:] = (0,0,0);
        snapshot.particles.position[1,:] = (diameter,0,0);
        system = init.read_snapshot(snapshot);
        mc = hpmc.integrate.sphere_union(d=0,a=0,seed=1);
        mc.shape_param.set('A',diameters=[diameter]*2, centers=[(0,0,-diameter/2),(0,0,diameter/2)],overlap=[0]*2);
        self.patch = jit.patch.user_union(mc=mc, r_cut=2.5, array_size=2, r_cut_iso=2.5, array_size_iso=2, \
                                          code=square_well, code_iso=soft_repulsion);
        self.patch.set_params('A',positions=[(0,0,-diameter/2),(0,0,diameter/2)], typeids=[0,0])
        self.logger = analyze.log(filename=None, quantities=["hpmc_patch_energy"], period=1);

    def test_alphas(self):

        with self.assertRaises(ValueError):
            self.patch.alpha_iso[:] = [1]*4;

        with self.assertRaises(ValueError):
            self.patch.alpha_union[:] = [1]*4;

        self.assertAlmostEqual(len(self.patch.alpha_iso), 2);
        self.assertAlmostEqual(len(self.patch.alpha_union), 2);

        self.patch.alpha_iso[:] = [-1.2, 43.0];
        self.patch.alpha_union[:] = [2.67, -1.7];
        np.testing.assert_allclose(self.patch.alpha_iso, [-1.2, 43.0]);
        np.testing.assert_allclose(self.patch.alpha_union, [2.67, -1.7]);

        self.patch.alpha_iso[0] = 2.5;
        self.patch.alpha_iso[1] = 1.3;
        self.patch.alpha_union[0] = 2.5;
        self.patch.alpha_union[1] = -1.7;
        hoomd.run(1)
        self.assertAlmostEqual(self.logger.query("hpmc_patch_energy"), -5.5);

        self.patch.alpha_union[0] = 0;
        hoomd.run(1)
        self.assertAlmostEqual(self.logger.query("hpmc_patch_energy"), self.patch.alpha_iso[1]);
        self.patch.alpha_union[0] = 2.5;

        self.patch.alpha_iso[0] = 0;
        hoomd.run(1)
        self.assertAlmostEqual(self.logger.query("hpmc_patch_energy"), 4*self.patch.alpha_union[1]);
        self.patch.alpha_iso[0] = 2.5;

    def tearDown(self):
        del self.logger
        del self.patch
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
