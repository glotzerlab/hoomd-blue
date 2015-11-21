# -*- coding: iso-8859-1 -*-
# Maintainer: jproc

from hoomd_script import *
import unittest
import os
import numpy as np

#test wall.group()
class wall_group_tests(unittest.TestCase):

    # basic test of creation for walls structure
    def test(self):
        walls=wall.group();

    # test each type of geometry can be added and deleted
    def test_add_sphere(self):
        walls=wall.group();
        walls.add_sphere(r=4, origin=(0.0, 0.0, 0.0), inside=True);
        walls.del_sphere(0);

    def test_add_cylinder(self):
        walls=wall.group();
        walls.add_cylinder(r=4, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), inside=True);
        walls.del_cylinder(0);

    def test_add_plane(self):
        walls=wall.group();
        walls.add_plane(normal=(1.0, 0.0, 0.0), origin=(-4.0, 0.0, 0.0), inside=True);
        walls.add_plane(normal=(-1.0, 0.0, 0.0), origin=(4.0, 0.0, 0.0), inside=False);
        walls.del_plane([0,1]);


# test lj wall force in standard mode
class wall_lj_tests (unittest.TestCase):

    def setUp(self):
        self.s=init.create_random(N=100, box=data.boxdim(L=5));
        updater=update.box_resize(L = 15, Period=None);
        updater.set_params(scale_particles = False);
        walls=wall.group();
        walls.add_sphere(r=5, origin=(0.0, 0.0, 0.0), inside=True);

    # test to see that se can create a wall.lj
    def test_create(self):
        wall.lj(walls);

    # test coefficient not set checking
    def test_force_coeff_fail(self):
        lj_wall = wall.lj(walls);
        lj_wall.force_coeff.set('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 10);

    # test setting coefficients
    def test_force_coeff(self):
        lj_wall = wall.lj(walls, r_cut=3.0);
        lj_wall.force_coeff.set('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);

    def test_overload_structure_fail(self):
        walls.spheres=[wall.sphere()]*21;
        lj_wall = wall.lj(walls, r_cut=3.0);
        lj_wall.force_coeff.set('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 10);

    def tearDown(self):
        del self.s
        init.reset();

# test lj wall force in shifted mode
class wall_shift_tests (unittest.TestCase):

    def setUp(self):
        snap= data.make_snapshot(N=4, box=data.boxdim(L=10))
        coords=[[0.0,0.0,0.0],[1.0,1.0,1.0],[4.0,-2.0,1.0],[-3.2,1.0,-0.5]];
        for i in range(4):
            snap.particles.position[i]=coords[i]
        self.s=init.read_snapshot(snap)
        walls=wall.group();
        walls.add_plane(origin=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), inside=True);
        all = group.all();
        integrate.mode_standard(dt=0.0);
        integrate.nve(all);
        lj_wall=wall.lj(walls,r_cut=300)
        lj_wall.force_coeff.set('A', r_shift=1.1, epsilon=1.0, sigma=1.0)
        run(5)

    # test forces
    def test_forces(self):
        self.assertAlmostEqual(1.5881, self.s.particles.pdata.getPNetForce(0).x);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(0).y);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(0).z);
        self.assertAlmostEqual(-0.130145, self.s.particles.pdata.getPNetForce(1).x);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(1).y);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(1).z);
        self.assertAlmostEqual(-0.000267406, self.s.particles.pdata.getPNetForce(2).x);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(2).y);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(2).z);
        self.assertAlmostEqual(1.5881, self.s.particles.pdata.getPNetForce(3).x);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(3).y);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(3).z);

    def test_energy(self):
        self.assertAlmostEqual(-0.983372, self.s.particles.pdata.getPNetForce(0).w);
        self.assertAlmostEqual(-0.0460947, self.s.particles.pdata.getPNetForce(1).w);
        self.assertAlmostEqual(-0.000227308, self.s.particles.pdata.getPNetForce(2).w);
        self.assertAlmostEqual(4.09853, self.s.particles.pdata.getPNetForce(3).w);

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
