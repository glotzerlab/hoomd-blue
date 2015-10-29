# -*- coding: iso-8859-1 -*-
# Maintainer: jproc

from hoomd_script import *
import unittest
import os

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
        walls.add_plane(normal=(1.0, 0.0, 0.0), origin=(-4.0, 0.0, 0.0));
        walls.add_plane(normal=(-1.0, 0.0, 0.0), origin=(4.0, 0.0, 0.0));
        walls.del_plane([0,1]);

    def test_overload_structure_fail(self):
        walls=wall.group();
        walls.spheres=[wall.sphere_wall()]*(walls._max_n_sphere_walls+1);
        self.assertRaises(RuntimeError, walls.update);


# test lj wall force
class wall_lj_tests (unittest.TestCase):

    def setUp(self):
        init.create_random(N=100, box=data.boxdim(L=5));
        updater=update.box_resize(L = 15, Period=None);
        updater.set_params(scale_particles = False);
        walls=wall.group();
        walls.add_sphere(r=4, origin=(0.0, 0.0, 0.0), inside=True);
        run(1)

    # test to see that se can create a wall.lj
    def test_create(self):
        wall.lj(walls);

    # test setting coefficients
    def test_set_coeff(self):
        lj_wall = wall.lj(walls);
        lj_wall.set_coeff('A', r_cut=3.0, epsilon=1.0, sigma=1.0, alpha=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);

    # test coefficient not set checking
    def test_set_coeff_fail(self):
        lj_wall = wall.lj(walls);
        lj_wall.set_coeff('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);

    def tearDown(self):
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
