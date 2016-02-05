# -*- coding: iso-8859-1 -*-
# Maintainer: jproc

from hoomd_script import *
import unittest
import os
import numpy as np

context.initialize()

#test wall.group()
class wall_group_tests(unittest.TestCase):
    def setUp(self):
        snapshot = data.make_snapshot(N=2,
                                      box=data.boxdim(L=20),
                                      particle_types=['A']);

        snapshot.particles.position[:] = [[0,0,-1], [0,0,1]];
        init.read_snapshot(snapshot)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);

    # basic test of creation for walls structure
    def test(self):
        walls=wall.group();

    # test each type of geometry can be added and deleted
    def test_add_sphere(self):
        walls=wall.group();
        walls.add_sphere(r=4, origin=(0.0, 0.0, 0.0), inside=True);
        walls.del_sphere(0);

        lj_wall = wall.lj(walls);
        lj_wall.force_coeff.set('A', sigma=1.0, alpha=1.0)
        run(1);

    def test_add_cylinder(self):
        walls=wall.group();
        walls.add_cylinder(r=4, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), inside=True);
        walls.del_cylinder(0);

        lj_wall = wall.lj(walls);
        lj_wall.force_coeff.set('A', sigma=1.0, alpha=1.0)
        run(1);

    def test_add_plane(self):
        walls=wall.group();
        walls.add_plane(normal=(1.0, 0.0, 0.0), origin=(-4.0, 0.0, 0.0), inside=True);
        walls.add_plane(normal=(-1.0, 0.0, 0.0), origin=(4.0, 0.0, 0.0), inside=False);
        walls.del_plane([0,1]);

        lj_wall = wall.lj(walls);
        lj_wall.force_coeff.set('A', sigma=1.0, alpha=1.0)
        run(1);

    def test_add_multiple(self):
        walls = wall.group(wall.plane(origin=(0,0,4), normal=(0,0,-1)))

        lj_wall = wall.lj(walls);
        lj_wall.force_coeff.set('A', sigma=1.0, alpha=1.0)
        run(1);

    def tearDown(self):
        init.reset();

# test lj wall force in standard mode
class wall_lj_tests (unittest.TestCase):
    def setUp(self):
        self.s=init.create_random(N=100, box=data.boxdim(L=5));
        updater=update.box_resize(L = 15, period=None);
        updater.set_params(scale_particles = False);
        self.walls=wall.group();
        self.walls.add_sphere(r=5, origin=(0.0, 0.0, 0.0), inside=True);

    # test to see that se can create a wall.lj
    def test_create(self):
        wall.lj(self.walls);

    # test coefficient not set checking
    def test_force_coeff_fail(self):
        lj_wall = wall.lj(self.walls);
        lj_wall.force_coeff.set('A', sigma=1.0, alpha=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 10);

    # test particle at the center
    def test_center(self):
        self.s.particles[0].position=(0,0,0)
        # specify Lennard-Jones interactions between particle pairs
        lj = pair.lj(r_cut=2.5)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        run(100)


    # test setting coefficients
    def test_force_coeff(self):
        lj_wall = wall.lj(self.walls, r_cut=3.0);
        lj_wall.force_coeff.set('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);

    def test_overload_structure_fail(self):
        self.walls.spheres=[wall.sphere()]*21;
        lj_wall = wall.lj(self.walls, r_cut=3.0);
        lj_wall.force_coeff.set('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 10);

    def test_NPT_fail(self):
        lj_wall = wall.lj(self.walls, r_cut=3.0);
        lj_wall.force_coeff.set('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.npt(group=all, T=1.0, tau=0.5, tauP=1.0, P=2.0);
        self.assertRaises(RuntimeError, run, 10);

    def tearDown(self):
        del self.s
        del self.walls
        init.reset();

# test lj wall force in shifted mode
class wall_shift_tests (unittest.TestCase):
    def setUp(self):
        snap= data.make_snapshot(N=4, box=data.boxdim(L=10))
        coords=[[1.1,0.0,0.0],[2.1,1.0,1.0],[4.0,-2.0,1.0],[-2.1,1.0,-0.5]];
        for i in range(4):
            snap.particles.position[i]=coords[i]
        self.s=init.read_snapshot(snap)
        self.walls=wall.group();
        self.walls.add_plane(origin=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), inside=True);
        all = group.all();
        integrate.mode_standard(dt=0.0);
        integrate.nve(all);
        lj_wall=wall.lj(self.walls,r_cut=3.5)
        lj_wall.force_coeff.set('A', r_extrap=1.1, epsilon=1.0, sigma=1.0)
        run(5)

    # test forces
    # errors allowed to be larger due to sqrt usage
    def test_forces(self):
        self.assertAlmostEqual(round(1.5880953898240548,4), round(self.s.particles.pdata.getPNetForce(0).x,4));
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(0).y);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(0).z);
        self.assertAlmostEqual(round(-0.1301453977354605,4), round(self.s.particles.pdata.getPNetForce(1).x,4));
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(1).y);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(1).z);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(2).x);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(2).y);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(2).z);
        self.assertAlmostEqual(round(1.5880953898240548,4), round(self.s.particles.pdata.getPNetForce(3).x,4));
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(3).y);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(3).z);

    def test_energy(self):
        self.assertAlmostEqual(-0.9811976689820279, self.s.particles.pdata.getPNetForce(0).w);
        self.assertAlmostEqual(-0.0439198953569452, self.s.particles.pdata.getPNetForce(1).w);
        self.assertAlmostEqual(0.0, self.s.particles.pdata.getPNetForce(2).w);
        self.assertAlmostEqual(round(4.100707578454948,4), round(self.s.particles.pdata.getPNetForce(3).w,4));

    def tearDown(self):
        del self.s
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
