# -*- coding: iso-8859-1 -*-
# Maintainer: jproc

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os
import numpy as np

#test md.wall.group()
class wall_group_tests(unittest.TestCase):
    def setUp(self):
        snapshot = data.make_snapshot(N=2,
                                      box=data.boxdim(L=20),
                                      particle_types=['A']);

        if comm.get_rank() == 0:
            snapshot.particles.position[:] = [[0,0,-1], [0,0,1]];

        init.read_snapshot(snapshot)

        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);

    # basic test of creation for walls structure
    def test(self):
        walls=md.wall.group();

    # test each type of geometry can be added and deleted
    def test_add_sphere(self):
        walls=md.wall.group();
        walls.add_sphere(r=4, origin=(0.0, 0.0, 0.0), inside=True);
        walls.del_sphere(0);

        lj_wall = md.wall.lj(walls, r_cut=3.0);
        lj_wall.force_coeff.set('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        run(1);

    def test_add_cylinder(self):
        walls=md.wall.group();
        walls.add_cylinder(r=4, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), inside=True);
        walls.del_cylinder(0);

        lj_wall = md.wall.lj(walls, r_cut=3.0);
        lj_wall.force_coeff.set('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        run(1);

    def test_add_plane(self):
        walls=md.wall.group();
        walls.add_plane(normal=(1.0, 0.0, 0.0), origin=(-4.0, 0.0, 0.0), inside=True);
        walls.add_plane(normal=(-1.0, 0.0, 0.0), origin=(4.0, 0.0, 0.0), inside=False);
        walls.del_plane([0,1]);

        lj_wall = md.wall.lj(walls, r_cut=3.0);
        lj_wall.force_coeff.set('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        run(1);

    def test_add_multiple(self):
        walls = md.wall.group(md.wall.plane(origin=(0,0,4), normal=(0,0,-1)))

        lj_wall = md.wall.lj(walls, r_cut=3.0);
        lj_wall.force_coeff.set('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        run(1);

    def tearDown(self):
        context.initialize();

# test lj wall force in standard mode
class wall_lj_tests (unittest.TestCase):
    def setUp(self):
        self.s = init.create_lattice(lattice.sc(a=1.0),n=[5,5,4]);
        updater=update.box_resize(L = 15, period=None, scale_particles = False);
        self.walls=md.wall.group();
        self.walls.add_sphere(r=5, origin=(0.0, 0.0, 0.0), inside=True);

    # test to see that se can create a md.wall.lj
    def test_create(self):
        md.wall.lj(self.walls);

    # test coefficient not set checking
    def test_force_coeff_fail(self):
        lj_wall = md.wall.lj(self.walls);
        lj_wall.force_coeff.set('A', sigma=1.0, alpha=1.0)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        self.assertRaises(RuntimeError, run, 10);

    # test particle at the center
    def test_center(self):
        self.s.particles[0].position=(0,0,0)
        # specify Lennard-Jones interactions between particle pairs
        nl = md.nlist.cell()
        lj = md.pair.lj(r_cut=2.5, nlist = nl)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        run(100)

    # test missing coefficients
    def test_missing_A(self):
        lj_wall = md.wall.lj(self.walls);
        self.assertRaises(RuntimeError, lj_wall.update_coeffs)

    # test setting coefficients
    def test_force_coeff(self):
        lj_wall = md.wall.lj(self.walls, r_cut=3.0);
        lj_wall.force_coeff.set('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100);

    def test_overload_structure_fail(self):
        self.walls.spheres=[md.wall.sphere()]*21;
        lj_wall = md.wall.lj(self.walls, r_cut=3.0);
        lj_wall.force_coeff.set('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        self.assertRaises(RuntimeError, run, 10);

    # def test_NPT_fail(self):
    #     lj_wall = md.wall.lj(self.walls, r_cut=3.0);
    #     lj_wall.force_coeff.set('A', epsilon=1.0, sigma=1.0, alpha=1.0)
    #     all = group.all();
    #     md.integrate.mode_standard(dt=0.005);
    #     md.integrate.npt(group=all, kT=1.0, tau=0.5, tauP=1.0, P=2.0);
    #     self.assertRaises(RuntimeError, run, 10);

    def tearDown(self):
        del self.s
        del self.walls
        context.initialize();

# test lj wall force in shifted mode
class wall_shift_tests (unittest.TestCase):
    def setUp(self):
        snap= data.make_snapshot(N=4, box=data.boxdim(L=10))
        coords=[[1.1,0.0,0.0],[2.1,1.0,1.0],[4.0,-2.0,1.0],[-2.1,1.0,-0.5]];
        if comm.get_rank() == 0:
            for i in range(4):
                snap.particles.position[i]=coords[i]
        self.s=init.read_snapshot(snap)
        self.walls=md.wall.group();
        self.walls.add_plane(origin=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), inside=True);
        all = group.all();
        md.integrate.mode_standard(dt=0.0);
        md.integrate.nve(all);
        lj_wall=md.wall.lj(self.walls,r_cut=3.5)
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
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
