# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# tests for data access
class particle_data_access_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

    # tests reading/setting of the box
    def test_box(self):
        self.s.box = data.boxdim(Lx=15, Ly=20, Lz=30, xy=1.0, xz=0.5, yz=2.0);
        b = self.s.box;
        self.assertAlmostEqual(15, b.Lx, 5)
        self.assertAlmostEqual(20, b.Ly, 5)
        self.assertAlmostEqual(30, b.Lz, 5)
        self.assertAlmostEqual(1.0, b.xy, 5)
        self.assertAlmostEqual(0.5, b.xz, 5)
        self.assertAlmostEqual(2.0, b.yz, 5)

    # test particles
    def test_particles(self):
        self.assertEqual(100, len(self.s.particles));
        for p in self.s.particles:
            # just access values to check that they can be read
            t = p.tag;
            t = p.acceleration;
            t = p.typeid;
            t = p.position;
            t = p.image;
            t = p.velocity;
            t = p.charge;
            t = p.mass;
            t = p.diameter;
            t = p.type;
            t = p.net_force;
            t = p.net_energy;
            t = p.orientation;
            t = p.net_torque;

        # test setting properties for just one particle
        self.s.particles[0].position = (1,2,3);
        t = self.s.particles[0].position;
        self.assertAlmostEqual(1, t[0], 5)
        self.assertAlmostEqual(2, t[1], 5)
        self.assertAlmostEqual(3, t[2], 5)

        self.s.particles[0].velocity = (4,5,6);
        t = self.s.particles[0].velocity;
        self.assertAlmostEqual(4, t[0], 5)
        self.assertAlmostEqual(5, t[1], 5)
        self.assertAlmostEqual(6, t[2], 5)

        self.s.particles[0].image = (7,8,9)
        t = self.s.particles[0].image;
        self.assertAlmostEqual(7, t[0], 5)
        self.assertAlmostEqual(8, t[1], 5)
        self.assertAlmostEqual(9, t[2], 5)

        self.s.particles[0].charge = 5.6;
        self.assertAlmostEqual(5.6, self.s.particles[0].charge, 5)

        self.s.particles[0].mass = 7.9;
        self.assertAlmostEqual(7.9, self.s.particles[0].mass, 5)

        self.s.particles[0].diameter= 8.7;
        self.assertAlmostEqual(8.7, self.s.particles[0].diameter, 5)

        self.s.particles[0].orientation = (1,2,3,5);
        t = self.s.particles[0].orientation;
        self.assertAlmostEqual(1, t[0], 5)
        self.assertAlmostEqual(2, t[1], 5)
        self.assertAlmostEqual(3, t[2], 5)
        self.assertAlmostEqual(5, t[3], 5)

    def tearDown(self):
        del self.s
        init.reset();

# tests for bond, angle, dihedral, and improper data access
class bond_data_access_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_empty(N=100, box=data.boxdim(L=10), n_bond_types=2, n_angle_types=2, n_dihedral_types=2, n_improper_types=2);

        sorter.set_params(grid=8)

    # tests bonds
    def test_bonds(self):
        self.assertEqual(0, len(self.s.bonds));

        # add some bonds
        b0 = self.s.bonds.add('bondA', 0, 1);
        self.assertEqual(1, len(self.s.bonds));
        b1 = self.s.bonds.add('bondA', 10, 11);
        self.assertEqual(2, len(self.s.bonds));
        b2 = self.s.bonds.add('bondB', 50, 20);
        self.assertEqual(3, len(self.s.bonds));

        # check that we can get all bond parameters
        for b in self.s.bonds:
            b.typeid
            b.type
            b.a
            b.b

        ####################################
        ## The bond deletion feature is currently disabled
        # test deletion by tag
        #self.s.bonds.remove(b1);
        #self.assertEqual(2, len(self.s.bonds));

        # test deletion by index (check bond a value to delete the bond with a=0)
        #if self.s.bonds[0].tag == b0:
        #    del self.s.bonds[0];
        #else:
        #    del self.s.bonds[1];

        #self.assertEqual(b2, self.s.bonds[0].tag);
        #self.assertEqual(50, self.s.bonds[0].a);
        #self.assertEqual(20, self.s.bonds[0].b);
        #self.assertEqual('bondB', self.s.bonds[0].type);

    # tests angles
    def test_angles(self):
        self.assertEqual(0, len(self.s.angles));

        # add some angles
        b0 = self.s.angles.add('angleA', 0, 1, 2);
        self.assertEqual(1, len(self.s.angles));
        b1 = self.s.angles.add('angleA', 10, 11, 12);
        self.assertEqual(2, len(self.s.angles));
        b2 = self.s.angles.add('angleB', 50, 20, 10);
        self.assertEqual(3, len(self.s.angles));

        # check that we can get all bond parameters
        for b in self.s.angles:
            b.tag
            b.typeid
            b.type
            b.a
            b.b
            b.c

        # test deletion by tag
        self.s.angles.remove(b1);
        self.assertEqual(2, len(self.s.angles));

        # test deletion by index (check bond a value to delete the bond with a=0)
        if self.s.angles[0].tag == b0:
            del self.s.angles[0];
        else:
            del self.s.angles[1];

        self.assertEqual(b2, self.s.angles[0].tag);
        self.assertEqual(50, self.s.angles[0].a);
        self.assertEqual(20, self.s.angles[0].b);
        self.assertEqual(10, self.s.angles[0].c);
        self.assertEqual('angleB', self.s.angles[0].type);

    # tests dihedrals
    def test_dihedrals(self):
        self.assertEqual(0, len(self.s.dihedrals));

        # add some dihedrals
        b0 = self.s.dihedrals.add('dihedralA', 0, 1, 2, 3);
        self.assertEqual(1, len(self.s.dihedrals));
        b1 = self.s.dihedrals.add('dihedralA', 10, 11, 12, 13);
        self.assertEqual(2, len(self.s.dihedrals));
        b2 = self.s.dihedrals.add('dihedralB', 50, 20, 10, 1);
        self.assertEqual(3, len(self.s.dihedrals));

        # check that we can get all bond parameters
        for b in self.s.dihedrals:
            b.tag
            b.typeid
            b.type
            b.a
            b.b
            b.c
            b.d

        # test deletion by tag
        self.s.dihedrals.remove(b1);
        self.assertEqual(2, len(self.s.dihedrals));

        # test deletion by index (check bond a value to delete the bond with a=0)
        if self.s.dihedrals[0].tag == b0:
            del self.s.dihedrals[0];
        else:
            del self.s.dihedrals[1];

        self.assertEqual(b2, self.s.dihedrals[0].tag);
        self.assertEqual(50, self.s.dihedrals[0].a);
        self.assertEqual(20, self.s.dihedrals[0].b);
        self.assertEqual(10, self.s.dihedrals[0].c);
        self.assertEqual(1, self.s.dihedrals[0].d);
        self.assertEqual('dihedralB', self.s.dihedrals[0].type);

    # tests impropers
    def test_impropers(self):
        self.assertEqual(0, len(self.s.impropers));

        # add some impropers
        b0 = self.s.impropers.add('improperA', 0, 1, 2, 3);
        self.assertEqual(1, len(self.s.impropers));
        b1 = self.s.impropers.add('improperA', 10, 11, 12, 13);
        self.assertEqual(2, len(self.s.impropers));
        b2 = self.s.impropers.add('improperB', 50, 20, 10, 1);
        self.assertEqual(3, len(self.s.impropers));

        # check that we can get all bond parameters
        for b in self.s.impropers:
            b.tag
            b.typeid
            b.type
            b.a
            b.b
            b.c
            b.d

        # test deletion by tag
        self.s.impropers.remove(b1);
        self.assertEqual(2, len(self.s.impropers));

        # test deletion by index (check bond a value to delete the bond with a=0)
        if self.s.impropers[0].tag == b0:
            del self.s.impropers[0];
        else:
            del self.s.impropers[1];

        self.assertEqual(b2, self.s.impropers[0].tag);
        self.assertEqual(50, self.s.impropers[0].a);
        self.assertEqual(20, self.s.impropers[0].b);
        self.assertEqual(10, self.s.impropers[0].c);
        self.assertEqual(1, self.s.impropers[0].d);
        self.assertEqual('improperB', self.s.impropers[0].type);

    def tearDown(self):
        del self.s
        init.reset();

# pair.lj
class pair_access_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        lj = pair.lj(r_cut=3.0);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, r_cut=2.5, r_on=2.0);
        lj.update_coeffs();
        for p in lj.forces:
            f = p.force;
            f = p.energy;
            f = p.virial;
            f = p.torque;

    def tearDown(self):
        init.reset();


# tests for body data access
class body_data_access_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

        for p in self.s.particles:
            p.body = p.tag % 50

        self.s.sysdef.getRigidData().initializeData()

    # test bodies
    def test_bodies(self):
        self.assertEqual(50, len(self.s.bodies));

        for b in self.s.bodies:
            # just access values to check that they can be read
            t = b.COM
            t = b.orientation
            t = b.velocity
            t = b.mass
            t = b.angular_momentum
            t = b.moment_inertia
            t = b.particle_tags
            t = b.particle_disp
            t = b.net_force
            t = b.net_torque

        self.s.bodies[0].mass = 7.9;
        self.assertAlmostEqual(7.9, self.s.bodies[0].mass, 5)

        self.s.bodies[0].moment_inertia = (4,5,6);
        t = self.s.bodies[0].moment_inertia;
        self.assertAlmostEqual(4, t[0], 5)
        self.assertAlmostEqual(5, t[1], 5)
        self.assertAlmostEqual(6, t[2], 5)

        self.s.bodies[0].orientation = (0,0,0,1);
        t = self.s.bodies[0].orientation;
        self.assertAlmostEqual(0, t[0], 5)
        self.assertAlmostEqual(0, t[1], 5)
        self.assertAlmostEqual(0, t[2], 5)
        self.assertAlmostEqual(1, t[3], 5)

        self.s.bodies[0].particle_disp = [(0,0,0),(1,0,0)];
        t = self.s.bodies[0].particle_disp;
        self.assertAlmostEqual(0, t[0][0], 5)
        self.assertAlmostEqual(0, t[0][1], 5)
        self.assertAlmostEqual(0, t[0][2], 5)
        self.assertAlmostEqual(1, t[1][0], 5)
        self.assertAlmostEqual(0, t[1][1], 5)
        self.assertAlmostEqual(0, t[1][2], 5)

        self.s.bodies[0].velocity = (1,2,3);
        t = self.s.bodies[0].velocity;
        self.assertAlmostEqual(1, t[0], 5)
        self.assertAlmostEqual(2, t[1], 5)
        self.assertAlmostEqual(3, t[2], 5)

        self.s.bodies[0].angular_momentum = (3,2,1);
        t = self.s.bodies[0].angular_momentum;
        self.assertAlmostEqual(3, t[0], 5)
        self.assertAlmostEqual(2, t[1], 5)
        self.assertAlmostEqual(1, t[2], 5)

    def tearDown(self):
        del self.s
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
