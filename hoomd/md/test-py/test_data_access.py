# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os

# tests for data access
class particle_data_access_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05

        context.current.sorter.set_params(grid=8)

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
        l = [b.get_lattice_vector(0), b.get_lattice_vector(1), b.get_lattice_vector(2)]
        self.assertAlmostEqual(l[0][0], 15, 5)
        self.assertAlmostEqual(l[0][1], 0, 5)
        self.assertAlmostEqual(l[0][2], 0, 5)
        self.assertAlmostEqual(l[1][0], 20*1.0, 5)
        self.assertAlmostEqual(l[1][1], 20, 5)
        self.assertAlmostEqual(l[1][2], 0)
        self.assertAlmostEqual(l[1][2], 0)
        self.assertAlmostEqual(l[2][0], 30*0.5, 5)
        self.assertAlmostEqual(l[2][1], 30*2.0)
        self.assertAlmostEqual(l[2][2], 30)
        v = (1+l[0][0],2+l[0][1],3+l[0][2])
        img = (0,0,0)
        v,img = b.wrap(v,img)
        self.assertEqual(img[0],1)
        self.assertEqual(img[1],0)
        self.assertEqual(img[2],0)
        self.assertAlmostEqual(v[0],1)
        self.assertAlmostEqual(v[1],2)
        self.assertAlmostEqual(v[2],3)
        b = b.scale(s=2)
        self.assertAlmostEqual(30, b.Lx, 5)
        self.assertAlmostEqual(40, b.Ly, 5)
        self.assertAlmostEqual(60, b.Lz, 5)
        b = b.scale(sx=2)
        self.assertAlmostEqual(60, b.Lx, 5)
        self.assertAlmostEqual(40, b.Ly, 5)
        self.assertAlmostEqual(60, b.Lz, 5)
        assert(b.set_volume(1.0).get_volume(),1.0,5)
        b = data.boxdim(Lx=2,Ly=3,Lz=4)
        v = (0.6*2,0.2*3,-0.7*4)
        u = b.min_image(v)
        self.assertAlmostEqual(u[0], -0.4*2)
        self.assertAlmostEqual(u[1], 0.2*3)
        self.assertAlmostEqual(u[2], 0.3*4)
        u = b.make_fraction(v)
        self.assertAlmostEqual(u[0],0.6+0.5)
        self.assertAlmostEqual(u[1],0.2+0.5)
        self.assertAlmostEqual(u[2],-0.7+0.5)

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
        print(self.s.particles[0])
        self.s.particles[0].position = (1,2,3);
        print(self.s.particles[0])
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

        # add some particles
        t0 = self.s.particles.add('A');
        self.assertEqual(101, len(self.s.particles));
        t1 = self.s.particles.add('A');
        self.assertEqual(102, len(self.s.particles));
        t2 = self.s.particles.add('A');
        self.assertEqual(103, len(self.s.particles));

        with self.assertRaises(RuntimeError):
            self.s.particles.add('B');

        self.assertEqual(self.s.particles[100].tag,t0);
        self.assertEqual(self.s.particles.get(t0).tag,t0);

        self.assertEqual(self.s.particles[101].tag,t1);
        self.assertEqual(self.s.particles.get(t1).tag,t1);

        self.assertEqual(self.s.particles[102].tag,t2);
        self.assertEqual(self.s.particles.get(t2).tag,t2);

        # check that we can get particle parameters
        for p in self.s.particles:
            p.tag
            p.type
            p.position

        # check that we can set/get the position
        self.s.particles.get(t2).position = (.5,.7,.9)
        pos = self.s.particles.get(t2).position
        self.assertAlmostEqual(pos[0],.5,5)
        self.assertAlmostEqual(pos[1],.7,5)
        self.assertAlmostEqual(pos[2],.9,5)

        # mass shold be one
        self.assertAlmostEqual(self.s.particles[100].mass,1.0,5)
        self.assertAlmostEqual(self.s.particles[101].mass,1.0,5)
        self.assertAlmostEqual(self.s.particles[102].mass,1.0,5)

        # test deletion by tag
        self.s.particles.remove(t1);
        self.assertEqual(102, len(self.s.particles));

        # test deletion by index
        if self.s.particles[100].tag == t0:
            del self.s.particles[100];
        else:
            del self.s.particles[101];

        t = self.s.particles.types
        self.assertEqual(len(t),1)
        self.assertEqual(t[0], 'A')
        t[0] = 'B'
        self.assertEqual(t[0], 'B')
        self.assertEqual(t.add('C'),1)
        # should print a warning
        self.assertEqual(t.add('C'),1)
        self.assertEqual(len(t),2)
        self.assertEqual(t[1], 'C')

    def tearDown(self):
        del self.s
        context.initialize();

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

# tests for bond, angle, dihedral, and improper data access
class bond_data_access_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = create_empty(N=100, box=data.boxdim(L=10),
                                   particle_types=['A'],
                                   bond_types=['bondA', 'bondB'],
                                   angle_types=['angleA', 'angleB'],
                                   dihedral_types=['dihedralA', 'dihedralB'],
                                   improper_types=['improperA', 'improperB']);

        context.current.sorter.set_params(grid=8)

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
        # test deletion by tag
        self.s.bonds.remove(b1);
        self.assertEqual(2, len(self.s.bonds));

        # test deletion by index (check bond a value to delete the bond with a=0)
        if self.s.bonds[0].tag == b0:
            del self.s.bonds[0];
        else:
            del self.s.bonds[1];

        self.assertEqual(b2, self.s.bonds[0].tag);
        self.assertEqual(b2, self.s.bonds.get(b2).tag);
        self.assertEqual(50, self.s.bonds[0].a);
        self.assertEqual(20, self.s.bonds[0].b);
        self.assertEqual('bondB', self.s.bonds[0].type);

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

        self.assertEqual(b2, self.s.angles.get(b2).tag);
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
        self.assertEqual(b2, self.s.dihedrals.get(b2).tag);
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

    # test that removing a particle invalidates the corresponding bond
    def test_remove_bonded_particle(self):
        # add some bonds
        b0 = self.s.bonds.add('bondA', 0, 1);
        self.assertEqual(1, len(self.s.bonds));
        b1 = self.s.bonds.add('bondA', 10, 11);
        self.assertEqual(2, len(self.s.bonds));
        b2 = self.s.bonds.add('bondB', 50, 20);
        self.assertEqual(3, len(self.s.bonds));

        l = len(self.s.particles)
        del(self.s.particles[50])
        self.assertEqual(len(self.s.particles),l-1)
        l_bonds = len(self.s.bonds)
        with self.assertRaises(RuntimeError):
            ptag = self.s.bonds[l_bonds-1].a

    # tests constraints
    def test_constraints(self):
        self.assertEqual(0, len(self.s.constraints));

        # add some constraints
        b0 = self.s.constraints.add(0, 1,1.5);
        self.assertEqual(1, len(self.s.constraints));
        b1 = self.s.constraints.add(10, 11,2.5);
        self.assertEqual(2, len(self.s.constraints));
        b2 = self.s.constraints.add(50, 20,3.5);
        self.assertEqual(3, len(self.s.constraints));

        # check that we can get all constraint parameters
        for c in self.s.constraints:
            c.d
            c.a
            c.b

        ####################################
        # test deletion by tag
        self.s.constraints.remove(b1);
        self.assertEqual(2, len(self.s.constraints));

        # test deletion by index (check constraint a value to delete the constraint with a=0)
        if self.s.constraints[0].tag == b0:
            del self.s.constraints[0];
        else:
            del self.s.constraints[1];

        self.assertEqual(b2, self.s.constraints[0].tag);
        self.assertEqual(b2, self.s.constraints.get(b2).tag);
        self.assertEqual(50, self.s.constraints[0].a);
        self.assertEqual(20, self.s.constraints[0].b);
        self.assertEqual(3.5, self.s.constraints[0].d)

    def tearDown(self):
        del self.s
        context.initialize();

# md.pair.lj
class pair_access_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05

        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        nl = md.nlist.cell()
        lj = md.pair.lj(r_cut=3.0, nlist = nl);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, r_cut=2.5, r_on=2.0);
        lj.update_coeffs();
        for p in lj.forces:
            f = p.force;
            f = p.energy;
            f = p.virial;
            f = p.torque;

    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
