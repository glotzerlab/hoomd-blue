# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
import hoomd;
context.initialize()
import unittest
import os
import sys
import numpy

# unit tests for init.take_snapshot and init.restore_snapshot
class init_take_restore_snapshot (unittest.TestCase):
    def setUp(self):
        self.s = init.read_gsd(os.path.join(os.path.dirname(os.path.dirname(__file__)),'md','test-py','test_data_polymer_system_small.gsd'));
        self.assertTrue(self.s);
        self.assertTrue(self.s.sysdef);

        # add some constraints
        self.s.constraints.add(0, 1, 0.1)
        self.s.constraints.add(0, 2, 0.2)
        self.s.constraints.add(3, 4, 3.4)

    # test taking a snapshot and re-initializing
    def test(self):
        snapshot = self.s.take_snapshot(all=True)
        self.s.restore_snapshot(snapshot)

    # tests options to take_snapshot
    def test_options(self):
        snapshot = self.s.take_snapshot(particles=True)
        self.s.restore_snapshot(snapshot)
        snapshot = self.s.take_snapshot(bonds=True)
        snapshot = self.s.take_snapshot(integrators=True)

    def test_read_snapshot(self):
        snapshot = self.s.take_snapshot(all=True)
        del self.s
        context.initialize()
        self.s = init.read_snapshot(snapshot)

    # test that adding and removing bonds works with take/restore snapshot
    def test_add_remove_bonds(self):
        l = len(self.s.bonds)
        del(self.s.bonds[l-1])
        del(self.s.bonds[2])
        bonds = []
        for b in self.s.bonds:
            bonds.append((b.a,b.b,b.type))
        self.assertEqual(len(self.s.bonds),l-2)
        snapshot = self.s.take_snapshot(bonds=True)
        self.s.restore_snapshot(snapshot)
        self.assertEqual(len(self.s.bonds),l-2)
        for (b,old_b) in zip(self.s.bonds,bonds):
            new_b = (b.a, b.b, b.type)
            self.assertEqual(new_b,old_b)
        self.s.bonds.add('polymer',0, 10)
        l_new = len(self.s.bonds)
        self.assertEqual(l_new,l-1)
        snapshot = self.s.take_snapshot(bonds=True)
        self.s.restore_snapshot(snapshot)
        l_new = len(self.s.bonds)
        self.assertEqual(l_new,l-1)
        self.assertEqual(self.s.bonds[l_new-1].a,0)
        self.assertEqual(self.s.bonds[l_new-1].b,10)
        self.assertEqual(self.s.bonds[l_new-1].type,'polymer')

    # test that adding and removing bonds works with take/restore snapshot
    def test_add_remove_constraint(self):
        l = len(self.s.constraints)
        self.assertEqual(l,3)
        del(self.s.constraints[l-1])
        constraints = []
        for c in self.s.constraints:
            constraints.append((c.a,c.b,c.d))
        self.assertEqual(len(self.s.constraints),l-1)
        snapshot = self.s.take_snapshot(bonds=True)
        self.s.restore_snapshot(snapshot)
        self.assertEqual(len(self.s.constraints),l-1)
        for (c,old_c) in zip(self.s.constraints,constraints):
            new_c = (c.a, c.b, c.d)
            self.assertEqual(new_c,old_c)
        self.s.constraints.add(0, 10,1.5)
        l_new = len(self.s.constraints)
        self.assertEqual(l_new,l)
        snapshot = self.s.take_snapshot(bonds=True)
        self.s.restore_snapshot(snapshot)
        l_new = len(self.s.constraints)
        self.assertEqual(l_new,l)
        self.assertEqual(self.s.constraints[l_new-1].a,0)
        self.assertEqual(self.s.constraints[l_new-1].b,10)
        self.assertEqual(self.s.constraints[l_new-1].d,1.5)


    # test removing and adding particles before taking the snapshot
    def test_add_remove_particle(self):
        l = len(self.s.particles)
        l_bonds = len(self.s.bonds)
        tags = []
        # remove the bonds that connect to the particle
        for b in self.s.bonds:
            if b.a == 2 or b.b == 2:
                tags.append(b.tag)

        for t in tags:
            self.s.bonds.remove(t)
        # we should have removed two bonds
        self.assertEqual(len(self.s.bonds),l_bonds-2)

        # remove particle
        del(self.s.particles[2])
        l_new = len(self.s.particles)
        self.assertEqual(l_new,l-1)

        # add particles
        t1 = self.s.particles.add('A')
        t2 = self.s.particles.add('B')
        l_new = len(self.s.particles)
        self.assertEqual(l_new, l+1)
        self.assertEqual(self.s.particles.get(t1).type,'A')
        self.assertEqual(self.s.particles.get(t2).type,'B')
        snapshot = self.s.take_snapshot(all=True)
        self.s.restore_snapshot(snapshot)
        self.assertEqual(len(self.s.particles), l_new)

    # test what happens when no bond snapshot is requested
    def test_no_bonds(self):
        snap = self.s.take_snapshot(particles=False, bonds=False, integrators=False);
        self.assertEqual(len(snap.particles.types), 0);
        self.assertEqual(len(snap.bonds.types), 0);
        self.assertEqual(len(snap.impropers.types), 0);
        self.assertEqual(len(snap.angles.types), 0);
        self.assertEqual(len(snap.dihedrals.types), 0);

    def tearDown(self):
        del self.s
        context.initialize();

# test take_snapshot with the numpy API
class init_verify_npy_dtype (unittest.TestCase):
    def setUp(self):
        self.s = init.read_gsd(os.path.join(os.path.dirname(os.path.dirname(__file__)),'md','test-py','test_data_polymer_system_small2.gsd'));
        self.assertTrue(self.s);
        self.assertTrue(self.s.sysdef);

        # add some constraints
        self.s.constraints.add(0, 1, 0.1)
        self.s.constraints.add(0, 2, 0.2)
        self.s.constraints.add(3, 4, 3.4)

    def test_take_snapshot_double(self):
        snapshot = self.s.take_snapshot(all=True, dtype='double');

        if comm.get_rank() == 0:
            float_type = numpy.float64;

            # check the particles
            self.assertEqual(snapshot.particles.N, 9)
            self.assertEqual(snapshot.particles.position.shape, (9,3));
            self.assertEqual(snapshot.particles.position.dtype, float_type);
            self.assertEqual(sys.getrefcount(snapshot.particles.position), 1);

            self.assertEqual(snapshot.particles.velocity.shape, (9,3));
            self.assertEqual(snapshot.particles.velocity.dtype, float_type);
            self.assertEqual(sys.getrefcount(snapshot.particles.velocity), 1);

            self.assertEqual(snapshot.particles.acceleration.shape, (9,3));
            self.assertEqual(snapshot.particles.acceleration.dtype, float_type);
            self.assertEqual(sys.getrefcount(snapshot.particles.acceleration), 1);

            self.assertEqual(snapshot.particles.typeid.shape, (9,));
            self.assertEqual(snapshot.particles.typeid.dtype, numpy.uint32);
            self.assertEqual(sys.getrefcount(snapshot.particles.typeid), 1);

            self.assertEqual(snapshot.particles.mass.shape, (9,));
            self.assertEqual(snapshot.particles.mass.dtype, float_type);
            self.assertEqual(sys.getrefcount(snapshot.particles.mass), 1);

            self.assertEqual(snapshot.particles.charge.shape, (9,));
            self.assertEqual(snapshot.particles.charge.dtype, float_type);
            self.assertEqual(sys.getrefcount(snapshot.particles.charge), 1);

            self.assertEqual(snapshot.particles.diameter.shape, (9,));
            self.assertEqual(snapshot.particles.diameter.dtype, float_type);
            self.assertEqual(sys.getrefcount(snapshot.particles.diameter), 1);

            self.assertEqual(snapshot.particles.image.shape, (9,3));
            self.assertEqual(snapshot.particles.image.dtype, numpy.int32);
            self.assertEqual(sys.getrefcount(snapshot.particles.image), 1);

            self.assertEqual(snapshot.particles.body.shape, (9,));
            self.assertEqual(snapshot.particles.body.dtype, numpy.uint32);
            self.assertEqual(sys.getrefcount(snapshot.particles.body), 1);

            self.assertEqual(snapshot.particles.types, ["A", "B"]);

            for i in range(0,9):
                self.assertEqual(list(snapshot.particles.velocity[i]), [0,0,0]);
                self.assertEqual(list(snapshot.particles.acceleration[i]), [0,0,0]);
                self.assertEqual(snapshot.particles.mass[i], 1.0);
                self.assertEqual(snapshot.particles.diameter[i], 1.0);
                self.assertEqual(snapshot.particles.charge[i], 0.0);
                # don't check image because the polymer generator can place particles with non-zero images
                self.assertEqual(snapshot.particles.body[i], 4294967295);

            self.assertEqual(list(snapshot.particles.typeid), [0, 1, 1, 0, 1, 1, 0, 1, 1]);

            # check the bonds
            self.assertEqual(snapshot.bonds.N, 6);
            self.assertEqual(snapshot.bonds.typeid.shape, (6,));
            self.assertEqual(snapshot.bonds.typeid.dtype, numpy.uint32);
            self.assertEqual(sys.getrefcount(snapshot.bonds.typeid), 1);

            self.assertEqual(snapshot.bonds.group.shape, (6,2));
            self.assertEqual(snapshot.bonds.group.dtype, numpy.uint32);
            self.assertEqual(sys.getrefcount(snapshot.bonds.group), 1);

            self.assertEqual(snapshot.bonds.types, ['polymer']);

            self.assertEqual(list(snapshot.bonds.group[0]), [0,1]);
            self.assertEqual(list(snapshot.bonds.group[1]), [1,2]);
            self.assertEqual(list(snapshot.bonds.group[2]), [3,4]);
            self.assertEqual(list(snapshot.bonds.group[3]), [4,5]);
            self.assertEqual(list(snapshot.bonds.group[4]), [6,7]);
            self.assertEqual(list(snapshot.bonds.group[5]), [7,8]);

            # check the constraints
            self.assertEqual(snapshot.constraints.N, 3)
            self.assertEqual(snapshot.constraints.value.shape, (3,))
            self.assertTrue(snapshot.constraints.value.dtype == numpy.float32 or snapshot.constraints.value.dtype == numpy.float64)
            self.assertEqual(sys.getrefcount(snapshot.constraints.value), 1);

            self.assertEqual(snapshot.constraints.group.shape, (3,2))
            self.assertEqual(snapshot.constraints.group.dtype, numpy.uint32)
            self.assertEqual(sys.getrefcount(snapshot.constraints.group), 1);

            self.assertEqual(list(snapshot.constraints.group[0]), [0,1])
            self.assertEqual(list(snapshot.constraints.group[1]), [0,2])
            self.assertEqual(list(snapshot.constraints.group[2]), [3,4])

            self.assertAlmostEqual(float(snapshot.constraints.value[0]), 0.1, 5)
            self.assertAlmostEqual(float(snapshot.constraints.value[1]), 0.2, 5)
            self.assertAlmostEqual(float(snapshot.constraints.value[2]), 3.4, 5)

    def test_take_snapshot_float(self):
        snapshot = self.s.take_snapshot(all=True, dtype='float');

        if comm.get_rank() == 0:
            float_type = numpy.float32;

            # check the particles
            self.assertEqual(snapshot.particles.N, 9)
            self.assertEqual(snapshot.particles.position.shape, (9,3));
            self.assertEqual(snapshot.particles.position.dtype, float_type);
            self.assertEqual(snapshot.particles.velocity.shape, (9,3));
            self.assertEqual(snapshot.particles.velocity.dtype, float_type);
            self.assertEqual(snapshot.particles.acceleration.shape, (9,3));
            self.assertEqual(snapshot.particles.acceleration.dtype, float_type);
            self.assertEqual(snapshot.particles.typeid.shape, (9,));
            self.assertEqual(snapshot.particles.typeid.dtype, numpy.uint32);
            self.assertEqual(snapshot.particles.mass.shape, (9,));
            self.assertEqual(snapshot.particles.mass.dtype, float_type);
            self.assertEqual(snapshot.particles.charge.shape, (9,));
            self.assertEqual(snapshot.particles.charge.dtype, float_type);
            self.assertEqual(snapshot.particles.diameter.shape, (9,));
            self.assertEqual(snapshot.particles.diameter.dtype, float_type);

            self.assertEqual(snapshot.particles.image.shape, (9,3));
            self.assertEqual(snapshot.particles.image.dtype, numpy.int32);

            self.assertEqual(snapshot.particles.body.shape, (9,));
            self.assertEqual(snapshot.particles.body.dtype, numpy.uint32);

            self.assertEqual(snapshot.particles.types, ["A", "B"]);

            for i in range(0,9):
                self.assertEqual(list(snapshot.particles.velocity[i]), [0,0,0]);
                self.assertEqual(list(snapshot.particles.acceleration[i]), [0,0,0]);
                self.assertEqual(snapshot.particles.mass[i], 1.0);
                self.assertEqual(snapshot.particles.diameter[i], 1.0);
                self.assertEqual(snapshot.particles.charge[i], 0.0);
                # don't check image because the polymer generator can place particles with non-zero images
                self.assertEqual(snapshot.particles.body[i], 4294967295);

            self.assertEqual(list(snapshot.particles.typeid), [0, 1, 1, 0, 1, 1, 0, 1, 1]);

            # check the bonds
            self.assertEqual(snapshot.bonds.N, 6);
            self.assertEqual(snapshot.bonds.typeid.shape, (6,));
            self.assertEqual(snapshot.bonds.typeid.dtype, numpy.uint32);
            self.assertEqual(snapshot.bonds.group.shape, (6,2));
            self.assertEqual(snapshot.bonds.group.dtype, numpy.uint32);
            self.assertEqual(snapshot.bonds.types, ['polymer']);

            self.assertEqual(list(snapshot.bonds.group[0]), [0,1]);
            self.assertEqual(list(snapshot.bonds.group[1]), [1,2]);
            self.assertEqual(list(snapshot.bonds.group[2]), [3,4]);
            self.assertEqual(list(snapshot.bonds.group[3]), [4,5]);
            self.assertEqual(list(snapshot.bonds.group[4]), [6,7]);
            self.assertEqual(list(snapshot.bonds.group[5]), [7,8]);

            # check the constraints
            self.assertEqual(snapshot.constraints.N, 3)
            self.assertEqual(snapshot.constraints.value.shape, (3,))
            self.assertTrue(snapshot.constraints.value.dtype == numpy.float32 or snapshot.constraints.value.dtype == numpy.float64)
            self.assertEqual(snapshot.constraints.group.shape, (3,2))
            self.assertEqual(snapshot.constraints.group.dtype, numpy.uint32)

            self.assertEqual(list(snapshot.constraints.group[0]), [0,1])
            self.assertEqual(list(snapshot.constraints.group[1]), [0,2])
            self.assertEqual(list(snapshot.constraints.group[2]), [3,4])

            self.assertAlmostEqual(float(snapshot.constraints.value[0]), 0.1, 5)
            self.assertAlmostEqual(float(snapshot.constraints.value[1]), 0.2, 5)
            self.assertAlmostEqual(float(snapshot.constraints.value[2]), 3.4, 5)

    def tearDown(self):
        del self.s
        context.initialize();


# test make_snapshot, read_snapshot and broadcast
class init_take_snapshot_float (unittest.TestCase):
    def setUp(self):
        self.snapshot = data.make_snapshot(N=4, box=data.boxdim(L=10), dtype='float');
        if comm.get_rank() == 0:
            # particles
            self.snapshot.particles.position[0] = [0,1,2];
            self.snapshot.particles.position[1] = [1,2,3];
            self.snapshot.particles.position[2] = [0,-1,-2];
            self.snapshot.particles.position[3] = [-1, -2, -3];
            self.snapshot.particles.velocity[0] = [10, 11, 12];
            self.snapshot.particles.velocity[1] = [11, 12, 13];
            self.snapshot.particles.velocity[2] = [12, 13, 14];
            self.snapshot.particles.velocity[3] = [13, 14, 15];
            self.snapshot.particles.acceleration[0] = [20, 21, 22];
            self.snapshot.particles.acceleration[1] = [21, 22, 23];
            self.snapshot.particles.acceleration[2] = [22, 23, 24];
            self.snapshot.particles.acceleration[3] = [23, 24, 25];
            self.snapshot.particles.typeid[:] = [0,0,1,1];
            self.snapshot.particles.mass[:] = [33, 34, 35,  36];
            self.snapshot.particles.charge[:] = [44, 45, 46, 47];
            self.snapshot.particles.diameter[:] = [55, 56, 57, 58];
            self.snapshot.particles.image[0] = [60, 61, 62];
            self.snapshot.particles.image[1] = [61, 62, 63];
            self.snapshot.particles.image[2] = [62, 63, 64];
            self.snapshot.particles.image[3] = [63, 64, 65];
            self.snapshot.particles.types = ['p1', 'p2'];

            # bonds
            self.snapshot.bonds.types = ['b1', 'b2'];
            self.snapshot.bonds.resize(2);
            self.snapshot.bonds.typeid[:] = [0, 1];
            self.snapshot.bonds.group[0] = [0, 1];
            self.snapshot.bonds.group[1] = [2, 3];

            # angles
            self.snapshot.angles.types = ['a1', 'a2'];
            self.snapshot.angles.resize(2);
            self.snapshot.angles.typeid[:] = [1, 0];
            self.snapshot.angles.group[0] = [0, 1, 2];
            self.snapshot.angles.group[1] = [2, 3, 0];

            # dihedrals
            self.snapshot.dihedrals.types = ['d1'];
            self.snapshot.dihedrals.resize(1);
            self.snapshot.dihedrals.typeid[:] = [0];
            self.snapshot.dihedrals.group[0] = [0, 1, 2, 3];

            # impropers
            self.snapshot.impropers.types = ['i1'];
            self.snapshot.impropers.resize(1);
            self.snapshot.impropers.typeid[:] = [0];
            self.snapshot.impropers.group[0] = [3, 2, 1, 0];

            # constraints
            self.snapshot.constraints.resize(1)
            self.snapshot.constraints.group[0] = [0, 1]
            self.snapshot.constraints.value[0] = 2.5

    def test_read_snapshot(self):
        s = init.read_snapshot(self.snapshot);
        self.assertTrue(s);
        self.assertTrue(s.sysdef);

        # particles
        self.assertEqual(len(s.particles), 4);
        self.assertEqual(s.particles[0].position, (0,1,2));
        self.assertEqual(s.particles[0].velocity, (10,11,12));
        self.assertEqual(s.particles[0].acceleration, (20,21,22));
        self.assertEqual(s.particles[0].type, 'p1');
        self.assertEqual(s.particles[0].mass, 33);
        self.assertEqual(s.particles[0].charge, 44);
        self.assertEqual(s.particles[0].diameter, 55);
        self.assertEqual(s.particles[0].image, (60,61,62));

        self.assertEqual(s.particles[1].position, (1,2,3));
        self.assertEqual(s.particles[1].velocity, (11,12,13));
        self.assertEqual(s.particles[1].acceleration, (21,22,23));
        self.assertEqual(s.particles[1].type, 'p1');
        self.assertEqual(s.particles[1].mass, 34);
        self.assertEqual(s.particles[1].charge, 45);
        self.assertEqual(s.particles[1].diameter, 56);
        self.assertEqual(s.particles[1].image, (61,62,63));

        self.assertEqual(s.particles[2].position, (0,-1,-2));
        self.assertEqual(s.particles[2].velocity, (12,13,14));
        self.assertEqual(s.particles[2].acceleration, (22,23,24));
        self.assertEqual(s.particles[2].type, 'p2');
        self.assertEqual(s.particles[2].mass, 35);
        self.assertEqual(s.particles[2].charge, 46);
        self.assertEqual(s.particles[2].diameter, 57);
        self.assertEqual(s.particles[2].image, (62,63,64));

        self.assertEqual(s.particles[3].position, (-1,-2,-3));
        self.assertEqual(s.particles[3].velocity, (13,14,15));
        self.assertEqual(s.particles[3].acceleration, (23,24,25));
        self.assertEqual(s.particles[3].type, 'p2');
        self.assertEqual(s.particles[3].mass, 36);
        self.assertEqual(s.particles[3].charge, 47);
        self.assertEqual(s.particles[3].diameter, 58);
        self.assertEqual(s.particles[3].image, (63,64,65));

        # bonds
        self.assertEqual(len(s.bonds), 2);
        self.assertEqual(s.bonds[0].type, 'b1');
        self.assertEqual(s.bonds[0].a, 0);
        self.assertEqual(s.bonds[0].b, 1);

        self.assertEqual(s.bonds[1].type, 'b2');
        self.assertEqual(s.bonds[1].a, 2);
        self.assertEqual(s.bonds[1].b, 3);

        # angles
        self.assertEqual(len(s.angles), 2);
        self.assertEqual(s.angles[0].type, 'a2');
        self.assertEqual(s.angles[0].a, 0);
        self.assertEqual(s.angles[0].b, 1);
        self.assertEqual(s.angles[0].c, 2);

        self.assertEqual(s.angles[1].type, 'a1');
        self.assertEqual(s.angles[1].a, 2);
        self.assertEqual(s.angles[1].b, 3);
        self.assertEqual(s.angles[1].c, 0);

        # dihedrals
        self.assertEqual(len(s.dihedrals), 1);
        self.assertEqual(s.dihedrals[0].type, 'd1');
        self.assertEqual(s.dihedrals[0].a, 0);
        self.assertEqual(s.dihedrals[0].b, 1);
        self.assertEqual(s.dihedrals[0].c, 2);
        self.assertEqual(s.dihedrals[0].d, 3);

        # impropers
        self.assertEqual(len(s.impropers), 1);
        self.assertEqual(s.impropers[0].type, 'i1');
        self.assertEqual(s.impropers[0].a, 3);
        self.assertEqual(s.impropers[0].b, 2);
        self.assertEqual(s.impropers[0].c, 1);
        self.assertEqual(s.impropers[0].d, 0);

        # constraints
        self.assertEqual(len(s.constraints), 1)
        self.assertAlmostEqual(s.constraints[0].d, 2.5, 5)
        self.assertEqual(s.constraints[0].a, 0)
        self.assertEqual(s.constraints[0].b, 1)

    def test_bcast(self):
        # broadcast to all ranks
        self.snapshot.broadcast()

        # particles
        self.assertEqual(self.snapshot.particles.N, 4);
        self.assertEqual(tuple(self.snapshot.particles.position[0]), (0,1,2));
        self.assertEqual(tuple(self.snapshot.particles.velocity[0]), (10,11,12));
        self.assertEqual(tuple(self.snapshot.particles.acceleration[0]), (20,21,22));
        self.assertEqual(self.snapshot.particles.typeid[0], 0);
        self.assertEqual(self.snapshot.particles.mass[0], 33);
        self.assertEqual(self.snapshot.particles.charge[0], 44);
        self.assertEqual(self.snapshot.particles.diameter[0], 55);
        self.assertEqual(tuple(self.snapshot.particles.image[0]), (60,61,62));

        self.assertEqual(tuple(self.snapshot.particles.position[1]), (1,2,3));
        self.assertEqual(tuple(self.snapshot.particles.velocity[1]), (11,12,13));
        self.assertEqual(tuple(self.snapshot.particles.acceleration[1]), (21,22,23));
        self.assertEqual(self.snapshot.particles.typeid[1], 0);
        self.assertEqual(self.snapshot.particles.mass[1], 34);
        self.assertEqual(self.snapshot.particles.charge[1], 45);
        self.assertEqual(self.snapshot.particles.diameter[1], 56);
        self.assertEqual(tuple(self.snapshot.particles.image[1]), (61,62,63));

        self.assertEqual(tuple(self.snapshot.particles.position[2]), (0,-1,-2));
        self.assertEqual(tuple(self.snapshot.particles.velocity[2]), (12,13,14));
        self.assertEqual(tuple(self.snapshot.particles.acceleration[2]), (22,23,24));
        self.assertEqual(self.snapshot.particles.typeid[2], 1);
        self.assertEqual(self.snapshot.particles.mass[2], 35);
        self.assertEqual(self.snapshot.particles.charge[2], 46);
        self.assertEqual(self.snapshot.particles.diameter[2], 57);
        self.assertEqual(tuple(self.snapshot.particles.image[2]), (62,63,64));

        self.assertEqual(tuple(self.snapshot.particles.position[3]), (-1,-2,-3));
        self.assertEqual(tuple(self.snapshot.particles.velocity[3]), (13,14,15));
        self.assertEqual(tuple(self.snapshot.particles.acceleration[3]), (23,24,25));
        self.assertEqual(self.snapshot.particles.typeid[3], 1);
        self.assertEqual(self.snapshot.particles.mass[3], 36);
        self.assertEqual(self.snapshot.particles.charge[3], 47);
        self.assertEqual(self.snapshot.particles.diameter[3], 58);
        self.assertEqual(tuple(self.snapshot.particles.image[3]), (63,64,65));

        # bonds
        self.assertEqual(self.snapshot.bonds.N, 2);
        self.assertEqual(self.snapshot.bonds.typeid[0], 0);
        self.assertEqual(self.snapshot.bonds.group[0,0], 0);
        self.assertEqual(self.snapshot.bonds.group[0,1], 1);

        self.assertEqual(self.snapshot.bonds.typeid[1], 1);
        self.assertEqual(self.snapshot.bonds.group[1,0], 2);
        self.assertEqual(self.snapshot.bonds.group[1,1], 3);

        # angles
        self.assertEqual(self.snapshot.angles.N, 2);
        self.assertEqual(self.snapshot.angles.typeid[0], 1);
        self.assertEqual(self.snapshot.angles.group[0,0], 0);
        self.assertEqual(self.snapshot.angles.group[0,1], 1);
        self.assertEqual(self.snapshot.angles.group[0,2], 2);

        self.assertEqual(self.snapshot.angles.typeid[1], 0);
        self.assertEqual(self.snapshot.angles.group[1,0], 2);
        self.assertEqual(self.snapshot.angles.group[1,1], 3);
        self.assertEqual(self.snapshot.angles.group[1,2], 0);

        # dihedrals
        self.assertEqual(self.snapshot.dihedrals.N, 1);
        self.assertEqual(self.snapshot.dihedrals.typeid[0], 0);
        self.assertEqual(self.snapshot.dihedrals.group[0,0], 0);
        self.assertEqual(self.snapshot.dihedrals.group[0,1], 1);
        self.assertEqual(self.snapshot.dihedrals.group[0,2], 2);
        self.assertEqual(self.snapshot.dihedrals.group[0,3], 3);

        # impropers
        self.assertEqual(self.snapshot.impropers.N, 1);
        self.assertEqual(self.snapshot.impropers.typeid[0], 0);
        self.assertEqual(self.snapshot.impropers.group[0,0], 3);
        self.assertEqual(self.snapshot.impropers.group[0,1], 2);
        self.assertEqual(self.snapshot.impropers.group[0,2], 1);
        self.assertEqual(self.snapshot.impropers.group[0,3], 0);

        # constraints
        self.assertEqual(self.snapshot.constraints.N, 1)
        self.assertAlmostEqual(self.snapshot.constraints.value[0], 2.5, 5)
        self.assertEqual(self.snapshot.constraints.group[0,0], 0)
        self.assertEqual(self.snapshot.constraints.group[0,1], 1)

    def tearDown(self):
        context.initialize();


# test make_snapshot and read_snapshot in double precision
class init_take_snapshot_double (unittest.TestCase):
    def setUp(self):
        self.snapshot = data.make_snapshot(N=4, box=data.boxdim(L=10), dtype='double');
        if comm.get_rank() == 0:
            # particles
            self.snapshot.particles.position[0] = [0,1,2];
            self.snapshot.particles.position[1] = [1,2,3];
            self.snapshot.particles.position[2] = [0,-1,-2];
            self.snapshot.particles.position[3] = [-1, -2, -3];
            self.snapshot.particles.velocity[0] = [10, 11, 12];
            self.snapshot.particles.velocity[1] = [11, 12, 13];
            self.snapshot.particles.velocity[2] = [12, 13, 14];
            self.snapshot.particles.velocity[3] = [13, 14, 15];
            self.snapshot.particles.acceleration[0] = [20, 21, 22];
            self.snapshot.particles.acceleration[1] = [21, 22, 23];
            self.snapshot.particles.acceleration[2] = [22, 23, 24];
            self.snapshot.particles.acceleration[3] = [23, 24, 25];
            self.snapshot.particles.typeid[:] = [0,0,1,1];
            self.snapshot.particles.mass[:] = [33, 34, 35,  36];
            self.snapshot.particles.charge[:] = [44, 45, 46, 47];
            self.snapshot.particles.diameter[:] = [55, 56, 57, 58];
            self.snapshot.particles.image[0] = [60, 61, 62];
            self.snapshot.particles.image[1] = [61, 62, 63];
            self.snapshot.particles.image[2] = [62, 63, 64];
            self.snapshot.particles.image[3] = [63, 64, 65];
            self.snapshot.particles.types = ['p1', 'p2'];

            # bonds
            self.snapshot.bonds.types = ['b1', 'b2'];
            self.snapshot.bonds.resize(2);
            self.snapshot.bonds.typeid[:] = [0, 1];
            self.snapshot.bonds.group[0] = [0, 1];
            self.snapshot.bonds.group[1] = [2, 3];

            # angles
            self.snapshot.angles.types = ['a1', 'a2'];
            self.snapshot.angles.resize(2);
            self.snapshot.angles.typeid[:] = [1, 0];
            self.snapshot.angles.group[0] = [0, 1, 2];
            self.snapshot.angles.group[1] = [2, 3, 0];

            # dihedrals
            self.snapshot.dihedrals.types = ['d1'];
            self.snapshot.dihedrals.resize(1);
            self.snapshot.dihedrals.typeid[:] = [0];
            self.snapshot.dihedrals.group[0] = [0, 1, 2, 3];

            # impropers
            self.snapshot.impropers.types = ['i1'];
            self.snapshot.impropers.resize(1);
            self.snapshot.impropers.typeid[:] = [0];
            self.snapshot.impropers.group[0] = [3, 2, 1, 0];

            # constraints
            self.snapshot.constraints.resize(1)
            self.snapshot.constraints.group[0] = [0, 1]
            self.snapshot.constraints.value[0] = 2.5

    def test_read_snapshot(self):
        s = init.read_snapshot(self.snapshot);
        self.assertTrue(s);
        self.assertTrue(s.sysdef);

        # particles
        self.assertEqual(len(s.particles), 4);
        self.assertEqual(s.particles[0].position, (0,1,2));
        self.assertEqual(s.particles[0].velocity, (10,11,12));
        self.assertEqual(s.particles[0].acceleration, (20,21,22));
        self.assertEqual(s.particles[0].type, 'p1');
        self.assertEqual(s.particles[0].mass, 33);
        self.assertEqual(s.particles[0].charge, 44);
        self.assertEqual(s.particles[0].diameter, 55);
        self.assertEqual(s.particles[0].image, (60,61,62));

        self.assertEqual(s.particles[1].position, (1,2,3));
        self.assertEqual(s.particles[1].velocity, (11,12,13));
        self.assertEqual(s.particles[1].acceleration, (21,22,23));
        self.assertEqual(s.particles[1].type, 'p1');
        self.assertEqual(s.particles[1].mass, 34);
        self.assertEqual(s.particles[1].charge, 45);
        self.assertEqual(s.particles[1].diameter, 56);
        self.assertEqual(s.particles[1].image, (61,62,63));

        self.assertEqual(s.particles[2].position, (0,-1,-2));
        self.assertEqual(s.particles[2].velocity, (12,13,14));
        self.assertEqual(s.particles[2].acceleration, (22,23,24));
        self.assertEqual(s.particles[2].type, 'p2');
        self.assertEqual(s.particles[2].mass, 35);
        self.assertEqual(s.particles[2].charge, 46);
        self.assertEqual(s.particles[2].diameter, 57);
        self.assertEqual(s.particles[2].image, (62,63,64));

        self.assertEqual(s.particles[3].position, (-1,-2,-3));
        self.assertEqual(s.particles[3].velocity, (13,14,15));
        self.assertEqual(s.particles[3].acceleration, (23,24,25));
        self.assertEqual(s.particles[3].type, 'p2');
        self.assertEqual(s.particles[3].mass, 36);
        self.assertEqual(s.particles[3].charge, 47);
        self.assertEqual(s.particles[3].diameter, 58);
        self.assertEqual(s.particles[3].image, (63,64,65));

        # bonds
        self.assertEqual(len(s.bonds), 2);
        self.assertEqual(s.bonds[0].type, 'b1');
        self.assertEqual(s.bonds[0].a, 0);
        self.assertEqual(s.bonds[0].b, 1);

        self.assertEqual(s.bonds[1].type, 'b2');
        self.assertEqual(s.bonds[1].a, 2);
        self.assertEqual(s.bonds[1].b, 3);

        # angles
        self.assertEqual(len(s.angles), 2);
        self.assertEqual(s.angles[0].type, 'a2');
        self.assertEqual(s.angles[0].a, 0);
        self.assertEqual(s.angles[0].b, 1);
        self.assertEqual(s.angles[0].c, 2);

        self.assertEqual(s.angles[1].type, 'a1');
        self.assertEqual(s.angles[1].a, 2);
        self.assertEqual(s.angles[1].b, 3);
        self.assertEqual(s.angles[1].c, 0);

        # dihedrals
        self.assertEqual(len(s.dihedrals), 1);
        self.assertEqual(s.dihedrals[0].type, 'd1');
        self.assertEqual(s.dihedrals[0].a, 0);
        self.assertEqual(s.dihedrals[0].b, 1);
        self.assertEqual(s.dihedrals[0].c, 2);
        self.assertEqual(s.dihedrals[0].d, 3);

        # impropers
        self.assertEqual(len(s.impropers), 1);
        self.assertEqual(s.impropers[0].type, 'i1');
        self.assertEqual(s.impropers[0].a, 3);
        self.assertEqual(s.impropers[0].b, 2);
        self.assertEqual(s.impropers[0].c, 1);
        self.assertEqual(s.impropers[0].d, 0);

        # constraints
        self.assertEqual(len(s.constraints), 1)
        self.assertAlmostEqual(s.constraints[0].d, 2.5, 5)
        self.assertEqual(s.constraints[0].a, 0)
        self.assertEqual(s.constraints[0].b, 1)

    def test_bcast(self):
        # broadcast to all ranks
        self.snapshot.broadcast()

        # particles
        self.assertEqual(self.snapshot.particles.N, 4);
        self.assertEqual(tuple(self.snapshot.particles.position[0]), (0,1,2));
        self.assertEqual(tuple(self.snapshot.particles.velocity[0]), (10,11,12));
        self.assertEqual(tuple(self.snapshot.particles.acceleration[0]), (20,21,22));
        self.assertEqual(self.snapshot.particles.typeid[0], 0);
        self.assertEqual(self.snapshot.particles.mass[0], 33);
        self.assertEqual(self.snapshot.particles.charge[0], 44);
        self.assertEqual(self.snapshot.particles.diameter[0], 55);
        self.assertEqual(tuple(self.snapshot.particles.image[0]), (60,61,62));

        self.assertEqual(tuple(self.snapshot.particles.position[1]), (1,2,3));
        self.assertEqual(tuple(self.snapshot.particles.velocity[1]), (11,12,13));
        self.assertEqual(tuple(self.snapshot.particles.acceleration[1]), (21,22,23));
        self.assertEqual(self.snapshot.particles.typeid[1], 0);
        self.assertEqual(self.snapshot.particles.mass[1], 34);
        self.assertEqual(self.snapshot.particles.charge[1], 45);
        self.assertEqual(self.snapshot.particles.diameter[1], 56);
        self.assertEqual(tuple(self.snapshot.particles.image[1]), (61,62,63));

        self.assertEqual(tuple(self.snapshot.particles.position[2]), (0,-1,-2));
        self.assertEqual(tuple(self.snapshot.particles.velocity[2]), (12,13,14));
        self.assertEqual(tuple(self.snapshot.particles.acceleration[2]), (22,23,24));
        self.assertEqual(self.snapshot.particles.typeid[2], 1);
        self.assertEqual(self.snapshot.particles.mass[2], 35);
        self.assertEqual(self.snapshot.particles.charge[2], 46);
        self.assertEqual(self.snapshot.particles.diameter[2], 57);
        self.assertEqual(tuple(self.snapshot.particles.image[2]), (62,63,64));

        self.assertEqual(tuple(self.snapshot.particles.position[3]), (-1,-2,-3));
        self.assertEqual(tuple(self.snapshot.particles.velocity[3]), (13,14,15));
        self.assertEqual(tuple(self.snapshot.particles.acceleration[3]), (23,24,25));
        self.assertEqual(self.snapshot.particles.typeid[3], 1);
        self.assertEqual(self.snapshot.particles.mass[3], 36);
        self.assertEqual(self.snapshot.particles.charge[3], 47);
        self.assertEqual(self.snapshot.particles.diameter[3], 58);
        self.assertEqual(tuple(self.snapshot.particles.image[3]), (63,64,65));

        # bonds
        self.assertEqual(self.snapshot.bonds.N, 2);
        self.assertEqual(self.snapshot.bonds.typeid[0], 0);
        self.assertEqual(self.snapshot.bonds.group[0,0], 0);
        self.assertEqual(self.snapshot.bonds.group[0,1], 1);

        self.assertEqual(self.snapshot.bonds.typeid[1], 1);
        self.assertEqual(self.snapshot.bonds.group[1,0], 2);
        self.assertEqual(self.snapshot.bonds.group[1,1], 3);

        # angles
        self.assertEqual(self.snapshot.angles.N, 2);
        self.assertEqual(self.snapshot.angles.typeid[0], 1);
        self.assertEqual(self.snapshot.angles.group[0,0], 0);
        self.assertEqual(self.snapshot.angles.group[0,1], 1);
        self.assertEqual(self.snapshot.angles.group[0,2], 2);

        self.assertEqual(self.snapshot.angles.typeid[1], 0);
        self.assertEqual(self.snapshot.angles.group[1,0], 2);
        self.assertEqual(self.snapshot.angles.group[1,1], 3);
        self.assertEqual(self.snapshot.angles.group[1,2], 0);

        # dihedrals
        self.assertEqual(self.snapshot.dihedrals.N, 1);
        self.assertEqual(self.snapshot.dihedrals.typeid[0], 0);
        self.assertEqual(self.snapshot.dihedrals.group[0,0], 0);
        self.assertEqual(self.snapshot.dihedrals.group[0,1], 1);
        self.assertEqual(self.snapshot.dihedrals.group[0,2], 2);
        self.assertEqual(self.snapshot.dihedrals.group[0,3], 3);

        # impropers
        self.assertEqual(self.snapshot.impropers.N, 1);
        self.assertEqual(self.snapshot.impropers.typeid[0], 0);
        self.assertEqual(self.snapshot.impropers.group[0,0], 3);
        self.assertEqual(self.snapshot.impropers.group[0,1], 2);
        self.assertEqual(self.snapshot.impropers.group[0,2], 1);
        self.assertEqual(self.snapshot.impropers.group[0,3], 0);

        # constraints
        self.assertEqual(self.snapshot.constraints.N, 1)
        self.assertAlmostEqual(self.snapshot.constraints.value[0], 2.5, 5)
        self.assertEqual(self.snapshot.constraints.group[0,0], 0)
        self.assertEqual(self.snapshot.constraints.group[0,1], 1)

    def tearDown(self):
        context.initialize();


class init_take_snapshot (unittest.TestCase):
    def test_make_snapshot1(self):
        snapshot = data.make_snapshot(N=10,
                                      box=data.boxdim(L=20),
                                      particle_types=['p1', 'p2'],
                                      bond_types=['polymer'],
                                      angle_types=['a1', 'a2'],
                                      dihedral_types=['dihedralA', 'dihedralB'],
                                      improper_types=['improperA', 'improperB', 'improperC']);

        if comm.get_rank() == 0:
            self.assertEqual(snapshot.particles.N, 10);
            self.assertEqual(snapshot.particles.types, ['p1', 'p2']);
            self.assertEqual(snapshot.bonds.N, 0);
            self.assertEqual(snapshot.bonds.types, ['polymer']);
            self.assertEqual(snapshot.angles.N, 0);
            self.assertEqual(snapshot.angles.types, ['a1', 'a2']);
            self.assertEqual(snapshot.dihedrals.N, 0);
            self.assertEqual(snapshot.dihedrals.types, ['dihedralA', 'dihedralB']);
            self.assertEqual(snapshot.impropers.N, 0);
            self.assertEqual(snapshot.impropers.types, ['improperA', 'improperB', 'improperC']);
            self.assertEqual(snapshot.constraints.N, 0)

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
