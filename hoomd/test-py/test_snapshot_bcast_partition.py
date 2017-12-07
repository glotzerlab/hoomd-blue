# -*- coding: iso-8859-1 -*-
# Maintainer: jglaser

from hoomd import *
import hoomd;

if hoomd._hoomd.is_MPI_available():
    # initialize with every rank == one partition
    context.initialize('--nrank=1')
else:
    context.initialize('')

import unittest
import os
import sys
import numpy

# test make_snapshot, read_snapshot and broadcast
class test_bcast_float (unittest.TestCase):
    def setUp(self):
        self.snapshot = data.make_snapshot(N=4, box=data.boxdim(L=10), dtype='float');
        if comm.get_partition() == 0:
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

    def test_bcast_all(self):
        # broadcast to all ranks
        self.snapshot.broadcast_all()

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
        if hoomd._hoomd.is_MPI_available():
            # initialize with every rank == one partition
            context.initialize('--nrank=1')
        else:
            context.initialize('')

# test make_snapshot and read_snapshot in double precision
class test_bcast_double (unittest.TestCase):
    def setUp(self):
        self.snapshot = data.make_snapshot(N=4, box=data.boxdim(L=10), dtype='double');
        if comm.get_partition() == 0:
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

    def test_bcast_all(self):
        # broadcast to all ranks
        self.snapshot.broadcast_all()

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
        if hoomd._hoomd.is_MPI_available():
            # initialize with every rank == one partition
            context.initialize('--nrank=1')
        else:
            context.initialize('')

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
