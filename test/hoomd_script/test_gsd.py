# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import hoomd_script;
context.initialize()
import unittest
import os

# unit tests for dump.xml
class gsd_write_tests (unittest.TestCase):
    def setUp(self):
        print
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

        self.s = init.read_snapshot(self.snapshot);
        sorter.set_params(grid=8)

    # tests basic creation of the dump
    def test(self):
        dump.gsd(filename="test.gsd", group=group.all(), period=1);
        run(5);

    # tests with phase
    def test_phase(self):
        dump.gsd(filename="test.gsd", group=group.all(), period=1, phase=0);
        run(1);

    # tests overwrite
    def test_overwrite(self):
        with open('test.gsd', 'wt') as f:
            f.write('Hello');
        dump.gsd(filename="test.gsd", group=group.all(), period=1, overwrite=True);
        run(1);

    # tests truncate
    def test_truncate(self):
        dump.gsd(filename="test.gsd", group=group.all(), period=1, truncate=True);
        run(5);

    # test all static quantities
    def test_all_static(self):
        dump.gsd(filename="test.gsd", group=group.all(), period=1, static=['attribute', 'property', 'momentum', 'topology']);
        run(1);

    # test write file
    def test_write_immediate(self):
        dump.gsd(filename="test.gsd", group=group.all(), period=None, time_step=1000);

    def tearDown(self):
        os.remove('test.gsd');
        context.initialize();

# TODO: test reading in data with init.read_gsd and compare to the given snapshot

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
