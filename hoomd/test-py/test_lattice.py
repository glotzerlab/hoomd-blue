# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
import hoomd;
context.initialize()
import unittest
import os
import numpy
import math

# unit tests for sq
class lattice_sq_test (unittest.TestCase):
    def test_init(self):
        sysdef = init.create_lattice(unitcell=lattice.sq(a=2.0),
                                     n=[1,2]);

        snap = sysdef.take_snapshot();
        if comm.get_rank() == 0:
            self.assertEqual(snap.box.dimensions, 2);
            numpy.testing.assert_allclose(snap.box.Lx, 2.0);
            numpy.testing.assert_allclose(snap.box.Ly, 4.0);
            self.assertEqual(snap.particles.types, ['A']);
            numpy.testing.assert_allclose(snap.particles.position, [[0,-1,0], [0,1,0]]);

    def test_type(self):
        sysdef = init.create_lattice(unitcell=lattice.sq(a=1.0, type_name='B'),
                                     n=1);

        snap = sysdef.take_snapshot();
        if comm.get_rank() == 0:
            self.assertEqual(snap.particles.types, ['B']);

    def tearDown(self):
        context.initialize();

# unit tests for hex
class lattice_hex_test (unittest.TestCase):
    def test_init(self):
        sysdef = init.create_lattice(unitcell=lattice.hex(a=2.0),
                                     n=[1,2]);

        snap = sysdef.take_snapshot();
        if comm.get_rank() == 0:
            self.assertEqual(snap.box.dimensions, 2);
            numpy.testing.assert_allclose(snap.box.Lx, 2.0);
            numpy.testing.assert_allclose(snap.box.Ly, 4.0*math.sqrt(3));
            self.assertEqual(snap.particles.types, ['A']);
            # work around box wrapping bug in single precision
            if snap.particles.position[1,1] > 0:
                snap.particles.position[1,1] *= -1;
            numpy.testing.assert_allclose(snap.particles.position, [[0,-1*math.sqrt(3),0], [-1,-2*math.sqrt(3),0],
                                                                    [0,math.sqrt(3),0], [-1,0,0]],
                                                                    rtol=1e-5);

    def test_type(self):
        sysdef = init.create_lattice(unitcell=lattice.hex(a=1.0, type_name='B'),
                                     n=1);

        snap = sysdef.take_snapshot();
        if comm.get_rank() == 0:
            self.assertEqual(snap.particles.types, ['B']);

    def tearDown(self):
        context.initialize();

# unit tests for sc
class lattice_sc_test (unittest.TestCase):
    def test_init(self):
        sysdef = init.create_lattice(unitcell=lattice.sc(a=2.0),
                                     n=[2,2,1]);

        snap = sysdef.take_snapshot();
        if comm.get_rank() == 0:
            self.assertEqual(snap.box.dimensions, 3);
            numpy.testing.assert_allclose(snap.box.Lx, 4.0);
            numpy.testing.assert_allclose(snap.box.Ly, 4.0);
            numpy.testing.assert_allclose(snap.box.Lz, 2.0);
            self.assertEqual(snap.particles.types, ['A']);
            numpy.testing.assert_allclose(snap.particles.position, [[-1,-1,0],
                                                                    [-1,1,0],
                                                                    [1,-1,0],
                                                                    [1,1,0]]);

    def test_type(self):
        sysdef = init.create_lattice(unitcell=lattice.sc(a=1.0, type_name='B'),
                                     n=1);

        snap = sysdef.take_snapshot();
        if comm.get_rank() == 0:
            self.assertEqual(snap.particles.types, ['B']);

    def tearDown(self):
        context.initialize();

# unit tests for bcc
class lattice_bcc_test (unittest.TestCase):
    def test_init(self):
        sysdef = init.create_lattice(unitcell=lattice.bcc(a=2.0),
                                     n=1);

        snap = sysdef.take_snapshot();
        if comm.get_rank() == 0:
            self.assertEqual(snap.box.dimensions, 3);
            numpy.testing.assert_allclose(snap.box.Lx, 2.0);
            numpy.testing.assert_allclose(snap.box.Ly, 2.0);
            numpy.testing.assert_allclose(snap.box.Lz, 2.0);
            self.assertEqual(snap.particles.types, ['A']);
            numpy.testing.assert_allclose(snap.particles.position, [[0,0,0],
                                                                    [-1,-1,-1]])

    def test_type(self):
        sysdef = init.create_lattice(unitcell=lattice.bcc(a=1.0, type_name='B'),
                                     n=1);

        snap = sysdef.take_snapshot();
        if comm.get_rank() == 0:
            self.assertEqual(snap.particles.types, ['B']);

    def tearDown(self):
        context.initialize();

# unit tests for fcc
class lattice_fcc_test (unittest.TestCase):
    def test_init(self):
        sysdef = init.create_lattice(unitcell=lattice.fcc(a=2.0),
                                     n=1);

        snap = sysdef.take_snapshot();
        if comm.get_rank() == 0:
            self.assertEqual(snap.box.dimensions, 3);
            numpy.testing.assert_allclose(snap.box.Lx, 2.0);
            numpy.testing.assert_allclose(snap.box.Ly, 2.0);
            numpy.testing.assert_allclose(snap.box.Lz, 2.0);
            self.assertEqual(snap.particles.types, ['A']);
            numpy.testing.assert_allclose(snap.particles.position, [[0,0,0],
                                                                    [0,-1,-1],
                                                                    [-1,0,-1],
                                                                    [-1,-1,0]])

    def test_type(self):
        sysdef = init.create_lattice(unitcell=lattice.fcc(a=1.0, type_name='B'),
                                     n=1);

        snap = sysdef.take_snapshot();
        if comm.get_rank() == 0:
            self.assertEqual(snap.particles.types, ['B']);

    def tearDown(self):
        context.initialize();

# unit tests for uc
class lattice_unitcell_test (unittest.TestCase):
    def test_init(self):
        uc = hoomd.lattice.unitcell(N = 3,
                            a1 = [1, 0, 0],
                            a2 = [0, 1, 0],
                            a3 = [0, 0, 1],
                            dimensions = 3,
                            type_name = ["A", "A", "B"]);
        sysdef = init.create_lattice(unitcell=uc, n=1);
        snap = sysdef.take_snapshot();

    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
