# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# tests for data access
class data_access_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_random(N=100, phi_p=0.05);
        import __main__;
        __main__.sorter.set_params(grid=8)

    # tests reading/setting of the box
    def test_box(self):
        self.s.box = (15, 20, 30);
        b = self.s.box;
        self.assertEqual(3, len(b));
        self.assertAlmostEqual(15, b[0], 5)
        self.assertAlmostEqual(20, b[1], 5)
        self.assertAlmostEqual(30, b[2], 5)

    # test reading/setting of the dimensions
    def test_dimensions(self):
        self.s.dimensions = 2;
        self.assertEqual(self.s.dimensions, 2);
        self.s.dimensions = 3;

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
    
    def tearDown(self):
        del self.s
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

