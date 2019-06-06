# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os

# tests md.force.constant
class force_constant_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05

        context.current.sorter.set_params(grid=8)

    # test to see that se can create a md.force.constant
    def test_create(self):
        md.force.constant(fx=1.0, fy=0.5, fz=0.74);

    # test changing the force
    def test_change_force(self):
        const = md.force.constant(fx=1.0, fy=0.5, fz=0.74);
        const.set_force(fx=1.45, fy=0.25, fz=-0.1);

    # test changing the force
    def test_change_particle_force(self):
        const = md.force.constant(fx=1.0, fy=0.5, fz=0.74);
        const.set_force(tag=1, fx=1.45, fy=0.25, fz=-0.1);
        const.set_force(tag=99, fx=2.34, fy=1.43, fz=-2.3);

        md.integrate.mode_standard(dt=0.005)
        nve = md.integrate.nve(group=group.all())

        run(1000)

        self.assertRaises(RuntimeError, const.set_force, tag=100, fx=23,fy=44,fz=55)

        self.assertAlmostEqual(const.forces[0].force[0], 1.0, 5)
        self.assertAlmostEqual(const.forces[0].force[1], 0.5, 5)
        self.assertAlmostEqual(const.forces[0].force[2], 0.74, 5)

        self.assertAlmostEqual(const.forces[1].force[0], 1.45, 5)
        self.assertAlmostEqual(const.forces[1].force[1], 0.25, 5)
        self.assertAlmostEqual(const.forces[1].force[2], -0.1, 5)

        self.assertAlmostEqual(const.forces[99].force[0], 2.34, 5)
        self.assertAlmostEqual(const.forces[99].force[1], 1.43, 5)
        self.assertAlmostEqual(const.forces[99].force[2], -2.3, 5)

    def test_callback(self):
        self.count = 0
        const = md.force.constant(fx=1.0, fy=0.5, fz=0.74)

        def callback(timestep):
            self.count += 1
        const.set_callback(callback)

        md.integrate.mode_standard(dt=0.005)
        nve = md.integrate.nve(group=group.all())

        run(1000)

        self.assertGreaterEqual(self.count,1001)
        count_old = self.count

        const.set_callback(None)
        run(100)

        self.assertEqual(self.count,count_old)

    # test the initialization checks
    def test_init_checks(self):
        const = md.force.constant(fx=1.0, fy=0.5, fz=0.74);
        const.cpp_force = None;

        self.assertRaises(RuntimeError, const.set_force, fx=1.45, fy=0.25, fz=-0.1);
        self.assertRaises(RuntimeError, const.enable);
        self.assertRaises(RuntimeError, const.disable);

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
