# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# tests for md.update.zero_momentum
class update_zero_momentum_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05

        context.current.sorter.set_params(grid=8)

    # tests basic creation of the updater
    def test(self):
        md.update.zero_momentum()
        run(100);

    # tests with phase
    def test_phase(self):
        md.update.zero_momentum(period=10, phase=0)
        run(100);

    # test variable periods
    def test_variable(self):
        md.update.zero_momentum(period = lambda n: n*100);
        run(100);

    # test if it actually zeros the momentum
    def test_zero(self):
        # set ONE particle one velocity on a particular rank to a nonzero value
        self.s.particles[0].velocity = (len(self.s.particles),0,0)
        log = analyze.log(filename=None, quantities=['momentum'],period=1)
        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group=group.all())
        run(1)
        self.assertAlmostEqual(log.query('momentum'),1.0,5)
        zero = md.update.zero_momentum(period=1)
        run(1)
        self.assertAlmostEqual(log.query('momentum'),0,5)

    def tearDown(self):
        del self.s
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
