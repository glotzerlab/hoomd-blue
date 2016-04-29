# -*- coding: iso-8859-1 -*-

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# test multiple contexts with a simple run
class multi_context(unittest.TestCase):
    def test_run(self):
        c1 = context.SimulationContext()
        c2 = context.SimulationContext()

        with c1:
            init.create_random(N=2000, phi_p=0.2)
            lj = md.pair.lj(r_cut=3.0)
            lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

            md.integrate.mode_standard(dt=0.005)
            md.integrate.nvt(group=group.all(), T=1.2, tau=0.5)

        with c2:
            init.create_random(N=1000, phi_p=0.02)
            lj = md.pair.lj(r_cut=3.0)
            lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

            md.integrate.mode_standard(dt=0.005)
            md.integrate.nvt(group=group.all(), T=1.2, tau=0.5)

        with c1:
            run(10)

        with c2:
            run(10)

    def test_replace(self):
        old = context.current;
        c1 = context.SimulationContext()

        with c1:
            pass

        # current context should be reset to the old
        print(context.current)
        print(old)
        print(c1)
        self.assertTrue(context.current is old);

    def test_set_current(self):
        c1 = context.SimulationContext()
        c1.set_current();
        self.assertTrue(context.current is c1);
        c1.on_gpu();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

