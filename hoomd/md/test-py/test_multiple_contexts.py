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
            init.create_lattice(lattice.sc(a=1.3782337338022654),n=[10,10,20]) #target a packing fraction of 0.2
            nl = md.nlist.cell()
            lj = md.pair.lj(r_cut=3.0, nlist = nl)
            lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

            md.integrate.mode_standard(dt=0.005)
            md.integrate.nvt(group=group.all(), kT=1.2, tau=0.5)

        with c2:
            init.create_lattice(lattice.sc(a=2.9693145670757697),n=[10,10,10]) #target a packing fraction of 0.02
            nl = md.nlist.cell()
            lj = md.pair.lj(r_cut=3.0, nlist = nl)
            lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

            md.integrate.mode_standard(dt=0.005)
            md.integrate.nvt(group=group.all(), kT=1.2, tau=0.5)

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
