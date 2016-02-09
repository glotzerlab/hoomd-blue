# -*- coding: iso-8859-1 -*-

from hoomd_script import *
import hoomd_script;
context.initialize()
import unittest
import os

# unit test to run a simple polymer system with pair and bond potentials
class multi_context(unittest.TestCase):
    def test_run(self):
        self.c1 = context.SimulationContext()
        self.c2 = context.SimulationContext()

        with self.c1:
            init.create_random(N=500, phi_p=0.2)
            lj = pair.lj(r_cut=3.0)
            lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

            integrate.mode_standard(dt=0.005)
            integrate.nvt(group=group.all(), T=1.2, tau=0.5)

        with self.c2:
            init.create_random(N=200, phi_p=0.02)
            lj = pair.lj(r_cut=3.0)
            lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

            integrate.mode_standard(dt=0.005)
            integrate.nvt(group=group.all(), T=1.2, tau=0.5)

        with self.c1:
            run(10)

        with self.c2:
            run(10)

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
