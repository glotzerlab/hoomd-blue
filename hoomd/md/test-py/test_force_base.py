# -*- coding: iso-8859-1 -*-
# Maintainer: jglaser

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# md.pair.lj
class force_base_tests (unittest.TestCase):
    def setUp(self):
        print
        self.N=1000;
        self.s = init.create_random(N=self.N, phi_p=0.05);
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        lj = md.pair.lj(r_cut=3.0);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        lj.update_coeffs();

        all = group.all();
        md.integrate.mode_standard(dt=0.0)
        md.integrate.nvt(group=all, T=1.2, tau=0.5)
        run(1, quiet=True);

        g = group.tag_list(name='ptl0', tags=[0])
        energy = lj.get_energy(g)
        self.assertAlmostEqual(energy, self.s.particles.get(0).net_energy, places=5);

    def tearDown(self):
        self.s = None
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
