# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# pair.lj
class pair_set_energy_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_random(N=100, phi_p=0.05);
        sorter.set_params(grid=8) # not sure what this does

    # basic test of creation
    def test(self):
        lj = pair.lj(r_cut=3.0);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        lj.update_coeffs();

        all = group.all();
        integrate.mode_standard(dt=0.005)
        integrate.nvt(group=all, T=1.2, tau=0.5)
        run(100, quiet=True);
        import numpy as np
        t1 = np.array([0], dtype=np.int64);
        t2 = np.linspace(1, 99, 99, dtype=np.int64);
        eng = lj.compute_energy(t1.tolist(), t2.tolist());
        self.assertAlmostEqual(eng/2.0, self.s.particles.get(0).net_energy, places=5); # do this for all particles?

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
