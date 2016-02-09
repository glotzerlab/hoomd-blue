# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import hoomd_script;
context.initialize()
import unittest
import os
import numpy

# pair.lj
class pair_set_energy_tests (unittest.TestCase):
    def setUp(self):
        print
        self.N=1000;
        self.s = init.create_random(N=self.N, phi_p=0.05);
        sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        lj = pair.lj(r_cut=3.0);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        lj.update_coeffs();

        all = group.all();
        integrate.mode_standard(dt=0.0)
        integrate.nvt(group=all, T=1.2, tau=0.5)
        run(1, quiet=True);

        t1 = numpy.array([0], dtype=numpy.int32);
        t2 = numpy.array(numpy.linspace(1, self.N-1, self.N-1), dtype=numpy.int32);
        eng = lj.compute_energy(t1, t2);

        # tags = numpy.linspace(0, self.N-1, self.N, dtype=numpy.int32);
        # print("Even and odd Energy = ", lj.compute_energy(tags1=numpy.array(tags[0:self.N:2]), tags2=numpy.array(tags[1:self.N:2])))

        self.assertAlmostEqual(eng/2.0, self.s.particles.get(0).net_energy, places=5);

    def tearDown(self):
        self.s = None
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
