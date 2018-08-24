# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os
import numpy

# md.pair.lj
class pair_set_energy_tests (unittest.TestCase):
    def setUp(self):
        print
        self.N=1000;
        self.s = init.create_lattice(lattice.sc(a=1.5),n=[10,10,10]); # move particles close so they interact
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        lj = md.pair.lj(r_cut=3.0, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        lj.update_coeffs();

        all = group.all();
        md.integrate.mode_standard(dt=0.0)
        md.integrate.nve(group=all)
        run(1, quiet=True);

        t1 = numpy.array([0], dtype=numpy.int32);
        t2 = numpy.array(numpy.linspace(1, self.N-1, self.N-1), dtype=numpy.int32);
        eng = lj.compute_energy(t1, t2);

        # tags = numpy.linspace(0, self.N-1, self.N, dtype=numpy.int32);
        # print("Even and odd Energy = ", lj.compute_energy(tags1=numpy.array(tags[0:self.N:2]), tags2=numpy.array(tags[1:self.N:2])))

        self.assertAlmostEqual(eng/2.0, self.s.particles.get(0).net_energy, places=5);

    def tearDown(self):
        del self.s, self.nl
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
