# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os
import numpy

# tests md.improper.harmonic
class improper_harmonic_tests (unittest.TestCase):
    def setUp(self):
        print
        snap = data.make_snapshot(N=40,
                                  box=data.boxdim(L=100),
                                  particle_types = ['A'],
                                  bond_types = [],
                                  angle_types = [],
                                  dihedral_types = [],
                                  improper_types = ['improperA'])

        if comm.get_rank() == 0:
            snap.impropers.resize(10);

            for i in range(10):
                x = numpy.array([i, 0, 0], dtype=numpy.float32)
                snap.particles.position[4*i+0,:] = x;
                x += numpy.random.random(3)
                snap.particles.position[4*i+1,:] = x;
                x += numpy.random.random(3)
                snap.particles.position[4*i+2,:] = x;
                x += numpy.random.random(3)
                snap.particles.position[4*i+3,:] = x;

                snap.impropers.group[i,:] = [4*i+0, 4*i+1, 4*i+2, 4*i+3];

        init.read_snapshot(snap)

        context.current.sorter.set_params(grid=8)

    # test to see that se can create an angle.harmonic
    def test_create(self):
        md.improper.harmonic();

    # test setting coefficients
    def test_set_coeff(self):
        harmonic = md.improper.harmonic();
        harmonic.improper_coeff.set('improperA', k=30.0, chi=1.57)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100);

    # test coefficient not set checking
    def test_set_coeff_fail(self):
        harmonic = md.improper.harmonic();
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
