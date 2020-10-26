# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os
import numpy

# tests md.dihedral.harmonic
class dihedral_harmonic_tests (unittest.TestCase):
    def setUp(self):
        print
        snap = data.make_snapshot(N=40,
                                  box=data.boxdim(L=100),
                                  particle_types = ['A'],
                                  bond_types = [],
                                  angle_types = [],
                                  dihedral_types = ['dihedralA'],
                                  improper_types = [])

        if comm.get_rank() == 0:
            snap.dihedrals.resize(10);
            for i in range(10):
                x = numpy.array([i, 0, 0], dtype=numpy.float32)
                snap.particles.position[4*i+0,:] = x;
                x += numpy.random.random(3)
                snap.particles.position[4*i+1,:] = x;
                x += numpy.random.random(3)
                snap.particles.position[4*i+2,:] = x;
                x += numpy.random.random(3)
                snap.particles.position[4*i+3,:] = x;

                snap.dihedrals.group[i,:] = [4*i+0, 4*i+1, 4*i+2, 4*i+3];

        init.read_snapshot(snap)

        context.current.sorter.set_params(grid=8)

    # test to see that se can create an angle.harmonic
    def test_create(self):
        md.dihedral.harmonic();

    # test setting coefficients
    def test_set_coeff(self):
        harmonic = md.dihedral.harmonic();
        harmonic.dihedral_coeff.set('dihedralA', k=1.0, d=1, n=4, phi_0=0)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100);
        
    # test setting coefficients with default phi_0
    def test_set_coeff_default(self):
        harmonic = md.dihedral.harmonic();
        harmonic.dihedral_coeff.set('dihedralA', k=1.0, d=1, n=4)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100);

    # test coefficient not set checking
    def test_set_coeff_fail(self):
        harmonic = md.dihedral.harmonic();
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);

    def tearDown(self):
        context.initialize();



if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
