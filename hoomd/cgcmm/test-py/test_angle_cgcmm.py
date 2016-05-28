# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
from hoomd import cgcmm
context.initialize()
import unittest
import os
import numpy

# tests cgcmm.angle.cgcmm
class angle_cgcmm_tests (unittest.TestCase):
    def setUp(self):
        print
        # create a polymer system and add a few angles to it
        snap = data.make_snapshot(N=40,
                                  box=data.boxdim(L=100),
                                  particle_types = ['A'],
                                  bond_types = [],
                                  angle_types = ['angleA'],
                                  dihedral_types = [],
                                  improper_types = [])

        if comm.get_rank() == 0:
            snap.angles.resize(10);
            for i in range(10):
                x = numpy.array([i, 0, 0], dtype=numpy.float32)
                snap.particles.position[4*i+0,:] = x;
                x += numpy.random.random(3)
                snap.particles.position[4*i+1,:] = x;
                x += numpy.random.random(3)
                snap.particles.position[4*i+2,:] = x;
                x += numpy.random.random(3)
                snap.particles.position[4*i+3,:] = x;

                snap.angles.group[i,:] = [4*i+0, 4*i+1, 4*i+2];

        init.read_snapshot(snap)

        context.current.sorter.set_params(grid=8)

    # test to see that se can create an cgcmm.angle.cgcmm
    def test_create(self):
        cgcmm.angle.cgcmm();

    # test setting coefficients
    def test_set_coeff(self):
        cg = cgcmm.angle.cgcmm();
        cg.set_coeff('angleA', k=3.0, t0=0.7851, exponents=126, epsilon=1.0, sigma=0.53)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100);

    # test coefficient not set checking
    def test_set_coeff_fail(self):
        cg = cgcmm.angle.cgcmm();
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);

    def tearDown(self):
        context.initialize();



if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
