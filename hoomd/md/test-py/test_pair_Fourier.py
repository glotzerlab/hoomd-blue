# -*- coding: iso-8859-1 -*-
# Maintainer: Pengji Zhou

import math as m
from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os
import numpy as np

# md.pair.Fourier
class pair_Fourier_test (unittest.TestCase):
    def setUp(self):
        context.initialize()
        snapshot = data.make_snapshot(N=2, box=data.boxdim(L=1000.0))
        if comm.get_rank() == 0:
            # suppose spherical particles
            snapshot.particles.position[0] = [0.0, 0.0, 0.0]
            snapshot.particles.position[1] = [0.5, 0.75, 1.0]
        self.system = init.read_snapshot(snapshot)


    def test(self):

        self.nl = md.nlist.cell()
        Fourier = md.pair.Fourier(r_cut = 3.0, nlist = self.nl)
        fourier_a = [0.08658918, -0.00177933, -0.0886236]
        fourier_b = [-0.18217308, -0.04460936,0.06499778]
        Fourier.pair_coeff.set('A', 'A', fourier_a = fourier_a, fourier_b = fourier_b)
        md.integrate.mode_standard(dt = 0.0)
        md.integrate.nve(group = group.all())
        run(1)
        force_x = 0.036262125554899126
        force_y = 0.05439318833234869
        force_z = 0.07252425110979825
        energy = -0.03563268310097793
        force_fourier_1 = self.system.particles[0].net_force
        potential_fourier_1 = self.system.particles[0].net_energy
        print(force_fourier_1)
        print(potential_fourier_1)
        np.testing.assert_allclose(force_x , force_fourier_1[0], rtol = 1e-8)
        np.testing.assert_allclose(force_y , force_fourier_1[1], rtol = 1e-8)
        np.testing.assert_allclose(force_z , force_fourier_1[2], rtol = 1e-8)
        np.testing.assert_allclose(energy , potential_fourier_1, rtol = 1e-8)

        force_fourier_2 = self.system.particles[1].net_force
        potential_fourier_2 = self.system.particles[1].net_energy
        print(force_fourier_1)
        print(potential_fourier_1)
        np.testing.assert_allclose(-force_x , force_fourier_2[0], rtol = 1e-8)
        np.testing.assert_allclose(-force_y , force_fourier_2[1], rtol = 1e-8)
        np.testing.assert_allclose(-force_z , force_fourier_2[2], rtol = 1e-8)
        np.testing.assert_allclose(energy , potential_fourier_2, rtol = 1e-8)



    def tearDown(self):
        del self.system, self.nl
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
