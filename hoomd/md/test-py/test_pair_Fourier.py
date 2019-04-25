# -*- coding: iso-8859-1 -*-
# Maintainer: Pengji Zhou

# import math as m
# from hoomd import *
# from hoomd import md
# context.initialize()
# import unittest
# import os
# import numpy as np
#
# def fourier(para,r):
#     # do fourier potential calculation by python
#     r_cut = 3.0
#     degree = 3
#     if r>r_cut:
#         energy = 0
#     else:
#         energy = 1/r**12
#         for i in range(0,degree):
#             energy += 1/r**2 * (para[i] * (np.cos(np.pi*r/r_cut*(i+2)) + np.cos(np.pi*r/r_cut) * (-1)**i))
#         for i in range(2,11):
#             j = i + 7
#             energy += 1/r**2 * (para[j] * (np.sin(np.pi*r/r_cut*(i)) + i * np.sin(np.pi*r/r_cut) * (-1)**i))
#
#     print('python calculated energy: %.8f' %energy)
#     return energy,force
#
#     Energy_python = fourier(para,r)
#
#
# class pair_Fourier_test (unittest.TestCase):
#     def setUp(self):
#         context.initialize()
#         snapshot = data.make_snapshot(N=2, box=data.boxdim(L=1000.0))
#         if comm.get_rank() == 0:
#             # suppose spherical particles
#             snapshot.particles.position[0] = [0.0, 0.0, 0.0]
#             snapshot.particles.position[1] = [0.5, 0.75, 1.0]
#         self.system = init.read_snapshot(snapshot)
#
#
#     def test(self):
#
#         self.nl = md.nlist.cell()
#         Fourier = md.pair.Fourier(r_cut = 3.0, nlist = self.nl)
#         fourier_a = [0.08658918, -0.00177933, -0.0886236]
#         fourier_b = [-0.18217308, -0.04460936,0.06499778]
#         Fourier.pair_coeff.set('A', 'A', fourier_a = fourier_a, fourier_b = fourier_b)
#         md.integrate.mode_standard(dt = 0.0)
#         md.integrate.nve(group = group.all())
#         run(1)
#
#         force_foureir_1 = self.system.particles[0].net_force
#         potential_fourier_1 = self.system.particles[0].net_energy
#         np.testing.assert_allclose(-0.90639243 , force_fourier_1[0], rtol = 1e-8)
#         np.testing.assert_allclose(-0.50984574 , force_fourier_1[1], rtol = 1e-8)
#         np.testing.assert_allclose(-1.01969148 , force_fourier_1[2], rtol = 1e-8)
#         np.testing.assert_allclose(0.23740489 , potential_fourier_1, rtol = 1e-8)
#
#
#         force_fourier_2 = self.system.particles[1].net_force
#         potential_fourier_2 = self.system.particles[1].net_energy
#         print('kphi0_force_fourier_2', force_fourier_2)
#         print('kphi0_potential_fourier_2', potential_fourier_2)
#         np.testing.assert_allclose(0.90639243 , force_fourier_2[0], rtol = 1e-8)
#         np.testing.assert_allclose(0.50984574 , force_fourier_2[1], rtol = 1e-8)
#         np.testing.assert_allclose(1.01969148 , force_fourier_2[2], rtol = 1e-8)
#         np.testing.assert_allclose(0.23740489 , potential_fourier_2, rtol = 1e-8)
#
#
#
#     def tearDown(self):
#         del self.system, self.nl
#         context.initialize();
#
# if __name__ == '__main__':
#     unittest.main(argv = ['test.py', '-v'])
