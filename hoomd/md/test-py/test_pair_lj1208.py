# -*- coding: iso-8859-1 -*-
# Maintainer: unassigned

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# md.pair.lj1208
class pair_lj1208_tests (unittest.TestCase):
    def setUp(self):
        self.s = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4])

        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        lj1208 = md.pair.lj1208(r_cut=3.0, nlist = self.nl);
        lj1208.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, r_cut=2.5, r_on=2.0);
        lj1208.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        lj1208 = md.pair.lj1208(r_cut=3.0, nlist = self.nl);
        lj1208.pair_coeff.set('A', 'A', sigma=1.0, alpha=1.0);
        self.assertRaises(RuntimeError, lj1208.update_coeffs);

    # test missing coefficients
    def test_set_missing_sigma(self):
        lj1208 = md.pair.lj1208(r_cut=3.0, nlist = self.nl);
        lj1208.pair_coeff.set('A', 'A', epsilon=1.0, alpha=1.0);
        self.assertRaises(RuntimeError, lj1208.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        lj1208 = md.pair.lj1208(r_cut=3.0, nlist = self.nl);
        self.assertRaises(RuntimeError, lj1208.update_coeffs);

    # test set params
    def test_set_params(self):
        lj1208 = md.pair.lj1208(r_cut=3.0, nlist = self.nl);
        lj1208.set_params(mode="no_shift");
        lj1208.set_params(mode="shift");
        lj1208.set_params(mode="xplor");
        self.assertRaises(RuntimeError, lj1208.set_params, mode="blah");

    # test default coefficients
    def test_default_coeff(self):
        lj1208 = md.pair.lj1208(r_cut=3.0, nlist = self.nl);
        # (alpha, r_cut, and r_on are default)
        lj1208.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
        lj1208.update_coeffs()

    # test max rcut
    def test_max_rcut(self):
        lj1208 = md.pair.lj1208(r_cut=2.5, nlist = self.nl);
        lj1208.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
        self.assertAlmostEqual(2.5, lj1208.get_max_rcut());
        lj1208.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.assertAlmostEqual(2.0, lj1208.get_max_rcut());

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        lj1208 = md.pair.lj1208(r_cut=2.5, nlist = self.nl);

        lj1208.pair_coeff.set('A', 'A', sigma = 1.0, epsilon=1.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'));

        lj1208.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'));

    # test coeff list
    def test_coeff_list(self):
        lj1208 = md.pair.lj1208(r_cut=3.0, nlist = self.nl);
        lj1208.pair_coeff.set(['A', 'B'], ['A', 'C'], epsilon=1.0, sigma=1.0, alpha=1.0, r_cut=2.5, r_on=2.0);
        lj1208.update_coeffs();

    # test adding types
    def test_type_add(self):
        lj1208 = md.pair.lj1208(r_cut=3.0, nlist = self.nl);
        lj1208.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0);
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, lj1208.update_coeffs);
        lj1208.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0)
        lj1208.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0)
        lj1208.update_coeffs();

    # test value of the pair potential (adapted from test_special_pair_lj.py)
    def test_pair_lj1208_value(self):
        snap = data.make_snapshot(N=3,
                                  box=data.boxdim(L=100),
                                  particle_types = ['A'],
                                  pair_types = [],
                                  angle_types = [],
                                  dihedral_types = [],
                                  improper_types = [])

        if comm.get_rank() == 0:
            snap.particles.position[0] = (0, 0, 0)
            snap.particles.position[1] = (1.5, 0, 0)
            snap.particles.position[2] = (-0.75, 0, 0)

        self.s.restore_snapshot(snap)

        lj1208 = md.pair.lj1208(r_cut=3.0, nlist = self.nl);
        lj1208.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0, r_cut=3.0)
        all = group.all();
        md.integrate.mode_standard(dt=0);
        md.integrate.nve(all);
        run(1)

        # U =4 * epsilon [(sigma / r)^12 - (sigma / r)^8]
        # sum(U(i=0)) = U(i=0, j=1) + U(i=0, j=2)
        # sum(U(i=1)) = U(i=1, j=0) + U(i=1, j=2)
        # sum(U(i=2)) = U(i=2, j=0) + U(i=2, j=1)
        self.assertAlmostEqual(lj1208.forces[0].energy, 0.5 * (-0.125244 + 86.322282), 3)
        self.assertAlmostEqual(lj1208.forces[1].energy, 0.5 * (-0.125244 - 0.005852), 3)
        self.assertAlmostEqual(lj1208.forces[2].energy, 0.5 * (86.322282 - 0.005852), 3)

    def tearDown(self):
        del self.s, self.nl
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
