# -*- coding: iso-8859-1 -*-
# Maintainer: unassigned

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# md.pair.buckingham
class pair_buckingham_tests (unittest.TestCase):
    def setUp(self):
        self.s = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4])

        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        buckingham = md.pair.buckingham(r_cut=3.0, nlist = self.nl);
        buckingham.pair_coeff.set('A', 'A', A=1.0, rho=1.0, C=1.0, r_cut=2.5, r_on=2.0);
        buckingham.update_coeffs();

    # test missing coefficients
    def test_set_missing_A(self):
        buckingham = md.pair.buckingham(r_cut=3.0, nlist = self.nl);
        buckingham.pair_coeff.set('A', 'A', rho=1.0, C=1.0);
        self.assertRaises(RuntimeError, buckingham.update_coeffs);

    # test missing coefficients
    def test_set_missing_rho(self):
        buckingham = md.pair.buckingham(r_cut=3.0, nlist = self.nl);
        buckingham.pair_coeff.set('A', 'A', A=1.0, C=1.0);
        self.assertRaises(RuntimeError, buckingham.update_coeffs);

    # test missing coefficients
    def test_set_missing_C(self):
        buckingham = md.pair.buckingham(r_cut=3.0, nlist = self.nl);
        buckingham.pair_coeff.set('A', 'A', A=1.0, rho=1.0);
        self.assertRaises(RuntimeError, buckingham.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        buckingham = md.pair.buckingham(r_cut=3.0, nlist = self.nl);
        self.assertRaises(RuntimeError, buckingham.update_coeffs);

    # test set params
    def test_set_params(self):
        buckingham = md.pair.buckingham(r_cut=3.0, nlist = self.nl);
        buckingham.set_params(mode="no_shift");
        buckingham.set_params(mode="shift");
        buckingham.set_params(mode="xplor");
        self.assertRaises(RuntimeError, buckingham.set_params, mode="blah");

    # test default coefficients
    def test_default_coeff(self):
        buckingham = md.pair.buckingham(r_cut=3.0, nlist = self.nl);
        # (alpha, r_cut, and r_on are default)
        buckingham.pair_coeff.set('A', 'A', A=1.0, rho=1.0, C=1.0)
        buckingham.update_coeffs()

    # test max rcut
    def test_max_rcut(self):
        buckingham = md.pair.buckingham(r_cut=2.5, nlist = self.nl);
        buckingham.pair_coeff.set('A', 'A', A=1.0, rho=1.0, C=1.0)
        self.assertAlmostEqual(2.5, buckingham.get_max_rcut());
        buckingham.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.assertAlmostEqual(2.0, buckingham.get_max_rcut());

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        buckingham = md.pair.buckingham(r_cut=2.5, nlist = self.nl);

        buckingham.pair_coeff.set('A', 'A', A=1.0, rho=1.0, C=1.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'));

        buckingham.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'));

    # test coeff list
    def test_coeff_list(self):
        buckingham = md.pair.buckingham(r_cut=3.0, nlist = self.nl);
        buckingham.pair_coeff.set(['A', 'B'], ['A', 'C'], A=1.0, rho=1.0, C=1.0, r_cut=2.5, r_on=2.0);
        buckingham.update_coeffs();

    # test adding types
    def test_type_add(self):
        buckingham = md.pair.buckingham(r_cut=3.0, nlist = self.nl);
        buckingham.pair_coeff.set('A', 'A', A=1.0, rho=1.0, C=1.0);
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, buckingham.update_coeffs);
        buckingham.pair_coeff.set('A', 'B', A=1.0, rho=1.0, C=1.0)
        buckingham.pair_coeff.set('B', 'B', A=1.0, rho=1.0, C=1.0)
        buckingham.update_coeffs();

    # test value of the pair potential (adapted from test_special_pair_lj.py)
    def test_pair_buckingham_value(self):
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

        buckingham = md.pair.buckingham(r_cut=3.0, nlist = self.nl);
        buckingham.pair_coeff.set('A', 'A', A=100.0, rho=0.6, C=10.0, r_cut=3.0)
        all = group.all();
        md.integrate.mode_standard(dt=0);
        md.integrate.nve(all);
        run(1)

        self.assertAlmostEqual(buckingham.forces[0].energy, 0.5 * (7.330585 - 27.536077), 3)
        self.assertAlmostEqual(buckingham.forces[1].energy, 0.5 * (7.330585 + 2.274701), 3)
        self.assertAlmostEqual(buckingham.forces[2].energy, 0.5 * (2.274701 - 27.536077), 3)

    def tearDown(self):
        del self.s, self.nl
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
