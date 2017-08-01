# -*- coding: iso-8859-1 -*-
# Maintainer: jglaser

from hoomd import *
from hoomd import md
import unittest
import os

context.initialize()

# md.pair.ewald
class pair_ewald_tests (unittest.TestCase):
    def setUp(self):
        print
        system = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        ewald = md.pair.ewald(r_cut=3.0, nlist = self.nl);
        ewald.pair_coeff.set('A', 'A', r_cut=1.0, kappa=1.0, alpha = 1.0);
        ewald.update_coeffs();

    # test missing coefficients
    def test_set_missing_kappa(self):
        ewald = md.pair.ewald(r_cut=3.0, nlist = self.nl);
        ewald.pair_coeff.set('A', 'A', alpha=1.0);
        self.assertRaises(RuntimeError, ewald.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        ewald = md.pair.ewald(r_cut=3.0, nlist = self.nl);
        self.assertRaises(RuntimeError, ewald.update_coeffs);

    # test nlist subscribe
    def test_nlist_subscribe(self):
        ewald = md.pair.ewald(r_cut=2.5, nlist = self.nl);

        ewald.pair_coeff.set('A', 'A', kappa=1.0, alpha=1.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'));

        ewald.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'));

    def tearDown(self):
        del self.nl
        context.initialize();

# test the validity of the pair potential
class test_pair_reaction_field_potential(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=2, box=data.boxdim(L=10),particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.charge[0] = 1
            snap.particles.charge[1] = 2
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (0.5,0,0)
        init.read_snapshot(snap)
        self.nl = md.nlist.cell()

    # test the calculation of force and potential
    def test_potential(self):
        ewald = md.pair.ewald(r_cut=2.0, nlist = self.nl)

        # basic test case
        ewald.pair_coeff.set('A','A', kappa=1.3, alpha=0.7)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        f0 = ewald.forces[0].force
        f1 = ewald.forces[1].force
        e0 = ewald.forces[0].energy
        e1 = ewald.forces[1].energy

        self.assertAlmostEqual(e0,0.5*1.38135,5)
        self.assertAlmostEqual(e1,0.5*1.38135,5)

        self.assertAlmostEqual(f0[0],-6.53712,5)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],6.53712,5)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    def tearDown(self):
        del self.nl
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
