# -*- coding: iso-8859-1 -*-
# Maintainer: jglaser

from hoomd import *
from hoomd import md
import unittest
import os

context.initialize()

# md.pair.DLVO
class pair_DLVO_tests (unittest.TestCase):
    def setUp(self):
        print
        system = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]);
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        rf = md.pair.DLVO(r_cut=3.0, nlist = self.nl);
        rf.pair_coeff.set('A', 'A', A=1.0, Z= 1.0, kappa=1.0);
        rf.update_coeffs();

    # test missing coefficients
    def test_missing_AA(self):
        rf = md.pair.DLVO(r_cut=3.0, nlist = self.nl);
        self.assertRaises(RuntimeError, rf.update_coeffs);

    # test set params
    def test_set_params(self):
        rf = md.pair.DLVO(r_cut=3.0, nlist = self.nl);
        rf.set_params(mode="no_shift");
        rf.set_params(mode="shift");
        rf.set_params(mode="xplor");
        self.assertRaises(RuntimeError, rf.set_params, mode="blah");

    # test nlist subscribe
    def test_nlist_subscribe(self):
        rf = md.pair.DLVO(r_cut=2.5, nlist = self.nl);

        rf.pair_coeff.set('A', 'A', epsilon=1.0, eps_rf=1.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'));

        rf.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'));

    def tearDown(self):
        del self.nl
        context.initialize();

# test the validity of the pair potential
class test_pair_DLVO_potential(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=2, box=data.boxdim(L=20),particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (1.5,0,0)
            snap.particles.diameter[0] = 0.2
            snap.particles.diameter[1] = 1.5
        init.read_snapshot(snap)
        self.nl = md.nlist.cell()

    # test the calculation of force and potential
    def test_potential(self):
        rf = md.pair.DLVO(r_cut=2.0, nlist = self.nl)

        # basic test case
        rf.pair_coeff.set('A','A', A=2.0, kappa=3.0, Z = 4.0)
        rf.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        f0 = rf.forces[0].force
        f1 = rf.forces[1].force
        e0 = rf.forces[0].energy
        e1 = rf.forces[1].energy

        self.assertAlmostEqual(e0,0.5*(0.0498935),6)
        self.assertAlmostEqual(e1,0.5*(0.0498935),5)

        self.assertAlmostEqual(f0[0],-0.148911,5)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],0.148911,5)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        # test energy shift
        rf.set_params(mode="shift")
        run(1)

        f0 = rf.forces[0].force
        f1 = rf.forces[1].force
        e0 = rf.forces[0].energy
        e1 = rf.forces[1].energy

        self.assertAlmostEqual(e0,0.5*(0.0323866),5)
        self.assertAlmostEqual(e1,0.5*(0.0323866),5)

        self.assertAlmostEqual(f0[0],-0.148911,5)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],0.148911,5)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    def tearDown(self):
        del self.nl
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
