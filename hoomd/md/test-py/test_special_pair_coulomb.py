# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# tests md.special_pair.coulomb
class special_pair_coulomb_tests (unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=4,
                                  box=data.boxdim(L=100),
                                  particle_types = ['A'],
                                  pair_types = ['pairtype_1','pairtype_2','pairtype_3'],
                                  angle_types = [],
                                  dihedral_types = [],
                                  improper_types = [])

        if comm.get_rank() == 0:
            snap.pairs.resize(3);
            snap.particles.position[0] = (0, 0, 0)
            snap.particles.position[1] = (0, 0, 3.5)
            snap.particles.position[2] = (-0.5 , 0, 0)
            snap.particles.position[3] = (0 , 2.5, 0)
            snap.particles.charge[0] = 1.0
            snap.particles.charge[1] = -3.0
            snap.particles.charge[2] = 2.5
            snap.particles.charge[3] = -7.0
            snap.pairs.group[0] = [0, 1]
            snap.pairs.group[1] = [2, 3]
            snap.pairs.group[2] = [1, 2]
            snap.pairs.typeid[0] = 0
            snap.pairs.typeid[1] = 1
            snap.pairs.typeid[2] = 2

        self.s = init.read_snapshot(snap)

    # test to see that se can create a md.force.constant
    def test_create(self):
        md.special_pair.coulomb();

    # test setting coefficients
    def test_set_coeff(self):
        coulomb = md.special_pair.coulomb();
        coulomb.pair_coeff.set('pairtype_1', alpha=0.5, r_cut=3.0)
        coulomb.pair_coeff.set('pairtype_2', alpha=0.5, r_cut=5.0)
        coulomb.pair_coeff.set('pairtype_3', alpha=0.5, r_cut=5.0)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100);

    # test coefficient not set checking
    def test_set_coeff_fail(self):
        coulomb = md.special_pair.coulomb();
        coulomb.pair_coeff.set('pairtype_1', alpha=0.5, r_cut=5.0)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);

    # test remove particle fails
    def test_special_pair_fail(self):
        coulomb = md.special_pair.coulomb();
        coulomb.pair_coeff.set('pairtype_1', alpha=0.5, r_cut=3.0)
        coulomb.pair_coeff.set('pairtype_2', alpha=0.5, r_cut=3.0)
        coulomb.pair_coeff.set('pairtype_3', alpha=0.5, r_cut=3.0)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        # remove a particle
        del(self.s.particles[0])
        if comm.get_num_ranks() == 1:
            self.assertRaises(RuntimeError, run, 100);
        else:
            # in MPI simulations, we cannot check for an assertion during a simulation
            # the program will terminate with MPI_Abort
            #self.assertRaises(RuntimeError, run, 100);
            pass

    # check the value of the pair potential
    def test_special_pair_coulomb_value(self):
        coulomb = md.special_pair.coulomb();
        coulomb.pair_coeff.set('pairtype_1', alpha=0.5, r_cut=3.0)
        coulomb.pair_coeff.set('pairtype_2', alpha=1.0, r_cut=5.0)
        coulomb.pair_coeff.set('pairtype_3', alpha=0.5, r_cut=5.0)
        all = group.all();
        md.integrate.mode_standard(dt=0);
        md.integrate.nve(all);
        run(1)

        # Should be zero due to distance > r_cut
        self.assertAlmostEqual(coulomb.forces[0].energy, 0.0, 3)
        # Should be non-zero
        self.assertAlmostEqual(coulomb.forces[1].energy, 0.5 * -1.060660, 3)
        self.assertAlmostEqual(coulomb.forces[2].energy, 0.5 * (-1.060660 + -6.864067), 3)
        self.assertAlmostEqual(coulomb.forces[3].energy, 0.5 * -6.864067, 3)

    def tearDown(self):
        del self.s
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
