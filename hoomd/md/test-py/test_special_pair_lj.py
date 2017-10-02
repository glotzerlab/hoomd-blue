# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# tests md.special_pair.lj
class special_pair_lj_tests (unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=3,
                                  box=data.boxdim(L=100),
                                  particle_types = ['A'],
                                  pair_types = ['pairtype_1','pairtype_2'],
                                  angle_types = [],
                                  dihedral_types = [],
                                  improper_types = [])

        if comm.get_rank() == 0:
            snap.pairs.resize(2);
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (0,0,1.5)
            snap.particles.position[2] = (-0.5,0,0)
            snap.pairs.group[0,:] = [0,1]
            snap.pairs.group[1,:] = [1,2]
            snap.pairs.typeid[0] = 0
            snap.pairs.typeid[1] = 1

        self.s = init.read_snapshot(snap)

    # test to see that se can create a md.force.constant
    def test_create(self):
        md.special_pair.lj();

    # test setting coefficients
    def test_set_coeff(self):
        lj = md.special_pair.lj();
        lj.pair_coeff.set('pairtype_1', sigma=1.0, epsilon=1.0, r_cut=3.0)
        lj.pair_coeff.set('pairtype_2', sigma=1.0, epsilon=1.0, r_cut=3.0)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100);

    # test coefficient not set checking
    def test_set_coeff_fail(self):
        lj = md.special_pair.lj();
        lj.pair_coeff.set('pairtype_1', sigma=1.0, epsilon=1.0, r_cut=3.0)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);

    # test remove particle fails
    def test_special_pair_fail(self):
        lj = md.special_pair.lj();
        lj.pair_coeff.set('pairtype_1', sigma=1.0, epsilon=1.0, r_cut=3.0)
        lj.pair_coeff.set('pairtype_2', sigma=1.0, epsilon=1.0, r_cut=3.0)
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

    # check that cutoff is correctly enforced
    def test_special_pair_lj_cutoff(self):
        lj = md.special_pair.lj();
        lj.pair_coeff.set('pairtype_1', sigma=1.0, epsilon=1.0, r_cut=1.49)
        lj.pair_coeff.set('pairtype_2', sigma=1.5, epsilon=2.5, r_cut=1.7)
        all = group.all();
        md.integrate.mode_standard(dt=0);
        md.integrate.nve(all);
        run(1)

        # Should be 0.0 due to being outside cutoff
        self.assertAlmostEqual(lj.forces[0].energy, 0.0,3)

        # Should only have contribution from [1,2] as [0,1] is outside cutoff
        self.assertAlmostEqual(lj.forces[1].energy, -0.5*1.9756,3)
        self.assertAlmostEqual(lj.forces[2].energy, -0.5*1.9756,3)

    # check the value of the pair potential
    def test_special_pair_lj_value(self):
        lj = md.special_pair.lj();
        lj.pair_coeff.set('pairtype_1', sigma=1.0, epsilon=1.0, r_cut=3.0)
        lj.pair_coeff.set('pairtype_2', sigma=1.5, epsilon=2.5, r_cut=3.0)
        all = group.all();
        md.integrate.mode_standard(dt=0);
        md.integrate.nve(all);
        run(1)

        self.assertAlmostEqual(lj.forces[0].energy, -0.5*.320337,3)
        self.assertAlmostEqual(lj.forces[1].energy, -0.5*.320337-0.5*1.9756,3)
        self.assertAlmostEqual(lj.forces[2].energy, -0.5*1.9756,3)

    def tearDown(self):
        del self.s
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
