# -*- coding: iso-8859-1 -*-
# Maintainer: jglaser

from hoomd import *
from hoomd import md
import unittest
import os

context.initialize()

# md.pair.reaction_field
class pair_reaction_field_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        rf = md.pair.reaction_field(r_cut=3.0);
        rf.pair_coeff.set('A', 'A', epsilon=1.0, eps_rf = 1.0);
        rf.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        rf = md.pair.reaction_field(r_cut=3.0);
        rf.pair_coeff.set('A', 'A', eps_rf=1.0);
        self.assertRaises(RuntimeError, rf.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        rf = md.pair.reaction_field(r_cut=3.0);
        self.assertRaises(RuntimeError, rf.update_coeffs);

    # test set params
    def test_set_params(self):
        rf = md.pair.reaction_field(r_cut=3.0);
        rf.set_params(mode="no_shift");
        rf.set_params(mode="shift");
        rf.set_params(mode="xplor");
        self.assertRaises(RuntimeError, rf.set_params, mode="blah");

    # test nlist subscribe
    def test_nlist_global_subscribe(self):
        rf = md.pair.reaction_field(r_cut=2.5);
        rf.pair_coeff.set('A', 'A', epsilon=1.0, eps_rf=1.0)
        context.current.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.5, context.current.neighbor_list.r_cut.get_pair('A','A'));

        rf.pair_coeff.set('A', 'A', r_cut = 2.0)
        context.current.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.0, context.current.neighbor_list.r_cut.get_pair('A','A'));

    # test nlist subscribe
    def test_nlist_subscribe(self):
        nl = md.nlist.cell()
        rf = md.pair.reaction_field(r_cut=2.5, nlist=nl);
        self.assertEqual(context.current.neighbor_list, None)

        rf.pair_coeff.set('A', 'A', epsilon=1.0, eps_rf=1.0)
        nl.update_rcut();
        self.assertAlmostEqual(2.5, nl.r_cut.get_pair('A','A'));

        rf.pair_coeff.set('A', 'A', r_cut = 2.0)
        nl.update_rcut();
        self.assertAlmostEqual(2.0, nl.r_cut.get_pair('A','A'));

    def tearDown(self):
        context.initialize();

# test the validity of the pair potential
class test_pair_reaction_field_potential(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=2, box=data.boxdim(L=10),particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (1.5,0,0)
        init.read_snapshot(snap)

    # test the calculation of force and potential
    def test_potential(self):
        rf = md.pair.reaction_field(r_cut=2.0)

        # basic test case
        rf.pair_coeff.set('A','A', epsilon=2.0, eps_rf=3.0)
        rf.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        f0 = rf.forces[0].force
        f1 = rf.forces[1].force
        e0 = rf.forces[0].energy
        e1 = rf.forces[1].energy

        self.assertAlmostEqual(e0,0.5*1.49405,5)
        self.assertAlmostEqual(e1,0.5*1.49405,5)

        self.assertAlmostEqual(f0[0],-0.674603,5)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],0.674603,5)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        # test energy shift
        rf.set_params(mode="shift")
        run(1)

        f0 = rf.forces[0].force
        f1 = rf.forces[1].force
        e0 = rf.forces[0].energy
        e1 = rf.forces[1].energy

        self.assertAlmostEqual(e0,0.5*(1.49405-1.28571),5)
        self.assertAlmostEqual(e1,0.5*(1.49405-1.28571),5)

        self.assertAlmostEqual(f0[0],-0.674603,5)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],0.674603,5)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        # test infinite eps_rf
        rf.pair_coeff.set('A','A', epsilon=2.0, eps_rf=0)
        rf.set_params(mode="no_shift")

        run(1)

        f0 = rf.forces[0].force
        f1 = rf.forces[1].force
        e0 = rf.forces[0].energy
        e1 = rf.forces[1].energy

        self.assertAlmostEqual(e0,0.5*1.61458,5)
        self.assertAlmostEqual(e1,0.5*1.61458,5)

        self.assertAlmostEqual(f0[0],-0.513889,5)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],0.513889,5)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        rf.pair_coeff.set('A','A', epsilon=2.0, eps_rf=0)
        rf.set_params(mode="no_shift")

    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
