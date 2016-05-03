# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os

import math

#---
# tests md.bond.harmonic
class constrain_distance_tests (unittest.TestCase):
    def setUp(self):
        print
        snap = data.make_snapshot(N=4,box=data.boxdim(L=25),particle_types=['A'])
        self.system = init.read_snapshot(snap)

        self.system.particles[0].position = (0,0,0)
        self.system.particles[1].position = (1.5,0,0)
        self.system.particles[2].position = (0,-1.5,0)
        self.system.particles[3].position = (-9,0,0)

        self.system.particles[0].mass = 0.7;
        self.system.particles[1].mass = 0.95;
        self.system.particles[2].mass = 0.12;
        self.system.particles[3].mass = 1.04;

        self.system.constraints.add(0,1,1.5)
        self.system.constraints.add(0,2,1.5)
        self.system.constraints.add(1,2,math.sqrt(1.5**2.0+1.5**2.0))

        self.nl = md.nlist.cell()

    # test to see that se can create a md.force.constant
    def test_create(self):
        md.constrain.distance();

    # test setting coefficients
    def test_constraint(self):
        constraint = md.constrain.distance()

        md.integrate.mode_standard(dt=0.005)

        md.integrate.nve(group=group.all())

        lj = md.pair.lj(r_cut=2.5, nlist = self.nl)
        lj.pair_coeff.set('A','A',epsilon=1.0,sigma=1.0)
        lj.set_params(mode="shift")

        log = analyze.log(quantities = ['potential_energy', 'kinetic_energy'], period = 10, filename=None);

        run(100)

        K0 = log.query('kinetic_energy');
        U0 = log.query('potential_energy');
        E0 = K0 + U0

        # check that distances are maintained
        box = self.system.box
        pos0 = self.system.particles[0].position
        pos1 = self.system.particles[1].position
        pos2 = self.system.particles[2].position

        pos01 = box.min_image((pos0[0]-pos1[0], pos0[1]-pos1[1], pos0[2]-pos1[2]))
        pos02 = box.min_image((pos0[0]-pos2[0], pos0[1]-pos2[1], pos0[2]-pos2[2]))
        pos12 = box.min_image((pos2[0]-pos1[0], pos2[1]-pos1[1], pos2[2]-pos1[2]))

        self.assertAlmostEqual(pos01[0]*pos01[0]+pos01[1]*pos01[1]+pos01[2]*pos01[2],1.5*1.5,4)
        self.assertAlmostEqual(pos02[0]*pos02[0]+pos02[1]*pos02[1]+pos02[2]*pos02[2],1.5*1.5,4)
        self.assertAlmostEqual(pos12[0]*pos12[0]+pos12[1]*pos12[1]+pos12[2]*pos12[2],2.0*1.5*1.5,4)

        # test energy conservation
        run(1000)
        K1 = log.query('kinetic_energy');
        U1 = log.query('potential_energy');
        E1 = K1 + U1

        self.assertAlmostEqual(E0,E1,3)

    # test coefficient not set checking
    def test_set_params(self):
        constraint = md.constrain.distance()
        constraint.set_params(rel_tol=0.01)

    # test remove particle fails
    def test_constraint_fail(self):
        constraint =  md.constrain.distance();
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        # remove a particle
        del(self.system.particles[0])
        if comm.get_num_ranks() == 1:
            self.assertRaises(RuntimeError, run, 100);
        else:
            # in MPI simulations, we cannot check for an assertion during a simulation
            # the program will terminate with MPI_Abort
            #self.assertRaises(RuntimeError, run, 100);
            pass

    # test exclusions in neighbor list
    def test_exclusions(self):
        distance = md.constrain.distance();
        lj = md.pair.lj(r_cut=3.0, nlist = self.nl)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0);
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100)

        self.assertEqual(self.nl.cpp_nlist.getNumExclusions(2), 3)
        self.assertEqual(self.nl.cpp_nlist.getNumExclusions(1), 0)

    # test exceeding the maximum contraint length in MPI
    def test_mpi(self):
        # add a long constraint
        self.system.constraints.add(1,3,10.5)
        distance = md.constrain.distance();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(group.all());
        if comm.get_num_ranks() > 1:
            # unfortunately, we cannot catch an MPI_Abort
            #self.assertRaises(RuntimeError, run, 1)
            pass
        else:
            run(1)

    def tearDown(self):
        del self.system
        del self.nl
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
