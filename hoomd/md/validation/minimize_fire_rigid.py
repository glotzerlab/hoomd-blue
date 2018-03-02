from hoomd import *
from hoomd import md

import math

import numpy as np

import unittest

context.initialize()

class minimize_rigid_test(unittest.TestCase):
    n = 7
    def setUp(self):
        context.initialize()

        snapshot = data.make_snapshot(N=2,particle_types=['A','B'],box=data.boxdim(L=25))
        if comm.get_rank() == 0:
            snapshot.particles.typeid[0] = 0
            snapshot.particles.typeid[1] = 1
            snapshot.particles.position[0] = (0,0,0)
            snapshot.particles.position[1] = (0,0,1)
            snapshot.particles.orientation[0]= (1,0,0,0)

            # rotate minimally away from the parallel configuration
            angle_deg = .01
            snapshot.particles.orientation[1]= (math.cos(0.5*angle_deg*math.pi/180),0,0,math.sin(0.5*angle_deg*math.pi/180))
            snapshot.particles.moment_inertia[0] = (0.5,1,1)
            snapshot.particles.moment_inertia[1] = (0.5,1,1)
        self.s = init.read_snapshot(snapshot)

        self.nl = md.nlist.cell()

        # two dumbbells
        self.s.particles.types.add('sphere')
        self.rigid = md.constrain.rigid()
        self.rigid.set_param('A', positions=[(-.5,0,0),(.5,0,0)], types=['sphere']*2)
        self.rigid.set_param('B', positions=[(-.5,0,0),(.5,0,0)], types=['sphere']*2)
        self.rigid.create_bodies()

        self.nl = md.nlist.cell()
        self.lj = md.pair.lj(nlist=self.nl,r_cut=2.5)
        self.lj.pair_coeff.set(['A','B','sphere'],['A','B'], epsilon=0, sigma=0, r_cut=False)
        self.lj.pair_coeff.set('sphere','sphere',epsilon=1,sigma=1)


    def test_minimize(self):
        # only update particle 1 (keep 0 fixed)
        typeB = group.type(name='typeB',type='B')
        fire = md.integrate.mode_minimize_fire(dt=0.005,wtol=1e-10,ftol=1e-10)
        nve = md.integrate.nve(group=typeB)
        max_conv = 10000
        log = analyze.log(quantities=['potential_energy'],period=100,filename=None)
        run(1)
        old_energy = log.query('potential_energy')
        while not(fire.has_converged()) and get_step() < max_conv:
            run(100)
        self.assertTrue(fire.has_converged())

        # check that dumbbell 1 is now forming a 'cross' with dumbbell 0
        self.assertAlmostEqual(abs(self.s.particles[1].orientation[0]),1/math.sqrt(2),6)
        self.assertTrue(abs(self.s.particles[1].orientation[1])<0.001)
        self.assertTrue(abs(self.s.particles[1].orientation[2])<0.001)
        self.assertAlmostEqual(abs(self.s.particles[1].orientation[3]),1/math.sqrt(2),6)
        self.assertTrue(log.query('potential_energy') < old_energy)

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
