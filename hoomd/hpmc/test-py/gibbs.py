from __future__ import division
from hoomd import *
from hoomd import hpmc

import unittest

import math

# this script needs to be run on two ranks

# initialize with one rank per partitions
context.initialize(args="--nrank=1")

class gibbs_ensemble_test(unittest.TestCase):
    def setUp(self):
        p = comm.get_partition()
        phi=0.2
        a = (1/6*math.pi / phi)**(1/3)

        unitcell=lattice.sc(a=a, type_name='A')
        self.system = init.create_lattice(unitcell=unitcell, n=5)

        self.system.particles.types.add('B')
        self.mc = hpmc.integrate.sphere(seed=123+p,implicit=True,depletant_mode='overlap_regions')
        self.mc.set_params(d=0.1)

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

    def test_spheres(self):
        # within two-phase region of hard spheres phase diagram
        q=0.8
        etap=0.7
        ntrial = 20
        p = comm.get_partition()

        self.mc.set_params(ntrial=ntrial)
        nR = etap/(math.pi/6.0*math.pow(q,3.0))
        self.mc.set_params(nR=nR,depletant_type='B')

        self.mc.shape_param.set('A', diameter=1.0)
        self.mc.shape_param.set('B', diameter=q)

        # needs to be run with 2 partitions
        muvt=hpmc.update.muvt(mc=self.mc,seed=456,ngibbs=2,transfer_types=['A'])

        muvt.set_params(dV=0.01)
        muvt.set_params(move_ratio=.01)

        run(100)

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
