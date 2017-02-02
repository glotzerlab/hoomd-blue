# -*- coding: iso-8859-1 -*-
# Maintainer: bvansade 

import numpy as np
import hoomd
import hoomd.md
import unittest

class test_oneD(unittest.TestCase):
    def setUp(self):
        hoomd.context.initialize()
        self.system = hoomd.init.create_lattice(hoomd.lattice.fcc(a=4.0), n=[2,2,2])
        self.system.box = hoomd.data.boxdim(Lx=self.system.box.Lx*2, Ly=self.system.box.Ly*2, Lz=self.system.box.Lz*2)
        self.initial = self.system.take_snapshot(particles=True)
        lj = hoomd.md.pair.lj(r_cut=3.0, nlist=hoomd.md.nlist.cell())
        lj.pair_coeff.set('A','A', epsilon=1.0, sigma=1.0)
        hoomd.md.integrate.mode_standard(dt=5e-3)
        nvt = hoomd.md.integrate.nvt(group=hoomd.group.type(type='A'), kT=1.0, tau=1e-1)

    def test_y(self):
        constraint_vec = [0,1,0]
        oneD = hoomd.md.constrain.oneD(group=hoomd.group.type('A'), constraint_vector=constraint_vec)
        hoomd.md.force.constant(fx=1, fy=0, fz=0, group=hoomd.group.type('A'))
        hoomd.run(1e2)
        self.final = self.system.take_snapshot(particles=True)

        pos_diff = np.asarray(self.final.particles.position) - np.asarray(self.initial.particles.position)
        dots = np.dot(pos_diff,np.asarray(constraint_vec))/(np.linalg.norm(pos_diff,axis=1)*np.linalg.norm(np.asarray(constraint_vec)))
        tolerance = 1e-3
        assert(np.abs(np.abs(dots).sum() - self.final.particles.N)<tolerance)

    def tearDown(self):
        del(self.system)

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
