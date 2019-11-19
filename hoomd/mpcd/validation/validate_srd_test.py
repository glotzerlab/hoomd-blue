# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import unittest
import numpy as np

import hoomd
from hoomd import md
from hoomd import mpcd

class mpcd_srd_validation(unittest.TestCase):
    def setUp(self):
        hoomd.context.initialize()

        # box size L and solvent density 5
        L = 20
        self.density = 5.

        # solute initially on simple cubic lattice
        self.solute = hoomd.init.create_lattice(hoomd.lattice.sc(a=1.0), L)
        snap = self.solute.take_snapshot(all=True)
        if hoomd.comm.get_rank() == 0:
            snap.particles.mass[:] = self.density
        self.solute.restore_snapshot(snap)

        # srd
        self.solvent = mpcd.init.make_random(N=int(self.density*L**3), kT=1.0, seed=42)
        mpcd.integrator(dt=0.1)
        self.srd = mpcd.collide.srd(seed=791, period=1, angle=130., kT=1.0)
        mpcd.stream.bulk(period=1)
        md.integrate.nve(hoomd.group.all())

    def test_solvent(self):
        """Test momentum conservation for SRD solvent."""

        # initial momentum of both should be zero
        slv = self.solvent.take_snapshot()
        slt = self.solute.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            slv_p0 = np.sum(slv.particles.velocity, axis=0)
            slt_p0 = self.density*np.sum(slt.particles.velocity, axis=0)
            np.testing.assert_allclose(slv_p0, [0,0,0], atol=1.e-6)
            np.testing.assert_allclose(slt_p0, [0,0,0], atol=1.e-6)

        hoomd.run(100)

        # both groups should still have zero momentum, since the solute is not coupled to solvent
        slv = self.solvent.take_snapshot()
        slt = self.solute.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            slv_p1 = np.sum(slv.particles.velocity, axis=0)
            slt_p1 = self.density*np.sum(slt.particles.velocity, axis=0)
            np.testing.assert_allclose(slv_p1, [0,0,0], atol=1.e-6)
            np.testing.assert_allclose(slt_p1, [0,0,0], atol=1.e-6)

    def test_embed(self):
        """Test momentum conservation for SRD solvent + embedded particles."""

        self.srd.embed(hoomd.group.all())

        # initial momentum of both should be zero
        slv = self.solvent.take_snapshot()
        slt = self.solute.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            slv_p0 = np.sum(slv.particles.velocity, axis=0)
            slt_p0 = self.density*np.sum(slt.particles.velocity, axis=0)
            np.testing.assert_allclose(slv_p0, [0,0,0], atol=1.e-6)
            np.testing.assert_allclose(slt_p0, [0,0,0], atol=1.e-6)

        hoomd.run(100)

        # each group should not have zero momentum, but total momentum should be zero
        slv = self.solvent.take_snapshot()
        slt = self.solute.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            slv_p1 = np.sum(slv.particles.velocity, axis=0)
            slt_p1 = self.density*np.sum(slt.particles.velocity, axis=0)
            self.assertFalse(np.allclose(slv_p1, [0,0,0]))
            self.assertFalse(np.allclose(slt_p1, [0,0,0]))
            np.testing.assert_allclose(slv_p1 + slt_p1, [0,0,0], atol=1.e-3)

    def tearDown(self):
        del self.solute, self.solvent, self.srd

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
