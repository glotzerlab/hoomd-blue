# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import hoomd
from hoomd import md
from hoomd import mpcd
hoomd.context.initialize()
import unittest
import numpy as np

# unit tests for slit pore bounce back geometry
class integrate_slit_pore_tests(unittest.TestCase):
    """ Unit tests for slit pore integrator.

    Most of the physics is already tested in hoomd.mpcd, so
    these unit tests are focused on the python API.

    """
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()

        # set the decomposition in z for mpi builds
        if hoomd.comm.get_num_ranks() > 1:
            hoomd.comm.decomposition(nz=2)

        # default testing configuration
        snap = hoomd.data.make_snapshot(N=2, box=hoomd.data.boxdim(L=10.))
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:] = [[4.95,-4.95,3.85],[0.,0.,-3.8]]
            snap.particles.velocity[:] = [[1.,-1.,1.],[-1.,-1.,-1.]]
            snap.particles.mass[:] = [1.,2.]
        self.s = hoomd.init.read_snapshot(snap)
        self.group = hoomd.group.all()

        md.integrate.mode_standard(dt=0.1)

    # test creation can happen (with all parameters set)
    def test_create(self):
        mpcd.integrate.slit_pore(group=self.group, H=4., L=2., boundary="no_slip")

    # test for setting parameters
    def test_set_params(self):
        slit_pore = mpcd.integrate.slit_pore(group=self.group, H=4., L=2.)
        self.assertAlmostEqual(slit_pore.H, 4.)
        self.assertAlmostEqual(slit_pore.L, 2.)
        self.assertEqual(slit_pore.boundary, "no_slip")
        self.assertAlmostEqual(slit_pore.cpp_method.geometry.getH(), 4.)
        self.assertAlmostEqual(slit_pore.cpp_method.geometry.getL(), 2.)
        self.assertEqual(slit_pore.cpp_method.geometry.getBoundaryCondition(), hoomd.mpcd._mpcd.boundary.no_slip)

        # change H and also ensure other parameters stay the same
        slit_pore.set_params(H=2.)
        self.assertAlmostEqual(slit_pore.H, 2.)
        self.assertAlmostEqual(slit_pore.L, 2.)
        self.assertEqual(slit_pore.boundary, "no_slip")
        self.assertAlmostEqual(slit_pore.cpp_method.geometry.getH(), 2.)
        self.assertAlmostEqual(slit_pore.cpp_method.geometry.getL(), 2.)
        self.assertEqual(slit_pore.cpp_method.geometry.getBoundaryCondition(), hoomd.mpcd._mpcd.boundary.no_slip)

        # change L
        slit_pore.set_params(L=3.)
        self.assertAlmostEqual(slit_pore.L, 3.)
        self.assertAlmostEqual(slit_pore.cpp_method.geometry.getL(), 3.)

        # change BCs
        slit_pore.set_params(boundary="slip")
        self.assertEqual(slit_pore.boundary, "slip")
        self.assertEqual(slit_pore.cpp_method.geometry.getBoundaryCondition(), hoomd.mpcd._mpcd.boundary.slip)

    # test for invalid boundary conditions being set
    def test_bad_boundary(self):
        slit_pore = mpcd.integrate.slit_pore(group=self.group, H=4., L=2.)
        slit_pore.set_params(boundary="no_slip")
        slit_pore.set_params(boundary="slip")

        with self.assertRaises(ValueError):
            slit_pore.set_params(boundary="invalid")

    def tearDown(self):
        del self.s, self.group

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
