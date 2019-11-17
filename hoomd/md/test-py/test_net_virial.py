# -*- coding: iso-8859-1 -*-

from __future__ import print_function
from __future__ import division

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os
import numpy

numpy.random.seed(1)

# unit tests for analyze.msd
class compute_net_virial_tests (unittest.TestCase):
    def setUp(self):
        self.L = 6.
        self.N = 1000
        self.snap = data.make_snapshot(N=self.N, particle_types=['A'], box=data.boxdim(L=self.L))

        if comm.get_rank() == 0:
            for i in range(self.N):
                self.snap.particles.position[i] = numpy.random.random( 3)
                #no kinetic contribution to the pressure wanted
                self.snap.particles.velocity[i] = numpy.zeros(3)

        init.read_snapshot(self.snap)
        context.current.sorter.set_params(grid=8)
        nl = md.nlist.cell()
        self.dpd = md.pair.dpd_conservative(r_cut=1.0,nlist=nl)
        self.dpd.pair_coeff.set('A','A',A=1)
        #no kinetic contribution to pressure wanted. Thus freeze the configuration
        md.integrate.mode_standard(dt=0.)
        md.integrate.nve(group=group.all())

    # API test: tests basic calling the function
    def test_api(self):
        run(1);
        dpd_virial = self.dpd.get_net_virial(group.all())

    # Unit test: Validate virial and pressure computation
    def test_virial_pressure(self):
        #log the pressure quantities
        qr = ["pressure_xx","pressure_xy","pressure_xz","pressure_yy","pressure_yz","pressure_zz"]
        log = analyze.log(None,qr,period=1)

        run(1);

        #access only the DPD virial
        dpd_virial = self.dpd.get_net_virial(group.all())

        volume = self.L**3
        for i in range(6):
            log_pressure = log.query(qr[i])
            numpy.testing.assert_allclose(log_pressure, dpd_virial[i]/volume)


    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
