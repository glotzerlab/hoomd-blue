# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os
import numpy
import math

def almost_equal(u, v, e=0.001):
    return math.fabs(u-v)/math.fabs(u) <= e and math.fabs(u-v) / math.fabs(v) <= e;

# unit tests for md.integrate.brownian
class integrate_brownian_script_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05

        context.current.sorter.set_params(grid=8)

    # tests basic creation of the integration method
    def test(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        bd = md.integrate.brownian(all, kT=1.2, seed=52);
        run(5);
        bd.disable();
        bd = md.integrate.brownian(all, kT=1.2, seed=1, dscale=1.0);
        run(5);
        bd.disable();
        bd = md.integrate.brownian(all, kT=1.2, seed=1, dscale=1.0, noiseless_t=True);
        run(5);
        bd.disable();
        bd = md.integrate.brownian(all, kT=1.2, seed=1, dscale=1.0, noiseless_r=True);
        run(5);
        bd.disable();
        bd = md.integrate.brownian(all, kT=1.2, seed=1, dscale=1.0, noiseless_t=True, noiseless_r=True);
        run(5);
        bd.disable();

    # test set_params
    def test_set_params(self):
        all = group.all();
        bd = md.integrate.brownian(all, kT=1.2, seed=1);
        bd.set_params(kT=1.3);

    # test set_gamma
    def test_set_gamma(self):
        all = group.all();
        bd = md.integrate.brownian(all, kT=1.2, seed=1);
        bd.set_gamma('A', 0.5);
        bd.set_gamma('B', 1.0);

        # test set_gamma
    def test_set_gamma_r(self):
        all = group.all();
        bd = md.integrate.brownian(all, kT=1.2, seed=1);
        bd.set_gamma_r('A', 0.5);
        bd.set_gamma_r('B', (1.0,2.0,3.0));

    def tearDown(self):
        context.initialize();


# validate brownian diffusion
class integrate_brownian_diffusion (unittest.TestCase):
    def setUp(self):
        print
        snap = data.make_snapshot(N=1000, box=data.boxdim(L=100000), particle_types=['A'])
        # this defaults all particles to position 0, which is what we want for this test
        self.s = init.read_snapshot(snap)

        context.current.sorter.set_params(grid=8)

    def test_gamma(self):
        # Setup an ideal gas with a gamma and T and validate the MSD
        T=1.8
        gamma=3241;
        dt=0.01;
        steps=5000;

        md.integrate.mode_standard(dt=dt);
        bd = md.integrate.brownian(group.all(), kT=T, seed=1, dscale=False);
        bd.set_gamma('A', gamma);

        run(steps);

        snap = self.s.take_snapshot();
        if comm.get_rank() == 0:
            msd = numpy.mean(snap.particles.position[:,0] * snap.particles.position[:,0] +
                             snap.particles.position[:,1] * snap.particles.position[:,1] +
                             snap.particles.position[:,2] * snap.particles.position[:,2])

            D = msd / (6*dt*steps);

            # check for a very crude overlap - we are not doing much averaging here to keep the test short
            self.assert_(almost_equal(D, T/gamma, 0.1))

    def test_noiseless_t(self):
        # Setup an ideal gas with a gamma and T and validate the MSD
        T=1.8
        gamma=3241;
        dt=0.01;
        steps=5000;

        md.integrate.mode_standard(dt=dt);
        bd = md.integrate.brownian(group.all(), kT=T, seed=1, dscale=False, noiseless_t=True);
        bd.set_gamma('A', gamma);

        run(steps);

        snap = self.s.take_snapshot();
        if comm.get_rank() == 0:
            msd = numpy.mean(snap.particles.position[:,0] * snap.particles.position[:,0] +
                             snap.particles.position[:,1] * snap.particles.position[:,1] +
                             snap.particles.position[:,2] * snap.particles.position[:,2])

            D = msd / (6*dt*steps);

            # check for a very crude overlap - we are not doing much averaging here to keep the test short
            self.assert_(math.fabs(D) < 0.1)

    def test_gamma(self):
        # Setup an ideal gas with a gamma and T and validate the MSD
        T=1.8
        gamma=3241;
        dt=0.01;
        steps=5000;

        md.integrate.mode_standard(dt=dt);
        bd = md.integrate.brownian(group.all(), kT=T, seed=1, dscale=False, noiseless_t=True);
        bd.set_gamma('A', gamma);

        run(steps);

        snap = self.s.take_snapshot();
        if comm.get_rank() == 0:
            msd = numpy.mean(snap.particles.position[:,0] * snap.particles.position[:,0] +
                             snap.particles.position[:,1] * snap.particles.position[:,1] +
                             snap.particles.position[:,2] * snap.particles.position[:,2])

            D = msd / (6*dt*steps);

            # check for a very crude overlap - we are not doing much averaging here to keep the test short
            self.assert_(math.fabs(D) < 0.1)

    def test_dscale(self):
        # Setup an ideal gas with a gamma and T and validate the MSD
        T=1.8
        gamma=3591;
        dt=0.01;
        steps=5000;

        md.integrate.mode_standard(dt=dt);
        bd = md.integrate.brownian(group.all(), kT=T, seed=1, dscale=gamma);

        run(steps);

        snap = self.s.take_snapshot();
        if comm.get_rank() == 0:
            msd = numpy.mean(snap.particles.position[:,0] * snap.particles.position[:,0] +
                             snap.particles.position[:,1] * snap.particles.position[:,1] +
                             snap.particles.position[:,2] * snap.particles.position[:,2])

            D = msd / (6*dt*steps);

            # check for a very crude overlap - we are not doing much averaging here to keep the test short
            self.assert_(almost_equal(D, T/gamma, 0.1))

    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
