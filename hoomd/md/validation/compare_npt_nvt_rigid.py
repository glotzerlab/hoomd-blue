from hoomd import *
from hoomd import md

import numpy as np
import math

import BlockAverage

import unittest

context.initialize()

p = int(option.get_user()[0])
rho_list = [0.1,0.2,0.25]
rho = rho_list[p]
n = 10
a = (1/rho)**(1./3.)

nsteps = 1e5

class npt_rigid_validation(unittest.TestCase):
    def setUp(self):
        self.system = init.create_lattice(unitcell=lattice.sc(a=a),n=n)
        N = len(self.system.particles)

        # generate a system of N=8 AB diblocks
        nl = md.nlist.cell()

        for p in self.system.particles:
            p.moment_inertia = (.5,.5,1)

        # create rigid spherocylinders out of two particles (not including the central particle)
        len_cyl =0.5

        # create constituent particle types
        self.system.particles.types.add('const')

        md.integrate.mode_standard(dt=0.005)

        wca = md.pair.lj(r_cut=False, nlist = nl)

        # central particles
        sigma = 1.0
        wca.pair_coeff.set('A', self.system.particles.types, epsilon=0, sigma=0, r_cut=False)

        # constituent particle coefficients (WCA)
        wca.pair_coeff.set('const','const', epsilon=1.0, sigma=sigma, r_cut=sigma*2**(1./6))
        wca.set_params(mode="shift")

        rigid = md.constrain.rigid()
        rigid.set_param('A', types=['const','const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)])
        rigid.create_bodies()

        #rigid.disable()
        self.center = group.rigid_center()

    def test_compare_nvt_npt(self):
        # thermalize
        langevin = md.integrate.langevin(group=self.center,kT=1.0,seed=123)
        langevin.set_gamma('A',2.0)
        run(50)
        langevin.disable()

        # store system data for later re-initialization
        snap = self.system.take_snapshot()

        # run system in NVT
        nvt = md.integrate.nvt(group=self.center,kT=1.0,tau=1.0)
        log = analyze.log(filename=None,quantities=['volume','pressure'],period=10,overwrite=True)

        Pval = []
        def accumulate_P(timestep):
            Pval.append(log.query('pressure'))

        run(nsteps,callback=accumulate_P, callback_period=100)

        block = BlockAverage.BlockAverage(Pval)
        P_avg = np.mean(np.array(Pval))
        i, P_err = block.get_error_estimate()

        context.msg.notice(1,'rho={:.3f} P = {:.5f}+-{:.5f}\n'.format(rho,P_avg,P_err))
        nvt.disable()

        # re-initialize
        self.system.restore_snapshot(snap)

        # set up NPT at measured pressure from NVT
        npt = md.integrate.npt(group=self.center,P=P_avg,tauP=0.5,kT=1.0,tau=1.0)

        rho_val = []
        N = len(self.center)
        def accumulate_rho(timestep):
            rho_val.append(N/log.query('volume'))

        run(nsteps,callback=accumulate_rho, callback_period=100)

        block = BlockAverage.BlockAverage(rho_val)
        rho_avg = np.mean(np.array(rho_val))
        i, rho_err = block.get_error_estimate()

        context.msg.notice(1,'P={:.5f} rho_avg = {:.5f}+-{:.5f}\n'.format(P_avg,rho_avg,rho_err))

        # max error 0.5 %
        self.assertLessEqual(rho_err/rho_avg,0.005)

        # confidence interval, 0.95 quantile of the normal distribution
        ci = 1.96

        # compare density in NVT vs average density in NPT
        self.assertLessEqual(math.fabs(rho_avg-rho),ci*(rho+rho_err))

    def tearDown(self):
        del self.center
        del self.system
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
