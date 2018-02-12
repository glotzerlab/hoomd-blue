from hoomd import *
from hoomd import md

import numpy as np
import math

import BlockAverage

import unittest

# Compute the WCA dimer equation of state

context.initialize()

# identify the state point by a user parameter
p = int(option.get_user()[0])

# rho d^3

# reference values from MC simulation in NPT ensemble (N=216 particles), using enthalpy-hpmc
rho_star_list = [0.86902,1.00270,1.02583]

# l/sigma
l_list = [1.0,0.6,0.3]

# normalized pressure for the chosen state points
# from the Monte Carlo dumbbells reported in Table I
P_star_ref_list = [15.0,20.0,20.0]

# relative error on the reference values = 0.5%
P_star_rel_err = 0.005

P_star_ref = P_star_ref_list[p]
rho_star = rho_star_list[p]
len_cyl = l_list[p]

# linear dimension of lattice
n = 10

sigma = 1
V_intersection = 1./12.*math.pi*(2*sigma+len_cyl)*(sigma-len_cyl)*(sigma-len_cyl)
V_dumbbell = 2*math.pi/6*sigma**3.0 - V_intersection
d_eff = (V_dumbbell*6/(math.pi))**(1./3.)
rho = rho_star/(d_eff**3.0)

class npt_rigid_validation(unittest.TestCase):
    def setUp(self):
        uc = lattice.unitcell(N = 1,
                            a1 = [1,0,0],
                            a2 = [0,1,0],
                            a3 = [0,0,1/rho],
                            dimensions = 3,
                            position = [[0,0,0]],
                            type_name = ['A'],
                            mass = [1],
                            moment_inertia = [[1,1,1]])

        self.system = init.create_lattice(unitcell=uc,n=n)
        nl = md.nlist.cell()

        # create rigid spherocylinders out of two particles (not including the central particle)

        # create constituent particle types
        self.system.particles.types.add('const')

        md.integrate.mode_standard(dt=0.005)

        wca = md.pair.lj(r_cut=False, nlist = nl)

        # central particles
        wca.pair_coeff.set('A', self.system.particles.types, epsilon=0, sigma=0, r_cut=False)

        # constituent particle coefficients (WCA)
        wca.pair_coeff.set('const','const', epsilon=1.0, sigma=sigma, r_cut=sigma*2**(1./6.))
        wca.set_params(mode="shift")

        rigid = md.constrain.rigid()
        rigid.set_param('A', types=['const','const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)])
        rigid.create_bodies()

        #rigid.disable()
        self.center = group.rigid_center()

    def test_virial_pressure(self):
        # thermalize
        langevin = md.integrate.langevin(group=self.center,kT=1.0,seed=123)
        langevin.set_gamma('A',2.0)
        run(5000)
        langevin.disable()

        # run system in NVT
        nvt = md.integrate.nvt(group=self.center,kT=1.0,tau=1.0)
        log = analyze.log(filename=None,quantities=['volume','pressure'],period=10,overwrite=True)

        Pval = []
        def accumulate_P(timestep):
            Pval.append(log.query('pressure'))

        run(5e5,callback=accumulate_P, callback_period=100)

        block = BlockAverage.BlockAverage(Pval)
        P_avg = np.mean(np.array(Pval))
        i, P_err = block.get_error_estimate()

        context.msg.notice(1,'rho_star={:.3f} P_star = {:.5f}+-{:.5f}\n'.format(rho_star,P_avg*d_eff**3.0,P_err*d_eff**3.0))

        # max error 0.5 %
        self.assertLessEqual(P_err/P_avg,0.005)

        # confidence interval, 0.95 quantile of the normal distribution
        ci = 1.96

        # compare if error is within confidence interval
        self.assertLessEqual(math.fabs(P_avg*d_eff**3.0-P_star_ref),ci*(P_star_ref*P_star_rel_err+P_err*d_eff**3.0))

    def tearDown(self):
        del self.center
        del self.system
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
