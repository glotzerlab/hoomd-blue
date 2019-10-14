from __future__ import division

from hoomd import *
from hoomd import hpmc

import numpy as np
import math

import unittest

context.initialize()

#seed_list=[123, 456]
seed_list = [123]
#phi_c_list=[0.01, 0.05, 0.10, 0.2, 0.3]
phi_c_list=[0.1]
#eta_p_r_list=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
eta_p_r_list=[0.4]

import itertools
params = []
params = list(itertools.product(seed_list, phi_c_list, eta_p_r_list))

context.current.device.cpp_msg.notice(1,"{} parameters\n".format(len(params)))

# choose a random state point
import random
p = int(option.get_user()[0])
(seed, phi_c, eta_p_r) = params[p % len(params)]

context.current.device.cpp_msg.notice(1,"parameter {} seed {} phi_c {:.3f} eta_p_r {:.3f}\n".format(p,seed, phi_c, eta_p_r))

# test the equation of state for the free volume fraction of hard spheres, when simultaneously
# applying depletion with a positive and negative coefficients so that both cancel

# reference data key = (phi_c, eta_p_r), value = (alpha, error)
# 128 spheres
alpha_ref=dict()
alpha_ref[(0.1,0.4)] = (0.375450,0.000130)

# number of spheres along one dimension
n = 5
N = n**3
d_sphere = 1.0
V_sphere = math.pi/6.0*math.pow(d_sphere,3.0)

# depletant-colloid size ratio
q=1.0

L_target= math.pow(N*V_sphere/phi_c,1.0/3.0)

class depletion_test(unittest.TestCase):
    def setUp(self):
        # initialize random configuration
        a = L_target/n
        self.system = init.create_lattice(unitcell=lattice.sc(a=a), n=n);

        self.system.particles.types.add('B')
        self.system.particles.types.add('C')

    def test_measure_etap(self):
        self.mc = hpmc.integrate.sphere(seed=seed)
        self.mc.set_params(d=0.1,a=0.1)
        self.mc.shape_param.set('A', diameter=d_sphere)
        self.mc.shape_param.set('B', diameter=d_sphere*q)
        self.mc.shape_param.set('C', diameter=d_sphere*q)

        self.mc_tune = hpmc.util.tune(self.mc, tunables=['d'],max_val=[d_sphere],gamma=1,target=0.2)
        for i in range(10):
            run(100, quiet=True)
            self.mc_tune.update()
        # warm up
        run(2000);


        # set depletant fugacity
        nR = eta_p_r/(math.pi/6.0*math.pow(d_sphere*q,3.0))
        self.mc.set_fugacity('B',nR)

        # set negative fugacity to same amount to cancel
        self.mc.set_fugacity('C',-nR)

        free_volume = hpmc.compute.free_volume(mc=self.mc, seed=seed, nsample=10000, test_type='B')
        log=analyze.log(filename=None, quantities=['hpmc_overlap_count','volume','hpmc_free_volume'], overwrite=True,period=1000)

        alpha_measure = []
        def log_callback(timestep):
            v = log.query('hpmc_free_volume')/log.query('volume')
            alpha_measure.append(v)
            self.assertEqual(log.query('hpmc_overlap_count'),0)

#            if context.current.device.comm.rank == 0:
#                print('alpha =', v);

        run(4e5,callback=log_callback,callback_period=100)

        import BlockAverage
        block = BlockAverage.BlockAverage(alpha_measure)
        alpha_avg = np.mean(np.array(alpha_measure))
        i, alpha_err = block.get_error_estimate()

        if context.current.device.comm.rank == 0:
            print(i)
            (n, num, err, err_err) = block.get_hierarchical_errors()

            print('Hierarchical error analysis:')
            for (i, num_samples, e, ee) in zip(n, num, err, err_err):
                print('{0} {1} {2} {3}'.format(i,num_samples,e,ee))

        if context.current.device.comm.rank == 0:
            print('avg: {:.6f} +- {:.6f}'.format(alpha_avg, alpha_err))
            print('tgt: {:.6f} +- {:.6f}'.format(alpha_ref[(phi_c,eta_p_r)][0], alpha_ref[(phi_c,eta_p_r)][1]))

        # max error 0.5%
        self.assertLessEqual(alpha_err/alpha_avg,0.005)

        # confidence interval, 0.95 quantile of the normal distribution
        ci = 1.96

        # check against reference value within reference error + measurement error
        self.assertLessEqual(math.fabs(alpha_avg-alpha_ref[(phi_c,eta_p_r)][0]),ci*(alpha_ref[(phi_c,eta_p_r)][1]+alpha_err))

    def tearDown(self):
        del self.system
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
