from __future__ import division

from hoomd import *
from hoomd import hpmc

import numpy as np
import math

import unittest

context.initialize()

#seed_list=[123, 456]
seed_list = [123]
phi_c_list=[0.15]
eta_p_r_list=[0.35]

import itertools
params = []
params = list(itertools.product(seed_list, phi_c_list, eta_p_r_list))

context.current.device.cpp_msg.notice(1,"{} parameters\n".format(len(params)))

# choose a random state point
import random
p = int(option.get_user()[0])
(seed, phi_c, eta_p_r) = params[p % len(params)]

# are we using update.cluster?
use_clusters = p//len(params)

context.current.device.cpp_msg.notice(1,"parameter {} seed {} phi_c {:.3f} eta_p_r {:.3f}\n".format(p,seed, phi_c, eta_p_r))
# test the equation of state of spheres with penetrable depletant disks
# the reference values have been generated in HPMC, as we are not aware of published EOS on 2d penetrable disks

# reference data key = (phi_c, eta_p_r) value = (eta_p, error)
# 125 disks
eta_p_ref=dict()
eta_p_ref[(0.15,0.35)] = (0.186032,0.000033)

# number of spheres
n = 6
N = n**2
d_sphere = 1.0
A_sphere = math.pi/4.0*math.pow(d_sphere,2.0)

# depletant-colloid size ratio
q=1.0

L_target= math.pow(N*A_sphere/phi_c,1.0/2.0)

class implicit_test (unittest.TestCase):
    def setUp(self):
        # initialize random configuration
        a = L_target/n
        self.system = init.create_lattice(unitcell=lattice.sq(a=a), n=n);

        self.system.particles.types.add('B')

    def test_measure_etap(self):
        self.mc = hpmc.integrate.sphere(seed=seed)
        self.mc.set_params(d=0.1,a=0.1)
        self.mc.shape_param.set('A', diameter=d_sphere)
        self.mc.shape_param.set('B', diameter=d_sphere*q)

        self.mc_tune = hpmc.util.tune(self.mc, tunables=['d'],max_val=[d_sphere],gamma=1,target=0.2)
        for i in range(10):
            run(100, quiet=True)
            self.mc_tune.update()
        # warm up
        run(2000);

        # set depletant fugacity
        nR = eta_p_r/(math.pi/6.0*math.pow(d_sphere*q,2.0))
        self.mc.set_fugacity('B',nR)

        free_volume = hpmc.compute.free_volume(mc=self.mc, seed=seed, nsample=10000, test_type='B')
        log=analyze.log(filename=None, quantities=['hpmc_overlap_count','volume','hpmc_free_volume','hpmc_fugacity_B'], overwrite=True,period=1000)

        eta_p_measure = []
        def log_callback(timestep):
            v = math.pi/6.0*log.query('hpmc_free_volume')/log.query('volume')*log.query('hpmc_fugacity_B')
            eta_p_measure.append(v)
            self.assertEqual(log.query('hpmc_overlap_count'),0)

            if context.current.device.comm.rank == 0:
               print('eta_p =', v);

        if use_clusters:
            hpmc.update.clusters(self.mc,period=1,seed=seed+1)

        run(4e5,callback=log_callback,callback_period=100)

        import BlockAverage
        block = BlockAverage.BlockAverage(eta_p_measure)
        eta_p_avg = np.mean(np.array(eta_p_measure))
        i, eta_p_err = block.get_error_estimate()

        if context.current.device.comm.rank == 0:
            print(i)
            (n, num, err, err_err) = block.get_hierarchical_errors()

            print('Hierarchical error analysis:')
            for (i, num_samples, e, ee) in zip(n, num, err, err_err):
                print('{0} {1} {2} {3}'.format(i,num_samples,e,ee))

        if context.current.device.comm.rank == 0:
            print('avg: {:.6f} +- {:.6f}'.format(eta_p_avg, eta_p_err))
            print('tgt: {:.6f} +- {:.6f}'.format(eta_p_ref[(phi_c,eta_p_r)][0], eta_p_ref[(phi_c,eta_p_r)][1]))

        # max error 0.5%
        self.assertLessEqual(eta_p_err/eta_p_avg,0.005)

        # confidence interval, 0.95 quantile of the normal distribution
        ci = 1.96

        # check against reference value within reference error + measurement error
        self.assertLessEqual(math.fabs(eta_p_avg-eta_p_ref[(phi_c,eta_p_r)][0]),ci*(eta_p_ref[(phi_c,eta_p_r)][1]+eta_p_err))
        del self.mc

    def tearDown(self):
        del self.system
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
