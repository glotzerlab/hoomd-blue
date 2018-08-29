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

context.msg.notice(1,"{} parameters\n".format(len(params)))

# choose a random state point
import random
p = int(option.get_user()[0])
(seed, phi_c, eta_p_r) = params[p % len(params)]

# are we using update.cluster?
use_clusters = p//len(params)

context.msg.notice(1,"parameter {} seed {} phi_c {:.3f} eta_p_r {:.3f}\n".format(p,seed, phi_c, eta_p_r))
# test the equation of state of spheres with penetrable depletant spheres
# see M. Dijkstra et al. Phys. Rev. E 73, p. 41404, 2006, Fig. 2 and
# J. Glaser et al., JCP 143 18, p. 184110, 2015.

# reference data key = (phi_c, eta_p_r) value = (eta_p, error)
# 128 spheres
eta_p_ref=dict()
#eta_p_ref[(0.01,0.)] = (0.,nan)
eta_p_ref[(0.01,0.2)] = (0.184438,7.79773e-6)
eta_p_ref[(0.01,0.4)] = (0.369113,0.000015426)
eta_p_ref[(0.01,0.6)] = (0.554202,0.0000226575)
eta_p_ref[(0.01,0.8)] = (0.73994,0.0000319383)
eta_p_ref[(0.01,1.)] = (0.926884,0.0000312035)
eta_p_ref[(0.01,1.2)] = (1.11685,0.0000215032)
eta_p_ref[(0.01,1.4)] = (1.32721,0.000331169)
eta_p_ref[(0.01,1.6)] = (1.52856,0.0000769524)
eta_p_ref[(0.01,1.8)] = (1.72593,0.000131199)
eta_p_ref[(0.01,2.)] = (1.92188,0.000436138)
#eta_p_ref[(0.05,0.)] = (0.,nan)
eta_p_ref[(0.05,0.2)] = (0.130102,0.000017169)
eta_p_ref[(0.05,0.4)] = (0.263677,0.0000296967)
eta_p_ref[(0.05,0.6)] = (0.402265,0.0000358007)
eta_p_ref[(0.05,0.8)] = (0.549098,0.0000542385)
eta_p_ref[(0.05,1.)] = (0.712581,0.000143215)
eta_p_ref[(0.05,1.2)] = (0.900993,0.000116858)
eta_p_ref[(0.05,1.4)] = (1.08466,0.0001577)
eta_p_ref[(0.05,1.6)] = (1.26389,0.000312563)
eta_p_ref[(0.05,1.8)] = (1.43957,0.000490628)
eta_p_ref[(0.05,2.)] = (1.61347,0.000118301)
#eta_p_ref[(0.1,0.)] = (0.,nan)
eta_p_ref[(0.1,0.2)] = (0.0777986,0.0000224789)
eta_p_ref[(0.1,0.4)] = (0.162055,0.0000391019)
eta_p_ref[(0.1,0.6)] = (0.25512,0.0000917089)
eta_p_ref[(0.1,0.8)] = (0.361985,0.000081159)
eta_p_ref[(0.1,1.)] = (0.491528,0.000211232)
eta_p_ref[(0.1,1.2)] = (0.644402,0.0000945081)
eta_p_ref[(0.1,1.4)] = (0.797721,0.000114195)
eta_p_ref[(0.1,1.6)] = (0.947405,0.000266665)
eta_p_ref[(0.1,1.8)] = (1.09756,0.000207732)
eta_p_ref[(0.1,2.)] = (1.24626,0.00085732)
#eta_p_ref[(0.2,0.)] = (0.,nan)
eta_p_ref[(0.2,0.2)] = (0.0180642,8.88676e-7)
eta_p_ref[(0.2,0.4)] = (0.0394307,0.0000491992)
eta_p_ref[(0.2,0.6)] = (0.0652104,0.0000840904)
eta_p_ref[(0.2,0.8)] = (0.0975177,0.0000992883)
eta_p_ref[(0.2,1.)] = (0.141602,0.000141207)
eta_p_ref[(0.2,1.2)] = (0.20416,0.000278241)
eta_p_ref[(0.2,1.4)] = (0.289024,0.000340248)
eta_p_ref[(0.2,1.6)] = (0.383491,0.000357631)
eta_p_ref[(0.2,1.8)] = (0.483246,0.000338302)
eta_p_ref[(0.2,2.)] = (0.594751,0.00061228)
#eta_p_ref[(0.3,0.)] = (0.,nan)
eta_p_ref[(0.3,0.2)] = (0.00154793,6.84185e-6)
eta_p_ref[(0.3,0.4)] = (0.00328478,0.0000103679)
eta_p_ref[(0.3,0.6)] = (0.00521468,0.0000212988)
eta_p_ref[(0.3,0.8)] = (0.00746148,7.85157e-6)
eta_p_ref[(0.3,1.)] = (0.0100912,3.00293e-6)
eta_p_ref[(0.3,1.2)] = (0.0131242,0.0000590406)
eta_p_ref[(0.3,1.4)] = (0.0169659,0.0000524466)
eta_p_ref[(0.3,1.6)] = (0.021623,0.0000828658)
eta_p_ref[(0.3,1.8)] = (0.0283405,0.000133873)
eta_p_ref[(0.3,2.)] = (0.0387704,0.000167702)

# number of spheres
n = 5
N = n**3
d_sphere = 1.0
V_sphere = math.pi/6.0*math.pow(d_sphere,3.0)

# depletant-colloid size ratio
q=1.0

L_target= math.pow(N*V_sphere/phi_c,1.0/3.0)

class implicit_test (unittest.TestCase):
    def setUp(self):
        # initialize random configuration
        a = L_target/n
        self.system = init.create_lattice(unitcell=lattice.sc(a=a), n=n);

        self.system.particles.types.add('B')

    def test_measure_etap_new(self):
        self.mc = hpmc.integrate.sphere(seed=seed,implicit=True, depletant_mode='overlap_regions')
        self.mc.set_params(d=0.1,a=0.1)
        self.mc.set_params(depletant_type='B')
        self.mc.shape_param.set('A', diameter=d_sphere)
        self.mc.shape_param.set('B', diameter=d_sphere*q)

        # no depletants during tuning
        self.mc.set_params(nR=0)

        self.mc_tune = hpmc.util.tune(self.mc, tunables=['d'],max_val=[d_sphere],gamma=1,target=0.2)
        for i in range(10):
            run(100, quiet=True)
            self.mc_tune.update()
        # warm up
        run(2000);

        # set depletant fugacity
        nR = eta_p_r/(math.pi/6.0*math.pow(d_sphere*q,3.0))
        self.mc.set_params(nR=nR)

        free_volume = hpmc.compute.free_volume(mc=self.mc, seed=seed, nsample=10000, test_type='B')
        log=analyze.log(filename=None, quantities=['hpmc_overlap_count','volume','hpmc_free_volume','hpmc_fugacity'], overwrite=True,period=1000)

        eta_p_measure = []
        def log_callback(timestep):
            v = math.pi/6.0*log.query('hpmc_free_volume')/log.query('volume')*log.query('hpmc_fugacity')
            eta_p_measure.append(v)
            self.assertEqual(log.query('hpmc_overlap_count'),0)

            # if comm.get_rank() == 0:
            #    print('eta_p =', v);

        if use_clusters:
            hpmc.update.clusters(self.mc,period=1,seed=seed+1)

        run(4e5,callback=log_callback,callback_period=100)

        import BlockAverage
        block = BlockAverage.BlockAverage(eta_p_measure)
        eta_p_avg = np.mean(np.array(eta_p_measure))
        i, eta_p_err = block.get_error_estimate()

        if comm.get_rank() == 0:
            print(i)
            (n, num, err, err_err) = block.get_hierarchical_errors()

            print('Hierarchical error analysis:')
            for (i, num_samples, e, ee) in zip(n, num, err, err_err):
                print('{0} {1} {2} {3}'.format(i,num_samples,e,ee))

        if comm.get_rank() == 0:
            print('avg: {:.6f} +- {:.6f}'.format(eta_p_avg, eta_p_err))
            print('tgt: {:.6f} +- {:.6f}'.format(eta_p_ref[(phi_c,eta_p_r)][0], eta_p_ref[(phi_c,eta_p_r)][1]))

        # max error 0.5%
        self.assertLessEqual(eta_p_err/eta_p_avg,0.005)

        # confidence interval, 0.95 quantile of the normal distribution
        ci = 1.96

        # check against reference value within reference error + measurement error
        self.assertLessEqual(math.fabs(eta_p_avg-eta_p_ref[(phi_c,eta_p_r)][0]),ci*(eta_p_ref[(phi_c,eta_p_r)][1]+eta_p_err))

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
