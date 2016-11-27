#! /usr/bin/env hoomd
from hoomd import *
from hoomd import hpmc

import numpy as np
import math

import unittest

context.initialize()

seed_list=[123, 456]
phi_c_list=[0.01, 0.05, 0.10, 0.2, 0.3]
eta_p_r_list=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
ntrial_list=[0,5,10,50]

import itertools
params = []
params = list(itertools.product(seed_list, phi_c_list, eta_p_r_list, ntrial_list))

context.msg.notice(1,"{} parameters\n".format(len(params)))

# choose a random state point
import random
p = int(option.get_user()[0])
(seed, phi_c, eta_p_r, ntrial) = params[p]

context.msg.notice(1,"parameter {} seed {} phi_c {:.3f} eta_p_r {:.3f} ntrial {}\n".format(p,seed, phi_c, eta_p_r, ntrial))
# test the equation of state of spheres with penetrable depletant spheres
# see M. Dijkstra et al. Phys. Rev. E 73, p. 41404, 2006, Fig. 2 and
# J. Glaser et al., JCP 143 18, p. 184110, 2015.

# reference data key = (phi_c, eta_p_r) value = (eta_p, error)
# 128 spheres
eta_p_ref = dict()
#eta_p_ref[(0.01,0.)] = (0.,nan)
eta_p_ref[(0.01,0.2)] = (0.184429,3.30672e-6)
eta_p_ref[(0.01,0.4)] = (0.369163,8.87389e-6)
eta_p_ref[(0.01,0.6)] = (0.55419,0.0000356438)
eta_p_ref[(0.01,0.8)] = (0.740008,0.0000518881)
eta_p_ref[(0.01,1.)] = (0.926917,0.0000800465)
eta_p_ref[(0.01,1.2)] = (1.12308,0.00101186)
eta_p_ref[(0.01,1.4)] = (1.33082,0.000658831)
eta_p_ref[(0.01,1.6)] = (1.53151,0.000701636)
eta_p_ref[(0.01,1.8)] = (1.73043,0.000990081)
eta_p_ref[(0.01,2.)] = (1.92507,0.000640646)
#eta_p_ref[(0.05,0.)] = (0.,nan)
eta_p_ref[(0.05,0.2)] = (0.130129,0.000014551)
eta_p_ref[(0.05,0.4)] = (0.263853,0.0000637632)
eta_p_ref[(0.05,0.6)] = (0.402658,0.0000834948)
eta_p_ref[(0.05,0.8)] = (0.550195,0.000125953)
eta_p_ref[(0.05,1.)] = (0.733105,0.000403362)
eta_p_ref[(0.05,1.2)] = (0.937355,0.000134075)
eta_p_ref[(0.05,1.4)] = (1.12721,0.0000717065)
eta_p_ref[(0.05,1.6)] = (1.31048,0.000288656)
eta_p_ref[(0.05,1.8)] = (1.48942,0.00128758)
eta_p_ref[(0.05,2.)] = (1.66023,0.00360969)
#eta_p_ref[(0.1,0.)] = (0.,nan)
eta_p_ref[(0.1,0.2)] = (0.0779529,0.0000260589)
eta_p_ref[(0.1,0.4)] = (0.162446,0.0000596288)
eta_p_ref[(0.1,0.6)] = (0.256333,0.0000902474)
eta_p_ref[(0.1,0.8)] = (0.365615,0.0000481711)
eta_p_ref[(0.1,1.)] = (0.526111,0.000248716)
eta_p_ref[(0.1,1.2)] = (0.705206,0.000198212)
eta_p_ref[(0.1,1.4)] = (0.868799,0.000139591)
eta_p_ref[(0.1,1.6)] = (1.03247,0.000195703)
eta_p_ref[(0.1,1.8)] = (1.19063,0.000482943)
eta_p_ref[(0.1,2.)] = (1.34874,0.000210042)
#eta_p_ref[(0.2,0.)] = (0.,nan)
eta_p_ref[(0.2,0.2)] = (0.0182153,7.27051e-6)
eta_p_ref[(0.2,0.4)] = (0.0398498,0.0000194976)
eta_p_ref[(0.2,0.6)] = (0.0661762,0.0000448559)
eta_p_ref[(0.2,0.8)] = (0.100065,0.0000563893)
eta_p_ref[(0.2,1.)] = (0.152769,0.000160969)
eta_p_ref[(0.2,1.2)] = (0.268763,0.000208779)
eta_p_ref[(0.2,1.4)] = (0.385803,0.000217095)
eta_p_ref[(0.2,1.6)] = (0.51938,0.000115689)
eta_p_ref[(0.2,1.8)] = (0.635392,0.000185744)
eta_p_ref[(0.2,2.)] = (0.750144,0.0000670894)
#eta_p_ref[(0.3,0.)] = (0.,nan)
eta_p_ref[(0.3,0.2)] = (0.00158686,2.11827e-6)
eta_p_ref[(0.3,0.4)] = (0.00337131,3.96414e-6)
eta_p_ref[(0.3,0.6)] = (0.00536869,7.0547e-6)
eta_p_ref[(0.3,0.8)] = (0.00771994,0.0000210634)
eta_p_ref[(0.3,1.)] = (0.0104042,0.0000319104)
eta_p_ref[(0.3,1.2)] = (0.0137031,0.0000429746)
eta_p_ref[(0.3,1.4)] = (0.0180348,0.0000560252)
eta_p_ref[(0.3,1.6)] = (0.0239093,0.0000778583)
eta_p_ref[(0.3,1.8)] = (0.0663776,0.000145459)
eta_p_ref[(0.3,2.)] = (0.153222,0.000270273)

# number of spheres
N = 128
d_sphere = 1.0
V_sphere = math.pi/6.0*math.pow(d_sphere,3.0)

# depletant-colloid size ratio
q=1.0

# initial volume fraction
phi_c_ini = 0.01

L_ini= math.pow(N*V_sphere/phi_c_ini,1.0/3.0)
L_target= math.pow(N*V_sphere/phi_c,1.0/3.0)


class implicit_test (unittest.TestCase):
    def setUp(self):
        # initialize random configuration
        from hoomd import deprecated
        self.system = deprecated.init.create_random(N=N,box=data.boxdim(L=L_ini), min_dist=1.0)

        self.mc = hpmc.integrate.sphere(seed=seed,implicit=True)
        self.mc.set_params(d=0.1,a=0.1)

        self.system.particles.types.add('B')
        self.mc.set_params(depletant_type='B')

        self.mc.shape_param.set('A', diameter=d_sphere)
        self.mc.shape_param.set('B', diameter=d_sphere*q)

        # number of test depletant to throw for measuring free volume
        nsample = 10000

        # no depletants during compression
        self.mc.set_params(nR=0)

        self.mc_tune = hpmc.util.tune(self.mc, tunables=['d','a'],max_val=[4*d_sphere,0.5],gamma=1,target=0.3)

        # run for a bit to randomize
        run(1000);

        # shrink the box to the target size (if needed)
        scale = 0.99;
        L = L_ini;
        while L_target < L:
            # shrink the box
            L = max(L*scale, L_target);

            update.box_resize(Lx=L, Ly=L, Lz=L, period=None);
            overlaps = self.mc.count_overlaps();
            context.msg.notice(1,"phi =%f: overlaps = %d " % (((N*V_sphere) / (L*L*L)), overlaps));

            # run until all overlaps are removed
            while overlaps > 0:
                self.mc_tune.update()
                run(100, quiet=True);
                overlaps = self.mc.count_overlaps();
                context.msg.notice(1,"%d\n" % overlaps)

            context.msg.notice(1,"\n");

        # set the target L (this expands to the final L if it is larger than the start)
        update.box_resize(Lx=L_target, Ly=L_target, Lz=L_target, period=None);

    def test_measure_etap(self):
        # set depletant fugacity
        nR = eta_p_r/(math.pi/6.0*math.pow(d_sphere*q,3.0))
        self.mc.set_params(nR=nR, ntrial=ntrial)

        free_volume = hpmc.compute.free_volume(mc=self.mc, seed=seed, nsample=10000, test_type='B')
        log=analyze.log(filename=None, quantities=['hpmc_overlap_count','volume','hpmc_free_volume','hpmc_fugacity'], overwrite=True,period=1000)

        eta_p_measure = []
        def log_callback(timestep):
            eta_p_measure.append(math.pi/6.0*log.query('hpmc_free_volume')/log.query('volume')*log.query('hpmc_fugacity'))

        run(5e5,callback=log_callback,callback_period=1000)

        import BlockAverage
        block = BlockAverage.BlockAverage(eta_p_measure)
        eta_p_avg = np.mean(np.array(eta_p_measure))
        _, eta_p_err = block.get_error_estimate()

        # max error 0.5%
        self.assertLessEqual(eta_p_err/eta_p_avg,0.005)

        # check against reference value within 2*error
        self.assertLessEqual(math.fabs(eta_p_avg-eta_p_ref[(phi_c,eta_p_r)][0]),2*eta_p_err)

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
