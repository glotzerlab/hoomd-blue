from hoomd import *
from hoomd import hpmc

import math

import numpy as np

import unittest

context.initialize()

V = math.pi/6
#P_list = [0.29054,0.91912,2.2768,5.29102,8.06553,9.98979]
#npt_list = [1,2,3,5,6,7]
P_list = [5.29102]
npt_list = [2]

# Packing fractions from Carnahan Starling EOS
phi_p_ref = {0.29054: 0.1, 0.91912: 0.2, 2.2768: 0.3, 5.29102: 0.4, 8.06553: 0.45, 9.98979: 0.475}
rel_err_cs = 0.0015 # see for example Guang-Wen Wu and Richard J. Sadus, doi:10.1002.aic10233

import itertools
params = list(itertools.product(P_list,npt_list))

context.msg.notice(1,"{} parameters\n".format(len(params)))

p = int(option.get_user()[0])
P = params[p][0]
do_lengths = bool(params[p][1] & 1)
do_volume = bool(params[p][1] & 2)
do_shear = bool(params[p][1] & 4)

N=256

class sphereEOS_test(unittest.TestCase):
    def setUp(self):

        from hoomd import deprecated
        phi_p_ini = 0.05
        self.system = deprecated.init.create_random(N=N,phi_p=phi_p_ini, min_dist=1.0)

        self.mc = hpmc.integrate.sphere(seed=p)

        self.mc.shape_param.set('A',diameter=1.0)
        self.mc.set_params(d=0.1,a=0.5)

        mc_tune = hpmc.util.tune(self.mc, tunables=['d','a'],max_val=[4,0.5],gamma=1,target=0.3)

        self.log = analyze.log(filename=None, quantities = ['hpmc_overlap_count','volume','phi_p', 'hpmc_d','hpmc_a','time'], overwrite=True, period=100)
        self.log.register_callback('phi_p', lambda timestep: len(self.system.particles)*V/self.system.box.get_volume())

        tunables = []
        boxmc = hpmc.update.boxmc(self.mc,betaP=P,seed=123)
        if do_lengths:
            boxmc.length(delta=(0.1,0.1,0.1),weight=1)
            tunables.append('dLx')
            tunables.append('dLy')
            tunables.append('dLz')
        if do_volume:
            boxmc.volume(delta=1,weight=1)
            tunables.append('dV')
        if do_shear:
            boxmc.shear(delta=(0.01,0.01,0.01),weight=1)
            tunables.append('dxy')
            tunables.append('dxz')
            tunables.append('dyz')

        npt_tune = hpmc.util.tune_npt(boxmc, tunables = tunables, target = 0.3, gamma=1)

        for i in range(20):
            run(1e3)
            mc_tune.update()
            npt_tune.update()

    def test_measure_phi_p(self):
        phi_p_measure = []
        def log_callback(timestep):
            phi_p_measure.append(self.log.query('phi_p'))

        run(1e4,callback=log_callback, callback_period=100)

        import BlockAverage
        block = BlockAverage.BlockAverage(phi_p_measure)
        phi_p_avg = np.mean(np.array(phi_p_measure))
        i, phi_p_err = block.get_error_estimate()

        if comm.get_rank() == 0:
            (n, num, err, err_err) = block.get_hierarchical_errors()

            print('Hierarchical error analysis:')
            for (i, num_samples, e, ee) in zip(n, num, err, err_err):
                print('{0} {1} {2} {3}'.format(i,num_samples,e,ee))

        # max error 0.5%
        self.assertLessEqual(phi_p_err/phi_p_avg,0.005)

        # confidence interval, 0.95 quantile of the normal distribution
        ci = 1.96

        if comm.get_rank() == 0:
            print('avg {:.6f} +- {:.6f}'.format(phi_p_avg, phi_p_err))

        # check against reference value within reference error + measurement error
        self.assertLessEqual(math.fabs(phi_p_avg-phi_p_ref[P]),ci*(phi_p_ref[P]*rel_err_cs+phi_p_err))

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
