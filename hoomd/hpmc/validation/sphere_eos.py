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
params = list(P_list)

context.msg.notice(1,"{} parameters\n".format(len(params)))

p = int(option.get_user()[0])
P = params[p]

class sphereEOS_test(unittest.TestCase):
    n = 7
    def setUp(self):
        context.initialize()
        n = self.n
        a = (math.pi / (6*phi_p_ref[P]))**(1.0/3.0);

        self.system = init.create_lattice(unitcell=lattice.sc(a=a), n=n);

        self.mc = hpmc.integrate.sphere(seed=p)

        self.mc.shape_param.set('A',diameter=1.0)
        self.mc.set_params(d=0.1,a=0.5)

        mc_tune = hpmc.util.tune(self.mc, tunables=['d','a'],max_val=[4,0.5],gamma=1,target=0.3)

        self.log = analyze.log(filename=None, quantities = ['hpmc_overlap_count','volume','phi_p', 'hpmc_d','hpmc_a','time'], overwrite=True, period=100)
        self.log.register_callback('phi_p', lambda timestep: len(self.system.particles)*V/self.system.box.get_volume())

        tunables = []
        boxmc = hpmc.update.boxmc(self.mc,betaP=P,seed=123)
        boxmc.ln_volume(delta=0.001,weight=1)
        tunables = ['dlnV']

        npt_tune = hpmc.util.tune_npt(boxmc, tunables = tunables, target = 0.2, gamma=1)

        for i in range(10):
            run(1000, quiet=True)

            d = self.mc.get_d();
            translate_acceptance = self.mc.get_translate_acceptance();
            util.quiet_status()
            v = boxmc.ln_volume()['delta']
            util.unquiet_status()
            volume_acceptance = boxmc.get_ln_volume_acceptance();
            if comm.get_rank() == 0:
                print('d: {:3.2f} accept: {:3.2f} / v: {:3.2f} accept: {:3.2f}'.format(d,translate_acceptance,v,volume_acceptance));

            mc_tune.update()
            npt_tune.update()

    def test_measure_phi_p(self):
        phi_p_measure = []
        def log_callback(timestep):
            v = self.log.query('phi_p')
            phi_p_measure.append(v)
            if comm.get_rank() == 0:
                print('phi_p =', v);

        # equilibrate
        run(1e4)

        # sample
        run(16e4,callback=log_callback, callback_period=50)

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
            print('avg: {:.6f} +- {:.6f}'.format(phi_p_avg, phi_p_err))
            print('tgt: {:.6f} +- {:.6f}'.format(phi_p_ref[P], rel_err_cs))

        # check against reference value within reference error + measurement error
        self.assertLessEqual(math.fabs(phi_p_avg-phi_p_ref[P]),ci*(phi_p_ref[P]*rel_err_cs+phi_p_err))

class sphereEOS_test_noncubic(sphereEOS_test):
    n = [7, 8, 6]

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
