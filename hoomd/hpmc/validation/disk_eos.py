from hoomd import *
from hoomd import hpmc
import math
import numpy as np
import unittest
import BlockAverage

context.initialize()

phi_p_ref = 0.698;
rel_err_cs = 0.0001; # guesstimate
P_ref = 9.1709;
P_ref_err = 0.0002; # actual

n = 256;
N = n**2
a = math.sqrt(math.pi / (4*phi_p_ref));

class diskEOS_test(unittest.TestCase):
    def setUp(self):
        self.system = init.create_lattice(unitcell=lattice.sq(a=a), n=n);

        self.mc = hpmc.integrate.sphere(d = 0.2, seed=1)
        self.mc.shape_param.set('A',diameter=1.0)
        self.boxmc = hpmc.update.boxmc(self.mc,betaP=P_ref,seed=123)
        self.boxmc.volume(delta=0.42,weight=1)

        self.log = analyze.log(filename=None, quantities = ['hpmc_overlap_count','volume','phi_p', 'hpmc_d','hpmc_a','time'], overwrite=True, period=100)
        self.log.register_callback('phi_p', lambda timestep: len(self.system.particles)/self.system.box.get_volume() * math.pi / 4.0)

        # warm up
        run(1e3);

    def test_measure_phi_p(self):
        phi_p_measure = []
        def log_callback(timestep):
            v = self.log.query('phi_p');
            phi_p_measure.append(v)
            if comm.get_rank() == 0:
                print('phi_p =', v);

        run(10e3,callback=log_callback, callback_period=50)

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
        self.assertLessEqual(math.fabs(phi_p_avg-phi_p_ref),ci*(phi_p_ref*rel_err_cs+phi_p_err))

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
