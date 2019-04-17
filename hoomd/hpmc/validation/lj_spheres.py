from hoomd import *
from hoomd import hpmc

context.initialize()

import numpy as np
import math

import unittest
import BlockAverage

# Reference potential energy (U/N/eps) from MC simulations
# https://mmlapps.nist.gov/srs/LJ_PURE/mc.htm
mean_Uref = -5.5121E+00;
sigma_Uref = 4.55E-04;

# Interaction cut-off
rcut = 3.0;

# LJ length scale
sigma = 1.0;

# Temperature (kT/eps)
Tstar = 8.50E-01;
eps   = 1.0 / Tstar;

# Reduced density: rhostar = (N / V) * sigma**3
rho_star = 7.76E-01;

# Diameter of particles
diameter = sigma;

# linear lattice dimension
n = 8;

# Particle volume
V_p = math.pi/6.*diameter**3.;

# lattice constant (sc)
d_eff = (V_p*6/math.pi)**(1./3.);
a = (d_eff**3.0/rho_star)**(1./3.);

class nvt_lj_sphere_energy(unittest.TestCase):

    def test_statepoint(self):

        system = init.create_lattice(unitcell=lattice.sc(a=a), n=n);

        use_clusters = int(option.get_user()[0])%2
        use_depletants = int(option.get_user()[0])//2
        depletant_mode = 'overlap_regions'

        N = len(system.particles);

        if use_depletants:
            mc = hpmc.integrate.sphere(d=0.3,seed=54871,implicit=True,depletant_mode=depletant_mode);
        else:
            mc = hpmc.integrate.sphere(d=0.3,seed=65412);

        mc.shape_param.set('A',diameter=0)

        if use_depletants:
            # set up a dummy depletant
            system.particles.types.add('B')
            mc.set_params(depletant_type='B',nR=0)
            mc.shape_param.set('B', diameter=0)

        lennard_jones = """
                        float rsq = dot(r_ij, r_ij);
                        float rcut  = {};
                        if (rsq <= rcut*rcut)
                           {{
                           float sigma = {};
                           float eps   = {};
                           float sigmasq = sigma*sigma;
                           float rsqinv = sigmasq / rsq;
                           float r6inv = rsqinv*rsqinv*rsqinv;
                           return 4.0f*eps*r6inv*(r6inv-1.0f);
                           }}
                        else
                           {{
                           return 0.0f;
                           }}
                      """.format(rcut,sigma,eps);

        from hoomd import jit

        jit.patch.user(mc,r_cut=rcut, code=lennard_jones);

        log = analyze.log(filename=None, quantities=['hpmc_overlap_count','hpmc_patch_energy'],period=100,overwrite=True);

        energy_val = [];
        def accumulate_energy(timestep):
            energy = log.query('hpmc_patch_energy') / float(N) / eps;
            # apply long range correction (used in reference data)
            energy += 8/9.0 * math.pi * rho_star * ((1/rcut)**9-3*(1/rcut)**3)
            energy_val.append(energy);
            if (timestep % 100 == 0): context.msg.notice(1,'energy = {:.5f}\n'.format(energy));

        mc_tune = hpmc.util.tune(mc, tunables=['d','a'],max_val=[4,0.5],gamma=0.5,target=0.4);

        for i in range(5):
            run(100,quiet=True);
            d = mc.get_d();
            translate_acceptance = mc.get_translate_acceptance();
            util.quiet_status();
            print('d: {:3.2f} accept: {:3.2f}'.format(d,translate_acceptance));
            mc_tune.update();

        # Equilibrate
        run(500);

        if use_clusters:
            clusters = hpmc.update.clusters(mc, seed=99685)
            mc.set_params(d=0, a=0); # test cluster moves alone

        # Sample
        run(1000,callback=accumulate_energy, callback_period=10)

        block = BlockAverage.BlockAverage(energy_val)
        mean_U = np.mean(energy_val)
        i, sigma_U = block.get_error_estimate()

        context.msg.notice(1,'rho_star = {:.3f}, U = {:.5f}+-{:.5f}\n'.format(rho_star,mean_U,sigma_U))

        # 0.99 confidence interval
        ci = 2.576

        # compare if 0 is within the confidence interval around the difference of the means
        sigma_diff = (sigma_U**2 + sigma_Uref**2)**(1/2.);
        print(mean_U, mean_Uref, sigma_U, sigma_Uref);
        self.assertLessEqual(math.fabs(mean_U - mean_Uref), ci*sigma_diff)

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
