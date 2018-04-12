from hoomd import *
from hoomd import hpmc

context.initialize()

import numpy as np
import math

import unittest
import BlockAverage

# identify the state point by a user parameter
p = int(option.get_user()[0])

# l/sigma
l_list = [1.0,0.6,0.3]

# normalized pressure for the chosen state points
P_star_list = [15.0,20.0,20.0]

# reference densities, from an initial MC NPT run (I didn't find any literature results)
rho_star_ref_list = [0.86902,1.00270,1.02583]

# just for reference, values for perfectly hard dumbbell in Tildesey and Street 1980
# and Vega, Paras and Monson 1992 (Table I)
rho_star_hard = [0.881,1.021,1.039]

n_params = len(l_list)

# are we testing update.cluster?
use_clusters = p//n_params

# relative error on the reference values = 1%
rho_star_rel_err = 0.005

P_star = P_star_list[p % n_params]
len_cyl = l_list[p % n_params]
rho_star_ref = rho_star_ref_list[p % n_params]

# linear lattice dimension
n = 10

sigma = 1
V_intersection = 1./12.*math.pi*(2*sigma+len_cyl)*(sigma-len_cyl)*(sigma-len_cyl)
V_dumbbell = 2*math.pi/6*sigma**3.0 - V_intersection
d_eff = (V_dumbbell*6/(math.pi))**(1./3.)

rho_star_ini = 0.1
a = (d_eff**3.0/rho_star_ini)**(1./3.)

class npt_wca_dimer_eos(unittest.TestCase):
    def test_statepoint(self):
        uc = lattice.unitcell(N = 1,
                            a1 = [a,0,0],
                            a2 = [0,a,0],
                            a3 = [0,0,a],
                            dimensions = 3,
                            position = [[0,0,0]],
                            type_name = ['A'])

        system = init.create_lattice(unitcell=uc,n=n)

        N = len(system.particles)

        seed = 1234
        mc = hpmc.integrate.sphere_union(d=0.1,a=0.1,seed=seed)

        mc.shape_param.set('A',diameters=[sigma]*2,centers=[(0,0,-len_cyl/2),(0,0,len_cyl/2)],overlap=[0]*2,colors=['ff5984ff']*2)

        rcut_wca = sigma*2**(1./6.)
        rcut = len_cyl + rcut_wca
        eps = 1.0
        dumbbell = """float rsq = dot(r_ij, r_ij);
                  float rcut = {:.15f};
                  float len_cyl = {};
                  float rcut_wca = {:.15f};
                  float rcut_wca_sq = rcut_wca*rcut_wca;
                  float sigma = {};
                  float eps = {};
                  float pair_eng = 0;
                  if (rsq <= rcut*rcut)
                    {{
                    for (unsigned int k = 0; k < 2; ++k)
                        {{
                        vec3<float> r_k;
                        if (k==0)
                            r_k = rotate(q_i, vec3<float>(0,0,-0.5*len_cyl));
                        else
                            r_k = rotate(q_i, vec3<float>(0,0,0.5*len_cyl));

                        for (unsigned int l = 0; l < 2; ++l)
                            {{
                            vec3<float> r_l = r_ij;
                            if (l==0)
                                r_l += rotate(q_j, vec3<float>(0,0,-0.5*len_cyl));
                            else
                                r_l += rotate(q_j, vec3<float>(0,0,0.5*len_cyl));

                            vec3<float> r_lk = r_l - r_k;
                            float rlksq = dot(r_lk,r_lk);
                            if (rlksq <= rcut_wca_sq)
                                {{
                                float r2inv = sigma*sigma/rlksq;
                                float r6inv = r2inv * r2inv * r2inv;
                                pair_eng += r6inv * 4*eps * (r6inv - 1) + eps;
                                }}
                            }}
                        }}
                    }}
                return pair_eng;
                 """.format(rcut,len_cyl,rcut_wca,sigma,eps);

        from hoomd import jit
        jit.patch.user(mc,r_cut=rcut, code=dumbbell)
        boxmc = hpmc.update.boxmc(mc,betaP=P_star/d_eff**3.0, seed=seed+1)
        boxmc.ln_volume(delta=0.001,weight=1)

        mc_tune = hpmc.util.tune(mc, tunables=['d','a'],max_val=[4,0.5],gamma=1,target=0.3)
        npt_tune = hpmc.util.tune_npt(boxmc, tunables = ['dlnV'], target=0.3,gamma=1)

        log = analyze.log(filename=None, quantities=['hpmc_overlap_count','volume'],period=100,overwrite=True)

        rho_val = []
        def accumulate_rho(timestep):
            rho = N/log.query('volume')
            rho_val.append(rho)
            if (timestep % 1000 == 0): context.msg.notice(1,'rho_star = {:.5f}\n'.format(rho*d_eff**3.0))

        for i in range(10):
            run(1000)
            mc_tune.update()
            npt_tune.update()

        if use_clusters:
            hpmc.update.clusters(mc,period=1,seed=seed+2)

        run(1e4,callback=accumulate_rho, callback_period=100)

        block = BlockAverage.BlockAverage(rho_val)
        rho_avg = np.mean(rho_val)
        i, rho_err = block.get_error_estimate()

        context.msg.notice(1,'P_star = {:.3f} rho_star = {:.5f}+-{:.5f} (tgt: {:.5f})\n'.format(P_star,rho_avg*d_eff**3.0,rho_err*d_eff**3.0,rho_star_ref))

        # max error 0.5 %
        self.assertLessEqual(rho_err/rho_avg,0.005)

        # confidence interval, 0.95 quantile of the normal distribution
        ci = 1.96

        # compare if error is within confidence interval
        self.assertLessEqual(math.fabs(rho_avg*d_eff**3.0-rho_star_ref),ci*(rho_star_ref*rho_star_rel_err+rho_err*d_eff**3.0))

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
