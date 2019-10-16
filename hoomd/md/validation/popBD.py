import numpy
import hoomd
from hoomd import md
from hoomd import deprecated
from hoomd import *
import numpy as np
import random
import time


random.seed(time.time())
seed1 = random.randint(0,1000)
seed2 = random.randint(0,1000)
hoomd.context.initialize()

n_polymer = 40
r_true = 11.12
r_colloid = 11.38
gap = 4
center_from_origin = r_true + gap/2

k = 0.05
nK = 60
r0 = 60
delta_G = 10

r_cut = 50
r_buff = 15
# epsilon_cc =0
class PopBD:
    def setUp(self):
    snapshot = data.make_snapshot(N=2, box=data.boxdim(Lx=80, Ly=80, Lz=80),
    bond_types=['polymer'], particle_types=['colloid'])
        snapshot.particles.diameter[:] = [job.sp.r_colloid*2]*2
        snapshot.particles.position[0] = [-job.sp.center_from_origin, 0, 0]
        snapshot.particles.position[1] = [job.sp.center_from_origin, 0, 0]
        system = hoomd.init.read_snapshot(snapshot)

        # define potentials
        nl = md.nlist.tree(r_buff=job.sp.r_buff)
        slj = md.pair.slj(r_cut=2**(1/6)*job.sp.sigma_cc, nlist=nl)
        slj.pair_coeff.set('colloid', 'colloid', epsilon=job.sp.epsilon_cc, sigma=job.sp.sigma_cc)
        slj.set_params(mode="shift")

        fene = md.bond.fene()
        fene.bond_coeff.set(k=job.sp.k, r0=job.sp.r0, epsilon=0,
                            sigma=0, type='polymer')

        # define integrator
        integrator = md.integrate.mode_standard(dt=job.sp.dt)
        brownian = md.integrate.brownian(group=all, kT=1, seed=seed1, noiseless_r=True)
        brownian.set_gamma('colloid', job.sp.drag)
        brownian.set_gamma_r('colloid', job.sp.drag)

        popbd = md.update.dynamic_bond(group=all,
                                    nlist=nl,
                                    seed=seed2,
                                    integrator=integrator,
                                    period=1)


        popbd.set_params(r_cut=job.sp.r_cut,
                            r_true=job.sp.r_true,
                            bond_type='polymer',
                            delta_G=job.sp.delta_G,
                            n_polymer=job.sp.n_polymer,
                            nK=job.sp.nK)

         # equlibrate
            nl.reset_exclusions(exclusions=None)

            eql_vis = dump.gsd("eql.gsd", period=job.sp.vis_period, group=all, overwrite=True, dynamic=['topology'])
            hoomd.run(job.sp.eql_steps)
            eql_vis.disable()

            job.document['progress'] = 'equilibrated'

            # dump visualization and stress data
            dump.gsd("trajectory.gsd", period=job.sp.vis_period, group=all, overwrite=True, dynamic=['topology'])

            # run
            hoomd.run(job.sp.run_steps)

            f = gsd.pygsd.GSDFile(open(job.fn('trajectory.gsd'), 'rb'))
            traj = gsd.hoomd.HOOMDTrajectory(f)

            nbonds = []
            time = []
            print('counting bonds')
            for n, i in enumerate(range(f.nframes)):
                time.append(int(n*job.sp.vis_period))
                nbonds.append(traj[i].bonds.N)

            df = pd.DataFrame({'tstep':time, 'nbonds':nbonds})
            df.to_csv(job.fn('nbonds.txt'), index=None)