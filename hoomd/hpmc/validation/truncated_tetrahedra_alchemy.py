import hoomd
from hoomd import hpmc
import numpy as np
import coxeter
from coxeter.shape_classes import ConvexPolyhedron
import BlockAverage

ttf = coxeter.shape_families.TruncatedTetrahedronFamily()

class TruncatedTetrahedron:
    def __init__(self, trunc=1.0):
        self.shape_params = [trunc]
        self.exec_conf = hoomd.context.current.system_definition.getParticleData().getExecConf()

    def __call__(self, trunc_attempt):
        shape = ttf(1 - trunc_attempt[0])
        return hpmc._hpmc.make_poly3d_verts((shape.vertices / (shape.volume**(1/3))).tolist(),
                                            0.0, False,
                                            self.exec_conf)

hoomd.context.initialize("--mode=cpu");

# See the following paper for the expected truncation value:
#   van Anders, G., Klotsa, D., Karas, A. S., Dodd, P. M., & Glotzer, S. C.
#       (2015). Digital Alchemy for Materials Design: Colloids and Beyond.
#       ACS Nano, 9(10), 9542–9553.https://doi.org/10.1021/acsnano.5b04181
mean_trunc_ref = 0.3736
sigma_trunc_ref = 0.0001

init_trunc = 0.3736
phi_final = 0.6
initial_shape = ConvexPolyhedron(ttf(1 - init_trunc).vertices / (ttf(1 - init_trunc).volume**(1/3)))
a = (8 * initial_shape.volume / phi_final)**(1.0 / 3.0) # lattice constant
dim = 3

lattice_vectors = [[a, 0, 0], [0, a, 0], [0, 0, a]]
basis_vectors = [[0.125, 0.125, 0.125],
                 [0.875, 0.875, 0.875],
                 [0.875, 0.375, 0.375],
                 [0.625, 0.125, 0.625],
                 [0.375, 0.875, 0.375],
                 [0.625, 0.625, 0.125],
                 [0.375, 0.375, 0.875],
                 [0.125, 0.625, 0.625]]
basis_vectors = np.asarray(basis_vectors).dot(np.asarray(lattice_vectors))

N = len(basis_vectors)
n = 3
uc = hoomd.lattice.unitcell(N=N,
                            a1=lattice_vectors[0],
                            a2=lattice_vectors[1],
                            a3=lattice_vectors[2],
                            position=basis_vectors.tolist(),
                            dimensions=dim)
system = hoomd.init.create_lattice(unitcell=uc, n=n)

mc = hoomd.hpmc.integrate.convex_polyhedron(1, d=0.1, a=0.1, move_ratio=0.5);
mc.shape_param.set('A', vertices=initial_shape.vertices)
mc_tuner = hpmc.util.tune(mc, tunables=['d', 'a'], target=0.2, gamma=0.5)     
snap = system.take_snapshot()

field = hpmc.field.lattice_field(mc=mc,
                                 position=[list(pos) for pos in snap.particles.position],
                                 orientation=[[1, 0, 0, 0]]*len(snap.particles.position),
                                 k=10.0, q=0.0)
box_L_points = [(0, system.box.Lx * 2),
                (5e3, system.box.Lx),
                (1e6, system.box.Lx)]
box_updater = hoomd.update.box_resize(L=hoomd.variant.linear_interp(box_L_points))

for i in range(10):
    hoomd.run(1e3)
    mc_tuner.update()

shape_gen_fn = TruncatedTetrahedron(trunc=init_trunc);
shape_updater = hoomd.hpmc.update.alchemy(
                mc=mc,
                move_ratio=1.0,
                period=1,
                seed=1)
shape_updater.python_shape_move(shape_gen_fn, {'A': [init_trunc]}, stepsize=0.1, param_ratio=0.5)
shape_tuner = shape_updater.get_tuner(target=0.5, gamma=0.5)

log = hoomd.analyze.log(filename=None, quantities=['shape_param-0'],period=10,overwrite=True);

trunc_list = [];
def accumulate_truncation(timestep):
    trunc_list.append(log.query('shape_param-0'))

field.set_params(5.0, 0.0)
for i in range(20):
    print("truncation: " + str(round(log.query('shape_param-0'), 5)))
    hoomd.run(1e3, callback=accumulate_truncation, callback_period=10)
    shape_tuner.update()
    mc_tuner.update()


block = BlockAverage.BlockAverage(trunc_list)
mean_trunc = np.mean(trunc_list)
i, sigma_trunc = block.get_error_estimate()
n, num_samples, err_est, err_err = block.get_hierarchical_errors()
mean_trunc = np.mean(trunc_list[-num_samples[-1]:])

# max error 0.5%
assert sigma_trunc / mean_trunc <= 0.005

# 0.99 confidence interval
ci = 2.576

# compare if 0 is within the confidence interval around the difference of the means
sigma_diff = (sigma_trunc**2 + sigma_trunc_ref**2)**(1.0 / 2.0);
assert abs(mean_trunc - mean_trunc_ref) <= ci * sigma_diff

