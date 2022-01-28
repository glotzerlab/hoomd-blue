# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd import hpmc
import numpy as np
import coxeter
from coxeter.shapes import ConvexPolyhedron
import BlockAverage
import freud

ttf = coxeter.families.TruncatedTetrahedronFamily()


class TruncatedTetrahedron:

    def __init__(self, trunc=1.0):
        self.shape_params = [trunc]

    def __call__(self, trunc_attempt):
        shape = ttf.get_shape(1 - trunc_attempt[0])
        self.shape_params.append(trunc_attempt[0])
        args = {
            'vertices': (shape.vertices / (shape.volume**(1 / 3))).tolist(),
            'sweep_radius': 0.0,
            'ignore_statistics': 0
        }
        return hoomd.hpmc._hpmc.PolyhedronVertices(args)


# See the following paper for the expected truncation value:
#   van Anders, G., Klotsa, D., Karas, A. S., Dodd, P. M., & Glotzer, S. C.
#       (2015). Digital Alchemy for Materials Design: Colloids and Beyond.
#       ACS Nano, 9(10), 9542–9553.https://doi.org/10.1021/acsnano.5b04181
mean_trunc_ref = 0.3736
sigma_trunc_ref = 0.0001

init_trunc = 0.3736
phi_final = 0.6
initial_shape = ConvexPolyhedron(
    ttf.get_shape(1 - init_trunc).vertices /
    (ttf.get_shape(1 - init_trunc).volume**(1 / 3)))
a = (8 * initial_shape.volume / phi_final)**(1.0 / 3.0)  # lattice constant
dim = 3

lattice_vectors = [[a, 0, 0], [0, a, 0], [0, 0, a]]
basis_vectors = [[0.125, 0.125, 0.125], [0.875, 0.875, 0.875],
                 [0.875, 0.375, 0.375], [0.625, 0.125, 0.625],
                 [0.375, 0.875, 0.375], [0.625, 0.625, 0.125],
                 [0.375, 0.375, 0.875], [0.125, 0.625, 0.625]]
basis_vectors = np.asarray(basis_vectors).dot(10 * np.asarray(lattice_vectors))
uc = freud.data.UnitCell(
    freud.box.Box.from_matrix(10 * np.asarray(lattice_vectors)), basis_vectors)
initial_box, initial_pos = uc.generate_system(num_replicas=3)

basis_vectors = np.asarray(basis_vectors).dot(np.asarray(lattice_vectors))
uc = freud.data.UnitCell(freud.box.Box.from_matrix(lattice_vectors),
                         basis_vectors)
final_box, final_pos = uc.generate_system(num_replicas=3)

N = len(basis_vectors)
n = 3

cpu = hoomd.device.CPU()

s = hoomd.Snapshot()

if s.exists:
    s.configuration.box = [
        initial_box.Lx, initial_box.Ly, initial_box.Lz, initial_box.xy,
        initial_box.xz, initial_box.yz
    ]
    s.configuration.dimensions = dim

    s.particles.N = len(initial_pos)
    s.particles.types = ['A']

    s.particles.position[:] = initial_pos

sim = hoomd.Simulation(device=cpu)

sim.create_state_from_snapshot(s)

mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
mc.shape['A'] = {'vertices': initial_shape.vertices.tolist()}
tune = hoomd.hpmc.tune.MoveSize.scale_solver(
    moves=['a', 'd'], target=0.2, trigger=hoomd.trigger.Periodic(1000))
sim.operations.add(mc)
sim.operations.tuners.append(tune)
compress = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(1),
                                           seed=10,
                                           target_box=hoomd.Box(
                                               final_box.Lx,
                                               final_box.Ly,
                                               Lz=final_box.Lz))
sim.operations.add(compress)
sim.operations._schedule()
sim.run(1e3)
shape_gen_fn = TruncatedTetrahedron()
updater = hoomd.hpmc.update.Alchemy(mc=mc,
                                    move_ratio=1.0,
                                    seed=3832765,
                                    trigger=hoomd.trigger.Periodic(1),
                                    nselect=1)
updater.python_shape_move(shape_gen_fn, {'A': [init_trunc]},
                          stepsize=0.1,
                          param_ratio=0.5)
sim.operations.add(updater)
tuner = updater.get_tuner(hoomd.trigger.Periodic(1000), 0.5, gamma=0.5)
sim.operations.add(tuner)
log_file = open("truncations.txt", "w+")
logger = hoomd.logging.Logger(flags=['scalar'])
logger += updater
writer = hoomd.write.Table(hoomd.trigger.Periodic(1),
                           logger,
                           log_file,
                           max_header_len=1)
sim.operations.add(writer)
sim.operations._schedule()

# field = hpmc.field.lattice_field(mc=mc,
#                                  position=[list(pos) for pos in snap.particles.position],
#                                  orientation=[[1, 0, 0, 0]] * len(snap.particles.position),
#                                  k=10.0, q=0.0)

# field.set_params(5.0, 0.0)
for _ in range(20):
    sim.run(1e3)
    print(updater.shape_param)

truncations = np.hsplit(np.loadtxt("truncations.txt", skiprows=1), 3)[2]
block = BlockAverage.BlockAverage(truncations)
mean_trunc = np.mean(truncations)
i, sigma_trunc = block.get_error_estimate()
n, num_samples, err_est, err_err = block.get_hierarchical_errors()
mean_trunc = np.mean(truncations[-num_samples[-1]:])

# max error 0.5%
assert sigma_trunc / mean_trunc <= 0.005

# 0.99 confidence interval
ci = 2.576

# compare if 0 is within the confidence interval around the difference of the means
sigma_diff = (sigma_trunc**2 + sigma_trunc_ref**2)**(1.0 / 2.0)
assert abs(mean_trunc - mean_trunc_ref) <= ci * sigma_diff
