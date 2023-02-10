# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest

import math

import hoomd

# test the equation of state of spheres with penetrable depletant spheres
# see M. Dijkstra et al. Phys. Rev. E 73, p. 41404, 2006, Fig. 2 and
# J. Glaser et al., JCP 143 18, p. 184110, 2015.

# reference data key = (phi_c, eta_p_r) value = (eta_p, error)
# 128 spheres
eta_p_ref = dict()
# eta_p_ref[(0.01,0.)] = (0.,nan)
eta_p_ref[(0.01, 0.2)] = (0.184438, 7.79773e-6)
eta_p_ref[(0.01, 0.4)] = (0.369113, 0.000015426)
eta_p_ref[(0.01, 0.6)] = (0.554202, 0.0000226575)
eta_p_ref[(0.01, 0.8)] = (0.73994, 0.0000319383)
eta_p_ref[(0.01, 1.)] = (0.926884, 0.0000312035)
eta_p_ref[(0.01, 1.2)] = (1.11685, 0.0000215032)
eta_p_ref[(0.01, 1.4)] = (1.32721, 0.000331169)
eta_p_ref[(0.01, 1.6)] = (1.52856, 0.0000769524)
eta_p_ref[(0.01, 1.8)] = (1.72593, 0.000131199)
eta_p_ref[(0.01, 2.)] = (1.92188, 0.000436138)
# eta_p_ref[(0.05,0.)] = (0.,nan)
eta_p_ref[(0.05, 0.2)] = (0.130102, 0.000017169)
eta_p_ref[(0.05, 0.4)] = (0.263677, 0.0000296967)
eta_p_ref[(0.05, 0.6)] = (0.402265, 0.0000358007)
eta_p_ref[(0.05, 0.8)] = (0.549098, 0.0000542385)
eta_p_ref[(0.05, 1.)] = (0.712581, 0.000143215)
eta_p_ref[(0.05, 1.2)] = (0.900993, 0.000116858)
eta_p_ref[(0.05, 1.4)] = (1.08466, 0.0001577)
eta_p_ref[(0.05, 1.6)] = (1.26389, 0.000312563)
eta_p_ref[(0.05, 1.8)] = (1.43957, 0.000490628)
eta_p_ref[(0.05, 2.)] = (1.61347, 0.000118301)
# eta_p_ref[(0.1,0.)] = (0.,nan)
eta_p_ref[(0.1, 0.2)] = (0.0777986, 0.0000224789)
eta_p_ref[(0.1, 0.4)] = (0.162055, 0.0000391019)
eta_p_ref[(0.1, 0.6)] = (0.25512, 0.0000917089)
eta_p_ref[(0.1, 0.8)] = (0.361985, 0.000081159)
eta_p_ref[(0.1, 1.)] = (0.491528, 0.000211232)
eta_p_ref[(0.1, 1.2)] = (0.644402, 0.0000945081)
eta_p_ref[(0.1, 1.4)] = (0.797721, 0.000114195)
eta_p_ref[(0.1, 1.6)] = (0.947405, 0.000266665)
eta_p_ref[(0.1, 1.8)] = (1.09756, 0.000207732)
eta_p_ref[(0.1, 2.)] = (1.24626, 0.00085732)
# eta_p_ref[(0.2,0.)] = (0.,nan)
eta_p_ref[(0.2, 0.2)] = (0.0180642, 8.88676e-7)
eta_p_ref[(0.2, 0.4)] = (0.0394307, 0.0000491992)
eta_p_ref[(0.2, 0.6)] = (0.0652104, 0.0000840904)
eta_p_ref[(0.2, 0.8)] = (0.0975177, 0.0000992883)
eta_p_ref[(0.2, 1.)] = (0.141602, 0.000141207)
eta_p_ref[(0.2, 1.2)] = (0.20416, 0.000278241)
eta_p_ref[(0.2, 1.4)] = (0.289024, 0.000340248)
eta_p_ref[(0.2, 1.6)] = (0.383491, 0.000357631)
eta_p_ref[(0.2, 1.8)] = (0.483246, 0.000338302)
eta_p_ref[(0.2, 2.)] = (0.594751, 0.00061228)
# eta_p_ref[(0.3,0.)] = (0.,nan)
eta_p_ref[(0.3, 0.2)] = (0.00154793, 6.84185e-6)
eta_p_ref[(0.3, 0.4)] = (0.00328478, 0.0000103679)
eta_p_ref[(0.3, 0.6)] = (0.00521468, 0.0000212988)
eta_p_ref[(0.3, 0.8)] = (0.00746148, 7.85157e-6)
eta_p_ref[(0.3, 1.)] = (0.0100912, 3.00293e-6)
eta_p_ref[(0.3, 1.2)] = (0.0131242, 0.0000590406)
eta_p_ref[(0.3, 1.4)] = (0.0169659, 0.0000524466)
eta_p_ref[(0.3, 1.6)] = (0.021623, 0.0000828658)
eta_p_ref[(0.3, 1.8)] = (0.0283405, 0.000133873)
eta_p_ref[(0.3, 2.)] = (0.0387704, 0.000167702)


@pytest.mark.parametrize("phi", [0.1])
@pytest.mark.parametrize("eta_p", [0.4])
@pytest.mark.parametrize("ntrial", [1])
@pytest.mark.parametrize("use_clusters", [False, True])
def test_sphere(simulation_factory, lattice_snapshot_factory, phi, eta_p,
                ntrial, use_clusters):
    # number of spheres
    n = 5
    N = n**3
    d_sphere = 1.0
    V_sphere = math.pi / 6.0 * math.pow(d_sphere, 3.0)

    # depletant-colloid size ratio (unity, to emphasize many-body effects)
    q = 1.0

    L_target = math.pow(N * V_sphere / phi, 1.0 / 3.0)

    a = L_target / n
    sim = simulation_factory(
        lattice_snapshot_factory(particle_types=['A', 'B'],
                                 dimensions=3,
                                 a=a,
                                 n=n))

    mc = hoomd.hpmc.integrate.Sphere(seed=1, default_d=0.1, default_a=0.1)
    sim.operations.integrator = mc

    mc.shape['A'] = dict(diameter=d_sphere)
    mc.shape['B'] = dict(diameter=d_sphere * q)

    tune = hoomd.hpmc.tune.MoveSize.scale_solver(trigger=hoomd.trigger.And(
        [hoomd.trigger.Periodic(100),
         hoomd.trigger.Before(1000)]),
                                                 moves=['d'],
                                                 types=['A'],
                                                 max_translation_move=d_sphere,
                                                 gamma=1,
                                                 target=0.2)
    sim.operations.tuners.append(tune)

    sim.run(1000)

    # warm up
    sim.run(2000)

    # set depletant fugacity
    nR = eta_p / (math.pi / 6.0 * math.pow(d_sphere * q, 3.0))
    mc.depletant_fugacity[('B', 'B')] = nR
    mc.depletant_ntrial[('B', 'B')] = ntrial

    if use_clusters:
        cl = hoomd.hpmc.update.Clusters(trigger=1, seed=1)
        sim.operations.updaters.append(cl)

        # use clusters exclusively to equilibrate
        mc.d['A'] = 0

    sim.run(100)
