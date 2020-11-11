import hoomd
import hoomd.write
import hoomd.hpmc
import numpy as np
import pytest
import hoomd.hpmc.pytest.conftest
from copy import deepcopy
import coxeter
from coxeter.shape_classes import ConvexPolyhedron


def test_vertex_moves(device, simulation_factory, lattice_snapshot_factory):
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
    mc.shape['A'] = {'vertices': np.array([(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5),
                                           (-0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5),
                                           (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)])}
    sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=4))
    sim.operations.add(mc)
    updater = hoomd.hpmc.update.Alchemy(mc=mc, move_ratio=1.0, seed=3832765, trigger=hoomd.trigger.Periodic(1), nselect=1)
    updater.vertex_shape_move(stepsize=0.01, param_ratio=0.2, volume=1.0)
    sim.operations.add(updater)
    sim.operations._schedule()
    sim.run(10)


ttf = coxeter.shape_families.TruncatedTetrahedronFamily()
class TruncatedTetrahedron:
    def __init__(self, mc):
        self.mc = mc

    def __getitem__(self,trunc):
        shape = ConvexPolyhedron(ttf(trunc).vertices / (ttf(trunc).volume**(1/3)))
        return [v for v in shape.vertices]

    def __call__(self, params):
        verts = self.__getitem__(params[0])
        args = {'vertices': verts, 'sweep_radius': 0.0, 'ignore_statistics': 0}
        return hoomd.hpmc._hpmc.PolyhedronVertices(args)


def test_python(device, simulation_factory, lattice_snapshot_factory):
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
    initial_value = 1.0
    mc.shape['A'] = {'vertices': ConvexPolyhedron(ttf(initial_value).vertices / (ttf(initial_value).volume**(1/3))).vertices}
    sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=4))
    sim.operations.add(mc)
    updater = hoomd.hpmc.update.Alchemy(mc=mc, move_ratio=1.0, seed=3832765, trigger=hoomd.trigger.Periodic(1), nselect=1)
    shape_gen_fn = TruncatedTetrahedron(mc=mc)
    updater.python_shape_move(shape_gen_fn, {'A': [initial_value]}, stepsize=0.05, param_ratio=1.0)
    sim.operations.add(updater)
    sim.operations._schedule()
    shape_param_changing = False
    for _ in range(10):
        sim.run(1)
        if updater.shape_param != initial_value:
            shape_param_changing = True
    assert shape_param_changing


def test_constant_moves(device, simulation_factory, lattice_snapshot_factory):
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
    mc.shape['A'] = {'vertices': ConvexPolyhedron(ttf(0.0).vertices / (ttf(0.0).volume**(1/3))).vertices}
    sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=4))
    sim.operations.add(mc)
    updater = hoomd.hpmc.update.Alchemy(mc=mc, move_ratio=1.0, seed=3832765, trigger=hoomd.trigger.Periodic(1), nselect=1)
    updater.constant_shape_move(vertices=ConvexPolyhedron(ttf(1.0).vertices / (ttf(1.0).volume**(1/3))).vertices, ignore_statistics=0, sweep_radius=0.0)
    sim.operations.add(updater)
    sim.operations._schedule()
    sim.run(10)


def test_elastic_moves(device, simulation_factory, lattice_snapshot_factory):
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
    mc.shape['A'] = {'vertices': ConvexPolyhedron(ttf(0.0).vertices / (ttf(0.0).volume**(1/3))).vertices}
    sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=4))
    sim.operations.add(mc)
    updater = hoomd.hpmc.update.ElasticShape(mc=mc, trigger=hoomd.trigger.Periodic(1), stepsize=0.001, move_ratio=1.0, seed=3832765, stiffness=hoomd.variant.Ramp(1.0, 10.0, 0, 20), reference=dict(vertices=ConvexPolyhedron(ttf(1.0).vertices / (ttf(1.0).volume**(1/3))).vertices, ignore_statistics=0, sweep_radius=0.0), nselect=3, nsweeps=2, param_ratio=0.5)
    sim.operations.add(updater)
    sim.operations._schedule()
    sim.run(10)


def test_tuner(device, simulation_factory, lattice_snapshot_factory):
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
    mc.shape['A'] = {'vertices': np.array([(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5),
                                           (-0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5),
                                           (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)])}
    sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=4))
    sim.operations.add(mc)
    updater = hoomd.hpmc.update.Alchemy(mc=mc, move_ratio=1.0, seed=3832765, trigger=hoomd.trigger.Periodic(1), nselect=1)
    updater.vertex_shape_move(stepsize=0.01, param_ratio=0.2, volume=1.0)
    sim.operations.add(updater)
    tuner = updater.get_tuner(hoomd.trigger.Periodic(1), 0.2)
    sim.operations.add(tuner)
    sim.operations._schedule()
    sim.run(10)
    acceptance1 = float(updater.accepted_count) / float(updater.total_count)
    sim.run(100)
    acceptance2 = float(updater.accepted_count) / float(updater.total_count)
    assert abs(acceptance1 - 0.2) > abs(acceptance2 - 0.2)


def test_logger(device, simulation_factory, lattice_snapshot_factory, tmpdir):
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
    mc.shape['A'] = {'vertices': ConvexPolyhedron(ttf(0.0).vertices / (ttf(0.0).volume**(1/3))).vertices}
    n = 4
    sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=n))
    sim.operations.add(mc)
    stiffness = 10.0
    updater = hoomd.hpmc.update.ElasticShape(mc=mc, trigger=hoomd.trigger.Periodic(1), stepsize=0.001, move_ratio=1.0, seed=3832765, stiffness=hoomd.variant.Constant(stiffness), reference=dict(vertices=ConvexPolyhedron(ttf(1.0).vertices / (ttf(1.0).volume**(1/3))).vertices, ignore_statistics=0, sweep_radius=0.0), nselect=3, nsweeps=2, param_ratio=0.5)
    sim.operations.add(updater)

    file_name = tmpdir.mkdir("sub").join("tmp_updater.txt")
    log_file = open(file_name, "w+")
    logger = hoomd.logging.Logger(flags=['scalar'])
    logger += updater
    writer = hoomd.write.Table(hoomd.trigger.Periodic(1), logger, log_file, max_header_len=1)
    sim.operations.add(writer)
    sim.operations._schedule()

    shape_move_acceptance_ratios = []
    for _ in range(10):
        sim.run(1)
        shape_move_acceptance_ratios.append(float(updater.accepted_count) / float(updater.total_count))
    log_file.close()

    acceptance, volume, sp, stiffness_arr, energy = np.hsplit(np.loadtxt(file_name,
                                                                         skiprows=1),
                                                              5)
    assert len(acceptance[acceptance < 0]) == 0
    assert len(acceptance[acceptance > 1]) == 0
    assert len(energy[energy < 0]) == 0

    np.testing.assert_allclose(volume, np.array([n**3] * len(volume)).reshape(len(volume), 1))
    np.testing.assert_allclose(sp, np.array([0.0] * len(sp)).reshape(len(sp), 1))
    np.testing.assert_allclose(stiffness_arr, np.array([stiffness] * len(stiffness_arr)).reshape(len(stiffness_arr), 1))
