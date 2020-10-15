import hoomd
import hoomd.hpmc
import numpy as np
import pytest
import hoomd.hpmc.pytest.conftest
from copy import deepcopy
import coxeter
from coxeter.shape_classes import ConvexPolyhedron


def test_vertex_moves(device, simulation_factory, two_particle_snapshot_factory):
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
    mc.shape['A'] = {'vertices': np.array([(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5),
                                           (-0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5),
                                           (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)])}
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=2))
    sim.operations.add(mc)
    updater = hoomd.hpmc.update.alchemy(mc=mc, move_ratio=1.0, seed=3832765, trigger=hoomd.trigger.Periodic(1), nselect=1)
    sim.operations.add(updater)
    sim.operations.schedule()
    updater.vertex_shape_move(stepsize=0.01, param_ratio=0.2, volume=1.0)
    sim.operations.schedule()
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


def test_python(device, simulation_factory, two_particle_snapshot_factory):
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
    mc.shape['A'] = {'vertices': ConvexPolyhedron(ttf(0.0).vertices / (ttf(0.0).volume**(1/3))).vertices}
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=2))
    sim.operations.add(mc)
    updater = hoomd.hpmc.update.alchemy(mc=mc, move_ratio=1.0, seed=3832765, trigger=hoomd.trigger.Periodic(1), nselect=1)
    sim.operations.add(updater)
    sim.operations.schedule()
    shape_gen_fn = TruncatedTetrahedron(mc=mc)
    updater.python_shape_move(shape_gen_fn, {'A': [0]}, stepsize=0.5, param_ratio=0.5)
    sim.operations.schedule()
    sim.run(10)


def test_constant_moves(device, simulation_factory, two_particle_snapshot_factory):
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
    mc.shape['A'] = {'vertices': ConvexPolyhedron(ttf(0.0).vertices / (ttf(0.0).volume**(1/3))).vertices}
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=2))
    sim.operations.add(mc)
    updater = hoomd.hpmc.update.alchemy(mc=mc, move_ratio=1.0, seed=3832765, trigger=hoomd.trigger.Periodic(1), nselect=1)
    sim.operations.add(updater)
    sim.operations.schedule()
    updater.constant_shape_move(vertices=ConvexPolyhedron(ttf(1.0).vertices / (ttf(1.0).volume**(1/3))).vertices, ignore_statistics=0, sweep_radius=0.0)
    sim.operations.schedule()
    sim.run(10)

def test_elastic_moves(device, simulation_factory, two_particle_snapshot_factory):
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
    mc.shape['A'] = {'vertices': ConvexPolyhedron(ttf(0.0).vertices / (ttf(0.0).volume**(1/3))).vertices}
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=2))
    sim.operations.add(mc)
    updater = hoomd.hpmc.update.elastic_shape(mc=mc, trigger=hoomd.trigger.Periodic(1), stepsize=0.001, move_ratio=1.0, seed=3832765, stiffness=hoomd.variant.Ramp(1.0, 10.0, 0, 20), reference=dict(vertices=ConvexPolyhedron(ttf(1.0).vertices / (ttf(1.0).volume**(1/3))).vertices, ignore_statistics=0, sweep_radius=0.0), nselect=3, nsweeps=2, param_ratio=0.5)
    sim.operations.add(updater)
    # logger = hoomd.analyze.log(filename="shape_updater.log", period=10, overwrite=True, quantities=['shape_move_energy', 'shape_move_stiffness', 'shape_move_acceptance_ratio', 'shape_move_particle_volume'])
    logger = hoomd.logging.Logger()
    logger += updater
    sim.operations.schedule()
    print(updater.shape_move_energy())
    # lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell())
    # lj.params[('A', 'A')] = {'sigma': 1, 'epsilon': 0.5}
    # logger += lj
    # sim.operations.add(logger)
    # sim.operations.schedule()
    sim.run(10)
    print(updater.shape_move_energy())
    assert False
