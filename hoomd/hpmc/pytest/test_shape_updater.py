import hoomd
import hoomd.write
import hoomd.hpmc
import numpy as np
import pytest
from copy import deepcopy
import coxeter
from coxeter.shapes import ConvexPolyhedron


# def test_vertex_moves(device, simulation_factory, lattice_snapshot_factory):
#     mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
#     mc.shape['A'] = {'vertices': np.array([(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5),
#                                            (-0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5),
#                                            (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)])}
#     sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=4))
#     sim.seed = 0
#     sim.operations.add(mc)
#     vertex_move = hoomd.hpmc.shape_move.Vertex(stepsize={'A': 0.01}, param_ratio=0.2, volume=1.0)
#     shape_updater = hoomd.hpmc.update.Shape(shape_move=vertex_move, move_ratio=1.0, trigger=hoomd.trigger.Periodic(1), nselect=1)
#     sim.operations.add(shape_updater)
#     sim.run(0)


ttf = coxeter.families.TruncatedTetrahedronFamily()
class TruncatedTetrahedron:
    def __getitem__(self, trunc):
        shape = ConvexPolyhedron(ttf.get_shape(trunc).vertices / (ttf.get_shape(trunc).volume**(1/3)))
        return [v for v in shape.vertices]

    def __call__(self, params):
        verts = self.__getitem__(params[0])
        args = {'vertices': verts, 'sweep_radius': 0.0, 'ignore_statistics': 0}
        return hoomd.hpmc._hpmc.PolyhedronVertices(args)


# def test_python(device, simulation_factory, lattice_snapshot_factory):
#     mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
#     initial_value = 1.0
#     mc.shape['A'] = {'vertices': ConvexPolyhedron(ttf.get_shape(initial_value).vertices / (ttf.get_shape(initial_value).volume**(1/3))).vertices}
#     sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=4))
#     sim.seed = 0
#     sim.operations.add(mc)
#     python_move = hoomd.hpmc.shape_move.Python(callback=TruncatedTetrahedron(), params=[[initial_value]], stepsize={'A': 0.05}, param_ratio=1.0)
#     shape_updater = hoomd.hpmc.update.Shape(shape_move=python_move, move_ratio=1.0, trigger=hoomd.trigger.Periodic(1), nselect=1)
#     sim.operations.add(shape_updater)
#     sim.run(10)
#     assert shape_updater.shape_param != initial_value


def test_constant_moves(device, simulation_factory, lattice_snapshot_factory):
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
    mc.shape['A'] = {'vertices': ConvexPolyhedron(ttf.get_shape(0.0).vertices / (ttf.get_shape(0.0).volume**(1/3))).vertices}
    sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=4))
    sim.seed = 0
    sim.operations.add(mc)
    constant_move = hoomd.hpmc.shape_move.Constant(shape_params={'A': dict(vertices=ConvexPolyhedron(ttf.get_shape(1.0).vertices / (ttf.get_shape(1.0).volume**(1/3))).vertices, ignore_statistics=0, sweep_radius=0.0)})
    shape_updater = hoomd.hpmc.update.Shape(shape_move=constant_move, move_ratio=1.0, trigger=hoomd.trigger.Periodic(1), nselect=1)
    sim.operations.add(shape_updater)
    sim.run(0)


# def test_elastic_moves(device, simulation_factory, lattice_snapshot_factory):
#     mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
#     mc.shape['A'] = {'vertices': ConvexPolyhedron(ttf.get_shape(0.0).vertices / (ttf.get_shape(0.0).volume**(1/3))).vertices}
#     sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=4))
#     sim.seed = 0
#     sim.operations.add(mc)
#     elastic_move = hoomd.hpmc.shape_move.Elastic(stiffness=hoomd.variant.Constant(1.0),
#                                                  reference={'A': dict(vertices=ConvexPolyhedron(ttf.get_shape(1.0).vertices / (ttf.get_shape(1.0).volume**(1/3))).vertices,
#                                                                       ignore_statistics=0,
#                                                                       sweep_radius=0.0)},
#                                                  stepsize={'A': 0.001},
#                                                  param_ratio=0.5)
#     shape_updater = hoomd.hpmc.update.Shape(shape_move=elastic_move, move_ratio=1.0, trigger=hoomd.trigger.Periodic(1), nselect=1)
#     sim.operations.add(shape_updater)
#     sim.run(0)
