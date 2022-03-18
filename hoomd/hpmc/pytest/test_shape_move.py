# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from collections.abc import Sequence, Mapping
import math
from numbers import Number
import hoomd
import hoomd.write
import hoomd.hpmc
import numpy as np
import pytest
from copy import deepcopy
import coxeter
from coxeter.shapes import ConvexPolyhedron
from hoomd.logging import LoggerCategories
from hoomd.conftest import logging_check

_seed = 0


def _equivalent_data_structures(reference, struct_2):
    """Compare arbitrary data structures for equality.

    ``reference`` is expected to be the reference data structure. Cannot handle
    set like data structures.
    """
    if isinstance(reference, np.ndarray):
        return np.allclose(reference, struct_2)
    if isinstance(reference, Mapping):
        # if the non-reference value does not have all the keys
        # we don't check for the exact same keys, since some values may have
        # defaults.
        if set(reference.keys()) - set(struct_2.keys()):
            return False
        return all(
            _equivalent_data_structures(reference[key], struct_2[key])
            for key in reference)
    if isinstance(reference, Sequence):
        if len(reference) != len(struct_2):
            return False
        return all(
            _equivalent_data_structures(value_1, value_2)
            for value_1, value_2 in zip(reference, struct_2))
    if isinstance(reference, Number):
        return math.isclose(reference, struct_2, rel_tol=1e-4)


ttf = coxeter.families.TruncatedTetrahedronFamily()


class TruncatedTetrahedron(hoomd.hpmc.shape_move.Callback):

    def __getitem__(self, trunc):
        shape = ConvexPolyhedron(
            ttf.get_shape(trunc).vertices /
            (ttf.get_shape(trunc).volume**(1 / 3)))
        return [v for v in shape.vertices]

    def __call__(self, params):
        verts = self.__getitem__(params[0])
        args = {'vertices': verts, 'sweep_radius': 0.0, 'ignore_statistics': 0}
        return hoomd.hpmc._hpmc.PolyhedronVertices(args)


_ttf1_verts = ConvexPolyhedron(
    ttf.get_shape(1.0).vertices / (ttf.get_shape(1.0).volume**(1 / 3))).vertices
_ttf2_verts = ConvexPolyhedron(
    ttf.get_shape(0.5).vertices / (ttf.get_shape(0.5).volume**(1 / 3))).vertices

_constant_args = [
    dict(shape_params={
        'A': dict(vertices=_ttf1_verts, ignore_statistics=0, sweep_radius=0.0)
    }),
    dict(shape_params={
        'A': dict(vertices=_ttf2_verts, ignore_statistics=1, sweep_radius=0.5)
    })
]

_elastic_args = [
    dict(stiffness=hoomd.variant.Constant(1.0),
         reference=dict(vertices=_ttf1_verts,
                        ignore_statistics=0,
                        sweep_radius=0.0),
         stepsize=0.001,
         param_ratio=0.5),
    dict(stiffness=hoomd.variant.Ramp(0.5, 5.0, 1, 100),
         reference=dict(vertices=_ttf2_verts,
                        ignore_statistics=0,
                        sweep_radius=0.0),
         stepsize=0.005,
         param_ratio=0.75)
]

# _python_args = [
#     dict(callback=TruncatedTetrahedron(),
#          params={'A': [1.0]},
#          stepsize={'A': 0.05},
#          param_ratio=1.0),
#     dict(callback=TruncatedTetrahedron(),
#          params={'A': [0.5]},
#          stepsize={'A': 0.02},
#          param_ratio=0.7)
]

# _vertex_args = [
#     dict(stepsize={'A': 0.01}, param_ratio=0.2, volume=1.0),
#     dict(stepsize={'A': 0.02}, param_ratio=0.5, volume=0.5)
# ]


# def get_move_and_args():
#     move_and_args = [(hoomd.hpmc.shape_move.Constant, _constant_args),
#                      (hoomd.hpmc.shape_move.Elastic, _elastic_args),
#                      (hoomd.hpmc.shape_move.Python, _python_args),
#                      (hoomd.hpmc.shape_move.Vertex, _vertex_args)]
#     return move_and_args

shape_updater_constructor_args = [
    dict(trigger = hoomd.)
]

python_move_constructor_args = [
    dict(
        callback = lambda typeid, param_list: {"vertices": [[],[]],
                                               "sweep_radius": 0,
                                               "ignore_statistics": True},
        param_move_probability = 1
        ),
    dict(
        callback = lambda typeid, param_list: {"vertices": [[],[]],
                                               "sweep_radius": 0,
                                               "ignore_statistics": True},
        param_move_probability = 0.4
        ),
]

@pytest.mark.parametrize("constructor_args", python_move_constructor_args)
def test_valid_construction_python_shape_move(constructor_args):
    move = hpmc.shape_move.PythonShapeMove(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(move, attr) == value

@pytest.mark.parametrize("constructor_args", python_move_constructor_args)
def test_valid_construction_and_attach_python_shape_move(device,
        simulation_factory, two_particle_snapshot_factory, constructor_args):
        move = hpmc.shape_move.PythonShapeMove(**constructor_args)
        move.params["A"] = [0.2, 0.23, 0.33]

        mc = hoomd.hpmc.integrate.ConvexPolyhedron()
        mc.shape['A'] = dict(vertices=_vertices)

        # create simulation & attach objects
        sim = simulation_factory(two_particle_snapshot_factory())
        sim.operations.integrator = mc

        # catch the error if running on the GPU
        if isinstance(device, hoomd.device.GPU):
            with pytest.raises(RuntimeError):
                sim.run(0)

def test_valid_construction_vertex_shape_move(args):
    pass

def test_valid_construction_elastic_shape_move(args):
    pass

@pytest.mark.parametrize("move_and_args", get_move_and_args())
def test_before_attaching(move_and_args):
    move, move_args = move_and_args
    shape_move = move(**move_args[0])
    for key, val in move_args[1].items():
        if key not in ['callback', 'reference', 'stiffness']:
            assert _equivalent_data_structures(getattr(shape_move, key),
                                               move_args[0][key])
            setattr(shape_move, key, val)
            assert _equivalent_data_structures(getattr(shape_move, key), val)


@pytest.mark.parametrize("move_and_args", get_move_and_args())
def test_after_attaching(device, simulation_factory, lattice_snapshot_factory,
                         move_and_args):
    move, move_args = move_and_args
    shape_move = move(**move_args[0])
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(_seed)
    mc.shape['A'] = {'vertices': _ttf1_verts}
    sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=3))
    sim.seed = _seed
    sim.operations.add(mc)
    shape_updater = hoomd.hpmc.update.Shape(shape_move=shape_move,
                                            move_ratio=1.0,
                                            trigger=hoomd.trigger.Periodic(1),
                                            nselect=1)
    sim.operations.add(shape_updater)
    sim.run(0)
    for key, val in move_args[1].items():
        if key not in ['callback', 'reference', 'stiffness']:
            setattr(shape_move, key, val)
            assert _equivalent_data_structures(getattr(shape_move, key), val)


def test_vertex(device, simulation_factory, lattice_snapshot_factory):
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(_seed)
    mc.shape['A'] = {'vertices': _ttf1_verts}
    sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=3))
    sim.seed = _seed
    sim.operations.add(mc)
    vertex_move = hoomd.hpmc.shape_move.Vertex(stepsize={'A': 0.01},
                                               param_ratio=0.2,
                                               volume=1.0)
    shape_updater = hoomd.hpmc.update.Shape(shape_move=vertex_move,
                                            move_ratio=1.0,
                                            trigger=hoomd.trigger.Periodic(1),
                                            nselect=1)
    sim.operations.add(shape_updater)
    sim.run(10)
    assert not np.allclose(np.asarray(mc.shape['A']['vertices']), _ttf1_verts)


def test_python(device, simulation_factory, lattice_snapshot_factory):
    initial_value = 1.0
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(_seed)
    mc.shape['A'] = {'vertices': _ttf1_verts}
    sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=3))
    sim.seed = _seed
    sim.operations.add(mc)
    python_move = hoomd.hpmc.shape_move.Python(callback=TruncatedTetrahedron(),
                                               params={'A': [initial_value]},
                                               stepsize={'A': 0.05},
                                               param_ratio=1.0)
    shape_updater = hoomd.hpmc.update.Shape(shape_move=python_move,
                                            move_ratio=1.0,
                                            trigger=hoomd.trigger.Periodic(1),
                                            nselect=1)
    sim.operations.add(shape_updater)
    expected_verts = np.asarray(
        TruncatedTetrahedron().__getitem__(initial_value))
    sim.run(0)
    assert np.allclose(np.asarray(mc.shape['A']["vertices"]), expected_verts)

    sim.run(10)
    expected_verts = np.asarray(TruncatedTetrahedron().__getitem__(
        python_move.params['A'][0]))
    assert np.allclose(np.asarray(mc.shape['A']["vertices"]), expected_verts)
    logging_check(
        hoomd.hpmc.shape_move.Python, ('hpmc', 'shape_move'),
        {'shape_param': {
            'category': LoggerCategories.object,
            'default': True
        }})


def test_elastic_moves(device, simulation_factory, lattice_snapshot_factory):
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
    mc.shape['A'] = {
        'vertices':
            ConvexPolyhedron(
                ttf.get_shape(0.0).vertices /
                (ttf.get_shape(0.0).volume**(1 / 3))).vertices
    }
    sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=4))
    sim.seed = 0
    sim.operations.add(mc)
    c = hoomd.variant.Constant(1.0)
    elastic_move = hoomd.hpmc.shape_move.Elastic(stiffness=c,
                                                 reference=dict(
                                                     vertices=_ttf1_verts,
                                                     ignore_statistics=0,
                                                     sweep_radius=0.0),
                                                 stepsize=0.001,
                                                 param_ratio=0.5)
    assert elastic_move.stiffness is c
    shape_updater = hoomd.hpmc.update.Shape(shape_move=elastic_move,
                                            move_ratio=1.0,
                                            trigger=hoomd.trigger.Periodic(1),
                                            nselect=1)
    sim.operations.add(shape_updater)
    sim.run(2)
    assert elastic_move.stiffness is c
    assert not np.isclose(shape_updater.shape_move_energy, 0.0)
