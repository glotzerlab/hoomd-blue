# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import hoomd.conftest
from hoomd import hpmc
import numpy as np
import pytest
from hoomd.hpmc.shape_move import ShapeMove, Vertex, ShapeSpace, Elastic

shape_move_classes = [Vertex, ShapeSpace, Elastic]


def _test_callback(typeid, param_list):
    pass


# vertices of a regular cube: used for testing vertex and elastic shape moves
verts = np.asarray([[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5],
                    [0.5, -0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, -0.5, 0.5],
                    [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]])

shape_move_constructor_args = [
    (Vertex, dict(vertex_move_probability=0.7)),
    (ShapeSpace, dict(callback=_test_callback, param_move_probability=1)),
    (Elastic,
     dict(stiffness=hoomd.variant.Constant(10),
          mc=hpmc.integrate.ConvexPolyhedron,
          normal_shear_ratio=0.5)),
]

shape_move_valid_attrs = [
    (Vertex(), "vertex_move_probability", 0.1),
    (ShapeSpace(callback=_test_callback), "param_move_probability", 0.1),
    (ShapeSpace(callback=_test_callback), "callback",
     lambda type, param_list: {}),
    (Elastic(1, hpmc.integrate.ConvexPolyhedron), "normal_shear_ratio", 0.5),
    (Elastic(1, hpmc.integrate.ConvexPolyhedron), "stiffness",
     hoomd.variant.Constant(10)),
    (Elastic(1, hpmc.integrate.ConvexPolyhedron), "stiffness",
     hoomd.variant.Ramp(1, 5, 0, 100)),
    (Elastic(1, hpmc.integrate.ConvexPolyhedron), "stiffness",
     hoomd.variant.Cycle(1, 5, 0, 10, 20, 10, 15)),
    (Elastic(1, hpmc.integrate.ConvexPolyhedron), "stiffness",
     hoomd.variant.Power(1, 5, 3, 0, 100))
]

shape_updater_valid_attrs = [("trigger", hoomd.trigger.Periodic(10)),
                             ("trigger", hoomd.trigger.After(100)),
                             ("trigger", hoomd.trigger.Before(100)),
                             ("type_select", 2), ("nweeps", 4),
                             ("shape_move", Vertex()),
                             ("shape_move",
                              ShapeSpace(callback=_test_callback)),
                             ("shape_move",
                              Elastic(stiffness=10,
                                      mc=hpmc.integrate.ConvexPolyhedron))]

updater_constructor_args = [
    dict(trigger=hoomd.trigger.Periodic(10), shape_move=ShapeMove()),
    dict(trigger=hoomd.trigger.After(100),
         shape_move=Vertex(),
         type_select=4,
         nsweeps=2),
    dict(trigger=hoomd.trigger.Before(100),
         shape_move=ShapeSpace(callback=_test_callback),
         nsweeps=4),
    dict(trigger=hoomd.trigger.Periodic(1000),
         shape_move=ShapeSpace(callback=_test_callback),
         type_select=1),
    dict(trigger=hoomd.trigger.Periodic(10),
         shape_move=Elastic(stiffness=10, mc=hpmc.integrate.ConvexPolyhedron),
         type_select=3,
         pretend=True)
]

type_parameters = [
    (ShapeSpace(callback=_test_callback), "params", [0.1, 0.3, 0.4]),
    (ShapeSpace(callback=_test_callback), "step_size", 0.4),
    (Vertex(), "volume", 1.2), (Vertex(), "step_size", 0.1),
    (Elastic(stiffness=10.0,
             mc=hpmc.integrate.ConvexPolyhedron), "reference_shape", {
                 "vertices": verts
             }),
    (Elastic(stiffness=10.0,
             mc=hpmc.integrate.ConvexPolyhedron), "step_size", 0.2)
]


@pytest.mark.parametrize("shape_move_class,params", shape_move_constructor_args)
def test_valid_construction_shape_moves(shape_move_class, params):

    move = shape_move_class(**params)

    # validate the params were set properly
    for attr, value in params.items():
        if hasattr(move, attr):
            assert getattr(move, attr) == value


@pytest.mark.parametrize("updater_constructor_args", updater_constructor_args)
def test_valid_construction_shape_updater(updater_constructor_args):

    updater = hpmc.update.Shape(**updater_constructor_args)

    # validate the params were set properly
    for attr, value in updater_constructor_args.items():
        assert getattr(updater, attr) == value


@pytest.mark.parametrize("shape_move_obj,attr,value", shape_move_valid_attrs)
def test_valid_setattr_shape_move(shape_move_obj, attr, value):
    """Test that the shape move classes can get and set attributes."""
    setattr(shape_move_obj, attr, value)
    assert getattr(shape_move_obj, attr) == value


@pytest.mark.parametrize("attr,value", shape_updater_valid_attrs)
def test_valid_setattr_shape_updater(attr, value):
    """Test that the Shape updater can get and set attributes."""
    updater = hpmc.update.Shape(trigger=1, shape_move=ShapeMove())

    setattr(updater, attr, value)
    assert getattr(updater, attr) == value


@pytest.mark.parametrize("obj,attr,value", type_parameters)
def test_type_parameters(obj, attr, value):
    getattr(obj, attr)["A"] = value
    hoomd.conftest.equality_check(getattr(obj, attr)["A"], value)


def test_vertex_shape_move(simulation_factory, two_particle_snapshot_factory):

    move = Vertex(default_step_size=0.2)
    move.volume["A"] = 1

    updater = hpmc.update.Shape(trigger=1, shape_move=move, nsweeps=2)
    updater.shape_move = move

    mc = hoomd.hpmc.integrate.ConvexPolyhedron()
    mc.d["A"] = 0
    mc.a["A"] = 0
    mc.shape["A"] = dict(vertices=verts)

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory(d=10))
    sim.operations.integrator = mc
    sim.operations += updater

    # test attachmet before first run
    assert not move._attached
    assert not updater._attached
    sim.run(0)

    # test attachmet after first run
    assert move._attached
    assert updater._attached

    # run with 0 probability of performing a move:
    #  - shape should remain unchanged
    move.vertex_move_probability = 0
    sim.run(10)
    assert np.allclose(mc.shape["A"]["vertices"], verts)

    # always attempt a shape move:
    #  - shape should change
    #  - volume should remain unchanged
    move.vertex_move_probability = 1
    sim.run(10)
    assert np.sum(updater.shape_moves) == 20
    assert not np.allclose(mc.shape["A"]["vertices"], verts)
    assert np.isclose(updater.particle_volumes[0], 1)


def test_python_callback_shape_move_ellipsoid(simulation_factory,
                                              lattice_snapshot_factory):
    """Test ShapeSpace with a toy class that randomly squashes spheres \
           into oblate ellipsoids with constant volume."""

    class ScaleEllipsoid:

        def __init__(self, a, b, c):
            self.vol_factor = 4 * np.pi / 3
            self.volume = self.vol_factor * a * b * c
            self.default_dict = dict(ignore_statistics=True)

        def __call__(self, type_id, param_list):
            x = param_list[0]
            b = (self.volume / x / (self.vol_factor))**(1 / 3)
            ret = dict(a=x * b, b=b, c=b, **self.default_dict)
            return ret

    ellipsoid = dict(a=1, b=1, c=1)

    move = ShapeSpace(callback=ScaleEllipsoid(**ellipsoid),
                      default_step_size=0.2)
    move.params["A"] = [1]

    updater = hpmc.update.Shape(trigger=1, shape_move=move, nsweeps=2)
    updater.shape_move = move

    mc = hoomd.hpmc.integrate.Ellipsoid()
    mc.d["A"] = 0
    mc.a["A"] = 0
    mc.shape["A"] = ellipsoid

    # create simulation & attach objects
    sim = simulation_factory(lattice_snapshot_factory(a=2.75, n=(3, 3, 5)))
    sim.operations.integrator = mc
    sim.operations += updater

    # test attachmet before first run
    assert not move._attached
    assert not updater._attached

    sim.run(0)

    # test attachmet after first run
    assert move._attached
    assert updater._attached

    # run with 0 probability of performing a move:
    #  - shape and params should remain unchanged
    #  - no shape moves proposed
    move.param_move_probability = 0
    sim.run(10)
    assert np.allclose(mc.shape["A"]["a"], ellipsoid["a"])
    assert np.allclose(mc.shape["A"]["b"], ellipsoid["b"])
    assert np.allclose(mc.shape["A"]["c"], ellipsoid["c"])
    assert np.allclose(move.params["A"], [1])

    # always attempt a shape move:
    #  - shape and params should change
    #  - volume should remain unchanged
    move.param_move_probability = 1
    sim.run(50)
    assert np.sum(updater.shape_moves) == 100

    # Check that the shape parameters have changed
    assert not np.allclose(mc.shape["A"]["a"], ellipsoid["a"])
    assert not np.allclose(mc.shape["A"]["b"], ellipsoid["b"])
    assert not np.allclose(mc.shape["A"]["c"], ellipsoid["c"])
    assert not np.allclose(move.params["A"], [1])

    # Check that the shape parameters map back to the correct geometry
    assert np.allclose(move.params["A"], [
        mc.shape["A"]["a"] / mc.shape["A"]["b"],
        mc.shape["A"]["a"] / mc.shape["A"]["c"]
    ])

    # Check that the callback is conserving volume properly
    assert np.allclose(updater.particle_volumes, 4 * np.pi / 3)


def test_python_callback_shape_move_pyramid(simulation_factory,
                                            two_particle_snapshot_factory):
    """Test ShapeSpace with a toy class that randomly stretches square \
        pyramids."""

    def square_pyramid_factory(h):
        """Generate a square pyramid with unit volume."""
        theta = np.arange(0, 2 * np.pi, np.pi / 2)
        base_vertices = np.array(
            [np.cos(theta), np.sin(theta),
             np.zeros_like(theta)]).T * np.sqrt(3 / 2)
        vertices = np.vstack([base_vertices, [0, 0, h]])
        return vertices / np.cbrt(h), base_vertices

    class ScalePyramid:

        def __init__(self, h=1.1):
            _, self.base_vertices = square_pyramid_factory(h=1.1)
            self.default_dict = dict(sweep_radius=0, ignore_statistics=True)

        def __call__(self, type_id, param_list):
            h = param_list[0] + 0.1  # Prevent a 0-height pyramid
            new_vertices = np.vstack([self.base_vertices, [0, 0, h]])
            new_vertices /= np.cbrt(h)  # Rescale to unit volume
            ret = dict(vertices=new_vertices, **self.default_dict)
            return ret

    initial_pyramid, _ = square_pyramid_factory(h=1.1)

    move = ShapeSpace(callback=ScalePyramid(), default_step_size=0.2)
    move.params["A"] = [1]

    updater = hpmc.update.Shape(trigger=1, shape_move=move, nsweeps=2)
    updater.shape_move = move

    mc = hoomd.hpmc.integrate.ConvexPolyhedron()
    mc.d["A"] = 0
    mc.a["A"] = 0
    mc.shape["A"] = dict(vertices=initial_pyramid)

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory(d=2.5))
    sim.operations.integrator = mc
    sim.operations += updater

    # test attachmet before first run
    assert not move._attached
    assert not updater._attached

    sim.run(0)

    # test attachmet after first run
    assert move._attached
    assert updater._attached

    # run with 0 probability of performing a move:
    #  - shape and params should remain unchanged
    #  - all moves accepted
    move.param_move_probability = 0
    sim.run(10)
    assert np.allclose(mc.shape["A"]["vertices"], initial_pyramid)
    assert np.allclose(move.params["A"], [1])
    assert np.allclose(updater.particle_volumes, 1)

    # always attempt a shape move:
    #  - shape and params should change
    #  - volume should remain unchanged
    move.param_move_probability = 1
    sim.run(50)
    assert np.sum(updater.shape_moves) == 100

    # Check that the shape parameters have changed
    current_h = move.params["A"][0]
    assert not np.allclose(mc.shape["A"]["vertices"], initial_pyramid)
    assert not np.isclose(current_h, 1)

    # Check that the shape parameters map back to the correct geometry
    assert np.allclose(
        square_pyramid_factory(current_h + 0.1)[0], mc.shape["A"]["vertices"])

    # Check that the callback is conserving volume properly
    assert np.allclose(updater.particle_volumes, 1)


def test_elastic_shape_move(simulation_factory, two_particle_snapshot_factory):

    mc = hoomd.hpmc.integrate.ConvexPolyhedron()
    mc.d["A"] = 0
    mc.a["A"] = 0
    mc.shape["A"] = dict(vertices=verts)

    move = Elastic(stiffness=1, mc=mc, default_step_size=0.1)
    move.reference_shape["A"] = dict(vertices=verts)

    updater = hpmc.update.Shape(trigger=1, shape_move=move, nsweeps=2)

    # create simulation & attach objects
    sim = simulation_factory(two_particle_snapshot_factory(d=10))
    sim.operations.integrator = mc
    sim.operations += updater

    # test attachmet before first run
    assert not move._attached
    assert not updater._attached

    sim.run(0)

    # test attachmet after first run
    assert move._attached
    assert updater._attached

    # always attempt a shape move:
    #  - shape should change
    #  - volume should remain unchanged
    sim.run(10)
    assert np.sum(updater.shape_moves) == 20
    assert not np.allclose(mc.shape["A"]["vertices"], verts)
    assert np.allclose(updater.particle_volumes, 1)

    # compute mean vertex displacement at this k
    dr_k1 = np.linalg.norm(mc.shape["A"]["vertices"] - verts, axis=1).mean()
    # reset shape, ramp up stiffness and run again
    mc.shape["A"] = dict(vertices=verts)
    move.stiffness = 500
    sim.run(10)

    # mean vertex displacement should be smaller for stiffer particles
    dr_k100 = np.linalg.norm(mc.shape["A"]["vertices"] - verts, axis=1).mean()
    assert dr_k100 < dr_k1
