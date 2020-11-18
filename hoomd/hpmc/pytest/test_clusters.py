# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test hoomd.hpmc.update.Clusters."""

import hoomd
import pytest
import math
import hoomd.hpmc.pytest.conftest

_3d_verts = [(0.5, 0.5, 0.5),
             (0.5, -0.5, -0.5),
             (-0.5, 0.5, -0.5),
             (-0.5, -0.5, 0.5)]

_2d_verts = [(-0.5, -0.5),
             (0.5, -0.5),
             (0.5, 0.5),
             (-0.5, 0.5)]

integrators = [
    ('Sphere', dict(diameter=1)),
    ('ConvexPolyhedron', dict(vertices=_3d_verts)),
    ('ConvexSpheropolyhedron', dict(vertices=_3d_verts, sweep_radius=0.1)),
    ('Ellipsoid', dict(a=0.5, b=0.25, c=0.125))
]


               #
               # dict(shape='ConvexSpheropolyhedron',
               #      params=dict(vertices=_3d_verts),
               #                 sweep_radius=0.1),
               # dict(shape='ConvexPolygon',
               #      params=dict(vertices=_2d_verts)),
               # dict(shape='SimplePolygon',
               #      params=dict(vertices=_2d_verts)),
               # dict(shape='ConvexSpheropolygon',
               #      params=dict(vertices=_2d_verts),
               #      sweep_radius=0.1)]

# integrators = [dict(shape='Sphere',
#                     params=dict(diameter=1)),
#                dict(shape='ConvexPolyhedron',
#                     params=dict(vertices=_3d_verts)),
#                dict(shape='ConvexSpheropolyhedron',
#                     params=dict(vertices=_3d_verts),
#                                 sweep_radius=0.1),
#                dict(shape='Ellipsoid',
#                     params=dict(a=0.5, b=0.25, c=0.125)),
#                dict(shape='ConvexSpheropolyhedron',
#                     params=dict(vertices=_3d_verts),
#                                sweep_radius=0.1),
#                dict(shape='ConvexPolygon',
#                     params=dict(vertices=_2d_verts)),
#                dict(shape='SimplePolygon',
#                     params=dict(vertices=_2d_verts)),
#                dict(shape='ConvexSpheropolygon',
#                     params=dict(vertices=_2d_verts),
#                     sweep_radius=0.1)]

               # 'FacetedEllipsoid',
               # 'SphereUnion', 'ConvexPolyhedronUnion',
               # 'FacetedEllipsoidUnion', 'Polyhedron',
               # 'Sphinx']

# note: The parameterized tests validate parameters so we can't pass in values
# here that require preprocessing
valid_constructor_args = [
    dict(trigger=hoomd.trigger.Periodic(10),
         swap_type_pair=[],
         move_ratio=0.1,
         flip_probability=0.8,
         swap_move_ratio=0.1,
         seed=1),
    dict(trigger=hoomd.trigger.After(100),
         swap_type_pair=['A', 'B'],
         move_ratio=0.7,
         flip_probability=1,
         swap_move_ratio=0.1,
         seed=4),
    dict(trigger=hoomd.trigger.Before(100),
         swap_type_pair=[],
         move_ratio=0.7,
         flip_probability=1,
         swap_move_ratio=0.1,
         seed=4),
    dict(trigger=hoomd.trigger.Periodic(1000),
         swap_type_pair=['A', 'B'],
         move_ratio=0.7,
         flip_probability=1,
         swap_move_ratio=0.1,
         seed=4),
]

valid_attrs = [
    ('trigger', hoomd.trigger.Periodic(10000)),
    ('trigger', hoomd.trigger.After(100)),
    ('trigger', hoomd.trigger.Before(12345)),
    ('swap_type_pair', []),
    ('swap_type_pair', ['A', 'B']),
    ('flip_probability', 0.2),
    ('flip_probability', 0.5),
    ('flip_probability', 0.8),
    ('move_ratio', 0.2),
    ('move_ratio', 0.5),
    ('move_ratio', 0.8),
    ('swap_move_ratio', 0.2),
    ('swap_move_ratio', 0.5),
    ('swap_move_ratio', 0.8),
]


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction(constructor_args):
    """Test that Clusters can be constructed with valid arguments."""
    cl = hoomd.hpmc.update.Clusters(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(cl, attr) == value


@pytest.mark.parametrize("integrator,params", integrators)
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_and_attach(simulation_factory,
                                       two_particle_snapshot_factory,
                                       constructor_args,
                                       integrator, params):
    """Test that Clusters can be attached with valid arguments."""
    cl = hoomd.hpmc.update.Clusters(**constructor_args)

    sim = simulation_factory(two_particle_snapshot_factory(particle_types=['A', 'B']))
    sim.operations.updaters.append(cl)

    # Clusters requires an HPMC integrator
    mc = hoomd.hpmc.integrate.__dict__[integrator](seed=1)
    mc.shape['A'] = params
    mc.shape['B'] = params
    sim.operations.integrator = mc

    sim.operations._schedule()

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(cl, attr) == value


@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr(attr, value):
    """Test that Clusters can get and set attributes."""
    cl = hoomd.hpmc.update.Clusters(trigger=hoomd.trigger.Periodic(10),
                                    swap_type_pair=['A', 'B'],
                                    seed=1)

    setattr(cl, attr, value)
    assert getattr(cl, attr) == value


@pytest.mark.parametrize("integrator,params", integrators)
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_attached(attr, value, integrator, params,
                                simulation_factory,
                                two_particle_snapshot_factory):
    """Test that Clusters can get and set attributes while attached."""
    cl = hoomd.hpmc.update.Clusters(trigger=hoomd.trigger.Periodic(10),
                                    swap_type_pair=['A', 'B'],
                                    seed=1)

    sim = simulation_factory(two_particle_snapshot_factory(particle_types=['A', 'B']))
    sim.operations.updaters.append(cl)

    # Clusters requires an HPMC integrator
    mc = hoomd.hpmc.integrate.__dict__[integrator](seed=1)
    mc.shape['A'] = params
    mc.shape['B'] = params
    sim.operations.integrator = mc

    sim.operations._schedule()

    setattr(cl, attr, value)
    assert getattr(cl, attr) == value
