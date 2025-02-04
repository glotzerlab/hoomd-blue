# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test HPMC with scaled shapes."""

import hoomd
import pytest
import math


@pytest.mark.parametrize("scale", [1e-9, 1, 1000, 1e9])
@pytest.mark.parametrize("shape", ['ConvexPolygon', 'SimplePolygon'])
@pytest.mark.parametrize("offset", [-100, -10, -1, 1, 10, 100])
@pytest.mark.serial
@pytest.mark.cpu
def test_polygon(scale, shape, offset, simulation_factory,
                 two_particle_snapshot_factory):
    """Test polygons at a variety of scales."""
    # make a many sided polygon to ensure that the overlap check is non-trivial
    a = 0.5 * scale
    vertices = []
    n_points = 12
    for i in range(12):
        theta = 2 * math.pi / n_points * i
        vertices.append([math.cos(theta) * a, math.sin(theta) * a])

    epsilon = 1e-5
    d = scale * (1 + offset * epsilon)
    initial_snap = two_particle_snapshot_factory(dimensions=2, d=d, L=scale * 3)
    initial_snap.particles.position[:, 1] = 0
    sim = simulation_factory(initial_snap)

    mc = getattr(hoomd.hpmc.integrate, shape)(default_d=0)
    mc.shape['A'] = dict(vertices=vertices)
    sim.operations.integrator = mc
    sim.run(0)

    if offset < 0:
        assert mc.overlaps == 1
    else:
        assert mc.overlaps == 0


@pytest.mark.parametrize("scale", [1e-9, 1, 1000, 1e9])
@pytest.mark.parametrize("offset", [-100, -10, -1, 1, 10, 100])
@pytest.mark.serial
@pytest.mark.cpu
def test_convex_polyhedron(scale, offset, simulation_factory,
                           two_particle_snapshot_factory):
    """Test convex polyhedrons at a variety of scales."""
    # make a many sized prism to ensure that the overlap check is non-trivial
    a = 0.5 * scale
    vertices = []
    n_points = 12
    for i in range(12):
        theta = 2 * math.pi / n_points * i
        vertices.append([math.cos(theta) * a, math.sin(theta) * a, -a])
        vertices.append([math.cos(theta) * a, math.sin(theta) * a, a])

    epsilon = 1e-5
    d = scale * (1 + offset * epsilon)
    initial_snap = two_particle_snapshot_factory(dimensions=3, d=d, L=scale * 3)
    initial_snap.particles.position[:, 2] = 0
    sim = simulation_factory(initial_snap)

    mc = hoomd.hpmc.integrate.ConvexPolyhedron(default_d=0)
    mc.shape['A'] = dict(vertices=vertices)
    sim.operations.integrator = mc
    sim.run(0)

    if offset < 0:
        assert mc.overlaps == 1
    else:
        assert mc.overlaps == 0
