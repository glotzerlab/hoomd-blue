# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

import hoomd
import hoomd.conftest
from hoomd import md


def test_type_shapes(simulation_factory, two_particle_snapshot_factory):
    alj = md.pair.aniso.ALJ(md.nlist.Cell(buffer=0.1))
    sim = simulation_factory(two_particle_snapshot_factory(d=2.0))
    sim.operations.integrator = md.Integrator(0.005, forces=[alj])

    alj.r_cut.default = 2.5
    octahedron = [(0.5, 0, 0), (-0.5, 0, 0), (0, 0.5, 0), (0, -0.5, 0),
                  (0, 0, 0.5), (0, 0, -0.5)]
    faces = [[5, 3, 1], [0, 3, 5], [1, 3, 4], [4, 3, 0], [5, 2, 0], [1, 2, 5],
             [0, 2, 4], [4, 2, 1]]
    rounding_radius = 0.1
    alj.shape["A"] = {
        "vertices": octahedron,
        "faces": faces,
        "rounding_radii": rounding_radius
    }
    # We use a non-zero sigma_i to ensure that it is added appropriately to the
    # rounding radius.
    alj.params[("A", "A")] = {
        "epsilon": 1.0,
        "sigma_i": 0.1,
        "sigma_j": 0.1,
        "alpha": 1
    }
    with pytest.raises(hoomd.error.DataAccessError):
        alj.type_shapes

    def get_rounding_radius(base, param_spec):
        modification = param_spec["sigma_i"] * param_spec["contact_ratio_i"]
        return rounding_radius + modification / 2

    sim.run(0)
    shape_spec = alj.type_shapes
    assert len(shape_spec) == 1
    shape_spec = shape_spec[0]
    assert shape_spec["type"] == "ConvexPolyhedron"
    assert np.allclose(shape_spec["vertices"], octahedron)
    assert np.isclose(
        shape_spec["rounding_radius"],
        get_rounding_radius(rounding_radius, alj.params[("A", "A")]))

    ellipse_axes = (0.1, 0.2, 0.3)
    alj.shape["A"] = {
        "vertices": [],
        "faces": [],
        "rounding_radii": ellipse_axes
    }
    shape_spec = alj.type_shapes
    assert len(shape_spec) == 1
    shape_spec = shape_spec[0]
    assert shape_spec["type"] == "Ellipsoid"
    assert np.isclose(
        shape_spec["a"],
        get_rounding_radius(ellipse_axes[0], alj.params[("A", "A")]))
    assert np.isclose(
        shape_spec["a"],
        get_rounding_radius(ellipse_axes[1], alj.params[("A", "A")]))
    assert np.isclose(
        shape_spec["a"],
        get_rounding_radius(ellipse_axes[2], alj.params[("A", "A")]))

    sim.operations.integrator.forces.remove(alj)

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=2, d=2))
    sim.operations.integrator = md.Integrator(0.005, forces=[alj])
    square = [(0.5, 0, 0), (-0.5, 0, 0), (-0.5, -0.5, 0), (0.5, 0.5, 0)]
    alj.shape["A"] = {
        "vertices": square,
        "faces": [],
        "rounding_radii": rounding_radius
    }

    sim.run(0)
    shape_spec = alj.type_shapes
    assert len(shape_spec) == 1
    shape_spec = shape_spec[0]
    assert shape_spec["type"] == "Polygon"
    assert np.allclose(shape_spec["vertices"], np.array(square)[:, :2])
    assert np.isclose(
        shape_spec["rounding_radius"],
        get_rounding_radius(rounding_radius, alj.params[("A", "A")]))

    alj.shape["A"] = {
        "vertices": [],
        "faces": [],
        "rounding_radii": ellipse_axes
    }
    shape_spec = alj.type_shapes
    assert len(shape_spec) == 1
    shape_spec = shape_spec[0]
    assert shape_spec["type"] == "Ellipsoid"
    assert np.isclose(
        shape_spec["a"],
        get_rounding_radius(ellipse_axes[0], alj.params[("A", "A")]))
    assert np.isclose(
        shape_spec["a"],
        get_rounding_radius(ellipse_axes[1], alj.params[("A", "A")]))
