# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

import hoomd
import hoomd.conftest
from hoomd import md


class ConservationDataStorer(hoomd.custom.Action):
    """Store energy and velocities from a simulation."""

    def __init__(self, sim, thermo):
        self._sim = sim
        self._thermo = thermo
        self._velocities = []
        self._energies = []

    def act(self, timestep):
        with self._sim.state.cpu_local_snapshot as snap:
            self._energies.append(self._thermo.potential_energy
                                  + self._thermo.kinetic_energy)
            self._velocities.append(np.array(snap.particles.velocity,
                                             copy=True))

    @property
    def velocities(self):
        return np.array(self._velocities)

    @property
    def total_energies(self):
        return np.array(self._energies)


@pytest.mark.validate
def test_conservation(simulation_factory, lattice_snapshot_factory):
    # For test, use a unit area hexagon.
    coxeter = pytest.importorskip("coxeter")

    particle_vertices = np.array([[6.20403239e-01, 0.00000000e+00],
                                  [3.10201620e-01, 5.37284966e-01],
                                  [-3.10201620e-01, 5.37284966e-01],
                                  [-6.20403239e-01, 7.59774841e-17],
                                  [-3.10201620e-01, -5.37284966e-01],
                                  [3.10201620e-01, -5.37284966e-01]])
    hexagon = coxeter.shapes.ConvexPolygon(particle_vertices)

    circumcircle_diameter = 2 * hexagon.circumcircle_radius

    # Just initialize in a simple cubic lattice.
    sim = simulation_factory(
        lattice_snapshot_factory(a=4 * circumcircle_diameter,
                                 n=10,
                                 dimensions=2))
    sim.seed = 175

    # Initialize moments of inertia since original simulation was HPMC.
    mass = hexagon.area
    # https://math.stackexchange.com/questions/2004798/moment-of-inertia-for-a-n-sided-regular-polygon # noqa
    moment_inertia = (mass * circumcircle_diameter**2 / 6)
    moment_inertia *= (1 + 2 * np.cos(np.pi / hexagon.num_vertices)**2)

    with sim.state.cpu_local_snapshot as snapshot:
        snapshot.particles.mass[:] = mass
        snapshot.particles.moment_inertia[:] = np.array([0, 0, moment_inertia])
        # Not sure if this should be incircle or circumcircle;
        # probably doesn't matter based on current usage, but may
        # matter in the future for the potential if it's modified
        # to actually use diameter.
        snapshot.particles.diameter[:] = circumcircle_diameter
    kT = 0.3
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT)

    # Create box resize updater
    packing_fraction = 0.4
    final_area = hexagon.area * sim.state.N_particles / packing_fraction
    L_final = np.sqrt(final_area)
    final_box = hoomd.Box.square(L_final)

    n_compression_start = int(1e4)
    n_compression_end = int(1e5)
    n_compression_total = n_compression_end - n_compression_start

    box_resize = hoomd.update.BoxResize(
        box1=sim.state.box,
        box2=final_box,
        trigger=int(n_compression_total / 10000),
        variant=hoomd.variant.Ramp(0, 1, n_compression_start,
                                   n_compression_total),
        filter=hoomd.filter.All())
    sim.operations += box_resize

    # Define forces and methods
    r_cut_scale = 1.3
    kernel_scale = (1 / np.cos(np.pi / hexagon.num_vertices))
    incircle_diameter = 2 * hexagon.incircle_radius
    r_cut_set = incircle_diameter * kernel_scale * r_cut_scale

    alj = md.pair.aniso.ALJ(default_r_cut=r_cut_set, nlist=md.nlist.Cell(0.4))

    alj.shape["A"] = {
        "vertices": hexagon.vertices,
        "faces": [],
        "rounding_radii": 0
    }

    alpha = 0  # Make it WCA-only (no attraction)
    eps_att = 1.0
    alj.params[("A", "A")] = {
        "epsilon": eps_att,
        "sigma_i": incircle_diameter,
        "sigma_j": incircle_diameter,
        "alpha": alpha
    }

    nve = md.methods.NVE(filter=hoomd.filter.All())
    integrator = md.Integrator(dt=1e-4,
                               forces=[alj],
                               methods=[nve],
                               integrate_rotational_dof=True)
    sim.operations.integrator = integrator

    # Compress box
    sim.run(n_compression_end)

    thermo = md.compute.ThermodynamicQuantities(hoomd.filter.All())
    sim.operations += thermo

    # Reset velocities after the compression, and equilibriate
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT)
    sim.run(1000)

    # run sim and get values back
    storer = ConservationDataStorer(sim, thermo)
    writer = hoomd.write.CustomWriter(action=storer, trigger=1)
    sim.operations.writers.append(writer)
    sim.run(1000)
    total_energies = storer.total_energies
    velocities = storer.velocities

    # Ensure energy conservation up to the 3 digit per-particle.
    assert np.std(total_energies) / sim.state.N_particles < 1e-3

    # Test momentum conservation.
    assert np.std(np.linalg.norm(np.sum(velocities, axis=1), axis=-1)) < 1e-6


def test_type_shapes(simulation_factory, two_particle_snapshot_factory):
    pytest.importorskip("coxeter")
    alj = md.pair.aniso.ALJ(md.nlist.Cell(buffer=0.1))
    sim = simulation_factory(two_particle_snapshot_factory(d=2.0))
    sim.operations.integrator = md.Integrator(0.005, forces=[alj])

    alj.r_cut.default = 2.5
    octahedron = [(0.5, 0, 0), (-0.5, 0, 0), (0, 0.5, 0), (0, -0.5, 0),
                  (0, 0, 0.5), (0, 0, -0.5)]
    faces = md.pair.aniso.ALJ.get_ordered_vertices(octahedron)[1]
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
