import numpy as np
import pytest

import hoomd
from hoomd import md


@pytest.mark.validation
def test_conservation(simulation_factory, lattice_snapshot_factory):
    # For test, use a unit area hexagon.
    coxeter = pytest.importorskip("coxeter",
                                  "coxeter is required for this test.")

    particle_vertices = np.array([[6.20403239e-01, 0.00000000e+00],
                                  [3.10201620e-01, 5.37284966e-01],
                                  [-3.10201620e-01, 5.37284966e-01],
                                  [-6.20403239e-01, 7.59774841e-17],
                                  [-3.10201620e-01, -5.37284966e-01],
                                  [3.10201620e-01, -5.37284966e-01]])
    hexagon = coxeter.shape.ConvexPolygon(particle_vertices)

    circumcircle_diameter = 2 * hexagon.circumcircle_radius

    # Just initialize in a simple cubic lattice.
    sim = simulation_factory(
        lattice_snapshot_factory(a=4 * circumcircle_diameter,
                                 n=10,
                                 dimensions=2))

    # Initialize moments of inertia since original simulation was HPMC.
    mass = hexagon.area
    # https://math.stackexchange.com/questions/2004798/moment-of-inertia-for-a-n-sided-regular-polygon # noqa
    moment_inertia = ((mass * circumcircle_diameter**2 / 6) *
                      (1 + 2 * np.cos(np.pi / hexagon.num_vertices)**2))
    with sim.state.cpu_local_snapshot as snapshot:
        snapshot.particles.mass[:] = mass
        snapshot.particles.moment_inertia[:] = np.array([0, 0, moment_inertia])
        # Not sure if this should be incircle or circumcircle;
        # probably doesn't matter based on current usage, but may
        # matter in the future for the potential if it's modified
        # to actually use diameter.
        snapshot.particles.diameter[:] = circumcircle_diameter
    kT = 0.3
    sim.state.thermalize_particles(kT, 43)

    # Create box resize updater
    packing_fraction = 0.4
    final_area = hexagon.area * sim.state.N_particles / packing_fraction
    L_final = np.sqrt(final_area)
    final_box = hoomd.box.square(L_final)

    n_compression_start = 1e4
    n_compression_end = 1e5
    n_compression_total = n_compression_end - n_compression_start
    n_total = 1e6

    box_resize = hoomd.update.BoxResize(
        box1=sim.state.box,
        box2=final_box,
        trigger=int(n_compression_total / 10000),
        variant=hoomd.variant.Ramp(0, 1, n_compression_start,
                                   n_compression_total),
        scale_particles=hoomd.filter.All())
    sim.operations += box_resize

    # Define forces and methods
    r_cut_scale = 1.3
    kernel_scale = 2 * (1 / np.cos(np.pi / hexagon.num_vertices))
    incircle_diameter = hexagon.incircle_radius
    r_cut_set = incircle_diameter * kernel_scale * r_cut_scale

    alj = md.pair.aniso.ALJ(default_r_cut=r_cut_set, nlist=md.nlist.Cell())

    alj.shape["A"] = {
        "vertices": hexagon.vertices,
        "faces": [],
        "rounding_radius": 0
    }

    alpha = 0  # Make it WCA-only (no attraction)
    eps_att = 1.0
    alj.params[("A", "A")] = {
        "epsilon": eps_att,
        "sigma_i": incircle_diameter,
        "sigma_j": incircle_diameter,
        "alpha": alpha
    }

    integrator = md.Integrator(dt=1e-4, aniso=True)
    nve = md.methods.NVE(filter=hoomd.filter.All())
    integrator.methods.append(nve)

    # Compress box
    sim.run(n_compression_end)

    thermo = md.compute.ThermodynamicQuantities(hoomd.filter.All())
    sim.operations += thermo

    # Reset velocities after the compression, and equilibriate
    sim.state.thermalize_particles(kT, 43)
    velocities = []
    total_energies = []
    while sim.timestep < n_total:
        sim.run(1_000)
        with sim.state.cpu_local_snapshot as snapshot:
            velocities.append(np.array(snapshot.particles.velocity, copy=True))
            total_energies.append(thermo.kinetic_energy
                                  + thermo.potential_energy)

    # Ensure energy conservation up to the 4 digit per-particle.
    assert np.std(total_energies) / sim.state.N_particles < 1e-4

    # Test momentum conservation.
    velocities = np.asarray(velocities)
    assert np.std(np.linalg.norm(np.sum(velocities, axis=1), axis=-1)) < 1e-6
