# Copyright (c) 2009-2026 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy
import pytest

import hoomd

rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))

_methods = [
    pytest.param(
        lambda: hoomd.md.methods.Brownian(rigid_centers_and_free, 1.5), id="Brownain"
    ),
    pytest.param(
        lambda: hoomd.md.methods.ConstantVolume(rigid_centers_and_free),
        id="ConstantVolume",
    ),
    pytest.param(
        lambda: hoomd.md.methods.ConstantPressure(
            rigid_centers_and_free, 0.1, 1.0, "xy"
        ),
        id="ConstantPressure",
    ),
    pytest.param(
        lambda: hoomd.md.methods.DisplacementCapped(rigid_centers_and_free, 1.5),
        id="DisplacementCapped",
    ),
    pytest.param(
        lambda: hoomd.md.methods.Langevin(rigid_centers_and_free, 1.5), id="Langevin"
    ),
    pytest.param(
        lambda: hoomd.md.methods.OverdampedViscous(
            rigid_centers_and_free, default_gamma=0.01, default_gamma_r=(0.1, 0.1, 0.1)
        ),
        id="OverdampedViscous",
    ),
]


@pytest.mark.serial
@pytest.mark.parametrize("create_method", _methods)
def test_2d_rigid(simulation_factory, create_method):
    # Test case from: https://github.com/glotzerlab/hoomd-blue/discussions/2299
    dimer_positions = [[-1.2, 0, 0], [1.2, 0, 0]]
    moment_of_inertia = [0, 2.8, 2.8]

    initial_snapshot = hoomd.Snapshot()
    initial_snapshot.particles.types = ["dimer", "A"]
    initial_snapshot.particles.N = 3
    initial_snapshot.particles.position[:] = [
        [-2.5, -2.5, 0],
        [-2.5, 2.5, 0],
        [2.5, 2.5, 0],
    ]
    initial_snapshot.configuration.box = [10, 10, 0, 0, 0, 0]
    initial_snapshot.particles.mass[:] = [2] * 3
    initial_snapshot.particles.moment_inertia[:] = [moment_of_inertia] * 3

    simulation_1 = simulation_factory(initial_snapshot)

    rigid_1 = hoomd.md.constrain.Rigid()
    rigid_1.body["dimer"] = {
        "constituent_types": ["A", "A"],
        "positions": dimer_positions,
        "orientations": [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
    }

    rigid_1.create_bodies(simulation_1.state)

    integrator_1 = hoomd.md.Integrator(dt=0.005, integrate_rotational_dof=True)
    simulation_1.operations.integrator = integrator_1
    integrator_1.rigid = rigid_1

    integrator_1.methods.append(create_method())

    cell_1 = hoomd.md.nlist.Cell(buffer=0, exclusions=["body"])
    lj_1 = hoomd.md.pair.LJ(nlist=cell_1)
    lj_1.params[("A", "A")] = dict(epsilon=1, sigma=2)
    lj_1.r_cut[("A", "A")] = 2.5
    lj_1.params[("dimer", "dimer"), ("dimer", "A")] = dict(epsilon=0, sigma=0)
    lj_1.r_cut[("dimer", "dimer"), ("dimer", "A")] = 0
    integrator_1.forces.append(lj_1)

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    simulation_1.operations.add(thermo)

    simulation_1.state.thermalize_particle_momenta(
        filter=rigid_centers_and_free, kT=1.5
    )
    simulation_1.run(1000)

    snapshot_1 = simulation_1.state.get_snapshot()

    numpy.testing.assert_array_equal(snapshot_1.particles.position[:, 2], [0.0] * 9)
    numpy.testing.assert_array_equal(snapshot_1.particles.velocity[:, 2], [0.0] * 9)
    assert thermo.rotational_degrees_of_freedom == 3.0

    simulation_2 = simulation_factory(snapshot_1)

    rigid_2 = hoomd.md.constrain.Rigid()
    rigid_2.body["dimer"] = {
        "constituent_types": ["A", "A"],
        "positions": dimer_positions,
        "orientations": [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
    }

    integrator_2 = hoomd.md.Integrator(dt=0.005, integrate_rotational_dof=True)
    simulation_2.operations.integrator = integrator_2
    integrator_2.rigid = rigid_2

    integrator_2.methods.append(create_method())

    cell_2 = hoomd.md.nlist.Cell(buffer=0, exclusions=["body"])
    lj_2 = hoomd.md.pair.LJ(nlist=cell_2)
    lj_2.params[("A", "A")] = dict(epsilon=1, sigma=2)
    lj_2.r_cut[("A", "A")] = 2.5
    lj_2.params[("dimer", "dimer"), ("dimer", "A")] = dict(epsilon=0, sigma=0)
    lj_2.r_cut[("dimer", "dimer"), ("dimer", "A")] = 0
    integrator_2.forces.append(lj_2)

    simulation_2.run(100)
    snapshot_2 = simulation_2.state.get_snapshot()

    numpy.testing.assert_array_equal(snapshot_2.particles.position[:, 2], [0.0] * 9)
    numpy.testing.assert_array_equal(snapshot_2.particles.velocity[:, 2], [0.0] * 9)
