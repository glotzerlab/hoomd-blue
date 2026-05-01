# Copyright (c) 2009-2026 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
import numpy

TOLERANCES = {"rtol": 1e-2, "atol": 1e-5}

friction_test_parameters = [
    # test for LJLinear friction model
    (
        hoomd.md.pair.friction.FrictionLJLinear,
        {},
        {
            "epsilon": 1,
            "sigma": 1,
            "gamma_f": 6,
            "kT": 1,
        },
        [[-0.35, -0.35, -0.1], [0.35, 0.35, 0.1]],
        [[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        [
            [0.19503148, -0.97515738, 0.0390063, 0.09751574],
            [0.81649658, 0.0, -0.40824829, -0.40824829],
        ],
        [38.39953386, -152.42732975, 299.06283406],
        [118.73327022, -99.83859521, -66.13136255],
    ),
    # test for the LJCoulomb friction model
    (
        hoomd.md.pair.friction.FrictionLJCoulomb,
        {},
        {
            "epsilon": 1,
            "sigma": 1,
            "kappa_f": 5,
            "kT": 1,
        },
        [[-0.35, -0.35, -0.1], [0.35, 0.35, 0.1]],
        [[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        [
            [0.19503148, -0.97515738, 0.0390063, 0.09751574],
            [0.81649658, 0.0, -0.40824829, -0.40824829],
        ],
        [1.57890627, -54.46188002, 85.05595653],
        [34.8688109, -29.31994622, -19.4210264],
    ),
    # test for the LJCoulombNewton friction model
    (
        hoomd.md.pair.friction.FrictionLJCoulombNewton,
        {},
        {
            "epsilon": 1,
            "sigma": 1,
            "gamma_f": 6,
            "kappa_f": 5,
            "kT": 1,
        },
        [[-0.35, -0.35, -0.1], [0.35, 0.35, 0.1]],
        [[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        [
            [0.19503148, -0.97515738, 0.0390063, 0.09751574],
            [0.81649658, 0.0, -0.40824829, -0.40824829],
        ],
        [-11.09870197, -20.73166752, 11.37184162],
        [5.99367133, -5.03986561, -3.33832001],
    ),
]


@pytest.fixture(scope="session")
def friction_snapshot_factory(device):
    def make_snapshot(
        position_i=numpy.array([[]]),
        position_j=numpy.array([]),
        velocity_i=numpy.array([]),
        velocity_j=numpy.array([]),
        angmom_i=numpy.array([]),
        angmom_j=numpy.array([]),
        orientation_i=(),
        orientation_j=(),
        dimensions=3,
        L=20,
    ):
        snapshot = hoomd.Snapshot(device.communicator)
        if snapshot.communicator.rank == 0:
            N = 2
            box = [L, L, L, 0, 0, 0]
            if dimensions == 2:
                box[2] = 0
            snapshot.configuration.box = box
            snapshot.particles.N = N
            snapshot.particles.diameter[:] = [1] * N
            snapshot.particles.mass[:] = [1] * N
            snapshot.particles.position[:] = [position_i, position_j]
            snapshot.particles.velocity[:] = [velocity_i, velocity_j]
            snapshot.particles.angmom[:] = [angmom_i, angmom_j]
            snapshot.particles.orientation[:] = [orientation_i, orientation_j]
            snapshot.particles.types = ["A", "B"]
            snapshot.particles.typeid[:] = [0, 1]
            snapshot.particles.moment_inertia[:] = [(0.1, 0.1, 0.1)] * N

        return snapshot

    return make_snapshot


@pytest.mark.parametrize(
    "friction_cls, friction_args, params, positions,"
    "velocities, angmom, orientations, force, torque",
    friction_test_parameters,
)
def test_params(
    friction_cls,
    friction_args,
    params,
    positions,
    velocities,
    angmom,
    orientations,
    force,
    torque,
):
    rcut = 2.0 ** (1.0 / 6.0) * params["sigma"]
    friction = friction_cls(
        nlist=hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=rcut, **friction_args
    )

    friction.params.default = params
    for key in params:
        assert friction.params[("A", "A")][key] == pytest.approx(params[key])
        assert friction.params[("A", "B")][key] == pytest.approx(params[key])
        assert friction.params[("B", "B")][key] == pytest.approx(params[key])


# Test the only the deterministic forces and torques (zero temperature)
@pytest.mark.parametrize(
    "friction_cls, friction_args, params, positions,"
    "velocities, angmom, orientations, force, torque",
    friction_test_parameters,
)
def test_deterministic_forces_torques(
    friction_snapshot_factory,
    simulation_factory,
    friction_cls,
    friction_args,
    params,
    positions,
    velocities,
    angmom,
    orientations,
    force,
    torque,
):
    snapshot = friction_snapshot_factory(
        position_i=positions[0],
        position_j=positions[1],
        velocity_i=velocities[0],
        velocity_j=velocities[1],
        angmom_i=angmom[0],
        angmom_j=angmom[1],
        orientation_i=orientations[0],
        orientation_j=orientations[1],
    )
    sim = simulation_factory(snapshot)
    rcut = 2.0 ** (1.0 / 6.0) * params["sigma"]
    friction = friction_cls(
        nlist=hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=rcut, **friction_args
    )
    # Set temperature to zero
    params["kT"] = 0
    friction.params.default = params

    sim.operations.integrator = hoomd.md.Integrator(
        dt=0.001, forces=[friction], integrate_rotational_dof=True
    )
    sim.run(0)

    sim_forces = friction.forces
    sim_torques = friction.torques
    if sim.device.communicator.rank == 0:
        sim.orientations = snapshot.particles.orientation

        numpy.testing.assert_allclose(sim_forces[0], force, **TOLERANCES)

        numpy.testing.assert_allclose(
            sim_forces[1], [-force[0], -force[1], -force[2]], **TOLERANCES
        )

        numpy.testing.assert_allclose(sim_torques[0], torque, **TOLERANCES)

        numpy.testing.assert_allclose(sim_torques[1], torque, **TOLERANCES)


# Test the frictional model for some timesteps (nonzero temperature)
@pytest.mark.parametrize(
    "friction_cls, friction_args, params, positions,"
    "velocities, angmom, orientations, force, torque",
    friction_test_parameters,
)
def test_forces_torques(
    friction_snapshot_factory,
    simulation_factory,
    friction_cls,
    friction_args,
    params,
    positions,
    velocities,
    angmom,
    orientations,
    force,
    torque,
):
    snapshot = friction_snapshot_factory(
        position_i=positions[0],
        position_j=positions[1],
        velocity_i=velocities[0],
        velocity_j=velocities[1],
        angmom_i=angmom[0],
        angmom_j=angmom[1],
        orientation_i=orientations[0],
        orientation_j=orientations[1],
    )
    sim = simulation_factory(snapshot)
    rcut = 2.0 ** (1.0 / 6.0) * params["sigma"]
    friction = friction_cls(
        nlist=hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=rcut, **friction_args
    )
    friction.params.default = params
    sim.operations.integrator = hoomd.md.Integrator(
        dt=0.001, forces=[friction], integrate_rotational_dof=True
    )
    # Run some timesteps
    sim.run(5)
    sim_forces = friction.forces
    sim_torques = friction.torques
    if sim.device.communicator.rank == 0:
        # check nonzero force and torques
        for x in sim_forces[0]:
            assert x != 0
        for x in sim_forces[1]:
            assert x != 0
        for x in sim_torques[0]:
            assert x != 0
        for x in sim_torques[1]:
            assert x != 0
