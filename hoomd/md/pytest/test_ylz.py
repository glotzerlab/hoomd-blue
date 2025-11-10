# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
import numpy

TOLERANCES = {"rtol": 1e-2, "atol": 1e-5}

ylz_test_parameters = [
    (
        # test for mu equals (1,0,0) at positions less than rmin
        hoomd.md.pair.aniso.YLZ,
        {},
        {
            "eps": 0.34512,
            "phi": 0.314159265358979,
            "beta": 1.774532,
            "rmin": 1.122462048309373,
            "twozeta": 8,
        },
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        [[0.0, 0.0, 0.36123102], [0.0, 0.0, -0.56123102]],
        [
            [0.19503148, -0.97515738, 0.0390063, 0.09751574],
            [0.81649658, 0.0, -0.40824829, -0.40824829],
        ],
        [-0.63895967, 0.11570992, 1.06497178],
        0.5592034600953706,
        [
            [0.75155544, 1.16806815, 0.35549139],
            [-0.64481743, -0.5786521, -0.35549139],
        ],
    ),
    # test for the same mu at positions less than rmin
    (
        hoomd.md.pair.aniso.YLZ,
        {},
        {
            "eps": 0.34512,
            "phi": 0.314159265358979,
            "beta": 1.774532,
            "rmin": 1.122462048309373,
            "twozeta": 8,
        },
        (0.6172134, 0.77151675, 0.15430335),
        (0.6172134, 0.77151675, 0.15430335),
        [[0.0, 0.0, 0.0], [0.64805377, 0.64805377, 0.44805377]],
        [
            [0.19503148, -0.97515738, 0.0390063, 0.09751574],
            [0.81649658, 0.0, -0.40824829, -0.40824829],
        ],
        [1.10203576, -0.66602597, -1.41636802],
        0.07150985496802698,
        [
            [-0.88451758, -0.12648429, -0.62613117],
            [0.26505036, 1.53813829, -0.51966794],
        ],
    ),
    # test for same mu at positions greater than rmin
    (
        hoomd.md.pair.aniso.YLZ,
        {},
        {
            "eps": 0.34512,
            "phi": 0.314159265358979,
            "beta": 1.774532,
            "rmin": 1.122462048309373,
            "twozeta": 8,
        },
        (0.6172134, 0.77151675, 0.15430335),
        (0.6172134, 0.77151675, 0.15430335),
        [[0.0, 0.0, 0.0], [0.64805377, 0.64805377, 0.84805377]],
        [
            [0.19503148, -0.97515738, 0.0390063, 0.09751574],
            [0.81649658, 0.0, -0.40824829, -0.40824829],
        ],
        [0.40294018, -0.08796581, -0.13414379],
        -0.06304066405886016,
        [
            [-0.12743464, 0.02748067, -0.14581856],
            [0.11510198, 0.40116666, -0.17231492],
        ],
    ),
    # test for different mu at positions less than rmin
    (
        hoomd.md.pair.aniso.YLZ,
        {},
        {
            "eps": 0.34512,
            "phi": 0.314159265358979,
            "beta": 1.774532,
            "rmin": 1.122462048309373,
            "twozeta": 8,
        },
        (0.6172134, 0.77151675, 0.15430335),
        (-0.6172134, -0.77151675, 0.15430335),
        [[0.0, 0.0, 0.0], [0.64805377, 0.64805377, 0.44805377]],
        [
            [0.19503148, -0.97515738, 0.0390063, 0.09751574],
            [0.81649658, 0.0, -0.40824829, -0.40824829],
        ],
        [-0.4641542, -0.37992727, 0.43512397],
        0.6391219785568611,
        [
            [0.67477282, 0.80716419, -0.38706132],
            [-0.22256124, -1.29711395, 0.44164489],
        ],
    ),
    # test for different mu at positions greater than rmin
    (
        hoomd.md.pair.aniso.YLZ,
        {},
        {
            "eps": 0.34512,
            "phi": 0.314159265358979,
            "beta": 1.774532,
            "rmin": 1.122462048309373,
            "twozeta": 8,
        },
        (0.6172134, 0.77151675, 0.15430335),
        (-0.6172134, -0.77151675, 0.15430335),
        [[0.0, 0.0, 0.0], [0.64805377, 0.64805377, 0.84805377]],
        [
            [0.19503148, -0.97515738, 0.0390063, 0.09751574],
            [0.81649658, 0.0, -0.40824829, -0.40824829],
        ],
        [-0.40952632, -0.40854062, -0.43412407],
        0.6267230129371016,
        [
            [0.15022915, 0.23237846, -0.15026583],
            [-0.08510048, -0.29834306, 0.15090462],
        ],
    ),
    # test cosine function with exponent 4
    (
        hoomd.md.pair.aniso.YLZ,
        {},
        {
            "eps": 0.34512,
            "phi": 0.314159265358979,
            "beta": 1.774532,
            "rmin": 1.122462048309373,
            "twozeta": 4,
        },
        (0.6172134, 0.77151675, 0.15430335),
        (-0.6172134, -0.77151675, 0.15430335),
        [[0.0, 0.0, 0.0], [0.64805377, 0.64805377, 0.84805377]],
        [
            [0.19503148, -0.97515738, 0.0390063, 0.09751574],
            [0.81649658, 0.0, -0.40824829, -0.40824829],
        ],
        [-0.23102846, -0.23000652, -0.19679651],
        0.649764911188693,
        [
            [0.15575243, 0.24092201, -0.15579045],
            [-0.08822926, -0.30931185, 0.15645273],
        ],
    ),
    # test cosine function with exponent 9
    (
        hoomd.md.pair.aniso.YLZ,
        {},
        {
            "eps": 0.34512,
            "phi": 0.314159265358979,
            "beta": 1.774532,
            "rmin": 1.122462048309373,
            "twozeta": 9,
        },
        (0.6172134, 0.77151675, 0.15430335),
        (-0.6172134, -0.77151675, 0.15430335),
        [[0.0, 0.0, 0.0], [0.64805377, 0.64805377, 0.84805377]],
        [
            [0.19503148, -0.97515738, 0.0390063, 0.09751574],
            [0.81649658, 0.0, -0.40824829, -0.40824829],
        ],
        [-0.45209962, -0.45112277, -0.49075084],
        0.6210913615279404,
        [
            [0.14887922, 0.23029034, -0.14891556],
            [-0.08433578, -0.29566219, 0.14954861],
        ],
    ),
    # test cosine function with exponent 15
    (
        hoomd.md.pair.aniso.YLZ,
        {},
        {
            "eps": 0.34512,
            "phi": 0.314159265358979,
            "beta": 1.774532,
            "rmin": 1.122462048309373,
            "twozeta": 15,
        },
        (0.6172134, 0.77151675, 0.15430335),
        (-0.6172134, -0.77151675, 0.15430335),
        [[0.0, 0.0, 0.0], [0.64805377, 0.64805377, 0.84805377]],
        [
            [0.19503148, -0.97515738, 0.0390063, 0.09751574],
            [0.81649658, 0.0, -0.40824829, -0.40824829],
        ],
        [-0.69115472, -0.69022937, -0.80890019],
        0.5883483906974297,
        [
            [0.14103053, 0.21814979, -0.14106496],
            [-0.07988973, -0.28007534, 0.14166464],
        ],
    ),
    # test cosine function with exponent 20
    (
        hoomd.md.pair.aniso.YLZ,
        {},
        {
            "eps": 0.34512,
            "phi": 0.314159265358979,
            "beta": 1.774532,
            "rmin": 1.122462048309373,
            "twozeta": 20,
        },
        (0.6172134, 0.77151675, 0.15430335),
        (-0.6172134, -0.77151675, 0.15430335),
        [[0.0, 0.0, 0.0], [0.64805377, 0.64805377, 0.84805377]],
        [
            [0.19503148, -0.97515738, 0.0390063, 0.09751574],
            [0.81649658, 0.0, -0.40824829, -0.40824829],
        ],
        [-0.87006137, -0.86917685, -1.04723716],
        0.562385097654067,
        [
            [0.13480698, 0.20852303, -0.13483989],
            [-0.07636427, -0.26771586, 0.1354131],
        ],
    ),
]


@pytest.fixture(scope="session")
def ylz_snapshot_factory(device):
    def make_snapshot(
        position_i=numpy.array([0, 0, 0]),
        position_j=numpy.array([2, 0, 0]),
        orientation_i=(1, 0, 0, 0),
        orientation_j=(1, 0, 0, 0),
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
            snapshot.particles.position[:] = [position_i, position_j]
            snapshot.particles.orientation[:] = [orientation_i, orientation_j]
            snapshot.particles.types = ["A", "B"]
            snapshot.particles.typeid[:] = [0, 1]
            snapshot.particles.moment_inertia[:] = [(1, 1, 1)] * N
            snapshot.particles.angmom[:] = [(0, 0, 0, 0)] * N
        return snapshot

    return make_snapshot


@pytest.mark.parametrize(
    "ylz_cls, ylz_args, params, n_A, n_B, positions,"
    "orientations, force, energy, torques",
    ylz_test_parameters,
)
def test_before_attaching(
    ylz_cls,
    ylz_args,
    params,
    n_A,
    n_B,
    positions,
    orientations,
    force,
    energy,
    torques,
):
    potential = ylz_cls(
        nlist=hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=2.6, **ylz_args
    )
    potential.params.default = params
    potential.mu["A"] = n_A
    potential.mu["B"] = n_B
    for key in params:
        assert potential.params[("A", "A")][key] == pytest.approx(params[key])
        assert potential.params[("A", "B")][key] == pytest.approx(params[key])
        assert potential.params[("B", "B")][key] == pytest.approx(params[key])

    for i in range(len(n_A)):
        assert potential.mu["A"][i] == pytest.approx(n_A[i])
    for i in range(len(n_B)):
        assert potential.mu["B"][i] == pytest.approx(n_B[i])


@pytest.mark.parametrize(
    "ylz_cls, ylz_args, params, n_A, n_B, positions,"
    "orientations, force, energy, torques",
    ylz_test_parameters,
)
def test_after_attaching(
    ylz_snapshot_factory,
    simulation_factory,
    ylz_cls,
    ylz_args,
    params,
    n_A,
    n_B,
    positions,
    orientations,
    force,
    energy,
    torques,
):
    sim = simulation_factory(ylz_snapshot_factory())
    potential = ylz_cls(
        nlist=hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=2.6, **ylz_args
    )
    potential.params.default = params
    potential.mu["A"] = n_A
    potential.mu["B"] = n_B

    sim.operations.integrator = hoomd.md.Integrator(
        dt=0.05, forces=[potential], integrate_rotational_dof=True
    )
    sim.run(0)
    for key in params:
        assert potential.params[("A", "A")][key] == pytest.approx(params[key])
        assert potential.params[("A", "B")][key] == pytest.approx(params[key])
        assert potential.params[("B", "B")][key] == pytest.approx(params[key])

    for i in range(len(n_A)):
        assert potential.mu["A"][i] == pytest.approx(n_A[i])

    for i in range(len(n_B)):
        assert potential.mu["B"][i] == pytest.approx(n_B[i])


@pytest.mark.parametrize(
    "ylz_cls, ylz_args, params, n_A, n_B, positions,"
    "orientations, force, energy, torques",
    ylz_test_parameters,
)
def test_forces_energies_torques(
    ylz_snapshot_factory,
    simulation_factory,
    ylz_cls,
    ylz_args,
    params,
    n_A,
    n_B,
    positions,
    orientations,
    force,
    energy,
    torques,
):
    snapshot = ylz_snapshot_factory(
        position_i=positions[0],
        position_j=positions[1],
        orientation_i=orientations[0],
        orientation_j=orientations[1],
    )
    sim = simulation_factory(snapshot)

    potential = ylz_cls(
        nlist=hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=2.6, **ylz_args
    )
    potential.params.default = params
    potential.mu["A"] = n_A
    potential.mu["B"] = n_B

    sim.operations.integrator = hoomd.md.Integrator(
        dt=0.005, forces=[potential], integrate_rotational_dof=True
    )
    sim.run(0)

    sim_forces = potential.forces
    sim_energy = potential.energy
    sim_torques = potential.torques
    if sim.device.communicator.rank == 0:
        sim_orientations = snapshot.particles.orientation

        numpy.testing.assert_allclose(sim_orientations, orientations, **TOLERANCES)

        numpy.testing.assert_allclose(sim_energy, energy, **TOLERANCES)

        numpy.testing.assert_allclose(sim_forces[0], force, **TOLERANCES)

        numpy.testing.assert_allclose(
            sim_forces[1], [-force[0], -force[1], -force[2]], **TOLERANCES
        )

        numpy.testing.assert_allclose(sim_torques[0], torques[0], **TOLERANCES)

        numpy.testing.assert_allclose(sim_torques[1], torques[1], **TOLERANCES)
