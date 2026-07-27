# Copyright (c) 2009-2026 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

import hoomd
from hoomd.conftest import pickling_check


@pytest.fixture
def small_snap():
    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [20, 20, 20, 0, 0, 0]
        snap.particles.N = 1
        snap.particles.types = ["A"]
        snap.mpcd.N = 1
        snap.mpcd.types = ["A"]
    return snap


def test_cell_list(small_snap, simulation_factory):
    if small_snap.communicator.rank == 0:
        small_snap.configuration.box = [20, 30, 40, 0, 0, 0]
    sim = simulation_factory(small_snap)

    cl = hoomd.mpcd.collide.CellList()

    cl._attach(sim)
    assert cl.num_cells == (20, 30, 40)


@pytest.mark.parametrize(
    "cls, init_args",
    [
        (
            hoomd.mpcd.collide.AndersenThermostat,
            {
                "kT": 1.0,
            },
        ),
        (
            hoomd.mpcd.collide.StochasticRotationDynamics,
            {
                "angle": 90,
            },
        ),
    ],
    ids=["AndersenThermostat", "StochasticRotationDynamics"],
)
class TestCollisionMethod:
    def test_create(self, small_snap, simulation_factory, cls, init_args):
        sim = simulation_factory(small_snap)
        cm = cls(period=5, **init_args)
        ig = hoomd.mpcd.Integrator(dt=0.02, collision_method=cm)
        sim.operations.integrator = ig

        assert ig.collision_method is cm
        assert cm.embedded_particles is None
        assert cm.period == 5
        if "kT" in init_args:
            assert isinstance(cm.kT, hoomd.variant.Constant)
            assert cm.kT(0) == init_args["kT"]

        sim.run(0)
        assert ig.collision_method is cm
        assert cm.embedded_particles is None
        assert cm.period == 5
        if "kT" in init_args:
            assert isinstance(cm.kT, hoomd.variant.Constant)
            assert cm.kT(0) == init_args["kT"]

    def test_pickling(self, small_snap, simulation_factory, cls, init_args):
        cm = cls(period=1, **init_args)
        pickling_check(cm)

        sim = simulation_factory(small_snap)
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02, collision_method=cm)
        sim.run(0)
        pickling_check(cm)

    def test_embed(self, small_snap, simulation_factory, cls, init_args):
        sim = simulation_factory(small_snap)
        cm = cls(period=1, embedded_particles=hoomd.filter.All(), **init_args)
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02, collision_method=cm)

        assert isinstance(cm.embedded_particles, hoomd.filter.All)
        sim.run(0)
        assert isinstance(cm.embedded_particles, hoomd.filter.All)

    def test_temperature(self, small_snap, simulation_factory, cls, init_args):
        sim = simulation_factory(small_snap)
        if "kT" not in init_args:
            init_args["kT"] = 1.0
            kT_required = False
        else:
            kT_required = True
        cm = cls(period=1, **init_args)
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02, collision_method=cm)

        assert isinstance(cm.kT, hoomd.variant.Constant)
        assert cm.kT(0) == 1.0
        sim.run(0)
        assert isinstance(cm.kT, hoomd.variant.Constant)

        ramp = hoomd.variant.Ramp(1.0, 2.0, 0, 10)
        cm.kT = ramp
        assert cm.kT is ramp
        sim.run(0)
        assert cm.kT is ramp

        if not kT_required:
            cm.kT = None
            assert cm.kT is None
            sim.run(0)
            assert cm.kT is None

    def test_run(self, small_snap, simulation_factory, cls, init_args):
        sim = simulation_factory(small_snap)
        cm = cls(period=1, **init_args)
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02, collision_method=cm)

        # test that one step can run without error with only solvent
        sim.run(1)

        # test that one step can run without error with embedded particles
        if "kT" not in init_args:
            init_args["kT"] = 1.0
        sim.operations.integrator.collision_method = cls(
            period=1, embedded_particles=hoomd.filter.All(), **init_args
        )
        sim.run(1)

    @pytest.mark.parametrize(
        "velo_rigid", [[0, 0, 0], [1, 2, 3]], ids=["Stationary", "Moving"]
    )
    @pytest.mark.parametrize(
        "angmom_rigid", [[0, 0, 0, 0], [0, 2, 3, 4]], ids=["Nonrotating", "Rotating"]
    )
    @pytest.mark.parametrize(
        "pos_rigid", [[0, 0, 0], [5, 5, 5]], ids=["center", "edge"]
    )
    @pytest.mark.parametrize(
        "def_rigid,properties_rigid",
        [
            (
                {
                    "constituent_types": ["B", "B"],
                    "positions": [[-2.4, 0, 0], [2.4, 0, 0]],
                    "orientations": [[1, 0, 0, 0], [1, 0, 0, 0]],
                },
                {"inertia": [0.0, 11.52, 11.52], "mass": np.array([2, 1, 1])},
            ),
            (
                {
                    "constituent_types": ["B", "B", "B", "B"],
                    "positions": np.array(
                        [
                            [1, 0, -1 / (2 ** (1.0 / 2.0))],
                            [-1, 0, -1 / (2 ** (1.0 / 2.0))],
                            [0, -1, 1 / (2 ** (1.0 / 2.0))],
                            [0, 1, 1 / (2 ** (1.0 / 2.0))],
                        ]
                    )
                    * 2,
                    "orientations": [(1.0, 0.0, 0.0, 0.0)] * 4,
                },
                {
                    "inertia": [1.0, 1.0, 1.0],
                    "mass": np.array([1 / 4, 1 / 16, 1 / 16, 1 / 16, 1 / 16]),
                },
            ),
            (
                {
                    "constituent_types": ["B", "B", "B", "B"],
                    "positions": np.array(
                        [
                            [1.0, 0.0, -1.0],
                            [-1.0, 0.0, -1.0],
                            [0.0, -1.0, 0.5],
                            [0.0, 1.0, 0.5],
                        ]
                    )
                    * 2,
                    "orientations": [(1.0, 0.0, 0.0, 0.0)] * 4,
                },
                {
                    "inertia": [7.0, 5.0, 6.0],
                    "mass": np.array([1.5, 1 / 4, 1 / 4, 1 / 2, 1 / 2]),
                },
            ),
        ],
        ids=["dimer", "cross", "uneven-mass"],
    )
    @pytest.mark.serial
    def test_rigid_collide(
        self,
        one_particle_snapshot_factory,
        simulation_factory,
        cls,
        init_args,
        velo_rigid,
        angmom_rigid,
        pos_rigid,
        def_rigid,
        properties_rigid,
    ):
        if "kT" not in init_args:
            init_args["kT"] = 1.0
        L = 11  # length of box

        N_mpcd = len(def_rigid["constituent_types"])
        rng = np.random.default_rng(seed=42)
        velo_mpcd = rng.normal(0.0, np.sqrt(init_args["kT"]), (N_mpcd, 3))
        velo_mpcd -= np.mean(velo_mpcd, axis=0)

        # create simulation
        initial_snap = one_particle_snapshot_factory(
            particle_types=["A", "B"], position=pos_rigid, L=L
        )
        total_mass = properties_rigid["mass"][0]
        if initial_snap.communicator.rank == 0:
            initial_snap.particles.moment_inertia[:] = [properties_rigid["inertia"]]
            initial_snap.particles.mass[:] = [total_mass]
            initial_snap.particles.velocity[:] = [velo_rigid]
            initial_snap.particles.angmom[:] = [angmom_rigid]

            # place the mpcd particles on top of constituents, accounting for PBCs
            positions = np.add(def_rigid["positions"], pos_rigid)
            positions[positions < -L * 0.5] = positions[positions < -L * 0.5] + L
            positions[positions > L * 0.5] = positions[positions > L * 0.5] - L
            initial_snap.mpcd.N = N_mpcd
            initial_snap.mpcd.types = ["C"]
            initial_snap.mpcd.position[:] = positions
            initial_snap.mpcd.velocity[:] = velo_mpcd

        sim = simulation_factory(initial_snap)
        sim.seed = 5

        rigid = hoomd.md.constrain.Rigid()
        rigid.body["A"] = def_rigid
        rigid.create_bodies(sim.state, masses={"A": properties_rigid["mass"][1:]})

        sim.operations.integrator = hoomd.mpcd.Integrator(
            dt=0, integrate_rotational_dof=True, rigid=rigid
        )
        sim.operations.integrator.cell_list.shift = False
        sim.operations.integrator.collision_method = cls(
            period=1,
            embedded_particles=hoomd.filter.Rigid(flags=("constituent",)),
            **init_args,
        )

        # run simulation
        sim.run(1)
        new_snap = sim.state.get_snapshot()
        if new_snap.communicator.rank == 0:
            assert np.array_equal(properties_rigid["mass"], new_snap.particles.mass)
            central_flag = new_snap.particles.typeid == new_snap.particles.types.index(
                "A"
            )
            constit_flag = new_snap.particles.typeid == new_snap.particles.types.index(
                "B"
            )
            new_velo_central = new_snap.particles.velocity[central_flag]
            new_velo_constituent = new_snap.particles.velocity[constit_flag]
            new_velo_mpcd = new_snap.mpcd.velocity
            # solve for expected central particle velocity based on linear momentum
            initial_momentum_mpcd = np.sum(velo_mpcd, axis=0)
            initial_momentum_md = np.array(velo_rigid) * total_mass
            initial_momentum = initial_momentum_md + initial_momentum_mpcd

            final_momentum_mpcd = np.sum(new_velo_mpcd * new_snap.mpcd.mass, axis=0)
            final_linmom_md = initial_momentum - final_momentum_mpcd
            final_velocity_md = final_linmom_md / total_mass
            assert np.allclose(final_velocity_md, new_velo_central)

            # solve for expected angular momentum change based on solvent
            # multiply expected change by 2 to get quaternion
            change_momentum_mpcd = (new_velo_mpcd - velo_mpcd) * new_snap.mpcd.mass
            expected_change_angmom_md = np.zeros(4)
            expected_change_angmom_md[1:] = (
                np.sum(
                    np.cross(def_rigid["positions"], -1 * change_momentum_mpcd), axis=0
                )
                * 2
            )
            change_angmom_md = new_snap.particles.angmom[0] - angmom_rigid
            assert np.allclose(expected_change_angmom_md, change_angmom_md)

            # check the constituent velocities match the central particle
            # since orientation is stuck at [1, 0, 0, 0], angular velocity
            # is 0.5 * real part of angmom / inertia.
            new_angmom = new_snap.particles.angmom[0]
            omega = [
                0 if i == 0 else 0.5 * a / i
                for i, a in zip(properties_rigid["inertia"], new_angmom[1:])
            ]
            expected_tangential_velocity = np.cross(omega, def_rigid["positions"])
            expected_velocity = np.add(expected_tangential_velocity, new_velo_central)
            assert np.allclose(new_velo_constituent, expected_velocity)

    @pytest.mark.parametrize(
        "embedded_filter_flags",
        [("B", "D"), ("B", "D", "C"), ("B", "C")],
        ids=["Non-participatory", "Participatory", "Constituent-non-participatory"],
    )
    @pytest.mark.serial
    def test_rigid_collide_free(
        self,
        two_particle_snapshot_factory,
        simulation_factory,
        cls,
        init_args,
        embedded_filter_flags,
    ):
        def_rigid = {
            "constituent_types": ["B", "B", "B", "D"],
            "positions": np.array(
                [
                    [1, 0, -1 / (2 ** (1.0 / 2.0))],
                    [-1, 0, -1 / (2 ** (1.0 / 2.0))],
                    [0, -1, 1 / (2 ** (1.0 / 2.0))],
                    [0, 1, 1 / (2 ** (1.0 / 2.0))],
                ]
            )
            * 2,
            "orientations": [(1.0, 0.0, 0.0, 0.0)] * 4,
        }
        properties_rigid = {
            "inertia": [1.0, 1.0, 1.0],
            "mass": np.array([1 / 4, 1 / 16, 1 / 16, 1 / 16, 1 / 16]),
        }
        if "kT" not in init_args:
            init_args["kT"] = 1.0
        N_mpcd = len(def_rigid["constituent_types"])
        rng = np.random.default_rng(seed=42)
        velo_mpcd = rng.normal(0.0, np.sqrt(init_args["kT"]), (N_mpcd, 3))
        velo_mpcd -= np.mean(velo_mpcd, axis=0)

        # create simulation
        total_mass = properties_rigid["mass"][0]
        initial_snap = two_particle_snapshot_factory(
            particle_types=["A", "B", "C", "D"], L=11
        )
        if initial_snap.communicator.rank == 0:
            # put a free particle that doesn't participate in collision on top of
            # a constituent particle that does
            initial_snap.particles.position[:] = [[0, 0, 0], def_rigid["positions"][0]]
            initial_snap.particles.moment_inertia[:] = [
                properties_rigid["inertia"],
                [1, 1, 1],
            ]
            initial_snap.particles.mass[:] = [total_mass, 1]
            initial_snap.particles.velocity[:] = [[0, 2, 0], [0, 1, 0]]
            initial_snap.particles.angmom[:] = [[0, 0, 0, 0], [0, 0, 0, 0]]
            initial_snap.particles.typeid[:] = [0, 2]

            # place the mpcd particles on top of constituents
            initial_snap.mpcd.N = N_mpcd
            initial_snap.mpcd.types = ["C"]
            initial_snap.mpcd.position[:] = def_rigid["positions"]
            initial_snap.mpcd.velocity[:] = velo_mpcd

        sim = simulation_factory(initial_snap)
        sim.seed = 5

        rigid = hoomd.md.constrain.Rigid()
        rigid.body["A"] = def_rigid
        rigid.create_bodies(sim.state, masses={"A": properties_rigid["mass"][1:]})

        sim.operations.integrator = hoomd.mpcd.Integrator(
            dt=0, integrate_rotational_dof=True, rigid=rigid
        )
        sim.operations.integrator.cell_list.shift = False
        sim.operations.integrator.collision_method = cls(
            period=1,
            embedded_particles=hoomd.filter.Type(embedded_filter_flags),
            **init_args,
        )

        # run simulation
        sim.run(1)
        new_snap = sim.state.get_snapshot()
        if new_snap.communicator.rank == 0:
            # check if the free particle participated in collisions
            participated_flag = "C" in embedded_filter_flags
            free_flag = new_snap.particles.typeid == new_snap.particles.types.index("C")
            assert np.array_equiv(
                new_snap.particles.velocity[free_flag], [0.0, 1.0, 0.0]
            ) == (not participated_flag)
            assert np.array_equiv(
                new_snap.particles.angmom[free_flag], [0.0, 0.0, 0.0, 0.0]
            )

            # get velocities of the different types of particles
            central_flag = new_snap.particles.typeid == new_snap.particles.types.index(
                "A"
            )
            constit_flag = np.logical_or(
                new_snap.particles.typeid == new_snap.particles.types.index("B"),
                new_snap.particles.typeid == new_snap.particles.types.index("D"),
            )

            new_velo_central = new_snap.particles.velocity[central_flag]
            new_velo_constituent = new_snap.particles.velocity[constit_flag]
            new_velo_free = new_snap.particles.velocity[free_flag]
            new_velo_mpcd = new_snap.mpcd.velocity

            # solve for expected central particle velocity based on linear momentum
            initial_momentum_nonrigid = (
                np.sum(velo_mpcd, axis=0) + np.array([0, 1, 0]) * participated_flag
            )
            initial_momentum_md = np.array([0, 2, 0]) * total_mass
            initial_momentum = initial_momentum_md + initial_momentum_nonrigid

            final_momentum_nonrigid = (
                np.sum(new_velo_mpcd * new_snap.mpcd.mass, axis=0)
                + new_velo_free * participated_flag
            )
            final_linmom_md = initial_momentum - final_momentum_nonrigid
            final_velocity_md = final_linmom_md / total_mass
            assert np.allclose(final_velocity_md, new_velo_central)

            # solve for expected angular momentum change based on solvent
            # multiply expected change by 2 to get quaternion
            # the free particle interacts with the first constituent
            change_momentum_nonrigid = (new_velo_mpcd - velo_mpcd) * new_snap.mpcd.mass
            change_momentum_nonrigid[0] += (
                np.squeeze(np.subtract(new_velo_free, [0, 1, 0])) * participated_flag
            )
            expected_change_angmom_md = np.zeros(4)
            expected_change_angmom_md[1:] = (
                np.sum(
                    np.cross(def_rigid["positions"], -1 * change_momentum_nonrigid),
                    axis=0,
                )
                * 2
            )
            change_angmom_md = new_snap.particles.angmom[central_flag] - np.array(
                [0, 0, 0, 0]
            )
            assert np.allclose(expected_change_angmom_md, change_angmom_md)

            # check the constituent velocities match the central particle
            # since orientation is stuck at [1, 0, 0, 0], angular velocity
            # is 0.5 * real part of angmom / inertia.
            new_angmom = new_snap.particles.angmom[central_flag]
            omega = [
                0 if i == 0 else 0.5 * a / i
                for i, a in zip(properties_rigid["inertia"], new_angmom[0, 1:])
            ]
            expected_tangential_velocity = np.cross(omega, def_rigid["positions"])

            expected_velocity = np.add(expected_tangential_velocity, new_velo_central)
            assert np.allclose(new_velo_constituent, expected_velocity)


@pytest.mark.parametrize(
    "def_rigid,masses,init_args",
    [
        (
            {
                "constituent_types": ["B", "B"],
                "positions": [[-2.4, 0, 0], [2.4, 0, 0]],
                "orientations": [[1, 0, 0, 0], [1, 0, 0, 0]],
            },
            np.array([2, 1, 1]),
            {},
        ),
        (
            {
                "constituent_types": ["B", "B"],
                "positions": [[-2.4, 0, 0], [2.4, 0, 0]],
                "orientations": [[1, 0, 0, 0], [1, 0, 0, 0]],
            },
            np.array([1, 1, 1]),
            {"kT": 1},
        ),
        (
            {
                "constituent_types": ["B", "B", "B", "B"],
                "positions": [[-2.4, 0, 0], [2.4, 0, 0], [0, -2.4, 0], [0, 2.4, 0]],
                "orientations": [
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                ],
            },
            np.array([2, 1, 1, 0, 0]),
            {"kT": 1},
        ),
        (
            {
                "constituent_types": ["B", "B"],
                "positions": [[-2.4, 0, 0], [2.4, 0, 0]],
                "orientations": [[1, 0, 0, 0], [1, 0, 0, 0]],
            },
            np.array([2, 0.5, 1.5]),
            {"kT": 1},
        ),
    ],
    ids=["thermostat_error", "summation_error", "zero_mass_error", "center_loc_error"],
)
@pytest.mark.serial
def test_rigid_mass_errors(
    small_snap, simulation_factory, def_rigid, masses, init_args
):
    # create simulation
    initial_snap = small_snap
    if initial_snap.communicator.rank == 0:
        initial_snap.particles.types = ["A", "B"]
        initial_snap.particles.mass[:] = [masses[0]]
    sim = simulation_factory(initial_snap)
    sim.seed = 5

    rigid = hoomd.md.constrain.Rigid()
    rigid.body["A"] = def_rigid
    rigid.create_bodies(sim.state, masses={"A": masses[1:]})

    sim.operations.integrator = hoomd.mpcd.Integrator(dt=0, rigid=rigid)
    sim.operations.integrator.collision_method = (
        hoomd.mpcd.collide.StochasticRotationDynamics(
            period=1,
            embedded_particles=hoomd.filter.Rigid(flags=("constituent",)),
            angle=90,
            **init_args,
        )
    )

    # run simulation
    with pytest.raises(RuntimeError):
        sim.run(1)


@pytest.mark.serial
def test_rigid_nonparticipatory_zero_mass(small_snap, simulation_factory):
    # create simulation
    initial_snap = small_snap
    if initial_snap.communicator.rank == 0:
        initial_snap.particles.types = ["A", "B", "C"]
        initial_snap.particles.mass[:] = [2]
    sim = simulation_factory(initial_snap)
    sim.seed = 5

    rigid = hoomd.md.constrain.Rigid()
    rigid.body["A"] = {
        "constituent_types": ["B", "B", "C"],
        "positions": [[-2.4, 0, 0], [2.4, 0, 0], [0, 1, 0]],
        "orientations": [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
    }
    rigid.create_bodies(sim.state, masses={"A": np.array([1, 1, 0])})

    sim.operations.integrator = hoomd.mpcd.Integrator(dt=0, rigid=rigid)
    sim.operations.integrator.collision_method = (
        hoomd.mpcd.collide.StochasticRotationDynamics(
            period=1,
            embedded_particles=hoomd.filter.Type("B"),
            angle=90,
            kT=1,
        )
    )

    # run simulation
    # velocities should all stay zero because there is no solvent to collide with
    sim.run(1)
    new_snap = sim.state.get_snapshot()
    if new_snap.communicator.rank == 0:
        assert np.array_equal(np.zeros((4, 3)), new_snap.particles.velocity)
