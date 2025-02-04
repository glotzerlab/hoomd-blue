# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from collections.abc import Sequence
import itertools
import numpy as np
import pytest
from hoomd.conftest import autotuned_kernel_parameter_check

try:
    import rowan
    skip_rowan = False
except ImportError:
    skip_rowan = True

import hoomd
import hoomd.md as md

skip_rowan = pytest.mark.skipif(skip_rowan, reason="rowan cannot be imported.")


@pytest.fixture
def valid_body_definition():
    return {
        "constituent_types": ["B", "B", "B", "B"],
        "positions": [
            [1, 0, -1 / (2**(1. / 2.))],
            [-1, 0, -1 / (2**(1. / 2.))],
            [0, -1, 1 / (2**(1. / 2.))],
            [0, 1, 1 / (2**(1. / 2.))],
        ],
        "orientations": [(1.0, 0.0, 0.0, 0.0)] * 4,
    }


def test_body_setting(valid_body_definition):
    invalid_body_definitions = {
        "constituent_types": [[4], "hello", ("A", 4)],
        "positions": [[(1, 2)], [(1.0, 4.0, "foo")], 1.0, "hello"],
        "orientations": [[(1, 2, 3)], [(1.0, 4.0, 5.0, "foo")], [1.0], 1.0,
                         "foo"],
    }

    rigid = md.constrain.Rigid()
    current_body_definition = {**valid_body_definition}
    rigid.body["A"] = current_body_definition

    for key, value in rigid.body["A"].items():
        if (isinstance(value, Sequence) and len(value) > 0
                and not isinstance(value[0], str)):
            assert np.allclose(value, current_body_definition[key])
        else:
            assert value == current_body_definition[key]

    # Test dictionaries with a single invalid input
    for key, values in invalid_body_definitions.items():
        for invalid_value in values:
            current_body_definition[key] = invalid_value
            with pytest.raises(hoomd.data.typeconverter.TypeConversionError):
                rigid.body["A"] = current_body_definition
        # Reset the body definition to a valid state to ensure only one key is
        # ever invalid.
        current_body_definition[key] = valid_body_definition[key]


def check_bodies(snapshot, definition, charges=None):
    """Non-general assumes a snapshot from two_particle_snapshot_factory.

    This is just to prevent duplication of code from test_create_bodies and
    test_running_simulation.
    """
    assert snapshot.particles.N == 10
    assert all(snapshot.particles.typeid[3:] == 1)

    assert snapshot.particles.body[0] == 0
    assert all(snapshot.particles.body[2:6] == 0)
    assert snapshot.particles.body[1] == 1
    assert all(snapshot.particles.body[6:] == 1)

    # check charges
    if charges is not None:
        for i in range(4):
            assert snapshot.particles.charge[i + 2] == charges[i]
            assert snapshot.particles.charge[i + 6] == charges[i]

    particle_one = (snapshot.particles.position[0],
                    snapshot.particles.orientation[0])
    particle_two = (snapshot.particles.position[1],
                    snapshot.particles.orientation[1])

    # Check positions
    def check_position(central_position, central_orientation,
                       constituent_position, local_position):
        d_pos = rowan.rotate(central_orientation, local_position)
        assert np.allclose(central_position + d_pos, constituent_position)

    for i in range(4):
        check_position(*particle_one, snapshot.particles.position[i + 2],
                       definition["positions"][i])
        check_position(*particle_two, snapshot.particles.position[i + 6],
                       definition["positions"][i])

    # check orientation
    def check_orientation(central_orientation, constituent_orientation,
                          local_orientation):
        expected_orientation = rowan.normalize(
            rowan.multiply(central_orientation, local_orientation))
        assert np.allclose(expected_orientation, local_orientation)

    for i in range(4):
        check_orientation(particle_one[1],
                          snapshot.particles.orientation[i + 2],
                          definition["orientations"][i])
        check_orientation(particle_two[1],
                          snapshot.particles.orientation[i + 6],
                          definition["orientations"][i])


@skip_rowan
def test_create_bodies(simulation_factory, two_particle_snapshot_factory,
                       lattice_snapshot_factory, valid_body_definition):
    rigid = md.constrain.Rigid()
    rigid.body["A"] = valid_body_definition

    initial_snapshot = two_particle_snapshot_factory()
    if initial_snapshot.communicator.rank == 0:
        initial_snapshot.particles.types = ["A", "B"]
    sim = simulation_factory(initial_snapshot)

    charges = [1.0, 2.0, 3.0, 4.0]
    rigid.create_bodies(sim.state, charges={"A": charges})
    snapshot = sim.state.get_snapshot()
    if snapshot.communicator.rank == 0:
        check_bodies(snapshot, valid_body_definition, charges)

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005, rigid=rigid)
    # Ensure validate bodies passes
    sim.run(0)

    # Second test with more general testing
    # detach rigid
    sim.operations.integrator.rigid = None

    initial_snapshot = lattice_snapshot_factory(n=10)
    if initial_snapshot.communicator.rank == 0:
        initial_snapshot.particles.types = ["C", "A", "B"]
        # Grab the middle particles and a random one to ensure that particle
        # type ordering with respect to particle tag does not matter for
        # create_bodies.
        initial_snapshot.particles.typeid[100:800] = 1
        initial_snapshot.particles.typeid[55] = 1

    sim = simulation_factory(initial_snapshot)
    rigid.create_bodies(sim.state)
    snapshot = sim.state.get_snapshot()
    if snapshot.communicator.rank == 0:
        # Check central particles
        central_tags = np.empty(701, dtype=int)
        central_tags[0] = 55
        central_tags[1:] = np.arange(100, 800)
        print
        assert np.all(snapshot.particles.body[central_tags] == central_tags)
        # Check free bodies
        assert np.all(snapshot.particles.body[:55] == -1)
        assert np.all(snapshot.particles.body[56:100] == -1)
        assert np.all(snapshot.particles.body[800:1000] == -1)
        # Check constituent_particles
        assert np.all(
            snapshot.particles.body[1000:] == np.repeat(central_tags, 4))

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005, rigid=rigid)
    # Ensure validate bodies passes
    sim.run(0)


def test_attaching(simulation_factory, two_particle_snapshot_factory,
                   valid_body_definition):
    rigid = md.constrain.Rigid()
    rigid.body["A"] = valid_body_definition
    langevin = md.methods.Langevin(kT=2.0, filter=hoomd.filter.Rigid())
    integrator = md.Integrator(dt=0.005, methods=[langevin])
    integrator.rigid = rigid

    initial_snapshot = two_particle_snapshot_factory()
    if initial_snapshot.communicator.rank == 0:
        initial_snapshot.particles.types = ["A", "B"]
    sim = simulation_factory(initial_snapshot)

    rigid.create_bodies(sim.state)
    sim.operations += integrator
    sim.run(0)

    for key, value in rigid.body["A"].items():
        if (isinstance(value, Sequence) and len(value) > 0
                and not isinstance(value[0], str)):
            assert np.allclose(value, valid_body_definition[key])
        else:
            assert value == valid_body_definition[key]


@pytest.mark.serial
def test_error_on_invalid_body(simulation_factory,
                               two_particle_snapshot_factory,
                               valid_body_definition):
    """Tests that Simulation fails when bodies are not present in state."""
    rigid = md.constrain.Rigid()
    rigid.body["A"] = valid_body_definition
    langevin = md.methods.Langevin(kT=2.0, filter=hoomd.filter.Rigid())
    integrator = md.Integrator(dt=0.005, methods=[langevin])
    integrator.rigid = rigid

    initial_snapshot = two_particle_snapshot_factory()
    if initial_snapshot.communicator.rank == 0:
        initial_snapshot.particles.types = ["A", "B"]
    sim = simulation_factory(initial_snapshot)

    sim.operations += integrator
    with pytest.raises(RuntimeError):
        sim.run(0)


@skip_rowan
def test_running_simulation(simulation_factory, two_particle_snapshot_factory,
                            valid_body_definition):
    rigid = md.constrain.Rigid()
    rigid.body["A"] = valid_body_definition
    langevin = md.methods.Langevin(kT=2.0, filter=hoomd.filter.Rigid())
    lj = hoomd.md.pair.LJ(nlist=md.nlist.Cell(buffer=0.4), mode="shift")
    lj.params.default = {"epsilon": 0.0, "sigma": 1}
    lj.params[("A", "A")] = {"epsilon": 1.0}
    lj.params[("B", "B")] = {"epsilon": 1.0}
    lj.r_cut.default = 2**(1.0 / 6.0)
    integrator = md.Integrator(dt=0.005, methods=[langevin], forces=[lj])
    integrator.rigid = rigid

    initial_snapshot = two_particle_snapshot_factory()
    if initial_snapshot.communicator.rank == 0:
        initial_snapshot.particles.types = ["A", "B"]
    sim = simulation_factory(initial_snapshot)
    sim.seed = 5

    charges = [1.0, 2.0, 3.0, 4.0]
    rigid.create_bodies(sim.state, charges={"A": charges})
    sim.operations += integrator
    sim.run(5)
    snapshot = sim.state.get_snapshot()
    if sim.device.communicator.rank == 0:
        check_bodies(snapshot, valid_body_definition, charges)

    autotuned_kernel_parameter_check(instance=rigid,
                                     activate=lambda: sim.run(1))


def test_running_without_body_definition(simulation_factory,
                                         two_particle_snapshot_factory):
    rigid = md.constrain.Rigid()
    langevin = md.methods.Langevin(kT=2.0, filter=hoomd.filter.Rigid())
    lj = hoomd.md.pair.LJ(nlist=md.nlist.Cell(buffer=0.4), mode="shift")
    lj.params.default = {"epsilon": 0.0, "sigma": 1}
    lj.params[("A", "A")] = {"epsilon": 1.0}
    lj.params[("B", "B")] = {"epsilon": 1.0}
    lj.r_cut.default = 2**(1.0 / 6.0)
    integrator = md.Integrator(dt=0.005, methods=[langevin], forces=[lj])
    integrator.rigid = rigid

    initial_snapshot = two_particle_snapshot_factory()
    if initial_snapshot.communicator.rank == 0:
        initial_snapshot.particles.types = ["A", "B"]
    sim = simulation_factory(initial_snapshot)
    sim.seed = 5

    sim.operations += integrator
    sim.run(1)


@pytest.mark.serial
def test_setting_body_after_attaching(simulation_factory,
                                      two_particle_snapshot_factory,
                                      valid_body_definition):
    """Test updating body definition without updating sim particles fails."""
    rigid = md.constrain.Rigid()
    langevin = md.methods.Langevin(kT=2.0, filter=hoomd.filter.Rigid())
    lj = hoomd.md.pair.LJ(nlist=md.nlist.Cell(buffer=0.4), mode="shift")
    lj.params.default = {"epsilon": 0.0, "sigma": 1}
    lj.params[("A", "A")] = {"epsilon": 1.0}
    lj.params[("B", "B")] = {"epsilon": 1.0}
    lj.r_cut.default = 2**(1.0 / 6.0)
    integrator = md.Integrator(dt=0.005, methods=[langevin], forces=[lj])
    integrator.rigid = rigid

    initial_snapshot = two_particle_snapshot_factory()
    if initial_snapshot.communicator.rank == 0:
        initial_snapshot.particles.types = ["A", "B"]
    sim = simulation_factory(initial_snapshot)
    sim.seed = 5

    sim.operations += integrator
    sim.run(1)
    rigid.body["A"] = valid_body_definition
    # This should error because the bodies have not been updated, but the
    # setting should be fine.
    with pytest.raises(RuntimeError):
        sim.run(1)


def test_rigid_body_restart(simulation_factory, valid_body_definition):
    s = hoomd.Snapshot()
    N = 1000

    if s.communicator.rank == 0:
        s.particles.N = N
        s.particles.position[:] = [[-0.5, 0, 0]] * N
        s.particles.body[:] = [x for x in range(N)]
        s.particles.types = ['A', 'B']
        s.particles.typeid[:] = [0] * N
        s.configuration.box = [2, 2, 2, 0, 0, 0]

    # create simulation object and add integrator
    sim = simulation_factory(s)
    integrator = hoomd.md.Integrator(dt=0.001)
    sim.operations.integrator = integrator

    # create bodies
    rigid = hoomd.md.constrain.Rigid()
    rigid.body["A"] = valid_body_definition
    rigid.create_bodies(sim.state)
    sim.run(0)

    snapshot = sim.state.get_snapshot()
    N_const = len(valid_body_definition['constituent_types'])
    if snapshot.communicator.rank == 0:
        assert np.all(snapshot.particles.typeid[:N] == 0)
        assert np.all(snapshot.particles.typeid[N:] == 1)
        assert np.all(snapshot.particles.body[:N] == np.arange(N))
        should_be = np.arange(N * N_const) // N_const
        assert np.all(snapshot.particles.body[N:] == should_be)


@pytest.mark.parametrize("reload_snapshot, n_free",
                         itertools.product([False, True], [0, 10]))
def test_rigid_dof(lattice_snapshot_factory, simulation_factory,
                   valid_body_definition, reload_snapshot, n_free):
    n = 7
    n_bodies = n**3 - n_free
    initial_snapshot = lattice_snapshot_factory(particle_types=['A', 'B'],
                                                n=n,
                                                dimensions=3,
                                                a=5)

    if initial_snapshot.communicator.rank == 0:
        initial_snapshot.particles.body[:n_bodies] = range(n_bodies)
        initial_snapshot.particles.moment_inertia[:n_bodies] = (1, 1, 1)
        initial_snapshot.particles.typeid[n_bodies:] = 1

    sim = simulation_factory(initial_snapshot)
    rigid = hoomd.md.constrain.Rigid()
    rigid.body["A"] = valid_body_definition
    rigid.create_bodies(sim.state)

    if reload_snapshot:
        # In C++, createRigidBodies follows a different code path than
        # validateRigidBodies. When reload_snapshot is True, test the latter.
        snapshot_with_constituents = sim.state.get_snapshot()
        sim.state.set_snapshot(snapshot_with_constituents)

    integrator = hoomd.md.Integrator(dt=0.0, integrate_rotational_dof=True)
    integrator.rigid = rigid

    thermo_all = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All())
    thermo_two = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.Tags([0, 1]))
    thermo_central = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.Rigid(flags=("center",)))
    thermo_central_free = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.Rigid(flags=("center", "free")))
    thermo_constituent = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.Rigid(flags=("constituent",)))

    sim.operations.computes.extend([
        thermo_all, thermo_two, thermo_central, thermo_central_free,
        thermo_constituent
    ])
    sim.operations.integrator = integrator
    integrator.methods.append(
        hoomd.md.methods.ConstantVolume(filter=hoomd.filter.Rigid(
            flags=("center", "free"))))

    sim.run(0)

    assert thermo_all.translational_degrees_of_freedom == (n_bodies
                                                           + n_free) * 3 - 3
    assert thermo_two.translational_degrees_of_freedom == 2 * 3 - 3 * (
        2 / (n_bodies + n_free))
    assert thermo_central.translational_degrees_of_freedom == (
        n_bodies * 3 - 3 * (n_bodies / (n_bodies + n_free)))
    assert thermo_central_free.translational_degrees_of_freedom == (
        n_bodies + n_free) * 3 - 3
    assert thermo_constituent.translational_degrees_of_freedom == 0

    # Test again with the rigid body constraints removed. Now the integration
    # method is applied only to part of the system, so the 3 degrees of freedom
    # are not removed.
    integrator.rigid = None
    sim.state.update_group_dof()

    sim.run(0)

    assert thermo_all.translational_degrees_of_freedom == (n_bodies
                                                           + n_free) * 3
    assert thermo_two.translational_degrees_of_freedom == 2 * 3
    assert thermo_central.translational_degrees_of_freedom == n_bodies * 3
    assert thermo_central_free.translational_degrees_of_freedom == (
        n_bodies + n_free) * 3
    assert thermo_constituent.translational_degrees_of_freedom == 0
