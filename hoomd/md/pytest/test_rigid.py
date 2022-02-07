# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from collections.abc import Sequence

import numpy as np
import pytest

import hoomd
import hoomd.md as md


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
        "charges": [0.0, 1.0, 2.0, 3.5],
        "diameters": [1.0, 1.5, 0.5, 1.0]
    }


def test_body_setting(valid_body_definition):
    invalid_body_definitions = {
        "constituent_types": [[4], "hello", ("A", 4)],
        "positions": [[(1, 2)], [(1.0, 4.0, "foo")], 1.0, "hello"],
        "orientations": [[(1, 2, 3)], [(1.0, 4.0, 5.0, "foo")], [1.0], 1.0,
                         "foo"],
        "charges": [0.0, ["foo"]],
        "diameters": [1.0, "foo", ["foo"]]
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


def check_bodies(snapshot, definition):
    """Non-general assumes a snapshot from two_particle_snapshot_factory.

    This is just to prevent duplication of code from test_create_bodies and
    test_running_simulation.
    """
    rowan = pytest.importorskip("rowan")

    assert snapshot.particles.N == 10
    assert all(snapshot.particles.typeid[3:] == 1)

    assert snapshot.particles.body[0] == 0
    assert all(snapshot.particles.body[2:6] == 0)
    assert snapshot.particles.body[1] == 1
    assert all(snapshot.particles.body[6:] == 1)

    # check charges
    for i in range(4):
        assert snapshot.particles.charge[i + 2] == definition["charges"][i]
        assert snapshot.particles.charge[i + 6] == definition["charges"][i]

    # check diameters
    for i in range(4):
        assert snapshot.particles.diameter[i + 2] == definition["diameters"][i]
        assert snapshot.particles.diameter[i + 6] == definition["diameters"][i]

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


def test_create_bodies(simulation_factory, two_particle_snapshot_factory,
                       lattice_snapshot_factory, valid_body_definition):
    rowan = pytest.importorskip("rowan")  # noqa F841 - used by called method

    rigid = md.constrain.Rigid()
    rigid.body["A"] = valid_body_definition

    initial_snapshot = two_particle_snapshot_factory()
    if initial_snapshot.communicator.rank == 0:
        initial_snapshot.particles.types = ["A", "B"]
    sim = simulation_factory(initial_snapshot)

    rigid.create_bodies(sim.state)
    snapshot = sim.state.get_snapshot()
    if snapshot.communicator.rank == 0:
        check_bodies(snapshot, valid_body_definition)

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


def test_running_simulation(simulation_factory, two_particle_snapshot_factory,
                            valid_body_definition):
    rowan = pytest.importorskip("rowan")  # noqa F841 - used by called method

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

    rigid.create_bodies(sim.state)
    sim.operations += integrator
    sim.run(5)
    snapshot = sim.state.get_snapshot()
    if sim.device.communicator.rank == 0:
        check_bodies(snapshot, valid_body_definition)


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
