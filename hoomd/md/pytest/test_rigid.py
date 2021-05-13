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
        if (isinstance(value, Sequence)
                and len(value) > 0 and not isinstance(value[0], str)):
            assert np.allclose(value, current_body_definition[key])
        else:
            assert value == current_body_definition[key]

    for key, values in invalid_body_definitions.items():
        for invalid_value in values:
            current_body_definition[key] = invalid_value
            with pytest.raises(hoomd.data.typeconverter.TypeConversionError):
                rigid.body["A"] = current_body_definition
        current_body_definition[key] = valid_body_definition[key]


def check_bodies(snapshot, definition):
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

    # Check positions
    particle_one = snapshot.particles.position[0]
    particle_two = snapshot.particles.position[1]
    for i in range(4):
        assert np.allclose(snapshot.particles.position[i + 2] - particle_one,
                           definition["positions"][i])

        assert np.allclose(snapshot.particles.position[i + 6] - particle_two,
                           definition["positions"][i])

    # check orientation (note since the central particles have the default
    # orientation no rotation should be needed
    for i in range(4):
        assert np.allclose(snapshot.particles.orientation[i + 2],
                           definition["orientations"][i])
        assert np.allclose(snapshot.particles.orientation[i + 6],
                           definition["orientations"][i])

    # check charges
    for i in range(4):
        assert (snapshot.particles.charge[i + 2] ==
                definition["charges"][i])
        assert (snapshot.particles.charge[i + 6] ==
                definition["charges"][i])

    # check diameters
    for i in range(4):
        assert (snapshot.particles.diameter[i + 2] ==
                definition["diameters"][i])
        assert (snapshot.particles.diameter[i + 6] ==
                definition["diameters"][i])


def test_create_bodies(simulation_factory, two_particle_snapshot_factory,
                       valid_body_definition):
    rigid = md.constrain.Rigid()
    rigid.body["A"] = valid_body_definition

    initial_snapshot = two_particle_snapshot_factory()
    initial_snapshot.particles.types = ["A", "B"]
    sim = simulation_factory(initial_snapshot)

    rigid.create_bodies(sim.state)
    snapshot = sim.state.snapshot
    if sim.device.communicator.rank == 0:
        check_bodies(snapshot, valid_body_definition)


def test_attaching(simulation_factory, two_particle_snapshot_factory,
                   valid_body_definition):
    rigid = md.constrain.Rigid()
    rigid.body["A"] = valid_body_definition
    langevin = md.methods.Langevin(kT=2.0, filter=hoomd.filter.Rigid())
    integrator = md.Integrator(dt=0.005, methods=[langevin])
    integrator.rigid = rigid

    initial_snapshot = two_particle_snapshot_factory()
    initial_snapshot.particles.types = ["A", "B"]
    sim = simulation_factory(initial_snapshot)

    rigid.create_bodies(sim.state)
    sim.operations += integrator
    sim.run(0)

    for key, value in rigid.body["A"].items():
        if (isinstance(value, Sequence)
                and len(value) > 0 and not isinstance(value[0], str)):
            assert np.allclose(value, valid_body_definition[key])
        else:
            assert value == valid_body_definition[key]


def test_error_on_invalid_body(simulation_factory,
                               two_particle_snapshot_factory,
                               valid_body_definition):
    rigid = md.constrain.Rigid()
    rigid.body["A"] = valid_body_definition
    langevin = md.methods.Langevin(kT=2.0, filter=hoomd.filter.Rigid())
    integrator = md.Integrator(dt=0.005, methods=[langevin])
    integrator.rigid = rigid

    initial_snapshot = two_particle_snapshot_factory()
    initial_snapshot.particles.types = ["A", "B"]
    sim = simulation_factory(initial_snapshot)

    sim.operations += integrator
    with pytest.raises(RuntimeError):
        sim.run(0)


def test_running_simulation(simulation_factory, two_particle_snapshot_factory,
                            valid_body_definition):
    rigid = md.constrain.Rigid()
    rigid.body["A"] = valid_body_definition
    langevin = md.methods.Langevin(kT=2.0, filter=hoomd.filter.Rigid())
    lj = hoomd.md.pair.LJ(nlist=md.nlist.Cell(), mode="shift")
    lj.params.default = {"epsilon": 0.0, "sigma": 1}
    lj.params[("A", "A")] = {"epsilon": 1.0}
    lj.params[("B", "B")] = {"epsilon": 1.0}
    lj.r_cut.default = 2 ** (1.0 / 6.0)
    integrator = md.Integrator(dt=0.005, methods=[langevin], forces=[lj])
    integrator.rigid = rigid

    initial_snapshot = two_particle_snapshot_factory()
    initial_snapshot.particles.types = ["A", "B"]
    sim = simulation_factory(initial_snapshot)

    rigid.create_bodies(sim.state)
    sim.operations += integrator
    sim.run(100)
    snapshot = sim.state.snapshot
    if sim.device.communicator.rank == 0:
        check_bodies(snapshot, valid_body_definition)
