# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from math import isclose
import pytest

from hoomd import hpmc
from hoomd.conftest import operation_pickling_check
from hoomd.hpmc.tune.volume_move_size import (_MoveSizeTuneDefinition,
                                              VolumeMoveSize)


@pytest.fixture(params=("volume", "aspect"))
def move_definition_dict(request):
    return {"attr": request.param, "target": 0.5, "domain": (1e-5, None)}


@pytest.fixture
def simulation(simulation_factory, lattice_snapshot_factory):
    snap = lattice_snapshot_factory(dimensions=2, r=1e-3, n=20)  # 400 particles
    sim = simulation_factory(snap)
    integrator = hpmc.integrate.Sphere(default_d=0.01)
    integrator.shape["A"] = dict(diameter=0.9)
    sim.operations.integrator = integrator
    return sim


@pytest.fixture
def boxmc(move_definition_dict):
    attr = move_definition_dict["attr"]
    boxmc = hpmc.update.BoxMC(0, 1)
    getattr(boxmc, attr)["weight"] = 1.0
    getattr(boxmc, attr)["delta"] = 1e-2
    if attr == "volume":
        boxmc.volume["mode"] = "ln"
    return boxmc


@pytest.fixture
def move_size_definition(move_definition_dict, boxmc):
    return _MoveSizeTuneDefinition(**move_definition_dict, boxmc=boxmc)


class TestMoveSizeTuneDefinition:

    def test_getting_attrs(self, move_definition_dict, move_size_definition):
        for attr in move_definition_dict:
            assert move_definition_dict[attr] == getattr(
                move_size_definition, attr)

    def test_setting_attrs(self, move_size_definition):
        move_size_definition.domain = (None, 5)
        assert move_size_definition.domain == (None, 5)
        move_size_definition.attr = "a"
        assert move_size_definition.attr == "a"
        move_size_definition.target = 0.9
        assert move_size_definition.target == 0.9

    def test_getting_acceptance_rate(self, move_size_definition, simulation,
                                     boxmc):
        simulation.operations += boxmc
        simulation.run(0)
        # needed to set previous values need to to calculate acceptance rate
        assert move_size_definition.y is None
        simulation.run(1000)
        move_attr = f"{move_size_definition.attr}_moves"
        accepted, rejected = getattr(boxmc, move_attr)
        calc_acceptance_rate = (accepted) / (accepted + rejected)
        assert isclose(move_size_definition.y, calc_acceptance_rate)
        # We do this twice to ensure that when the counter doesn"t change our
        # return value does not change either.
        assert isclose(move_size_definition.y, calc_acceptance_rate)
        simulation.run(10)
        assert not isclose(move_size_definition.y, calc_acceptance_rate)
        accepted, rejected = getattr(boxmc, move_attr)
        calc_acceptance_rate = accepted / (accepted + rejected)
        assert isclose(move_size_definition.y, calc_acceptance_rate)

    def test_getting_setting_move_size(self, boxmc, move_size_definition,
                                       simulation):
        attr = move_size_definition.attr
        simulation.operations += move_size_definition.boxmc
        assert move_size_definition.x == getattr(boxmc, attr)["delta"]
        getattr(boxmc, attr)["delta"] = getattr(boxmc, attr)["delta"] * 1.1
        assert move_size_definition.x == getattr(boxmc, attr)["delta"]
        move_size_definition.x = getattr(boxmc, attr)["delta"] * 1.1
        assert move_size_definition.x == getattr(boxmc, attr)["delta"]

    def test_hash(self, move_size_definition, move_definition_dict, simulation,
                  boxmc):
        identical_definition = _MoveSizeTuneDefinition(**move_definition_dict,
                                                       boxmc=boxmc)
        assert hash(identical_definition) == hash(move_size_definition)
        move_definition_dict["domain"] = (None, 5)
        different_definition = _MoveSizeTuneDefinition(**move_definition_dict,
                                                       boxmc=boxmc)
        assert hash(different_definition) != hash(move_size_definition)

    def test_eq(self, move_size_definition, move_definition_dict, simulation,
                boxmc):
        identical_definition = _MoveSizeTuneDefinition(**move_definition_dict,
                                                       boxmc=boxmc)
        assert identical_definition == move_size_definition
        move_definition_dict["domain"] = (None, 5)
        different_definition = _MoveSizeTuneDefinition(**move_definition_dict,
                                                       boxmc=boxmc)
        assert different_definition != move_size_definition


_move_size_options = [(VolumeMoveSize.scale_solver,
                       dict(trigger=300,
                            moves=["volume"],
                            target=0.5,
                            max_move_size={"volume": 5},
                            tol=1e-1)),
                      (VolumeMoveSize.secant_solver,
                       dict(
                           trigger=300,
                           moves=["aspect"],
                           target=0.6,
                       ))]


@pytest.fixture(params=_move_size_options,
                ids=lambda x: "MoveSize-" + x[0].__name__)
def move_size_tuner_pairs(request, boxmc):
    return (request.param[0], {**request.param[1], "boxmc": boxmc})


@pytest.fixture
def move_size_tuner(move_size_tuner_pairs):
    return move_size_tuner_pairs[0](**move_size_tuner_pairs[1])


class TestMoveSize:

    def test_construction(self, move_size_tuner_pairs):
        move_size_dict = move_size_tuner_pairs[1]
        move_size = move_size_tuner_pairs[0](**move_size_dict)
        for attr in move_size_dict:
            if attr == "trigger":
                assert getattr(move_size, attr).period == move_size_dict[attr]
                continue
            try:
                assert getattr(move_size, attr) == move_size_dict[attr]
            # We catch attribute errors since the solver may be the one to
            # have an attribute. This allows us to check that all attributes
            # are getting set correctly.
            except AttributeError:
                assert getattr(move_size.solver, attr) == move_size_dict[attr]

    def test_attach(self, move_size_tuner, simulation, boxmc):
        simulation.operations.tuners.append(move_size_tuner)
        simulation.operations += boxmc
        tunable = move_size_tuner._tunables[0]
        assert tunable.y is None
        simulation.run(0)
        assert move_size_tuner._attached
        assert tunable.y is None

    def test_detach(self, move_size_tuner, simulation, boxmc):
        simulation.operations.tuners.append(move_size_tuner)
        simulation.operations += boxmc
        simulation.run(0)
        assert move_size_tuner._attached
        move_size_tuner._detach()
        assert not move_size_tuner._attached

    def test_set_params(self, move_size_tuner):
        target = move_size_tuner.target
        assert all(target == t.target for t in move_size_tuner._tunables)
        target *= 1.1
        move_size_tuner.target = target
        assert all(target == t.target for t in move_size_tuner._tunables)
        assert target == move_size_tuner.target

        max_move = 4.
        move_size_tuner.max_move_size["volume"] = max_move
        assert move_size_tuner.max_move_size["volume"] == max_move

        move_size_tuner.max_move_size["aspect"] = max_move
        assert move_size_tuner.max_move_size["aspect"] == max_move

        with pytest.raises(ValueError):
            move_size_tuner.moves = ["f", "a"]
        move_size_tuner.moves = ["volume"]
        assert move_size_tuner.moves == ["volume"]

    # All tests (using differnt fixtures) combined take about 17 seconds, so
    # only test during validation
    @pytest.mark.validate
    def test_act(self, move_size_tuner_pairs, simulation, boxmc):
        move_size_tuner = move_size_tuner_pairs[0](**move_size_tuner_pairs[1])
        simulation.operations.tuners.append(move_size_tuner)
        simulation.operations += boxmc
        simulation.run(901)
        # breakpoint()
        assert False
        cnt = 0
        while not move_size_tuner.tuned and cnt < 4:
            simulation.run(2000)
            cnt += 1
        assert move_size_tuner.tuned
        simulation.run(10000)
        move_counts = simulation.operations.integrator.translate_moves
        acceptance_rate = move_counts[0] / sum(move_counts)
        tolerance = move_size_tuner.solver.tol
        assert abs(acceptance_rate - move_size_tuner.target) <= tolerance

    def test_pickling(self, move_size_tuner, simulation):
        operation_pickling_check(move_size_tuner, simulation)
