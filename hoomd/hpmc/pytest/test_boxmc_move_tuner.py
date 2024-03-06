# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from math import isclose
import pytest

import hoomd
from hoomd import hpmc
from hoomd.conftest import operation_pickling_check
from hoomd.hpmc.tune.boxmc_move_size import (_MoveSizeTuneDefinition,
                                             BoxMCMoveSize)

MOVE_TYPES = ("aspect", "volume", "shear_x", "shear_y", "shear_z", "length_x",
              "length_y", "length_z")


def generate_move_definition(rng, move=None):
    if move is None:
        move = rng.choose(MOVE_TYPES)
    target = rng.random()
    domain = (rng.uniform(1e-8, 1e-4), rng.uniform(5, 10))
    if rng.random() > 0.5:
        domain = (domain[0], None)
    return {"attr": move, "target": target, "domain": domain}


def get_move_acceptance_ratio(boxmc, attr):
    """Helps translate between tune move names and their acceptance ratio."""
    splits = attr.split("_")
    if splits[0].startswith("l"):
        return boxmc.volume_moves
    if len(splits) == 1:
        return getattr(boxmc, attr + "_moves")
    return boxmc.shear_moves


@pytest.fixture(params=MOVE_TYPES)
def move_definition_dict(rng, request):
    return generate_move_definition(rng, request.param)


@pytest.fixture
def simulation(device, simulation_factory, lattice_snapshot_factory):
    if isinstance(device, hoomd.device.CPU):
        n = (4, 4, 8)
    else:
        n = (6, 6, 8)
    snap = lattice_snapshot_factory(dimensions=3, r=1e-2, n=n,
                                    a=2)  # 72 particles
    sim = simulation_factory(snap)
    integrator = hpmc.integrate.Sphere(default_d=0.01)
    integrator.shape["A"] = dict(diameter=0.9)
    sim.operations.integrator = integrator
    return sim


@pytest.fixture
def boxmc(move_definition_dict):
    split = move_definition_dict["attr"].split("_")
    attr = split[0]
    boxmc = hpmc.update.BoxMC(trigger=1, betaP=0)
    getattr(boxmc, attr)["weight"] = 1.0
    if len(split) > 1:
        getattr(boxmc, attr)["delta"] = (1e-1,) * 3
    else:
        getattr(boxmc, attr)["delta"] = 1e-1
    if attr == "volume":
        boxmc.volume["mode"] = "ln"
    if attr == "shear":
        boxmc.shear["reduce"] = 1.0
    return boxmc


@pytest.fixture
def move_size_definition(move_definition_dict, boxmc):
    return _MoveSizeTuneDefinition(**move_definition_dict, boxmc=boxmc)


class TestMoveSizeTuneDefinition:

    def test_getting_attrs(self, move_definition_dict, move_size_definition):
        for attr in move_definition_dict:
            if attr == "attr":
                assert move_definition_dict[attr].split("_")[0] == getattr(
                    move_size_definition, attr)
                continue
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
        simulation.run(10)
        attr = move_size_definition.attr
        accepted, rejected = get_move_acceptance_ratio(boxmc, attr)
        calc_acceptance_rate = (accepted) / (accepted + rejected)
        assert isclose(move_size_definition.y, calc_acceptance_rate)
        # We do this twice to ensure that when the counter doesn"t change our
        # return value does not change either.
        assert isclose(move_size_definition.y, calc_acceptance_rate)
        simulation.run(10)
        accepted, rejected = get_move_acceptance_ratio(boxmc, attr)
        calc_acceptance_rate = (accepted) / (accepted + rejected)
        assert isclose(move_size_definition.y, calc_acceptance_rate)

    def test_getting_setting_move_size(self, rng, boxmc, move_size_definition,
                                       simulation):
        attr = move_size_definition.attr

        def set_move_size(new_value):
            definition = getattr(boxmc, attr)
            if move_size_definition.index < 0:
                definition["delta"] = new_value
                return new_value
            old_value = list(definition["delta"])
            old_value[move_size_definition.index] = new_value
            definition["delta"] = old_value
            return new_value

        def get_move_size():
            base_size = getattr(boxmc, move_size_definition.attr)["delta"]
            if move_size_definition.index < 0:
                return base_size
            return base_size[move_size_definition.index]

        simulation.operations += move_size_definition.boxmc
        assert move_size_definition.x == get_move_size()
        set_move_size(rng.uniform(0, 10))
        assert move_size_definition.x == get_move_size()
        move_size_definition.x = rng.uniform(0, 10)
        assert move_size_definition.x == get_move_size()

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


@pytest.fixture(params=MOVE_TYPES)
def boxmc_tuner_method_and_kwargs(request, rng):
    cls_methods = (BoxMCMoveSize.secant_solver, BoxMCMoveSize.scale_solver)
    cls = cls_methods[rng.integers(2)]
    max_move_sizes = {"volume": 10.0, "aspect": 1.0}
    max_move_sizes.update({f"shear_{x}": 1.0 for x in ("x", "y", "z")})
    max_move_sizes.update({f"length_{x}": 1.0 for x in ("x", "y", "z")})
    return cls, {
        "trigger": 100,
        "moves": [request.param],
        "tol": 0.05,
        "target": 0.2,
        "max_move_size": max_move_sizes,
    }


@pytest.fixture
def boxmc_with_tuner(rng, boxmc_tuner_method_and_kwargs):
    cls, move_size_kwargs = boxmc_tuner_method_and_kwargs
    move = move_size_kwargs["moves"][0]
    boxmc = hpmc.update.BoxMC(1, betaP=1.0)
    if move == "aspect":
        boxmc.aspect = {"weight": 1.0, "delta": 0.4}
    elif move == "volume":
        boxmc.volume = {"weight": 1.0, "delta": 6.5, "mode": "standard"}
    elif move.startswith("l"):
        delta = [0.0, 0.0, 0.0]
        delta[["x", "y", "z"].index(move[-1])] = 0.05
        setattr(boxmc,
                move.split("_")[0], {
                    "weight": 1.0,
                    "delta": tuple(delta)
                })
    else:
        boxmc.shear = {"weight": 1.0, "delta": (1e-1,) * 3, "reduce": 1.0}
    cls_methods = (BoxMCMoveSize.secant_solver, BoxMCMoveSize.scale_solver)
    cls = cls_methods[rng.integers(2)]
    return boxmc, cls(**move_size_kwargs, boxmc=boxmc)


class TestMoveSize:

    def test_construction(self, boxmc_tuner_method_and_kwargs, boxmc):
        cls, params = boxmc_tuner_method_and_kwargs
        move_size = cls(**params, boxmc=boxmc)
        for attr in params:
            if attr == "trigger":
                assert getattr(move_size, attr).period == params[attr]
                continue
            try:
                assert getattr(move_size, attr) == params[attr]
            # We catch attribute errors since the solver may be the one to
            # have an attribute. This allows us to check that all attributes
            # are getting set correctly.
            except AttributeError:
                assert getattr(move_size.solver, attr) == params[attr]
        assert boxmc is move_size.boxmc

    def test_attach(self, boxmc_with_tuner, simulation):
        boxmc, move_size_tuner = boxmc_with_tuner
        simulation.operations.tuners.append(move_size_tuner)
        simulation.operations += boxmc
        tunable = move_size_tuner._tunables[0]
        assert tunable.y is None
        simulation.run(0)
        assert move_size_tuner._attached
        assert tunable.y is None

    def test_detach(self, boxmc_with_tuner, simulation):
        boxmc, move_size_tuner = boxmc_with_tuner
        simulation.operations.tuners.append(move_size_tuner)
        simulation.operations += boxmc
        simulation.run(0)
        assert move_size_tuner._attached
        move_size_tuner._detach()
        assert not move_size_tuner._attached

    def test_set_params(self, boxmc_with_tuner):
        _, move_size_tuner = boxmc_with_tuner
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
    def test_act(self, boxmc_with_tuner, simulation):
        boxmc, move_size_tuner = boxmc_with_tuner
        if move_size_tuner.moves[0].startswith("sh"):
            pytest.skip("Do not test shear with validation.")
        simulation.run(1_000)
        simulation.operations.tuners.append(move_size_tuner)
        simulation.operations += boxmc
        cnt = 0
        max_count = 10
        steps = 1501
        while not move_size_tuner.tuned and cnt < max_count:
            simulation.run(steps)
            cnt += 1
        assert move_size_tuner.tuned
        simulation.operations.tuners.pop()
        simulation.run(500)
        move_counts = get_move_acceptance_ratio(boxmc, move_size_tuner.moves[0])
        acceptance_rate = move_counts[0] / sum(move_counts)
        # We must increase tolerance a bit since tuning to a tolerance is noisy
        # for BoxMC. Particularly shear move tuning.
        tolerance = 2.5 * move_size_tuner.solver.tol
        assert abs(acceptance_rate - move_size_tuner.target) <= tolerance

    def test_pickling(self, boxmc_with_tuner, simulation):
        _, move_size_tuner = boxmc_with_tuner
        operation_pickling_check(move_size_tuner, simulation)
