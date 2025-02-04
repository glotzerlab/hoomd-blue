# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

import hoomd
from hoomd import md
from hoomd.conftest import operation_pickling_check


@pytest.fixture
def nlist():
    return md.nlist.Cell(buffer=0.5)


@pytest.fixture
def simulation(simulation_factory, lattice_snapshot_factory, nlist):
    snap = lattice_snapshot_factory(dimensions=2, r=1e-3, n=20)  # 400 particles
    sim = simulation_factory(snap)
    lj = md.pair.LJ(nlist, default_r_cut=2.5)
    lj.params[("A", "A")] = {"sigma": 1.0, "epsilon": 1.0}
    thermostat = hoomd.md.methods.thermostats.MTTK(kT=1.0, tau=1.0)
    integrator = md.Integrator(
        dt=0.005,
        methods=[md.methods.ConstantVolume(hoomd.filter.All(), thermostat)],
        forces=[lj])
    sim.operations.integrator = integrator
    return sim


@pytest.fixture(params=("GradientDescent", "GridOptimizer"))
def nlist_tuner(nlist, request):
    if request.param == "GradientDescent":
        return md.tune.NeighborListBuffer.with_gradient_descent(
            trigger=5, nlist=nlist, maximum_buffer=1.5)
    return md.tune.NeighborListBuffer.with_grid(trigger=5,
                                                nlist=nlist,
                                                maximum_buffer=1.5)


class TestMoveSize:

    def test_invalid_construction(self, nlist):
        solver = hoomd.tune.solve.ScaleSolver()
        with pytest.raises(ValueError):
            md.tune.NeighborListBuffer(trigger=5,
                                       solver=solver,
                                       nlist=nlist,
                                       maximum_buffer=1.0)

        solver = hoomd.tune.solve.GradientDescent()
        with pytest.raises(TypeError):
            md.tune.NeighborListBuffer(trigger=5, solver=solver, nlist=nlist)

    def test_valid_construction(self, nlist):
        solver = hoomd.tune.solve.GradientDescent()
        attrs = {
            "solver": solver,
            "nlist": nlist,
            "trigger": 5,
            "maximum_buffer": 1.0
        }
        tuner = md.tune.NeighborListBuffer(**attrs)
        for attr, value in attrs.items():
            tuner_attr = getattr(tuner, attr)
            if attr == 'trigger':
                assert tuner_attr.period == value
            else:
                assert tuner_attr is value or tuner_attr == value

        # Test factory method with_gradient_descent
        attrs.pop("solver")
        solver_attrs = {
            "alpha": 0.1,
            "kappa": np.array([0.2, 0.15]),
            "tol": 1e-3,
            "max_delta": 0.4
        }
        tuner = md.tune.NeighborListBuffer.with_gradient_descent(
            **attrs, **solver_attrs)
        for attr, value in attrs.items():
            tuner_attr = getattr(tuner, attr)
            if attr == 'trigger':
                assert tuner_attr.period == value
            else:
                assert tuner_attr is value or tuner_attr == value
        kappa = solver_attrs.pop("kappa")
        assert np.array_equal(kappa, tuner.solver.kappa)
        for attr, value in solver_attrs.items():
            assert getattr(tuner.solver, attr) == value

    def test_attach(self, nlist_tuner, nlist, simulation):
        simulation.operations += nlist_tuner
        simulation.run(1)
        assert nlist_tuner._attached
        assert nlist_tuner._tunable.y != 0
        assert nlist_tuner._tunable.x == nlist.buffer
        new_buffer = 0.4
        nlist_tuner._tunable.x = new_buffer
        assert nlist_tuner._tunable.x == new_buffer
        assert nlist.buffer == new_buffer

    def test_detach(self, nlist_tuner, simulation):
        simulation.operations.tuners.append(nlist_tuner)
        simulation.run(0)
        assert nlist_tuner._attached
        simulation.operations -= nlist_tuner
        assert not nlist_tuner._attached

    def test_set_params(self, nlist_tuner):
        max_buffer = 4.
        nlist_tuner.maximum_buffer_size = max_buffer
        assert nlist_tuner.maximum_buffer_size == max_buffer
        trigger = hoomd.trigger.Before(400)
        nlist_tuner.trigger = trigger
        assert nlist_tuner.trigger == trigger

    def test_act(self, nlist, nlist_tuner, simulation):
        old_buffer = nlist.buffer
        simulation.operations.tuners.append(nlist_tuner)
        simulation.run(10)
        assert old_buffer != nlist.buffer

    def test_pickling(self, nlist_tuner, simulation):
        operation_pickling_check(nlist_tuner, simulation)
