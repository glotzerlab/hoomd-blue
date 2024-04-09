# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

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
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02,
                                                          collision_method=cm)
        sim.run(0)
        pickling_check(cm)

    def test_embed(self, small_snap, simulation_factory, cls, init_args):
        sim = simulation_factory(small_snap)
        cm = cls(period=1, embedded_particles=hoomd.filter.All(), **init_args)
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02,
                                                          collision_method=cm)

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
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02,
                                                          collision_method=cm)

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
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02,
                                                          collision_method=cm)

        # test that one step can run without error with only solvent
        sim.run(1)

        # test that one step can run without error with embedded particles
        if "kT" not in init_args:
            init_args["kT"] = 1.0
        sim.operations.integrator.collision_method = cls(
            period=1, embedded_particles=hoomd.filter.All(), **init_args)
        sim.run(1)
