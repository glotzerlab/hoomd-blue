# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest


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
                "period": 5,
                "kT": 1.0,
            },
        ),
        (
            hoomd.mpcd.collide.StochasticRotationDynamics,
            {
                "period": 5,
                "angle": 90,
            },
        ),
    ],
    ids=["AndersenThermostat", "StochasticRotationDynamics"],
)
class TestCollisionMethod:

    def test_create(self, small_snap, simulation_factory, cls, init_args):
        sim = simulation_factory(small_snap)
        cm = cls(**init_args)
        ig = hoomd.mpcd.Integrator(dt=0.02, collision_method=cm)
        sim.operations.integrator = ig

        sim.run(0)
        assert ig.collision_method is cm
        assert cm.period == 5

        ig.collision_method = None
        sim.run(0)
        assert ig.collision_method is None

        ig.collision_method = cm
        sim.run(0)
        assert ig.collision_method is cm

    def test_embed(self, small_snap, simulation_factory, cls, init_args):
        sim = simulation_factory(small_snap)
        cm = cls(**init_args, embedded_particles=hoomd.filter.All())
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02,
                                                          collision_method=cm)

        sim.run(0)
        assert isinstance(cm.embedded_particles, hoomd.filter.All)

    def test_temperature(self, small_snap, simulation_factory, cls, init_args):
        sim = simulation_factory(small_snap)
        if "kT" not in init_args:
            init_args["kT"] = 1.0
            kT_required = False
        else:
            kT_required = True
        cm = cls(**init_args)
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02,
                                                          collision_method=cm)

        sim.run(0)

        cm.kT = hoomd.variant.Ramp(1.0, 2.0, 0, 10)
        sim.run(0)

        if not kT_required:
            cm.kT = None
            sim.run(0)

    def test_run(self, small_snap, simulation_factory, cls, init_args):
        sim = simulation_factory(small_snap)
        cm = cls(**init_args)
        sim.operations.integrator = hoomd.mpcd.Integrator(dt=0.02,
                                                          collision_method=cm)

        sim.run(1)

        if "kT" not in init_args:
            init_args["kT"] = 1.0
        sim.operations.integrator.collision_method = cls(
            **init_args, embedded_particles=hoomd.filter.All())
        sim.run(1)
