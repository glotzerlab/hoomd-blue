# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest

import hoomd
from hoomd.conftest import pickling_check


@pytest.fixture
def make_simulation(simulation_factory):

    def _make_simulation():
        snap = hoomd.Snapshot()
        if snap.communicator.rank == 0:
            snap.configuration.box = [20, 20, 20, 0, 0, 0]
            snap.particles.N = 1
            snap.particles.types = ["A"]
            snap.mpcd.N = 1
            snap.mpcd.types = ["A"]
        return simulation_factory(snap)

    return _make_simulation


def test_create(make_simulation):
    sim = make_simulation()
    ig = hoomd.mpcd.Integrator(dt=0.1)
    sim.operations.integrator = ig

    sim.run(0)
    assert ig._attached
    assert ig.cell_list is not None
    assert ig.streaming_method is None
    assert ig.collision_method is None


def test_collision_method(make_simulation):
    sim = make_simulation()
    collide = hoomd.mpcd.collide.StochasticRotationDynamics(period=1, angle=130)

    # check that constructor assigns right
    ig = hoomd.mpcd.Integrator(dt=0.1, collision_method=collide)
    sim.operations.integrator = ig
    assert ig.collision_method is collide
    sim.run(0)
    assert ig.collision_method is collide

    # clear out by setter
    ig.collision_method = None
    assert ig.collision_method is None
    sim.run(0)
    assert ig.collision_method is None

    # assign by setter
    ig.collision_method = collide
    assert ig.collision_method is collide
    sim.run(0)
    assert ig.collision_method is collide


def test_streaming_method(make_simulation):
    sim = make_simulation()
    stream = hoomd.mpcd.stream.Bulk(period=1)

    # check that constructor assigns right
    ig = hoomd.mpcd.Integrator(dt=0.1, streaming_method=stream)
    sim.operations.integrator = ig
    assert ig.streaming_method is stream
    sim.run(0)
    assert ig.streaming_method is stream

    # clear out by setter
    ig.streaming_method = None
    assert ig.streaming_method is None
    sim.run(0)
    assert ig.streaming_method is None

    # assign by setter
    ig.streaming_method = stream
    assert ig.streaming_method is stream
    sim.run(0)
    assert ig.streaming_method is stream


def test_virtual_particle_fillers(make_simulation):
    sim = make_simulation()
    geom = hoomd.mpcd.geometry.ParallelPlates(separation=8.0)
    filler = hoomd.mpcd.fill.GeometryFiller(
        type="A",
        density=5.0,
        kT=1.0,
        geometry=geom,
    )

    # check that constructor assigns right
    ig = hoomd.mpcd.Integrator(dt=0.1, virtual_particle_fillers=[filler])
    sim.operations.integrator = ig
    assert len(ig.virtual_particle_fillers) == 1
    assert filler in ig.virtual_particle_fillers
    sim.run(0)
    assert len(ig.virtual_particle_fillers) == 1
    assert filler in ig.virtual_particle_fillers

    # attach a second filler
    filler2 = hoomd.mpcd.fill.GeometryFiller(
        type="A",
        density=1.0,
        kT=1.0,
        geometry=geom,
    )
    ig.virtual_particle_fillers.append(filler2)
    assert len(ig.virtual_particle_fillers) == 2
    assert filler in ig.virtual_particle_fillers
    assert filler2 in ig.virtual_particle_fillers
    sim.run(0)
    assert len(ig.virtual_particle_fillers) == 2
    assert filler in ig.virtual_particle_fillers
    assert filler2 in ig.virtual_particle_fillers

    # make sure synced list is working right with non-empty list assignment
    ig.virtual_particle_fillers = [filler]
    assert len(ig.virtual_particle_fillers) == 1
    assert len(ig._cpp_obj.fillers) == 1

    ig.virtual_particle_fillers = []
    assert len(ig.virtual_particle_fillers) == 0
    sim.run(0)
    assert len(ig.virtual_particle_fillers) == 0


def test_mpcd_particle_sorter(make_simulation):
    sim = make_simulation()
    sorter = hoomd.mpcd.tune.ParticleSorter(trigger=1)

    # check that constructor assigns right
    ig = hoomd.mpcd.Integrator(dt=0.1, mpcd_particle_sorter=sorter)
    sim.operations.integrator = ig
    assert ig.mpcd_particle_sorter is sorter
    sim.run(0)
    assert ig.mpcd_particle_sorter is sorter

    # clear out by setter
    ig.mpcd_particle_sorter = None
    assert ig.mpcd_particle_sorter is None
    sim.run(0)
    assert ig.mpcd_particle_sorter is None

    # assign by setter
    ig.mpcd_particle_sorter = sorter
    assert ig.mpcd_particle_sorter is sorter
    sim.run(0)
    assert ig.mpcd_particle_sorter is sorter


def test_attach_and_detach(make_simulation):
    sim = make_simulation()
    ig = hoomd.mpcd.Integrator(dt=0.1)
    sim.operations.integrator = ig

    # make sure attach works even without collision and streaming methods
    sim.run(0)
    assert ig._attached
    assert ig.cell_list._attached
    assert ig.streaming_method is None
    assert ig.collision_method is None
    assert len(ig.virtual_particle_fillers) == 0
    assert ig.mpcd_particle_sorter is None

    # attach with both methods
    ig.streaming_method = hoomd.mpcd.stream.Bulk(period=1)
    ig.collision_method = hoomd.mpcd.collide.StochasticRotationDynamics(
        period=1, angle=130)
    ig.mpcd_particle_sorter = hoomd.mpcd.tune.ParticleSorter(trigger=1)
    sim.run(0)
    assert ig.streaming_method._attached
    assert ig.collision_method._attached
    assert ig.mpcd_particle_sorter._attached

    # detach with everything
    sim.operations._unschedule()
    assert not ig._attached
    assert not ig.cell_list._attached
    assert not ig.streaming_method._attached
    assert not ig.collision_method._attached
    assert not ig.mpcd_particle_sorter._attached


def test_pickling(make_simulation):
    ig = hoomd.mpcd.Integrator(dt=0.1)
    pickling_check(ig)

    sim = make_simulation()
    sim.operations.integrator = ig
    sim.run(0)
    pickling_check(ig)
