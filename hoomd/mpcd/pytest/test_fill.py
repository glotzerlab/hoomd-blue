# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest

import hoomd
from hoomd.conftest import pickling_check


@pytest.fixture
def snap():
    snap_ = hoomd.Snapshot()
    if snap_.communicator.rank == 0:
        snap_.configuration.box = [10, 10, 10, 0, 0, 0]
        snap_.particles.N = 0
        snap_.particles.types = ["A"]
        snap_.mpcd.N = 1
        snap_.mpcd.types = ["A"]
    return snap_


@pytest.mark.parametrize(
    "cls, init_args",
    [
        (hoomd.mpcd.geometry.CosineChannel, {
            "amplitude": 4.0,
            "repeat_length": 20.0,
            "separation": 2.0
        }),
        (hoomd.mpcd.geometry.CosineExpansionContraction, {
            "expansion_separation": 8.0,
            "contraction_separation": 4.0,
            "repeat_length": 20.0,
        }),
        (hoomd.mpcd.geometry.ParallelPlates, {
            "separation": 8.0
        }),
        (hoomd.mpcd.geometry.PlanarPore, {
            "separation": 8.0,
            "length": 10.0
        }),
        (hoomd.mpcd.geometry.Sphere, {
            "radius": 4.0
        }),
    ],
    ids=[
        "CosineChannel", "CosineExpansionContraction", "ParallelPlates",
        "PlanarPore", "Sphere"
    ],
)
class TestGeometryFiller:

    def test_create_and_attributes(self, simulation_factory, snap, cls,
                                   init_args):
        geom = cls(**init_args)
        filler = hoomd.mpcd.fill.GeometryFiller(type="A",
                                                density=5.0,
                                                kT=1.0,
                                                geometry=geom)

        assert filler.geometry is geom
        assert filler.type == "A"
        assert filler.density == 5.0
        assert isinstance(filler.kT, hoomd.variant.Constant)
        assert filler.kT(0) == 1.0

        sim = simulation_factory(snap)
        filler._attach(sim)
        assert filler.geometry is geom
        assert filler.type == "A"
        assert filler.density == 5.0
        assert isinstance(filler.kT, hoomd.variant.Constant)
        assert filler.kT(0) == 1.0

        filler.density = 3.0
        filler.kT = hoomd.variant.Ramp(2.0, 1.0, 0, 10)
        assert filler.geometry is geom
        assert filler.type == "A"
        assert filler.density == 3.0
        assert isinstance(filler.kT, hoomd.variant.Ramp)
        assert filler.kT(0) == 2.0

    def test_run(self, simulation_factory, snap, cls, init_args):
        filler = hoomd.mpcd.fill.GeometryFiller(type="A",
                                                density=5.0,
                                                kT=1.0,
                                                geometry=cls(**init_args))
        sim = simulation_factory(snap)
        ig = hoomd.mpcd.Integrator(dt=0.1, virtual_particle_fillers=[filler])
        sim.operations.integrator = ig
        sim.run(1)

    def test_pickling(self, simulation_factory, snap, cls, init_args):
        geom = cls(**init_args)
        filler = hoomd.mpcd.fill.GeometryFiller(type="A",
                                                density=5.0,
                                                kT=1.0,
                                                geometry=geom)
        pickling_check(filler)

        sim = simulation_factory(snap)
        filler._attach(sim)
        pickling_check(filler)
