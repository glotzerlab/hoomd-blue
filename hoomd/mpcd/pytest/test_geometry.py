# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.conftest import pickling_check
import pytest


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


class TestParallelPlates:

    def test_default_init(self, simulation_factory, snap):
        geom = hoomd.mpcd.geometry.ParallelPlates(H=4.0)
        assert geom.H == 4.0
        assert geom.V == 0.0
        assert geom.no_slip

        sim = simulation_factory(snap)
        geom._attach(sim)
        assert geom.H == 4.0
        assert geom.V == 0.0
        assert geom.no_slip

    def test_nondefault_init(self, simulation_factory, snap):
        geom = hoomd.mpcd.geometry.ParallelPlates(H=5.0, V=1.0, no_slip=False)
        assert geom.H == 5.0
        assert geom.V == 1.0
        assert not geom.no_slip

        sim = simulation_factory(snap)
        geom._attach(sim)
        assert geom.H == 5.0
        assert geom.V == 1.0
        assert not geom.no_slip

    def test_pickling(self, simulation_factory, snap):
        geom = hoomd.mpcd.geometry.ParallelPlates(H=4.0)
        pickling_check(geom)

        sim = simulation_factory(snap)
        geom._attach(sim)
        pickling_check(geom)


class TestPlanarPore:

    def test_default_init(self, simulation_factory, snap):
        geom = hoomd.mpcd.geometry.PlanarPore(H=4.0, L=5.0)
        assert geom.H == 4.0
        assert geom.L == 5.0
        assert geom.no_slip

        sim = simulation_factory(snap)
        geom._attach(sim)
        assert geom.H == 4.0
        assert geom.L == 5.0
        assert geom.no_slip

    def test_nondefault_init(self, simulation_factory, snap):
        geom = hoomd.mpcd.geometry.PlanarPore(H=5.0, L=4.0, no_slip=False)
        assert geom.H == 5.0
        assert geom.L == 4.0
        assert not geom.no_slip

        sim = simulation_factory(snap)
        geom._attach(sim)
        assert geom.H == 5.0
        assert geom.L == 4.0
        assert not geom.no_slip

    def test_pickling(self, simulation_factory, snap):
        geom = hoomd.mpcd.geometry.PlanarPore(H=4.0, L=5.0)
        pickling_check(geom)

        sim = simulation_factory(snap)
        geom._attach(sim)
        pickling_check(geom)
