# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
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


class TestCosineChannel:

    def test_default_init(self, simulation_factory, snap):
        wavenumber = 2.0 * np.pi / 10.0
        geom = hoomd.mpcd.geometry.CosineChannel(amplitude=4.0,
                                                 wavenumber=wavenumber,
                                                 separation=4.0)
        assert geom.amplitude == 4.0
        assert geom.wavenumber == wavenumber
        assert geom.separation == 4.0
        assert geom.no_slip

        sim = simulation_factory(snap)
        geom._attach(sim)
        assert geom.amplitude == 4.0
        assert geom.wavenumber == wavenumber
        assert geom.separation == 4.0
        assert geom.no_slip

    def test_nondefault_init(self, simulation_factory, snap):
        wavenumber = 2.0 * np.pi / 10.0
        geom = hoomd.mpcd.geometry.CosineChannel(amplitude=4.0,
                                                 wavenumber=wavenumber,
                                                 separation=4.0,
                                                 no_slip=False)
        assert geom.amplitude == 4.0
        assert geom.wavenumber == wavenumber
        assert geom.separation == 4.0
        assert not geom.no_slip

        sim = simulation_factory(snap)
        geom._attach(sim)
        assert geom.amplitude == 4.0
        assert geom.wavenumber == wavenumber
        assert geom.separation == 4.0
        assert not geom.no_slip

    def test_pickling(self, simulation_factory, snap):
        geom = hoomd.mpcd.geometry.CosineChannel(amplitude=4.0,
                                                 wavenumber=2.0 * np.pi / 10.0,
                                                 separation=4.0)
        pickling_check(geom)

        sim = simulation_factory(snap)
        geom._attach(sim)
        pickling_check(geom)


class TestCosineExpansionContraction:

    def test_default_init(self, simulation_factory, snap):
        wavenumber = 2.0 * np.pi / 10.0
        geom = hoomd.mpcd.geometry.CosineExpansionContraction(
            expansion_separation=4,
            contraction_separation=2,
            wavenumber=wavenumber)

        assert geom.expansion_separation == 4.0
        assert geom.contraction_separation == 2.0
        assert geom.wavenumber == wavenumber
        assert geom.no_slip

        sim = simulation_factory(snap)
        geom._attach(sim)
        assert geom.expansion_separation == 4.0
        assert geom.contraction_separation == 2.0
        assert geom.wavenumber == wavenumber
        assert geom.no_slip

    def test_nondefault_init(self, simulation_factory, snap):
        wavenumber = 2.0 * np.pi / 10.0
        geom = hoomd.mpcd.geometry.CosineExpansionContraction(
            expansion_separation=4,
            contraction_separation=2,
            wavenumber=wavenumber,
            no_slip=False)
        assert geom.expansion_separation == 4.0
        assert geom.contraction_separation == 2.0
        assert geom.wavenumber == wavenumber
        assert not geom.no_slip

        sim = simulation_factory(snap)
        geom._attach(sim)
        assert geom.expansion_separation == 4.0
        assert geom.contraction_separation == 2.0
        assert geom.wavenumber == wavenumber
        assert not geom.no_slip

    def test_pickling(self, simulation_factory, snap):
        wavenumber = 2.0 * np.pi / 10.0
        geom = hoomd.mpcd.geometry.CosineExpansionContraction(
            expansion_separation=4,
            contraction_separation=2,
            wavenumber=wavenumber)

        pickling_check(geom)

        sim = simulation_factory(snap)
        geom._attach(sim)
        pickling_check(geom)


class TestParallelPlates:

    def test_default_init(self, simulation_factory, snap):
        geom = hoomd.mpcd.geometry.ParallelPlates(separation=8.0)
        assert geom.separation == 8.0
        assert geom.speed == 0.0
        assert geom.no_slip

        sim = simulation_factory(snap)
        geom._attach(sim)
        assert geom.separation == 8.0
        assert geom.speed == 0.0
        assert geom.no_slip

    def test_nondefault_init(self, simulation_factory, snap):
        geom = hoomd.mpcd.geometry.ParallelPlates(separation=10.0,
                                                  speed=1.0,
                                                  no_slip=False)
        assert geom.separation == 10.0
        assert geom.speed == 1.0
        assert not geom.no_slip

        sim = simulation_factory(snap)
        geom._attach(sim)
        assert geom.separation == 10.0
        assert geom.speed == 1.0
        assert not geom.no_slip

    def test_pickling(self, simulation_factory, snap):
        geom = hoomd.mpcd.geometry.ParallelPlates(separation=8.0)
        pickling_check(geom)

        sim = simulation_factory(snap)
        geom._attach(sim)
        pickling_check(geom)


class TestPlanarPore:

    def test_default_init(self, simulation_factory, snap):
        geom = hoomd.mpcd.geometry.PlanarPore(separation=8.0, length=10.0)
        assert geom.separation == 8.0
        assert geom.length == 10.0
        assert geom.no_slip

        sim = simulation_factory(snap)
        geom._attach(sim)
        assert geom.separation == 8.0
        assert geom.length == 10.0
        assert geom.no_slip

    def test_nondefault_init(self, simulation_factory, snap):
        geom = hoomd.mpcd.geometry.PlanarPore(separation=10.0,
                                              length=8.0,
                                              no_slip=False)
        assert geom.separation == 10.0
        assert geom.length == 8.0
        assert not geom.no_slip

        sim = simulation_factory(snap)
        geom._attach(sim)
        assert geom.separation == 10.0
        assert geom.length == 8.0
        assert not geom.no_slip

    def test_pickling(self, simulation_factory, snap):
        geom = hoomd.mpcd.geometry.PlanarPore(separation=8.0, length=10.0)
        pickling_check(geom)

        sim = simulation_factory(snap)
        geom._attach(sim)
        pickling_check(geom)
