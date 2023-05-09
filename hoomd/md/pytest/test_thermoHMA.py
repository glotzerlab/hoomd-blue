# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import math
from hoomd.logging import LoggerCategories
from hoomd.error import DataAccessError
from hoomd.conftest import logging_check, autotuned_kernel_parameter_check
import pytest


def test_before_attaching():
    filt = hoomd.filter.All()
    thermoHMA = hoomd.md.compute.HarmonicAveragedThermodynamicQuantities(
        filt, 1.0)
    assert thermoHMA._filter == filt
    assert thermoHMA.kT == 1.0
    assert thermoHMA.harmonic_pressure == 0.0
    with pytest.raises(DataAccessError):
        thermoHMA.potential_energy
    with pytest.raises(DataAccessError):
        thermoHMA.pressure

    thermoHMA = hoomd.md.compute.HarmonicAveragedThermodynamicQuantities(
        filt, 2.5, 0.6)
    assert thermoHMA.kT == 2.5
    assert thermoHMA.harmonic_pressure == 0.6


def test_after_attaching(simulation_factory, two_particle_snapshot_factory):
    filt = hoomd.filter.All()
    thermoHMA = hoomd.md.compute.HarmonicAveragedThermodynamicQuantities(
        filt, 1.0)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.add(thermoHMA)
    assert len(sim.operations.computes) == 1
    sim.run(0)
    assert math.isnan(thermoHMA.pressure)
    sim.always_compute_pressure = True
    assert not math.isnan(thermoHMA.pressure)
    assert isinstance(thermoHMA.pressure, float)
    assert isinstance(thermoHMA.potential_energy, float)

    sim.operations.remove(thermoHMA)
    assert len(sim.operations.computes) == 0
    with pytest.raises(DataAccessError):
        thermoHMA.potential_energy
    with pytest.raises(DataAccessError):
        thermoHMA.pressure


def test_kernel_parameters(simulation_factory, two_particle_snapshot_factory):
    filter_ = hoomd.filter.All()
    thermo = hoomd.md.compute.HarmonicAveragedThermodynamicQuantities(filter_, 1.0)
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.add(thermo)

    # add table writer so the thermo triggers every step
    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(thermo)
    w = hoomd.write.Table(trigger=1, logger=logger)
    sim.operations.add(w)
    sim.run(0)

    # now check that kernels are autotuned
    autotuned_kernel_parameter_check(instance=thermo,
                                     activate=lambda: sim.run(1))


def test_logging():
    logging_check(
        hoomd.md.compute.HarmonicAveragedThermodynamicQuantities,
        ('md', 'compute'), {
            'potential_energy': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'pressure': {
                'category': LoggerCategories.scalar,
                'default': True
            }
        })
