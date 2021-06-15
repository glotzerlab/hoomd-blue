import hoomd
import math
from hoomd.logging import LoggerCategories
from hoomd.error import DataAccessError
from hoomd.conftest import logging_check
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


def test_logging(simulation_factory, two_particle_snapshot_factory):
    filt = hoomd.filter.All()
    T = 1.0
    thermoHMA = hoomd.md.compute.HarmonicAveragedThermodynamicQuantities(
        filt, T)
    snap = two_particle_snapshot_factory()
    if snap.communicator.rank == 0:
        snap.particles.velocity[:] = [[-2, 0, 0], [2, 0, 0]]
    sim = simulation_factory(snap)
    sim.always_compute_pressure = True
    sim.operations.add(thermoHMA)

    integrator = hoomd.md.Integrator(dt=0.0001)
    integrator.methods.append(hoomd.md.methods.NVT(filt, tau=1, kT=T))
    sim.operations.integrator = integrator

    log = hoomd.logging.Logger()
    log += thermoHMA
    sim.run(5)
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
