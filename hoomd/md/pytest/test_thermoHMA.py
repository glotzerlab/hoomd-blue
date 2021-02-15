import hoomd


def test_before_attaching():
    filt = hoomd.filter.All()
    thermoHMA = hoomd.md.compute.ThermoHMA(filt, 1.0)
    assert thermoHMA._filter == filt
    assert thermoHMA.temperature == 1.0
    assert thermoHMA.harmonic_pressure == 0.0
    assert thermoHMA.potential_energy is None
    assert thermoHMA.pressure is None

    thermoHMA = hoomd.md.compute.ThermoHMA(filt, 2.5, 0.6)
    assert thermoHMA.temperature == 2.5
    assert thermoHMA.harmonic_pressure == 0.6


def test_after_attaching(simulation_factory, two_particle_snapshot_factory):
    filt = hoomd.filter.All()
    thermoHMA = hoomd.md.compute.ThermoHMA(filt, 1.0)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.add(thermoHMA)
    assert len(sim.operations.computes) == 1
    sim.run(0)
    assert isinstance(thermoHMA.potential_energy, float)
    assert isinstance(thermoHMA.pressure, float)

    sim.operations.remove(thermoHMA)
    assert len(sim.operations.computes) == 0
    assert thermoHMA.potential_energy is None
    assert thermoHMA.pressure is None
