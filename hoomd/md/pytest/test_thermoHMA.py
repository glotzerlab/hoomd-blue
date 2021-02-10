import hoomd


def test_before_attaching():
    filt = hoomd.filter.All()
    thermoHMA = hoomd.md.compute.ThermoHMA(filt, 1.0)
    assert thermoHMA._filter == filt
    assert thermoHMA.temperature == 1.0
    assert thermoHMA.harmonicPressure == 0.0
    assert thermoHMA.potential_energyHMA is None
    assert thermoHMA.pressureHMA is None

    thermoHMA = hoomd.md.compute.ThermoHMA(filt, 2.5, 0.6)
    assert thermoHMA.temperature == 2.5
    assert thermoHMA.harmonicPressure == 0.6


def test_after_attaching(simulation_factory, two_particle_snapshot_factory):
    filt = hoomd.filter.All()
    thermoHMA = hoomd.md.compute.ThermoHMA(filt, 1.0)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.add(thermoHMA)
    assert len(sim.operations.computes) == 1
    sim.run(0)
    assert isinstance(thermoHMA.potential_energyHMA, float)
    assert isinstance(thermoHMA.pressureHMA, float)

    sim.operations.remove(thermoHMA)
    assert len(sim.operations.computes) == 0
    assert thermoHMA.potential_energyHMA is None
    assert thermoHMA.pressureHMA is None
