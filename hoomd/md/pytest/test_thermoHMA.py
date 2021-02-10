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
