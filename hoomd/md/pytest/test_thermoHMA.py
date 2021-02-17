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
    sim.always_compute_pressure = True
    assert len(sim.operations.computes) == 1
    sim.run(0)
    assert isinstance(thermoHMA.potential_energy, float)
    assert isinstance(thermoHMA.pressure, float)

    sim.operations.remove(thermoHMA)
    assert len(sim.operations.computes) == 0
    assert thermoHMA.potential_energy is None
    assert thermoHMA.pressure is None


def test_logging(simulation_factory, two_particle_snapshot_factory):
    filt = hoomd.filter.All()
    T = 1.0
    thermoHMA = hoomd.md.compute.ThermoHMA(filt, T)
    snap = two_particle_snapshot_factory()
    if snap.exists:
        snap.particles.velocity[:] = [[-2, 0, 0], [2, 0, 0]]
    sim = simulation_factory(snap)
    sim.always_compute_pressure = True
    sim.operations.add(thermoHMA)

    integrator = hoomd.md.Integrator(dt=0.0001)
    integrator.methods.append(hoomd.md.methods.NVT(filt, tau=1, kT=T))
    sim.operations.integrator = integrator

    log = hoomd.logging.Logger()
    log += thermoHMA
    for _ in range(5):
        sim.run(5)
        for key, (val, _) in log.log()['md']['compute']['ThermoHMA']['state']['__params__'].items():
            assert val == getattr(thermoHMA, key)
