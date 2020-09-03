import hoomd

_thermo_qtys = [
    'temperature',
    'pressure',
    'pressureXX',
    'pressureXY',
    'pressureXZ',
    'pressureYY',
    'pressureYZ',
    'pressureZZ',
    'kineticEnergy',
    'translationalKineticEnergy',
    'rotationalKineticEnergy',
    'potentialEnergy',
    'NDOF',
    'translationalDOF',
    'rotationalDOF',
    'numParticles',
]

def test_attach_detach(simulation_factory, two_particle_snapshot_factory):
    # test before attaching to simulation
    group = hoomd.filter.All()
    thermo = hoomd.compute.ThermoQuantities(group)
    for qty in _thermo_qtys:
        assert getattr(thermo, qty) == None

    # make simulation and test state of operations
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.add(thermo)
    assert len(sim.operations.computes) == 1
    sim.operations.schedule()

    # make sure quantities are computable without failure
    for qty in _thermo_qtys:
        calc_qty = getattr(thermo, qty)
        assert type(calc_qty) == float

    # detach from simulation and test properties again
    sim.operations.remove(thermo)
    assert len(sim.operations.computes) == 0
    for qty in _thermo_qtys:
        assert getattr(thermo, qty) == None

