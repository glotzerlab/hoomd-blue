# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import io
import numpy


def test_table_pressure(simulation_factory, two_particle_snapshot_factory):
    """Test that write.table can log MD pressure values."""
    thermo = hoomd.md.compute.ThermodynamicQuantities(hoomd.filter.All())
    snap = two_particle_snapshot_factory()
    if snap.communicator.rank == 0:
        snap.particles.velocity[:] = [[-2, 0, 0], [2, 0, 0]]
    sim = simulation_factory(snap)
    sim.operations.add(thermo)

    integrator = hoomd.md.Integrator(dt=0.0)
    thermostat = hoomd.md.methods.thermostats.Bussi(kT=1.0)
    integrator.methods.append(
        hoomd.md.methods.ConstantVolume(hoomd.filter.All(), thermostat))
    sim.operations.integrator = integrator

    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(thermo, quantities=['pressure'])
    output = io.StringIO("")
    table_writer = hoomd.write.Table(1, logger, output)
    sim.operations.writers.append(table_writer)

    sim.run(1)

    ideal_gas_pressure = (2 * thermo.translational_kinetic_energy / 3
                          / sim.state.box.volume)
    if sim.device.communicator.rank == 0:
        output_lines = output.getvalue().split('\n')
        numpy.testing.assert_allclose(float(output_lines[1]),
                                      ideal_gas_pressure,
                                      rtol=0.2)
