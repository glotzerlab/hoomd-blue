import hoomd
import io


def test_table_pressure(simulation_factory, two_particle_snapshot_factory):
    """Test that write.table can log MD pressure values."""
    thermo = hoomd.md.compute.ThermodynamicQuantities(hoomd.filter.All())
    snap = two_particle_snapshot_factory()
    if snap.communicator.rank == 0:
        snap.particles.velocity[:] = [[-2, 0, 0], [2, 0, 0]]
    sim = simulation_factory(snap)
    sim.operations.add(thermo)

    integrator = hoomd.md.Integrator(dt=0.0001)
    integrator.methods.append(
        hoomd.md.methods.NVT(hoomd.filter.All(), tau=1, kT=1))
    sim.operations.integrator = integrator

    logger = hoomd.logging.Logger(categories=['scalar'])
    logger.add(thermo)
    output = io.StringIO("")
    table_writer = hoomd.write.Table(1, logger, output)
    sim.operations.writers.append(table_writer)

    sim.run(1)
