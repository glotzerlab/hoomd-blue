import hoomd

simulation = hoomd.util.make_example_simulation()

# Select mobile particles with a filter.
stationary_particles = hoomd.filter.Tags([0])
mobile_particles = hoomd.filter.SetDifference(hoomd.filter.All(),
                                              stationary_particles)

# Integrate the equations of motion of the mobile particles.
langevin = hoomd.md.methods.Langevin(filter=mobile_particles, kT=1.5)
simulation.operations.integrator = hoomd.md.Integrator(dt=0.001,
                                                       methods=[langevin])

simulation.run(100)
