import hoomd

simulation = hoomd.util.make_example_simulation()

# Include only particles that should move in the filter passed to the
# integration method.
particles_to_not_move = hoomd.filter.Tags([0])
particles_to_move = hoomd.filter.SetDifference(hoomd.filter.All(),
                                               particles_to_not_move)

langevin = hoomd.md.methods.Langevin(filter=particles_to_move, kT=1.5)
simulation.operations.integrator = hoomd.md.Integrator(dt=0.001,
                                                       methods=[langevin])

simulation.run(100)
