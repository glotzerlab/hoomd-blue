import hoomd

# Step 1: Use different types for stationary and mobile particles.
simulation = hoomd.util.make_example_simulation(
    particle_types=['A', 'A_no_motion'])

hpmc_ellipsoid = hoomd.hpmc.integrate.Ellipsoid()
hpmc_ellipsoid.shape['A'] = dict(a=0.5, b=0.25, c=0.125)

# Step 2: Set the move sizes of the stationary type to 0.
hpmc_ellipsoid.d['A_no_motion'] = 0
hpmc_ellipsoid.a['A_no_motion'] = 0

# Step 3: Set the shape of the stationary type accordingly.
hpmc_ellipsoid.shape['A_no_motion'] = hpmc_ellipsoid.shape['A']

simulation.operations.integrator = hpmc_ellipsoid

simulation.run(100)
