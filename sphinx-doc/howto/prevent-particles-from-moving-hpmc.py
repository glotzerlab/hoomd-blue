import hoomd

simulation = hoomd.util.make_example_simulation(
    particle_types=['A', 'A_no_motion'])

hpmc_ellipsoid = hoomd.hpmc.integrate.Ellipsoid()
hpmc_ellipsoid.shape['A'] = dict(a=0.5, b=0.25, c=0.125)

# Particles of type A_no_motion have the same shape as type A and do not move.
hpmc_ellipsoid.shape['A_no_motion'] = hpmc_ellipsoid.shape['A']
hpmc_ellipsoid.d['A_no_motion'] = 0
hpmc_ellipsoid.a['A_no_motion'] = 0

simulation.operations.integrator = hpmc_ellipsoid

simulation.run(100)
