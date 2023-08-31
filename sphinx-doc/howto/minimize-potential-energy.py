import hoomd

# Preparation: Create a MD simulation.
simulation = hoomd.util.make_example_simulation()

# Step 1: Use hoomd.md.minize.FIRE as the integrator.
constant_volume = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
fire = hoomd.md.minimize.FIRE(dt=0.001,
                              force_tol=1e-3,
                              angmom_tol=1e-3,
                              energy_tol=1e-6,
                              methods=[constant_volume])
simulation.operations.integrator = fire

# Step 2: Apply forces to the particles.
lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4))
lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
lj.r_cut[('A', 'A')] = 2.5

simulation.operations.integrator.forces = [lj]

# Step 3: Run simulation steps until the minimization converges.
while not fire.converged:
    simulation.run(100)
