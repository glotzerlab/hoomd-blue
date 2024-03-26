import hoomd

simulation = hoomd.util.make_example_simulation(particle_types=['A', 'B'])
simulation.state.replicate(nx=3, ny=3, nz=3)

mc = hoomd.hpmc.integrate.Ellipsoid()
mc.shape['A'] = dict(a=1.0, b=1.0, c=0.25)
mc.shape['B'] = dict(a=1.0, b=1.0, c=0.5)
simulation.operations.integrator = mc

# loop over particle types and set ignore_statistics = True
for ignored_type in simulation.state.particle_types:
    mc.shape[ignored_type]['ignore_statistics'] = True

# loop over particle types to tune move sizes for
for tuned_type in simulation.state.particle_types:
    move_size_tuner = hoomd.hpmc.tune.MoveSize.scale_solver(
        100, ['a', 'd'], 0.2, [tuned_type])
    simulation.operations.add(move_size_tuner)
    mc.shape[tuned_type]['ignore_statistics'] = False
    simulation.run(1000)
    mc.shape[tuned_type]['ignore_statistics'] = True
    simulation.operations.remove(move_size_tuner)

# stop ignoring statistics after tuning
for ignored_type in simulation.state.particle_types:
    mc.shape[ignored_type]['ignore_statistics'] = False
