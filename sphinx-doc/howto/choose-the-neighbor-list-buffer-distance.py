import hoomd

kT = 1.5
sample_steps = 1000

# Prepare a MD simulation.
device = hoomd.device.auto_select()
simulation = hoomd.Simulation(device=device)
simulation.create_state_from_gsd(filename='spheres.gsd')
simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)

neighbor_list = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=neighbor_list)
lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
lj.r_cut[('A', 'A')] = 2.5

bussi = hoomd.md.methods.thermostats.Bussi(kT=kT)
constant_volume = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(),
                                                  thermostat=bussi)
simulation.operations.integrator = hoomd.md.Integrator(
    dt=0.001, methods=[constant_volume], forces=[lj])

# Complete GPU kernel autotuning before making sensitive timing measurements.
if isinstance(device, hoomd.device.GPU):
    simulation.run(0)
    while not simulation.operations.is_tuning_complete:
        simulation.run(1000)

for buffer in [0, 0.05, 0.1, 0.2, 0.3]:
    neighbor_list.buffer = buffer
    simulation.run(sample_steps)
    device.notice(f'buffer={buffer}: TPS={simulation.tps:0.3g}, '
                  f'num_builds={neighbor_list.num_builds}')
