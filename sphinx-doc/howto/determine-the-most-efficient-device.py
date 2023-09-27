import hoomd
import argparse

kT = 1.2

# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='CPU')
parser.add_argument('--replicate', default=1, type=int)
parser.add_argument('--steps', default=10_000, type=int)
args = parser.parse_args()

# Create WCA MD simulation
device = getattr(hoomd.device, args.device)()
simulation = hoomd.Simulation(device=device, seed=1)
simulation.create_state_from_gsd(filename='spheres.gsd')
simulation.state.replicate(
    nx=args.replicate,
    ny=args.replicate,
    nz=args.replicate,
)
simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=kT)

cell = hoomd.md.nlist.Cell(buffer=0.2)
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A', 'A')] = dict(sigma=1, epsilon=1)
lj.r_cut[('A', 'A')] = 2**(1 / 6)

constant_volume = hoomd.md.methods.ConstantVolume(
    filter=hoomd.filter.All(),
    thermostat=hoomd.md.methods.thermostats.Bussi(kT=kT))

simulation.operations.integrator = hoomd.md.Integrator(
    dt=0.001, methods=[constant_volume], forces=[lj])

# Wait until GPU kernel parameter autotuning is complete.
if args.device == 'GPU':
    simulation.run(100)
    while not simulation.operations.is_tuning_complete:
        simulation.run(100)

# Warm up memory caches and pre-computed quantities.
simulation.run(args.steps)

# Run the benchmark and print the performance.
simulation.run(args.steps)
device.notice(f'TPS: {simulation.tps:0.5g}')
