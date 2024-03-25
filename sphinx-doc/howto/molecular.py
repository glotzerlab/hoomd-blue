import hoomd
import gsd.hoomd

frame = gsd.hoomd.Frame()

# Place a polymer in the box.
frame.particles.N = 5
frame.particles.position = [[-2, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0],
                            [2, 0, 0]]
frame.particles.types = ['A']
frame.particles.typeid = [0] * 5
frame.configuration.box = [20, 20, 20, 0, 0, 0]

# Connect particles with bonds.
frame.bonds.N = 4
frame.bonds.types = ['A-A']
frame.bonds.typeid = [0] * 4
frame.bonds.group = [[0, 1], [1, 2], [2, 3], [3, 4]]

with gsd.hoomd.open(name='molecular.gsd', mode='x') as f:
    f.append(frame)

# Apply the harmonic potential on the bonds.
harmonic = hoomd.md.bond.Harmonic()
harmonic.params['A-A'] = dict(k=100, r0=1.0)

# Perform the MD simulation.
sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
sim.create_state_from_gsd(filename='molecular.gsd')
langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
integrator = hoomd.md.Integrator(dt=0.005,
                                 methods=[langevin],
                                 forces=[harmonic])
gsd_writer = hoomd.write.GSD(filename='molecular_trajectory.gsd',
                             trigger=hoomd.trigger.Periodic(1000),
                             mode='xb')
sim.operations.integrator = integrator
sim.operations.writers.append(gsd_writer)
sim.run(10e3)
