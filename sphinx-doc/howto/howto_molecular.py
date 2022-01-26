# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import gsd.hoomd

snapshot = gsd.hoomd.Snapshot()

# Place a polymer in the box.
snapshot.particles.N = 5
snapshot.particles.position = [[-2, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0],
                               [2, 0, 0]]
snapshot.particles.types = ['A']
snapshot.particles.typeid = [0] * 5
snapshot.configuration.box = [20, 20, 20, 0, 0, 0]

# Connect particles with bonds.
snapshot.bonds.N = 4
snapshot.bonds.types = ['A-A']
snapshot.bonds.typeid = [0] * 4
snapshot.bonds.group = [[0, 1], [1, 2], [2, 3], [3, 4]]

with gsd.hoomd.open(name='howto_molecular.gsd', mode='xb') as f:
    f.append(snapshot)

# Apply the harmonic potential on the bonds.
harmonic = hoomd.md.bond.Harmonic()
harmonic.params['A-A'] = dict(k=100, r0=1.0)

# Perform the MD simulation.
sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
sim.create_state_from_gsd(filename='howto_molecular.gsd')
langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
integrator = hoomd.md.Integrator(dt=0.005,
                                 methods=[langevin],
                                 forces=[harmonic])
gsd_writer = hoomd.write.GSD(filename='howto_molecular_trajectory.gsd',
                             trigger=hoomd.trigger.Periodic(1000),
                             mode='xb')
sim.operations.integrator = integrator
sim.operations.writers.append(gsd_writer)
sim.run(10e3)
