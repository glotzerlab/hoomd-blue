# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import gsd.hoomd
import numpy
import math
import itertools
import datetime

sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=1)
sim.create_state_from_gsd(filename='multi_chain_Xi_0.1_NVT.gsd')

# Apply the harmonic potential on the bonds.
harmonic = hoomd.md.bond.Harmonic()
harmonic.params['polybond'] = dict(k=100, r0=1.0)

# Apply the harmonic potential on the angles.
harmonic_angle = hoomd.md.angle.Harmonic()
harmonic_angle.params['polyangle'] = dict(k=100, t0=2.0944)

# Apply the PCND potential on the chain.
PCND = hoomd.md.angle.PCND()
PCND.params['polyangle'] = dict(Xi=0.1, Tau=100.0)

# Lennard-Jones interactions
cell = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
lj.r_cut[('A', 'A')] = 2.5

# Perform the MD simulation.
NVT = hoomd.md.methods.NVT(filter=hoomd.filter.All(), kT=1.5, tau=1.0)
integrator = hoomd.md.Integrator(dt=0.005,
                                 methods=[NVT],
                                 forces=[harmonic, harmonic_angle, lj, PCND])
sim.operations.integrator = integrator
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)
logger = hoomd.logging.Logger()
logger.add(thermodynamic_properties)
logger_tps = hoomd.logging.Logger(categories=['scalar', 'string'])
logger_tps.add(sim, quantities=['timestep', 'tps'])
class Status():

    def __init__(self, sim):
            self.sim = sim

    @property
    def seconds_remaining(self):
        try:
            return (self.sim.final_timestep - self.sim.timestep) / self.sim.tps
        except ZeroDivisionError:
            return 0

    @property
    def etr(self):
            return str(datetime.timedelta(seconds=self.seconds_remaining))

status = Status(sim)
logger_tps[('Status', 'etr')] = (status, 'etr', 'string')
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=1000),
                          logger = logger_tps)
sim.operations.writers.append(table)
gsd_writer = hoomd.write.GSD(filename='multi_chain_trajectory_Xi_0.1_NVT.gsd',
                             trigger=hoomd.trigger.Periodic(1000),
                             mode='xb',
                             filter=hoomd.filter.All(),
                             dynamic=['property', 'momentum'],
                             log = logger)
sim.operations.writers.append(gsd_writer)
sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)
sim.run(10e5)
