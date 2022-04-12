# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Alchemical molecular dynamics.

Implements molecular dynamics simulations of an extended statistical
mechanical ensemble that includes alchemical degrees of freedom describing
particle attributes as thermodynamic variables.

Example::

    nvt = hoomd.md.methods.NVT(...)
    integrator.methods.append(nvt)
    ljg = hoomd.md.alchemy.pair.LJGauss(...)
    integrator.forces.append(ljg)
    sim.run(0)
    ar0 = ljg.create_alchemical_dof(('A', 'A'), 'r0')
    alchemostat = hoomd.md.alchemy.methods.NVT(
        period=period,
        alchemical_dof=[ar0],
        alchemical_kT=hoomd.variant.Constant(0.1),
    )
    integrator.methods.insert(0, alchemostat)
    sim.run(n_steps)
"""

from . import methods
from . import pair
