# Copyright (c) 2009-2024 The Regents of the University of Michigan.
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
    r0_alchemical_dof = ljg.r0[('A', 'A')]
    alchemostat = hoomd.md.alchemy.methods.NVT(
        period=period,
        alchemical_dof=[r0_alchemical_dof],
        alchemical_kT=hoomd.variant.Constant(0.1),
    )
    integrator.methods.insert(0, alchemostat)
    sim.run(n_steps)

.. versionadded:: 3.1.0
"""

from . import methods
from . import pair
