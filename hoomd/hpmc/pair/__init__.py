# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Pair Potentials for Monte Carlo.

Define :math:`U_{\\mathrm{pair},ij}` for use with
`hoomd.hpmc.integrate.HPMCIntegrator`. Assign a pair potential instance to
`hpmc.integrate.HPMCIntegrator.pair_potential` to activate the potential.
"""

from . import user
