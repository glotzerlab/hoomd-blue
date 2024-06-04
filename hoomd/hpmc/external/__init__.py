# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""External fields for Monte Carlo.

Define :math:`U_{\\mathrm{external},i}` for use with
`hoomd.hpmc.integrate.HPMCIntegrator`. Assign an external field instance to
`hpmc.integrate.HPMCIntegrator.external_potential` to activate the potential.
"""

from . import user
from . import field
from . import wall
from .external import External
from .linear import Linear
