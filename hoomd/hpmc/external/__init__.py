# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""External potentials define :math:`U_{\\mathrm{external},i}` for use with
`hoomd.hpmc.integrate.HPMCIntegrator`. Add external
potential instances to your integrator's
`external_potentials <hpmc.integrate.HPMCIntegrator.external_potentials>` list
to apply it during the simulation.
"""

from .external import External
from .linear import Linear
from .harmonic import Harmonic
from .wall import WallPotential

__all__ = [
    "External",
    "Harmonic",
    "Linear",
    "WallPotential",
]
