# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""External fields for Monte Carlo.

Define :math:`U_{\\mathrm{external},i}` for use with
`hoomd.hpmc.integrate.HPMCIntegrator`. Add external
potential instances to your integrator's
`external_potentials <hpmc.integrate.HPMCIntegrator.external_potentials>` list
to apply it during the simulation.

Note:
    The following class types can not be added to the ``external_potentials``
    list. You may set one of these in the
    `external_potential <hpmc.integrate.HPMCIntegrator.external_potential>`
    attribute.

    * `user.CPPExternalPotential`
    * `field.Harmonic`
    * `wall.WallPotential`

    In HOOMD-blue 5.0.0, `field.Harmonic` and `wall.WallPotential` will be
    replaced by similar classes that do support ``external_potentials``.
"""

from . import field
from . import wall
from .external import External
from .linear import Linear
