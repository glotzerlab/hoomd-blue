# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Define quantities that vary over the simulation.

A `Variant` object represents a scalar function of the time step. Some
operations accept `Variant` values for certain parameters, such as the
``kT`` parameter to `hoomd.md.methods.thermostats.Bussi`.

See `Variant` for details on creating user-defined variants or use one of the
provided subclasses.
"""

from hoomd.variant.scalar import Variant
from hoomd.variant.scalar import Constant
from hoomd.variant.scalar import Ramp
from hoomd.variant.scalar import Cycle
from hoomd.variant.scalar import Power
from hoomd.variant.scalar import variant_like

from hoomd.variant import box
