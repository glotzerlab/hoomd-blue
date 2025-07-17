# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""A `Variant` object represents a function of the time step. Some
operations accept `Variant` values for certain parameters, such as the
``kT`` parameter to `hoomd.md.methods.thermostats.Bussi`.

See `Variant` for details on creating user-defined variants or use one of the
provided subclasses.

.. autodata:: variant_like

   Objects that are like a variant.

   Any subclass of `Variant` is accepted along with float instances and objects
   convertible to float. They are internally converted to variants of type
   `Constant` via ``Constant(float(a))`` where ``a`` is the float or float
   convertible object.

   Attributes that are `Variant` objects can be set via a `variant_like`
   object.
"""

from hoomd.variant.scalar import Variant, Constant, Ramp, Cycle, Power, variant_like
from hoomd.variant import box

__all__ = [
    "Constant",
    "Cycle",
    "Power",
    "Ramp",
    "Variant",
    "box",
    "variant_like",
]
