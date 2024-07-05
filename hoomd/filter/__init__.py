# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Particle filters.

Particle filters describe criteria to select subsets of particles in the
system for use by various operations throughout HOOMD. To maintain high
performance, filters are **not** re-evaluated on every use. Instead, each unique
particular filter (defined by the class name and hash) is mapped to a **group**,
an internally maintained list of the selected particles. Subsequent uses of the
same particle filter specification (in the same `hoomd.Simulation`) will resolve
to the same group *and the originally selected particles*, **even if the state
of the system has changed.**

Groups are not completely static. HOOMD-blue automatically re-evaluates the
filter specifications and updates the group membership whenever the number of
particles in the simulation changes. Use `hoomd.update.FilterUpdater` to
manually trigger updates to group membership.

For molecular dynamics simulations, each group maintains a count of the number
of degrees of freedom given to the group by integration methods. This count is
used by `hoomd.md.compute.ThermodynamicQuantities` and the integration methods
themselves to compute the kinetic temperature. See
`hoomd.State.update_group_dof` for details on when HOOMD-blue updates this
count.
"""

import typing

from hoomd.filter.filter_ import ParticleFilter
from hoomd.filter.all_ import All
from hoomd.filter.null import Null
from hoomd.filter.rigid import Rigid
from hoomd.filter.set_ import Intersection, SetDifference, Union
from hoomd.filter.tags import Tags
from hoomd.filter.type_ import Type
from hoomd.filter.custom import CustomFilter

filter_like = typing.Union[ParticleFilter, CustomFilter]
"""
An object that acts like a particle filter.

Either a subclass of `ParticleFilter` or `CustomFilter`.
"""
