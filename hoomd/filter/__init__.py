# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Particle filters.

Particle filters describe criteria to select subsets of the particle in the
system for use by various operations throughout HOOMD. To maintain high
performance, filters are **not** re-evaluated on every use. Instead, each unique
particular filter (defined by the class name and hash) is mapped to a **group**,
an internally maintained list of the selected particles. Subsequent uses of the
same particle filter specification (in the same `hoomd.Simulation`) will resolve
to the same group *and the originally selected particles*, **even if the state
of the system has changed.**

Groups are not completely static. HOOMD-blue re-evaluates the filter
specifications and updates the group membership whenever the number of particles
in the simulation changes. Use `hoomd.update.FilterUpdater` to trigger updates
to groups.

For molecular dynamics simulations, each group maintains a count of the number
of degrees of freedom given to the group by integration methods. This count is
used by `hoomd.md.compute.ThermodynamicQuantities` and the integration methods
themselves to compute the kinetic temperature. See
`hoomd.State.update_group_dof` for details on when HOOMD-blue updates this
count.
"""

from hoomd.filter.filter_ import ParticleFilter
from hoomd.filter.all_ import All
from hoomd.filter.null import Null
from hoomd.filter.rigid import Rigid
from hoomd.filter.set_ import Intersection, SetDifference, Union
from hoomd.filter.tags import Tags
from hoomd.filter.type_ import Type
from hoomd.filter.custom import CustomFilter
