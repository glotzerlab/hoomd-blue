"""Particle filters.

Particle filters describe criteria to select subsets of the particle in the
system for use by various operations throughout HOOMD. To maintain high
performance, filters are **not** re-evaluated on every use. Instead, each unique
particular filter (defined by the class name and hash) is mapped to a **group**,
an internally maintained list of the selected particles. Subsequent uses of the
same particle filter specification (in the same `Simulation`) will resolve to
the same group *and the originally selected particles*, **even if the state of
the system has changed.**

Groups are not completely static. HOOMD-blue re-evaluates the filter
specifications and updates the group membership whenever the number of particles
in the simulation changes. A future release will include an operation that you
can schedule to periodically update groups on demand.

For molecular dynamics simulations, each group maintains a count of the number
of degrees of freedom given to the group by integration methods. This count is
used by `hoomd.md.compute.ThermodynamicQuantities` and the integration methods
themselves to compute the kinetic temperature. See
`hoomd.State.update_group_dof` for details on when HOOMD-blue updates this
count.
"""

from hoomd.filter.filter_ import ParticleFilter  # noqa
from hoomd.filter.all_ import All  # noqa
from hoomd.filter.set_ import Intersection, SetDifference, Union  # noqa
from hoomd.filter.tags import Tags  # noqa
from hoomd.filter.type_ import Type  # noqa
