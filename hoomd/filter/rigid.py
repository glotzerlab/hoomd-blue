# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Define the Rigid filter."""

from hoomd.filter.filter_ import ParticleFilter
from hoomd._hoomd import ParticleFilterRigid


class Rigid(ParticleFilter, ParticleFilterRigid):
    """Select particles based on inclusion in rigid bodies.

    Args:
        flags (`tuple` [`str` ], optional):
            A tuple of strings of values "center", "constituent", or "free".
            These string flags specify what kinds of particles to filter:
            "center" will include central particles in a rigid body,
            "constituent" will include non-central particles in a rigid body,
            and "free" will include all particles not in a rigid body.
            Specifying all three is the same as `hoomd.filter.All`. The default
            is ``("center",)``

    Base: `ParticleFilter`

    .. rubric:: Examples:

    .. code-block:: python

        rigid_center_and_free = hoomd.filter.Rigid(flags=('center', 'free'))

    .. code-block:: python

        rigid_center = hoomd.filter.Rigid(flags=('center',))
    """

    def __init__(self, flags=("center",)):
        if not all(flag in {"center", "constituent", "free"} for flag in flags):
            raise ValueError(
                "Only allowed flags are 'center', 'constituent', and 'free'.")
        ParticleFilter.__init__(self)
        ParticleFilterRigid.__init__(self, flags)
        self._flags = flags

    def __hash__(self):
        """Return a hash of the filter parameters."""
        return hash(self._flags)

    def __eq__(self, other):
        """Test for equality between two particle filters."""
        return type(self) is type(other) and self._flags == other._flags

    def __reduce__(self):
        """Enable (deep)copying and pickling of `Rigid` particle filters."""
        return (type(self), (self._flags,))
