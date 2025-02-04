# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Define particle filter set operations.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    filter1 = hoomd.filter.All()
    filter2 = hoomd.filter.Tags([0])
"""

from hoomd.filter.filter_ import ParticleFilter
from hoomd import _hoomd


class _ParticleFilterSetOperations(ParticleFilter):
    """An abstract class for particle filters with set operations.

    Should not be instantiated directly.
    """

    @property
    def _cpp_cls_name(self):
        """The name of the C++ class in the `_hoomd` module.

        Used for Python class's inheritance.
        """
        raise NotImplementedError

    @property
    def _symmetric(self):
        """Whether the class implements a symmetric set operation.

        Determines behavior of __eq__.
        """
        raise NotImplementedError

    def __init__(self, f, g):
        ParticleFilter.__init__(self)

        if f == g:
            raise ValueError("Cannot use same filter for {}"
                             "".format(self.__class__.__name__))
        else:
            self._f = f
            self._g = g
        # Grab the C++ class constructor for the set operation using the class
        # variable _cpp_cls_name
        getattr(_hoomd, self._cpp_cls_name).__init__(self, f, g)

    def __hash__(self):
        return hash(hash(self._f) + hash(self._g))

    def __eq__(self, other):
        if self._symmetric:
            return type(self) is type(other) and \
                (self._f == other._f or self._f == other._g) and \
                (self._g == other._g or self._g == other._f)
        else:
            return type(self) is type(other) and \
                self._f == other._f and self._g == other._g

    def __reduce__(self):
        """Enable (deep)copying and pickling of set based particle filters."""
        return (type(self), (self._f, self._g))


class SetDifference(_ParticleFilterSetOperations,
                    _hoomd.ParticleFilterSetDifference):
    r"""Set difference operation.

    Args:
        f (ParticleFilter): First set in the difference.
        g (ParticleFilter): Second set in the difference.

    `SetDifference` is a composite filter. It selects particles in the set
    difference :math:`f \setminus g`.

    Base: `ParticleFilter`

    .. rubric:: Example:

    .. code-block:: python

        set_difference = hoomd.filter.SetDifference(filter1, filter2)
    """
    _cpp_cls_name = 'ParticleFilterSetDifference'
    _symmetric = False


class Union(_ParticleFilterSetOperations, _hoomd.ParticleFilterUnion):
    r"""Set union operation.

    Args:
        f (ParticleFilter): First set in the union.
        g (ParticleFilter): Second set in the union.

    `Union` is a composite filter. It selects particles in the set
    union :math:`f \cup g`.

    Base: `ParticleFilter`

    .. rubric:: Example:

    .. code-block:: python

        union = hoomd.filter.Union(filter1, filter2)
    """
    _cpp_cls_name = 'ParticleFilterUnion'
    _symmetric = True


class Intersection(_ParticleFilterSetOperations,
                   _hoomd.ParticleFilterIntersection):
    r"""Set intersection operation.

    Args:
        f (ParticleFilter): First set in the intersection.
        g (ParticleFilter): Second set in the intersection.

    `Intersection` is a composite filter. It selects particles in the set
    intersection :math:`f \cap g`.

    Base: `ParticleFilter`

    .. rubric:: Example:

    .. code-block:: python

        intersection = hoomd.filter.Intersection(filter1, filter2)
    """
    _cpp_cls_name = 'ParticleFilterIntersection'
    _symmetric = True
