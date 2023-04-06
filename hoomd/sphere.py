# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Implement Sphere."""

import numpy as np
import hoomd._hoomd as _hoomd


class Sphere:
    """Define sphere.

    Args:
        R (float): radius of the bounding sphere :math:`[\\mathrm{length}]`.

    """

    # Constructors
    def __init__(self, R=1.0):
        if R < 0:
            raise ValueError(
                "Cannot set radius of the confining sphere to a negative number"
            )
        self._cpp_obj = _hoomd.Sphere(R)

    @classmethod
    def _from_cpp(cls, cpp_obj):
        """Wrap a C++ SphereDim.

        Does not copy the C++ object.
        """
        s = Sphere(1)
        s._cpp_obj = cpp_obj
        return s

    @classmethod
    def from_sphere(cls, sphere):
        R"""Initialize a Sphere instance from a sphere-like object.

        Args:
            sphere:
                A sphere-like object

        Returns:
            :class:`hoomd.Sphere`: The resulting sphere object.
        """
        try:
            # Handles hoomd.sphere.Sphere and objects with attributes
            R = sphere.R
        except AttributeError:
            try:
                # Handle dictionary-like
                R = sphere['R']
            except (IndexError, KeyError, TypeError):
                raise ValueError("List-like objects must have length 1 to be "
                                 "converted to freud.sphere.Sphere.")
        except:  # noqa
            raise

        return cls(R=R)

    @property
    def R(self):  # noqa: N802: Allow function name
        """float: The radius of the confining sphere \
        :math:`[\\mathrm{length}]`."""
        return self._cpp_obj.getR()

    @R.setter
    def R(self, value):  # noqa: N802: Allow function name
        R = float(value)
        self._cpp_obj.setR(R)

    @property
    def volume(self):
        """float: Volume of the sphere surface.

        When setting volume the aspect ratio of the sphere is maintained while
        the lengths are changed.
        """
        return self._cpp_obj.getVolume()

    @volume.setter
    def volume(self, volume):
        self.scale((volume / self.volume)**(0.5))

    def scale(self, s):
        R"""Scale sphere dimensions.

        Scales the sphere in place by the given scale factors. Tilt factors are
        not modified.

        Args:
            s (float or list[float]): scale factors in each dimension. If a
                single float is given then scale all dimensions by s; otherwise,
                s must be a sequence of 3 values used to scale each dimension.

        Returns:
            ``self``
        """
        s = np.asarray(s, dtype=float)
        self.R *= s
        return self

    # Magic Methods
    def __repr__(self):
        """Executable representation of the object."""
        return "hoomd.sphere.Sphere(R={})".format(self.R)

    def __eq__(self, other):
        """Test if spheres are equal."""
        if not isinstance(other, Sphere):
            return NotImplemented
        return self._cpp_obj == other._cpp_obj

    def __neq__(self, other):
        """Test if spheres are not equal."""
        if not isinstance(other, Sphere):
            return NotImplemented
        return self._cpp_obj != other._cpp_obj

    def __reduce__(self):
        """Reduce values to picklable format."""
        return (type(self), *self.R)
