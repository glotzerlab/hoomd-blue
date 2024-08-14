# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Manifolds.

A `Manifold` defines a lower dimensional manifold embedded in 3D space with an
implicit function :math:`F(x,y,z) = 0`. Use `Manifold` classes to define
positional constraints to a given set of particles with:

* `hoomd.md.methods.rattle`
* `hoomd.md.force.ActiveOnManifold`
"""

from hoomd.md import _md
from hoomd import _hoomd
from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyIf, to_type_converter
from hoomd.error import MutabilityError
from collections.abc import Sequence


# A manifold in hoomd reflects a Manifold in c++. It is responsible to define
# the manifold used for RATTLE integrators and the active force constraints.
class Manifold(_HOOMDBaseObject):
    r"""Base class manifold object.

    Warning:
        Users should not instantiate `Manifold` directly, but should
        instead instantiate one of its subclasses defining a specific manifold
        geometry.

    Warning:
        Only one manifold can be applied to a given method or active forces.
    """

    @staticmethod
    def _preprocess_unitcell(value):
        if isinstance(value, Sequence):
            if len(value) != 3:
                raise ValueError(
                    "Expected a single int or a sequence of three ints.")
            return tuple(value)
        else:
            return (value, value, value)

    def __eq__(self, other):
        """Test for equality."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in self._param_dict)

    def _setattr_param(self, attr, value):
        raise MutabilityError(attr)


class Cuboid(Manifold):
    r"""Cuboid manifold.

    Args:
        a (`tuple` [`float`, `float`, `float`]): edge lengths of the cuboid
            :math:`[\mathrm{length}]`.
        P (`tuple` [`float`, `float`, `float`]): point defining position of
            the cuboid center (default origin) :math:`[\mathrm{length}]`.

    `Cuboid` defines a cuboid:

    Example::

        cuboid1 = manifold.Cuboid(a=(10,10,10))
        cuboid2 = manifold.Cuboid(a=(5,6,7),P=(1,1,1))
    """

    def __init__(self, a, P=(0, 0, 0)):
        # initialize the base class
        param_dict = ParameterDict(
            a=(float, float, float),
            P=(float, float, float),
        )
        param_dict['P'] = P
        param_dict['a'] = a

        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _md.ManifoldCuboid(
            _hoomd.make_scalar3(self.a[0], self.a[1], self.a[2]), 
            _hoomd.make_scalar3(self.P[0], self.P[1], self.P[2]))

        super()._attach(self._simulation)

class Cylinder(Manifold):
    r"""Cylinder manifold.

    Args:
        r (float): radius of the cylinder constraint
          :math:`[\mathrm{length}]`.
        P (`tuple` [`float`, `float`, `float`]): point defining position of
            the cylinder axis (default origin) :math:`[\mathrm{length}]`.

    `Cylinder` defines a right circular cylinder along the z axis:

    .. math::
        F(x,y,z) = (x - P_x)^{2} + (y - P_y)^{2} - r^{2}

    Example::

        cylinder1 = manifold.Cylinder(r=10)
        cylinder2 = manifold.Cylinder(r=5,P=(1,1,1))
    """

    def __init__(self, r, P=(0, 0, 0)):
        # initialize the base class
        param_dict = ParameterDict(
            r=float(r),
            P=(float, float, float),
        )
        param_dict['P'] = P

        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _md.ManifoldZCylinder(
            self.r, _hoomd.make_scalar3(self.P[0], self.P[1], self.P[2]))

        super()._attach(self._simulation)


class Diamond(Manifold):
    r"""Triply periodic diamond manifold.

    Args:
        N (`tuple` [`int`, `int`, `int`] or `int`): number of unit cells in all
            3 directions.  :math:`[N_x, N_y, N_z]`. In case number of unit cells
            u in all direction the same (:math:`[u, u, u]`), use ``N = u``.
        epsilon (float): defines CMC companion of the Diamond surface (default
            0)

    `Diamond` defines a periodic diamond surface . The diamond (or
    Schwarz D) belongs to the family of triply periodic minimal surfaces:

    .. math::

        F(x,y,z) = \cos{\frac{2 \pi}{B_x} x} \cdot \cos{\frac{2 \pi}{B_y} y}
                   \cdot \cos{\frac{2 \pi}{B_z} z} - \sin{\frac{2 \pi}{B_x} x}
                   \cdot \sin{\frac{2 \pi}{B_y} y}
                   \cdot \sin{\frac{2 \pi}{B_z} z} - \epsilon

    is the nodal approximation of the diamond surface where
    :math:`[B_x,B_y,B_z]` is the periodicity length in the x, y and z direction.
    The periodicity length B is defined by the current box size L and the number
    of unit cells N :math:`B_i=\frac{L_i}{N_i}`.

    See Also:
        * `A. H. Schoen 1970 <https://ntrs.nasa.gov/citations/19700020472>`__
        * `P. J. F. Gandy et. al. 1999
          <https://doi.org/10.1016/S0009-2614(99)01000-3>`__
        * `H. G. von Schnering and R. Nesper 1991
          <https://doi.org/10.1007/BF01313411>`__

    Example::

        diamond1 = manifold.Diamond(N=1)
        diamond2 = manifold.Diamond(N=(1,2,2))
    """

    def __init__(self, N, epsilon=0):

        # store metadata
        param_dict = ParameterDict(
            N=OnlyIf(to_type_converter((int,) * 3),
                     preprocess=self._preprocess_unitcell),
            epsilon=float(epsilon),
        )
        param_dict['N'] = N
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _md.ManifoldDiamond(
            _hoomd.make_int3(self.N[0], self.N[1], self.N[2]), self.epsilon)

        super()._attach(self._simulation)


class Ellipsoid(Manifold):
    r"""Ellipsoid manifold.

    Args:
        a (float): length of the a-axis of the ellipsoidal constraint
            :math:`[\mathrm{length}]`.
        b (float): length of the b-axis of the ellipsoidal constraint
            :math:`[\mathrm{length}]`.
        c (float): length of the c-axis of the ellipsoidal constraint
            :math:`[\mathrm{length}]`.
        P (`tuple` [`float`, `float`, `float`]): center of the ellipsoid
            constraint (default origin) :math:`[\mathrm{length}]`.

    `Ellipsoid` defines an ellipsoid:

    .. rubric:: Implicit function

    .. math::
        F(x,y,z) = \frac{(x-P_x)^{2}}{a^{2}}
                 + \frac{(y-P_y)^{2}}{b^{2}}
                 + \frac{(z-P_z)^{2}}{c^{2}} - 1

    Example::

        ellipsoid1 = manifold.Ellipsoid(a=10,b=5,c=5)
        ellipsoid2 = manifold.Ellipsoid(a=5,b=10,c=10,P=(1,0.5,1))
    """

    def __init__(self, a, b, c, P=(0, 0, 0)):
        # store metadata
        param_dict = ParameterDict(
            a=float(a),
            b=float(b),
            c=float(c),
            P=(float, float, float),
        )
        param_dict['P'] = P

        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _md.ManifoldEllipsoid(
            self.a, self.b, self.c,
            _hoomd.make_scalar3(self.P[0], self.P[1], self.P[2]))

        super()._attach(self._simulation)


class Gyroid(Manifold):
    r"""Triply periodic gyroid manifold.

    Args:
        N (`tuple` [`int`, `int`, `int`] or `int`): number of unit cells in all
            3 directions.  :math:`[N_x, N_y, N_z]`. In case number of unit cells
            u in all direction the same (:math:`[u, u, u]`), use ``N = u``.
        epsilon (float): defines CMC companion of the Gyroid surface (default
            0)

    `Gyroid` defines a periodic gyroid surface. The gyroid belongs to
    the family of triply periodic minimal surfaces:

    .. math::
        F(x,y,z) = \sin{\frac{2 \pi}{B_x} x} \cdot \cos{\frac{2 \pi}{B_y} y}
                 + \sin{\frac{2 \pi}{B_y} y} \cdot \cos{\frac{2 \pi}{B_z} z}
                 + \sin{\frac{2 \pi}{B_z} z} \cdot \cos{\frac{2 \pi}{B_x} x}
                 - \epsilon

    is the nodal approximation of the diamond surface where
    :math:`[B_x,B_y,B_z]` is the periodicity length in the x, y and z direction.
    The periodicity length B is defined by the current box size L and the number
    of unit cells N :math:`B_i=\frac{L_i}{N_i}`.

    See Also:
        * `A. H. Schoen 1970 <https://ntrs.nasa.gov/citations/19700020472>`__
        * `P. J.F. Gandy et. al. 2000
          <https://doi.org/10.1016/S0009-2614(00)00373-0>`__
        * `H. G. von Schnering and R. Nesper 1991
          <https://doi.org/10.1007/BF01313411>`__

    Example::

        gyroid1 = manifold.Gyroid(N=1)
        gyroid2 = manifold.Gyroid(N=(1,2,2))
    """

    def __init__(self, N, epsilon=0):

        # initialize the base class
        super().__init__()
        # store metadata
        param_dict = ParameterDict(
            N=OnlyIf(to_type_converter((int,) * 3),
                     preprocess=self._preprocess_unitcell),
            epsilon=float(epsilon),
        )
        param_dict['N'] = N
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _md.ManifoldGyroid(
            _hoomd.make_int3(self.N[0], self.N[1], self.N[2]), self.epsilon)

        super()._attach(self._simulation)


class Plane(Manifold):
    r"""Plane manifold.

    Args:
        shift (float): z-shift of the xy-plane :math:`[\mathrm{length}]`.

    `Plane` defines an xy-plane at a given value of z:

    .. math::
        F(x,y,z) = z - \textrm{shift}

    Example::

        plane1 = manifold.Plane()
        plane2 = manifold.Plane(shift=0.8)
    """

    def __init__(self, shift=0):
        param_dict = ParameterDict(shift=float(shift),)

        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _md.ManifoldXYPlane(self.shift)

        super()._attach(self._simulation)


class Primitive(Manifold):
    r"""Triply periodic primitive manifold.

    Args:
        N (`tuple` [`int`, `int`, `int`] or `int`): number of unit cells in all
            3 directions.  :math:`[N_x, N_y, N_z]`. In case number of unit cells
            u in all direction the same (:math:`[u, u, u]`), use ``N = u``.
        epsilon (float): defines CMC companion of the Primitive surface
            (default 0)

    `Primitive` specifies a periodic primitive surface as a constraint.  The
    primitive (or Schwarz P) belongs to the family of triply periodic minimal
    surfaces:

    .. math::
        F(x,y,z) = \cos{\frac{2 \pi}{B_x} x} + \cos{\frac{2 \pi}{B_y} y}
                 + \cos{\frac{2 \pi}{B_z} z} - \epsilon

    is the nodal approximation of the diamond surface where
    :math:`[B_x,B_y,B_z]` is the periodicity length in the x, y and z direction.
    The periodicity length B is defined by the current box size L and the number
    of unit cells N. :math:`B_i=\frac{L_i}{N_i}`

    See Also:
        * `A. H. Schoen 1970 <https://ntrs.nasa.gov/citations/19700020472>`__)
        * `P. J.F. Gandy et. al. 2000
          <https://doi.org/10.1016/S0009-2614(00)00373-0>`__
        * `H. G. von Schnering and R. Nesper 1991
          <https://doi.org/10.1007/BF01313411>`__

    Example::

        primitive1 = manifold.Primitive(N=1)
        primitive2 = manifold.Primitive(N=(1,2,2))
    """

    def __init__(self, N, epsilon=0):

        # store metadata
        param_dict = ParameterDict(
            N=OnlyIf(to_type_converter((int,) * 3),
                     preprocess=self._preprocess_unitcell),
            epsilon=float(epsilon),
        )
        param_dict['N'] = N
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _md.ManifoldPrimitive(
            _hoomd.make_int3(self.N[0], self.N[1], self.N[2]), self.epsilon)

        super()._attach(self._simulation)


class Sphere(Manifold):
    """Sphere manifold.

    Args:
        r (float): radius of the a-axis of the spherical constraint
          :math:`[\\mathrm{length}]`.
        P (`tuple` [`float`, `float`, `float`] ): center of the spherical
            constraint (default origin) :math:`[\\mathrm{length}]`.

    `Sphere` defines a sphere:

    .. math::
        F(x,y,z) = (x-P_x)^{2} + (y-P_y)^{2} + (z-P_z)^{2} - r^{2}

    Example::

        sphere1 = manifold.Sphere(r=10)
        sphere2 = manifold.Sphere(r=5,P=(1,0,1.5))
    """

    def __init__(self, r, P=(0, 0, 0)):
        # initialize the base class
        super().__init__()
        param_dict = ParameterDict(
            r=float(r),
            P=(float, float, float),
        )
        param_dict['P'] = P

        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _md.ManifoldSphere(
            self.r, _hoomd.make_scalar3(self.P[0], self.P[1], self.P[2]))

        super()._attach(self._simulation)
