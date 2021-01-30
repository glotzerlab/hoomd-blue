# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

import hoomd;
from hoomd.md import _md
from hoomd import _hoomd
from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeconverter import OnlyIf, to_type_converter
from collections.abc import Sequence

## \internal
# \brief Base class for manifold
#
# A manifold in hoomd reflects a Manifold in c++. It is respnsible to define the manifold 
# used for RATTLE intergrators and the active force constraints.
class Manifold(_HOOMDBaseObject):
    r"""Base class manifold object.

    Manifold defines a positional constraint to a given set of particles. A manifold can 
    be applied to an integrator (RATTLE) and/or the active force class. The degrees of freedom removed from 
    the system by constraints are correctly taken into account, i.e. when computing temperature 
    for thermostatting and/or logging.

    Sphere, Ellipsoid, Plane, Cylinder, Gyroid, Diamond and Primitive are subclasses of this class. All
    manifolds are described by implicit functions.

    Attention:
        Users should not instantiate :py:class:`Manifold` directly.

    Warning:
        Only one manifold can be applied to the integrators/active forces.

    """
    def __init__(self):

        self._cpp_manifold = None;

        self.name = None;

    ## \var cpp_manifold
    # \internal
    # \brief Stores the C++ side Manifold managed by this class

    def implicit_function(self, point):
        """Compute the implicit function evaluated at a point in space.

        Args:
            point (tuple): The point applied to the implicit function."""
        return self._cpp_manifold.implicit_function(_hoomd.make_scalar3(point[0], point[1], point[2]))

    def derivative(self, point):
        """Compute the deriviative of the implicit function evaluated at a point in space.

        Args:
            point (tuple): The point applied to the implicit function."""
        return self._cpp_manifold.derivative(_hoomd.make_scalar3(point[0], point[1], point[2]))


class Sphere(Manifold):
    """ Sphere manifold.

    Args:
        r (`float`): raduis of the a-axis of the spherical constraint (in distance units).
        P (`tuple`): center of the spherical constraint (default origin).

    :py:class:`Sphere` specifies that a spherical manifold is defined as 
    a constraint. 

    .. rubric:: Implicit function

    .. math::
        F(x,y,z) = x^{2} + y^{2} + z^{2} - r^{2}

    Example::

        sphere1 = manifold.Sphere(r=10)
        sphere2 = manifold.Sphere(r=5,P=(1,0,1.5))
    """
    def __init__(self,r, P=[0,0,0] ):
        # initialize the base class
        super().__init__();
        param_dict = ParameterDict(
            r=float(r),
            P=(float, float,float),
        )
        param_dict.update(
            dict(P=(P[0], P[1], P[2])))

        self._param_dict.update(param_dict)
        # set defaults

    def _attach(self):
        self._cpp_manifold = _md.ManifoldClassSphere(self.r, _hoomd.make_scalar3( self.P[0], self.P[1], self.P[2]) );

        self.name = "Sphere"

        super()._attach()

