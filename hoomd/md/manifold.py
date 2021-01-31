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


class Cylinder(Manifold):
    R""" Cylinder manifold.

    Args:
        r (`float`): radius of the cylinder constraint (in distance units).
        P (`tuple`): point defining position of the cylinder axis (default origin).

    :py:class:`Cylinder` specifies that a cylindric manifold is defined as
    a constraint.

    Note:
        The cylinder axis is parallel to the z-direction.

    .. rubric:: Implicit function

    .. math::
        F(x,y,z) = x^{2} + y^{2} - r^{2}

    Example::

        cylinder1 = manifold.Cylinder(r=10)
        cylinder2 = manifold.Cylinder(r=5,P=(1,1,1))
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
        self._cpp_manifold = _md.ManifoldClassCylinder(self.r, _hoomd.make_scalar3( self.P[0], self.P[1], self.P[2]) );

        self.name = "Cylinder"

        super()._attach()

class Diamond(Manifold):

    R""" Triply periodic diamond manifold.

    Args:
        N (`list` [ `int` ] or `int`): number of unit cells in all 3 directions.
            :math:`[N_x, N_y, N_z]`. In case number of unit cells u in all
            direction the same (:math:`[u, u, u]`), use ``N = u``.
        epsilon (`float`): defines CMC companion of the Diamond surface (default 0) 

    :py:class:`Diamond` specifies a periodic diamond surface as a constraint.
    The diamond (or Schwarz D) belongs to the family of triply periodic minimal surfaces.

    
    For the diamond surface, see:
    * `A. H. Schoen 1970  <https://ntrs.nasa.gov/citations/19700020472>`_
    * `P. J. F. Gandy et. al. 1999  <https://doi.org/10.1016/S0009-2614(99)01000-3>`_
    * `H. G. von Schnering and R. Nesper 1991  <https://doi.org/10.1007/BF01313411>`_
    
    .. rubric:: Implicit function
    
    .. math::
        F(x,y,z) = \cos{\frac{2 \pi}{L_x} x}\cdot\cos{\frac{2 \pi}{L_y} y}\cdot\cos{\frac{2 \pi}{L_z} z} - \sin{\frac{2 \pi}{L_x} x}\cdot\sin{\frac{2 \pi}{L_y} y}\cdot\sin{\frac{2 \pi}{L_z} z}

    is the nodal approximation of the diamond surface where :math:`[L_x,L_y,L_z]` is the periodicity length in the x, y and z direction.
    The periodicity length L is defined by the box size B and the number of unit cells N. :math:`L=\frac{B}{N}`
    
    Example::
    
        diamond1 = manifold.Diamond(N=1)
        diamond2 = manifold.Diamond(N=(1,2,2))
    """
    def __init__(self,N,epsilon=0):

        # initialize the base class
        super().__init__();
        # store metadata
        param_dict = ParameterDict(
            N=OnlyIf(to_type_converter((int,)*3), preprocess=self.__preprocess_unitcell),
            epsilon=float(epsilon),
        )
        param_dict.update(
            dict(N=N))
        self._param_dict.update(param_dict)

    def _attach(self):
        self._cpp_manifold = _md.ManifoldClassDiamond(self.N[0], self.N[1], self.N[2], self.epsilon );

        self.name = "Diamond"

        super()._attach()

    def __preprocess_unitcell(self,value):
        if isinstance(value, Sequence):
            if len(value) != 3:
                raise ValueError(
                    "Expected a single int or six.")
            return tuple(value)
        else:
            return (value,value,value)

class Ellipsoid(Manifold):
    """ Ellipsoid manifold.

    Args:
        a (`float`): length of the a-axis of the ellipsoidal constraint (in distance units).
        b (`float`): length of the b-axis of the ellipsoidal constraint (in distance units).
        c (`float`): length of the c-axis of the ellipsoidal constraint (in distance units).
        P (`tuple`): center of the ellipsoidal constraint (default origin).
    
    :py:class:`Ellipsoid` specifies that a ellipsoidal manifold is defined as a constraint. 
    
    .. rubric:: Implicit function
    
    .. math::
        F(x,y,z) = \\frac{x^{2}}{a^{2}} + \\frac{y^{2}}{b^{2}} + \\frac{z^{2}}{c^{2}} - 1      
    
    Example::
    
        ellipsoid1 = manifold.Ellipsoid(a=10,b=5,c=5)
        ellipsoid2 = manifold.Ellipsoid(a=5,b=10,c=10,P=(1,0.5,1))
    """
    def __init__(self,a,b,c, P=[0,0,0] ):
        # initialize the base class
        super().__init__();
        # store metadata
        param_dict = ParameterDict(
            a=float(a),
            b=float(b),
            c=float(c),
            P=(float, float,float),
        )
        param_dict.update(
            dict(P=(P[0], P[1], P[2])))

        self._param_dict.update(param_dict)

    def _attach(self):
        self._cpp_manifold = _md.ManifoldClassEllipsoid(self.a, self.b, self.c,  _hoomd.make_scalar3( self.P[0], self.P[1], self.P[2]) );

        self.name = "Ellipsoid"

        super()._attach()

class Gyroid(Manifold):

    R""" Triply periodic gyroid manifold.

    Args:
        N (`list` [ `int` ] or `int`): number of unit cells in all 3 directions.
            :math:`[N_x, N_y, N_z]`. In case number of unit cells u in all
            direction the same (:math:`[u, u, u]`), use ``N = u``.
        epsilon (`float`): defines CMC companion of the Gyroid surface (default 0) 
        
    :py:class:`Gyroid` specifies a periodic gyroid surface as a constraint.
    The gyroid belongs to the family of triply periodic minimal surfaces.

    For the gyroid surface, see:
    
    * `A. H. Schoen 1970  <https://ntrs.nasa.gov/citations/19700020472>`_
    * `P. J.F. Gandy et. al. 2000  <https://doi.org/10.1016/S0009-2614(00)00373-0>`_
    * `H. G. von Schnering and R. Nesper 1991  <https://doi.org/10.1007/BF01313411>`_
    
    .. rubric:: Implicit function
    
    .. math::
        F(x,y,z) = \sin{\frac{2 \pi}{L_x} x}\cdot\cos{\frac{2 \pi}{L_y} y} + \sin{\frac{2 \pi}{L_y} y}\cdot\cos{\frac{2 \pi}{L_z} z} + \sin{\frac{2 \pi}{L_z} z}\cdot\cos{\frac{2 \pi}{L_x} x}
    
    is the nodal approximation of the gyroid surface where :math:`[L_x,L_y,L_z]` is the periodicity length in the x, y and z direction.
    The periodicity length L is defined by the box size B and the number of unit cells N. :math:`L=\frac{B}{N}`
    
    Example::
    
        gyroid1 = manifold.Gyroid(N=1)
        gyroid2 = manifold.Gyroid(N=(1,2,2))
    """
    def __init__(self,N,epsilon=0):

        # initialize the base class
        super().__init__();
        # store metadata
        param_dict = ParameterDict(
            N=OnlyIf(to_type_converter((int,)*3), preprocess=self.__preprocess_unitcell),
            epsilon=float(epsilon),
        )
        param_dict.update(
            dict(N=N))
        self._param_dict.update(param_dict)

    def _attach(self):
        self._cpp_manifold = _md.ManifoldClassGyroid(self.N[0], self.N[1], self.N[2], self.epsilon );

        self.name = "Gyroid"

        super()._attach()

    def __preprocess_unitcell(self,value):
        if isinstance(value, Sequence):
            if len(value) != 3:
                raise ValueError(
                    "Expected a single int or six.")
            return tuple(value)
        else:
            return (value,value,value)

class Plane(Manifold):
    R""" Plane manifold.
    
    Args:
        shift (`float`): z-shift of the xy-plane (in distance units).

    :py:class:`Plane` specifies that a xy-plane manifold is defined as 
    a constraint. 

    .. rubric:: Implicit function

    .. math::
        F(x,y,z) = z - \textrm{shift}

    Example::

        plane1 = manifold.Plane()
        plane2 = manifold.Plane(shift=0.8)
    """
    def __init__(self,shift=0):
        # initialize the base class
        super().__init__();
        param_dict = ParameterDict(
            shift=float(shift),
        )

        self._param_dict.update(param_dict)
        # set defaults

    def _attach(self):
        self._cpp_manifold = _md.ManifoldClassPlane(self.shift);

        self.name = "Plane"

        super()._attach()

class Primitive(Manifold):

    R""" Triply periodic primitive manifold.

    Args:
        N (`list` [ `int` ] or `int`): number of unit cells in all 3 directions.
            :math:`[N_x, N_y, N_z]`. In case number of unit cells u in all
            direction the same (:math:`[u, u, u]`), use ``N = u``.
        epsilon (`float`): defines CMC companion of the Primitive surface (default 0) 
        
    :py:class:`Primitive` specifies a periodic primitive surface as a constraint.
    The primitive (or Schwarz P) belongs to the family of triply periodic minimal surfaces.

    For the primitive surface, see:
    
    * `A. H. Schoen 1970  <https://ntrs.nasa.gov/citations/19700020472>`_
    * `P. J. F. Gandy et. al. 2000  <https://doi.org/10.1016/S0009-2614(00)00453-X>`_
    * `H. G. von Schnering and R. Nesper 1991  <https://doi.org/10.1007/BF01313411>`_
    
    .. rubric:: Implicit function
    
    .. math::
        F(x,y,z) = \cos{\frac{2 \pi}{L_x} x} + \cos{\frac{2 \pi}{L_y} y} + \cos{\frac{2 \pi}{L_z} z}

    is the nodal approximation of the primitive surface where :math:`[L_x,L_y,L_z]` is the periodicity length in the x, y and z direction.
    The periodicity length L is defined by the box size B and the number of unit cells N. :math:`L=\frac{B}{N}`
    
    Example::
    
        primitive1 = manifold.Primitive(N=1)
        primitive2 = manifold.Primitive(N=(1,2,2))
    """
    def __init__(self,N,epsilon=0):

        # initialize the base class
        super().__init__();
        # store metadata
        param_dict = ParameterDict(
            N=OnlyIf(to_type_converter((int,)*3), preprocess=self.__preprocess_unitcell),
            epsilon=float(epsilon),
        )
        param_dict.update(
            dict(N=N))
        self._param_dict.update(param_dict)

    def _attach(self):
        self._cpp_manifold = _md.ManifoldClassPrimitive(self.N[0], self.N[1], self.N[2], self.epsilon );

        self.name = "Primitive"

        super()._attach()

    def __preprocess_unitcell(self,value):
        if isinstance(value, Sequence):
            if len(value) != 3:
                raise ValueError(
                    "Expected a single int or six.")
            return tuple(value)
        else:
            return (value,value,value)


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

