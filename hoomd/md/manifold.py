# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Manifold.

Manifold defining a positional constraint to a given set of particles. For example, a group of particles 
can be constrained to the surface of a sphere with :py:class:`sphere`.

Warning:
    Only one manifold can be applied to the integrators/active forces.

The degrees of freedom removed from the system by constraints are correctly taken into account when computing the
temperature for thermostatting and logging.
"""

from hoomd.md import _md
import hoomd;
from hoomd.manifold import _Manifold
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict

class Cylinder(_Manifold):
    def __init__(self,r, P=(0,0,0) ):
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
        cpp_sys_def = self._simulation.state._cpp_sys_def
        self._cpp_manifold = _md.CylinderManifold(cpp_sys_def, self.r, _hoomd.make_scalar3( self.P[0], self.P[1], self.P[2]) );

        super()._attach()

class Diamond(_Manifold):
    def __init__(self,N=None,Nx=None,Ny=None,Nz=None):
        if N is not None:
            Nx = Ny = Nz = N;

        # initialize the base class
        super().__init__();
        # store metadata
        param_dict = ParameterDict(
            N=(int, int, int),
        )
        param_dict.update(
            dict(N=(Nx, Ny, Nz)))
        self._param_dict.update(param_dict)

    def _attach(self):
        cpp_sys_def = self._simulation.state._cpp_sys_def
        self._cpp_manifold = _md.TPMSManifold(cpp_sys_def, 'D', self.N[0], self.N[1], self.N[2] );

        super()._attach()

class Ellipsoid(_Manifold):
    def __init__(self,a,b,c, P=(0,0,0) ):
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
        cpp_sys_def = self._simulation.state._cpp_sys_def
        self._cpp_manifold = _md.EllipsoidManifold(cpp_sys_def, self.a, self.b, self.c,  _hoomd.make_scalar3( self.P[0], self.P[1], self.P[2]) );

        super()._attach()

class Gyroid(_Manifold):
    def __init__(self,N=None,Nx=None,Ny=None,Nz=None):
        if N is not None:
            Nx = Ny = Nz = N;

        # initialize the base class
        super().__init__();
        # store metadata
        param_dict = ParameterDict(
            N=(int, int, int),
        )
        param_dict.update(
            dict(N=(Nx, Ny, Nz)))
        self._param_dict.update(param_dict)

    def _attach(self):
        cpp_sys_def = self._simulation.state._cpp_sys_def
        self._cpp_manifold = _md.TPMSManifold(cpp_sys_def, 'G', self.N[0], self.N[1], self.N[2] );

        super()._attach()


class Plane(_Manifold):
    def __init__(self, shift=0):
        # initialize the base class
        super().__init__();

        param_dict = ParameterDict(
            shift=float(shift),
        )
        self._param_dict.update(param_dict)

    def _attach(self):
        cpp_sys_def = self._simulation.state._cpp_sys_def
        self._cpp_manifold = _md.FlatManifold(cpp_sys_def, self.shift);

        super()._attach()

class Primitive(_Manifold):
    def __init__(self,N=None,Nx=None,Ny=None,Nz=None):
        if N is not None:
            Nx = Ny = Nz = N;

        # initialize the base class
        super().__init__();
        # store metadata
        param_dict = ParameterDict(
            N=(int, int, int),
        )
        param_dict.update(
            dict(N=(Nx, Ny, Nz)))
        self._param_dict.update(param_dict)

    def _attach(self):
        cpp_sys_def = self._simulation.state._cpp_sys_def
        self._cpp_manifold = _md.TPMSManifold(cpp_sys_def, 'P', self.N[0], self.N[1], self.N[2] );

        super()._attach()

class Sphere(_Manifold):
    def __init__(self,r, P=(0,0,0) ):
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
        cpp_sys_def = self._simulation.state._cpp_sys_def
        self._cpp_manifold = _md.SphereManifold(cpp_sys_def, self.r, _hoomd.make_scalar3( self.P[0], self.P[1], self.P[2]) );

        super()._attach()

