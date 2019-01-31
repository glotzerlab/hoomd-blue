# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R""" Define lattices.

:py:mod:`hoomd.lattice` provides a general interface to define lattices to initialize systems.

See Also:
    :py:func:`hoomd.init.create_lattice`.
"""

import numpy
import hoomd
import math

# Multiply two quaternions
# Apply quaternion multiplication per http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
# (requires numpy)
# \param q1 quaternion
# \param q2 quaternion
# \returns q1*q2
def _quatMult(q1, q2):
    s = q1[0]
    v = q1[1:]
    t = q2[0]
    w = q2[1:]
    q = numpy.empty((4,), dtype=numpy.float64)
    q[0] = s*t - numpy.dot(v, w)
    q[1:] = s*w + t*v + numpy.cross(v,w)
    return q

# Rotate a vector by a unit quaternion
# Quaternion rotation per http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
# (requires numpy)
# \param q rotation quaternion
# \param v 3d vector to be rotated
# \returns q*v*q^{-1}
def _quatRot(q, v):
    v = numpy.asarray(v)
    q = numpy.asarray(q)
    # assume q is a unit quaternion
    w = q[0]
    r = q[1:]
    vnew = numpy.empty((3,), dtype=v.dtype)
    vnew = v + 2*numpy.cross(r, numpy.cross(r,v) + w*v)
    return vnew

# Given a set of lattice vectors, rotate to produce an upper triangular right-handed box
# as a hoomd boxdim object and a rotation quaternion that brings particles in the original coordinate system to the new one.
# The conversion preserves handedness, so it is left to the user to provide a right-handed set of lattice vectors
# (E.g. returns (data.boxdim(Lx=10, Ly=20, Lz=30, xy=1.0, xz=0.5, yz=0.1), q) )
# (requires numpy)
# \param a1 first lattice vector
# \param a2 second lattice vector
# \param a3 third lattice vector
# \returns (box, q) tuple of boxdim object and rotation quaternion
def _latticeToHoomd(a1, a2, a3=[0.,0.,1.], ndim=3):
    a1 = numpy.array(a1, dtype=numpy.float64)
    a2 = numpy.array(a2, dtype=numpy.float64)
    a3 = numpy.array(a3, dtype=numpy.float64)
    a1.resize((3,))
    a2.resize((3,))
    a3.resize((3,))
    xhat = numpy.array([1.,0.,0.])
    yhat = numpy.array([0.,1.,0.])
    zhat = numpy.array([0.,0.,1.])

    # Find quaternion to rotate first lattice vector to x axis
    a1mag = numpy.sqrt(numpy.dot(a1,a1))
    v1 = a1/a1mag + xhat
    v1mag = numpy.sqrt(numpy.dot(v1,v1))
    if v1mag > 1e-6:
        u1 = v1/v1mag
    else:
        # a1 is antialigned with xhat, so rotate around any unit vector perpendicular to xhat
        u1 = yhat
    q1 = numpy.concatenate(([numpy.cos(numpy.pi/2)], numpy.sin(numpy.pi/2)*u1))

    # Find quaternion to rotate second lattice vector to xy plane after applying above rotation
    a2prime = _quatRot(q1, a2)
    angle = -1*numpy.arctan2(a2prime[2], a2prime[1])
    q2 = numpy.concatenate(([numpy.cos(angle/2)], numpy.sin(angle/2)*xhat))

    q = _quatMult(q2,q1)

    Lx = numpy.sqrt(numpy.dot(a1, a1))
    a2x = numpy.dot(a1, a2) / Lx
    Ly = numpy.sqrt(numpy.dot(a2,a2) - a2x*a2x)
    xy = a2x / Ly
    v0xv1 = numpy.cross(a1, a2)
    v0xv1mag = numpy.sqrt(numpy.dot(v0xv1, v0xv1))
    Lz = numpy.dot(a3, v0xv1) / v0xv1mag
    a3x = numpy.dot(a1, a3) / Lx
    xz = a3x / Lz
    yz = (numpy.dot(a2, a3) - a2x*a3x) / (Ly*Lz)

    box = hoomd.data.boxdim(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz, dimensions=ndim)

    return box, q


class unitcell(object):
    R""" Define a unit cell.

    Args:
        N (int): Number of particles in the unit cell.
        a1 (list): Lattice vector (3-vector).
        a2 (list): Lattice vector (3-vector).
        a3 (list): Lattice vector (3-vector). Set to [0,0,1] in 2D lattices.
        dimensions (int): Dimensionality of the lattice (2 or 3).
        position (list): List of particle positions.
        type_name (list): List of particle type names.
        mass (list): List of particle masses.
        charge (list): List of particle charges.
        diameter (list): List of particle diameters.
        moment_inertia (list): List of particle moments of inertia.
        orientation (list): List of particle orientations.

    A unit cell is a box definition (*a1*, *a2*, *a3*, *dimensions*), and particle properties for *N* particles.
    You do not need to specify all particle properties. Any property omitted will be initialized to defaults (see
    :py:func:`hoomd.data.make_snapshot`). :py:class:`hoomd.init.create_lattice` initializes the system with many
    copies of a unit cell.

    :py:class:`unitcell` is a completely generic unit cell representation. See other classes in the :py:mod:`hoomd.lattice`
    module for convenience wrappers for common lattices.

    Example::

        uc = hoomd.lattice.unitcell(N = 2,
                                    a1 = [1,0,0],
                                    a2 = [0.2,1.2,0],
                                    a3 = [-0.2,0, 1.0],
                                    dimensions = 3,
                                    position = [[0,0,0], [0.5, 0.5, 0.5]],
                                    type_name = ['A', 'B'],
                                    mass = [1.0, 2.0],
                                    charge = [0.0, 0.0],
                                    diameter = [1.0, 1.3],
                                    moment_inertia = [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                                    orientation = [[0.707, 0, 0, 0.707], [1.0, 0, 0, 0]]);

    Note:
        *a1*, *a2*, *a3* must define a right handed coordinate system.

    """

    def __init__(self,
                 N,
                 a1,
                 a2,
                 a3,
                 dimensions = 3,
                 position = None,
                 type_name = None,
                 mass = None,
                 charge = None,
                 diameter = None,
                 moment_inertia = None,
                 orientation = None):

        self.N = N;
        self.a1 = a1;
        self.a2 = a2;
        self.a3 = a3;
        self.dimensions = dimensions;

        if position is None:
            self.position = numpy.array([(0,0,0)] * self.N, dtype=numpy.float64);
        else:
            self.position = numpy.asarray(position, dtype=numpy.float64);
            if len(self.position) != N:
                raise ValueError("Particle properties must have length N");

        if type_name is None:
            self.type_name = ['A'] * self.N
        else:
            self.type_name = type_name;
            if len(self.type_name) != N:
                raise ValueError("Particle properties must have length N");

        if mass is None:
            self.mass = numpy.array([1.0] * self.N, dtype=numpy.float64);
        else:
            self.mass = numpy.asarray(mass, dtype=numpy.float64);
            if len(self.mass) != N:
                raise ValueError("Particle properties must have length N");

        if charge is None:
            self.charge = numpy.array([0.0] * self.N, dtype=numpy.float64);
        else:
            self.charge = numpy.asarray(charge, dtype=numpy.float64);
            if len(self.charge) != N:
                raise ValueError("Particle properties must have length N");

        if diameter is None:
            self.diameter = numpy.array([1.0] * self.N, dtype=numpy.float64);
        else:
            self.diameter = numpy.asarray(diameter, dtype=numpy.float64);
            if len(self.diameter) != N:
                raise ValueError("Particle properties must have length N");

        if moment_inertia is None:
            self.moment_inertia = numpy.array([(0,0,0)] * self.N, dtype=numpy.float64);
        else:
            self.moment_inertia = numpy.asarray(moment_inertia, dtype=numpy.float64);
            if len(self.moment_inertia) != N:
                raise ValueError("Particle properties must have length N");

        if orientation is None:
            self.orientation = numpy.array([(1,0,0,0)] * self.N, dtype=numpy.float64);
        else:
            self.orientation = numpy.asarray(orientation, dtype=numpy.float64);
            if len(self.orientation) != N:
                raise ValueError("Particle properties must have length N");

    def get_type_list(self):
        R""" Get a list of the unique type names in the unit cell.

        Returns:
            A :py:class:`list` of the unique type names present in the unit cell.
        """

        type_list = [];
        for name in self.type_name:
            if not name in type_list:
                type_list.append(name);

        return type_list;

    def get_typeid_mapping(self):
        R""" Get a type name to typeid mapping.

        Returns:
            A :py:class:`dict` that maps type names to integer type ids.
        """

        mapping = {};
        idx = 0;

        for name in self.type_name:
            if not name in mapping:
                mapping[name] = idx;
                idx = idx + 1;

        return mapping;

    def get_snapshot(self):
        R""" Get a snapshot.

        Returns:
            A snapshot representing the lattice.

        .. attention::
            HOOMD-blue requires upper-triangular box matrices. The general box matrix *(a1, a2, a3)* set for this
            :py:class:`unitcell` and the particle positions and orientations will be rotated from provided values
            into upper triangular form.
        """

        box, q = _latticeToHoomd(self.a1, self.a2, self.a3, ndim=self.dimensions)
        snap = hoomd.data.make_snapshot(N=self.N, box=box, dtype='double');
        mapping = self.get_typeid_mapping();

        if hoomd.comm.get_rank() == 0:
            snap.particles.types = self.get_type_list();
            snap.particles.typeid[:] = [mapping[name] for name in self.type_name];
            snap.particles.mass[:] = self.mass[:];
            snap.particles.charge[:] = self.charge[:];
            snap.particles.diameter[:] = self.diameter[:];
            snap.particles.moment_inertia[:] = self.moment_inertia[:];

            for i in range(self.N):
                snap.particles.position[i] = _quatRot(q, self.position[i])
                snap.particles.position[i], img = box.wrap(snap.particles.position[i])
                snap.particles.orientation[i] = _quatMult(q, self.orientation[i])

        return snap;

def sc(a, type_name='A'):
    R""" Create a simple cubic lattice (3D).

    Args:
        a (float): Lattice constant.
        type_name (str): Particle type name.

    The simple cubic unit cell has 1 particle:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{r}& =& \left(\begin{array}{ccc} 0 & 0 & 0 \\
                             \end{array}\right)
        \end{eqnarray*}

    And the box matrix:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \mathbf{h}& =& \left(\begin{array}{ccc} a & 0 & 0 \\
                                                0 & a & 0 \\
                                                0 & 0 & a \\
                             \end{array}\right)
        \end{eqnarray*}
    """
    hoomd.util.print_status_line();
    return unitcell(N=1, type_name=[type_name], a1=[a,0,0], a2=[0,a,0], a3=[0,0,a], dimensions=3);

def bcc(a, type_name='A'):
    R""" Create a body centered cubic lattice (3D).

    Args:
        a (float): Lattice constant.
        type_name (str): Particle type name.

    The body centered cubic unit cell has 2 particles:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{r}& =& \left(\begin{array}{ccc} 0 & 0 & 0 \\
                                             \frac{a}{2} & \frac{a}{2} & \frac{a}{2} \\
                             \end{array}\right)
        \end{eqnarray*}

    And the box matrix:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \mathbf{h}& =& \left(\begin{array}{ccc} a & 0 & 0 \\
                                                0 & a & 0 \\
                                                0 & 0 & a \\
                             \end{array}\right)
        \end{eqnarray*}
    """
    hoomd.util.print_status_line();
    return unitcell(N=2,
                    type_name=[type_name, type_name],
                    position=[[0,0,0],[a/2,a/2,a/2]],
                    a1=[a,0,0],
                    a2=[0,a,0],
                    a3=[0,0,a],
                    dimensions=3);

def fcc(a, type_name='A'):
    R""" Create a face centered cubic lattice (3D).

    Args:
        a (float): Lattice constant.
        type_name (str): Particle type name.

    The face centered cubic unit cell has 4 particles:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{r}& =& \left(\begin{array}{ccc} 0 & 0 & 0 \\
                                             0 & \frac{a}{2} & \frac{a}{2} \\
                                             \frac{a}{2} & 0 & \frac{a}{2} \\
                                             \frac{a}{2} & \frac{a}{2} & 0\\
                             \end{array}\right)
        \end{eqnarray*}

    And the box matrix:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \mathbf{h}& =& \left(\begin{array}{ccc} a & 0 & 0 \\
                                                0 & a & 0 \\
                                                0 & 0 & a \\
                             \end{array}\right)
        \end{eqnarray*}
    """
    hoomd.util.print_status_line();
    return unitcell(N=4,
                    type_name=[type_name]*4,
                    position=[[0,0,0],[0,a/2,a/2],[a/2,0,a/2],[a/2,a/2,0]],
                    a1=[a,0,0],
                    a2=[0,a,0],
                    a3=[0,0,a],
                    dimensions=3);

def sq(a, type_name='A'):
    R""" Create a square lattice (2D).

    Args:
        a (float): Lattice constant.
        type_name (str): Particle type name.

    The simple square unit cell has 1 particle:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{r}& =& \left(\begin{array}{ccc} 0 & 0 \\
                             \end{array}\right)
        \end{eqnarray*}

    And the box matrix:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \mathbf{h}& =& \left(\begin{array}{ccc} a & 0 \\
                                                0 & a \\
                             \end{array}\right)
        \end{eqnarray*}
    """
    hoomd.util.print_status_line();
    return unitcell(N=1, type_name=[type_name], a1=[a,0,0], a2=[0,a,0], a3=[0,0,1], dimensions=2);

def hex(a, type_name='A'):
    R""" Create a hexagonal lattice (2D).

    Args:
        a (float): Lattice constant.
        type_name (str): Particle type name.

    :py:class:`hex` creates a hexagonal lattice in a rectangular box. It has 2 particles, one at the
    corner and one at the center of the rectangle. This is not the primitive unit cell, but is more convenient to
    work with because of its shape.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{r}& =& \left(\begin{array}{ccc} 0 & 0 \\
                                             \frac{a}{2} & \sqrt{3} \frac{a}{2} \\
                             \end{array}\right)
        \end{eqnarray*}

    And the box matrix:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \mathbf{h}& =& \left(\begin{array}{ccc} a & 0 \\
                                                0 & \sqrt{3} a \\
                                                0 & 0 \\
                             \end{array}\right)
        \end{eqnarray*}
    """

    hoomd.util.print_status_line();
    return unitcell(N=2,
                    type_name=[type_name, type_name],
                    position=[[0,0,0],[a/2,math.sqrt(3)*a/2,0]],
                    a1=[a,0,0],
                    a2=[0,math.sqrt(3)*a,0],
                    a3=[0,0,1],
                    dimensions=2);
