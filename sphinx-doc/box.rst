.. _boxdim:

Periodic boundary conditions
============================

Introduction
------------

All simulations executed in HOOMD-blue occur in a triclinic simulation box with periodic boundary conditions in all
three directions. A triclinic box is defined by six values: the extents :math:`L_x`, :math:`L_y` and :math:`L_z` of the box
in the three directions, and three tilt factors :math:`xy`, :math:`xz` and :math:`yz`.

The parameter matrix :math:`\mathbf{h}` is defined in terms of the lattice vectors
:math:`\vec a_1`, :math:`\vec a_2` and :math:`\vec a_3`:

.. math::

    \mathbf{h} \equiv \left( \vec a_1, \vec a_2, \vec a_3 \right)

By convention, the first lattice vector
:math:`\vec a_1` is parallel to the unit vector :math:`\vec e_x = (1,0,0)`. The tilt factor
:math:`xy` indicates how the second lattice vector :math:`\vec a_2` is tilted with respect to the first one. It specifies
many units along the x-direction correspond to one unit of the second lattice vector. Similarly, :math:`xz` and
:math:`yz` indicate the tilt of the third lattice vector :math:`\vec a_3` with respect to the first and second lattice
vector.

Definitions and formulas for the cell parameter matrix
------------------------------------------------------

The full cell parameter matrix is:

.. math::
    :nowrap:

    \begin{eqnarray*}
    \mathbf{h}& =& \left(\begin{array}{ccc} L_x & xy L_y & xz L_z \\
                                            0   & L_y    & yz L_z \\
                                            0   & 0      & L_z    \\
                         \end{array}\right)
    \end{eqnarray*}

The tilt factors :math:`xy`, :math:`xz` and :math:`yz` are dimensionless.
The relationships between the tilt factors and the box angles :math:`\alpha`,
:math:`\beta` and :math:`\gamma` are as follows:

.. math::
    :nowrap:

    \begin{eqnarray*}
    \cos\gamma \equiv \cos(\angle\vec a_1, \vec a_2) &=& \frac{xy}{\sqrt{1+xy^2}}\\
    \cos\beta \equiv \cos(\angle\vec a_1, \vec a_3) &=& \frac{xz}{\sqrt{1+xz^2+yz^2}}\\
    \cos\alpha \equiv \cos(\angle\vec a_2, \vec a_3) &=& \frac{xy \cdot xz + yz}{\sqrt{1+xy^2} \sqrt{1+xz^2+yz^2}}
    \end{eqnarray*}

Given an arbitrarily oriented lattice with box vectors :math:`\vec v_1, \vec v_2, \vec v_3`, the HOOMD-blue
box parameters for the rotated box can be found as follows.

.. math::
    :nowrap:

    \begin{eqnarray*}
    L_x &=& v_1\\
    a_{2x} &=& \frac{\vec v_1 \cdot \vec v_2}{v_1}\\
    L_y &=& \sqrt{v_2^2 - a_{2x}^2}\\
    xy &=& \frac{a_{2x}}{L_y}\\
    L_z &=& \vec v_3 \cdot \frac{\vec v_1 \times \vec v_2}{\left| \vec v_1 \times \vec v_2 \right|}\\
    a_{3x} &=& \frac{\vec v_1 \cdot \vec v_3}{v_1}\\
    xz &=& \frac{a_{3x}}{L_z}\\
    yz &=& \frac{\vec v_2 \cdot \vec v_3 - a_{2x}a_{3x}}{L_y L_z}
    \end{eqnarray*}

Example::

    # boxMatrix contains an arbitrarily oriented right-handed box matrix.
    v[0] = boxMatrix[:,0]
    v[1] = boxMatrix[:,1]
    v[2] = boxMatrix[:,2]
    Lx = numpy.sqrt(numpy.dot(v[0], v[0]))
    a2x = numpy.dot(v[0], v[1]) / Lx
    Ly = numpy.sqrt(numpy.dot(v[1],v[1]) - a2x*a2x)
    xy = a2x / Ly
    v0xv1 = numpy.cross(v[0], v[1])
    v0xv1mag = numpy.sqrt(numpy.dot(v0xv1, v0xv1))
    Lz = numpy.dot(v[2], v0xv1) / v0xv1mag
    a3x = numpy.dot(v[0], v[2]) / Lx
    xz = a3x / Lz
    yz = (numpy.dot(v[1],v[2]) - a2x*a3x) / (Ly*Lz)

Initializing a system with a triclinic box
------------------------------------------

You can specify all parameters of a triclinic box in a GSD file.

You can also pass a :py:class:`hoomd.data.boxdim` argument to the constructor of any initialization method. Here is an
example for :py:func:`hoomd.deprecated.init.create_random`::

    init.create_random(box=data.boxdim(L=18, xy=0.1, xz=0.2, yz=0.3), N=1000))

This creates a triclinic box with edges of length 18, and tilt factors
:math:`xy =0.1`, :math:`xz=0.2` and :math:`yz=0.3`.

You can also specify a 2D box to any of the initialization methods::

    init.create_random(N=1000, box=data.boxdim(xy=1.0, volume=2000, dimensions=2), min_dist=1.0)


Change the simulation box
-------------------------

The triclinic unit cell can be updated in various ways.

Resizing the box
^^^^^^^^^^^^^^^^

The simulation box can be gradually resized during a simulation run using
:py:class:`hoomd.update.box_resize`.

To update the tilt factors continuously during the simulation (shearing
the simulation box with **Lees-Edwards** boundary conditions), use::

    update.box_resize(xy = variant.linear_interp([(0,0), (1e6, 1)]))

This command applies shear in the :math:`xy` -plane so that the angle between the *x*
and *y*-directions changes continuously from 0 to :math:`45^\circ` during :math:`10^6` time steps.

:py:class:`hoomd.update.box_resize` can change any or all of the six box parameters.

NPT or NPH integration
^^^^^^^^^^^^^^^^^^^^^^

In a constant pressure ensemble, the box is updated every time step, according to the anisotropic stresses in the
system. This is supported by:

- :py:class:`hoomd.md.integrate.npt`
- :py:class:`hoomd.md.integrate.nph`
