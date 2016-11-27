Anisotropic Particles
=====================

Overview
--------

HOOMD-blue natively supports the integration of rotational degrees of freedom of anisotropic particles. When
any anisotropic potential is defined in the system, integrators automatically integrate the rotational and
translational degrees of freedom of the system. Anisotropic integration can also be explicitly enabled or
disabled through the ``aniso`` argument of :py:class:`hoomd.md.integrate.mode_standard`

.. note::

    Rotational degrees of freedom given a moment of inertia of 0 (the default value) are not integrated or
    considered for temperature computation. Always set the moment of inertia for rotational degrees of freedom
    which should be integrated.

Anisotropic particles have a number of properties accessible using the particle data API (:py:mod:`hoomd.data`):

 - orientation, Quaternion to rotate the particle from its base orientation to its current orientation, in order :math:`(real, imag_x, imag_y, imag_z)`
 - angular_momentum, Conjugate quaternion representing the particle's angular momentum
 - moment_inertia, principal moments of inertia :math:`(I_{xx}, I_{yy}, I_{zz})`
 - net_torque, net torque on the particle in the global reference frame

Quaternions for angular momentum
--------------------------------

Particle angular momenta are stored in quaternion form as defined in `Kamberaj 2005 <http://dx.doi.org/10.1063/1.1906216>`_ : the
angular momentum quaternion :math:`\mathbf{P}` is defined with respect to the orientation quaternion of the
particle :math:`\mathbf{q}` and the angular momentum of the particle, lifted into pure imaginary quaternion form
:math:`\mathbf{S}^{(4)}` as:

.. math::

    \mathbf{P} = 2 \mathbf{q} \times \mathbf{S}^{(4)}

in other words, the angular momentum vector :math:`\vec{S}` with respect to the principal axis of the particle is

.. math::

    \vec{S} = \frac{1}{2}im(\mathbf{q}^* \times \mathbf{P})

where :math:`\mathbf{q}^*` is the conjugate of the particle's orientation quaternion and :math:`\times` is
quaternion multiplication.
