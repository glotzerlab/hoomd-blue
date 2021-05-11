# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" External forces.

Apply an external force to all particles in the simulation. This module organizes all external forces.
As an example, a force derived from a :py:class:`periodic` potential can be used to induce a concentration modulation
in the system.
"""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import force;
import hoomd;

import sys;
import math;


class External(force.Force):

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(_md, self._cpp_class_name)
        else:
            cls = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cls(self._simulation._cpp_sys_def)
        super()._attach()


class Periodic(External):
    """ One-dimension periodic potential.

    `Periodic` specifies that an external force should be added to every
    particle in the simulation to induce a periodic modulation in the particle
    concentration. The force parameters can be set on a per particle type basis.
    The potential can e.g. be used to induce an ordered phase in a
    block-copolymer melt.

    The external potential :math:`V(\\vec{r})` is implemented using the following
    formula:

    .. math::

       V(\\vec{r}) = A * \\tanh\\left[\\frac{1}{2 \\pi p w} \\cos\\left(
       p \\vec{b}_i\\cdot\\vec{r}\\right)\\right]

    where :math:`A` is the ordering parameter, :math:`\\vec{b}_i` is the
    reciprocal lattice vector direction :math:`i=0..2`, :math:`p` the
    periodicity and :math:`w` the interface width (relative to the distance
    :math:`2\\pi/|\\mathbf{b_i}|` between planes in the :math:`i`-direction).
    The modulation is one-dimensional. It extends along the lattice vector
    :math:`\\mathbf{a}_i` of the simulation cell.

    Examples::

        # Apply a periodic composition modulation along the first lattice vector
        periodic = external.Periodic()
        periodic.force_coeff.set('A', A=1.0, i=0, w=0.02, p=3)
        periodic.force_coeff.set('B', A=-1.0, i=0, w=0.02, p=3)
    """
    _cpp_class_name = "PotentialExternalPeriodic"
    def __init__(self):
        params = TypeParameter('params', 'particle_types',
                               TypeParameterDict(i=int, A=float, w=float, p=int,
                                                 len_keys=1)
                               )
        self._add_typeparam(params)

    def process_coeff(self, coeff):
        A = coeff['A'];
        i = coeff['i'];
        w = coeff['w'];
        p = coeff['p'];

        return _hoomd.make_scalar4(_hoomd.int_as_scalar(i), A, w, _hoomd.int_as_scalar(p));

class e_field(_external_force):
    R""" Electric field.

    :py:class:`e_field` specifies that an external force should be
    added to every particle in the simulation that results from an electric field.

    The external potential :math:`V(\vec{r})` is implemented using the following formula:

    .. math::

       V(\vec{r}) = - q_i \vec{E} \cdot \vec{r}


    where :math:`q_i` is the particle charge and :math:`\vec{E}` is the field vector

    Example::

        # Apply an electric field in the x-direction
        e_field = external.e_field((1,0,0))
    """
    def __init__(self, field, name=""):

        # initialize the base class
        _external_force.__init__(self, name);

        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_force = _md.PotentialExternalElectricField(hoomd.context.current.system_definition,self.name);
        else:
            self.cpp_force = _md.PotentialExternalElectricFieldGPU(hoomd.context.current.system_definition,self.name);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs = None;

        self.field_coeff = tuple(field)

    def process_coeff(self, coeff):
        pass;

    def process_field_coeff(self, field):
        return _hoomd.make_scalar3(field[0],field[1],field[2])
