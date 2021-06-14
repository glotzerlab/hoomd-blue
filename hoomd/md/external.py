# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" External forces.

Apply an external force to all particles in the simulation. This module
organizes all external forces. As an example, a force derived from a `Periodic`
potential can be used to induce a concentration modulation in the system.
"""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import force
import hoomd

import sys
import math

from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter


class External(force.Force):
    """
    Common External potential documentation.

    Users should not invoke `External` directly. Documentation common to all
    external potentials is located here. External potentials represent forces
    which are applied to all particle in the simulation by an external agent.
    """

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(_md, self._cpp_class_name)
        else:
            cls = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cls(self._simulation.state._cpp_sys_def)
        super()._attach()


class Periodic(External):
    """ One-dimension periodic potential.

    `Periodic` specifies that an external force should be added to every
    particle in the simulation to induce a periodic modulation in the particle
    concentration. The modulation is one-dimensional and extends along the
    lattice vector :math:`\\mathbf{a}_i` of the simulation cell. The force
    parameters can be set on a per particle type basis. This potential can, for
    example, be used to induce an ordered phase in a block-copolymer melt.

    The external potential :math:`V(\\vec{r})` is implemented using the following
    formula:

    .. math::

       V(\\vec{r}) = A * \\tanh\\left[\\frac{1}{2 \\pi p w} \\cos\\left(
       p \\vec{b}_i\\cdot\\vec{r}\\right)\\right]

    The coefficients above must be set per unique particle type.

    .. py:attribute:: params

        The `Periodic` external potential parameters. The dictionary has the
        following keys:

        * ``A`` (`float`, **required**) - Ordering parameter :math:`A`
            (in energy units).
        * ``i`` (`int`, **required**) - :math:`\\vec{b}_i`, :math:`i=0, 1, 2`,
            is the simulation box's reciprocal lattice vector in the :math:`i`
            direction (dimensionless).
        * ``w`` (`float`, **required**) - The interface width :math:`w`
            relative to the distance :math:`2\\pi/|\\mathbf{b_i}|` between
            planes in the :math:`i`-direction. (dimensionless).
        * ``p`` (`int`, **required**) - The periodicity :math:`p` of the
            modulation (dimensionless).

        Type: `TypeParameter` [``particle_type``, `dict`]

    Example::

        # Apply a periodic composition modulation along the first lattice vector
        periodic = external.Periodic()
        periodic.params['A'] = dict(A=1.0, i=0, w=0.02, p=3)
        periodic.params['B'] = dict(A=-1.0, i=0, w=0.02, p=3)
    """
    _cpp_class_name = "PotentialExternalPeriodic"
    def __init__(self):
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(i=int, A=float, w=float, p=int, len_keys=1))
        self._add_typeparam(params)


class ElectricField(External):
    """ Electric field.

    `ElectricField` specifies that an external force should be added to every
    particle in the simulation that results from an electric field.

    The external potential :math:`V(\\vec{r})` is implemented using the following
    formula:

    .. math::

       V(\\vec{r}) = - q_i \\vec{E} \\cdot \\vec{r}


    where :math:`q_i` is the particle charge and :math:`\\vec{E}` is the field
    vector. The field vector :math:`\\vec{E}` must be set per unique particle
    types.

    .. py:attribute:: E

        The electric field vector, :math:`E`, as a tuple (i.e.
        :math:`(E_x, E_y, E_z)`) (units: [energy] [distance^{-1}] [length^{-1}])

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float`]]

    Example::

        # Apply an electric field in the x-direction
        e_field = external.ElectricField()
        e_field.E['A'] = (1, 0, 0)
    """
    _cpp_class_name = "PotentialExternalElectricField"
    def __init__(self):
        params = TypeParameter(
            'E', 'particle_types',
            TypeParameterDict((float, float, float), len_keys=1))
        self._add_typeparam(params)
