# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""External field potentials."""

import hoomd
from hoomd.md import _md
from hoomd.md import force
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter


class Field(force.Force):
    """Constructs the external field potential.

    External potentials represent forces which are applied to all particles in
    the simulation by an external agent.

    Note:
        `Field` is the base class for all external field potentials.
        Users should not instantiate this class directly.
    """

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(_md, self._cpp_class_name)
        else:
            cls = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cls(self._simulation.state._cpp_sys_def)
        super()._attach()


class Periodic(Field):
    """One-dimension periodic potential.

    `Periodic` specifies that an external force should be added to every
    particle in the simulation to induce a periodic modulation in the particle
    concentration. The modulation is one-dimensional and extends along the
    lattice vector :math:`\\mathbf{a}_i` of the simulation cell. The force
    parameters can be set on a per particle type basis. This potential can, for
    example, be used to induce an ordered phase in a block-copolymer melt.

    The external potential :math:`V(\\vec{r})` is implemented using the
    following formula:

    .. math::

       V(\\vec{r}) = A \\tanh\\left[\\frac{1}{2 \\pi p w} \\cos\\left(
       p \\vec{b}_i\\cdot\\vec{r}\\right)\\right]

    The coefficients above must be set per unique particle type.

    .. py:attribute:: params

        The `Periodic` external potential parameters. The dictionary has the
        following keys:

        * ``A`` (`float`, **required**) - Ordering parameter :math:`A` \
            :math:`[\\mathrm{energy}]`.
        * ``i`` (`int`, **required**) - :math:`\\vec{b}_i`, :math:`i=0, 1, 2`, \
            is the simulation box's reciprocal lattice vector in the :math:`i` \
            direction :math:`[\\mathrm{dimensionless}]`.
        * ``w`` (`float`, **required**) - The interface width :math:`w` \
            relative to the distance :math:`2\\pi/|\\mathbf{b_i}|` between \
            planes in the :math:`i`-direction :math:`[\\mathrm{dimensionless}]`.
        * ``p`` (`int`, **required**) - The periodicity :math:`p` of the \
            modulation :math:`[\\mathrm{dimensionless}]`.

        Type: `TypeParameter` [``particle_type``, `dict`]

    Example::

        # Apply a periodic composition modulation along the first lattice vector
        periodic = external.field.Periodic()
        periodic.params['A'] = dict(A=1.0, i=0, w=0.02, p=3)
        periodic.params['B'] = dict(A=-1.0, i=0, w=0.02, p=3)
    """
    _cpp_class_name = "PotentialExternalPeriodic"

    def __init__(self):
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(i=int, A=float, w=float, p=int, len_keys=1))
        self._add_typeparam(params)


class Electric(Field):
    """Electric field.

    `Electric` specifies that an external force should be added to every
    particle in the simulation that results from an electric field.

    The external potential :math:`V(\\vec{r})` is implemented using the
    following formula:

    .. math::

       V(\\vec{r}) = - q_i \\vec{E} \\cdot \\vec{r}


    where :math:`q_i` is the particle charge and :math:`\\vec{E}` is the field
    vector. The field vector :math:`\\vec{E}` must be set per unique particle
    types.

    .. py:attribute:: E

        The electric field vector :math:`\\vec{E}` as a tuple
        :math:`(E_x, E_y, E_z)` :math:`[\\mathrm{energy} \\cdot
        \\mathrm{charge}^{-1} \\cdot \\mathrm{length^{-1}}]`.

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float`]]

    Example::

        # Apply an electric field in the x-direction
        e_field = external.field.Electric()
        e_field.E['A'] = (1, 0, 0)
    """
    _cpp_class_name = "PotentialExternalElectricField"

    def __init__(self):
        params = TypeParameter(
            'E', 'particle_types',
            TypeParameterDict((float, float, float), len_keys=1))
        self._add_typeparam(params)
