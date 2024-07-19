# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""External field forces.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    simulation.operations.integrator = hoomd.md.Integrator(dt=0.001)
"""

import hoomd
from hoomd.md import _md
from hoomd.md import force
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter


class Field(force.Force):
    """Base class external field force.

    External potentials represent forces which are applied to all particles in
    the simulation by an external agent.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """

    # Module where the C++ class is defined. Reassign this when developing an
    # external plugin.
    _ext_module = _md

    def _attach_hook(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(_md, self._cpp_class_name)
        else:
            cls = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cls(self._simulation.state._cpp_sys_def)


class Periodic(Field):
    """One-dimension periodic force.

    `Periodic` computes forces and energies that induce a periodic modulation in
    the particle concentration. The modulation is one-dimensional and extends
    along the lattice vector :math:`\\mathbf{a}_i` of the simulation cell. This
    force can, for example, be used to induce an ordered phase in a
    block-copolymer melt.

    The force is computed commensurate with the potential energy:

    .. math::

       U_i(\\vec{r_j}) = A \\tanh\\left[\\frac{1}{2 \\pi p w} \\cos\\left(
       p \\vec{b}_i\\cdot\\vec{r_j}\\right)\\right]

    `Periodic` results in no virial stress due functional dependence on box
    scaled coordinates.

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

    .. rubric:: Example:

    .. code-block:: python

        periodic = hoomd.md.external.field.Periodic()
        periodic.params['A'] = dict(A=1.0, i=0, w=0.02, p=3)
        periodic.params['B'] = dict(A=-1.0, i=0, w=0.02, p=3)
        simulation.operations.integrator.forces = [periodic]
    """
    _cpp_class_name = "PotentialExternalPeriodic"

    def __init__(self):
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(i=int, A=float, w=float, p=int, len_keys=1))
        self._add_typeparam(params)


class Electric(Field):
    """Electric field force.

    `Electric` computes forces, virials, and energies on all particles in
    the simulation state which are consistent with:

    .. math::

       U_i = - q_i \\vec{E} \\cdot \\vec{r}_i


    where :math:`q_i` is the particle charge and :math:`\\vec{E}` is the field
    vector. The field vector :math:`\\vec{E}` must be set per unique particle
    type.

    .. py:attribute:: E

        The electric field vector :math:`\\vec{E}` as a tuple
        :math:`(E_x, E_y, E_z)` :math:`[\\mathrm{energy} \\cdot
        \\mathrm{charge}^{-1} \\cdot \\mathrm{length^{-1}}]`.

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float`]]

    .. rubric:: Example:

    .. code-block:: python

        electric = hoomd.md.external.field.Electric()
        electric.E['A'] = (1, 0, 0)
        simulation.operations.integrator.forces = [electric]
    """
    _cpp_class_name = "PotentialExternalElectricField"

    def __init__(self):
        params = TypeParameter(
            'E', 'particle_types',
            TypeParameterDict((float, float, float), len_keys=1))
        self._add_typeparam(params)


class Magnetic(Field):
    """Magnetic field torque on a magnetic dipole.

    `Magnetic` computes torques and energies on all particles in the simulation
    state which are consistent with:

    .. math::

       U_i = -\\vec{\\mu}_i \\cdot \\vec{B}


    where :math:`\\vec{\\mu}_i` is the magnetic dipole moment of particle
    :math:`i` and :math:`\\vec{B}` is the field vector.

    .. py:attribute:: params

        The `Magnetic` external potential parameters. The dictionary has the
        following keys:

        * ``B`` (`tuple` [`float`, `float`, `float`], **required**) - The
          magnetic field vector in the global reference frame
          :math:`[\\mathrm{energy} \\cdot \\mathrm{time} \\cdot
          \\mathrm{charge}^{-1} \\cdot \\mathrm{length}^{-2} ]`.
        * ``mu`` (`tuple` [`float`, `float`, `float`], **required**) - The
          magnetic dipole moment of the particles type in the particle
          reference frame :math:`[\\mathrm{charge} \\cdot \\mathrm{length}^2
          \\cdot \\mathrm{time}^{-1}]`.

        Type: `TypeParameter` [``particle_type``, `dict`]

    .. rubric:: Example:

    .. code-block:: python

        magnetic = hoomd.md.external.field.Magnetic()
        magnetic.params['A'] = dict(B=(1.0,0.0,0.0), mu=(1.0,0.0,0.0))
        simulation.operations.integrator.forces = [magnetic]
    """
    _cpp_class_name = "PotentialExternalMagneticField"

    def __init__(self):
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(B=(float, float, float),
                              mu=(float, float, float),
                              len_keys=1))
        self._add_typeparam(params)
