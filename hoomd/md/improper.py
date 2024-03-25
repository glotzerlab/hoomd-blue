# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Improper forces.

.. skip: next if(not hoomd.version.md_built)

.. invisible-code-block: python

    if hoomd.version.md_built:
        simulation = hoomd.util.make_example_simulation()
        simulation.operations.integrator = hoomd.md.Integrator(dt=0.001)

Improper force classes apply a force and virial on every particle in the
simulation state commensurate with the potential energy:

.. math::

    U_\mathrm{improper} = \sum_{(i,j,k,l) \in \mathrm{impropers}}
    U_{ijkl}(\chi)

Each improper is defined by an ordered quadruplet of particle tags in the
`hoomd.State` member ``improper_group``. HOOMD-blue does not construct improper
groups, users must explicitly define impropers in the initial condition.

.. image:: md-improper.svg
    :alt: Definition of the improper bond between particles i, j, k, and l.

In an improper group (i,j,k,l), :math:`\chi` is the signed improper angle
between the planes passing through (:math:`\vec{r}_i, \vec{r}_j, \vec{r}_k`) and
(:math:`\vec{r}_j, \vec{r}_k, \vec{r}_l`). This is the same definition used in
dihedrals. Typically, researchers use impropers to force molecules to be planar.

.. rubric Per-particle energies and virials

Improper force classes assign 1/4 of the potential energy to each of the
particles in the improper group:

.. math::

    U_m = \frac{1}{4} \sum_{(i,j,k,l) \in \mathrm{impropers}}
    U_{ijkl}(\chi) [m=i \lor m=j \lor m=k \lor m=l]

and similarly for virials.
"""

import hoomd
from hoomd import md
from hoomd.md import _md


class Improper(md.force.Force):
    """Base class improper force.

    `Improper` is the base class for all improper forces.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """

    # Module where the C++ class is defined. Reassign this when developing an
    # external plugin.
    _ext_module = _md

    def __init__(self):
        super().__init__()

    def _attach_hook(self):
        # check that some impropers are defined
        if self._simulation.state._cpp_sys_def.getImproperData().getNGlobal(
        ) == 0:
            self._simulation.device._cpp_msg.warning(
                "No impropers are defined.\n")

        # Instantiate the c++ implementation.
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_class = getattr(self._ext_module, self._cpp_class_name)
        else:
            cpp_class = getattr(self._ext_module, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_class(self._simulation.state._cpp_sys_def)


class Harmonic(Improper):
    """Harmonic improper force.

    `Harmonic` computes forces, virials, and energies on all impropers in the
    simulation state with:

    .. math::

        U(r) = \\frac{1}{2}k \\left( \\chi - \\chi_{0}  \\right )^2

    Attributes:
        params(`TypeParameter` [``improper type``, `dict`]):
            The parameter of the harmonic impropers for each improper type. The
            dictionary has the following keys:

            * ``k`` (`float`, **required**), potential constant :math:`k`
              :math:`[\\mathrm{energy}]`.
            * ``chi0`` (`float`, **required**), equilibrium angle
              :math:`\\chi_0` :math:`[\\mathrm{radian}]`.

    Example::

        harmonic = hoomd.md.improper.Harmonic()
        harmonic.params['A-B-C-D'] = dict(k=1.0, chi0=0)
    """
    _cpp_class_name = "HarmonicImproperForceCompute"

    def __init__(self):
        super().__init__()
        params = hoomd.data.typeparam.TypeParameter(
            'params', 'improper_types',
            hoomd.data.parameterdicts.TypeParameterDict(
                k=float,
                chi0=hoomd.data.typeconverter.nonnegative_real,
                len_keys=1))
        self._add_typeparam(params)


class Periodic(Improper):
    """Periodic improper force.

    `Periodic` computes forces, virials, and energies on all impropers in the
    simulation state with:

    .. math::

        U(\\chi) = k \\left( 1 + d \\cos(n \\chi - \\chi_{0})  \\right )

    Attributes:
        params(`TypeParameter` [``improper type``, `dict`]):
            The parameter of the harmonic impropers for each improper type. The
            dictionary has the following keys:

            * ``k`` (`float`, **required**), potential constant :math:`k`
              :math:`[\\mathrm{energy}]`.
            * ``chi0`` (`float`, **required**), equilibrium angle
              :math:`\\chi_0` :math:`[\\mathrm{radian}]`.
            * ``n`` (`int`, **required**), periodic number
              :math:`n` :math:`[\\mathrm{dimensionless}]`.
            * ``d`` (`float`, **required**), sign factor
              :math:`d` :math:`[\\mathrm{dimensionless}]`.

    .. rubric:: Example:

    .. code-block:: python

        periodic = hoomd.md.improper.Periodic()
        periodic.params['A-B-C-D'] = dict(k=1.0, n = 1, chi0=0, d=1.0)

    """
    _cpp_class_name = "PeriodicImproperForceCompute"

    def __init__(self):
        super().__init__()
        params = hoomd.data.typeparam.TypeParameter(
            'params', 'improper_types',
            hoomd.data.parameterdicts.TypeParameterDict(
                k=float,
                n=int,
                d=int,
                chi0=hoomd.data.typeconverter.nonnegative_real,
                len_keys=1))
        self._add_typeparam(params)
