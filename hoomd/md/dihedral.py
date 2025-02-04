# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Dihedral forces.

Dihedral force classes apply a force and virial on every particle in the
simulation state commensurate with the potential energy:

.. math::

    U_\mathrm{dihedral} = \sum_{(i,j,k,l) \in \mathrm{dihedrals}}
    U_{ijkl}(\phi)

Each dihedral is defined by an ordered quadruplet of particle tags in the
`hoomd.State` member ``dihedral_group``. HOOMD-blue does not construct dihedral
groups, users must explicitly define dihedrals in the initial condition.

.. image:: md-dihedral.svg
    :alt: Definition of the dihedral bond between particles i, j, k, and l.

In the dihedral group (i,j,k,l), :math:`\phi` is the signed dihedral angle
between the planes passing through (:math:`\vec{r}_i, \vec{r}_j, \vec{r}_k`) and
(:math:`\vec{r}_j, \vec{r}_k, \vec{r}_l`).

.. rubric Per-particle energies and virials

Dihedral force classes assign 1/4 of the potential energy to each of the
particles in the dihedral group:

.. math::

    U_m = \frac{1}{4} \sum_{(i,j,k,l) \in \mathrm{dihedrals}}
    U_{ijkl}(\phi) [m=i \lor m=j \lor m=k \lor m=l]

and similarly for virials.

Important:
    There are multiple conventions pertaining to the dihedral angle in the
    literature. HOOMD-blue utilizes the convention where :math:`\phi = \pm \pi`
    in the anti-parallel stretched state ( /\\/ ) and :math:`\phi = 0` in the
    parallel compact state ( \|_\| ).
"""

from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
import hoomd

import numpy


class Dihedral(Force):
    """Base class dihedral force.

    `Dihedral` is the base class for all dihedral forces.

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
        # check that some dihedrals are defined
        if self._simulation.state._cpp_sys_def.getDihedralData().getNGlobal(
        ) == 0:
            self._simulation.device._cpp_msg.warning(
                "No dihedrals are defined.\n")

        # create the c++ mirror class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_class = getattr(self._ext_module, self._cpp_class_name)
        else:
            cpp_class = getattr(self._ext_module, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_class(self._simulation.state._cpp_sys_def)


class Periodic(Dihedral):
    r"""Periodic dihedral force.

    `Periodic` computes forces, virials, and energies on all dihedrals in the
    simulation state with:

    .. math::

        U(\phi) = \frac{1}{2}k \left( 1 + d \cos\left(n \phi - \phi_0 \right)
               \right)

    Attributes:
        params (`TypeParameter` [``dihedral type``, `dict`]):
            The parameter of the harmonic bonds for each dihedral type. The
            dictionary has the following keys:

            * ``k`` (`float`, **required**) - potential constant :math:`k`
              :math:`[\mathrm{energy}]`
            * ``d`` (`float`, **required**) - sign factor :math:`d`
            * ``n`` (`int`, **required**) - angle multiplicity factor :math:`n`
            * ``phi0`` (`float`, **required**) - phase shift :math:`\phi_0`
              :math:`[\mathrm{radians}]`

    Examples::

        harmonic = dihedral.Periodic()
        harmonic.params['A-A-A-A'] = dict(k=3.0, d=-1, n=3, phi0=0)
        harmonic.params['A-B-C-D'] = dict(k=100.0, d=1, n=4, phi0=math.pi/2)
    """
    _cpp_class_name = "HarmonicDihedralForceCompute"

    def __init__(self):
        super().__init__()
        params = TypeParameter(
            'params', 'dihedral_types',
            TypeParameterDict(k=float, d=float, n=int, phi0=float, len_keys=1))
        self._add_typeparam(params)


class Table(Dihedral):
    """Tabulated dihedral force.

    Args:
        width (int): Number of points in the table.

    `Table` computes computes forces, virials, and energies on all dihedrals
    in the simulation given the user defined tables :math:`U` and :math:`\\tau`.

    The torque :math:`\\tau` is:

    .. math::
        \\tau(\\phi) = \\tau_\\mathrm{table}(\\phi)

    and the potential :math:`U(\\phi)` is:

    .. math::
        U(\\phi) = U_\\mathrm{table}(\\phi)

    Provide :math:`\\tau_\\mathrm{table}(\\phi)` and
    :math:`U_\\mathrm{table}(\\phi)` on evenly spaced grid points points
    in the range :math:`\\phi \\in [-\\pi,\\pi]`. `Table` linearly
    interpolates values when :math:`\\phi` lies between grid points. The
    torque must be specificed commensurate with the potential: :math:`\\tau =
    -\\frac{\\partial U}{\\partial \\phi}`.

    Attributes:
        params (`TypeParameter` [``dihedral type``, `dict`]):
          The potential parameters. The dictionary has the following keys:

          * ``U`` ((*width*,) `numpy.ndarray` of `float`, **required**) -
            the tabulated energy values :math:`[\\mathrm{energy}]`. Must have
            a size equal to `width`.

          * ``tau`` ((*width*,) `numpy.ndarray` of `float`, **required**) -
            the tabulated torque values :math:`[\\mathrm{force} \\cdot
            \\mathrm{length}]`. Must have a size equal to `width`.

        width (int): Number of points in the table.
    """

    def __init__(self, width):
        super().__init__()
        param_dict = hoomd.data.parameterdicts.ParameterDict(width=int)
        param_dict['width'] = width
        self._param_dict = param_dict

        params = TypeParameter(
            "params", "dihedral_types",
            TypeParameterDict(
                U=hoomd.data.typeconverter.NDArrayValidator(numpy.float64),
                tau=hoomd.data.typeconverter.NDArrayValidator(numpy.float64),
                len_keys=1))
        self._add_typeparam(params)

    def _attach_hook(self):
        """Create the c++ mirror class."""
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = _md.TableDihedralForceCompute
        else:
            cpp_cls = _md.TableDihedralForceComputeGPU

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def, self.width)


class OPLS(Dihedral):
    r"""OPLS dihedral force.

    `OPLS` computes forces, virials, and energies on all dihedrals in the
    simulation state with:

    .. math::

        U(\phi) = \frac{1}{2}k_1 \left( 1 + \cos\left(\phi \right) \right) +
                  \frac{1}{2}k_2 \left( 1 - \cos\left(2 \phi \right) \right) +
                  \frac{1}{2}k_3 \left( 1 + \cos\left(3 \phi \right) \right) +
                  \frac{1}{2}k_4 \left( 1 - \cos\left(4 \phi \right) \right)

    :math:`k_n` are the force coefficients in the Fourier series.

    Attributes:
        params (`TypeParameter` [``dihedral type``, `dict`]):
            The parameter of the OPLS bonds for each particle type.
            The dictionary has the following keys:

            * ``k1`` (`float`, **required**) -  force constant of the
              first term :math:`[\mathrm{energy}]`

            * ``k2`` (`float`, **required**) -  force constant of the
              second term :math:`[\mathrm{energy}]`

            * ``k3`` (`float`, **required**) -  force constant of the
              third term :math:`[\mathrm{energy}]`

            * ``k4`` (`float`, **required**) -  force constant of the
              fourth term :math:`[\mathrm{energy}]`

    Examples::

        opls = dihedral.OPLS()
        opls.params['A-A-A-A'] = dict(k1=1.0, k2=1.0, k3=1.0, k4=1.0)
    """
    _cpp_class_name = "OPLSDihedralForceCompute"

    def __init__(self):
        super().__init__()
        # check that some dihedrals are defined
        params = TypeParameter(
            'params', 'dihedral_types',
            TypeParameterDict(k1=float,
                              k2=float,
                              k3=float,
                              k4=float,
                              len_keys=1))
        self._add_typeparam(params)
