# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Dihedral potentials.

Dihedrals add forces between specified quadruplets of particles and used to
model rotation about chemical bonds.

By themselves, dihedrals that have been specified in an input file do nothing.
Only when you specify an dihedral force (e.g. `dihedral.Harmonic`), are forces
actually calculated between the listed particles.

Important: There are multiple conventions pertaining to the dihedral angle (phi)
in the literature. HOOMD utilizes the convention shown in the following figure,
where vectors are defined from the central particles to the outer particles.
These vectors correspond to a stretched state (:math:`\phi = 180` degrees) when
they are anti-parallel and a compact state (:math:`\phi = 0`) when they are
parallel.

.. image:: dihedral-angle-definition.png
    :width: 400 px
    :align: center
    :alt: Dihedral angle definition
"""

from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
import hoomd

import numpy
import math


class Dihedral(Force):
    """Constructs the dihedral bond potential.

    Note:
        :py:class:`Dihedral` is the base class for all dihedral potentials.
        Users should not instantiate this class directly.
    """

    def __init__(self):
        super().__init__()

    def _attach(self):
        # check that some dihedrals are defined
        if self._simulation.state._cpp_sys_def.getDihedralData().getNGlobal(
        ) == 0:
            self._simulation.device._cpp_msg.warning(
                "No dihedrals are defined.\n")

        # create the c++ mirror class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_class = getattr(_md, self._cpp_class_name)
        else:
            cpp_class = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_class(self._simulation.state._cpp_sys_def)
        super()._attach()


class Harmonic(Dihedral):
    r"""Harmonic dihedral potential.

    :py:class:`Harmonic` specifies a harmonic dihedral potential energy between
    every defined dihedral quadruplet of particles in the simulation:

    .. math::

        V(\phi) = \frac{1}{2}k \left( 1 + d \cos\left(n \phi - \phi_0 \right)
               \right)

    where :math:`\phi` is the angle between two sides of the dihedral.

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

        harmonic = dihedral.Harmonic()
        harmonic.params['polymer'] = dict(k=3.0, d=-1, n=3, phi0=0)
        harmonic.params['backbone'] = dict(k=100.0, d=1, n=4, phi0=math.pi/2)
    """
    _cpp_class_name = "HarmonicDihedralForceCompute"

    def __init__(self):
        super().__init__()
        params = TypeParameter(
            'params', 'dihedral_types',
            TypeParameterDict(k=float, d=float, n=int, phi0=float, len_keys=1))
        self._add_typeparam(params)


def _table_eval(theta, V, T, width):
    dth = (2 * math.pi) / float(width - 1)
    i = int(round((theta + math.pi) / dth))
    return (V[i], T[i])


class Table(Dihedral):
    """Tabulated dihedral potential.

    Args:
        width (int): Number of points in the table.

    `Table` computes a user-defined potential and force applied to each
    dihedral.

    The torque :math:`\\tau` is:

    .. math::
        \\tau(\\phi) = \\tau_\\mathrm{table}(\\phi)

    and the potential :math:`V(\\phi)` is:

    .. math::
        V(\\phi) =V_\\mathrm{table}(\\phi)

    where :math:`\\phi` is the dihedral angle for the particles A,B,C,D
    in the dihedral.

    Provide :math:`\\tau_\\mathrm{table}(\\phi)` and
    :math:`V_\\mathrm{table}(\\phi)` on evenly spaced grid points points
    in the range :math:`\\phi \\in [-\\pi,\\pi]`. `Table` linearly
    interpolates values when :math:`\\phi` lies between grid points. The
    torque must be specificed commensurate with the potential: :math:`\\tau =
    -\\frac{\\partial V}{\\partial \\phi}`.

    Attributes:
        params (`TypeParameter` [``dihedral type``, `dict`]):
          The potential parameters. The dictionary has the following keys:

          * ``V`` ((*width*,) `numpy.ndarray` of `float`, **required**) -
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
                V=hoomd.data.typeconverter.NDArrayValidator(numpy.float64),
                tau=hoomd.data.typeconverter.NDArrayValidator(numpy.float64),
                len_keys=1))
        self._add_typeparam(params)

    def _attach(self):
        """Create the c++ mirror class."""
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = _md.TableDihedralForceCompute
        else:
            cpp_cls = _md.TableDihedralForceComputeGPU

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def, self.width)

        Force._attach(self)


class OPLS(Dihedral):
    r"""OPLS dihedral force.

    :py:class:`OPLS` specifies an OPLS-style dihedral potential energy between
    every defined dihedral.

    .. math::

        V(\phi) = \frac{1}{2}k_1 \left( 1 + \cos\left(\phi \right) \right) +
                  \frac{1}{2}k_2 \left( 1 - \cos\left(2 \phi \right) \right) +
                  \frac{1}{2}k_3 \left( 1 + \cos\left(3 \phi \right) \right) +
                  \frac{1}{2}k_4 \left( 1 - \cos\left(4 \phi \right) \right)

    where :math:`\phi` is the angle between two sides of the dihedral and
    :math:`k_n` are the force coefficients in the Fourier series (in energy
    units).

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
        opls.params['backbone'] = dict(k1=1.0, k2=1.0, k3=1.0, k4=1.0)
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
