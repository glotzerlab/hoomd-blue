# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Angle forces.

Angle force classes apply a force and virial on every particle in the simulation
state commensurate with the potential energy:

.. math::

    U_\mathrm{angle} = \sum_{(i,j,k) \in \mathrm{angles}} U_{ijk}(\theta)

Each angle is defined by an ordered triplet of particle tags in the
`hoomd.State` member ``angle_group``. HOOMD-blue does not construct angle
groups, users must explicitly define angles in the initial condition.

.. image:: md-angle.svg
    :alt: Definition of the angle bond between particles i, j, and k.

In the angle group (i,j,k), :math:`\theta` is the angle between the vectors
:math:`\vec{r}_{ij}` and :math:`\vec{r}_{kj}`.

.. rubric Per-particle energies and virials

Angle force classes assign 1/3 of the potential energy to each of the particles
in the angle group:

.. math::

    U_l = \frac{1}{3} \sum_{(i,j,k) \in \mathrm{angles}}
    U_{ijk}(\theta) [l=i \lor l=j \lor l=k]

and similarly for virials.
"""

from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
import hoomd
import numpy


class Angle(Force):
    """Base class angle force.

    `Angle` is the base class for all angle forces.

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
        # check that some angles are defined
        if self._simulation.state._cpp_sys_def.getAngleData().getNGlobal() == 0:
            self._simulation.device._cpp_msg.warning("No angles are defined.\n")

        # create the c++ mirror class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(self._ext_module, self._cpp_class_name)
        else:
            cpp_cls = getattr(self._ext_module, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def)


class Harmonic(Angle):
    r"""Harmonic angle force.

    `Harmonic` computes forces, virials, and energies on all angles in the
    simulation state with:

    .. math::

        U(\theta) = \frac{1}{2} k \left( \theta - \theta_0 \right)^2

    Attributes:
        params (TypeParameter[``angle type``, dict]):
            The parameter of the harmonic bonds for each particle type.
            The dictionary has the following keys:

            * ``k`` (`float`, **required**) - potential constant :math:`k`
              :math:`[\mathrm{energy} \cdot \mathrm{radians}^{-2}]`

            * ``t0`` (`float`, **required**) - rest angle :math:`\theta_0`
              :math:`[\mathrm{radians}]`

    Examples::

        harmonic = angle.Harmonic()
        harmonic.params['A-A-A'] = dict(k=3.0, t0=0.7851)
        harmonic.params['A-B-A'] = dict(k=100.0, t0=1.0)

    """

    _cpp_class_name = 'HarmonicAngleForceCompute'

    def __init__(self):
        super().__init__()
        params = TypeParameter('params', 'angle_types',
                               TypeParameterDict(t0=float, k=float, len_keys=1))
        self._add_typeparam(params)


class CosineSquared(Angle):
    r"""Cosine squared angle force.

    `CosineSquared` computes forces, virials, and energies on all angles in the
    simulation state with:

    .. math::

        U(\theta) = \frac{1}{2} k \left( \cos\theta - \cos\theta_0 \right)^2

    `CosineSquared` is used in the gromos96 and MARTINI force fields.

    Attributes:
        params (TypeParameter[``angle type``, dict]):
            The parameter of the harmonic bonds for each particle type.
            The dictionary has the following keys:

            * ``k`` (`float`, **required**) - potential constant :math:`k`
              :math:`[\mathrm{energy}]`

            * ``t0`` (`float`, **required**) - rest angle :math:`\theta_0`
              :math:`[\mathrm{radians}]`

    Examples::

        cosinesq = angle.CosineSquared()
        cosinesq.params['A-A-A'] = dict(k=3.0, t0=0.7851)
        cosinesq.params['A-B-A'] = dict(k=100.0, t0=1.0)
    """

    _cpp_class_name = 'CosineSqAngleForceCompute'

    def __init__(self):
        super().__init__()
        params = TypeParameter('params', 'angle_types',
                               TypeParameterDict(t0=float, k=float, len_keys=1))
        self._add_typeparam(params)


class Table(Angle):
    """Tabulated bond force.

    Args:
        width (int): Number of points in the table.

    `Table` computes computes forces, virials, and energies on all angles
    in the simulation given the user defined tables :math:`U` and :math:`\\tau`.

    The torque :math:`\\tau` is:

    .. math::
        \\tau(\\theta) = \\tau_\\mathrm{table}(\\theta)

    and the potential :math:`V(\\theta)` is:

    .. math::
        U(\\theta) =U_\\mathrm{table}(\\theta)

    Provide :math:`\\tau_\\mathrm{table}(\\theta)` and
    :math:`U_\\mathrm{table}(\\theta)` on evenly spaced grid points points
    in the range :math:`\\theta \\in [0,\\pi]`. `Table` linearly
    interpolates values when :math:`\\theta` lies between grid points. The
    torque must be specificed commensurate with the potential: :math:`\\tau =
    -\\frac{\\partial U}{\\partial \\theta}`.

    Attributes:
        params (`TypeParameter` [``angle type``, `dict`]):
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
            "params", "angle_types",
            TypeParameterDict(
                U=hoomd.data.typeconverter.NDArrayValidator(numpy.float64),
                tau=hoomd.data.typeconverter.NDArrayValidator(numpy.float64),
                len_keys=1))
        self._add_typeparam(params)

    def _attach_hook(self):
        """Create the c++ mirror class."""
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = _md.TableAngleForceCompute
        else:
            cpp_cls = _md.TableAngleForceComputeGPU

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def, self.width)
