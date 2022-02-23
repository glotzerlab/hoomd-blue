# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Angle potentials."""

from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
import hoomd
import numpy


class Angle(Force):
    """Constructs the angular bond potential.

    Note:
        :py:class:`Angle` is the base class for all angular potentials.
        Users should not instantiate this class directly.
    """

    def __init__(self):
        super().__init__()

    def _attach(self):
        # check that some angles are defined
        if self._simulation.state._cpp_sys_def.getAngleData().getNGlobal() == 0:
            self._simulation.device._cpp_msg.warning("No angles are defined.\n")

        # create the c++ mirror class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(_md, self._cpp_class_name)
        else:
            cpp_cls = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def)

        super()._attach()


class Harmonic(Angle):
    r"""Harmonic angle potential.

    :py:class:`Harmonic` specifies a harmonic potential energy between
    every triplet of particles with an angle specified between them.

    .. math::

        V(\theta) = \frac{1}{2} k \left( \theta - \theta_0 \right)^2

    where :math:`\theta` is the angle between the triplet of particles.

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
        harmonic.params['polymer'] = dict(k=3.0, t0=0.7851)
        harmonic.params['backbone'] = dict(k=100.0, t0=1.0)

    """

    _cpp_class_name = 'HarmonicAngleForceCompute'

    def __init__(self):
        super().__init__()
        params = TypeParameter('params', 'angle_types',
                               TypeParameterDict(t0=float, k=float, len_keys=1))
        self._add_typeparam(params)


class CosineSquared(Angle):
    r"""Cosine squared angle potential.

    :py:class:`CosineSquared` specifies a cosine squared potential energy
    between every triplet of particles with an angle specified between them.

    .. math::

        V(\theta) = \frac{1}{2} k \left( \cos\theta - \cos\theta_0 \right)^2

    where :math:`\theta` is the angle between the triplet of particles.
    This angle style is also known as g96, since they were used in the
    gromos96 force field. These are also the types of angles used with the
    coarse-grained MARTINI force field.

    Attributes:
        params (TypeParameter[``angle type``, dict]):
            The parameter of the harmonic bonds for each particle type.
            The dictionary has the following keys:

            * ``k`` (`float`, **required**) - potential constant :math:`k`
              :math:`[\mathrm{energy}]`

            * ``t0`` (`float`, **required**) - rest angle :math:`\theta_0`
              :math:`[\mathrm{radians}]`

    Parameters :math:`k` and :math:`\theta_0` must be set for each type of
    angle in the simulation.  Note that the value of :math:`k` for this angle
    potential is not comparable to the value of :math:`k` for harmonic angles,
    as they have different units.

    Examples::

        cosinesq = angle.CosineSquared()
        cosinesq.params['polymer'] = dict(k=3.0, t0=0.7851)
        cosinesq.params['backbone'] = dict(k=100.0, t0=1.0)
    """

    _cpp_class_name = 'CosineSqAngleForceCompute'

    def __init__(self):
        super().__init__()
        params = TypeParameter('params', 'angle_types',
                               TypeParameterDict(t0=float, k=float, len_keys=1))
        self._add_typeparam(params)


class Table(Angle):
    """Tabulated bond potential.

    Args:
        width (int): Number of points in the table.

    `Table` computes a user-defined potential and force applied to each angle.

    The torque :math:`\\tau` is:

    .. math::
        \\tau(\\theta) = \\tau_\\mathrm{table}(\\theta)

    and the potential :math:`V(\\theta)` is:

    .. math::
        V(\\theta) =V_\\mathrm{table}(\\theta)

    where :math:`\\theta` is the angle between the vectors
    :math:`\\vec{r}_A - \\vec{r}_B` and :math:`\\vec{r}_C - \\vec{r}_B` for
    particles A,B,C in the angle.

    Provide :math:`\\tau_\\mathrm{table}(\\theta)` and
    :math:`V_\\mathrm{table}(\\theta)` on evenly spaced grid points points
    in the range :math:`\\theta \\in [0,\\pi]`. `Table` linearly
    interpolates values when :math:`\\theta` lies between grid points. The
    torque must be specificed commensurate with the potential: :math:`\\tau =
    -\\frac{\\partial V}{\\partial \\theta}`.

    Attributes:
        params (`TypeParameter` [``angle type``, `dict`]):
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
            "params", "angle_types",
            TypeParameterDict(
                V=hoomd.data.typeconverter.NDArrayValidator(numpy.float64),
                tau=hoomd.data.typeconverter.NDArrayValidator(numpy.float64),
                len_keys=1))
        self._add_typeparam(params)

    def _attach(self):
        """Create the c++ mirror class."""
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = _md.TableAngleForceCompute
        else:
            cpp_cls = _md.TableAngleForceComputeGPU

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def, self.width)

        Force._attach(self)


class PCND(Angle):
    r"""Protracted colored noise dynamics.

    :py:class:`PCND` specifies a correlated stochastic force acting on the central particle of 
    every triplet of particles directed along the backbone of a polymer chain. The magnitude of 
    the time-correlated noise \epsilon (t) is obtained via the following equation.

    .. math::

            \frac{d\epsilon (t)}{dt} = \frac{\xi \sqrt{\tau} \eta (t) - \epsilon(t)}{\tau}

    where :math:`\xi` is the root mean square magnitude of the forces, :math:`\tau` is the
    correlation time, and :math:`\eta (t)` is white noise which is uncorrelated in time.

    Attributes:
        params (TypeParameter[``angle type``, dict]):
            The parameter of the PCND bonds for each particle type.
            The dictionary has the following keys:

            * ``Xi`` (`float`, **required**) - RMS force magnitude :math:`\Xi`

            * ``Tau`` (`float`, **required**) - correlation time :math:`\tau`
              :math:`[\mathrm{time}]`

    Example::

        PCND = angle.PCND()
        PCND.params['polymer'] = dict(Xi=0.1, Tau=1000.0)

    """

    _cpp_class_name = 'PCNDAngleForceCompute'

    def __init__(self):
        super().__init__()
        params = TypeParameter('params', 'angle_types',
                               TypeParameterDict(Xi=float, Tau=float, len_keys=1))
        self._add_typeparam(params)
