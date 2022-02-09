# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Angle potentials."""

from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
import hoomd


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


class Cosinesq(Angle):
    r"""Cosine squared angle potential.

    :py:class:`Cosinesq` specifies a cosine squared potential energy
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

        cosinesq = angle.Cosinesq()
        cosinesq.params['polymer'] = dict(k=3.0, t0=0.7851)
        cosinesq.params['backbone'] = dict(k=100.0, t0=1.0)
    """

    _cpp_class_name = 'CosineSqAngleForceCompute'

    def __init__(self):
        super().__init__()
        params = TypeParameter('params', 'angle_types',
                               TypeParameterDict(t0=float, k=float, len_keys=1))
        self._add_typeparam(params)


class PCND(Angle):
    R""" PCND angle potential.

    The command angle.pcnd defines a regular harmonic potential energy between every defined triplet
    of particles in the simulation, but in addition in adds the repulsive part of a PCND pair potential
    between the first and the third particle.

    `B. Levine et. al. 2011 <http://dx.doi.org/10.1021/ct2005193>`_ describes the PCND implementation details in
    HOOMD-blue. Cite it if you utilize the PCND potential in your work.

    The total potential is thus:

    .. math::

        V(\theta) = \frac{1}{2} k \left( \theta - \theta_0 \right)^2

    where :math:`\theta` is the current angle between the three particles
    and either:

    .. math::

        V_{\mathrm{LJ}}(r_{13}) -V_{\mathrm{LJ}}(r_c) \mathrm{~with~~~} V_{\mathrm{LJ}}(r) = 4 \varepsilon \left[
        \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^{6} \right]
        \mathrm{~~~~for~} r <= r_c \mathrm{~~~} r_c = \sigma \cdot 2^{\frac{1}{6}}


    .. math::

        V_{\mathrm{LJ}}(r_{13}) -V_{\mathrm{LJ}}(r_c) \mathrm{~with~~~}
        V_{\mathrm{LJ}}(r) = \frac{27}{4} \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{9} -
        \left( \frac{\sigma}{r} \right)^{6} \right]
        \mathrm{~~~~for~} r <= r_c \mathrm{~~~} r_c = \sigma \cdot \left(\frac{3}{2}\right)^{\frac{1}{3}}


    .. math::

        V_{\mathrm{LJ}}(r_{13}) -V_{\mathrm{LJ}}(r_c) \mathrm{~with~~~}
        V_{\mathrm{LJ}}(r) = \frac{3\sqrt{3}}{2} \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
        \left( \frac{\sigma}{r} \right)^{4} \right]
        \mathrm{~~~~for~} r <= r_c \mathrm{~~~} r_c = \sigma \cdot 3^{\frac{1}{8}}

    with :math:`r_{13}` being the distance between the two outer particles of the angle.

    Coefficients:

    - :math:`\theta_0` - rest angle ``t0`` (in radians)
    - :math:`k` - potential constant ``k`` (in units of energy/radians^2)
    - :math:`\varepsilon` - strength of potential ``epsilon`` (in energy units)
    - :math:`\sigma` - distance of interaction ``sigma`` (in distance units)

    Coefficients :math:`k, \theta_0, \varepsilon``, and :math:`\sigma` and Lennard-Jones exponents pair must be set for
    each type of angle in the simulation using :py:meth:`set_coeff()`.
    """
    
    __cpp_class_name = 'PCNDAngleForceCompute'

    def __init__(self):
        super().__init__()
        params = TypeParameter('params', 'angle_types',
                               TypeParameterDict(Xi=float, tau=float, len_keys=1))
        self._add_typeparam(params)
