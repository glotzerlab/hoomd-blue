# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Improper potentials.

Impropers add forces between specified quadruplets of particles and are
typically used to model rotation about chemical bonds without having bonds to
connect the atoms. Their most common use is to keep structural elements flat,
i.e. model the effect of conjugated double bonds, like in benzene rings and its
derivatives.
"""

import hoomd
from hoomd import md  # required because hoomd.md is not yet available


class Improper(md.force.Force):
    """Improper potential base class.

    Note:
        Use one of the subclasses. Users should not instantiate this class
        directly.
    """

    def __init__(self):
        super().__init__()

    def _attach(self):
        # check that some impropers are defined
        if self._simulation.state._cpp_sys_def.getImproperData().getNGlobal(
        ) == 0:
            self._simulation.device._cpp_msg.warning(
                "No impropers are defined.\n")

        # Instantiate the c++ implementation.
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_class = getattr(hoomd.md._md, self._cpp_class_name)
        else:
            cpp_class = getattr(hoomd.md._md, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_class(self._simulation.state._cpp_sys_def)
        super()._attach()


class Harmonic(Improper):
    """Harmonic improper potential.

    :py:class:`Harmonic` computes the harmonic improper potential energy for
    every defined dihedral quadruplet of particles in the simulation:

    .. math::

        V(r) = \\frac{1}{2}k \\left( \\chi - \\chi_{0}  \\right )^2

    where :math:`\\chi` is angle between two sides of the improper.

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
            hoomd.data.parameterdicts.TypeParameterDict(k=float,
                                                        chi0=float,
                                                        len_keys=1))
        self._add_typeparam(params)
