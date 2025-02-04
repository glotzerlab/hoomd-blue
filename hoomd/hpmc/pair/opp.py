# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Oscillating pair potential.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    sphere = hoomd.hpmc.integrate.Sphere()
    sphere.shape['A'] = dict(diameter=0.0)
    simulation.operations.integrator = sphere
"""

import hoomd

from .pair import Pair


@hoomd.logging.modify_namespace(('hpmc', 'pair', 'OPP'))
class OPP(Pair):
    """Oscillating pair potential (HPMC).

    Args:
        default_r_cut (float): Default cutoff radius :math:`[\\mathrm{length}]`.
        default_r_on (float): Default XPLOR on radius
          :math:`[\\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `OPP` computes the oscillating pair potential between every pair
    of particles in the simulation state. The functional form of the potential,
    including its behavior under shifting modes, is identical to that in
    the MD pair potential `hoomd.md.pair.OPP`.

    See Also:
        `hoomd.md.pair.OPP`

        `hoomd.md.pair`

    .. rubric:: Example

    .. code-block:: python

        opp = hoomd.hpmc.pair.OPP()
        opp.params[('A', 'A')] = dict(
            C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=3.14, r_cut=3.0
        )
        simulation.operations.integrator.pair_potentials = [opp]

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``C1`` (`float`, **required**) -
          Energy scale of the first term :math:`C_1`
          :math:`[\\mathrm{energy}]`.
        * ``C2`` (`float`, **required**) -
          Energy scale of the second term :math:`C_2`
          :math:`[\\mathrm{energy}]`
        * ``eta1`` (`float`, **required**) -
          The inverse power to take :math:`r` to in the first term,
          :math:`\\eta_1` :math:`[\\mathrm{dimensionless}]`.
        * ``eta2`` (`float`, **required**) -
          The inverse power to take :math:`r` to in the second term
          :math:`\\eta_2` :math:`[\\mathrm{dimensionless}]`.
        * ``k`` (`float`, **required**) -
          oscillation frequency :math:`k` :math:`[\\mathrm{length}^{-1}]`
        * ``phi`` (`float`, **required**) -
          potential phase shift :math:`\\phi` :math:`[\\mathrm{dimensionless}]`
        * ``r_cut`` (`float`): Cutoff radius :math:`[\\mathrm{length}]`.
          Defaults to the value given in ``default_r_cut`` on construction.
        * ``r_on`` (`float`): XPLOR on radius :math:`[\\mathrm{length}]`.
          Defaults to the value given in ``default_r_on`` on construction.

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        The energy shifting/smoothing mode: Possible values are:
        ``"none"``, ``"shift"``, and ``"xplor"``.

        .. rubric:: Example

        .. code-block:: python

            opp.mode = 'shift'

        Type: `str`
    """
    _cpp_class_name = "PairPotentialOPP"

    def __init__(self, default_r_cut=None, default_r_on=0.0, mode='none'):
        if default_r_cut is None:
            default_r_cut = float
        else:
            default_r_cut = float(default_r_cut)

        params = hoomd.data.typeparam.TypeParameter(
            'params', 'particle_types',
            hoomd.data.parameterdicts.TypeParameterDict(
                C1=float,
                C2=float,
                eta1=float,
                eta2=float,
                k=float,
                phi=float,
                r_cut=default_r_cut,
                r_on=float(default_r_on),
                len_keys=2))
        self._add_typeparam(params)

        self._param_dict.update(
            hoomd.data.parameterdicts.ParameterDict(
                mode=hoomd.data.typeconverter.OnlyFrom(("none", "shift",
                                                        "xplor"))))
        self.mode = mode
