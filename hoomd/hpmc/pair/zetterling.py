# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Zetterling pair potential.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    sphere = hoomd.hpmc.integrate.Sphere()
    sphere.shape['A'] = dict(diameter=0.0)
    simulation.operations.integrator = sphere
"""

import hoomd
import inspect

from .pair import Pair


@hoomd.logging.modify_namespace(("hpmc", "pair", "Zetterling"))
class Zetterling(Pair):
    """Zetterling pair potential (HPMC).

    Args:
        default_r_cut (float): Default cutoff radius :math:`[\\mathrm{length}]`.
        default_r_on (float): Default XPLOR on radius
          :math:`[\\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Zetterling` computes the oscillating pair potential between every pair of
    particles in the simulation state. The functional form of the potential,
    including its behavior under shifting modes, is identical to that in
    the MD pair potential `hoomd.md.pair.Zetterling`.

    See Also:
    `hoomd.md.pair.Zetterling`

    `hoomd.md.pair`

    .. rubric:: Example

    .. code-block:: python

        zetterling = hoomd.hpmc.pair.Zetterling(mode="shift")
        zetterling.params[("A", "A")] = {
            "A": 1.58,
            "alpha": -0.22,
            "kf": 4.12,
            "B": 0.95533,
            "sigma": 1.0,
            "n": 18.0,
            "r_cut": 2.649,
        }
        simulation.operations.integrator.pair_potentials = [zetterling]

    {inherited}

    ----------

    **Members defined in** `Zetterling`:

    .. py:attribute:: params

        The Zetterling potential parameters. The dictionary has the following keys:

        * ``A`` (`float`, **required**) -
          Energy scale of the first term :math:`A`
          :math:`[\\mathrm{energy}]`
        * ``alpha`` (`float`, **required**) -
          Screening factor :math:`\\alpha`
          :math:`[\\mathrm{length}^{-1}]`
        * ``kf`` (`float`, **required**) -
          Wave number to mimic the Friedel oscillations effect :math:`k_F`
          :math:`[\\mathrm{length}^{-1}]`.
        * ``B`` (`float`, **required**) -
          Energy scale of the second term :math:`B`
          :math:`[\\mathrm{energy}]`.
        * ``sigma`` (`float`, **required**) -
          Repulsive core size :math:`\\sigma` :math:`[\\mathrm{length}]`
        * ``n`` (`float`, **required**) -
          The power to take \\sigma/r in the second term :math:`n`
          :math:`[\\mathrm{dimensionless}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        The energy shifting/smoothing mode: Possible values are:
        ``"none"``, ``"shift"``, and ``"xplor"``.

        .. rubric:: Example

        .. code-block:: python

            zetterling.mode = "shift"

        Type: `str`
    """

    _cpp_class_name = "PairPotentialZetterling"
    __doc__ = inspect.cleandoc(__doc__).replace(
        "{inherited}", inspect.cleandoc(Pair._doc_inherited)
    )

    def __init__(self, default_r_cut=None, default_r_on=0.0, mode="none"):
        if default_r_cut is None:
            default_r_cut = float
        else:
            default_r_cut = float(default_r_cut)

        params = hoomd.data.typeparam.TypeParameter(
            "params",
            "particle_types",
            hoomd.data.parameterdicts.TypeParameterDict(
                A=float,
                alpha=float,
                kf=float,
                B=float,
                sigma=float,
                n=float,
                r_cut=default_r_cut,
                r_on=float(default_r_on),
                len_keys=2,
            ),
        )
        self._add_typeparam(params)

        self._param_dict.update(
            hoomd.data.parameterdicts.ParameterDict(
                mode=hoomd.data.typeconverter.OnlyFrom(("none", "shift", "xplor"))
            )
        )
        self.mode = mode
