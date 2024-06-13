# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Expanded Gaussian pair potential.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    sphere = hoomd.hpmc.integrate.Sphere()
    sphere.shape['A'] = dict(diameter=0.0)
    simulation.operations.integrator = sphere
"""

import hoomd

from .pair import Pair


@hoomd.logging.modify_namespace(('hpmc', 'pair', 'ExpandedGaussian'))
class ExpandedGaussian(Pair):
    """Expanded Gaussian pair potential (HPMC).

    Args:
        default_r_cut (float): Default cutoff radius :math:`[\\mathrm{length}]`.
        default_r_on (float): Default XPLOR on radius
          :math:`[\\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `ExpandedGaussian` computes the Expanded Gaussian pair potential between 
    every pair of particles in the simulation state. The functional form of the 
    potential, including its behavior under shifting modes, is identical to that
    in the MD pair potential `hoomd.md.pair.ExpandedGaussian`.

    See Also:
        `hoomd.md.pair.ExpandedGaussian`

        `hoomd.md.pair`

    .. rubric:: Example

    .. code-block:: python

        expanded_gaussian =  hoomd.hpmc.pair.ExpandedGaussian()
        expanded_gaussian.params[('A', 'A')] = dict(epsilon=1.0, 
                                                    sigma=1.0, 
                                                    delta=1.0, 
                                                    r_cut=2.5)
        simulation.operations.integrator.pair_potentials = [expanded_gaussian]

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          Energy well depth :math:`\\varepsilon` :math:`[\\mathrm{energy}]`.
        * ``sigma`` (`float`, **required**) -
          Characteristic length scale :math:`\\sigma`
          :math:`[\\mathrm{length}]`.
        * ``delta`` (`float`, **required**) -
          Characteristic length scale :math:`\\delta`
          :math:`[\\mathrm{length}]`.
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

            expanded_gaussian.mode = 'shift'

        Type: `str`
    """
    _cpp_class_name = "PairPotentialExpandedGaussian"

    def __init__(self, default_r_cut=None, default_r_on=0.0, mode='none'):
        if default_r_cut is None:
            default_r_cut = float
        else:
            default_r_cut = float(default_r_cut)

        params = hoomd.data.typeparam.TypeParameter(
            'params', 'particle_types',
            hoomd.data.parameterdicts.TypeParameterDict(
                epsilon=float,
                sigma=float,
                delta=float,
                r_cut=default_r_cut,
                r_on=float(default_r_on),
                len_keys=2))
        self._add_typeparam(params)

        self._param_dict.update(
            hoomd.data.parameterdicts.ParameterDict(
                mode=hoomd.data.typeconverter.OnlyFrom(("none", "shift",
                                                        "xplor"))))
        self.mode = mode
