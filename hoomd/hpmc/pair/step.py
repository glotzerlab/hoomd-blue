# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Step function pair potential.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    sphere = hoomd.hpmc.integrate.Sphere()
    sphere.shape['A'] = dict(diameter=0.0)
    simulation.operations.integrator = sphere
"""

import hoomd

from .pair import Pair
from hoomd.data.typeconverter import OnlyIf, to_type_converter


@hoomd.logging.modify_namespace(('hpmc', 'pair', 'Step'))
class Step(Pair):
    r"""Step function pair potential (HPMC).

    `Step` computes a user-defined step function pair potential between every
    pair of particles in the simulation state. The functional form of the
    potential is:

    .. math::

        U(r) =
        \begin{cases}
        \varepsilon_0 & r < r_0 \\
        \varepsilon_k & r_{k-1} \le r < r_{k}, k \in [1,n-1] \\
        0 & r \ge r_{n-1} \\
        \end{cases}

    Where :math:`\varepsilon_k` is the element :math:`k` of the ``epsilon`` list
    and :math:`r_k` is the element :math:`k` of the ``r`` list.

    .. rubric:: Example

    .. code-block:: python

        step =  hoomd.hpmc.pair.Step()
        step.params[('A', 'A')] = dict(epsilon=[1, -1], r=[0.5, 1.5])
        simulation.operations.integrator.pair_potentials = [step]

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`list` [`float`], **required**) - Function values
          :math:`\varepsilon` :math:`[\mathrm{energy}]`.
        * ``r`` (`list` [`float`], **required**) - Points at which function
          values are defined :math:`[\mathrm{length}]`. The values of ``r``
          *must* be listed in monotonically increasing order.

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]
    """
    _cpp_class_name = "PairPotentialStep"

    def __init__(self):
        params = hoomd.data.typeparam.TypeParameter(
            'params', 'particle_types',
            hoomd.data.parameterdicts.TypeParameterDict(OnlyIf(
                to_type_converter({
                    'epsilon': [float],
                    'r': [float],
                }),
                allow_none=True),
                                                        len_keys=2))
        self._add_typeparam(params)
