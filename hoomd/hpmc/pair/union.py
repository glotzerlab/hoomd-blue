# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Union pair potential.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation(particle_types=['A','R'])
    sphere = hoomd.hpmc.integrate.Sphere()
    sphere.shape['A'] = dict(diameter=0.0)
    sphere.shape['R'] = dict(diameter=0.0)
    simulation.operations.integrator = sphere

    lennard_jones =  hoomd.hpmc.pair.LennardJones()
    lennard_jones.params[('A', 'A')] = dict(epsilon=1, sigma=1, r_cut=2.5)
    lennard_jones.params[('A', 'R')] = dict(epsilon=0, sigma=0, r_cut=0)
    lennard_jones.params[('R', 'R')] = dict(epsilon=0, sigma=0, r_cut=0)
"""

import hoomd
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyIf, OnlyTypes, to_type_converter

from .pair import Pair


@hoomd.logging.modify_namespace(('hpmc', 'pair', 'Union'))
class Union(Pair):
    r"""Treat particles as extended bodies.

    Args:
        constituent_potential (hoomd.hpmc.pair.Pair): The pair potential to
            apply between constituent points.

        leaf_capacity (int):  Maximum number of leaf nodes in the tree data
            structure used by this class. The default ``leaf_capacity=0`` uses
            an all N*M code path.

    `Union` computes the potential energy between sets of constituent points
    that rigidly transform about each particle. The union potential between
    a pair of particles is:

    .. math::
        U(\vec{r}_{ij}, \mathbf{q}_i, \mathbf{q}_j)) =
        \sum_{a=1}^{N_{\mathrm{constituents},i}}
        \sum_{b=1}^{N_{\mathrm{constituents},j}}
        U_\mathrm{constituent}(\mathbf{q}_j \vec{P}_{j,b} \mathbf{q}_j^* -
        \vec{P}_{i,a},
        \mathbf{q}_i \mathbf{Q}_{i,a}, \mathbf{q}_j \mathbf{Q}_{j,b})

    where :math:`N_{\mathrm{constituents},i}` is the number of constituents on
    the :math:`i` particle and :math:`U_\mathrm{constituent}` is the potential
    evaluated by the given ``constituent_potential``. :math:`\vec{P}_{i,a}` and
    :math:`\mathbf{Q}_{i,a}` are the constituent postitions and orientations
    with index :math:`a` on the :math:`i` particle.
    :math:`U_\mathrm{constituent}` also depends on the constituent particle
    types and charges (not shown in the equation).

    See Also:
        `hoomd.md.constrain.Rigid` implements a similar evaluation for MD
        simulations.

    Important:
        Unlike `hoomd.md.constrain.Rigid`, `Union` does *not* automatically
        include the central particle in the evaluation. You must add a
        constituent particle with position 0,0,0 when desired.

    Note:
        ``constituent_potential`` may be isotropic or have orientation
        dependence, but it **may not** be another `Union` potential.

    Tip:
        The default ``leaf_capacity=0`` performs best for unions with small
        numbers of constituents. ``leaf_capacity=4`` (or any other non-zero
        value) activates a tree algorithm that may perform better when you have
        many constituents.

    .. rubric:: Example

    .. code-block:: python

        union = hoomd.hpmc.pair.Union(constituent_potential=lennard_jones)
        union.body['R'] = dict(types=['A', 'A', 'A'],
                               positions=[(-1,0,0), (0,0,0), (1,0,0)])
        union.body['A'] = None

        simulation.operations.integrator.pair_potentials = [union]

    The particle types used as constituents must be particle types present in
    the system state, even when there are no actual particles of that type.
    As shown above, set the body for constituent types to ``None`` (which is
    equivalent to ``dict(types=[], positions=[])``).

    .. py:attribute:: body

        The body definition.

        Define the position and orientation of each constituent point relative
        to the position and orientation of the particle (i.e. in the particle
        reference frame). Set a particle type name for each constituent point
        which will be used to determine the constituent potential parameters.

        The dictionary has the following keys:

        - ``types`` (`list` [`str`]): List of types of constituent points.

        - ``positions`` (`list` [`tuple` [`float`, `float`, `float`]]): List of
          relative positions of constituent points.

        - ``orientations`` (`list` [`tuple` [`float`, `float`, `float`,
          `float`]]): List of orientations (as quaternions) of constituent
          points (optional, defaults to ``[(1,0,0,0)] * len(positions)``).

        - ``charges`` (`list` [`float`]): List of charges of constituent points
          (optional, defaults to ``[0] * len(positions)``).

        Type: `TypeParameter` [``particle_type``, `dict`] or `None`

    Attributes:
        leaf_capacity (int):
            Maximum number of leaf nodes in the tree data structure used by this
            class. Set ``leaf_capacity=0`` to use an all N*M code path.

            .. rubric:: Example

            .. code-block:: python

                union.leaf_capacity = 4
    """
    _cpp_class_name = "PairPotentialUnion"

    def __init__(self, constituent_potential, leaf_capacity=0):
        body = TypeParameter(
            'body', 'particle_types',
            TypeParameterDict(OnlyIf(to_type_converter(
                dict(types=[str],
                     positions=[(float,) * 3],
                     orientations=OnlyIf(to_type_converter([(float,) * 4]),
                                         allow_none=True),
                     charges=OnlyIf(to_type_converter([float]),
                                    allow_none=True))),
                                     allow_none=True),
                              len_keys=1,
                              _defaults={
                                  'orientations': None,
                                  'charges': None
                              }))
        self._add_typeparam(body)

        param_dict = ParameterDict(
            leaf_capacity=OnlyTypes(int, allow_none=True))
        param_dict.update(dict(leaf_capacity=leaf_capacity))
        self._param_dict.update(param_dict)

        if not isinstance(constituent_potential, hoomd.hpmc.pair.Pair):
            raise TypeError(
                "constituent_potential must subclass hoomd.hpmc.pair.Pair")
        self._constituent_potential = constituent_potential

    @property
    def constituent_potential(self):
        """hpmc.pair.Pair: Interactions between constituent points.

        .. rubric:: Example

        .. code-block:: python

            union.constituent_potential
        """
        return self._constituent_potential

    def _make_cpp_obj(self):
        cpp_sys_def = self._simulation.state._cpp_sys_def
        cls = getattr(hoomd.hpmc._hpmc, self._cpp_class_name)
        return cls(cpp_sys_def, self.constituent_potential._cpp_obj)

    def _attach_hook(self):
        self.constituent_potential._attach(self._simulation)
        super()._attach_hook()
        self.constituent_potential._cpp_obj.setParent(self._cpp_obj)

    def _detach_hook(self):
        if self.constituent_potential is not None:
            self.constituent_potential._detach()

        super()._detach_hook()
