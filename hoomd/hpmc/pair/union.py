# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Union pair potential."""

import hoomd
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyIf, OnlyTypes, to_type_converter

from .pair import Pair


@hoomd.logging.modify_namespace(('hpmc', 'pair', 'Union'))
class Union(Pair):
    """Potential of a set of points surrounding a particle body.

    `Union` computes the potential energy of a set of constituent points rigidly
    attached to a particle body. The constituent points on one particle interact
    with consituent points on other particles in a pairwise fashion defined by
    the constituent potential.

    The position and orientation of the constituent points are defined relative
    to the position and orientation of the particle (i.e. in the particle
    reference frame).

    .. py:attribute:: body

        - ``types`` (`list` [`str`]): List of types of constituent points.

        - ``positions`` (`list` [`tuple` [`float`, `float`, `float`]]): List of
          relative positions of constituent points.

        - ``orientations`` (`list` [`tuple` [`float`, `float`, `float`,
          `float`]]): List of orientations (as quaternions) of constituent
          points.

        - ``charges`` (`list` [`float`]): List of charges of constituent points.

        Type: `TypeParameter` [``particle_type``, `dict`]

    Attributes:
        consituent_potential (`hpmc.pair.Pair`):
            Pair potential class defining the interactions of constituent
            points.
        leaf_capacity (int):
            Maximum number of leaf nodes in the tree data structure used by this
            class.

    """
    _cpp_class_name = "PairPotentialUnion"

    def __init__(self, constituent_potential, leaf_capacity=4):
        body = TypeParameter(
            'body', 'particle_types',
            TypeParameterDict(OnlyIf(to_type_converter(
                dict(types=[str],
                     positions=[(float,) * 3],
                     orientations=[(float,) * 4],
                     charges=[float])),
                                     allow_none=True),
                              len_keys=1))
        self._add_typeparam(body)

        param_dict = ParameterDict(
            constituent_potential=OnlyTypes(hoomd.hpmc.pair.Pair,
                                            allow_none=True),
            leaf_capacity=OnlyTypes(int, allow_none=True))
        param_dict.update(
            dict(constituent_potential=constituent_potential,
                 leaf_capacity=leaf_capacity))
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        # attach the constituent potential
        self.constituent_potential._attach(self._simulation)

        # attach the cur
        cpp_sys_def = self._simulation.state._cpp_sys_def
        cls = getattr(hoomd.hpmc._hpmc, self._cpp_class_name)
        self._cpp_obj = cls(cpp_sys_def, self.constituent_potential._cpp_obj)

        self.constituent_potential._cpp_obj.setParent(self._cpp_obj)

        super()._attach_hook()

    def _detach_hook(self):
        if self.constituent_potential is not None:
            self.constituent_potential._detach()
