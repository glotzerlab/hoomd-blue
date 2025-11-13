# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""MeshTrianglePair forces."""

import copy
import warnings
import inspect

import hoomd
from hoomd.md import _md
from hoomd.md.mesh.potential import MeshPotential
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
import numpy as np
from hoomd.data.typeconverter import (
    OnlyFrom,
    nonnegative_real,
    positive_real,
    positive_int,
)


class MeshTrianglePair(MeshPotential):
    r"""Base class mesh triangle pair force.

    `MeshTrianglePair` is the base class for all mesh triangle pair forces.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.

    {inherited}

    ----------

    **Members defined in** `MeshTrianglePair`:

    .. py:attribute:: nlist

        Neighbor list used to compute the pair force.

        Type: `hoomd.md.nlist.NeighborList`

    .. py:attribute:: mode

        *mode*, *optional*: defaults to ``"none"``.
        Possible values: ``"none"``, ``"shift"``, ``"xplor"``

        Type: `str`

    .. py:attribute:: r_cut

        Cuttoff radius beyond which the energy and force are 0
        :math:`[\mathrm{length}]`. *Optional*: defaults to the value
        ``default_r_cut`` specified on construction.

        Type: `TypeParameter` [``particle_type``,
        `float`])
    """

    __doc__ = inspect.cleandoc(__doc__).replace(
        "{inherited}", inspect.cleandoc(MeshPotential._doc_inherited)
    )
    _doc_inherited = (
        MeshPotential._doc_inherited
        + """
    ----------

    **Members inherited from**
    `Pair <hoomd.md.mesh.triangle_pair.MeshTrianglePair>`:

    .. py:attribute:: nlist

        Neighbor list used to compute the pair force.
        `Read more... <hoomd.md.pair.Pair.nlist>`

    .. py:attribute:: mode

        Energy smoothing/cutoff mode.
        `Read more... <hoomd.md.pair.Pair.mode>`

    .. py:attribute:: r_cut

        Cuttoff radius beyond which the energy and force are 0.
        `Read more... <hoomd.md.pair.Pair.r_cut>`
    """
    )

    # The accepted modes for the potential. Should be reset by subclasses with
    # restricted modes.
    _accepted_modes = ("none", "shift", "xplor")

    # Module where the C++ class is defined. Reassign this when developing an
    # external plugin.
    _ext_module = _md

    def __init__(self, nlist, mesh,default_r_cut=None, default_nlist_r_cut=None, mode="none"):
        super().__init__(mesh)
        tp_r_cut = TypeParameter(
            "r_cut", "particle_types", TypeParameterDict(nonnegative_real, len_keys=1)
        )
        if default_r_cut is not None:
            tp_r_cut.default = default_r_cut

        tp_nlist_r_cut = TypeParameter(
            "nlist_r_cut", "particle_types", TypeParameterDict(nonnegative_real, len_keys=1)
        )
        if default_nlist_r_cut is not None:
            tp_nlist_r_cut.default = default_nlist_r_cut

        type_params = [tp_r_cut,tp_nlist_r_cut]

        self._extend_typeparam(type_params)
        self._param_dict.update(
            ParameterDict(
                mode=OnlyFrom(self._accepted_modes), nlist=hoomd.md.nlist.NeighborList
            )
        )
        self.mode = mode
        self.nlist = nlist

    def _attach_hook(self):
        if self.nlist._attached and self._simulation != self.nlist._simulation:
            warnings.warn(
                f"{self} object is creating a new equivalent neighbor list."
                f" This is happending since the force is moving to a new "
                f"simulation. Set a new nlist to suppress this warning.",
                RuntimeWarning,
            )
            self.nlist = copy.deepcopy(self.nlist)
        self.nlist._attach(self._simulation)

        self._attach_mesh()

        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(self._ext_module, self._cpp_class_name)
        else:
            cls = getattr(self._ext_module, self._cpp_class_name + "GPU")
        self.nlist._cpp_obj.setStorageMode(_md.NeighborList.storageMode.full)
        self._cpp_obj = cls(self._simulation.state._cpp_sys_def, self.nlist._cpp_obj,self._mesh._cpp_obj)

    def _detach_hook(self):
        self.nlist._detach()
        self._mesh._detach()

    def _setattr_param(self, attr, value):
        if attr == "nlist":
            self._nlist_setter(value)
            return
        super()._setattr_param(attr, value)

    def _nlist_setter(self, new_nlist):
        if new_nlist is self.nlist:
            return
        if self._attached:
            raise RuntimeError("nlist cannot be set after scheduling.")
        self._param_dict._dict["nlist"] = new_nlist

    def _apply_typeparam_dict(self, cpp_obj, simulation):
        for typeparam in self._typeparam_dict.values():
            try:
                typeparam._attach(cpp_obj, simulation.state)
            except ValueError as err:
                raise err.__class__(
                    f"For {type(self)} in TypeParameter {typeparam.name} {err!s}"
                )


class WCA(MeshTrianglePair):
    r"""Weeks-Chandler-Andersen mesh triangle pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `WCA` computes the Weeks-Chandler-Andersen pair force between the mesh and every particle in the
    simulation state.

    .. math::
        U(r) = 4 \varepsilon \left[ \left(
        \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r}
        \right)^{6} \right]

    Example::

        nl = nlist.Cell()
        wca = triangle_pair.WCA(nl, default_r_cut=3.0)
        wca.params[("A", "A")] = {"sigma": 1.0, "epsilon": 1.0}
        wca.r_cut[("A", "B")] = 3.0

    {inherited}

    ----------

    **Members defined in** `WCA`:

    .. py:attribute:: params

        The LJ potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    """

    _cpp_class_name = "PotentialMeshTriangleLJ"
    __doc__ = inspect.cleandoc(__doc__).replace(
        "{inherited}", inspect.cleandoc(MeshTrianglePair._doc_inherited)
    )

    def __init__(
        self,
        nlist,
        mesh,
        default_r_cut=None,
        default_nlist_r_cut=None,
        mode="none",
    ):
        super().__init__(nlist, mesh, default_r_cut, default_nlist_r_cut, mode)
        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(epsilon=float, sigma=float, len_keys=1),
        )
        self._add_typeparam(params)

__all__ = [
    "WCA",
]
