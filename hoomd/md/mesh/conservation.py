# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Mesh Bond potentials."""

from hoomd.md.mesh.potential import MeshPotential
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict


class Volume(MeshPotential):
    r"""Volume conservation potential.

    :py:class:`Volume` specifies a volume constrainton the whole mesh
    surface.

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure constraint.

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the harmonic bonds for the defined mesh.
            As the mesh can only have one type a type name does not have
            to be stated. The dictionary has the following keys:

            * ``k`` (`float`, **required**) - potential constant
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`

            * ``V0`` (`float`, **required**) - rest length
              :math:`[\mathrm{length}]`

    Examples::

        volume = mesh.conservation.Volume(mesh)
        volume.params["mesh"] = dict(k=10.0, r0=100)
    """
    _cpp_class_name = "VolumeConservationMeshForceCompute"

    def __init__(self, mesh):
        params = TypeParameter("params", "types",
                               TypeParameterDict(k=float, V0=float, len_keys=1))
        self._add_typeparam(params)

        super().__init__(mesh)
