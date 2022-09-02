# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Mesh surface tension."""

from hoomd.md.mesh.potential import MeshPotential
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict


class SurfaceTension3D(MeshPotential):
    r"""surface tension potential.

    `SurfaceTension3D` applies a surface tension to all mesh triangles
    in ``mesh`` and computes its forces, virials, and energies.

    .. math::

        U_\mathrm{surface tension} = \sigma\sum_{(ijk) \in
        \mathrm{mesh triangles}} A_{ijk}

    For a mesh triangle (ijk), :math:`A_{ijk}` is the area of the triangle
    spanned by mesh vertex particles.

    Args:
        mesh (hoomd.mesh.Mesh): Mesh data structure constraint.

    Attributes:
        params (TypeParameter[``mesh name``,dict]):
            The parameter of the surface tsension for the defined mesh.
            The mesh type name defaults to "mesh". The dictionary has
            the following key:

            * ``sigma`` (`float`, **required**) - surface tension constant
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`

    Examples::

        surface_tension = hoomd.md.mesh.SurfaceTension(mesh)
        surface_tension.params["mesh"] = dict(sigma=10.0)
    """
    _cpp_class_name = "SurfaceTensionMeshForceCompute"

    def __init__(self, mesh):
        params = TypeParameter("params", "types",
                               TypeParameterDict(sigma=float, len_keys=1))
        self._add_typeparam(params)

        super().__init__(mesh)
