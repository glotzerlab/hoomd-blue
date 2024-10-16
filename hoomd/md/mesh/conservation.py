# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Mesh Conservation potential.

Mesh conservation force classes apply a force and virial to every mesh vertex
particle based on a global or local quantity :math:`A` of the given mesh
triangulation :math:`T`.

.. math::

    U_\mathrm{conservation} = U(A(T))

See Also:
   See the documentation in `hoomd.mesh.Mesh` for more information on the
   initialization of the mesh object.

"""

from hoomd.md.mesh.potential import MeshConvervationPotential
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.logging import log


class Volume(MeshConvervationPotential):
    r"""Volume conservation potential.

    :py:class:`Volume` specifies a volume constraint on the whole mesh
    surface:

    .. math::

        U(r) = k \frac{( V(r) - V_0 )^2}{2 \cdot V_0}

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure constraint.
        ignore_type (`bool`, *optional*): ignores mesh type if set to `True`
            and calculates the conservation energy considering all triangles in
            the mesh. Defaults to `False`.

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the volume constraint for the defined mesh.
            The dictionary has the following keys:

            * ``k`` (`float`, **required**) - potential constant
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-3}]`

            * ``V0`` (`float`, **required**) - target volume
              :math:`[\mathrm{length}^{3}]`

    Examples::

        volume = mesh.conservation.Volume(mesh)
        volume.params["mesh"] = dict(k=10.0, V0=100)
    """
    _cpp_class_name = "VolumeConservationMeshForceCompute"

    def __init__(self, mesh, ignore_type=False):
        params = TypeParameter("params", "types",
                               TypeParameterDict(k=float, V0=float, len_keys=1))
        self._add_typeparam(params)

        super().__init__(mesh, ignore_type)

    @log(requires_run=True)
    def volume(self):
        """Volume of the mesh triangulation."""
        return self._cpp_obj.getVolume()
