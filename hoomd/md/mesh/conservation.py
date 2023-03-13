# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Mesh Conservation potential."""

from hoomd.md.mesh.potential import MeshPotential
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.logging import log


class Area(MeshPotential):
    r"""Area conservation potential.

    :py:class:`Area` specifies a area conservation energy of each
    triangle mesh unit applied to all particles within the mesh.

    .. math::

        U(r) = k \frac{( A(r) - A_0 )^2}{2 \cdot A_0}
    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure constraint.

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the area conservation for the defined mesh.
            As the mesh can only have one type a type name does not have
            to be stated. The dictionary has the following keys:
            * ``k`` (`float`, **required**) - area conservation coefficient
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`
            * ``A0`` (`float`, **required**) - targeted sureface area
              of the whole mesh
              :math:`[\mathrm{length}]^2]`

    Examples::
        area_conservation_potential =
        mesh.conservation.Area(mesh)
        tringle_area_conservation_potential.parameter = dict(k=10.0, A0=250)
    """
    _cpp_class_name = "AreaConservationMeshForceCompute"

    def __init__(self, mesh):
        params = TypeParameter("params", "types",
                               TypeParameterDict(k=float, A0=float, len_keys=1))
        self._add_typeparam(params)

        super().__init__(mesh)

    @log(requires_run=True)
    def area(self):
        """Area of the mesh."""
        return self._cpp_obj.getArea()


class TriangleArea(MeshPotential):
    r"""Triangle Area conservation potential.

    :py:class:`TriangleArea` specifies a area conservation energy by
    applying an area constraint to each triangle of the mesh.

    .. math::
        U(r) = k \sum_{i,j,k \in \mathrm{mesh{ }triangle}}
        \frac{( N_\mathrm{tri} \cdot A_{ijk} - A_0 )^2}
        {2 \cdot A_0 \cdot N_\mathrm{tri}}

    The potential sums over the area :math:`A_{ijk}` of all triplets
    :math:`i`, :math:`j` and :math:`k` that make up a trinagle in the
    mesh. The number of triangles in the mesh are symbolized by
    :math:`N_\mathrm{tri}`.

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure constraint.

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the area conservation for the defined mesh.
            As the mesh can only have one type a type name does not have
            to be stated. The dictionary has the following keys:
            * ``k`` (`float`, **required**) - area conservation coefficient
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`
            * ``A0`` (`float`, **required**) - targeted total surface area
              of the whole mesh
              :math:`[\mathrm{length}]^2`

    Examples::
        tringle_area_conservation_potential =
        mesh.conservation.TriangleArea(mesh)
        tringle_area_conservation_potential.parameter = dict(k=10.0, A0=250)
    """
    _cpp_class_name = "TriangleAreaConservationMeshForceCompute"

    def __init__(self, mesh):
        params = TypeParameter("params", "types",
                               TypeParameterDict(k=float, A0=float, len_keys=1))
        self._add_typeparam(params)

        super().__init__(mesh)

    @log(requires_run=True)
    def area(self):
        """Area of the mesh."""
        return self._cpp_obj.getArea()

class Volume(MeshPotential):
    r"""Volume conservation potential.

    :py:class:`Volume` specifies a volume constraint on the whole mesh
    surface:

    .. math::

        U(r) = k \frac{( V(r) - V_0 )^2}{2 \cdot V_0}

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure constraint.

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the volume constraint for the defined mesh.
            A type name does not have to be stated as the mesh can only
            have one type. The dictionary has the following keys:

            * ``k`` (`float`, **required**) - potential constant
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-3}]`

            * ``V0`` (`float`, **required**) - target volume
              :math:`[\mathrm{length}^{3}]`

    Examples::

        volume = mesh.conservation.Volume(mesh)
        volume.params["mesh"] = dict(k=10.0, V0=100)
    """
    _cpp_class_name = "VolumeConservationMeshForceCompute"

    def __init__(self, mesh):
        params = TypeParameter("params", "types",
                               TypeParameterDict(k=float, V0=float, len_keys=1))
        self._add_typeparam(params)

        super().__init__(mesh)

    @log(requires_run=True)
    def volume(self):
        """Volume of the mesh triangulation."""
        return self._cpp_obj.getVolume()
