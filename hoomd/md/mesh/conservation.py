# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Mesh Conservation potentials."""

from hoomd.md.mesh.potential import MeshPotential
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.logging import log


class Area(MeshPotential):
    r"""Area conservation potential.

    :py:class:`AreaConservation` specifies a area conservation energy of each
    triangle mesh unit applied to all particles within the mesh.

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure constraint.

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the area conservation for the defined mesh.
            As the mesh can only have one type a type name does not have
            to be stated. The dictionary has the following keys:
            * ``k`` (`float`, **required**) - area conservation coefficient
              :math:`[\mathrm{energy}]`
            * ``A_mesh`` (`float`, **required**) - desired total sureface area
              of the whole mesh
              :math:`[\mathrm{length}]^2`

    Examples::
        area_conservation_potential =
        mesh.conservation.Area(mesh)
        tringle_area_conservation_potential.parameter = dict(k=10.0, A_mesh=250)
    """
    _cpp_class_name = "AreaConservationMeshForceCompute"

    def __init__(self, mesh):
        params = TypeParameter(
            "params", "types",
            TypeParameterDict(k=float, A_mesh=float, len_keys=1))
        self._add_typeparam(params)

        super().__init__(mesh)

    @log(requires_run=True)
    def area(self):
        """Area of the mesh."""
        return self._cpp_obj.getArea()


class TriangleArea(MeshPotential):
    r"""Triangle Area conservation potential.

    :py:class:`AreaConservation` specifies a area conservation energy of each
    triangle mesh unit applied to all particles within the mesh.

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure constraint.

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the area conservation for the defined mesh.
            As the mesh can only have one type a type name does not have
            to be stated. The dictionary has the following keys:
            * ``k`` (`float`, **required**) - area conservation coefficient
              :math:`[\mathrm{energy}]`
            * ``A_mesh`` (`float`, **required**) - desired total sureface area
              of the whole mesh
              :math:`[\mathrm{length}]^2`

    Examples::
        tringle_area_conservation_potential =
        mesh.conservation.TriangleArea(mesh)
        tringle_area_conservation_potential.parameter = dict(k=10.0, A_mesh=250)
    """
    _cpp_class_name = "TriangleAreaConservationMeshForceCompute"

    def __init__(self, mesh):
        params = TypeParameter(
            "params", "types",
            TypeParameterDict(k=float, A_mesh=float, len_keys=1))
        self._add_typeparam(params)

        super().__init__(mesh)

    @log(requires_run=True)
    def area(self):
        """Area of the mesh."""
        return self._cpp_obj.getArea()


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
