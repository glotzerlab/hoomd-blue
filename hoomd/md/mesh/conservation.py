# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Mesh Area Conservation potential."""

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
            * ``A0`` (`float`, **required**) - desired total sureface area
              :math:`[\mathrm{length}]^2`

    Examples::
        area_conservation_potential = mesh.bond.AreaConservation(mesh)
        area_conservation_potential.parameter = dict(k=10.0, A0=250)
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
