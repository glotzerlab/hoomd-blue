"""Mesh Bond potentials."""

from hoomd.md.mesh.bond import MeshBond
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict


class Helfrich(MeshBond):
    r"""Helfrich bending potential.

    :py:class:`Helfrich` specifies a Helfrich bending energy applied to
    all particles within the mesh.

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure constraint.

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the Helfrich energy for the defined mesh.
            As the mesh can only have one type a type name does not have
            to be stated. The dictionary has the following keys:

            * ``k`` (`float`, **required**) - bending stiffness
              :math:`[\mathrm{energy}]`

    Examples::

        helfrich_potential = mesh.bond.Helfrich(mesh)
        helfrich_potential.parameter = dict(k=10.0)
    """
    _cpp_class_name = "HelfrichMeshForceCompute"

    def __init__(self, mesh):
        params = TypeParameter("params", "types",
                               TypeParameterDict(k=float, len_keys=1))
        self._add_typeparam(params)

        super().__init__(mesh)
