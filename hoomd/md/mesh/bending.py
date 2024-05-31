# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Mesh Bending potentials.

Mesh bending force classes apply a force and virial to every mesh vertex
particle based on the local curvature :math:`K` of the given mesh triangulation.

.. math::

    U_\mathrm{mesh bending} = \sum_{j \in \mathrm{mesh}}
    U_{j}(K(\mathbf{r}_j))

The curvature at each vertex particle :math:`j` is determined at its position
:math:`\mathbf{r}_j`.

See Also:
   See the documentation in `hoomd.mesh.Mesh` for more information on the
   initialization of the mesh object.

"""

from hoomd.md.mesh.potential import MeshPotential
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict


class BendingRigidity(MeshPotential):
    r"""Bending potential.

    :py:class:`BendingRigidity` specifies a bending energy applied to
    all particles within the mesh.

    .. math::

        U(i) = \frac{1}{2} k \sum_{j \in \mathrm{Neigh}(i)}
        ( 1 - cos(\theta_{ij}))

    with :math:`\theta_{ij}` is the angle between the two normal
    directors of the bordering triangles of bond :math:`i` and :math:`j`.

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure constraint.

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the bending energy for the defined mesh:
            * ``k`` (`float`, **required**) - bending stiffness
              :math:`[\mathrm{energy}]`

    Examples::

        bending_potential = mesh.bond.BendingRigidity(mesh)
        bending_potential.parameter = dict(k=10.0)
    """
    _cpp_class_name = "BendingRigidityMeshForceCompute"

    def __init__(self, mesh):
        params = TypeParameter("params", "types",
                               TypeParameterDict(k=float, len_keys=1))
        self._add_typeparam(params)

        super().__init__(mesh)
