# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Mesh Bond potentials."""

from hoomd.md.mesh.potential import MeshPotential
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict


class Harmonic(MeshPotential):
    r"""Harmonic bond potential.

    :py:class:`Harmonic` specifies a harmonic potential energy between all
    particles that share an edge within the mesh. For more details see
    :py:class:`hoomd.md.bond.Harmonic`.

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure constraint.

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the harmonic bonds for the defined mesh.
            As the mesh can only have one type a type name does not have
            to be stated. The dictionary has the following keys:

            * ``k`` (`float`, **required**) - potential constant
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`

            * ``r0`` (`float`, **required**) - rest length
              :math:`[\mathrm{length}]`

    Examples::

        harmonic = mesh.bond.Harmonic(mesh)
        harmonic.params["mesh"] = dict(k=10.0, r0=1.0)
    """
    _cpp_class_name = "PotentialMeshBondHarmonic"

    def __init__(self, mesh):
        params = TypeParameter("params", "types",
                               TypeParameterDict(k=float, r0=float, len_keys=1))
        self._add_typeparam(params)

        super().__init__(mesh)


class FENEWCA(MeshPotential):
    r"""FENE and WCA bond potential.

    :py:class:`FENEWCA` specifies a FENE potential energy between all
    particles that share an edge within the mesh. For more details see
    :py:class:`hoomd.md.bond.FENEWCA`.

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure constraint.

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the FENE bonds for the defined mesh.
            As the mesh can only have one type a type name does not have
            to be stated. The dictionary has the following keys:

            * ``k`` (`float`, **required**) - attractive force strength
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`

            * ``r0`` (`float`, **required**) - size parameter
              :math:`[\mathrm{length}]`

            * ``epsilon`` (`float`, **required**) - repulsive force strength
              :math:`[\mathrm{energy}]`

            * ``sigma`` (`float`, **required**) - repulsive force interaction
              :math:`[\mathrm{length}]`

    Examples::

        bond_potential = mesh.bond.FENE(mesh)
        bond_potential.params["mesh"] = dict(k=10.0, r0=1.0,
                                            epsilon=0.8, sigma=1.2, delta=0.0)

    """
    _cpp_class_name = "PotentialMeshBondFENE"

    def __init__(self, mesh):
        params = TypeParameter(
            "params", "types",
            TypeParameterDict(k=float,
                              r0=float,
                              epsilon=float,
                              sigma=float,
                              delta=float,
                              len_keys=1))
        self._add_typeparam(params)

        super().__init__(mesh)


class Tether(MeshPotential):
    r"""Tethering bond potential.

    :py:class:`Tether` specifies a Tethering potential energy between all
    particles that share an edge within the mesh. For more details see
    :py:class:`hoomd.md.bond.Tethering`.

    Args:
        mesh (:py:mod:`hoomd.mesh.Mesh`): Mesh data structure constraint.

    Attributes:
        parameter (TypeParameter[dict]):
            The parameter of the Tethering bonds for the defined mesh.
            As the mesh can only have one type a type name does not have
            to be stated. The dictionary has the following keys:

            * ``k_b`` (`float`, **required**) - bond stiffness
              :math:`[\mathrm{energy}]`

            * ``l_min`` (`float`, **required**) - minimum bond length
              :math:`[\mathrm{length}]`

            * ``l_c1`` (`float`, **required**) - cutoff distance of repulsive
              part :math:`[\mathrm{length}]`

            * ``l_c0`` (`float`, **required**) - cutoff distance of attractive
              part :math:`[\mathrm{length}]`

            * ``l_max`` (`float`, **required**) - maximum bond length
              :math:`[\mathrm{length}]`

    Examples::

        bond_potential = mesh.bond.Tether(mesh)
        bond_potential.params["mesh"] = dict(k_b=10.0, l_min=0.9, l_c1=1.2,
                                         l_c0=1.8, l_max=2.1)
    """
    _cpp_class_name = "PotentialMeshBondTether"

    def __init__(self, mesh):
        params = TypeParameter(
            "params", "types",
            TypeParameterDict(k_b=float,
                              l_min=float,
                              l_c1=float,
                              l_c0=float,
                              l_max=float,
                              len_keys=1))
        self._add_typeparam(params)

        super().__init__(mesh)
