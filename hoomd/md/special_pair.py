# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Special pair forces.

Special pair force classes apply a force and virial on every particle in the
simulation state commensurate with the potential energy:

.. math::

    U_\mathrm{special~pairs} = \sum_{(j,k) \in \mathrm{special~pairs}} U_{jk}(r)

Special pairs are used to implement interactions between designated pairs of
particles. They act much like bonds, except that the interaction potential is
typically a pair potential, such as LJ.

Each special pair is defined by an ordered pair of particle tags in the
`hoomd.State` member ``pair_group``. HOOMD-blue does not compute special pair
groups, users must explicitly define special pairs in the initial condition.

.. image:: md-bond.svg
    :alt: Definition of the special pair between particles j and k.

In the special pair group (j,k), :math:`r` is the length of the vector between
the particle positions :math:`r= |\mathrm{minimum\_image}(\vec{r}_k -
\vec{r}_j)|`.

.. rubric Per-particle energies and virials

Special pair force classes assign 1/2 of the potential energy to each of the
particles in the bond group:

.. math::

    U_i = \frac{1}{2} \sum_{(j,k) \in \mathrm{special~pairs}}
    U_{jk}(r) [i=j \lor i=k]

and similarly for virials.
"""

from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
import hoomd


class SpecialPair(Force):
    """Base class special pair forces.

    `SpecialPair` is the base class for all special pair forces.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """

    # Module where the C++ class is defined. Reassign this when developing an
    # external plugin.
    _ext_module = _md

    def __init__(self):
        super().__init__()

    def _attach_hook(self):
        # check that some special pairs are defined
        if self._simulation.state._cpp_sys_def.getPairData().getNGlobal() == 0:
            self._simulation.device._cpp_msg.error("No pairs are defined.\n")

        # create the c++ mirror class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(self._ext_module, self._cpp_class_name)
        else:
            cpp_cls = getattr(self._ext_module, self._cpp_class_name + "GPU")
        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def)


class LJ(SpecialPair):
    r"""LJ special pair force.

    `LJ` computes forces, virials, and energies on all special pairs
    in the simulation state with:

    .. math::
        U(r) =
        \begin{cases}
         4 \varepsilon
            \left[ \left( \frac{\sigma}{r} \right)^{12} -
                   \left( \frac{\sigma}{r} \right)^{6} \right]
                              & r < r_{\mathrm{cut}} \\
         0 & r \ge r_{\mathrm{cut}} \\
        \end{cases}

    Note:
        Use `LJ` to implement special 1-4 interactions in atomistic force
        fields, such as the scaled 1-4 interactions in OPLS where both the 1-4
        `LJ` and `Coulomb` interactions are scaled by 0.5.

    Attributes:
        params (TypeParameter[``special pair type``, dict]):
            The parameter of the lj forces for each particle type.
            The dictionary has the following keys:

            * ``epsilon`` (`float`, **required**) - energy parameter
              :math:`[\mathrm{energy}]`

            * ``sigma`` (`float`, **required**) - particle size
              :math:`[\mathrm{length}]`

        r_cut (TypeParameter[``special pair type``, float]):
            The cut-off distance for special pair potential
            :math:`[\mathrm{length}]`

    Examples::

        lj = special_pair.LJ()
        lj.params['cluster'] = dict(epsilon=3, sigma=0.5)
        lj.r_cut['cluster'] = 5
    """

    _cpp_class_name = "PotentialSpecialPairLJ"

    def __init__(self):
        super().__init__()
        # setup the coefficient options
        params = TypeParameter(
            "params", "special_pair_types",
            TypeParameterDict(epsilon=float, sigma=float, len_keys=1))
        r_cut = TypeParameter("r_cut", "special_pair_types",
                              TypeParameterDict(float, len_keys=1))

        self._extend_typeparam([params, r_cut])


class Coulomb(SpecialPair):
    r"""Coulomb special pair force.

    `Coulomb` computes forces, virials, and energies on all special pairs
    in the simulation state with:

    .. math::
        U(r) =
        \begin{cases}
        \alpha \cdot \left[ \frac{q_{a}q_{b}}{r} \right]
        & r < r_{\mathrm{cut}} \\
        0 & r \ge r_{\mathrm{cut}} \\
        \end{cases}

    Note:
        Use `Coulomb` to implement special 1-4 interactions in atomistic force
        fields, such as the scaled 1-4 interactions in OPLS where both the 1-4
        `LJ` and `Coulomb` interactions are scaled by 0.5.

    Attributes:
        params (TypeParameter[``special pair type``, dict]):
            The parameter of the Coulomb forces for each particle type.
            The dictionary has the following keys:

            * ``alpha`` (`float`, **required**) - Coulomb scaling factor
              :math:`[\mathrm{energy}]`

        r_cut (TypeParameter[``special pair type``, float]):
            The cut-off distance for special pair potential
            :math:`[\mathrm{length}]`

    Examples::

        coulomb = special_pair.Coulomb()
        coulomb.params['cluster'] = dict(alpha=1.0)
        coulomb.r_cut['cluster'] = 2
    """

    _cpp_class_name = "PotentialSpecialPairCoulomb"

    def __init__(self):
        super().__init__()
        params = TypeParameter("params", "special_pair_types",
                               TypeParameterDict(alpha=float, len_keys=1))
        r_cut = TypeParameter("r_cut", "special_pair_types",
                              TypeParameterDict(float, len_keys=1))
        self._extend_typeparam([params, r_cut])
