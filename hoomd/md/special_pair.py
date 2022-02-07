# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Potentials between special pairs of particles.

Special pairs are used to implement interactions between designated pairs of
particles. They act much like bonds, except that the interaction potential is
typically a pair potential, such as LJ.

By themselves, special pairs that have been specified in an initial
configuration do nothing. Only when you specify an force (i.e.
special_pairs.lj), are forces actually calculated between the listed particles.
"""

from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
import hoomd


class SpecialPair(Force):
    """Base class special pair forces.

    Note:
        :py:class:`SpecialPair` is the base class for all special pair
        potentials. Users should not instantiate this class directly.

    """

    def __init__(self):
        super().__init__()

    def _attach(self):
        # check that some bonds are defined
        if self._simulation.state._cpp_sys_def.getPairData().getNGlobal() == 0:
            self._simulation.device._cpp_msg.error("No pairs are defined.\n")

        # create the c++ mirror class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(_md, self._cpp_class_name)
        else:
            cpp_cls = getattr(_md, self._cpp_class_name + "GPU")
        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def)
        super()._attach()


class LJ(SpecialPair):
    r"""LJ special pair potential.

    :py:class:`LJ` specifies a Lennard-Jones potential energy between the two
    particles in each defined pair.

    This is useful for implementing e.g. special 1-4 interactions in all-atom
    force fields.

    The pair potential uses the standard LJ definition.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{LJ}}(r)  = & 4 \varepsilon
            \left[ \left( \frac{\sigma}{r} \right)^{12} -
                   \left( \frac{\sigma}{r} \right)^{6} \right];
                              & r < r_{\mathrm{cut}} \\
                            = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    where :math:`\vec{r}` is the vector pointing from one particle to the other
    in the bond.

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

    .. versionadded:: 2.1

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
    r"""Coulomb special pair potential.

    :py:class:`Coulomb` specifies a Coulomb potential energy between the two
    particles in each defined pair.

    This is useful for implementing e.g. special 1-4 interactions in all-atom
    force fields. It uses a standard Coulomb interaction with a scaling
    parameter. This allows for using this for scaled 1-4 interactions like in
    OPLS where both the 1-4 LJ and Coulomb interactions are scaled by 0.5.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{Coulomb}}(r)  = & \alpha \cdot
                                     \left[ \frac{q_{a}q_{b}}{r} \right];
                                   & r < r_{\mathrm{cut}} \\
                                 = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    where :math:`\vec{r}` is the vector pointing from one particle to the other
    in the bond.

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

    .. versionadded:: 2.2
    .. versionchanged:: 2.2

    """

    _cpp_class_name = "PotentialSpecialPairCoulomb"

    def __init__(self):
        super().__init__()
        params = TypeParameter("params", "special_pair_types",
                               TypeParameterDict(alpha=float, len_keys=1))
        r_cut = TypeParameter("r_cut", "special_pair_types",
                              TypeParameterDict(float, len_keys=1))
        self._extend_typeparam([params, r_cut])
