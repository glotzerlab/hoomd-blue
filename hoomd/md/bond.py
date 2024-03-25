# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Bond forces.

Bond force classes apply a force and virial on every particle in the simulation
state commensurate with the potential energy:

.. math::

    U_\mathrm{bond} = \sum_{(j,k) \in \mathrm{bonds}} U_{jk}(r)

Each bond is defined by an ordered pair of particle tags in the
`hoomd.State` member ``bond_group``. HOOMD-blue does not construct bond groups,
users must explicitly define bonds in the initial condition.

.. image:: md-bond.svg
    :alt: Definition of the bond between particles j and k.

In the bond group (j,k), :math:`r` is the length of the bond between the
particle positions :math:`r= |\mathrm{minimum\_image}(\vec{r}_k - \vec{r}_j)|`.

.. rubric Per-particle energies and virials

Bond force classes assign 1/2 of the potential energy to each of the particles
in the bond group:

.. math::

    U_i = \frac{1}{2} \sum_{(j,k) \in \mathrm{bonds}}
    U_{jk}(r) [i=j \lor i=k]

and similarly for virials.
"""

from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
import hoomd

import numpy


class Bond(Force):
    """Base class bond force.

    `Bond` is the base class for all bond forces.

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
        """Create the c++ mirror class."""
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(self._ext_module, self._cpp_class_name)
        else:
            cpp_cls = getattr(self._ext_module, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def)


class Harmonic(Bond):
    r"""Harmonic bond force.

    `Harmonic` computes forces, virials, and energies on all bonds in the
    simulation state with:

    .. math::

        U(r) = \frac{1}{2} k \left( r - r_0 \right)^2

    Attributes:
        params (TypeParameter[``bond type``, dict]):
            The parameter of the harmonic bonds for each particle type.
            The dictionary has the following keys:

            * ``k`` (`float`, **required**) - potential constant
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`

            * ``r0`` (`float`, **required**) - rest length
              :math:`[\mathrm{length}]`

    Examples::

        harmonic = bond.Harmonic()
        harmonic.params['A-A'] = dict(k=3.0, r0=2.38)
        harmonic.params['A-B'] = dict(k=10.0, r0=1.0)
    """
    _cpp_class_name = "PotentialBondHarmonic"

    def __init__(self):
        super().__init__()
        params = TypeParameter("params", "bond_types",
                               TypeParameterDict(k=float, r0=float, len_keys=1))
        self._add_typeparam(params)


class FENEWCA(Bond):
    r"""FENE and WCA bond force.

    `FENEWCA` computes forces, virials, and energies on all bonds in the
    simulation state with:

    .. math::

        V(r) = - \frac{1}{2} k r_0^2 \ln \left( 1 - \left( \frac{r -
               \Delta}{r_0} \right)^2 \right) + U_{\mathrm{WCA}}(r)

    where

    .. math::
        U_{\mathrm{WCA}}(r)  =
        \begin{cases} 4 \varepsilon \left[ \left( \frac{\sigma}{r -
                                 \Delta} \right)^{12} - \left( \frac{\sigma}{r -
                                 \Delta} \right)^{6} \right]  + \varepsilon
                               & r-\Delta < 2^{\frac{1}{6}}\sigma\\
        0 & r-\Delta \ge 2^{\frac{1}{6}}\sigma
        \end{cases}

    , :math:`k` is the attractive force strength, :math:`r_0` is the size of the
    bond, :math:`\varepsilon` is the repulsive interaction energy, and
    :math:`\sigma` is the repulsive interaction width.

    Attributes:
        params (TypeParameter[``bond type``, dict]):
            The parameter of the FENEWCA potential bonds.
            The dictionary has the following keys:

            * ``k`` (`float`, **required**) - attractive force strength
              :math:`k` :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`.

            * ``r0`` (`float`, **required**) - size parameter
              :math:`r_0` :math:`[\mathrm{length}]`.

            * ``epsilon`` (`float`, **required**) - repulsive force strength
              :math:`\varepsilon` :math:`[\mathrm{energy}]`.

            * ``sigma`` (`float`, **required**) - repulsive force interaction
              width :math:`\sigma` :math:`[\mathrm{length}]`.

            * ``delta`` (`float`, **required**) - radial shift :math:`\Delta`
              :math:`[\mathrm{length}]`.

    Examples::

        fenewca = bond.FENEWCA()
        fenewca.params['A-A'] = dict(k=3.0, r0=2.38, epsilon=1.0, sigma=1.0,
                                     delta=0.0)
        fenewca.params['A-B'] = dict(k=10.0, r0=1.0, epsilon=0.8, sigma=1.2,
                                     delta=0.0)

    """
    _cpp_class_name = "PotentialBondFENE"

    def __init__(self):
        super().__init__()
        params = TypeParameter(
            "params", "bond_types",
            TypeParameterDict(k=float,
                              r0=float,
                              epsilon=float,
                              sigma=float,
                              delta=float,
                              len_keys=1))
        self._add_typeparam(params)


class Table(Bond):
    """Tabulated bond force.

    Args:
        width (int): Number of points in the table.

    `Table` computes computes forces, virials, and energies on all bonds
    in the simulation given the user defined tables :math:`U` and :math:`F`.

    Note:
        For potentials that diverge near r=0, to set *r_min* to a non-zero
        value.

    The force :math:`\\vec{F}` is:

    .. math::
        \\vec{F}(\\vec{r}) =
        \\begin{cases}
        0 & r < r_{\\mathrm{min}} \\\\
        F_\\mathrm{table}(r)\\hat{r}
        & r_{\\mathrm{min}} \\le r < r_{\\mathrm{max}} \\\\
        0 & r \\ge r_{\\mathrm{max}} \\\\
        \\end{cases}

    and the potential :math:`U(r)` is:

    .. math::
        U(r) =
        \\begin{cases}
        0 & r < r_{\\mathrm{min}} \\\\
        U_\\mathrm{table}(r)
        & r_{\\mathrm{min}} \\le r < r_{\\mathrm{max}} \\\\
        0 & r \\ge r_{\\mathrm{max}} \\\\
        \\end{cases}

    where :math:`\\vec{r}` is the vector pointing from one particle to the other
    in the bond.

    Warning:
        Bonds that stretch to a length :math:`r \\ge r_{\\mathrm{max}}` result
        in an error.

    Provide :math:`F_\\mathrm{table}(r)` and :math:`U_\\mathrm{table}(r)` on
    evenly spaced grid points points between :math:`r_{\\mathrm{min}}` and
    :math:`r_{\\mathrm{max}}`. `Table` linearly interpolates values when
    :math:`r` lies between grid points and between the last grid point and
    :math:`r=r_{\\mathrm{max}}`.  The force must be specificed commensurate with
    the potential: :math:`F = -\\frac{\\partial U}{\\partial r}`.

    Attributes:
        params (`TypeParameter` [``bond type``, `dict`]):
          The potential parameters. The dictionary has the following keys:

          * ``r_min`` (`float`, **required**) - the minimum distance to apply
            the tabulated potential, corresponding to the first element of the
            energy and force arrays :math:`[\\mathrm{length}]`.

          * ``r_max`` (`float`, **required**) - the minimum distance to apply
            the tabulated potential, corresponding to the first element of the
            energy and force arrays :math:`[\\mathrm{length}]`.

          * ``U`` ((*width*,) `numpy.ndarray` of `float`, **required**) -
            the tabulated energy values :math:`[\\mathrm{energy}]`. Must have
            a size equal to `width`.

          * ``F`` ((*width*,) `numpy.ndarray` of `float`, **required**) -
            the tabulated force values :math:`[\\mathrm{force}]`. Must have a
            size equal to `width`.

        width (int): Number of points in the table.
    """

    def __init__(self, width):
        super().__init__()
        param_dict = hoomd.data.parameterdicts.ParameterDict(width=int)
        param_dict['width'] = width
        self._param_dict = param_dict

        params = TypeParameter(
            "params", "bond_types",
            TypeParameterDict(
                r_min=float,
                r_max=float,
                U=hoomd.data.typeconverter.NDArrayValidator(numpy.float64),
                F=hoomd.data.typeconverter.NDArrayValidator(numpy.float64),
                len_keys=1))
        self._add_typeparam(params)

    def _attach_hook(self):
        """Create the c++ mirror class."""
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = self._ext_module.BondTablePotential
        else:
            cpp_cls = self._ext_module.BondTablePotentialGPU

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def, self.width)


class Tether(Bond):
    r"""Tether bond force.

    The tethered network is described in Refs. `Gompper, G. & Kroll, D. M.
    Statistical Mechanics of Membranes and Surfaces 2nd edn (eds Nelson, D. R.
    et al.) 359-426 (World Scientific, 2004)
    <https://www.worldscientific.com/worldscibooks/10.1142/5473>`__ and
    `Noguchi, H. & Gompper, G., Phys. Rev. E 72 011901 (2005)
    <https://link.aps.org/doi/10.1103/PhysRevE.72.011901>`__.

    `Tether` computes forces, virials, and energies on all bonds in the
    simulation state with:

    .. math::

        U(r) = U_{\mathrm{att}}(r) + U_{\mathrm{rep}}(r)

    where :math:`r` is the distance from one particle to the other in the bond.

    .. math::
        U_{\mathrm{att}}(r)  =
        \begin{cases}
        k_b \frac{\exp(1/(l_{c0}-r)}{l_{max}-r}  & r > l_{c0} \\
        0 & r \leq l_{c0} \\
        \end{cases}

    .. math::
        U_{\mathrm{rep}}(r)  =
        \begin{cases}
        k_b \frac{\exp(1/(r-l_{c1})}{r-l_{min}}  & r < l_{c1}\\
        0 & r \ge l_{c1}
        \end{cases}

    .. math::
        l_{min} < l_{c1} < l_{c0} < l_{max}

    Attributes:
        params (TypeParameter[``bond type``, dict]):
            The parameter of the Tethering potential bonds.
            The dictionary has the following keys:

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

        bond_potential = bond.Tether()
        bond_potential.params['A-A'] = dict(k_b=10.0, l_min=0.9, l_c1=1.2,
                                               l_c0=1.8, l_max=2.1)
    """
    _cpp_class_name = "PotentialBondTether"

    def __init__(self):
        super().__init__()
        params = TypeParameter(
            "params", "bond_types",
            TypeParameterDict(k_b=float,
                              l_min=float,
                              l_c1=float,
                              l_c0=float,
                              l_max=float,
                              len_keys=1))
        self._add_typeparam(params)
