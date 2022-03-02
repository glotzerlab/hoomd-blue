# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Bond potentials."""

from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
import hoomd

import numpy


class Bond(Force):
    """Constructs the bond potential.

    Note:
        :py:class:`Bond` is the base class for all bond potentials.
        Users should not instantiate this class directly.
    """

    def __init__(self):
        super().__init__()

    def _attach(self):
        """Create the c++ mirror class."""
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(_md, self._cpp_class_name)
        else:
            cpp_cls = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def)

        super()._attach()


class Harmonic(Bond):
    r"""Harmonic bond potential.

    :py:class:`Harmonic` specifies a harmonic potential energy between the two
    particles in each defined bond.

    .. math::

        V(r) = \frac{1}{2} k \left( r - r_0 \right)^2

    where :math:`r` is the distance from one particle to the other in the bond.

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
        harmonic.params['polymer'] = dict(k=3.0, r0=2.38)
        harmonic.params['backbone'] = dict(k=10.0, r0=1.0)
    """
    _cpp_class_name = "PotentialBondHarmonic"

    def __init__(self):
        super().__init__()
        params = TypeParameter("params", "bond_types",
                               TypeParameterDict(k=float, r0=float, len_keys=1))
        self._add_typeparam(params)


class FENEWCA(Bond):
    r"""Bond potential that adds FENE to a WCA repulsive potential.

    :py:class:`FENEWCA` computes the energy and force from the FENE and WCA
    potentials between the two particles in each defined bond:

    .. math::

        V(r) = - \frac{1}{2} k r_0^2 \ln \left( 1 - \left( \frac{r -
               \Delta}{r_0} \right)^2 \right) + V_{\mathrm{WCA}}(r)

    where

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{WCA}}(r)  = & 4 \varepsilon \left[ \left( \frac{\sigma}{r -
                                 \Delta} \right)^{12} - \left( \frac{\sigma}{r -
                                 \Delta} \right)^{6} \right]  + \varepsilon;
                               & r-\Delta < 2^{\frac{1}{6}}\sigma\\
                             = & 0;
                               & r-\Delta \ge 2^{\frac{1}{6}}\sigma
        \end{eqnarray*}

    , :math:`r` is the distance from one particle to the other in the bond,
    :math:`k` is the attractive force strength, :math:`r_0` is the size of the
    bond, :math:`\varepsilon` is the repulsive interaction energy, and
    :math:`sigma` is the repulsive interaction width.

    Attributes:
        params (TypeParameter[``bond type``, dict]):
            The parameter of the FENE potential bonds.
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
    """Tabulated bond potential.

    Args:
        width (int): Number of points in the table.

    `Table` computes a user-defined potential and force between the two
    particles in each bond.

    Note:
        For potentials that diverge near r=0, to set *r_min* to a non-zero
        value.

    The force :math:`\\vec{F}` is:

    .. math::
        :nowrap:

        \\begin{eqnarray*}
        \\vec{F}(\\vec{r}) = & 0; & r < r_{\\mathrm{min}} \\\\
                           = & F_\\mathrm{table}(r)\\hat{r};
                             & r_{\\mathrm{min}} \\le r < r_{\\mathrm{max}} \\\\
                           = & 0; & r \\ge r_{\\mathrm{max}} \\\\
        \\end{eqnarray*}

    and the potential :math:`V(r)` is:

    .. math::
        :nowrap:

        \\begin{eqnarray*}
        V(r) = & 0; & r < r_{\\mathrm{min}} \\\\
             = & V_\\mathrm{table}(r);
               & r_{\\mathrm{min}} \\le r < r_{\\mathrm{max}} \\\\
             = & 0; & r \\ge r_{\\mathrm{max}} \\\\
        \\end{eqnarray*}

    where :math:`\\vec{r}` is the vector pointing from one particle to the other
    in the bond.

    Warning:
        Bonds that stretch to a length :math:`r \\ge r_{\\mathrm{max}}` result
        in an error.

    Provide :math:`F_\\mathrm{table}(r)` and :math:`V_\\mathrm{table}(r)` on
    evenly spaced grid points points between :math:`r_{\\mathrm{min}}` and
    :math:`r_{\\mathrm{max}}`. `Table` linearly interpolates values when
    :math:`r` lies between grid points and between the last grid point and
    :math:`r=r_{\\mathrm{max}}`.  The force must be specificed commensurate with
    the potential: :math:`F = -\\frac{\\partial V}{\\partial r}`.

    Attributes:
        params (`TypeParameter` [``bond type``, `dict`]):
          The potential parameters. The dictionary has the following keys:

          * ``r_min`` (`float`, **required**) - the minimum distance to apply
            the tabulated potential, corresponding to the first element of the
            energy and force arrays :math:`[\\mathrm{length}]`.

          * ``r_max`` (`float`, **required**) - the minimum distance to apply
            the tabulated potential, corresponding to the first element of the
            energy and force arrays :math:`[\\mathrm{length}]`.

          * ``V`` ((*width*,) `numpy.ndarray` of `float`, **required**) -
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
                V=hoomd.data.typeconverter.NDArrayValidator(numpy.float64),
                F=hoomd.data.typeconverter.NDArrayValidator(numpy.float64),
                len_keys=1))
        self._add_typeparam(params)

    def _attach(self):
        """Create the c++ mirror class."""
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = _md.BondTablePotential
        else:
            cpp_cls = _md.BondTablePotentialGPU

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def, self.width)

        Force._attach(self)


class Tether(Bond):
    r"""Tethering bond potential.

    :py:class:`Tether` specifies a Tethering potential energy between two
    particles in each defined bond.

    The tethered network is described in Refs. `Gompper, G. & Kroll, D. M.
    Statistical Mechanics of Membranes and Surfaces 2nd edn (eds Nelson, D. R.
    et al.) 359-426 (World Scientific, 2004)
    <https://www.worldscientific.com/worldscibooks/10.1142/5473>`__ and
    `Noguchi, H. & Gompper, G., Phys. Rev. E 72 011901 (2005)
    <https://link.aps.org/doi/10.1103/PhysRevE.72.011901>`__.

    .. math::

        V(r) = V_{\mathrm{att}}(r) + V_{\mathrm{rep}}(r)

    where :math:`r` is the distance from one particle to the other in the bond.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{att}}(r)  = & k_b \frac{exp(1/(l_{c0}-r)}{l_{max}-r};
                                & r > l_{c0}\\
                                = & 0;
                                & r \leq l_{c0}
        \end{eqnarray*}

    .. math::

        \begin{eqnarray*}
        V_{\mathrm{rep}}(r)  = & k_b \frac{exp(1/(r-l_{c1})}{r-l_{min}};
                                & r < l_{c1}\\
                                = & 0;
                                & r \ge l_{c1}
        \end{eqnarray*}

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
        bond_potential.params['tether'] = dict(k_b=10.0, l_min=0.9, l_c1=1.2,
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
