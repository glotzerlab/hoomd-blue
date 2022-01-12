# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Pair potentials."""

import copy
import warnings

import hoomd
from hoomd.md import _md
from hoomd.md import force
from hoomd.md.nlist import NList
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
import numpy as np
from hoomd.data.typeconverter import (OnlyFrom, OnlyTypes, nonnegative_real)

validate_nlist = OnlyTypes(NList)


class Pair(force.Force):
    r"""Common pair potential documentation.

    Users should not invoke `Pair` directly. It is a base command
    that provides common features to all standard pair forces. Common
    documentation for all pair potentials is documented here.

    All pair force commands specify that a given potential energy and force be
    computed on all non-excluded particle pairs in the system within a short
    range cutoff distance :math:`r_{\mathrm{cut}}`.

    The force :math:`\vec{F}` applied between each pair of particles is:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{F}  = & -\nabla V(r); & r < r_{\mathrm{cut}} \\
                  = & 0;           & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    where :math:`\vec{r}` is the vector pointing from one particle to the other
    in the pair, and :math:`V(r)` is chosen by a mode switch:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  = & V_{\mathrm{pair}}(r); & \mathrm{mode\ is\ no\_shift} \\
              = & V_{\mathrm{pair}}(r) - V_{\mathrm{pair}}(r_{\mathrm{cut}});
              & \mathrm{mode\ is\ shift} \\
              = & S(r) \cdot V_{\mathrm{pair}}(r); & \mathrm{mode\ is\
              xplor\ and\ } r_{\mathrm{on}} < r_{\mathrm{cut}} \\
              = & V_{\mathrm{pair}}(r) - V_{\mathrm{pair}}(r_{\mathrm{cut}});
              & \mathrm{mode\ is\ xplor\ and\ } r_{\mathrm{on}} \ge
              r_{\mathrm{cut}}
        \end{eqnarray*}

    :math:`S(r)` is the XPLOR smoothing function:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        S(r) = & 1; & r < r_{\mathrm{on}} \\
             = & \frac{(r_{\mathrm{cut}}^2 - r^2)^2 \cdot
             (r_{\mathrm{cut}}^2 + 2r^2 -
             3r_{\mathrm{on}}^2)}{(r_{\mathrm{cut}}^2 -
             r_{\mathrm{on}}^2)^3};
               & r_{\mathrm{on}} \le r \le r_{\mathrm{cut}} \\
             = & 0; & r > r_{\mathrm{cut}} \\
         \end{eqnarray*}

    and :math:`V_{\mathrm{pair}}(r)` is the specific pair potential chosen by
    the respective command.

    Enabling the XPLOR smoothing function :math:`S(r)` results in both the
    potential energy and the force going smoothly to 0 at :math:`r =
    r_{\mathrm{cut}}`, reducing the rate of energy drift in long simulations.
    :math:`r_{\mathrm{on}}` controls the point at which the smoothing starts,
    so it can be set to only slightly modify the tail of the potential. It is
    suggested that you plot your potentials with various values of
    :math:`r_{\mathrm{on}}` in order to find a good balance between a smooth
    potential function and minimal modification of the original
    :math:`V_{\mathrm{pair}}(r)`. A good value for the LJ potential is
    :math:`r_{\mathrm{on}} = 2 \cdot \sigma`.

    The split smoothing / shifting of the potential when the mode is ``xplor``
    is designed for use in mixed WCA / LJ systems. The WCA potential and it's
    first derivative already go smoothly to 0 at the cutoff, so there is no need
    to apply the smoothing function. In such mixed systems, set
    :math:`r_{\mathrm{on}}` to a value greater than :math:`r_{\mathrm{cut}}`
    for those pairs that interact via WCA in order to enable shifting of the WCA
    potential to 0 at the cutoff.

    The following coefficients must be set per unique pair of particle types.
    See `hoomd.md.pair` for information on how to set coefficients.


    .. py:attribute:: r_cut

        Cuttoff radius beyond which the energy and force are 0
        :math:`[\mathrm{length}]`. *Optional*: defaults to the value
        ``default_r_cut`` specified on construction.

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `float`])

    .. py:attribute:: r_on

        Radius at which the smoothing modification to the potential starts
        :math:`[\mathrm{length}]`.  *Optional*: defaults to the value
        ``default_r_on`` specified on construction.

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `float`])

    .. py:attribute:: mode

        *mode*, *optional*: defaults to ``"none"``.
        Possible values: ``"none"``, ``"shift"``, ``"xplor"``

        Type: `str`
    """

    # The accepted modes for the potential. Should be reset by subclasses with
    # restricted modes.
    _accepted_modes = ("none", "shift", "xplor")

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        self._nlist = validate_nlist(nlist)
        tp_r_cut = TypeParameter(
            'r_cut', 'particle_types',
            TypeParameterDict(nonnegative_real, len_keys=2))
        if default_r_cut is not None:
            tp_r_cut.default = default_r_cut
        tp_r_on = TypeParameter('r_on', 'particle_types',
                                TypeParameterDict(nonnegative_real, len_keys=2))
        if default_r_on is not None:
            tp_r_on.default = default_r_on

        type_params = [tp_r_cut]
        if 'xplor' in self._accepted_modes:
            type_params.append(tp_r_on)

        self._extend_typeparam(type_params)
        self._param_dict.update(
            ParameterDict(mode=OnlyFrom(self._accepted_modes)))
        self.mode = mode

    def compute_energy(self, tags1, tags2):
        r"""Compute the energy between two sets of particles.

        Args:
            tags1 (``ndarray<int32>``): a numpy array of particle tags in the
                first group.
            tags2 (``ndarray<int32>``): a numpy array of particle tags in the
                second group.

        .. math::

            U = \sum_{i \in \mathrm{tags1}, j \in \mathrm{tags2}} V_{ij}(r)

        where :math:`V_{ij}(r)` is the pairwise energy between two particles
        :math:`i` and :math:`j`.

        Assumed properties of the sets *tags1* and *tags2* are:

        - *tags1* and *tags2* are disjoint
        - all elements in *tags1* and *tags2* are unique
        - *tags1* and *tags2* are contiguous numpy arrays of dtype int32

        None of these properties are validated.

        Examples::

            tags=numpy.linspace(0,N-1,1, dtype=numpy.int32)
            # computes the energy between even and odd particles
            U = mypair.compute_energy(tags1=numpy.array(tags[0:N:2]),
                                      tags2=numpy.array(tags[1:N:2]))

        """
        # TODO future versions could use np functions to test the assumptions
        # above and raise an error if they occur.
        return self._cpp_obj.computeEnergyBetweenSets(tags1, tags2)

    def _add(self, simulation):
        # if nlist was associated with multiple pair forces and is still
        # attached, we need to deepcopy existing nlist.
        nlist = self._nlist
        if (not self._attached and nlist._attached
                and nlist._simulation != simulation):
            warnings.warn(
                f"{self} object is creating a new equivalent neighbor list."
                f" This is happending since the force is moving to a new "
                f"simulation. To supress the warning explicitly set new nlist.",
                RuntimeWarning)
            self._nlist = copy.deepcopy(nlist)
        # We need to check if the force is added since if it is not then this is
        # being called by a SyncedList object and a disagreement between the
        # simulation and nlist._simulation is an error. If the force is added
        # then the nlist is compatible. We cannot just check the nlist's _added
        # property because _add is also called when the SyncedList is synced.
        elif (not self._added and nlist._added
              and nlist._simulation != simulation):
            raise RuntimeError(
                f"NeighborList associated with {self} is associated with "
                f"another simulation.")
        super()._add(simulation)
        # this ideopotent given the above check.
        self._nlist._add(simulation)
        # This is ideopotent, but we need to ensure that if we change
        # neighbor list when not attached we handle correctly.
        self._add_dependency(self._nlist)

    def _attach(self):
        # create the c++ mirror class
        if not self._nlist._added:
            self._nlist._add(self._simulation)
        else:
            if self._simulation != self._nlist._simulation:
                raise RuntimeError("{} object's neighbor list is used in a "
                                   "different simulation.".format(type(self)))
        if not self.nlist._attached:
            self.nlist._attach()
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(_md, self._cpp_class_name)
            self.nlist._cpp_obj.setStorageMode(
                _md.NeighborList.storageMode.half)
        else:
            cls = getattr(_md, self._cpp_class_name + "GPU")
            self.nlist._cpp_obj.setStorageMode(
                _md.NeighborList.storageMode.full)
        self._cpp_obj = cls(self._simulation.state._cpp_sys_def,
                            self.nlist._cpp_obj)

        super()._attach()

    @property
    def nlist(self):
        """Neighbor list used to compute the pair potential."""
        return self._nlist

    @nlist.setter
    def nlist(self, value):
        if self._attached:
            raise RuntimeError("nlist cannot be set after scheduling.")
        nlist = validate_nlist(value)
        if self._added:
            if nlist._added and self._simulation != nlist._simulation:
                raise RuntimeError(
                    "Neighbor lists and forces must belong to the same "
                    "simulation or SyncedList.")
            self._nlist._add(self._simulation)
        self._nlist = nlist

    @property
    def _children(self):
        return [self.nlist]


class LJ(Pair):
    r"""Lennard-Jones pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `LJ` specifies that a Lennard-Jones pair potential should be
    applied between every non-excluded particle pair in the simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{LJ}}(r)  = & 4 \varepsilon \left[ \left(
        \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r}
        \right)^{6} \right]; & r < r_{\mathrm{cut}} \\
        = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    See `Pair` for details on how forces are calculated and the available
    energy shifting and smoothing modes.

    .. py:attribute:: params

        The LJ potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        lj = pair.LJ(nl, default_r_cut=3.0)
        lj.params[('A', 'A')] = {'sigma': 1.0, 'epsilon': 1.0}
        lj.r_cut[('A', 'B')] = 3.0
    """
    _cpp_class_name = "PotentialPairLJ"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, len_keys=2))
        self._add_typeparam(params)


class Gauss(Pair):
    r"""Gaussian pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Gauss` specifies that a Gaussian pair potential should be applied
    between every non-excluded particle pair in the simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{gauss}}(r)  = & \varepsilon \exp \left[ -\frac{1}{2}
                                  \left( \frac{r}{\sigma} \right)^2 \right];
                                  & r < r_{\mathrm{cut}} \\
                                 = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    See `Pair` for details on how forces are calculated and the available
    energy shifting and smoothing modes.

    .. py:attribute:: params

        The Gauss potential parameters. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) - energy parameter
          :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) - particle size :math:`\sigma`
          :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        gauss = pair.Gauss(default_r_cut=3.0, nlist=nl)
        gauss.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
        gauss.r_cut[('A', 'B')] = 3.0
    """
    _cpp_class_name = "PotentialPairGauss"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, len_keys=2))
        self._add_typeparam(params)


class SLJ(Pair):
    r"""Shifted Lennard-Jones pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting mode.

    `SLJ` specifies that a shifted Lennard-Jones type pair potential
    should be applied between every non-excluded particle pair in the
    simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{SLJ}}(r)  = & 4 \varepsilon \left[ \left(
                                \frac{\sigma}{r - \Delta} \right)^{12} -
                                \left( \frac{\sigma}{r - \Delta}
                                \right)^{6} \right]; & r < (r_{\mathrm{cut}}
                                + \Delta) \\
                             = & 0; & r \ge (r_{\mathrm{cut}} + \Delta) \\
        \end{eqnarray*}

    where :math:`\Delta = (d_i + d_j)/2 - 1` and :math:`d_i` is the diameter of
    particle :math:`i`.

    See `Pair` for details on how forces are calculated and the
    available energy shifting and smoothing modes.

    Attention:
        Due to the way that `SLJ` modifies the cutoff criteria, a smoothing mode
        of *xplor* is not supported.

    Set the ``max_diameter`` property of the neighbor list object to the largest
    particle diameter in the system (where **diameter** is a per-particle
    property of the same name in `hoomd.State`).

    Warning:
        Failure to set ``max_diameter`` will result in missing pair
        interactions.

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) - energy parameter
          :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) - particle size :math:`\sigma`
          :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        nl.max_diameter = 2.0
        slj = pair.SLJ(default_r_cut=3.0, nlist=nl)
        slj.params[('A', 'B')] = dict(epsilon=2.0, r_cut=3.0)
        slj.r_cut[('B', 'B')] = 2**(1.0/6.0)
    """
    _cpp_class_name = 'PotentialPairSLJ'

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        if mode == 'xplor':
            raise ValueError("xplor is not a valid mode for SLJ potential")

        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, len_keys=2))
        self._add_typeparam(params)

        # mode not allowed to be xplor, so re-do param dict entry without that
        # option
        param_dict = ParameterDict(mode=OnlyFrom(['none', 'shift']))
        self._param_dict.update(param_dict)
        self.mode = mode

        # this potential needs diameter shifting on
        self._nlist.diameter_shift = True


class ExpandedLJ(Pair):
    r"""Expanded Lennard-Jones pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting mode.

    `ExpandedLJ` specifies that a radially-shifted Lennard-Jones
    type pair potential should be applied between every non-excluded
    particle pair in the simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{ExpandedLJ}}(r)  = & 4 \varepsilon \left[ \left(
                                \frac{\sigma}{r - \Delta} \right)^{12} -
                                \left( \frac{\sigma}{r - \Delta}
                                \right)^{6} \right]; & r < (r_{\mathrm{cut}}) \\
                             = & 0; & r \ge (r_{\mathrm{cut}}) \\
        \end{eqnarray*}

    See `Pair` for details on how forces are calculated and the
    available energy shifting and smoothing modes.

    Note:
        To replicate the behavior of the SLJ potential in HOOMD v2, set
        `hoomd.md.pair.Pair.r_cut` to ``r_cut_unshifted + delta``.

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) - energy parameter
          :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) - particle size :math:`\sigma`
          :math:`[\mathrm{length}]`
        * ``delta`` (`float`, **required**) - radial shift
          :math:`\Delta` :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        expanded_lj = pair.ExpandedLJ(default_r_cut=3.0, nlist=nl)
        expanded_lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0, delta=1.0)
        expanded_lj.params[('A', 'B')] = dict(
                                             epsilon=2.0,
                                             sigma=1.0,
                                             delta=0.75)
        expanded_lj.params[('B', 'B')] = dict(epsilon=1.0, sigma=1.0, delta=0.5)

    """
    _cpp_class_name = 'PotentialPairExpandedLJ'

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float,
                              sigma=float,
                              delta=float,
                              len_keys=2))
        self._add_typeparam(params)
        self.mode = mode


class Yukawa(Pair):
    r"""Yukawa pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Yukawa` specifies that a Yukawa pair potential should be applied between
    every non-excluded particle pair in the simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
          V_{\mathrm{yukawa}}(r) = & \varepsilon \frac{ \exp \left(
          -\kappa r \right) }{r}; & r < r_{\mathrm{cut}} \\
                                  = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    See `Pair` for details on how forces are calculated and the available
    energy shifting and smoothing modes.

    .. py:attribute:: params

        The Yukawa potential parameters. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) - energy parameter
          :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``kappa`` (`float`, **required**) - scaling parameter
          :math:`\kappa` :math:`[\mathrm{length}^{-1}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        yukawa = pair.Yukawa(default_r_cut=3.0, nlist=nl)
        yukawa.params[('A', 'A')] = dict(epsilon=1.0, kappa=1.0)
        yukawa.r_cut[('A', 'B')] = 3.0
    """
    _cpp_class_name = "PotentialPairYukawa"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(kappa=float, epsilon=float, len_keys=2))
        self._add_typeparam(params)


class Ewald(Pair):
    r"""Ewald pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    `Ewald` specifies that a Ewald pair potential should be applied between
    every non-excluded particle pair in the simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
         V_{\mathrm{ewald}}(r)  = & q_i q_j \left[\mathrm{erfc}\left(\kappa
                                    r + \frac{\alpha}{2\kappa}\right)
                                    \exp(\alpha r) \\
                                    + \mathrm{erfc}\left(\kappa r -
                                    \frac{\alpha}{2 \kappa}\right)
                                    \exp(-\alpha r)\right];
                                    & r < r_{\mathrm{cut}} \\
                            = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    Call `md.long_range.pppm.make_pppm_coulomb_forces` to create an instance
    of `Ewald` and `md.long_range.pppm.Coulomb` that together implement the PPPM
    method for electrostatics.

    See `Pair` for details on how forces are calculated. Note Ewald does not
    support energy shifting or smoothing.

    .. py:attribute:: params

        The Ewald potential parameters. The dictionary has the following keys:

        * ``kappa`` (`float`, **required**) - Splitting parameter
          :math:`\kappa` :math:`[\mathrm{length}^{-1}]`
        * ``alpha`` (`float`, **required**) - Debye screening length
          :math:`\alpha` :math:`[\mathrm{length}^{-1}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        ewald = pair.Ewald(default_r_cut=3.0, nlist=nl)
        ewald.params[('A', 'A')] = dict(kappa=1.0, alpha=1.5)
        ewald.r_cut[('A', 'B')] = 3.0
    """
    _cpp_class_name = "PotentialPairEwald"
    _accepted_modes = ("none",)

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist=nlist,
                         default_r_cut=default_r_cut,
                         default_r_on=0,
                         mode='none')
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(kappa=float, alpha=0.0, len_keys=2))

        self._add_typeparam(params)


class Table(Pair):
    """Tabulated pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\\mathrm{length}]`.

    `Table` specifies that a tabulated pair potential should be applied between
    every non-excluded particle pair in the simulation in the range
    :math:`[r_{\\mathrm{min}}, r_{\\mathrm{cut}})`

    Note:
        For potentials that diverge near r=0, to set *r_min* to a non-zero
        value.

    The force :math:`\\vec{F}` is:

    .. math::
        :nowrap:

        \\begin{eqnarray*}
        \\vec{F}(\\vec{r}) = & 0; & r < r_{\\mathrm{min}} \\\\
                           = & F(r)\\hat{r};
                             & r_{\\mathrm{min}} \\le r < r_{\\mathrm{cut}} \\\\
                           = & 0; & r \\ge r_{\\mathrm{cut}} \\\\
        \\end{eqnarray*}

    and the potential :math:`V(r)` is:

    .. math::
        :nowrap:

        \\begin{eqnarray*}
        V(r) = & 0; & r < r_{\\mathrm{min}} \\\\
             = & V(r);
               & r_{\\mathrm{min}} \\le r < r_{\\mathrm{cut}} \\\\
             = & 0; & r \\ge r_{\\mathrm{cut}} \\\\
        \\end{eqnarray*}

    where :math:`\\vec{r}` is the vector pointing from one particle to the other
    in the pair, ``r_min`` is defined in `params`, and ``r_cut`` is defined in
    `Pair.r_cut`.

    Provide :math:`F(r)` and :math:`V(r)` on an evenly space set of grid points
    points between :math:`r_{\\mathrm{min}}` and :math:`r_{\\mathrm{cut}}`.
    `Table` linearly interpolates values when :math:`r` lies between grid points
    and between the last grid point and :math:`r=r_{\\mathrm{cut}}`.  The force
    must be specificed commensurate with the potential: :math:`F =
    -\\frac{\\partial V}{\\partial r}`.

    `Table` does not support energy shifting or smoothing modes.

    Attributes:
        params (`TypeParameter` [\
          `tuple` [``particle_type``, ``particle_type``],\
          `dict`]):
          The potential parameters. The dictionary has the following keys:

          * ``r_min`` (`float`, **required**) - the minimum distance to apply
            the tabulated potential, corresponding to the first element of the
            energy and force arrays :math:`[\\mathrm{length}]`.

          * ``V`` ((*N*,) `numpy.ndarray` of `float`, **required**) -
            the tabulated energy values :math:`[\\mathrm{energy}]`.

          * ``F`` ((*N*,) `numpy.ndarray` of `float`, **required**) -
            the tabulated force values :math:`[\\mathrm{force}]`. Must have the
            same length as ``V``.

    Note:
        The implicitly defined :math:`r` values are those that would be returned
        by ``numpy.linspace(r_min, r_cut, len(V), endpoint=False)``.

    Tip:
        Define non-interacting potentials with::

            table.params[(type1, type2)] = dict(r_min=0, V=[0], F=[0])
            table.r_cut[(type1, type2)] = 0

        There must be at least one element in V and F, but the ``r_cut`` value
        of 0 disables the interaction entirely.
    """
    _cpp_class_name = "PotentialPairTable"
    _accepted_modes = ("none",)

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist,
                         default_r_cut=default_r_cut,
                         default_r_on=0,
                         mode='none')
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(
                r_min=float,
                V=hoomd.data.typeconverter.NDArrayValidator(np.float64),
                F=hoomd.data.typeconverter.NDArrayValidator(np.float64),
                len_keys=2))
        self._add_typeparam(params)


class Morse(Pair):
    r"""Morse pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Morse` specifies that a Morse pair potential should be applied between
    every non-excluded particle pair in the simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{morse}}(r) = & D_0 \left[ \exp \left(-2\alpha\left(
            r-r_0\right)\right) -2\exp \left(-\alpha\left(r-r_0\right)
            \right) \right]; & r < r_{\mathrm{cut}} \\
            = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    See `Pair` for details on how forces are calculated and the available
    energy shifting and smoothing modes.

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``D0`` (`float`, **required**) - depth of the potential at its
          minimum :math:`D_0` :math:`[\mathrm{energy}]`
        * ``alpha`` (`float`, **required**) - the width of the potential well
          :math:`\alpha` :math:`[\mathrm{length}^{-1}]`
        * ``r0`` (`float`, **required**) - position of the minimum
          :math:`r_0` :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        morse = pair.Morse(default_r_cut=3.0, nlist=nl)
        morse.params[('A', 'A')] = dict(D0=1.0, alpha=3.0, r0=1.0)
        morse.r_cut[('A', 'B')] = 3.0
    """

    _cpp_class_name = "PotentialPairMorse"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(D0=float, alpha=float, r0=float, len_keys=2))
        self._add_typeparam(params)


class DPD(Pair):
    r"""Dissipative Particle Dynamics.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list
        kT (`hoomd.variant` or `float`): Temperature of
            thermostat :math:`[\mathrm{energy}]`.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    `DPD` specifies that a DPD pair force should be applied between every
    non-excluded particle pair in the simulation, including an interaction
    potential, pairwise drag force, and pairwise random force. See `Groot and
    Warren 1997 <http://dx.doi.org/10.1063/1.474784>`_.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        F = F_{\mathrm{C}}(r) + F_{\mathrm{R,ij}}(r_{ij}) +
        F_{\mathrm{D,ij}}(v_{ij}) \\
        \end{eqnarray*}

    .. math::
        :nowrap:

        \begin{eqnarray*}
        F_{\mathrm{C}}(r) = & A \cdot  w(r_{ij}) \\
        F_{\mathrm{R, ij}}(r_{ij}) = & - \theta_{ij}\sqrt{3}
        \sqrt{\frac{2k_b\gamma T}{\Delta t}}\cdot w(r_{ij})  \\
        F_{\mathrm{D, ij}}(r_{ij}) = & - \gamma w^2(r_{ij})\left(
        \hat r_{ij} \circ v_{ij} \right)  \\
        \end{eqnarray*}

    .. math::
        :nowrap:

        \begin{eqnarray*}
        w(r_{ij}) = &\left( 1 - r/r_{\mathrm{cut}} \right);
        & r < r_{\mathrm{cut}} \\
                  = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    where :math:`\hat r_{ij}` is a normalized vector from particle i to
    particle j, :math:`v_{ij} = v_i - v_j`, and :math:`\theta_{ij}` is a
    uniformly distributed random number in the range [-1, 1].

    `C. L. Phillips et. al. 2011 <http://dx.doi.org/10.1016/j.jcp.2011.05.021>`_
    describes the DPD implementation details in HOOMD-blue. Cite it if you
    utilize the DPD functionality in your work.

    `DPD` does not implement any energy shift / smoothing modes due to the
    function of the force.

    To use the DPD thermostat, an `hoomd.md.methods.NVE` integrator
    must be applied to the system and the user must specify a temperature.  Use
    of the dpd thermostat pair force with other integrators will result in
    unphysical behavior. To use pair.dpd with a different conservative potential
    than :math:`F_C`, set A to zero and define the conservative pair potential
    separately.  Note that DPD thermostats are often defined in terms of
    :math:`\sigma` where :math:`\sigma = \sqrt{2k_b\gamma T}`.

    .. py:attribute:: params

        The force parameters. The dictionary has the following keys:

        * ``A`` (`float`, **required**) - :math:`A` :math:`[\mathrm{force}]`
        * ``gamma`` (`float`, **required**) - :math:`\gamma`
          :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        dpd = pair.DPD(nlist=nl, kT=1.0, default_r_cut=1.0)
        dpd.params[('A', 'A')] = dict(A=25.0, gamma=4.5)
        dpd.params[('A', 'B')] = dict(A=40.0, gamma=4.5)
        dpd.params[('B', 'B')] = dict(A=25.0, gamma=4.5)
        dpd.params[(['A', 'B'], ['C', 'D'])] = dict(A=40.0, gamma=4.5)
    """
    _cpp_class_name = "PotentialPairDPDThermoDPD"
    _accepted_modes = ("none",)

    def __init__(
        self,
        nlist,
        kT,
        default_r_cut=None,
    ):
        super().__init__(nlist=nlist,
                         default_r_cut=default_r_cut,
                         default_r_on=0,
                         mode='none')
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(A=float, gamma=float, len_keys=2))
        self._add_typeparam(params)
        param_dict = ParameterDict(kT=hoomd.variant.Variant)
        param_dict["kT"] = kT
        self._param_dict.update(param_dict)

    def _add(self, simulation):
        """Add the operation to a simulation.

        DPD uses RNGs. Warn the user if they did not set the seed.
        """
        if isinstance(simulation, hoomd.Simulation):
            simulation._warn_if_seed_unset()

        super()._add(simulation)


class DPDConservative(Pair):
    r"""DPD Conservative pair force.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    `DPDConservative` specifies the conservative part of the DPD pair potential
    should be applied between every non-excluded particle pair in the
    simulation. No thermostat (e.g. Drag Force and Random Force) is applied, as
    is in `DPD`.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{DPD-C}}(r) = & A \cdot \left( r_{\mathrm{cut}} - r
          \right) - \frac{1}{2} \cdot \frac{A}{r_{\mathrm{cut}}} \cdot
          \left(r_{\mathrm{cut}}^2 - r^2 \right);
          & r < r_{\mathrm{cut}} \\
                              = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}


    `DPDConservative` does not implement any energy shift / smoothing modes due
    to the function of the force.

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``A`` (`float`, **required**) - :math:`A` :math:`[\mathrm{force}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        dpdc = pair.DPDConservative(nlist=nl, default_r_cut=3.0)
        dpdc.params[('A', 'A')] = dict(A=1.0)
        dpdc.params[('A', 'B')] = dict(A=2.0, r_cut = 1.0)
        dpdc.params[(['A', 'B'], ['C', 'D'])] = dict(A=3.0)
    """
    _cpp_class_name = "PotentialPairDPD"
    _accepted_modes = ("none",)

    def __init__(self, nlist, default_r_cut=None):
        # initialize the base class
        super().__init__(nlist=nlist,
                         default_r_cut=default_r_cut,
                         default_r_on=0,
                         mode='none')
        params = TypeParameter('params', 'particle_types',
                               TypeParameterDict(A=float, len_keys=2))
        self._add_typeparam(params)


class DPDLJ(Pair):
    r"""Dissipative Particle Dynamics with a LJ conservative force.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        kT (`hoomd.variant` or `float`): Temperature of
            thermostat :math:`[\mathrm{energy}]`.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting mode.

    `DPDLJ` specifies that a DPD thermostat and a Lennard-Jones pair potential
    should be applied between every non-excluded particle pair in the
    simulation.

    `C. L. Phillips et. al. 2011 <http://dx.doi.org/10.1016/j.jcp.2011.05.021>`_
    describes the DPD implementation details in HOOMD-blue. Cite it if you
    utilize the DPD functionality in your work.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        F = F_{\mathrm{C}}(r) + F_{\mathrm{R,ij}}(r_{ij}) +
            F_{\mathrm{D,ij}}(v_{ij}) \\
        \end{eqnarray*}

    .. math::
        :nowrap:

        \begin{eqnarray*}
        F_{\mathrm{C}}(r) = & \partial V_{\mathrm{LJ}} / \partial r \\
        F_{\mathrm{R, ij}}(r_{ij}) = & - \theta_{ij}\sqrt{3}
            \sqrt{\frac{2k_b\gamma T}{\Delta t}}\cdot w(r_{ij})  \\
        F_{\mathrm{D, ij}}(r_{ij}) = & - \gamma w^2(r_{ij})
            \left( \hat r_{ij} \circ v_{ij} \right)  \\
        \end{eqnarray*}

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{LJ}}(r) = & 4 \varepsilon \left[ \left(
            \frac{\sigma}{r} \right)^{12} -
             \left( \frac{\sigma}{r} \right)^{6} \right];
            & r < r_{\mathrm{cut}} \\
                            = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    .. math::
        :nowrap:

        \begin{eqnarray*}
        w(r_{ij}) = &\left( 1 - r/r_{\mathrm{cut}} \right);
            & r < r_{\mathrm{cut}} \\
                  = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    where :math:`\hat r_{ij}` is a normalized vector from particle i to
    particle j, :math:`v_{ij} = v_i - v_j`, and :math:`\theta_{ij}` is a
    uniformly distributed random number in the range [-1, 1].

    To use the DPD thermostat, an `hoomd.md.methods.NVE` integrator
    must be applied to the system and the user must specify a temperature. Use
    of the dpd thermostat pair force with other integrators will result in
    unphysical behavior.

    See `Pair` for details on how forces are calculated and the
    available energy shifting and smoothing modes.

    .. py:attribute:: params

        The DPDLJ potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) - :math:`\varepsilon`
          :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) - :math:`\sigma`
          :math:`[\mathrm{length}]`
        * ``gamma`` (`float`, **required**) - :math:`\gamma`
          :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        dpdlj = pair.DPDLJ(nlist=nl, kT=1.0, default_r_cut=2.5)
        dpdlj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0, gamma=4.5)
        dpdlj.params[(['A', 'B'], ['C', 'D'])] = dict(
            epsilon=3.0, sigma=1.0, gamma=1.2)
        dpdlj.r_cut[('B', 'B')] = 2.0**(1.0/6.0)
    """
    _cpp_class_name = "PotentialPairDPDLJThermoDPD"
    _accepted_modes = ("none", "shifted")

    def __init__(self, nlist, kT, default_r_cut=None, mode='none'):

        super().__init__(nlist=nlist,
                         default_r_cut=default_r_cut,
                         default_r_on=0,
                         mode=mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float,
                              sigma=float,
                              gamma=float,
                              len_keys=2))
        self._add_typeparam(params)

        d = ParameterDict(kT=hoomd.variant.Variant)
        self._param_dict.update(d)

        self.kT = kT

    def _add(self, simulation):
        """Add the operation to a simulation.

        DPDLJ uses RNGs. Warn the user if they did not set the seed.
        """
        if isinstance(simulation, hoomd.Simulation):
            simulation._warn_if_seed_unset()

        super()._add(simulation)


class ForceShiftedLJ(Pair):
    r"""Force-shifted Lennard-Jones pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `ForceShiftedLJ` specifies that a modified Lennard-Jones pair force should
    be applied between non-excluded particle pair in the simulation. The force
    differs from the one calculated by  `LJ` by the subtraction of the
    value of the force at :math:`r_{\mathrm{cut}}`, such that the force
    smoothly goes to zero at the cut-off. The potential is modified by a linear
    function. This potential can be used as a substitute for `LJ`,
    when the exact analytical form of the latter is not required but a smaller
    cut-off radius is desired for computational efficiency. See `Toxvaerd et.
    al. 2011 <http://dx.doi.org/10.1063/1.3558787>`_ for a discussion of this
    potential.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r) = & 4 \varepsilon \left[ \left( \frac{\sigma}{r}
          \right)^{12} - \left( \frac{\sigma}{r} \right)^{6}
          \right] + \Delta V(r); & r < r_{\mathrm{cut}}\\
             = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    .. math::

        \Delta V(r) = -(r - r_{\mathrm{cut}}) \frac{\partial
          V_{\mathrm{LJ}}}{\partial r}(r_{\mathrm{cut}})

    See `Pair` for details on how forces are calculated.

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) - :math:`\varepsilon`
          :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) - :math:`\sigma`
          :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        fslj = pair.ForceShiftedLJ(nlist=nl, default_r_cut=1.5)
        fslj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
    """
    _cpp_class_name = "PotentialPairForceShiftedLJ"
    _accepted_modes = ("none",)

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist=nlist,
                         default_r_cut=default_r_cut,
                         default_r_on=0,
                         mode='none')

        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, len_keys=2))
        self._add_typeparam(params)


class Moliere(Pair):
    r"""Moliere pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Moliere` specifies that a Moliere type pair potential should be applied
    between every non-excluded particle pair in the simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{Moliere}}(r)
          = & \frac{Z_i Z_j e^2}{4 \pi \epsilon_0 r_{ij}} \left[ 0.35 \exp
          \left( -0.3 \frac{r_{ij}}{a_F} \right) + \\
          0.55 \exp \left( -1.2 \frac{r_{ij}}{a_F} \right) + 0.10 \exp
          \left( -6.0 \frac{r_{ij}}{a_F} \right) \right];
          & r < r_{\mathrm{cut}} \\
          = & 0; & r > r_{\mathrm{cut}} \\
        \end{eqnarray*}

    Where each parameter is defined as:

    - :math:`Z_i` - *Z_i* - Atomic number of species i
      :math:`[\mathrm{dimensionless}]`
    - :math:`Z_j` - *Z_j* - Atomic number of species j
      :math:`[\mathrm{dimensionless}]`
    - :math:`e` - *elementary_charge* - The elementary charge
      :math:`[\mathrm{charge}]`
    - :math:`a_F = \frac{0.8853 a_0}{\left( \sqrt{Z_i} + \sqrt{Z_j}
      \right)^{2/3}}`, where :math:`a_0` is the Bohr radius
      :math:`[\mathrm{length}]`

    See `Pair` for details on how forces are calculated and the available
    energy shifting and smoothing modes.

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``qi`` (`float`, **required**) -
          :math:`q_i = Z_i \frac{e}{\sqrt{4 \pi \epsilon_0}}`
          :math:`[\mathrm{charge}]`
        * ``qj`` (`float`, **required**) -
          :math:`q_j = Z_j \frac{e}{\sqrt{4 \pi \epsilon_0}}`
          :math:`[\mathrm{charge}]`

        * ``aF`` (`float`, **required**) -
          :math:`a_F = \frac{0.8853 a_0}{\left( \sqrt{Z_i} + \sqrt{Z_j}
          \right)^{2/3}}` :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        moliere = pair.Moliere(default_r_cut = 3.0, nlist=nl)

        Zi = 54
        Zj = 7
        e = 1
        a0 = 1
        aF = 0.8853 * a0 / (np.sqrt(Zi) + np.sqrt(Zj))**(2/3)

        moliere.params[('A', 'B')] = dict(qi=Zi*e, qj=Zj*e, aF=aF)
    """
    _cpp_class_name = "PotentialPairMoliere"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(qi=float, qj=float, aF=float, len_keys=2))
        self._add_typeparam(params)


class ZBL(Pair):
    r"""ZBL pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.

    `ZBL` specifies that a Ziegler-Biersack-Littmark pair potential
    should be applied between every non-excluded particle pair in the
    simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{ZBL}}(r) =
          & \frac{Z_i Z_j e^2}{4 \pi \epsilon_0 r_{ij}} \left[ 0.1818
          \exp \left( -3.2 \frac{r_{ij}}{a_F} \right) \\
          + 0.5099 \exp \left( -0.9423 \frac{r_{ij}}{a_F} \right) \\
          + 0.2802 \exp \left( -0.4029 \frac{r_{ij}}{a_F} \right) \\
          + 0.02817 \exp \left( -0.2016 \frac{r_{ij}}{a_F} \right) \right];
          & r < r_{\mathrm{cut}} \\
          = & 0; & r > r_{\mathrm{cut}} \\
        \end{eqnarray*}

    Where each parameter is defined as:

    - :math:`Z_i` - *Z_i* - Atomic number of species i
      :math:`[\mathrm{dimensionless}]`
    - :math:`Z_j` - *Z_j* - Atomic number of species j
      :math:`[\mathrm{dimensionless}]`
    - :math:`e` - *elementary_charge* - The elementary charge
      :math:`[\mathrm{charge}]`
    - :math:`a_F = \frac{0.8853 a_0}{ Z_i^{0.23} + Z_j^{0.23} }`, where
      :math:`a_0` is the Bohr radius :math:`[\mathrm{length}]`

    See `Pair` for details on how forces are calculated. Note ZBL does not
    support energy shifting or smoothing.

    .. py:attribute:: params

        The ZBL potential parameters. The dictionary has the following keys:

        * ``q_i`` (`float`, **required**) - :math:`q_i=Z_i \frac{e}{\sqrt{4
          \pi \epsilon_0}}` :math:`[\mathrm{charge}]`
        * ``q_j`` (`float`, **required**) - :math:`q_j=Z_j \frac{e}{\sqrt{4
          \pi \epsilon_0}}` :math:`[\mathrm{charge}]`
        * ``a_F`` (`float`, **required**) -
          :math:`a_F = \frac{0.8853 a_0}{ Z_i^{0.23} + Z_j^{0.23} }`
          :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        zbl = pair.ZBL(default_r_cut=3.0, nlist=nl)

        Zi = 54
        Zj = 7
        e = 1
        a0 = 1
        aF = 0.8853 * a0 / (Zi**(0.23) + Zj**(0.23))

        zbl.params[('A', 'B')] = dict(qi=Zi*e, qj=Zj*e, aF=aF)
    """
    _cpp_class_name = "PotentialPairZBL"
    _accepted_modes = ("none",)

    def __init__(self, nlist, default_r_cut=None, default_r_on=0.):

        super().__init__(nlist, default_r_cut, default_r_on, 'none')
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(qi=float, qj=float, aF=float, len_keys=2))
        self._add_typeparam(params)


class Mie(Pair):
    r"""Mie pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Mie` specifies that a Mie pair potential should be applied between every
    non-excluded particle pair in the simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{mie}}(r)
          = & \left( \frac{n}{n-m} \right) {\left( \frac{n}{m}
          \right)}^{\frac{m}{n-m}} \varepsilon \left[ \left(
          \frac{\sigma}{r} \right)^{n} - \left( \frac{\sigma}{r}
          \right)^{m} \right]; & r < r_{\mathrm{cut}} \\
          = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    See `Pair` for details on how forces are calculated and the available
    energy shifting and smoothing modes.

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) - :math:`\varepsilon`
          :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) - :math:`\sigma`
          :math:`[\mathrm{length}]`
        * ``n`` (`float`, **required**) - :math:`n`
          :math:`[\mathrm{dimensionless}]`
        * ``m`` (`float`, **required**) - :math:`m`
          :math:`[\mathrm{dimensionless}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        mie = pair.Mie(nlist=nl, default_r_cut=3.0)
        mie.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0, n=12, m=6)
        mie.r_cut[('A', 'A')] = 2**(1.0/6.0)
        mie.r_on[('A', 'A')] = 2.0
        mie.params[(['A', 'B'], ['C', 'D'])] = dict(epsilon=1.5, sigma=2.0)
    """
    _cpp_class_name = "PotentialPairMie"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):

        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float,
                              sigma=float,
                              n=float,
                              m=float,
                              len_keys=2))

        self._add_typeparam(params)


class ExpandedMie(Pair):
    """Expanded Mie pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius (in distance units).
        default_r_on (float): Default turn-on radius (in distance units).
        mode (str): Energy shifting/smoothing mode.

    `ExpandedMie` specifies that a radially shifted Mie pair potential should be
    applied between every non-excluded particle pair in the simulation.

    .. math::
        :nowrap:

        \\begin{eqnarray*}
        V_{\\mathrm{mie}}(r)
          = & \\left( \\frac{n}{n-m} \\right) {\\left( \\frac{n}{m}
          \\right)}^{\\frac{m}{n-m}} \\varepsilon \\left[ \\left(
          \\frac{\\sigma}{r-\\Delta} \\right)^{n} - \\left( \\frac
          {\\sigma}{r-\\Delta}
          \\right)^{m} \\right]; & r < r_{\\mathrm{cut}} \\\\
          = & 0; & r \\ge r_{\\mathrm{cut}} \\\\
        \\end{eqnarray*}

    See `Pair` for details on how forces are calculated and the available energy
    shifting and smoothing modes.

    .. py:attribute:: params

        The Expanded Mie potential parameters.
        The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          :math:`\\epsilon` :math:`[\\mathrm{energy}]`.
        * ``sigma`` (`float`, **required**) -
          :math:`\\sigma` :math:`[\\mathrm{length}]`.
        * ``n`` (`float`, **required**) -
          :math:`n` :math:`[\\mathrm{dimensionless}]`.
        * ``m`` (`float`, **required**) -
          :math:`m` :math:`[\\mathrm{dimensionless}]`.
        * ``delta`` (`float`, **required**) -
          :math:`\\Delta` :math:`[\\mathrm{length}]`.

        Type: `TypeParameter` [ `tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        expanded_mie = pair.ExpandedMie(nlist=nl, default_r_cut=3.0)
        mie.params[('A', 'B')] = {
            "epsilon": 1.0, "sigma": 1.0, "n": 12, "m": 6,
            "delta": 0.5}
        expanded_mie.r_cut[('A', 'B')] = 2**(1.0 / 6.0)
        expanded_mie.params[(['A', 'B'], ['C', 'D'])] = {
            "epsilon": 1.5, "sigma": 2.0, "n": 12, "m": 6,
            "delta": 0.5}
    """
    _cpp_class_name = "PotentialPairExpandedMie"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):

        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float,
                              sigma=float,
                              n=float,
                              m=float,
                              delta=float,
                              len_keys=2))

        self._add_typeparam(params)


class ReactionField(Pair):
    r"""Onsager reaction field pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `ReactionField` specifies that an Onsager reaction field pair potential
    should be applied between every non-excluded particle pair in the
    simulation.

    Reaction field electrostatics is an approximation to the screened
    electrostatic interaction, which assumes that the medium can be treated as
    an electrostatic continuum of dielectric constant :math:`\epsilon_{RF}`
    outside the cutoff sphere of radius :math:`r_{\mathrm{cut}}`. See: `Barker
    et. al. 1973 <http://dx.doi.org/10.1080/00268977300102101>`_.

    .. math::

       V_{\mathrm{RF}}(r) = \varepsilon \left[ \frac{1}{r} +
           \frac{(\epsilon_{RF}-1) r^2}{(2 \epsilon_{RF} + 1) r_c^3} \right]

    By default, the reaction field potential does not require charge or diameter
    to be set. Two parameters, :math:`\varepsilon` and :math:`\epsilon_{RF}`
    are needed. If :math:`\epsilon_{RF}` is specified as zero, it will
    represent infinity.

    If *use_charge* is set to True, the following formula is evaluated instead:

    .. math::

        V_{\mathrm{RF}}(r) = q_i q_j \varepsilon \left[ \frac{1}{r} +
          \frac{(\epsilon_{RF}-1) r^2}{(2 \epsilon_{RF} + 1) r_c^3} \right]

    where :math:`q_i` and :math:`q_j` are the charges of the particle pair.

    See `Pair` for details on how forces are calculated and the available
    energy shifting and smoothing modes.

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) - :math:`\varepsilon`
          :math:`[\mathrm{energy} \cdot \mathrm{length}]`
        * ``eps_rf`` (`float`, **required**) - :math:`\epsilon_{RF}`
          :math:`[\mathrm{dimensionless}]`
        * ``use_charge`` (`boolean`, **optional**) - evaluate pair potential
          using particle charges (*default*: False)

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        reaction_field = pair.reaction_field(nl, default_r_cut=3.0)
        reaction_field.params[('A', 'B')] = dict(epsilon=1.0, eps_rf=1.0)
        reaction_field.params[('B', 'B')] = dict(
            epsilon=1.0, eps_rf=0.0, use_charge=True)
    """
    _cpp_class_name = "PotentialPairReactionField"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float,
                              eps_rf=float,
                              use_charge=False,
                              len_keys=2))

        self._add_typeparam(params)


class DLVO(Pair):
    r"""DLVO colloidal interaction.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        name (str): Name of the force instance.
        mode (str): Energy shifting mode.

    `DLVO` computes the energy and force from the DLVO dispersion and
    electrostatic interaction between every non-excluded pair of particles.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{DLVO}}(r)  = & - \frac{A}{6} \left[
            \frac{2a_1a_2}{r^2 - (a_1+a_2)^2} +
            \frac{2a_1a_2}{r^2 - (a_1-a_2)^2} \\
            + \log \left(
            \frac{r^2 - (a_1+a_2)^2}{r^2 - (a_1-a_2)^2} \right) \right]
            & \\
            & + \frac{a_1 a_2}{a_1+a_2} Z e^{-\kappa(r - (a_1+a_2))};
            & r < r_{\mathrm{cut}} \\
            = & 0; & r \ge r_{\mathrm{cut}}
        \end{eqnarray*}

    where :math:`a_1` is the radius of first particle in the pair, :math:`a_2`
    is the radius of second particle in the pair, :math:`A` is the Hamaker
    constant, :math:`Z` is proportional to the surface electric potential,
    and :math:`\kappa` is the screening parameter.

    The first term corresponds to the attractive van der Waals interaction with
    and the second term to the repulsive double-layer interaction between two
    spherical surfaces. See "Intermolecular and Surface Forces" Israelachvili
    2011, pp. 317.

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``A`` (`float`, **required**) - Hamaker constant :math:`A`
          :math:`[\mathrm{energy}]`
        * ``a1`` (`float`, **required**) - Radius of first particle :math:`a_1`
          :math:`[\mathrm{length}]`
        * ``a2`` (`float`, **required**) - Radius of second particle :math:`a_2`
          :math:`[\mathrm{length}]`
        * ``kappa`` (`float`, **required**) - screening parameter
          :math:`\kappa` :math:`[\mathrm{length}^{-1}]`
        * ``Z`` surface electric potential (`float`, **required**) - :math:`Z`
          :math:`[\mathrm{energy} \cdot \mathrm{length}^{-1}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = hoomd.md.nlist.Cell()
        dlvo = hoomd.md.pair.DLVO(nlist=nl)
        dlvo.params[('A', 'A')] = dict(A=1.0, kappa=1.0, Z=2, a1=1, a2=1)
        dlvo.params[('A', 'B')] = dict(A=2.0, kappa=0.5, Z=3, a1=1, a2=3)
        dlvo.params[('B', 'B')] = dict(A=2.0, kappa=0.5, Z=3, a1=3, a2=3)
    """
    _cpp_class_name = "PotentialPairDLVO"
    _accepted_modes = ("none", "shift")

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        if mode == 'xplor':
            raise ValueError("xplor is not a valid mode for the DLVO potential")

        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(
                kappa=float,
                Z=float,
                A=float,
                a1=float,
                a2=float,
                len_keys=2,
            ))
        self._add_typeparam(params)

        # this potential needs diameter shifting on
        self._nlist.diameter_shift = True


class Buckingham(Pair):
    r"""Buckingham pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Buckingham` specifies that a Buckingham pair potential should be applied
    between every non-excluded particle pair in the simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{Buckingham}}(r) = & A \exp\left(-\frac{r}{\rho}\right)
          - \frac{C}{r^6}; & r < r_{\mathrm{cut}} \\
          = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    See `Pair` for details on how forces are calculated and the available
    energy shifting and smoothing modes.

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``A`` (`float`, **required**) - :math:`A` :math:`[\mathrm{energy}]`
        * ``rho`` (`float`, **required**) - :math:`\rho`
          :math:`[\mathrm{length}]`
        * ``C`` (`float`, **required**) - :math:`C` :math:`[\mathrm{energy}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        buck = pair.Buckingham(nl, default_r_cut=3.0)
        buck.params[('A', 'A')] = {'A': 2.0, 'rho'=0.5, 'C': 1.0}
        buck.params[('A', 'B')] = dict(A=1.0, rho=1.0, C=1.0)
        buck.params[('B', 'B')] = dict(A=2.0, rho=2.0, C=2.0)
    """

    _cpp_class_name = "PotentialPairBuckingham"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(A=float, rho=float, C=float, len_keys=2))
        self._add_typeparam(params)


class LJ1208(Pair):
    r"""Lennard-Jones 12-8 pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `LJ1208` specifies that a Lennard-Jones 12-8 pair potential should be
    applied between every non-excluded particle pair in the simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{LJ}}(r)
          = & 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
          \left( \frac{\sigma}{r} \right)^{8} \right];
          & r < r_{\mathrm{cut}} \\
          = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    See `Pair` for details on how forces are calculated and the available
    energy shifting and smoothing modes.

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) - energy parameter
          :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) - particle size :math:`\sigma`
          :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        lj1208 = pair.LJ1208(nl, default_r_cut=3.0)
        lj1208.params[('A', 'A')] = {'sigma': 1.0, 'epsilon': 1.0}
        lj1208.params[('A', 'B')] = dict(epsilon=2.0, sigma=1.0)
    """
    _cpp_class_name = "PotentialPairLJ1208"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, len_keys=2))
        self._add_typeparam(params)


class LJ0804(Pair):
    r"""Lennard-Jones 8-4 pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `LJ0804` specifies that a Lennard-Jones 8-4 pair potential should be
    applied between every non-excluded particle pair in the simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{LJ}}(r)
          = & 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{8} -
          \left( \frac{\sigma}{r} \right)^{4} \right];
          & r < r_{\mathrm{cut}} \\
          = & 0: & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    See `Pair` for details on how forces are calculated and the available
    energy shifting and smoothing modes.

    .. py:attribute:: params

        The LJ potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        lj0804 = pair.LJ0804(nl, default_r_cut=3.0)
        lj0804.params[('A', 'A')] = {'sigma': 1.0, 'epsilon': 1.0}
        lj0804.params[('A', 'B')] = dict(epsilon=2.0, sigma=1.0)
        lj0804.r_cut[('A', 'B')] = 3.0
    """
    _cpp_class_name = "PotentialPairLJ0804"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, len_keys=2))
        self._add_typeparam(params)


class Fourier(Pair):
    r"""Fourier pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Fourier` specifies that a Fourier pair potential should be applied between
    every non-excluded particle pair in the simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{Fourier}}(r)
          = & \frac{1}{r^{12}} + \frac{1}{r^2}\sum_{n=1}^4
          [a_n cos(\frac{n \pi r}{r_{cut}}) +
          b_n sin(\frac{n \pi r}{r_{cut}})];
          & r < r_{\mathrm{cut}}  \\
          = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

        where:
        \begin{eqnarray*}
        a_1 = \sum_{n=2}^4 (-1)^n a_n
        \end{eqnarray*}

        \begin{eqnarray*}
        b_1 = \sum_{n=2}^4 n (-1)^n b_n
        \end{eqnarray*}

        is calculated to enforce close to zero value at r_cut.

    See `Pair` for details on how forces are calculated and the available
    energy shifting and smoothing modes.

    .. py:attribute:: params

        The Fourier potential parameters. The dictionary has the following
        keys:

        * ``a`` (`float`, **required**) - array of 3 values corresponding to
          a2, a3 and a4 in the Fourier series :math:`[\mathrm{dimensionless}]`
        * ``b`` (`float`, **required**) - array of 3 values corresponding to
          b2, b3 and b4 in the Fourier series :math:`[\mathrm{dimensionless}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        fourier = pair.Fourier(default_r_cut=3.0, nlist=nl)
        fourier.params[('A', 'A')] = dict(a=[a2,a3,a4], b=[b2,b3,b4])
    """
    _cpp_class_name = "PotentialPairFourier"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(a=(float, float, float),
                              b=(float, float, float),
                              len_keys=2))
        self._add_typeparam(params)


class OPP(Pair):
    r"""Oscillating pair potential.

    Args:
        nlist (:py:mod:`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `OPP` specifies that an oscillating pair potential should be applied between
    every non-excluded particle pair in the simulation. The OPP potential can
    be used to model metallic interactions.

    .. math::
        :nowrap:

        \begin{equation*}
        V_{\mathrm{OPP}}(r) = C_1 r^{-\eta_1}
            + C_2 r^{-\eta_2} \cos{\left(k r - \phi\right)}
        \end{equation*}

    See `Pair` for details on how forces are calculate. Note OPP does not
    support energy shifting or smoothing.

    The potential comes from `Marek Mihalkovič and C. L. Henley 2012`_.

    .. _Marek Mihalkovič and C. L. Henley 2012:
       https://dx.doi.org/10.1103/PhysRevB.85.092102

    .. py:attribute:: params

        The OPP potential parameters. The dictionary has the following keys:

        * ``C1`` (`float`, **required**) -
          Energy scale of the first term :math:`C_1`
          :math:`[\mathrm{energy}]`
        * ``C2`` (`float`, **required**) -
          Energy scale of the second term :math:`C_2`
          :math:`[\mathrm{energy}]`
        * ``eta1`` (`float`, **required**) -
          The inverse power to take :math:`r` to in the first term,
          :math:`\eta_1` :math:`[\mathrm{dimensionless}]`.
        * ``eta2`` (`float`, **required**) -
          The inverse power to take :math:`r` to in the second term
          :math:`\eta_2` :math:`[\mathrm{dimensionless}]`.
        * ``k`` (`float`, **required**) -
          oscillation frequency :math:`k` :math:`[\mathrm{length}^{-1}]`
        * ``phi`` (`float`, **required**) -
          potential phase shift :math:`\phi` :math:`[\mathrm{dimensionless}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        opp = pair.OPP(nl, default_r_cut=3.0)
        opp.params[('A', 'A')] = {
            'C1': 1., 'C2': 1., 'eta1': 15,
            'eta2': 3, 'k': 1.0, 'phi': 3.14}
        opp.r_cut[('A', 'B')] = 3.0
    """
    _cpp_class_name = "PotentialPairOPP"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(C1=float,
                              C2=float,
                              eta1=float,
                              eta2=float,
                              k=float,
                              phi=float,
                              len_keys=2))
        self._add_typeparam(params)


class TWF(Pair):
    r"""Pair potential model for globular proteins.

    Args:
        nlist (:py:mod:`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius (in distance units).
        default_r_on (float): Default turn-on radius (in distance units).
        mode (str): Energy shifting/smoothing mode.

    This potential was introduced by Ten-wolde and Daan Frenkel in 1997 for
    studying globular protein crystallization. The potential has the following
    form:

    .. math::
        :nowrap:

        \begin{equation}
        V_{\mathrm{TWF}}(r) = \frac{4 \epsilon}{\alpha^2} {\left[
        {\left(\frac{\sigma^2}{r^2} - 1 \right)}^6 -
        \alpha {\left(\frac{\sigma^2}{r^2} - 1 \right)}^3\right]}
        \end{equation}

    See `Pair` for details on how forces are calculated and the available
    energy shifting and smoothing modes.

    The potential comes from `Pieter Rein ten Wolde and Daan Frenkel 1997`_.

    .. _Pieter Rein ten Wolde and Daan Frenkel 1997:
       https://dx.doi.org/10.1126/science.277.5334.1975

    .. py:attribute:: params

        The LJ potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[energy]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[length]`
        * ``alpha`` (`float`, **required**) -
          controls well-width :math:`\alpha` :math:`[dimensionless]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        twf = hoomd.md.pair.TWF(nl, default_r_cut=3.0)
        twf.params[('A', 'A')] = {'sigma': 1.0, 'epsilon': 1.0, 'alpha': 50.0}
        twf.r_cut[('A', 'B')] = 3.0
    """
    _cpp_class_name = "PotentialPairTWF"

    def __init__(self,
                 nlist,
                 default_r_cut=None,
                 default_r_on=0.0,
                 mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float,
                              sigma=float,
                              alpha=float,
                              len_keys=2))
        self._add_typeparam(params)
