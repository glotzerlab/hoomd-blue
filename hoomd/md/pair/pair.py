# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Pair forces."""

import copy
import warnings

import hoomd
from hoomd.md import _md
from hoomd.md import force
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
import numpy as np
from hoomd.data.typeconverter import OnlyFrom, nonnegative_real, positive_real


class Pair(force.Force):
    r"""Base class pair force.

    `Pair` is the base class for all pair forces.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.

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

    .. py:attribute:: nlist

        Neighbor list used to compute the pair force.

        Type: `hoomd.md.nlist.NeighborList`
    """

    # The accepted modes for the potential. Should be reset by subclasses with
    # restricted modes.
    _accepted_modes = ("none", "shift", "xplor")

    # Module where the C++ class is defined. Reassign this when developing an
    # external plugin.
    _ext_module = _md

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__()
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
            ParameterDict(mode=OnlyFrom(self._accepted_modes),
                          nlist=hoomd.md.nlist.NeighborList))
        self.mode = mode
        self.nlist = nlist

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

    def _attach_hook(self):
        if self.nlist._attached and self._simulation != self.nlist._simulation:
            warnings.warn(
                f"{self} object is creating a new equivalent neighbor list."
                f" This is happending since the force is moving to a new "
                f"simulation. Set a new nlist to suppress this warning.",
                RuntimeWarning)
            self.nlist = copy.deepcopy(self.nlist)
        self.nlist._attach(self._simulation)
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(self._ext_module, self._cpp_class_name)
            self.nlist._cpp_obj.setStorageMode(
                _md.NeighborList.storageMode.half)
        else:
            cls = getattr(self._ext_module, self._cpp_class_name + "GPU")
            self.nlist._cpp_obj.setStorageMode(
                _md.NeighborList.storageMode.full)
        self._cpp_obj = cls(self._simulation.state._cpp_sys_def,
                            self.nlist._cpp_obj)

    def _detach_hook(self):
        self.nlist._detach()

    def _setattr_param(self, attr, value):
        if attr == "nlist":
            self._nlist_setter(value)
            return
        super()._setattr_param(attr, value)

    def _nlist_setter(self, new_nlist):
        if new_nlist is self.nlist:
            return
        if self._attached:
            raise RuntimeError("nlist cannot be set after scheduling.")
        self._param_dict._dict["nlist"] = new_nlist


class LJ(Pair):
    r"""Lennard-Jones pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.
        tail_correction (bool): Whether to apply the isotropic integrated long
            range tail correction.

    `LJ` computes the Lennard-Jones pair force on every particle in the
    simulation state.

    .. math::
        U(r) = 4 \varepsilon \left[ \left(
        \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r}
        \right)^{6} \right]

    Example::

        nl = nlist.Cell()
        lj = pair.LJ(nl, default_r_cut=3.0)
        lj.params[('A', 'A')] = {'sigma': 1.0, 'epsilon': 1.0}
        lj.r_cut[('A', 'B')] = 3.0

    .. py:attribute:: params

        The LJ potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`

    .. py:attribute:: tail_correction

        Whether to apply the isotropic integrated long range tail correction.

        Type: `bool`
    """
    _cpp_class_name = "PotentialPairLJ"

    def __init__(self,
                 nlist,
                 default_r_cut=None,
                 default_r_on=0.,
                 mode='none',
                 tail_correction=False):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, len_keys=2))
        self._add_typeparam(params)
        self._param_dict.update(
            ParameterDict(tail_correction=bool(tail_correction)))


class Gaussian(Pair):
    r"""Gaussian pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Gaussian` computes the Gaussian pair force should on every particle in the
    simulation state:

    .. math::
        U(r) = \varepsilon \exp \left( -\frac{1}{2}
               \left( \frac{r}{\sigma} \right)^2 \right)

    Example::

        nl = nlist.Cell()
        gauss = pair.Gaussian(default_r_cut=3.0, nlist=nl)
        gauss.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
        gauss.r_cut[('A', 'B')] = 3.0

    .. py:attribute:: params

        The Gauss potential parameters. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) - energy parameter
          :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float` > 0, **required**) - particle size :math:`\sigma`
          :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
    """
    _cpp_class_name = "PotentialPairGauss"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=positive_real, len_keys=2))
        self._add_typeparam(params)


class ExpandedGaussian(Pair):
    r"""Expanded Gaussian pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `ExpandedGaussian` computes the radially-shifted Gaussian pair force should
    on every particle in the simulation state:

    .. math::
        U(r) = \varepsilon \exp \left( -\frac{1}{2}
               \left( \frac{r-\Delta}{\sigma} \right)^2 \right)

    Example::

        nl = nlist.Cell()
        expanded_gauss = pair.ExpandedGaussian(default_r_cut=3.0, nlist=nl)
        expanded_gauss.params[('A', 'A')] = dict(epsilon=1.0,
        sigma=1.0, delta=0.5)
        expanded_gauss.r_cut[('A', 'B')] = 3.0

    .. py:attribute:: params

        The expanded Gaussian potential parameters. The dictionary has the
        following keys:

        * ``epsilon`` (`float`, **required**) - energy parameter
          :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float` > 0, **required**) - particle size
          :math:`\sigma` :math:`[\mathrm{length}]`
        * ``delta`` (`float`, **required**) - shift distance
          :math:`\delta` :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
    """
    _cpp_class_name = "PotentialPairExpandedGaussian"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float,
                              sigma=positive_real,
                              delta=float,
                              len_keys=2))
        self._add_typeparam(params)


class ExpandedLJ(Pair):
    r"""Expanded Lennard-Jones pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting mode.

    `ExpandedLJ` computes the radially-shifted Lennard-Jones pair force on every
    particle in the simulation state:

    .. math::
        U(r) = 4 \varepsilon \left[ \left(
               \frac{\sigma}{r - \Delta} \right)^{12} -
               \left( \frac{\sigma}{r - \Delta}
               \right)^{6} \right]


    Example::

        nl = nlist.Cell()
        expanded_lj = pair.ExpandedLJ(default_r_cut=3.0, nlist=nl)
        expanded_lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0, delta=1.0)
        expanded_lj.params[('A', 'B')] = dict(
                                             epsilon=2.0,
                                             sigma=1.0,
                                             delta=0.75)
        expanded_lj.params[('B', 'B')] = dict(epsilon=1.0, sigma=1.0, delta=0.5)

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

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
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
    r"""Yukawa pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Yukawa` computes the Yukawa pair force son every particle in the simulation
    state:

    .. math::

        U(r) = \varepsilon \frac{ \exp \left(
               -\kappa r \right) }{r}

    Example::

        nl = nlist.Cell()
        yukawa = pair.Yukawa(default_r_cut=3.0, nlist=nl)
        yukawa.params[('A', 'A')] = dict(epsilon=1.0, kappa=1.0)
        yukawa.r_cut[('A', 'B')] = 3.0

    .. py:attribute:: params

        The Yukawa potential parameters. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) - energy parameter
          :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``kappa`` (`float`, **required**) - scaling parameter
          :math:`\kappa` :math:`[\mathrm{length}^{-1}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
    """
    _cpp_class_name = "PotentialPairYukawa"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(kappa=float, epsilon=float, len_keys=2))
        self._add_typeparam(params)


class Ewald(Pair):
    r"""Ewald pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    `Ewald` computes the Ewald pair force on every particle in the simulation
    state:

    .. math::

        U(r) = q_i q_j \left[\mathrm{erfc}\left(\kappa
               r + \frac{\alpha}{2\kappa}\right)
               \exp(\alpha r)
               + \mathrm{erfc}\left(\kappa r -
               \frac{\alpha}{2 \kappa}\right)
               \exp(-\alpha r)\right]

    Call `md.long_range.pppm.make_pppm_coulomb_forces` to create an instance
    of `Ewald` and `md.long_range.pppm.Coulomb` that together implement the PPPM
    method for electrostatics.

    Example::

        nl = nlist.Cell()
        ewald = pair.Ewald(default_r_cut=3.0, nlist=nl)
        ewald.params[('A', 'A')] = dict(kappa=1.0, alpha=1.5)
        ewald.r_cut[('A', 'B')] = 3.0

    .. py:attribute:: params

        The Ewald potential parameters. The dictionary has the following keys:

        * ``kappa`` (`float`, **required**) - Splitting parameter
          :math:`\kappa` :math:`[\mathrm{length}^{-1}]`
        * ``alpha`` (`float`, **required**) - Debye screening length
          :math:`\alpha` :math:`[\mathrm{length}^{-1}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``.

        Type: `str`
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
    """Tabulated pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\\mathrm{length}]`.

    `Table` computes the tabulated pair force on every particle in the
    simulation state.

    The force :math:`\\vec{F}` is:

    .. math::

        \\vec{F}(\\vec{r}) =
        \\begin{cases}
        0 & r < r_{\\mathrm{min}} \\\\
        F_\\mathrm{table}(r)\\hat{r} &
        r_{\\mathrm{min}} \\le r < r_{\\mathrm{cut}} \\\\
        0 & r \\ge r_{\\mathrm{cut}} \\\\
        \\end{cases}

    and the potential :math:`U(r)` is:

    .. math::

        U(r) =
        \\begin{cases}
        0 & r < r_{\\mathrm{min}} \\\\
        U_\\mathrm{table}(r)
        & r_{\\mathrm{min}} \\le r < r_{\\mathrm{cut}} \\\\
        0 & r \\ge r_{\\mathrm{cut}} \\\\
        \\end{cases}

    where :math:`\\vec{r}` is the vector pointing from one particle to the other
    in the pair, ``r_min`` is defined in `params`, and ``r_cut`` is defined in
    `Pair.r_cut`.

    Provide :math:`F_\\mathrm{table}(r)` and :math:`U_\\mathrm{table}(r)` on
    evenly spaced grid points points between :math:`r_{\\mathrm{min}}` and
    :math:`r_{\\mathrm{cut}}`. `Table` linearly interpolates values when
    :math:`r` lies between grid points and between the last grid point and
    :math:`r=r_{\\mathrm{cut}}`.  The force must be specificed commensurate with
    the potential: :math:`F = -\\frac{\\partial U}{\\partial r}`.

    `Table` does not support energy shifting or smoothing modes.

    Note:
        For potentials that diverge near r=0, to set *r_min* to a non-zero
        value.

    Note:
        The implicitly defined :math:`r` values are those that would be returned
        by ``numpy.linspace(r_min, r_cut, len(U), endpoint=False)``.

    Tip:
        Define non-interacting potentials with::

            table.params[(type1, type2)] = dict(r_min=0, U=[0], F=[0])
            table.r_cut[(type1, type2)] = 0

        There must be at least one element in U and F, and the ``r_cut`` value
        of 0 disables the interaction entirely.

    Attributes:
        params (`TypeParameter` [\
          `tuple` [``particle_type``, ``particle_type``],\
          `dict`]):
          The potential parameters. The dictionary has the following keys:

          * ``r_min`` (`float`, **required**) - the minimum distance to apply
            the tabulated potential, corresponding to the first element of the
            energy and force arrays :math:`[\\mathrm{length}]`.

          * ``U`` ((*N*,) `numpy.ndarray` of `float`, **required**) -
            the tabulated energy values :math:`[\\mathrm{energy}]`.

          * ``F`` ((*N*,) `numpy.ndarray` of `float`, **required**) -
            the tabulated force values :math:`[\\mathrm{force}]`. Must have the
            same length as ``U``.

        mode (str): Energy shifting/smoothing mode: ``"none"``.
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
                U=hoomd.data.typeconverter.NDArrayValidator(np.float64),
                F=hoomd.data.typeconverter.NDArrayValidator(np.float64),
                len_keys=2))
        self._add_typeparam(params)


class Morse(Pair):
    r"""Morse pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Morse` computes the Morse pair force on every particle in the simulation
    state:

    .. math::
        U(r) = D_0 \left[ \exp \left(-2\alpha\left(
            r-r_0\right)\right) -2\exp \left(-\alpha\left(r-r_0\right)
            \right) \right]

    Example::

        nl = nlist.Cell()
        morse = pair.Morse(default_r_cut=3.0, nlist=nl)
        morse.params[('A', 'A')] = dict(D0=1.0, alpha=3.0, r0=1.0)
        morse.r_cut[('A', 'B')] = 3.0

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

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
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
        nlist (hoomd.md.nlist.NeighborList): Neighbor list
        kT (`hoomd.variant` or `float`): Temperature of
            thermostat :math:`[\mathrm{energy}]`.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    `DPD` computes the DPD pair force on every particle in the simulation state.
    DPD includes a an interaction potential, pairwise drag force, and pairwise
    random force. See `Groot and Warren 1997
    <http://dx.doi.org/10.1063/1.474784>`_:

    .. math::

        F = F_{\mathrm{C}}(r) + F_{\mathrm{R,ij}}(r_{ij}) +
        F_{\mathrm{D,ij}}(v_{ij})

    where

    .. math::
        F_{\mathrm{C}}(r) &= A \cdot  w(r_{ij}), \\
        F_{\mathrm{R, ij}}(r_{ij}) &= - \theta_{ij}\sqrt{3}
        \sqrt{\frac{2k_b\gamma T}{\Delta t}}\cdot w(r_{ij}),  \\
        F_{\mathrm{D, ij}}(r_{ij}) &= - \gamma w^2(r_{ij})\left(
        \hat r_{ij} \circ v_{ij} \right), \\
        w(r_{ij}) &=
        \begin{cases}
        \left( 1 - r/r_{\mathrm{cut}} \right)
        & r < r_{\mathrm{cut}} \\
        0 & r \ge r_{\mathrm{cut}} \\
        \end{cases},


    :math:`\hat r_{ij}` is a normalized vector from particle i to
    particle j, :math:`v_{ij} = v_i - v_j`, and :math:`\theta_{ij}` is a
    uniformly distributed random number in the range :math:`[-1, 1]`.

    `C. L. Phillips et. al. 2011 <http://dx.doi.org/10.1016/j.jcp.2011.05.021>`_
    describes the DPD implementation details. Cite it if you utilize the DPD
    functionality in your work.

    `DPD` does not implement any energy shift / smoothing modes due to the
    function of the force.

    To use the DPD thermostat, apply the `hoomd.md.methods.ConstantVolume` or
    `hoomd.md.methods.ConstantPressure` integration method without thermostats
    along with `DPD` forces.  Use of the DPD thermostat pair force with
    other integrators will result in nonphysical behavior. To use `DPD` with a
    different conservative potential than :math:`F_C`, set A to zero and define
    the conservative pair force separately.

    Example::

        nl = nlist.Cell()
        dpd = pair.DPD(nlist=nl, kT=1.0, default_r_cut=1.0)
        dpd.params[('A', 'A')] = dict(A=25.0, gamma=4.5)
        dpd.params[('A', 'B')] = dict(A=40.0, gamma=4.5)
        dpd.params[('B', 'B')] = dict(A=25.0, gamma=4.5)
        dpd.params[(['A', 'B'], ['C', 'D'])] = dict(A=40.0, gamma=4.5)

    .. py:attribute:: params

        The force parameters. The dictionary has the following keys:

        * ``A`` (`float`, **required**) - :math:`A` :math:`[\mathrm{force}]`
        * ``gamma`` (`float`, **required**) - :math:`\gamma`
          :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``.

        Type: `str`
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

    def _attach_hook(self):
        """DPD uses RNGs. Warn the user if they did not set the seed."""
        self._simulation._warn_if_seed_unset()
        super()._attach_hook()


class DPDConservative(Pair):
    r"""DPD Conservative pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    `DPDConservative` computes the conservative part of the `DPD` pair force on
    every particle in the simulation state with:

    .. math::
        U(r) = A \cdot \left( r_{\mathrm{cut}} - r
          \right) - \frac{1}{2} \cdot \frac{A}{r_{\mathrm{cut}}} \cdot
          \left(r_{\mathrm{cut}}^2 - r^2 \right).

    `DPDConservative` does not implement any energy shift / smoothing modes due
    to the function of the force.

    Example::

        nl = nlist.Cell()
        dpdc = pair.DPDConservative(nlist=nl, default_r_cut=3.0)
        dpdc.params[('A', 'A')] = dict(A=1.0)
        dpdc.params[('A', 'B')] = dict(A=2.0, r_cut = 1.0)
        dpdc.params[(['A', 'B'], ['C', 'D'])] = dict(A=3.0)

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``A`` (`float`, **required**) - :math:`A` :math:`[\mathrm{force}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``.

        Type: `str`
    """
    _cpp_class_name = "PotentialPairConservativeDPD"
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
    r"""Dissipative Particle Dynamics with the LJ conservative force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        kT (`hoomd.variant` or `float`): Temperature of
            thermostat :math:`[\mathrm{energy}]`.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting mode.

    `DPDLJ` computes the `DPD` thermostat combined with the `LJ` pair force
    on every particle in the simulation state with:

    .. math::
        F &= F_{\mathrm{C}}(r) + F_{\mathrm{R,ij}}(r_{ij}) +
            F_{\mathrm{D,ij}}(v_{ij}), \\
        F_{\mathrm{C}}(r) &= \partial U / \partial r, \\
        F_{\mathrm{R, ij}}(r_{ij}) &= - \theta_{ij}\sqrt{3}
            \sqrt{\frac{2k_b\gamma T}{\Delta t}}\cdot w(r_{ij}), \\
        F_{\mathrm{D, ij}}(r_{ij}) &= - \gamma w^2(r_{ij})
            \left( \hat r_{ij} \circ v_{ij} \right), \\
        U(r) &= 4 \varepsilon \left[ \left(
               \frac{\sigma}{r} \right)^{12} -
               \left( \frac{\sigma}{r} \right)^{6} \right], \\
        w(r_{ij}) &=
        \begin{cases}
        \left( 1 - r/r_{\mathrm{cut}} \right)
        & r < r_{\mathrm{cut}} \\
        0 & r \ge r_{\mathrm{cut}} \\
        \end{cases},

    :math:`\hat r_{ij}` is a normalized vector from particle i to
    particle j, :math:`v_{ij} = v_i - v_j`, and :math:`\theta_{ij}` is a
    uniformly distributed random number in the range [-1, 1].

    `C. L. Phillips et. al. 2011 <http://dx.doi.org/10.1016/j.jcp.2011.05.021>`_
    describes the DPD implementation details. Cite it if you utilize the DPD
    functionality in your work.

    To use the DPD thermostat, apply the `hoomd.md.methods.ConstantVolume` or
    `hoomd.md.methods.ConstantPressure` integration method  without thermostat
    along with `DPD` forces.  Use of the DPD thermostat pair force with
    other integrators will result in nonphysical behavior.

    Example::

        nl = nlist.Cell()
        dpdlj = pair.DPDLJ(nlist=nl, kT=1.0, default_r_cut=2.5)
        dpdlj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0, gamma=4.5)
        dpdlj.params[(['A', 'B'], ['C', 'D'])] = dict(
            epsilon=3.0, sigma=1.0, gamma=1.2)
        dpdlj.r_cut[('B', 'B')] = 2.0**(1.0/6.0)

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

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"`` or ``"shift"``.

        Type: `str`
    """
    _cpp_class_name = "PotentialPairDPDThermoLJ"
    _accepted_modes = ("none", "shift")

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

    def _attach_hook(self):
        """DPDLJ uses RNGs. Warn the user if they did not set the seed."""
        self._simulation._warn_if_seed_unset()
        super()._attach_hook()


class ForceShiftedLJ(Pair):
    r"""Force-shifted Lennard-Jones pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `ForceShiftedLJ` computes the modified Lennard-Jones pair force on every
    particle in the simulation state.

    .. math::
        U(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r}
          \right)^{12} - \left( \frac{\sigma}{r} \right)^{6}
          \right] + \Delta V(r)

    .. math::

        \Delta V(r) = -(r - r_{\mathrm{cut}}) \frac{\partial
          V_{\mathrm{LJ}}}{\partial r}(r_{\mathrm{cut}})

    The force differs from the one calculated by  `LJ` by the subtraction of the
    value of the force at :math:`r_{\mathrm{cut}}`, such that the force smoothly
    goes to zero at the cut-off. The potential is modified by a linear function.
    See `Toxvaerd et. al. 2011 <http://dx.doi.org/10.1063/1.3558787>`_ for a
    discussion of this potential.

    Example::

        nl = nlist.Cell()
        fslj = pair.ForceShiftedLJ(nlist=nl, default_r_cut=1.5)
        fslj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) - :math:`\varepsilon`
          :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) - :math:`\sigma`
          :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``.

        Type: `str`
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
    r"""Moliere pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Moliere` computes the Moliere pair force on every particle in the
    simulation state:

    .. math::
        U(r) = \frac{Z_i Z_j e^2}{4 \pi \epsilon_0 r_{ij}} \left[ 0.35 \exp
          \left( -0.3 \frac{r_{ij}}{a_F} \right) + \\
          0.55 \exp \left( -1.2 \frac{r_{ij}}{a_F} \right) + 0.10 \exp
          \left( -6.0 \frac{r_{ij}}{a_F} \right) \right]

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

    Example::

        nl = nlist.Cell()
        moliere = pair.Moliere(default_r_cut = 3.0, nlist=nl)

        Zi = 54
        Zj = 7
        e = 1
        a0 = 1
        aF = 0.8853 * a0 / (np.sqrt(Zi) + np.sqrt(Zj))**(2/3)

        moliere.params[('A', 'B')] = dict(qi=Zi*e, qj=Zj*e, aF=aF)

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

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
    """
    _cpp_class_name = "PotentialPairMoliere"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(qi=float, qj=float, aF=float, len_keys=2))
        self._add_typeparam(params)


class ZBL(Pair):
    r"""ZBL pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.

    `ZBL` computes the Ziegler-Biersack-Littmark pair force on every particle in
    the simulation state:

    .. math::
        U(r) =
          \frac{Z_i Z_j e^2}{4 \pi \epsilon_0 r_{ij}} \left[ 0.1818
          \exp \left( -3.2 \frac{r_{ij}}{a_F} \right) \right. \\
          + 0.5099 \exp \left( -0.9423 \frac{r_{ij}}{a_F} \right) \\
          + 0.2802 \exp \left( -0.4029 \frac{r_{ij}}{a_F} \right) \\
          + \left. 0.02817 \exp \left( -0.2016 \frac{r_{ij}}{a_F} \right)
          \right]

    Where each parameter is defined as:

    - :math:`Z_i` - *Z_i* - Atomic number of species i
      :math:`[\mathrm{dimensionless}]`
    - :math:`Z_j` - *Z_j* - Atomic number of species j
      :math:`[\mathrm{dimensionless}]`
    - :math:`e` - *elementary_charge* - The elementary charge
      :math:`[\mathrm{charge}]`
    - :math:`a_F = \frac{0.8853 a_0}{ Z_i^{0.23} + Z_j^{0.23} }`, where
      :math:`a_0` is the Bohr radius :math:`[\mathrm{length}]`

    Example::

        nl = nlist.Cell()
        zbl = pair.ZBL(default_r_cut=3.0, nlist=nl)

        Zi = 54
        Zj = 7
        e = 1
        a0 = 1
        aF = 0.8853 * a0 / (Zi**(0.23) + Zj**(0.23))

        zbl.params[('A', 'B')] = dict(qi=Zi*e, qj=Zj*e, aF=aF)

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

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``.

        Type: `str`
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
    r"""Mie pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Mie` computes the Mie pair force on every particle in the simulation state.

    .. math::
        U(r) = \left( \frac{n}{n-m} \right) {\left( \frac{n}{m}
          \right)}^{\frac{m}{n-m}} \varepsilon \left[ \left(
          \frac{\sigma}{r} \right)^{n} - \left( \frac{\sigma}{r}
          \right)^{m} \right]

    Example::

        nl = nlist.Cell()
        mie = pair.Mie(nlist=nl, default_r_cut=3.0)
        mie.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0, n=12, m=6)
        mie.r_cut[('A', 'A')] = 2**(1.0/6.0)
        mie.r_on[('A', 'A')] = 2.0
        mie.params[(['A', 'B'], ['C', 'D'])] = dict(epsilon=1.5, sigma=2.0)

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

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
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
    r"""Expanded Mie pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `ExpandedMie` computes the radially shifted Mie pair force on every particle
    in the simulation state:

    .. math::
        U(r) = \left( \frac{n}{n-m} \right) {\left( \frac{n}{m}
          \right)}^{\frac{m}{n-m}} \varepsilon \left[ \left(
          \frac{\sigma}{r-\Delta} \right)^{n} - \left( \frac
          {\sigma}{r-\Delta}
          \right)^{m} \right]

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

    .. py:attribute:: params

        The Expanded Mie potential parameters.
        The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          :math:`\epsilon` :math:`[\mathrm{energy}]`.
        * ``sigma`` (`float`, **required**) -
          :math:`\sigma` :math:`[\mathrm{length}]`.
        * ``n`` (`float`, **required**) -
          :math:`n` :math:`[\mathrm{dimensionless}]`.
        * ``m`` (`float`, **required**) -
          :math:`m` :math:`[\mathrm{dimensionless}]`.
        * ``delta`` (`float`, **required**) -
          :math:`\Delta` :math:`[\mathrm{length}]`.

        Type: `TypeParameter` [ `tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
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
    r"""Onsager reaction field pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `ReactionField` computes the Onsager reaction field pair force
    on all particles in the simulation state.

    Reaction field electrostatics is an approximation to the screened
    electrostatic interaction, which assumes that the medium can be treated as
    an electrostatic continuum of dielectric constant :math:`\epsilon_{RF}`
    outside the cutoff sphere of radius :math:`r_{\mathrm{cut}}`. See: `Barker
    et. al. 1973 <http://dx.doi.org/10.1080/00268977300102101>`_.

    By default (``use_charge=False``), the reaction field potential ignores the
    particle charges. Two parameters, :math:`\varepsilon` and
    :math:`\epsilon_{RF}` are needed. If :math:`\epsilon_{RF}` is specified as
    zero, it will represent infinity:

    .. math::

       U(r) = \varepsilon \left[ \frac{1}{r} +
           \frac{(\epsilon_{RF}-1) r^2}{(2 \epsilon_{RF} + 1) r_c^3} \right]

    When *use_charge* is set to `True`, the following formula is evaluated
    instead:

    .. math::

        U(r) = q_i q_j \varepsilon \left[ \frac{1}{r} +
          \frac{(\epsilon_{RF}-1) r^2}{(2 \epsilon_{RF} + 1) r_c^3} \right]

    where :math:`q_i` and :math:`q_j` are the charges of the particle pair.

    Example::

        nl = nlist.Cell()
        reaction_field = pair.reaction_field(nl, default_r_cut=3.0)
        reaction_field.params[('A', 'B')] = dict(epsilon=1.0, eps_rf=1.0)
        reaction_field.params[('B', 'B')] = dict(
            epsilon=1.0, eps_rf=0.0, use_charge=True)

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) - :math:`\varepsilon`
          :math:`[\mathrm{energy} \cdot \mathrm{length}]`
        * ``eps_rf`` (`float`, **required**) - :math:`\epsilon_{RF}`
          :math:`[\mathrm{dimensionless}]`
        * ``use_charge`` (`bool`, **optional**) - evaluate pair force
          using particle charges (*default*: False)

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
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
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        name (str): Name of the force instance.
        mode (str): Energy shifting mode.

    `DLVO` computes the DLVO dispersion and electrostatic interaction pair force
    on every particle in the simulation state with:

    .. math::
        V_{\mathrm{DLVO}}(r) = &- \frac{A}{6} \left[
            \frac{2a_1a_2}{r^2 - (a_1+a_2)^2} +
            \frac{2a_1a_2}{r^2 - (a_1-a_2)^2} \\
            + \log \left(
            \frac{r^2 - (a_1+a_2)^2}{r^2 - (a_1-a_2)^2} \right) \right] \\
            & + \frac{a_1 a_2}{a_1+a_2} Z e^{-\kappa(r - (a_1+a_2))}

    where :math:`a_1` is the radius of first particle in the pair, :math:`a_2`
    is the radius of second particle in the pair, :math:`A` is the Hamaker
    constant, :math:`Z` is proportional to the surface electric potential,
    and :math:`\kappa` is the screening parameter.

    The first term corresponds to the attractive van der Waals interaction with
    and the second term to the repulsive double-layer interaction between two
    spherical surfaces. See "Intermolecular and Surface Forces" Israelachvili
    2011, pp. 317.

    Example::

        nl = hoomd.md.nlist.Cell()
        dlvo = hoomd.md.pair.DLVO(nlist=nl)
        dlvo.params[('A', 'A')] = dict(A=1.0, kappa=1.0, Z=2, a1=1, a2=1)
        dlvo.params[('A', 'B')] = dict(A=2.0, kappa=0.5, Z=3, a1=1, a2=3)
        dlvo.params[('B', 'B')] = dict(A=2.0, kappa=0.5, Z=3, a1=3, a2=3)

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

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"`` or ``"shift"``.

        Type: `str`
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


class Buckingham(Pair):
    r"""Buckingham pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Buckingham` computes the Buckingham pair force on every particle in the
    simulation state with:

    .. math::
        U(r) = A \exp\left(-\frac{r}{\rho}\right) - \frac{C}{r^6}

    Example::

        nl = nlist.Cell()
        buck = pair.Buckingham(nl, default_r_cut=3.0)
        buck.params[('A', 'A')] = {'A': 2.0, 'rho'=0.5, 'C': 1.0}
        buck.params[('A', 'B')] = dict(A=1.0, rho=1.0, C=1.0)
        buck.params[('B', 'B')] = dict(A=2.0, rho=2.0, C=2.0)

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``A`` (`float`, **required**) - :math:`A` :math:`[\mathrm{energy}]`
        * ``rho`` (`float`, **required**) - :math:`\rho`
          :math:`[\mathrm{length}]`
        * ``C`` (`float`, **required**) - :math:`C` :math:`[\mathrm{energy}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
    """

    _cpp_class_name = "PotentialPairBuckingham"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(A=float, rho=float, C=float, len_keys=2))
        self._add_typeparam(params)


class LJ1208(Pair):
    r"""Lennard-Jones 12-8 pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `LJ1208` computes the Lennard-Jones 12-8 pair force on every particle
    in the simulation state.

    Example::

        nl = nlist.Cell()
        lj1208 = pair.LJ1208(nl, default_r_cut=3.0)
        lj1208.params[('A', 'A')] = {'sigma': 1.0, 'epsilon': 1.0}
        lj1208.params[('A', 'B')] = dict(epsilon=2.0, sigma=1.0)

    .. math::
        U(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
          \left( \frac{\sigma}{r} \right)^{8} \right]

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) - energy parameter
          :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) - particle size :math:`\sigma`
          :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
    """
    _cpp_class_name = "PotentialPairLJ1208"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, len_keys=2))
        self._add_typeparam(params)


class LJ0804(Pair):
    r"""Lennard-Jones 8-4 pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `LJ0804` computes the Lennard-Jones 8-4 pair force on every particle in the
    simulation state:

    .. math::
        U(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{8} -
          \left( \frac{\sigma}{r} \right)^{4} \right]

    Example::

        nl = nlist.Cell()
        lj0804 = pair.LJ0804(nl, default_r_cut=3.0)
        lj0804.params[('A', 'A')] = {'sigma': 1.0, 'epsilon': 1.0}
        lj0804.params[('A', 'B')] = dict(epsilon=2.0, sigma=1.0)
        lj0804.r_cut[('A', 'B')] = 3.0

    .. py:attribute:: params

        The LJ potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
    """
    _cpp_class_name = "PotentialPairLJ0804"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, len_keys=2))
        self._add_typeparam(params)


class Fourier(Pair):
    r"""Fourier pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Fourier` computes the  Fourier pair force on every particle in the
    simulation state:

    .. math::
        U(r) = \frac{1}{r^{12}} + \frac{1}{r^2}\sum_{n=1}^4
          \left[ a_n cos \left( \frac{n \pi r}{r_{cut}} \right) +
          b_n sin \left( \frac{n \pi r}{r_{cut}} \right) \right]

    where

    .. math::

        a_1 &= \sum_{n=2}^4 (-1)^n a_n \\
        b_1 &= \sum_{n=2}^4 n (-1)^n b_n \\

    enforce :math:`U(r_\mathrm{cut}) = 0`.

    Example::

        nl = nlist.Cell()
        fourier = pair.Fourier(default_r_cut=3.0, nlist=nl)
        fourier.params[('A', 'A')] = dict(a=[a2,a3,a4], b=[b2,b3,b4])

    .. py:attribute:: params

        The Fourier potential parameters. The dictionary has the following
        keys:

        * ``a`` (`float`, **required**) - array of 3 values corresponding to
          a2, a3 and a4 in the Fourier series :math:`[\mathrm{dimensionless}]`
        * ``b`` (`float`, **required**) - array of 3 values corresponding to
          b2, b3 and b4 in the Fourier series :math:`[\mathrm{dimensionless}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"`` or ``"xplor"``.

        Type: `str`
    """
    _cpp_class_name = "PotentialPairFourier"
    _accepted_modes = ("none", "xplor")

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(a=(float, float, float),
                              b=(float, float, float),
                              len_keys=2))
        self._add_typeparam(params)


class OPP(Pair):
    r"""Oscillating pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `OPP` computes the oscillating pair force on all particles in the simulation
    state:

    .. math::
        U(r) = C_1 r^{-\eta_1}
              + C_2 r^{-\eta_2} \cos{\left(k r - \phi\right)}

    The potential was introduced in `Marek Mihalkovič and C. L. Henley 2012`_.

    .. _Marek Mihalkovič and C. L. Henley 2012:
       https://dx.doi.org/10.1103/PhysRevB.85.092102

    Example::

        nl = nlist.Cell()
        opp = pair.OPP(nl, default_r_cut=3.0)
        opp.params[('A', 'A')] = {
            'C1': 1., 'C2': 1., 'eta1': 15,
            'eta2': 3, 'k': 1.0, 'phi': 3.14}
        opp.r_cut[('A', 'B')] = 3.0

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

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
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
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `TWF` computes the Ten-wolde Frenkel potential on all particles in the
    simulation state:

    .. math::
        U(r) = \frac{4 \epsilon}{\alpha^2} {\left[
        {\left(\frac{\sigma^2}{r^2} - 1 \right)}^{-6} -
        \alpha {\left(\frac{\sigma^2}{r^2} - 1 \right)}^{-3}\right]}

    The potential was introdcued in `Pieter Rein ten Wolde and Daan Frenkel
    1997`_.

    .. _Pieter Rein ten Wolde and Daan Frenkel 1997:
       https://dx.doi.org/10.1126/science.277.5334.1975

    Example::

        nl = nlist.Cell()
        twf = hoomd.md.pair.TWF(nl, default_r_cut=3.0)
        twf.params[('A', 'A')] = {'sigma': 1.0, 'epsilon': 1.0, 'alpha': 50.0}
        twf.r_cut[('A', 'B')] = 3.0

    .. py:attribute:: params

        The LJ potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`
        * ``alpha`` (`float`, **required**) -
          controls well-width :math:`\alpha` :math:`[\mathrm{dimensionless}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
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


class LJGauss(Pair):
    r"""Lennard-Jones-Gauss pair potential.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `LJGauss` computes the Lennard-Jones Gauss force on all particles in the
    simulation state:

    .. math::
        U(r) = 1\ [\mathrm{energy}] \cdot \left[
                 \left ( \frac{1\ [\mathrm{length}]}{r} \right)^{12} -
                 2\  \left(\frac{1 [\mathrm{length}]}{r} \right)^{6} \right] -
            \epsilon
            \exp \left[- \frac{\left(r - r_{0}\right)^{2}}{2 \sigma^{2}} \right]

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float` > 0, **required**) -
          Gaussian width :math:`\sigma` :math:`[\mathrm{length}]`
        * ``r0`` (`float`, **required**) -
          Gaussian center :math:`r_0` :math:`[\mathrm{length}]`

    Example::

        nl = hoomd.md.nlist.Cell()
        ljg = pair.LJGauss(nl)
        ljg.params[('A', 'A')] = dict(epsilon=1.0, sigma=0.02, r0=1.6)
        ljg.params[('A', 'B')] = {'epsilon' : 2.0, 'sigma' : 0.02, 'r0' : 1.6}
        ljg.params[('A', 'B')] = {'epsilon' : 2.0, 'sigma' : 0.02, 'r0' : 1.6}

    .. versionadded:: 3.1.0
    """
    _cpp_class_name = "PotentialPairLJGauss"

    def __init__(self,
                 nlist,
                 default_r_cut=None,
                 default_r_on=0.0,
                 mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=positive_real, 
                              r0=float, len_keys=2))
        self._add_typeparam(params)
