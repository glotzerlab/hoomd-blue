# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement many body potentials."""

import hoomd
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeconverter import OnlyTypes, positive_real
from hoomd.data.typeparam import TypeParameter
from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.md.nlist import NList

validate_nlist = OnlyTypes(NList)


class Triplet(Force):
    r"""Common three body potential documentation.

    Users should not invoke :py:class:`Triplet` directly. It is a base class
    that provides common features to all standard triplet forces. Common
    documentation for all three-body potentials is documented here.

    All triplet force commands specify that a given force be computed on all
    particles which have at least two other particles within a short range
    cutoff distance :math:`r_{\mathrm{cut}}`.

    The force :math:`\vec{F}` applied to each particle :math:`i` is:

    .. math::
        :nowrap:

        \begin{eqnarray}
        \vec{F_i}
            =& -\nabla V(\vec r_{ij}, \vec r_{ik});
             & r_{ij} < r_{\mathrm{cut}}
                \textrm{ and } r_{ik} < r_{\mathrm{cut}} \\
           =& 0; & otherwise
        \end{eqnarray}

    Where
    :math:`\vec r_{\alpha \beta} = \vec r_{\alpha} - \vec r_{\beta}`.

    The following coefficients must be set per unique pair of particle types.
    See :class:`hoomd.md.many_body` for information on how to set coefficients.

    .. py:attribute:: r_cut

        *r_cut* (in distance units), *optional*: defaults to the value
        ``default_r_cut`` specified on construction.

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `float`])

    Warning:
        Currently HOOMD does not support reverse force communication between MPI
        domains on the GPU. Since reverse force communication is required for
        the calculation of three-body potentials, attempting to use this
        potential on the GPU with MPI will result in an error.
    """

    def __init__(self, nlist, default_r_cut=None):
        super().__init__()
        self._nlist = validate_nlist(nlist)
        r_cut_param = TypeParameter(
            'r_cut', 'particle_types',
            TypeParameterDict(positive_real, len_keys=2))
        if default_r_cut is not None:
            r_cut_param.default = default_r_cut
        self._add_typeparam(r_cut_param)

    def _attach(self):
        if not self._nlist._added:
            self._nlist._add(self._simulation)
        else:
            if self._simulation != self._nlist._simulation:
                raise RuntimeError("{} object's neighbor list is used in a "
                                   "different simulation.".format(type(self)))
        if not self.nlist._attached:
            self.nlist._attach()
        self.nlist._cpp_obj.setStorageMode(_md.NeighborList.storageMode.full)

        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(_md, self._cpp_class_name)
        else:
            cls = getattr(_md, self._cpp_class_name + "GPU")

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
        else:
            self._nlist = validate_nlist(value)

    @property
    def _children(self):
        return [self._nlist]


class Tersoff(Triplet):
    r"""Tersoff Potential.

    Args:
        nlist (:py:class:`hoomd.md.nlist.NList`): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    The Tersoff potential is a bond-order potential based on the Morse potential
    that accounts for the weakening of individual bonds with increasing
    coordination number. It does this by computing a modifier to the attractive
    term of the potential. The modifier contains the effects of third-bodies on
    the bond energies. The potential also includes a smoothing function around
    the cutoff. The smoothing function used in this work is exponential in
    nature as opposed to the sinusoid used by J. Tersoff 1988 (`paper`_).

    .. _paper: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.38.9902

    :py:class:`Tersoff` specifies that the Tersoff three-body potential should
    be applied to every non-bonded particle pair in the simulation. Despite the
    fact that the Tersoff potential accounts for the effects of third bodies,
    the species of the third body is irrelevant. It can thus use type-pair
    parameters similar to those of the pair potentials.

    .. math::

        V_{ij}(r) = \frac{1}{2} f_C(r_{ij})
            \left[f_R(r_{ij}) + b_{ij}f_A(r_{ij})\right]

    where

    .. math::

        f_R(r) = A_1e^{\lambda_1(r_D-r)}

        f_A(r) = A_2e^{\lambda_2(r_D-r)}

    .. math::
        :nowrap:

        \begin{equation}
            f_C(r) =
            \begin{cases}
                1 & r < r_{\mathrm{cut}} - r_{CT} \\
                \exp \left[-\alpha\frac{x(r)^3}{x(r)^3 - 1} \right]
                  & r_{\mathrm{cut}} - r_{CT} < r < r_{\mathrm{cut}} \\
                0 & r > r_{\mathrm{cut}}
            \end{cases}
        \end{equation}

    .. math::

        b_{ij} = (1 + \gamma^n\chi_{ij}^n)^{\frac{-1}{2n}}

    In the definition of :math:`f_C(r)`, there is a quantity :math:`x(r)`, which
    is defined as

    .. math::

        x(r) = \frac{r - (r_{\mathrm{cut}} - r_{CT})}{r_{CT}}

    which ensures continuity between the different regions of the potential. In
    the definition of :math:`b_{ij}`, there is a quantity :math:`\chi_{ij}`
    which is defined as

    .. math::

        \chi_{ij} = \sum_{k \neq i,j} f_C(r_{ik})
                     \cdot e^{\lambda_3^3 |r_{ij} - r_{ik}|^3}
        \cdot g(\theta_{ijk})

        g(\theta_{ijk}) = 1 + \frac{c^2}{d^2}
                           - \frac{c^2}{d^2 + |m - \cos(\theta_{ijk})|^2}

    The parameters of this potential are set via the ``params`` dictionary,
    they must be set for each unique pair of particle types.

    .. py:attribute:: params

        The Tersoff potential parameters. The dictionary has the following
        keys:

        * ``magnitudes`` (tuple[`float`, `float`]) - :math:`(A_1, A_2)` -
          Magnitudes of the repulsive and attractive
          terms (*default*: (1.0, 1.0)) :math:`[\mathrm{energy}]`
        * ``exp_factors`` (tuple[`float`, `float`]) -
          :math:`(\lambda_1, \lambda_2)` - exponential factors of the
          repulsive and attractive
          terms (*default*: 2.0) :math:`[\mathrm{length}^{-1}]`
        * ``lambda3`` (`float`) - :math:`\lambda_3` - exponential factor in
          :math:`\chi_{ij}` (*default*: 0.0) :math:`[\mathrm{length}^{-1}]`
        * ``dimer_r`` (`float`) - :math:`r_D` - length shift in attractive
          and repulsive terms (*default*: 1.5) :math:`[\mathrm{length}]`
        * ``cutoff_thickness`` (`float`) - :math:`r_{CT}` - distance which
          defines the different regions of the
          potential (*default*: 0.2) :math:`[\mathrm{length}]`
        * ``alpha`` (`float`) - :math:`\alpha` - decay rate of the cutoff
          term
          :math:`f_C(r)` (*default*: 3.0) :math:`[\mathrm{dimensionless}]`
        * ``n`` (`float`) - :math:`n` - power in
          :math:`b_{ij}` (*default*: 0.0) :math:`[\mathrm{dimensionless}]`
        * ``gamma`` (`float`) - :math:`\gamma` - coefficient in
          :math:`b_{ij}` (*default*: 0.0) :math:`[\mathrm{dimensionless}]`
        * ``c`` (`float`) - :math:`c` - coefficient in
          :math:`g(\theta)` (*default*: 0.0) :math:`[\mathrm{dimensionless}]`
        * ``d`` (`float`) - :math:`d` - coefficient in
          :math:`g(\theta)` (*default*: 1.0) :math:`[\mathrm{dimensionless}]`
        * ``m`` (`float`) - :math:`m` - coefficient in
          :math:`g(\theta)` (*default*: 0.0) :math:`[\mathrm{dimensionless}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = md.nlist.Cell()
        tersoff = md.many_body.Tersoff(default_r_cut=1.3, nlist=nl)
        tersoff.params[('A', 'B')] = dict(magnitudes=(2.0, 1.0), lambda3=5.0)
    """
    _cpp_class_name = "PotentialTersoff"

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist, default_r_cut)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(cutoff_thickness=0.2,
                              magnitudes=(1.0, 1.0),
                              exp_factors=(2.0, 1.0),
                              lambda3=0.0,
                              dimer_r=1.5,
                              n=0.0,
                              gamma=0.0,
                              c=0.0,
                              d=1.0,
                              m=0.0,
                              alpha=3.0,
                              len_keys=2))
        self._add_typeparam(params)


class RevCross(Triplet):
    r"""Reversible crosslinker three-body potential to model bond swaps.

    Args:
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    :py:class:`RevCross` specifies that the revcross three-body potential
    should be applied to every non-bonded particle pair in the simulation.
    Despite the fact that the revcross potential accounts for the effects of
    third bodies, it is actually just a combination of two body potential terms.
    It can thus use type-pair parameters similar to those of the pair
    potentials.

    The RevCross potential has been described in detail in
    `S. Ciarella and W.G. Ellenbroek 2019 <https://arxiv.org/abs/1912.08569>`_.
    It is based on a generalized Lennard-Jones pairwise attraction to form bonds
    between interacting particles:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{ij}(r)  =  4 \varepsilon \left[
            \left( \dfrac{ \sigma}{r_{ij}} \right)^{2n}
            - \left( \dfrac{ \sigma}{r_{ij}} \right)^{n}
        \right]; \qquad r < r_{cut}
        \end{eqnarray*}

    Then an additional three-body repulsion is evaluated to compensate the bond
    energies imposing single bond per particle condition:

    .. math::
        :nowrap:

            \begin{eqnarray*}
            v^{\left( 3b \right)}_{ijk} = \lambda \epsilon\,
                \hat{v}^{ \left( 2b \right)}_{ij}
            \left(\vec{r}_{ij}\right)
                \cdot \hat{v}^{ \left( 2b \right)}_{ik}
            \left(\vec{r}_{ik}\right)~,\
            \end{eqnarray*}

    where the two body potential is rewritten as:

    .. math::
        :nowrap:

            \begin{eqnarray*}
            \hat{v}^{ \left( 2b \right)}_{ij}\left(\vec{r}_{ij}\right) =
            \begin{cases}
            1; & r \le r_{min} \\
            - \dfrac{v_{ij}\left(\vec{r}_{ij}\right)}{\epsilon};
              & r > r_{min} \\
            \end{cases}
            \end{eqnarray*}

    .. attention::

        The RevCross potential models an asymmetric interaction between two
        different chemical moieties that can form a reversible bond. This
        requires the definition of (at least) two different types of particles.
        A reversible bond is only possible between two different species,
        otherwise :math:`v^{\left( 3b \right)}_{ijk}`, would prevent any bond.
        In our example we then set the interactions for types A and B with
        ``rev_c.params[[('A','B'),('A','B')]]`` to
        ``{"sigma": 0.0, "n": 0, "epsilon": 0, "lambda3": 0}``
        and the only non-zero energy only between the different types with
        setting ``rev_c.params[('A','B')]`` to
        ``{"sigma":1, "n": 100, "epsilon": 100, "lambda3": 1}``.
        Notice that the number of the minority species corresponds to the
        maximum number of bonds.


    This three-body term also tunes the energy required for a bond swap through
    the coefficient:- :math:`\lambda` - *lambda3* (unitless)
    in S. Ciarella and W.G. Ellenbroek 2019
    (`paper link <https://arxiv.org/abs/1912.08569>`__)
    is explained that setting :math:`\lambda=1` corresponds to no energy
    requirement to initiate bond swap, while this energy barrier scales roughly
    as :math:`\beta \Delta E_\text{sw} =\beta \varepsilon(\lambda-1)`.

    Note:
        Choosing :math:`\lambda=1` pushes the system towards clusterization
        because the three-body term is not enough to compensate the energy of
        multiple bonds, so it may cause unphysical situations.

    Use ``params`` dictionary to set potential coefficients. The coefficients
    must be set per unique pair of particle types.

    .. py:attribute:: params

        The revcross potential parameters. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) - :math:`\varepsilon`
          :math:`[\mathrm{energy}]`

        * ``sigma`` (`float`, **required**) - :math:`\sigma`
          :math:`[\mathrm{length}]`

        * ``n`` (`float`, **required**) - :math:`n`
          :math:`[\mathrm{dimensionless}]`

        * ``lambda3`` (`float`, **required**) - :math:`\lambda_3`
          :math:`[\mathrm{dimensionless}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = md.nlist.Cell()
        bond_swap = md.many_body.RevCross(default_r_cut=1.3,nlist=nl)
        bond_swap.params[(['A','B'],['A','B'])] = {
            "sigma":0,"n": 0, "epsilon": 0, "lambda3": 0}
        # a bond can be made only between A-B and not A-A or B-B
        bond_swap.params[('A','B')] = {
            "sigma": 1, "n": 100, "epsilon": 10, "lambda3": 1}
    """
    _cpp_class_name = "PotentialRevCross"

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist, default_r_cut)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(sigma=2.0,
                              n=1.0,
                              epsilon=1.0,
                              lambda3=1.0,
                              len_keys=2))
        self._add_typeparam(params)


class SquareDensity(Triplet):
    r"""Soft potential for simulating a van der Waals liquid.

    Args:
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    :py:class:`SquareDensity` specifies that the three-body potential should be
    applied to every non-bonded particle pair in the simulation, that is
    harmonic in the local density.

    The self energy per particle takes the form

    .. math::

        \Psi^{ex} = B (\rho - A)^2

    which gives a pair-wise additive, three-body force

    .. math::

        \vec{f}_{ij} = \left( B (n_i - A) + B (n_j - A) \right) w'_{ij}
        \vec{e}_{ij}

    Here, :math:`w_{ij}` is a quadratic, normalized weighting function,

    .. math::

        w(x) = \frac{15}{2 \pi r_{c,\mathrm{weight}}^3}
               (1-r/r_{c,\mathrm{weight}})^2

    The local density at the location of particle *i* is defined as

    .. math::

        n_i = \sum\limits_{j\neq i} w_{ij}
              \left(\big| \vec r_i - \vec r_j \big|\right)

    Use `params` dictionary to set potential coefficients. The coefficients
    must be set per unique pair of particle types.

    .. py:attribute:: params

        The SquareDensity potential parameters. The dictionary has the
        following keys:

        * ``A`` (`float`, **required**) - :math:`A` - mean density
          (*default*:0) :math:`[\mathrm{length}^{-2}]` in 2D and
          :math:`[\mathrm{length}^{-3}]` in 3D
        * ``B`` (`float`, **required**) - :math:`B` - coefficient of the
          harmonic density term
          :math:`[\mathrm{energy} \cdot \mathrm{length}^4]` in 2D and
          :math:`[\mathrm{energy} \cdot \mathrm{length}^6]` in 3D

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        sqd = md.many_body.SquareDensity(nl, default_r_cut=3.0)
        sqd.params[('A', 'B')] = dict(A=1.0, B=2.0)
        sqd.params[('B', 'B')] = dict(A=2.0, B=2.0, default_r_on=1.0)

    For further details regarding this multibody potential, see

    [1] P. B. Warren, "Vapor-liquid coexistence in many-body dissipative
    particle dynamics" Phys. Rev. E. Stat. Nonlin. Soft Matter Phys., vol. 68,
    no. 6 Pt 2, p. 066702, 2003.
    """
    _cpp_class_name = "PotentialSquareDensity"

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist, default_r_cut)
        params = TypeParameter('params', 'particle_types',
                               TypeParameterDict(A=0.0, B=float, len_keys=2))
        self._add_typeparam(params)
