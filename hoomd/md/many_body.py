import hoomd
from hoomd.md import _md
from hoomd.md.pair import _NBody
from hoomd.parameterdicts import TypeParameterDict
from hoomd.typeparam import TypeParameter


class _ThreeBody(_NBody):
    """Superclass for 3-body potentials. Each requires a full neighborlist."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _attach(self):
        super()._attach()
        self.nlist._cpp_obj.setStorageMode(
            _md.NeighborList.storageMode.full)



class Tersoff(_ThreeBody):
    R""" Tersoff Potential.

    Args:
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        r_cut (float): Default cutoff radius (in distance units).
        r_on (float): Default turn-on radius (in distance units).
        mode (str): Energy shifting mode.

    :py:class:`Tersoff` specifies that the Tersoff three-body potential should be applied to every
    non-bonded particle pair in the simulation.  Despite the fact that the Tersoff potential accounts
    for the effects of third bodies, the species of the third body is irrelevant. It can thus use
    type-pair parameters similar to those of the pair potentials.

    The Tersoff potential is a bond-order potential based on the Morse potential that accounts for the weakening of
    individual bonds with increasing coordination number. It does this by computing a modifier to the
    attractive term of the potential. The modifier contains the effects of third-bodies on the bond
    energies. The potential also includes a smoothing function around the cutoff. The smoothing function
    used in this work is exponential in nature as opposed to the sinusoid used by Tersoff. The exponential
    function provides continuity up (I believe) the second derivative.

    """
    _cpp_class_name = "PotentialTersoff"
    def __init__(self, nlist, r_cut=None, r_on=0., mode='none'):
        super().__init__(nlist, r_cut, r_on, mode);
        params = TypeParameter(
                'params',
                'particle_types',
                TypeParameterDict(cutoff_thickness=0.2,
                    C1=1.0,
                    C2=1.0,
                    lambda1=2.0,
                    lambda2=1.0,
                    lambda3=0.0,
                    dimer_r=1.5,
                    n=0.0,
                    gamma=0.0,
                    c=0.0,
                    d=1.0,
                    m=0.0,
                    alpha=3.0,
                    len_keys=2)
                )
        self._add_typeparam(params)


class RevCross(_ThreeBody):
    R""" Reversible crosslinker three-body potential to model bond swaps.

    Args:
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        r_cut (float): Default cutoff radius (in distance units).
        r_on (float): Default turn-on radius (in distance units).
        mode (str): Energy shifting mode.

    :py:class:`RevCross` specifies that the revcross three-body potential should be applied to every
    non-bonded particle pair in the simulation.  Despite the fact that the revcross potential accounts
    for the effects of third bodies, it is actually just a combination of two body potential terms.
    It can thus use type-pair parameters similar to those of the pair potentials.

    The RevCross potential has been described in detail in `S. Ciarella and W.G. Ellenbroek 2019 <https://arxiv.org/abs/1912.08569>`_. It is based on a generalized-Lennard-Jones pairwise
    attraction to form bonds between interacting particless:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{ij}(r)  =  4 \varepsilon \left[ \left( \dfrac{ \sigma}{r_{ij}} \right) ^{2n}- \left( \dfrac{ \sigma}{r_{ij}} \right)^{n} \right] \qquad r<r_{cut}
        \end{eqnarray*}

    with the following coefficients:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`\sigma` - *sigma* (in distance units)
    - :math:`n` - *n* (unitless)
    - :math:`m` - *m* (unitless)
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command

    Then an additional three-body repulsion is evaluated to compensate the bond energies imposing single bond per particle condition:

    .. math::
        :nowrap:

            \begin{eqnarray*}
            v^{\left( 3b \right)}_{ijk}=\lambda \epsilon\,\hat{v}^{ \left( 2b \right)}_{ij}\left(\vec{r}_{ij}\right) \cdot \hat{v}^{ \left( 2b \right)}_{ik}\left(\vec{r}_{ik}\right)~,\\
            \end{eqnarray*}

    where the two body potential is rewritten as:

    .. math::
        :nowrap:

            \begin{eqnarray*}
            \hat{v}^{ \left( 2b \right)}_{ij}\left(\vec{r}_{ij}\right) =
            \begin{cases}
            & 1 \qquad \qquad \; \; \qquad r\le r_{min}\\
            & - \dfrac{v_{ij}\left(\vec{r}_{ij}\right)}{\epsilon} \qquad r > r_{min}~.\\
            \end{cases}
            \end{eqnarray*}

    .. attention::

        The RevCross potential models an asymmetric interaction between two different chemical moieties that can form a reversible bond.
        This requires the definition of (at least) two different types of particles.
        A reversible bond is only possible between two different species, otherwise :math:`v^{\left( 3b \right)}_{ijk}`, would prevent any bond.
        In our example we then set the interactions for types A and B with ``potRevC.params[(['A','B'],['A','B'])] = dict(sigma=0.0,n=0,epsilon=0,lambda3=0)``
        and the only non-zero energy only between the different types ``potRevC.params[('A','B')] = dict(sigma=1,n=100,epsilon=100,lambda3=1)``.
        Notice that the number of the minoritary species corresponds to the maximum number of bonds.


    This three-body term also tunes the energy required for a bond swap through the coefficient:
    - :math:`\lambda` - *lambda3* (unitless)
    in `S. Ciarella and W.G. Ellenbroek 2019 <https://arxiv.org/abs/1912.08569>`_ is explained that setting :math:`\lambda=1` corresponds to no energy requirement to initiate bond swap, while this
    energy barrier scales roughly as :math:`\beta \Delta E_\text{sw} =\beta \varepsilon(\lambda-1)`.

    Note:

        Choosing :math:`\lambda=1` pushes the system towards clusterization because the three-body term is not enough to
        compensate the energy of multiple bonds, so it may cause unphysical situations.


    Example::

        nl = md.nlist.Cell()
        potBondSwap = md.many_body.RevCross(r_cut=1.3,nlist=nl)
        potBondSwap.params[(['A','B'],['A','B'])] = dict(sigma=0,n=0,epsilon=0,lambda3=0)
        # a bond can be made only between A-B and not A-A or B-B
        potBondSwap.params[('A','B')] = dict(sigma=1,n=100,epsilon=10,lambda3=1)
    """
    _cpp_class_name = "PotentialRevCross"
    def __init__(self, nlist, r_cut=None, r_on=0., mode='none'):
        super().__init__(nlist, r_cut, r_on, mode);
        params = TypeParameter('params', 'particle_types',
                               TypeParameterDict(sigma=2.0, n=1.0, epsilon=1.0,
                                   lambda3=1.0, len_keys=2))
        self._add_typeparam(params)


class SquareDensity(_ThreeBody):
    R""" Soft potential for simulating a van-der-Waals liquid

    Args:
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        r_cut (float): Default cutoff radius (in distance units).
        r_on (float): Default turn-on radius (in distance units).
        mode (str): Energy shifting mode.

    :py:class:`SquareDensity` specifies that the three-body potential should be applied to every
    non-bonded particle pair in the simulation, that is harmonic in the local density.

    The self energy per particle takes the form

    .. math:: \Psi^{ex} = B (\rho - A)^2

    which gives a pair-wise additive, three-body force

    .. math:: \vec{f}_{ij} = \left( B (n_i - A) + B (n_j - A) \right) w'_{ij} \vec{e}_{ij}

    Here, :math:`w_{ij}` is a quadratic, normalized weighting function,

    .. math:: w(x) = \frac{15}{2 \pi r_{c,\mathrm{weight}}^3} (1-r/r_{c,\mathrm{weight}})^2

    The local density at the location of particle *i* is defined as

    .. math:: n_i = \sum\limits_{j\neq i} w_{ij}\left(\big| \vec r_i - \vec r_j \big|\right)

    The following coefficients must be set per unique pair of particle types:

    - :math:`A` - *A* (in units of volume^-1) - mean density (*default*: 0)
    - :math:`B` - *B* (in units of energy*volume^2) - coefficient of the harmonic density term

    Example::

        nl = nlist.Cell()
        sqd = md.many_body.SquareDensity(r_cut=3.0, nlist=nl)
        sqd.params[('A', 'A')] = dict(A=0.1, B=1.0)

    For further details regarding this multibody potential, see

    Warning:
        Currently HOOMD does not support reverse force communication between MPI domains on the GPU.
        Since reverse force communication is required for the calculation of multi-body potentials, attempting to use the
        square_density potential on the GPU with MPI will result in an error.

    [1] P. B. Warren, "Vapor-liquid coexistence in many-body dissipative particle dynamics"
    Phys. Rev. E. Stat. Nonlin. Soft Matter Phys., vol. 68, no. 6 Pt 2, p. 066702, 2003.
    """
    _cpp_class_name = "PotentialSquareDensity"
    def __init__(self, nlist, r_cut=None, r_on=0., mode='none'):
        super().__init__(nlist, r_cut, r_on, mode);
        params = TypeParameter(
                'params',
                'particle_types',
                TypeParameterDict(A=0.0, B=float, len_keys=2))
        self._add_typeparam(params)



