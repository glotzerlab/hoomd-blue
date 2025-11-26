# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Frictional pair force classes apply a force, and torque on every
particle in the simulation state. The following general expression
for Markovian tangential friction forces is implemented for
interactions between two spherical particles and is discussed in detail in
`Hofmann et. al. 2025`_. For two particles :math:`i` and
:math:`j` with radii :math:`R_{i,j}`, center positions :math:`\mathbf{r}_{i,j}`,
angular velocities :math:`\mathbf{\omega}_{i,j}`, and translational velocities
:math:`\mathbf{v}_{i,j}`, their surface velocities at the contact point are given by

.. _Hofmann et. al. 2025: https://doi.org/10.48550/arXiv.2507.16388

.. math::

    \begin{align*}
    \mathbf{u}_i &= \mathbf{v}_i+\mathbf{\omega}_i \times \mathbf{\hat{r}}_{ij}R_i \\
    \mathbf{u}_j &= \mathbf{v}_j-\mathbf{\omega}_j \times \mathbf{\hat{r}}_{ij}R_j \, ,
    \end{align*}

where :math:`\mathbf{\hat{r}}_{ij}=\mathbf{r}_{ij}/r_{i,j}`. With these expressions,
we calculate the relative tangential velocity :math:`\mathbf{u}^\perp_{i,j}` at the
contact point

.. math::

    \mathbf{u}^\perp_{i,j} =  \mathbf{P}(\mathbf{\hat{r}}_{ij})(\mathbf{v}_j
                            -\mathbf{v}_i) -(\mathbf{\omega}_iR_i+\mathbf{\omega}_jR_j)
                            \times \mathbf{\hat{r}}_{ij}\, ,

wit the projection operator :math:`\mathbf{P}(\mathbf{\hat{r}}_{ij})=1
-\mathbf{\hat{r}}_{ij}\mathbf{\hat{r}}_{ij}`. We model the tangential
friction force at the contact point very general as

.. math::

    \mathbf{F}^\mathrm{f,contact}_i = -\mathbf{F}^\mathrm{f,contact}_j = f(u^\perp_{i,j}
                                        ,r_{i,j})\mathbf{\hat{u}}^\perp_{i,j}

with :math:`\mathbf{\hat{u}}^\perp_{i,j}=\mathbf{u}^\perp_{i,j}/u^\perp_{i,j}`, where
:math:`f(u^\perp_{i,j},r_{i,j})` is an arbitrary function. The surface force
:math:`\mathbf{F}^\mathrm{f,contact}_i` generates a center-of-mass force and torque
acting on particle :math:`i`,

.. math::

    \begin{align*}
    \mathbf{F}^\mathrm{f}_{ij} &= \mathbf{F}^\mathrm{f,contact}_i \\
    \mathbf{\tau}^\mathrm{f}_{ij} &= R_i\mathbf{\hat{r}}_{ij} \times
    \mathbf{F}^\mathrm{f,contact}_i\, ,
    \end{align*}

which is a pair friction force and torque resulting from the friction with the particle
:math:`j`.

The functional form of :math:`f(u^\perp_{i,j},r_{i,j})` specifies the frictional model.

`FrictionalPair` does not support any shifting modes.

.. invisible-code-block: python

    neighbor_list = hoomd.md.nlist.Cell(buffer = 0.4)
    simulation = hoomd.util.make_example_simulation()
    simulation.operations.integrator = hoomd.md.Integrator(
        dt=0.001,
        integrate_rotational_dof = True)
"""

import inspect

from hoomd.md.pair.pair import Pair
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter


class FrictionalPair(Pair):
    r"""Base class friction pair force.

    `FrictionalPair` is the base class for all frictional pair forces.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """

    __doc__ = inspect.cleandoc(__doc__) + "\n\n" + inspect.cleandoc(Pair._doc_inherited)

    _accepted_modes = ("none",)

    def __init__(self, nlist, default_r_cut=None, mode="none"):
        super().__init__(nlist, default_r_cut, 0.0, mode)


class FrictionLJCoulomb(FrictionalPair):
    r"""Coulomb frictional model pair force with the LJ conservative force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    `FrictionLJCoulomb` computes the frictional interaction
    between pairs of particles with a Coulomb friction model
    as described in `Hofmann et. al. 2025`_.

    .. _Hofmann et. al. 2025: https://doi.org/10.48550/arXiv.2507.16388

    The Coulomb friction model is defined by the function

    .. math::

        f_\mathrm{C}(u,r) = w(r)\kappa_\mathrm{f}\, .

    And a repulsive Weeks-Chandler-Anderson (WCA) potential

    .. math::

        \begin{align*}
        U_\mathrm{WCA}(r) &= 4 \varepsilon \left[ \left(
        \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r}
        \right)^{6} \right] \\
        w(r)&=-\frac{\mathrm{d}}{\mathrm{d}r}U_\mathrm{WCA}(r)
        \end{align*}

    Since the frictional force results from coarse-graining the microscopic equations
    of motion, a corresponding pairwise noise that satisfies a fluctuation-dissipation
    relation must be applied. For this a stochastic force and torque is added

    .. math::

        \begin{align*}
        D_\mathrm{C}(u,r) &= \frac{w(r)\kappa_\mathrm{f}\sqrt{\pi}}{\sqrt{2k_\mathrm{B}
                                T\nu}}\mathrm{e}^{\frac{u^2}{2k_\mathrm{B}T\nu}}
                                \mathrm{Erfc}\big(\frac{u}
                                {\sqrt{2k_\mathrm{B}T\nu}} \big) \\
        \mathbf{F}^\mathrm{R}_{ij} = -\mathbf{F}^\mathrm{R}_{ji} &= \sqrt{D_\mathrm{C}
                                    (u^\perp_{ij},r_{ij})}\Big[\mathbf{P}
                                    (\mathbf{\hat{r}}_{ij})\mathbf{\xi}^\mathrm{f}_{ij}
                                    - \mathbf{\hat{r}}_{ij} \times \mathbf{N}^\mathrm{f}
                                    _{ij}\Big] \\
                                    \frac{\mathbf{\tau}^\mathrm{R}_{ij}}{R_i} =
                                    \frac{\mathbf{\tau}^\mathrm{R}_{ji}}{R_j} &=
                                    \sqrt{D_\mathrm{C}(u^\perp_{ij},r_{ij})}\Big[
                                    \mathbf{\hat{r}}_{ij} \times
                                    \mathbf{\xi}^\mathrm{f}_{ij}+\mathbf{P}
                                    (\mathbf{e}_{ij})\mathbf{N}^\mathrm{f}_{ij}\Big]\, ,
        \end{align*}

    where :math:`\nu=(1/m_i+1/m_j)+(R_i^2/\mathbf{I}_i+R_j^2/\mathbf{I}_j))`,
    and :math:`\mathbf{\xi}^\mathrm{f}_{ij}` and :math:`\mathbf{N}^\mathrm{f}_{ij}`
    are three dimensional Gaussian white noise vectors with correlations

    .. math::

        \begin{align*}
        \langle \mathbf{\xi}^\mathrm{f}_{ij}(t) \mathbf{\xi}^\mathrm{f}_{kl}(t')
                                                \rangle &= \mathbf{1}k_\mathrm{B}T
                                                (\delta_{ik}\delta_{jl}-\delta_{il}
                                                \delta_{jk})\delta(t-t') \\
        \langle \mathbf{N}^\mathrm{f}_{ij}(t) \mathbf{N}^\mathrm{f}_{kl}(t') \rangle &=
                                                \mathbf{1}k_\mathrm{B}T(\delta_{ik}
                                                \delta_{jl}+\delta_{il}\delta_{jk})
                                                \delta(t-t')\, .
        \end{align*}

    .. rubric:: Example:

    .. code-block:: python

        coulomb_lj = hoomd.md.pair.friction.FrictionLJCoulomb(nlist=neighbor_list,
                                                                default_r_cut=3)

        coulomb_lj_params = {'epsilon':1, 'sigma':1, 'kappa_f':3, 'kT':1}

        coulomb_lj.params.default = coulomb_lj_params
        simulation.operations.integrator.forces = [coulomb_lj]

    {inherited}
    """

    _cpp_class_name = "FrictionPairFrictionLJCoulomb"
    __doc__ = inspect.cleandoc(__doc__).replace(
        "{inherited}", inspect.cleandoc(FrictionalPair._doc_inherited)
    )

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist, default_r_cut, "none")
        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(
                epsilon=float, sigma=float, kappa_f=float, kT=float, len_keys=2
            ),
        )
        self._add_typeparam(params)


class FrictionLJCoulombNewton(FrictionalPair):
    r"""CoulombNewton frictional model pair force with the LJ conservative force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    `FrictionLJCoulombNewton` computes the frictional interaction
    between pairs of particles with a Coulomb-Newton friction model as described in
    `Hofmann et. al. 2025`_.

    .. _Hofmann et. al. 2025: https://doi.org/10.48550/arXiv.2507.16388

    The Coulomb-Newton friction model is defined by the function

    .. math::

        f_\mathrm{CN}(u,r) = \mathrm{min}[\gamma_\mathrm{f}u,w(r)\kappa_\mathrm{f}]\, .

    And a repulsive Weeks-Chandler-Anderson (WCA) potential

    .. math::

        \begin{align*}
        U_\mathrm{WCA}(r) &= 4 \varepsilon \left[ \left(
        \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r}
        \right)^{6} \right] \\
        w(r)&=-\frac{\mathrm{d}}{\mathrm{d}r}U_\mathrm{WCA}(r)
        \end{align*}

    Since the frictional force results from coarse-graining the microscopic equations
    of motion, a corresponding pairwise noise that satisfies a fluctuation-dissipation
    relation must be applied. For this a stochastic force and torque is added

    .. math::

        \begin{align*}
        D_\mathrm{CN}(u,r) &= \begin{cases}
                                \frac{w(r)\kappa_\mathrm{f}\sqrt{\pi}}
                                {\sqrt{2k_\mathrm{B}T\nu}}\mathrm{e}^{\frac{u^2}
                                {2k_\mathrm{B}T\nu}}\mathrm{Erfc}\big(\frac{u}
                                {\sqrt{2k_\mathrm{B}T\nu}} \big)  & ,\, u\ge
                                \frac{w(r)\kappa_\mathrm{f}}{\gamma_\mathrm{f}} \\
                                \gamma_\mathrm{f}\left(1-\mathrm{e}^{\left(u^2
                                -(\frac{w(r)\kappa_\mathrm{f}}{\gamma_\mathrm{f}})^2
                                \right)/2k_\mathrm{B}T\nu} \right) + \frac{w(r)
                                \kappa_\mathrm{f}\sqrt{\pi}}{\sqrt{2k_\mathrm{B}T\nu}}
                                \mathrm{e}^\frac{u^2}{2k_\mathrm{B}T\nu}\mathrm{Erfc}
                                \left(\frac{w(r)\kappa_\mathrm{f}/\gamma_\mathrm{f}}
                                {\sqrt{2k_\mathrm{B}T\nu}}\right) &,\, u < \frac{w(r)
                                \kappa_\mathrm{f}}{\gamma_\mathrm{f}}
                             \end{cases} \\
        \mathbf{F}^\mathrm{R}_{ij} = -\mathbf{F}^\mathrm{R}_{ji} &= \sqrt{D_\mathrm{CN}
                                        (u^\perp_{ij},r_{ij})}\Big[\mathbf{P}
                                        (\mathbf{\hat{r}}_{ij})
                                        \mathbf{\xi}^\mathrm{f}_{ij} -
                                        \mathbf{\hat{r}}_{ij} \times
                                        \mathbf{N}^\mathrm{f}_{ij}\Big] \\
        \frac{\mathbf{\tau}^\mathrm{R}_{ij}}{R_i} = \frac{\mathbf{\tau}^\mathrm{R}_{ji}}
                                        {R_j} &= \sqrt{D_\mathrm{CN}
                                        (u^\perp_{ij},r_{ij})}\Big[\mathbf{\hat{r}}_{ij}
                                        \times \mathbf{\xi}^\mathrm{f}_{ij}+\mathbf{P}
                                        (\mathbf{e}_{ij})\mathbf{N}^\mathrm{f}_{ij}
                                        \Big]\, ,
        \end{align*}

    where :math:`\nu=(1/m_i+1/m_j)+(R_i^2/\mathbf{I}_i+R_j^2/\mathbf{I}_j))`, and
    :math:`\mathbf{\xi}^\mathrm{f}_{ij}` and :math:`\mathbf{N}^\mathrm{f}_{ij}` are
    three dimensional Gaussian white noise vectors with correlations

    .. math::

        \begin{align*}
        \langle \mathbf{\xi}^\mathrm{f}_{ij}(t) \mathbf{\xi}^\mathrm{f}_{kl}(t')
                                                \rangle &= \mathbf{1}k_\mathrm{B}T
                                                (\delta_{ik}\delta_{jl}-\delta_{il}
                                                \delta_{jk})\delta(t-t') \\
        \langle \mathbf{N}^\mathrm{f}_{ij}(t) \mathbf{N}^\mathrm{f}_{kl}(t') \rangle &=
                                                \mathbf{1}k_\mathrm{B}T
                                                (\delta_{ik}\delta_{jl}+\delta_{il}
                                                \delta_{jk})\delta(t-t')\, .
        \end{align*}

    .. rubric:: Example:

    .. code-block:: python

        coulombNewton_lj = hoomd.md.pair.friction.FrictionLJCoulombNewton(
                                                                nlist=neighbor_list,
                                                                default_r_cut=3)

        coulombNewton_lj_params = { 'epsilon':1, 'sigma':1, 'gamma_f':1, 'kappa_f':3,
                                    'kT':1}

        coulombNewton_lj.params.default = coulombNewton_lj_params
        simulation.operations.integrator.forces = [coulombNewton_lj]

    {inherited}
    """

    _cpp_class_name = "FrictionPairFrictionLJCoulombNewton"
    __doc__ = inspect.cleandoc(__doc__).replace(
        "{inherited}", inspect.cleandoc(FrictionalPair._doc_inherited)
    )

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist, default_r_cut, "none")
        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(
                epsilon=float,
                sigma=float,
                gamma_f=float,
                kappa_f=float,
                kT=float,
                len_keys=2,
            ),
        )
        self._add_typeparam(params)


class FrictionLJLinear(FrictionalPair):
    r"""Linear frictional model pair force with the LJ conservative force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    `FrictionLJLinear` computes the frictional interaction
    between pairs of particles with a linear friction model as described in
    `Hofmann et. al. 2025`_.

    .. _Hofmann et. al. 2025: https://doi.org/10.48550/arXiv.2507.16388

    The linear friction model is defined by the function

    .. math::

        f_\mathrm{l}(u,r) = w(r)\gamma_\mathrm{f}u\, .

    And a repulsive Weeks-Chandler-Anderson (WCA) potential

    .. math::

        \begin{align*}
        U_\mathrm{WCA}(r) &= 4 \varepsilon \left[ \left(
        \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r}
        \right)^{6} \right] \\
        w(r)&=-\frac{\mathrm{d}}{\mathrm{d}r}U_\mathrm{WCA}(r)
        \end{align*}

    Since the frictional force results from coarse-graining the microscopic equations
    of motion, a corresponding pairwise noise that satisfies a fluctuation-dissipation
    relation must be applied. For this a stochastic force and torque is added

    .. math::

        \begin{align*}
        D_\mathrm{l}(u,r) &= w(r)\gamma_\mathrm{f} \\
        \mathbf{F}^\mathrm{R}_{ij} = -\mathbf{F}^\mathrm{R}_{ji} &= \sqrt{D_\mathrm{l}
                                    (u^\perp_{ij},r_{ij})}\Big[\mathbf{P}
                                    (\mathbf{\hat{r}}_{ij})\mathbf{\xi}^\mathrm{f}_{ij}
                                    - \mathbf{\hat{r}}_{ij} \times
                                    \mathbf{N}^\mathrm{f}_{ij}\Big] \\
        \frac{\mathbf{\tau}^\mathrm{R}_{ij}}{R_i} = \frac{\mathbf{\tau}^\mathrm{R}_{ji}}
                                        {R_j} &= \sqrt{D_\mathrm{l}
                                        (u^\perp_{ij},r_{ij})}\Big[\mathbf{\hat{r}}_{ij}
                                        \times \mathbf{\xi}^\mathrm{f}_{ij}+\mathbf{P}
                                        (\mathbf{e}_{ij})\mathbf{N}^\mathrm{f}_{ij}
                                        \Big]\, ,
        \end{align*}

    where :math:`\nu=(1/m_i+1/m_j)+(R_i^2/\mathbf{I}_i+R_j^2/\mathbf{I}_j))`, and
    :math:`\mathbf{\xi}^\mathrm{f}_{ij}` and :math:`\mathbf{N}^\mathrm{f}_{ij}` are
    three dimensional Gaussian white noise vectors with correlations

    .. math::

        \begin{align*}
        \langle \mathbf{\xi}^\mathrm{f}_{ij}(t) \mathbf{\xi}^\mathrm{f}_{kl}(t')
                                                \rangle &= \mathbf{1}k_\mathrm{B}T
                                                (\delta_{ik}\delta_{jl}-\delta_{il}
                                                \delta_{jk})\delta(t-t') \\
        \langle \mathbf{N}^\mathrm{f}_{ij}(t) \mathbf{N}^\mathrm{f}_{kl}(t') \rangle &=
                                                 \mathbf{1}k_\mathrm{B}T(\delta_{ik}
                                                 \delta_{jl}+\delta_{il}\delta_{jk})
                                                 \delta(t-t')\, .
        \end{align*}

    .. rubric:: Example:

    .. code-block:: python

        linear_lj = hoomd.md.pair.friction.FrictionLJLinear(nlist=neighbor_list,
                                                                default_r_cut=3)

        linear_lj_params = {'epsilon':1, 'sigma':1, 'gamma_f':1, 'kT':1}

        linear_lj.params.default = linear_lj_params
        simulation.operations.integrator.forces = [linear_lj]

    {inherited}
    """

    _cpp_class_name = "FrictionPairFrictionLJLinear"
    __doc__ = inspect.cleandoc(__doc__).replace(
        "{inherited}", inspect.cleandoc(FrictionalPair._doc_inherited)
    )

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist, default_r_cut, "none")
        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(
                epsilon=float, sigma=float, gamma_f=float, kT=float, len_keys=2
            ),
        )
        self._add_typeparam(params)


__all__ = [
    "FrictionLJCoulomb",
    "FrictionLJCoulombNewton",
    "FrictionLJLinear",
    "FrictionalPair",
]
