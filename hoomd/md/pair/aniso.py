# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Anisotropic potentials."""

import json

from hoomd.md.pair.pair import Pair
from hoomd.logging import log
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter


class AnisotropicPair(Pair):
    r"""Generic anisotropic pair potential.

    Users should not instantiate `AnisotropicPair` directly. It is a base
    class that provides common features to all anisotropic pair forces.
    All anisotropic pair potential commands specify that a given potential
    energy, force and torque be computed on all non-excluded particle pairs in
    the system within a short range cutoff distance :math:`r_{\mathrm{cut}}`.
    The interaction energy, forces and torque depend on the inter-particle
    separation :math:`\vec r` and on the orientations :math:`\vec q_i`,
    :math:`q_j`, of the particles.

    `AnisotropicPair` is similar to `hoomd.md.pair.Pair` except it does not
    support the ``'xplor'`` shifting mode or `r_on`.

    Args:
        nlist (hoomd.md.nlist.NList) : The neighbor list.
        default_r_cut (`float`, optional) : The default cutoff for the
            potential, defaults to ``None`` which means no cutoff
            :math:`[\mathrm{length}]`.
        mode (`str`, optional) : the energy shifting mode, defaults to "none".
    """

    _accepted_modes = ("none", "shift")

    def __init__(self, nlist, default_r_cut=None, mode="none"):
        super().__init__(nlist, default_r_cut, 0.0, mode)

    def _return_type_shapes(self):
        type_shapes = self.cpp_force.getTypeShapesPy()
        ret = [json.loads(json_string) for json_string in type_shapes]
        return ret


class Dipole(AnisotropicPair):
    r"""Screened dipole-dipole interactions.

    Implements the force and energy calculations for both magnetic and
    electronic dipole-dipole interactions. When particles have charge as well as
    a dipole moment, the interactions are through electronic dipole moments. If
    the particles have no charge then the interaction is through magnetic or
    electronic dipoles. Note whether a dipole is magnetic or electronic does not
    change the functional form of the potential only the units associated with
    the potential parameters.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        mode (str): energy shifting/smoothing mode

    `Dipole` computes the (screened) interaction between pairs of
    particles with dipoles and electrostatic charges. The total energy
    computed is:

    .. math::

        U_{dipole} = U_{dd} + U_{de} + U_{ee}

        U_{dd} = A e^{-\kappa r}
            \left(\frac{\vec{\mu_i}\cdot\vec{\mu_j}}{r^3}
                  - 3\frac{(\vec{\mu_i}\cdot \vec{r_{ji}})
                           (\vec{\mu_j}\cdot \vec{r_{ji}})}
                          {r^5}
            \right)

        U_{de} = A e^{-\kappa r}
            \left(\frac{(\vec{\mu_j}\cdot \vec{r_{ji}})q_i}{r^3}
                - \frac{(\vec{\mu_i}\cdot \vec{r_{ji}})q_j}{r^3}
            \right)

        U_{ee} = A e^{-\kappa r} \frac{q_i q_j}{r}

    See `hoomd.md.pair.Pair` for details on how forces are calculated and the
    available energy shifting and smoothing modes.  Use ``params`` dictionary to
    set potential coefficients. The coefficients must be set per unique pair of
    particle types.

    Note:
       All units are given for electronic dipole moments.

    .. py:attribute:: params

        The dipole potential parameters. The dictionary has the following
        keys:

        * ``A`` (`float`, **required**) - :math:`A` - electrostatic energy
          scale (*default*: 1.0)
          :math:`[\mathrm{energy} \cdot \mathrm{length} \cdot
          \mathrm{charge}^{-2}]`
        * ``kappa`` (`float`, **required**) - :math:`\kappa` - inverse
          screening length :math:`[\mathrm{length}^{-1}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mu

        :math:`\mu` - the magnetic magnitude of the particle local reference
        frame as a tuple (i.e. :math:`(\mu_x, \mu_y, \mu_z)`)
        :math:`[\mathrm{charge} \cdot \mathrm{length}]`.

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float` ]]

    Example::

        nl = nlist.Cell()
        dipole = md.pair.Dipole(nl, default_r_cut=3.0)
        dipole.params[('A', 'B')] = dict(A=1.0, kappa=4.0)
        dipole.mu['A'] = (4.0, 1.0, 0.0)
    """
    _cpp_class_name = "AnisoPotentialPairDipole"

    def __init__(self, nlist, default_r_cut=None, mode='none'):
        super().__init__(nlist, default_r_cut, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(A=float, kappa=float, len_keys=2))
        mu = TypeParameter('mu', 'particle_types',
                           TypeParameterDict((float, float, float), len_keys=1))
        self._extend_typeparam((params, mu))


class GayBerne(AnisotropicPair):
    r"""Gay-Berne anisotropic pair potential.

    Warning: The code has yet to be updated to the current API.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        mode (str): energy shifting/smoothing mode.

    `GayBerne` computes the Gay-Berne potential between anisotropic
    particles.

    This version of the Gay-Berne potential supports identical pairs of uniaxial
    ellipsoids, with orientation-independent energy-well depth. The potential
    comes from the following paper Allen et. al. 2006 `paper link`_.

    .. _paper link: http://dx.doi.org/10.1080/00268970601075238

    The interaction energy for this anisotropic pair potential is

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{GB}}(\vec r, \vec e_i, \vec e_j)
            = & 4 \varepsilon \left[ \zeta^{-12} - \zeta^{-6} \right];
            & \zeta < \zeta_{\mathrm{cut}} \\
            = & 0; & \zeta \ge \zeta_{\mathrm{cut}} \\
        \end{eqnarray*}

    .. math::

        \zeta = \left(\frac{r-\sigma+\sigma_{\mathrm{min}}}
                           {\sigma_{\mathrm{min}}}\right)

        \sigma^{-2} = \frac{1}{2} \hat{\vec{r}}
            \cdot \vec{H^{-1}} \cdot \hat{\vec{r}}

        \vec{H} = 2 \ell_\perp^2 \vec{1}
            + (\ell_\parallel^2 - \ell_\perp^2)
              (\vec{e_i} \otimes \vec{e_i} + \vec{e_j} \otimes \vec{e_j})

    with :math:`\sigma_{\mathrm{min}} = 2 \min(\ell_\perp, \ell_\parallel)`.

    The cut-off parameter :math:`r_{\mathrm{cut}}` is defined for two particles
    oriented parallel along the **long** axis, i.e.
    :math:`\zeta_{\mathrm{cut}} = \left(\frac{r-\sigma_{\mathrm{max}}
    + \sigma_{\mathrm{min}}}{\sigma_{\mathrm{min}}}\right)`
    where :math:`\sigma_{\mathrm{max}} = 2 \max(\ell_\perp, \ell_\parallel)` .

    The quantities :math:`\ell_\parallel` and :math:`\ell_\perp` denote the
    semi-axis lengths parallel and perpendicular to particle orientation.

    Use ``params`` dictionary to set potential coefficients. The coefficients
    must be set per unique pair of particle types.

    .. py:attribute:: params

        The Gay-Berne potential parameters. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) - :math:`\varepsilon`
          :math:`[\mathrm{energy}]`
        * ``lperp`` (`float`, **required**) - :math:`\ell_\perp`
          :math:`[\mathrm{length}]`
        * ``lpar`` (`float`, **required**) -  :math:`\ell_\parallel`
          :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]


    Example::

        nl = nlist.Cell()
        gay_berne = md.pair.GayBerne(nlist=nl, default_r_cut=2.5)
        gay_berne.params[('A', 'A')] = dict(epsilon=1.0, lperp=0.45, lpar=0.5)
        gay_berne.r_cut[('A', 'B')] = 2 ** (1.0 / 6.0)

    """
    _cpp_class_name = "AnisoPotentialPairGB"

    def __init__(self, nlist, default_r_cut=None, mode='none'):
        super().__init__(nlist, default_r_cut, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float,
                              lperp=float,
                              lpar=float,
                              len_keys=2))
        self._add_typeparam(params)

    @log(category="object")
    def type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Example:

            >>> gay_berne.type_shapes
            [{'type': 'Ellipsoid', 'a': 1.0, 'b': 1.0, 'c': 1.5}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super()._return_type_shapes()
