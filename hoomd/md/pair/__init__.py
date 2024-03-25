# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Pair Potentials for molecular dynamics.

Pair force classes apply a force and virial on every particle in the
simulation state commensurate with the potential energy:

.. math::

    U_\mathrm{pair,total} = \frac{1}{2} \sum_{i=0}^\mathrm{N_particles-1}
                      \sum_{j \ne i, (i,j) \notin \mathrm{exclusions}}
                      U_\mathrm{pair}(r_{ij})

where :math:`\vec{r}_{ij} = \mathrm{minimum\_image}(\vec{r}_j - \vec{r}_i)`.
`Pair` applies a short range cutoff using a `hoomd.md.nlist.NeighborList` for
performance and assumes that both :math:`U(r)` and its derivatives are 0 when
:math:`r_{ij} \ge r_\mathrm{cut}`. `Pair` also ignores particle pairs that are
excluded in the neighbor list.

Specifically, the force :math:`\vec{F}` on each pair of particles :math:`i,j`
is:

.. math::

    \vec{F} =
    \begin{cases}
    -\nabla U_\mathrm{pair}(r) & r < r_{\mathrm{cut}} \\
    0 & r \ge r_{\mathrm{cut}} \\
    \end{cases}

where the cutoff radius :math:`r_{\mathrm{cut}}` is given by `Pair.r_cut`.

Tip:
    Set `Pair.r_cut` to 0 to skip computations for non-interacting pairs.

`Pair` splits half the energy from each pair interaction onto particles
:math:`i` and :math:`j`:

.. math::

    U_i = \frac{1}{2} \sum_{j \ne i, (i,j) \notin \mathrm{exclusions}}
    U_\mathrm{pair}(r_{ij})
    [r_{ij} < r_\mathrm{cut}]

and similarly for virials.

.. rubric:: Shifting/smoothing mode

The function :math:`U_\mathrm{pair}(r)` depends on the chosen form of the pair
potential :math:`U(r)` (by the `Pair` subclass)  and the mode (`Pair.mode`):

.. math::
    U_\mathrm{pair}(r) =
    \begin{cases}
    U_(r) & \mathrm{mode\ is\ \mathrm{none}} \\
    U(r) - U(r_{\mathrm{cut}}) & \mathrm{mode\ is\ shift} \\
    S(r) \cdot U_(r) & \mathrm{mode\ is\ xplor}
    \land r_{\mathrm{on}} < r_{\mathrm{cut}} \\
    U(r) - U(r_{\mathrm{cut}})
    & \mathrm{mode\ is\ xplor} \land r_{\mathrm{on}} \ge r_{\mathrm{cut}}
    \end{cases}

where :math:`S(r)` is the XPLOR smoothing function:

.. math::
    S(r) =
    \begin{cases}
     1 & r < r_{\mathrm{on}} \\
     \frac{(r_{\mathrm{cut}}^2 - r^2)^2 \cdot
     (r_{\mathrm{cut}}^2 + 2r^2 -
     3r_{\mathrm{on}}^2)}{(r_{\mathrm{cut}}^2 -
     r_{\mathrm{on}}^2)^3}
     & r_{\mathrm{on}} \le r \le r_{\mathrm{cut}} \\
     0 & r > r_{\mathrm{cut}} \\
     \end{cases}

where :math:`r_{\mathrm{on}}` is given by `Pair.r_on`.

The XPLOR smoothing function :math:`S(r)` ensures that both the potential energy
and the force going smoothly to 0 at :math:`r = r_{\mathrm{cut}}`, reducing the
rate of energy drift in long simulations. :math:`r_{\mathrm{on}}` controls the
point at which the smoothing starts. Set it to modify only the tail of the
potential. The WCA potential and it's first derivative already go smoothly to 0
at the cutoff, so there is no need to apply the smoothing function. In such
mixed systems, set :math:`r_{\mathrm{on}}` to a value greater than
:math:`r_{\mathrm{cut}}` for those pairs that interact via WCA in order to
enable shifting of the WCA potential to 0 at the cutoff.

.. rubric:: Tail correction

Some pair potentials can optionally apply isotropic integrated long range tail
corrections when the ``tail_correction`` parameter is ``True``. These
corrections are only valid when the shifting/smoothing mode is set to
``"none"``. Following `Sun 1998 <https://doi.org/10.1021/jp980939v>`_, the
pressure and energy corrections :math:`\Delta P` and :math:`\Delta E` are given
by:

.. math::

    \Delta P = \frac{-2\pi}{3} \sum_{i=1}^{n} \rho_i \sum_{j=1}^{n} \rho_j
    \int_{r_\mathrm{cut}}^{\infty} \left( r
    \frac{\mathrm{d}U_{ij}(r)}{\mathrm{d}r} \right) r^2 \mathrm{d}r

and

.. math::

    \Delta E = 2\pi \sum_{i=1}^{n} N_i \sum_{j=1}^{n} \rho_j
    \int_{r_\mathrm{cut}}^{\infty} U_{ij}(r) r^2 \mathrm{d}r,

where :math:`n` is the number of unique particle types in the system,
:math:`\rho_i` is the number density of particles of type :math:`i` in the
system, :math:`U_{ij}(r)` is the pair potential between particles of type
:math:`i` and :math:`j`, and :math:`N_i` is the number of particles of type
:math:`i` in the system. These expressions assume that the radial pair
distribution functions :math:`g_{ij}(r)` are unity at the cutoff and beyond.

The pressure shift :math:`\Delta P` appears in the additional virial term
:math:`W_\mathrm{additional}` (`Force.additional_virial`) and the energy shift
appears in the additional energy :math:`U_\mathrm{additional}`
(`Force.additional_energy`).

Warning:
    The value of the tail corrections depends on the number of each type of
    particle in the system, and these are precomputed when the pair
    potential object is initialized. If the number of any of the types of
    particles changes, the tail corrections will yield invalid results.

.. rubric:: Anisotropic potentials

For anisotropic potentials see `hoomd.md.pair.aniso`
"""

from . import aniso
from .pair import (
    Pair,
    LJ,
    Gaussian,
    ExpandedLJ,
    ExpandedGaussian,
    Yukawa,
    Ewald,
    Morse,
    DPD,
    DPDConservative,
    DPDLJ,
    ForceShiftedLJ,
    Moliere,
    ZBL,
    Mie,
    ExpandedMie,
    ReactionField,
    DLVO,
    Buckingham,
    LJ1208,
    LJ0804,
    Fourier,
    OPP,
    Table,
    TWF,
    LJGauss,
)
