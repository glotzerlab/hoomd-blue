# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

R""" Multiparticle collision dynamics.

Simulating complex fluids and soft matter using conventional molecular dynamics
methods (:py:mod:`hoomd.md`) can be computationally demanding due to large
disparities in the relevant length and time scales between molecular-scale
solvents and mesoscale solutes such as polymers, colloids, and deformable
materials like cells. One way to overcome this challenge is to simplify the model
for the solvent while retaining its most important interactions with the solute.
MPCD is a particle-based simulation method for resolving solvent-mediated
fluctuating hydrodynamic interactions with a microscopically detailed solute
model. This method has been successfully applied to a simulate a broad class
of problems, including polymer solutions and colloidal suspensions both in and
out of equilibrium.

.. rubric:: Algorithm

In MPCD, the solvent is represented by point particles having continuous
positions and velocities. The solvent particles propagate in alternating
streaming and collision steps. During the streaming step, particles evolve
according to Newton's equations of motion. Typically, no external forces are
applied to the solvent, and streaming is straightforward with a large time step.
Particles are then binned into local cells and undergo a stochastic multiparticle
collision within the cell. Collisions lead to the build up of hydrodynamic
interactions, and the frequency and nature of the collisions, along with the
solvent properties, determine the transport coefficients. All standard collision
rules conserve linear momentum within the cell and can optionally be made to
enforce angular-momentum conservation. Currently, we have implemented
the following collision rules with linear-momentum conservation only:

    * :py:obj:`~hoomd.mpcd.collide.srd` -- Stochastic rotation dynamics
    * :py:obj:`~hoomd.mpcd.collide.at` -- Andersen thermostat

Solute particles can be coupled to the solvent during the collision step. This
is particularly useful for soft materials like polymers. Standard molecular
dynamics integration can be applied to the solute. Coupling to the MPCD
solvent introduces both hydrodynamic interactions and a heat bath that acts as
a thermostat. In the future, fluid-solid coupling will also be introduced during
the streaming step to couple hard particles and boundaries.

Details of this implementation of the MPCD algorithm for HOOMD-blue can be found
in Howard et al. (2017).

.. rubric:: Getting started

MPCD is intended to be used as an add-on to the standard MD methods in
:py:mod:`hoomd.md`. To get started, take the following steps:

    1. Initialize any solute particles using standard methods (:py:mod:`hoomd.init`).
    2. Initialize the MPCD solvent particles using one of the methods in
       :py:mod:`.mpcd.init`. Additional details on how to manipulate the solvent
       particle data can be found in :py:mod:`.mpcd.data`.
    3. Setup an MD integrator and any interactions between solute particles.
    4. Setup an :py:obj:`~hoomd.mpcd.integrate.integrator` for the MPCD particles.
    5. Choose the appropriate collision rule from :py:mod:`.mpcd.collide`, and set
       the collision rule parameters. If necessary, adjust the MPCD cell size.
    6. Optionally, configure the sorting frequency to improve performance (see
       :py:obj:`update.sort`).
    7. Run your simulation!

.. rubric:: Stability

:py:mod:`hoomd.mpcd` is currently **unstable**. (It is under development.) When
upgrading from version 2.x to 2.y (y > x), existing job scripts may need to be
updated.

**Maintainer:** Michael P. Howard, Princeton University.
"""

# these imports are necessary in order to link derived types between modules
from hoomd import _hoomd
from hoomd.md import _md

from hoomd.mpcd import collide
from hoomd.mpcd import data
from hoomd.mpcd import init
from hoomd.mpcd import integrate
from hoomd.mpcd import update

# pull the integrator into the main module namespace for convenience
# (we want to type mpcd.integrator not mpcd.integrate.integrator)
from hoomd.mpcd.integrate import integrator
