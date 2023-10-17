# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r""" Multiparticle collision dynamics.

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
in `M. P. Howard et al. (2018) <https://doi.org/10.1016/j.cpc.2018.04.009>`_.

.. rubric:: Getting started

MPCD is intended to be used as an add-on to the standard MD methods in
:py:mod:`hoomd.md`. To get started, take the following steps:

    1. Initialize any solute particles using standard methods (`Simulation`).
    2. Initialize the MPCD solvent particles using one of the methods in
       :py:mod:`.mpcd.init`. Additional details on how to manipulate the solvent
       particle data can be found in :py:mod:`.mpcd.data`.
    3. Create an MPCD :py:obj:`~hoomd.mpcd.integrator`.
    4. Choose the appropriate streaming method from :py:mod:`.mpcd.stream`.
    5. Choose the appropriate collision rule from :py:mod:`.mpcd.collide`, and set
       the collision rule parameters.
    6. Setup an MD integrator and any interactions between solute particles.
    7. Optionally, configure the sorting frequency to improve performance (see
       :py:obj:`update.sort`).
    8. Run your simulation!

Example script for a pure bulk SRD fluid::

    import hoomd
    hoomd.context.initialize()
    from hoomd import mpcd

    # Initialize (empty) solute in box.
    box = hoomd.data.boxdim(L=100.)
    hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=box))

    # Initialize MPCD particles and set sorting period.
    s = mpcd.init.make_random(N=int(10*box.get_volume()), kT=1.0, seed=7)
    s.sorter.set_period(period=25)

    # Create MPCD integrator with streaming and collision methods.
    mpcd.integrator(dt=0.1)
    mpcd.stream.bulk(period=1)
    mpcd.collide.srd(seed=42, period=1, angle=130., kT=1.0)

    hoomd.run(2000)


"""

# these imports are necessary in order to link derived types between modules
import hoomd
from hoomd import _hoomd
from hoomd.md import _md

from hoomd.mpcd import collide
from hoomd.mpcd import force
from hoomd.mpcd import integrate
from hoomd.mpcd import stream
from hoomd.mpcd import update
