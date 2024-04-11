# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Multiparticle collision dynamics.

Simulating complex fluids and soft matter using conventional molecular dynamics
methods (`hoomd.md`) can be computationally demanding due to large disparities
in the relevant length and time scales between molecular-scale solvents and
mesoscale solutes such as polymers, colloids, or cells. One way to overcome this
challenge is to simplify the model for the solvent while retaining its most
important interactions with the solute. MPCD is a particle-based simulation
method for resolving solvent-mediated fluctuating hydrodynamic interactions with
a microscopically detailed solute model.

.. rubric:: Algorithm

In MPCD, a fluid is represented by point particles having continuous
positions and velocities. The MPCD particles propagate in alternating
streaming and collision steps. During the streaming step, particles evolve
according to Newton's equations of motion. Particles are then binned into local
cells and undergo a stochastic collision within the cell. Collisions lead to the
build up of hydrodynamic interactions, and the frequency and nature of the
collisions, along with the fluid properties, determine the transport
coefficients. All standard collision rules conserve linear momentum within the
cell and can optionally be made to enforce angular-momentum conservation.
Currently, we have implemented the following collision rules with
linear-momentum conservation only:

* :class:`~hoomd.mpcd.collide.StochasticRotationDynamics`
* :class:`~hoomd.mpcd.collide.AndersenThermostat`

Solute particles can be coupled to the MPCD particles during the collision step.
This is particularly useful for soft materials like polymers. Standard molecular
dynamics methods can be applied to the solute. Coupling to the MPCD particles
introduces both hydrodynamic interactions and a heat bath that acts as a
thermostat.

The MPCD particles can additionally be coupled to solid boundaries (with no-slip
or slip boundary conditions) during the streaming step.

Details of HOOMD-blue's implementation of the MPCD algorithm can be found
in `M. P. Howard et al. (2018) <https://doi.org/10.1016/j.cpc.2018.04.009>`_.
Note, though, that continued improvements to the code may cause some deviations.

.. rubric:: Getting started

MPCD is intended to be used as an add-on to the standard MD methods in
`hoomd.md`. Getting started can look like:

1. Initialize the MPCD particles through `Snapshot.mpcd`. You can include any
   solute particles in the snapshot as well.
2. Create the MPCD `Integrator`. Setup solute particle integration methods
   and interactions as you normally would to use `hoomd.md`.
3. Choose the streaming method from :mod:`.mpcd.stream`.
4. Choose the collision rule from :mod:`.mpcd.collide`. Couple the solute to the
   collision step.
5. Run your simulation!


"""

from hoomd.mpcd import collide
from hoomd.mpcd import fill
from hoomd.mpcd import force
from hoomd.mpcd import geometry
from hoomd.mpcd import integrate
from hoomd.mpcd.integrate import Integrator
from hoomd.mpcd import methods
from hoomd.mpcd import stream
from hoomd.mpcd import tune
