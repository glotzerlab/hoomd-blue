# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Hard particle Monte Carlo.

In hard particle Monte Carlo (HPMC) simulations, the particles in the system
state have extended shapes. The potential energy of the system is infinite when
any particle shapes overlap. Pair (:doc:`module-hpmc-pair`) and external
(:doc:`module-hpmc-external`) potentials compute the potential energy when there
are no shape overlaps. `hpmc` employs the Metropolis Monte Carlo algorithm to
sample equilibrium configurations of the system.

To perform HPMC simulations, assign a HPMC integrator (`hoomd.hpmc.integrate`)
to the `hoomd.Simulation` operations. The HPMC integrator defines the particle
shapes and performs local trial moves on the particle positions and
orientations. HPMC updaters (`hoomd.hpmc.update`) interoperate with the
integrator to perform additional types of trial moves, including box moves,
cluster moves, and particle insertion/removal. Use HPMC computes
(`hoomd.hpmc.compute`) to compute properties of the system state, such as the
free volume or pressure.

See Also:
    `Anderson 2016 <https://dx.doi.org/10.1016/j.cpc.2016.02.024>`_ further
    describes the theory and implementation.
"""

# need to import all submodules defined in this directory
from hoomd.hpmc import integrate
from hoomd.hpmc import update
from hoomd.hpmc import compute
from hoomd.hpmc import tune
from hoomd.hpmc import pair
from hoomd.hpmc import external
from hoomd.hpmc import nec
from hoomd.hpmc import shape_move
