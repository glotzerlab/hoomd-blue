# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Newtonian event chain Monte Carlo.

The integrators in `hoomd.hpmc.nec` implement Newtonian event chain Monte Carlo
as described in `Klement 2021 <https://doi.org/10.1021/acs.jctc.1c00311>`__.

Newtonian event chain Monte Carlo combines rejection free particle chain
translation moves with traditional local rotation trial moves to explore phase
space. This combination can lead to more efficient simulations. See the paper
for a full description of the method.

A chain move is rejection free and changes the positions of many particles, but
does not change their orientations. Mix one chain move with many rotation moves
to effectively explore phase space. The author suggests setting
``chain_probability`` to a low value, such as 0.1. As with traditional HPMC,
tune the maximum rotation move size ``a`` to achieve a target acceptance ratio,
such as 20%.

Important:
    Chain moves translate particles along their velocity vectors. You must set
    a non-zero velocity for every particle in the simulation state to use NEC.
    For example, start with a thermal distribution using:

    .. code::

        sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT=1)

The ``chain_time`` parameter determines the total length of each chain move.
Like the ``d`` parameter in HPMC with local trial moves, the value of
``chain_time`` greatly impacts the rate at which the simulation explores phase
space. See `Klement 2021 <https://doi.org/10.1021/acs.jctc.1c00311>`__ for a
full description on how to choose ``chain_time`` optimally. As a starting point,
use `hoomd.hpmc.nec.tune.ChainTime` to attain an average of 20 particles per
chain move.

The NEC method uses the ``d`` parameter as a search radius for collisions and
updates the translation move acceptance counter when it finds a collision
within the distance ``d``. Changing ``d`` will not change the chain moves NEC
makes, but it will adjust the wall time needed to complete the moves. Set ``d``
too high and performance will slow due to many narrow phase collisions checks.
See ``d`` too low and performance will slow due to many broad phase collision
checks. Adjust ``d`` to obtain optimal performance. The code author suggests
tuning ``d`` to an "acceptance ratio" of 10% as a starting point.

See Also:
    `hoomd.hpmc.integrate`
"""

from . import integrate
from . import tune
