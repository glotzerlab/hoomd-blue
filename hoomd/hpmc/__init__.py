# -*- coding: iso-8859-1 -*-
# this file exists to mark this directory as a python module

R""" Hard particle Monte Carlo

HPMC performs hard particle Monte Carlo simulations of a variety of classes of shapes.

.. rubric:: Overview

HPMC implements hard particle Monte Carlo in HOOMD-blue.

.. rubric:: Timestep definition

HOOMD-blue started as an MD code where **timestep** has a clear meaning. MC simulations are run
for timesteps. In exact terms, this means different things on the CPU and GPU and something slightly different when
using MPI. The behavior is approximately normalized so that user scripts do not need to drastically change
run() lengths when switching from one execution resource to another.

In the GPU implementation, one trial move is applied to a number of randomly chosen particles in each cell during one
timestep. The number of selected particles is ``nselect*ceil(avg particles per cell)`` where *nselect* is a user-chosen
parameter. The default value of *nselect* is 4, which achieves optimal performance for a wide variety of benchmarks.
Detailed balance is obeyed at the level of a timestep. In short: One timestep **is NOT equal** to one sweep,
but is approximately *nselect* sweeps, which is an overestimation.

In the single-threaded CPU implementation, one trial move is applied *nselect* times to each of the *N* particles
during one timestep. In parallel MPI runs, one trial moves is applied *nselect* times to each particle in the active
region. There is a small strip of inactive region near the boundaries between MPI ranks in the domain decomposition.
The trial moves are performed in a shuffled order so detailed balance is obeyed at the level of a timestep.
In short: One timestep **is approximately** *nselect* sweeps (*N* trial moves). In single-threaded runs, the
approximation is exact, but it is slightly underestimated in MPI parallel runs.

To approximate a fair comparison of dynamics between CPU and GPU timesteps, log the ``hpmc_sweep``
quantity to get the number sweeps completed so far at each logged timestep.

See `J. A. Anderson et. al. 2016 <http://dx.doi.org/10.1016/j.cpc.2016.02.024>`_ for design and implementation details.

.. rubric:: Depletants

HPMC supports integration with implicit depletants. *Depletants* are shapes that do not interact between themselves, but have
a finite excluded volume with respect to other particles (the *colloids*). Their ideal gas nature makes it possible to randomly insert
depletants into the overlap regions between the colloids, according to a Poisson point process to sample from the grand-canonical
ensemble. This insertion is efficiently performed in parallel on the CPU, using TBB when it is enabled (see
:doc:`Installation Guide </installation>`), or on the GPU.

Details on the depletant capability are documented in `J. Glaser et al. 2015 <https://doi.org/10.1063/1.4935175>`_, and
Glaser, to be published (2019).

Since release 3.0 HOOMD-blue supports *quermass integration*, which is a method
to define the excluded volume of the colloids independently from that of the
test particles. Every colloid is swept by a sphere of constant radius
**r_sweep** (see ``hoomd.hpmc.integrate.mode_hpmc.set_params``), similar
to implicit depletants with a spherical depletant. However, the test particle
(or mixture thereof) now intersects the region of intersection between the
sphere-swept colloids, as illustrated below. The name 'Quermass integration' of
the method emphasizes the fact that test particles of arbitrary shape and in
particular, convex test particles of arbitrary geometric measures (volume,
surface area, integrated mean and Gaussian curvature -- the four Minkowski measures in
three dimensions) can be used to realize a free energy functional that depends
on the corresponding measures of the system of particles in a general way. The
coefficients can have any sign, e.g. negative coefficients are realized
by negative test particle fugacities (see ``hoomd.hpmc.integrate.mode_hpmc.set_fugacity``).

.. image:: quermass.png
    :width: 450 px
    :align: center
    :alt: Cell list schematic

.. rubric:: Stability

:py:mod:`hoomd.hpmc` is **stable**. When upgrading from version 2.x to 2.y (y > x),
existing job scripts that follow *documented* interfaces for functions and classes
will not require any modifications. **Maintainer:** Joshua A. Anderson
"""

# need to import all submodules defined in this directory
from hoomd.hpmc import integrate
from hoomd.hpmc import update
from hoomd.hpmc import compute
from hoomd.hpmc import util
from hoomd.hpmc import field
from hoomd.hpmc import tune
