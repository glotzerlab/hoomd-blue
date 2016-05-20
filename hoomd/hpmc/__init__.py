# -*- coding: iso-8859-1 -*-
# this file exists to mark this directory as a python module

R""" Hard particle Monte Carlo

HPMC performs hard particle Monte Carlo simulations of a variety of classes of shapes.

.. rubric:: Overview

HPMC implements hard particle Monte Carlo in HOOMD-blue. It supports:

- Dimensions: 2D and 3D
- Box shape: triclinic
- Ensembles:
    - NVT
    - NPT (:py:class:`update.npt`)
    - Implicit depletants
    - Grand canonical ensemble (:py:class:`update.muvt`)
    - Gibbs ensemble (:py:class:`update.muvt`)
- Shapes:
    - Spheres / disks (:py:class:`integrate.sphere`)
    - Union of spheres (:py:class:`integrate.sphere_union`)
    - Convex polygons (:py:class:`integrate.convex_polygon`)
    - Convex spheropolygons (:py:class:`integrate.convex_spheropolygon`)
    - Simple polygons (:py:class:`integrate.simple_polygon`)
    - Ellipsoids / ellipses (:py:class:`integrate.ellipsoid`)
    - Convex polyhedra (:py:class:`integrate.convex_polyhedron`)
    - Convex spheropolyhedra (:py:class:`integrate.convex_spheropolyhedron`)
    - Faceted spheres (:py:class:`integrate.faceted_sphere`)
    - General polyhedra (:py:class:`integrate.polyhedron`)
- Execution:
    - Canonical hard particle MC on a single CPU core
    - Parallel updates on many CPU cores using MPI
    - Parallel update scheme on a single GPU
    - Frenkel-Ladd free energy determination on a single CPU core
- Analysis:
    - Scale distribution function for pressure determination in NVT (:py:class:`analyze.sdf`)
- File I/O:
    - Loose integration with pos_writer

.. rubric:: Logging

The following quantities are provided by the integrator for use in HOOMD-blue's :py:class:`hoomd.analyze.log`.

- ``hpmc_sweep`` - Number of sweeps completed since the start of the MC integrator
- ``hpmc_translate_acceptance`` - Fraction of translation moves accepted (averaged only over the last time step)
- ``hpmc_rotate_acceptance`` - Fraction of rotation moves accepted (averaged only over the last time step)
- ``hpmc_d`` - Maximum move displacement
- ``hpmc_a`` - Maximum rotation move
- ``hpmc_move_ratio`` - Probability of making a translation move (1- P(rotate move))
- ``hpmc_overlap_count`` - Count of the number of particle-particle overlaps in the current system configuration

With non-interacting depletant (**implicit=True**), the following log quantities are available:

- ``hpmc_fugacity`` - The current value of the depletant fugacity (in units of density, volume^-1)
- ``hpmc_ntrial`` - The current number of configurational bias attempts per overlapping depletant
- ``hpmc_insert_count`` - Number of depletants inserted per colloid
- ``hpmc_reinsert_count`` - Number of overlapping depletants reinserted per colloid by configurational bias MC
- ``hpmc_free_volume_fraction`` - Fraction of free volume to total sphere volume after a trial move has been proposed
  (sampled inside a sphere around the new particle position)
- ``hpmc_overlap_fraction`` - Fraction of deplatants in excluded volume after trial move to depletants in free volume before move
- ``hpmc_configurational_bias_ratio`` - Ratio of configurational bias attempts to depletant insertions

:py:class:`compute.free_volume` provides the following loggable quantities:
- ``hpmc_free_volume`` - The free volume estimate in the simulation box obtained by MC sampling (in volume units)

:py:class:`update.npt` provides the following loggable quantities:

- ``hpmc_npt_trial_count`` - Number of NPT box changes attempted since the start of the NPT updater
- ``hpmc_npt_volume_acceptance`` - Fraction of volume change trials accepted (averaged only over the last time step)
- ``hpmc_npt_shear_acceptance`` - Fraction of shear trials accepted (averaged only over the last time step)
- ``hpmc_npt_move_ratio`` Probability of a volume change move (1-P(shear move)) (averaged only over the last time step)
- ``hpmc_npt_dLx`` Current maximum trial length change of the first box vector
- ``hpmc_npt_dLy`` Current maximum trial change of the y-component of the second box vector
- ``hpmc_npt_dLz`` Current maximum trial change of the z-component of the third box vector
- ``hpmc_npt_dxy`` Current maximum trial change of the shear parameter for the second box vector
- ``hpmc_npt_dxz`` Current maximum trial change of the shear parameter for the third box vector in the x direction
- ``hpmc_npt_dyz`` Current maximum trial change of the shear parameter for the third box vector in the y direction
- ``hpmc_npt_pressure`` Current value of the :math:`\beta p` value of the NpT updater

:py:class:`update.muvt` provides the following loggable quantities.

- ``hpmc_muvt_insert_acceptance`` - Fraction of particle insertions accepted (averaged from start of run)
- ``hpmc_muvt_remove_acceptance`` - Fraction of particle removals accepted (averaged from start of run)
- ``hpmc_muvt_volume_acceptance`` - Fraction of particle removals accepted (averaged from start of run)

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

.. rubric:: Stability

:py:mod:`hoomd.hpmc` is **stable**. When upgrading from version 2.x to 2.y (y > x),
existing job scripts that follow *documented* interfaces for functions and classes
will not require any modifications. **Maintainer:** Joshua A. Anderson
"""

# need to import all submodules defined in this directory
from hoomd.hpmc import integrate
from hoomd.hpmc import update
from hoomd.hpmc import analyze
from hoomd.hpmc import compute
from hoomd.hpmc import util
from hoomd.hpmc import field
