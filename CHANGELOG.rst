Change Log
==========

v2.x
----

v2.9.7 (2021-08-03)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Support CUDA 11.5. A bug in CUDA 11.4 may result in the error
  `__global__ function call is not configure` when running HOOMD.

v2.9.6 (2021-03-16)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Support TBB 2021.

v2.9.5 (2021-03-15)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Support macos-arm64.
* Support TBB 2021.
* Fix memory leak in PPPM.

v2.9.4 (2021-02-05)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Support thrust 1.10
* Support LLVM11
* Fix Python syntax warnings
* Fix compile errors with gcc 10

v2.9.3 (2020-08-05)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Fix a compile error with CUDA 11

v2.9.2 (2020-06-26)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Fix a bug where repeatedly using objects with ``period=None`` would use
  significant amounts of memory.
* Support CUDA 11.
* Reccomend citing the 2020 Computational Materials Science paper
  10.1016/j.commatsci.2019.109363.

v2.9.1 (2020-05-28)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Fixed a minor bug where the variable period timestep would be off by one when
  the timestep got sufficiently large.
* Updated collections API to hide ``DeprecationWarning``.
* Fix scaling of cutoff in Gay-Berne potential to scale the current maximum
  distance based on the orientations of the particles, ensuring ellipsoidal
  energy isocontours.
* Misc documentation fixes.


v2.9.0 (2020-02-03)
^^^^^^^^^^^^^^^^^^^

*New features*

* General

  * Read and write GSD 2.0 files.

    * HOOMD >=2.9 can read and write GSD files created by HOOMD <= 2.8 or GSD
      1.x. HOOMD <= 2.8 cannot read GSD files created by HOOMD >=2.9 or GSD >=
      2.0.
    * OVITO >=3.0.0-dev652 reads GSD 2.0 files.
    * A future release of the ``gsd-vmd`` plugin will read GSD 2.0 files.

* HPMC

  * User-settable parameters in ``jit.patch``.
  * 2D system support in muVT updater.
  * Fix bug in HPMC where overlaps were not checked after adding new particle
    types.

* MD

  * The performance of ``nlist.tree`` has been drastically improved for a
    variety of systems.

v2.8.2 (2019-12-20)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Fix randomization of barostat and thermostat velocities with
  ``randomize_velocities()`` for non-unit temperatures.
* Improve MPCD documentation.
* Fix uninitialized memory in some locations which could have led to
  unreproducible results with HPMC in MPI, in particular with
  ``ALWAYS_USE_MANAGED_MEMORY=ON``.
* Fix calculation of cell widths in HPMC (GPU) and ``nlist.cell()`` with MPI.
* Fix potential memory-management issue in MPI for migrating MPCD particles and
  cell energy.
* Fix bug where exclusions were sometimes ignored when ``charge.pppm()`` is
  the only potential using the neighbor list.
* Fix bug where exclusions were not accounted for properly in the
  ``pppm_energy`` log quantity.
* Fix a bug where MD simulations with MPI start off without a ghost layer,
  leading to crashes or dangerous builds shortly after ``run()``.
* ``hpmc.update.remove_drift`` now communicates particle positions after
  updating them.

v2.8.1 (2019-11-26)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Fix a rare divide-by-zero in the ``collide.srd`` thermostat.
* Improve performance of first frame written by ``dump.gsd``.
* Support Python 3.8.
* Fix an error triggering migration of embedded particles for MPCD with MPI +
  GPU configurations.

v2.8.0 (2019-10-30)
^^^^^^^^^^^^^^^^^^^

*New Features*

- MD:

  - ``hoomd.md.dihedral.harmonic`` now accepts phase offsets, ``phi_0``, for CHARMM-style periodic dihedrals.
  - Enable per-type shape information for anisotropic pair potentials that complements the existing pair parameters struct.

- HPMC:

  - Enable the use of an array with adjustable parameters within the user defined pair potential.
  - Add muVT updater for 2D systems.


*Bug fixes*

- Fix missing header in external plugin builds.
- Enable ``couple='none'`` option to ``md.integrate.npt()`` when randomly initializing velocities.
- Documentation improvements.
- Skip gsd shape unit test when required modules are not compiled.
- Fix default particle properties when new particles are added to the system (e.g., via the muVT updater).
- Fix ``charge.pppm()`` execution on multiple GPUs.
- Enable ``with SimulationContext() as c``.
- Fix a bug for ``mpcd.collide.at`` with embedded particles, which may have given incorrect results or simulation crashes.

v2.7.0 (2019-10-01)
^^^^^^^^^^^^^^^^^^^

*New features*

- General:

  - Allow components to use ``Logger`` at the C++ level.
  - Drop support for python 2.7.
  - User-defined log quantities in ``dump.gsd``.
  - Add ``hoomd.dump.gsd.dump_shape`` to save particle shape information in GSD files.

- HPMC:

  - Add ``get_type_shapes`` to ``ellipsoid``.

- MPCD:

  - ``mpcd.stream.slit_pore`` allows for simulations through parallel-plate (lamellar) pores.
  - ``mpcd.integrate`` supports integration of MD (solute) particles with bounce-back rules in MPCD streaming geometries.

*Bug fixes*

- ``hoomd.hdf5.log.query`` works with matrix quantities.
- ``test_group_rigid.py`` is run out of the ``md`` module.
- Fix a bug in ``md.integrate.langevin()`` and ``md.integrate.bd()`` where on the GPU the value of ``gamma`` would be ignored.
- Fix documentation about interoperability between ``md.mode_minimize_fire()`` and MPI.
- Clarify ``dump.gsd`` documentation.
- Improve documentation of ``lattice_field`` and ``frenkel_ladd_energy`` classes.
- Clarify singularity image download documentation.
- Correctly document the functional form of the Buckingham pair potential.
- Correct typos in HPMC example snippets.
- Support compilation in WSL.

v2.6.0 (2019-05-28)
^^^^^^^^^^^^^^^^^^^

*New features*

- General:

  - Enable ``HPMC`` plugins.
  - Fix plug-in builds when ``ENABLE_TBB`` or ``ALWAYS_USE_MANAGED_MEMORY`` CMake parameters are set.
  - Remove support for compute 3.0 GPUs.
  - Report detailed CUDA errors on initialization.
  - Document upcoming feature removals and API changes.

- MD:

  - Exclude neighbors that belong to the same floppy molecule.
  - Add fourier potential.

- HPMC:

  - New shape class: ``hpmc.integrate.faceted_ellipsoid_union()``.
  - Store the *orientable* shape state.

- MPCD:

  - ``mpcd.stream.slit`` allows for simulations in parallel-plate channels. Users can implement other geometries as a plugin.
  - MPCD supports virtual particle filling in bounded geometries through the ``set_filler`` method of ``mpcd.stream`` classes.
  - ``mpcd.stream`` includes an external ``mpcd.force`` acting on the MPCD particles. A block force, a constant force, and a sine force are implemented.

*Bug fixes*

- Fix compile errors with LLVM 8 and ``-DBUILD_JIT=on``.
- Allow simulations with 0 bonds to specify bond potentials.
- Fix a problem where HOOMD could not be imported in ``mpi4py`` jobs.
- Validate snapshot input in ``restore_snapshot``.
- Fix a bug where rigid body energy and pressure deviated on the first time step after ``run()``.
- Fix a bug which could lead to invalid MPI simulations with ``nlist.cell()`` and ``nlist.stencil()``.

*C++ API changes*

- Refactor handling of ``MPI_Comm`` inside library
- Use ``random123`` for random number generation
- CMake version 2.8.10.1 is now a minimum requirement for compiling from source

v2.5.2 (2019-04-30)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

- Support LLVM 9 in ``jit``
- Fix error when importing ``jit`` before ``hpmc``
- HPMC integrators raise errors when ``restore_state=True`` and state information is missing
- Send messages to replaced ``sys.stdout`` and ``sys.stderr`` streams
- Add ``hpmc.update.clusters`` to documentation index
- Fix a bug in the MPCD Gaussian random number generator that could lead to NaN values
- Fix issue where an initially cubic box can become non-cubic with ``integrate.npt()`` and ``randomize_velocities()``
- Fix illegal memory access in NeighborListGPU with ``-DALWAYS_USE_MANAGED_MEMORY=ON`` on single GPUs
- Improve ``pair.table`` performance with multi-GPU execution
- Improve ``charge.pppm`` performance with multi-GPU execution
- Improve rigid body performance with multi-GPU execution
- Display correct cell list statistics with the ``-DALWAYS_USE_MANAGED_MEMORY=ON`` compile option
- Fix a sporadic data corruption / bus error issue when data structures are dynamically resized in simulations that use unified memory (multi-GPU, or with -DALWAYS_USE_MANAGED_MEMORY=ON compile time option)
- Improve ``integrate.nve`` and ``integrate.npt`` performance with multi-GPU execution
- Improve some angular degrees of freedom integrators with multi-GPU execution
- Improve rigid body pressure calculation performance with multi-GPU execution

v2.5.1 (2019-03-14)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

- fix out-of-range memory access in ``hpmc.integrate.convex_polyheron``
- Remove support for clang3.8 and 4.0
- Documentation improvements
- Fix a segfault when using ``SLURM_LOCALID``

v2.5.0 (2019-02-05)
^^^^^^^^^^^^^^^^^^^

*New features*

-  General:

   -  Fix BondedGroupData and CommunicatorGPU compile errors in certain
      build configurations

-  MD:

   -  Generalize ``md.integrate.brownian`` and ``md.integrate.langevin``
      to support anisotropic friction coefficients for rotational
      Brownian motion.
   -  Improve NVLINK performance with rigid bodies
   -  ``randomize_velocities`` now chooses random values for the
      internal integrator thermostat and barostat variables.
   -  ``get_net_force`` returns the net force on a group of particles
      due to a specific force compute

-  HPMC:

   -  Fix a bug where external fields were ignored with the HPMC
      implicit integrator unless a patch potential was also in use.

-  JIT:

   -  Add ``jit.external.user`` to specify user-defined external fields
      in HPMC.
   -  Use ``-DHOOMD_LLVMJIT_BUILD`` now instead of ``-DHOOMD_NOPYTHON``

v2.4.2 (2018-12-20)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Miscellaneous documentation updates
-  Fix compile error with ``with -DALWAYS_USE_MANAGED_MEMORY=ON``
-  Fix MuellerPlatheFlow, cast input parameter to int to avoid C++
   constructor type mismatch
-  Improve startup time with multi-GPU simulations
-  Correctly assign GPUs to MPI processes on Summit when launching with
   more than one GPU per resource set
-  Optimize multi-GPU performance with NVLINK
-  Do not use mapped memory with MPI/GPU anymore
-  Fix some cases where a multi-GPU simulation fails with an alignment
   error
-  Eliminate remaining instance of unsafe ``__shfl``
-  Hide CMake warnings regarding missing CPU math libraries
-  Hide CMake warning regarding missing MPI<->CUDA interoperability
-  Refactor memory management to fix linker errors with some compilers

*C++ API Changes*

-  May break some plug-ins which rely on ``GPUArray`` data type being
   returned from ``ParticleData`` and other classes (replace by
   ``GlobalArray``)

v2.4.1 (2018-11-27)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Install ``WarpTools.cuh`` for use by plugins
-  Fix potential violation of detailed balance with anisotropic
   particles with ``hpmc.update.clusters`` in periodic boundary
   conditions
-  Support llvm 7.0

v2.4.0 (2018-11-07)
^^^^^^^^^^^^^^^^^^^

*New features*

-  General:

   -  Misc documentation updates
   -  Accept ``mpi4py`` communicators in ``context.initialize``.
   -  CUDA 10 support and testing
   -  Sphinx 1.8 support
   -  Flush message output so that ``python -u`` is no longer required
      to obtain output on some batch job systems
   -  Support multi-GPU execution on dense nodes using CUDA managed
      memory. Execute with ``--gpu=0,1,..,n-1`` command line option to
      run on the first n GPUs (Pascal and above).

      -  Node-local acceleration is implemented for a subset of kernels.
         Performance improvements may vary.
      -  Improvements are only expected with NVLINK hardware. Use MPI
         when NVLINK is not available.
      -  Combine the ``--gpu=..`` command line option with mpirun to
         execute on many dense nodes

   -  Bundle ``libgetar`` v0.7.0 and remove ``sqlite3`` dependency
   -  When building with ENABLE_CUDA=on, CUDA 8.0 is now a minimum
      requirement

-  MD:

   -  *no changes*.

-  HPMC:

   -  Add ``convex_spheropolyhedron_union`` shape class.
   -  Correctly count acceptance rate when maximum particle move is is
      zero in ``hpmc.integrate.*``.
   -  Correctly count acceptance rate when maximum box move size is zero
      in ``hpmc.update.boxmc``.
   -  Fix a bug that may have led to overlaps between polygon soups with
      ``hpmc.integrate.polyhedron``.
   -  Improve performance in sphere trees used in
      ``hpmc.integrate.sphere_union``.
   -  Add ``test_overlap`` method to python API

-  API:

   -  Allow external callers of HOOMD to set the MPI communicator
   -  Removed all custom warp reduction and scan operations. These are
      now performed by CUB.
   -  Separate compilation of pair potentials into multiple files.
   -  Removed compute 2.0 workaround implementations. Compute 3.0 is now
      a hard minimum requirement to run HOOMD.
   -  Support and enable compilation for sm70 with CUDA 9 and newer.

-  Deprecated:

   -  HPMC: The implicit depletant mode ``circumsphere`` with
      ``ntrial > 0`` does not support compute 7.0 (Volta) and newer GPUs
      and is now disabled by default. To enable this functionality,
      configure HOOMD with option the ``-DENABLE_HPMC_REINSERT=ON``,
      which will not function properly on compute 7.0 (Volta) and newer
      GPUs.
   -  HPMC: ``convex_polyhedron_union`` is replaced by
      ``convex_spheropolyhedron_union`` (when sweep_radii are 0 for all
      particles)

v2.3.5 (2018-10-07)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Document ``--single-mpi`` command line option.
-  HPMC: Fix a bug where ``hpmc.field.lattice_field`` did not resize 2D
   systems properly in combination with ``update.box_resize``.

v2.3.4 (2018-07-30)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  ``init.read_gsd`` no longer applies the *time_step* override when
   reading the *restart* file
-  HPMC: Add ``hpmc_patch_energy`` and ``hpmc_patch_rcut`` loggable
   quantities to the documentation

v2.3.3 (2018-07-03)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix ``libquickhull.so`` not found regression on Mac OS X

v2.3.2 (2018-06-29)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix a bug where gsd_snapshot would segfault when called without an
   execution context.
-  Compile warning free with gcc8.
-  Fix compile error when TBB include files are in non-system directory.
-  Fix ``libquickhull.so`` not found error on additional platforms.
-  HOOMD-blue is now available on **conda-forge** and the **docker
   hub**.
-  MPCD: Default value for ``kT`` parameter is removed for
   ``mpcd.collide.at``. Scripts that are correctly running are not
   affected by this change.
-  MPCD: ``mpcd`` notifies the user of the appropriate citation.
-  MD: Correct force calculation between dipoles and point charge in
   ``pair.dipole``

*Deprecated*

-  The **anaconda** channel **glotzer** will no longer be updated. Use
   **conda-forge** to upgrade to v2.3.2 and newer versions.

v2.3.1 (2018-05-25)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix doxygen documentation syntax errors
-  Fix libquickhull.so not found error on some platforms
-  HPMC: Fix bug that allowed particles to pas through walls
-  HPMC: Check spheropolyhedra with 0 vertices against walls correctly
-  HPMC: Fix plane wall/spheropolyhedra overlap test
-  HPMC: Restore detailed balance in implicit depletant integrator
-  HPMC: Correctly choose between volume and lnV moves in
   ``hpmc.update.boxmc``
-  HPMC: Fix name of log quantity ``hpmc_clusters_pivot_acceptance``
-  MD: Fix image list for tree neighbor lists in 2d

v2.3.0 (2018-04-25)
^^^^^^^^^^^^^^^^^^^

*New features*

-  General:

   -  Store ``BUILD_*`` CMake variables in the hoomd cmake cache for use
      in external plugins.
   -  ``init.read_gsd`` and ``data.gsd_snapshot`` now accept negative
      frame indices to index from the end of the trajectory.
   -  Faster reinitialization from snapshots when done frequently.
   -  New command line option ``--single-mpi`` allows non-mpi builds of
      hoomd to launch within mpirun (i.e. for use with mpi4py managed
      pools of jobs)
   -  For users of the University of Michigan Flux system: A ``--mode``
      option is no longer required to run hoomd.

-  MD:

   -  Improve performance with ``md.constrain.rigid`` in multi-GPU
      simulations.
   -  New command ``integrator.randomize_velocities()`` sets a particle
      group’s linear and angular velocities to random values consistent
      with a given kinetic temperature.
   -  ``md.force.constant()`` now supports setting the force per
      particle and inside a callback

-  HPMC:

   -  Enabled simulations involving spherical walls and convex
      spheropolyhedral particle shapes.
   -  Support patchy energetic interactions between particles (CPU only)
   -  New command ``hpmc.update.clusters()`` supports geometric cluster
      moves with anisotropic particles and/or depletants and/or patch
      potentials. Supported move types: pivot and line reflection
      (geometric), and AB type swap.

-  JIT:

   -  Add new experimental ``jit`` module that uses LLVM to compile and
      execute user provided C++ code at runtime. (CPU only)
   -  Add ``jit.patch.user``: Compute arbitrary patch energy between
      particles in HPMC (CPU only)
   -  Add ``jit.patch.user_union``: Compute arbitrary patch energy
      between rigid unions of points in HPMC (CPU only)
   -  Patch energies operate with implicit depletant and normal HPMC
      integration modes.
   -  ``jit.patch.user_union`` operates efficiently with additive
      contributions to the cutoff.

-  MPCD:

   -  The ``mpcd`` component adds support for simulating hydrodynamics
      using the multiparticle collision dynamics method.

*Beta feature*

-  Node local parallelism (optional, build with ``ENABLE_TBB=on``):

   -  The Intel TBB library is required to enable this feature.
   -  The command line option ``--nthreads`` limits the number of
      threads HOOMD will use. The default is all CPU cores in the
      system.
   -  Only the following methods in HOOMD will take advantage of
      multiple threads:

      -  ``hpmc.update.clusters()``
      -  HPMC integrators with implicit depletants enabled
      -  ``jit.patch.user_union``

Node local parallelism is still under development. It is not enabled in
builds by default and only a few methods utilize multiple threads. In
future versions, additional methods in HOOMD may support multiple
threads.

To ensure future workflow compatibility as future versions enable
threading in more components, explicitly set –nthreads=1.

*Bug fixes*

-  Fixed a problem with periodic boundary conditions and implicit
   depletants when ``depletant_mode=circumsphere``
-  Fixed a rare segmentation fault with ``hpmc.integrate.*_union()`` and
   ``hpmc.integrate.polyhedron``
-  ``md.force.active`` and ``md.force.dipole`` now record metadata
   properly.
-  Fixed a bug where HPMC restore state did not set ignore flags
   properly.
-  ``hpmc_boxmc_ln_volume_acceptance`` is now available for logging.

*Other changes*

-  Eigen is now provided as a submodule. Plugins that use Eigen headers
   need to update include paths.
-  HOOMD now builds with pybind 2.2. Minor changes to source and cmake
   scripts in plugins may be necessary. See the updated example plugin.
-  HOOMD now builds without compiler warnings on modern compilers (gcc6,
   gcc7, clang5, clang6).
-  HOOMD now uses pybind11 for numpy arrays instead of ``num_util``.
-  HOOMD versions v2.3.x will be the last available on the anaconda
   channel ``glotzer``.

v2.2.5 (2018-04-20)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Pin cuda compatible version in conda package to resolve ``libcu*.so``
   not found errors in conda installations.

v2.2.4 (2018-03-05)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix a rare error in ``md.nlist.tree`` when particles are very close
   to each other.
-  Fix deadlock when ``init.read_getar`` is given different file names
   on different ranks.
-  Sample from the correct uniform distribution of depletants in a
   sphere cap with ``depletant_mode='overlap_regions'`` on the CPU
-  Fix a bug where ternary (or higher order) mixtures of small and large
   particles were not correctly handled with
   ``depletant_mode='overlap_regions'`` on the CPU
-  Improve acceptance rate in depletant simulations with
   ``depletant_mode='overlap_regions'``

v2.2.3 (2018-01-25)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Write default values to gsd frames when non-default values are
   present in frame 0.
-  ``md.wall.force_shifted_lj`` now works.
-  Fix a bug in HPMC where ``run()`` would not start after
   ``restore_state`` unless shape parameters were also set from python.
-  Fix a bug in HPMC Box MC updater where moves were attempted with zero
   weight.
-  ``dump.gsd()`` now writes ``hpmc`` shape state correctly when there
   are multiple particle types.
-  ``hpmc.integrate.polyhedron()`` now produces correct results on the
   GPU.
-  Fix binary compatibility across python minor versions.

v2.2.2 (2017-12-04)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  ``md.dihedral.table.set_from_file`` now works.
-  Fix a critical bug where forces in MPI simulations with rigid bodies
   or anisotropic particles were incorrectly calculated
-  Ensure that ghost particles are updated after load balancing.
-  ``meta.dump_metadata`` no longer reports an error when used with
   ``md.constrain.rigid``
-  Miscellaneous documentation fixes
-  ``dump.gsd`` can now write GSD files with 0 particles in a frame
-  Explicitly report MPI synchronization delays due to load imbalance
   with ``profile=True``
-  Correctly compute net torque of rigid bodies with anisotropic
   constituent particles in MPI execution on multiple ranks
-  Fix ``PotentialPairDPDThermoGPU.h`` for use in external plugins
-  Use correct ghost region with ``constrain.rigid`` in MPI execution on
   multiple ranks
-  ``hpmc.update.muvt()`` now works with
   ``depletant_mode='overlap_regions'``
-  Fix the sampling of configurations with in ``hpmc.update.muvt`` with
   depletants
-  Fix simulation crash after modifying a snapshot and re-initializing
   from it
-  The pressure in simulations with rigid bodies
   (``md.constrain.rigid()``) and MPI on multiple ranks is now computed
   correctly

v2.2.1 (2017-10-04)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Add special pair headers to install target
-  Fix a bug where ``hpmc.integrate.convex_polyhedron``,
   ``hpmc.integrate.convex_spheropolyhedron``,
   ``hpmc.integrate.polyedron``, ``hpmc.integrate.faceted_sphere``,
   ``hpmc.integrate.sphere_union`` and
   ``hpmc.integrate.convex_polyhedron_union`` produced spurious overlaps
   on the GPU

v2.2.0 (2017-09-08)
^^^^^^^^^^^^^^^^^^^

*New features*

-  General:

   -  Add ``hoomd.hdf5.log`` to log quantities in hdf5 format. Matrix
      quantities can be logged.
   -  ``dump.gsd`` can now save internal state to gsd files. Call
      ``dump_state(object)`` to save the state for a particular object.
      The following objects are supported:

      -  HPMC integrators save shape and trial move size state.

   -  Add *dynamic* argument to ``hoomd.dump.gsd`` to specify which
      quantity categories should be written every frame.
   -  HOOMD now inter-operates with other python libraries that set the
      active CUDA device.
   -  Add generic capability for bidirectional ghost communication,
      enabling multi body potentials in MPI simulation.

-  MD:

   -  Added support for a 3 body potential that is harmonic in the local
      density.
   -  ``force.constant`` and ``force.active`` can now apply torques.
   -  ``quiet`` option to ``nlist.tune`` to quiet the output of the
      embedded ``run()`` commands.
   -  Add special pairs as exclusions from neighbor lists.
   -  Add cosine squared angle potential ``md.angle.cosinesq``.
   -  Add ``md.pair.DLVO()`` for evaluation of colloidal dispersion and
      electrostatic forces.
   -  Add Lennard-Jones 12-8 pair potential.
   -  Add Buckingham (exp-6) pair potential.
   -  Add Coulomb 1-4 special_pair potential.
   -  Check that composite body dimensions are consistent with minimum
      image convention and generate an error if they are not.
   -  ``md.integrate.mode.minimize_fire()`` now supports anisotropic
      particles (i.e. composite bodies)
   -  ``md.integrate.mode.minimize_fire()`` now supports flexible
      specification of integration methods
   -  ``md.integrate.npt()/md.integrate.nph()`` now accept a friction
      parameter (gamma) for damping out box fluctuations during
      minimization runs
   -  Add new command ``integrate.mode_standard.reset_methods()`` to
      clear NVT and NPT integrator variables

-  HPMC:

   -  ``hpmc.integrate.sphere_union()`` takes new capacity parameter to
      optimize performance for different shape sizes
   -  ``hpmc.integrate.polyhedron()`` takes new capacity parameter to
      optimize performance for different shape sizes
   -  ``hpmc.integrate.convex_polyhedron`` and
      ``convex_spheropolyhedron`` now support arbitrary numbers of
      vertices, subject only to memory limitations (``max_verts`` is now
      ignored).
   -  HPMC integrators restore state from a gsd file read by
      ``init.read_gsd`` when the option ``restore_state`` is ``True``.
   -  Deterministic HPMC integration on the GPU (optional):
      ``mc.set_params(deterministic=True)``.
   -  New ``hpmc.update.boxmc.ln_volume()`` move allows logarithmic
      volume moves for fast equilibration.
   -  New shape: ``hpmc.integrate.convex_polyhedron_union`` performs
      simulations of unions of convex polyhedra.
   -  ``hpmc.field.callback()`` now enables MC energy evaluation in a
      python function
   -  The option ``depletant_mode='overlap_regions'`` for
      ``hpmc.integrate.*`` allows the selection of a new depletion
      algorithm that restores the diffusivity of dilute colloids in
      dense depletant baths

*Deprecated*

-  HPMC: ``hpmc.integrate.sphere_union()`` no longer needs the
   ``max_members`` parameter.
-  HPMC: ``hpmc.integrate.convex_polyhedron`` and
   ``convex_spheropolyhedron`` no longer needs the ``max_verts``
   parameter.
-  The *static* argument to ``hoomd.dump.gsd`` should no longer be used.
   Use *dynamic* instead.

*Bug fixes*

-  HPMC:

   -  ``hpmc.integrate.sphere_union()`` and
      ``hpmc.integrate.polyhedron()`` missed overlaps.
   -  Fix alignment error when running implicit depletants on GPU with
      ntrial > 0.
   -  HPMC integrators now behave correctly when the user provides
      different RNG seeds on different ranks.
   -  Fix a bug where overlapping configurations were produced with
      ``hpmc.integrate.faceted_sphere()``

-  MD:

   -  ``charge.pppm()`` with ``order=7`` now gives correct results
   -  The PPPM energy for particles excluded as part of rigid bodies now
      correctly takes into account the periodic boundary conditions

-  EAM:

   -  ``metal.pair.eam`` now produces correct results.

*Other changes*

-  Optimized performance of HPMC sphere union overlap check and
   polyhedron shape
-  Improved performance of rigid bodies in MPI simulations
-  Support triclinic boxes with rigid bodies
-  Raise an error when an updater is given a period of 0
-  Revised compilation instructions
-  Misc documentation improvements
-  Fully document ``constrain.rigid``
-  ``-march=native`` is no longer set by default (this is now a
   suggestion in the documentation)
-  Compiler flags now default to CMake defaults
-  ``ENABLE_CUDA`` and ``ENABLE_MPI`` CMake options default OFF. User
   must explicitly choose to enable optional dependencies.
-  HOOMD now builds on powerpc+CUDA platforms (tested on summitdev)
-  Improve performance of GPU PPPM force calculation
-  Use sphere tree to further improve performance of
   ``hpmc.integrate.sphere_union()``

v2.1.9 (2017-08-22)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix a bug where the log quantity ``momentum`` was incorrectly
   reported in MPI simulations.
-  Raise an error when the user provides inconsistent ``charge`` or
   ``diameter`` lists to ``md.constrain.rigid``.
-  Fix a bug where ``pair.compute_energy()`` did not report correct
   results in MPI parallel simulations.
-  Fix a bug where make rigid bodies with anisotropic constituent
   particles did not work on the GPU.
-  Fix hoomd compilation after the rebase in the cub repository.
-  ``deprecated.dump.xml()`` now writes correct results when particles
   have been added or deleted from the simulation.
-  Fix a critical bug where ``charge.pppm()`` calculated invalid forces
   on the GPU

v2.1.8 (2017-07-19)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  ``init.read_getar`` now correctly restores static quantities when
   given a particular frame.
-  Fix bug where many short calls to ``run()`` caused incorrect results
   when using ``md.integrate.langevin``.
-  Fix a bug in the Saru pseudo-random number generator that caused some
   double-precision values to be drawn outside the valid range [0,1) by
   a small amount. Both floats and doubles are now drawn on [0,1).
-  Fix a bug where coefficients for multi-character unicode type names
   failed to process in Python 2.

*Other changes*

-  The Saru generator has been moved into ``hoomd/Saru.h``, and plugins
   depending on Saru or SaruGPU will need to update their includes. The
   ``SaruGPU`` class has been removed. Use ``hoomd::detail::Saru``
   instead for both CPU and GPU plugins.

v2.1.7 (2017-05-11)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix PPM exclusion handling on the CPU
-  Handle ``r_cut`` for special pairs correctly
-  Fix tauP reference in NPH documentation
-  Fixed ``constrain.rigid`` on compute 5.x.
-  Fixed random seg faults when using sqlite getar archives with LZ4
   compression
-  Fixed XZ coupling with ``hoomd.md.integrate.npt`` integration
-  Fixed aspect ratio with non-cubic boxes in
   ``hoomd.hpmc.update.boxmc``

v2.1.6 (2017-04-12)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Document ``hpmc.util.tune_npt``
-  Fix dump.getar.writeJSON usage with MPI execution
-  Fix a bug where integrate.langevin and integrate.brownian correlated
   RNGs between ranks in multiple CPU execution
-  Bump CUB to version 1.6.4 for improved performance on Pascal
   architectures. CUB is now embedded using a git submodule. Users
   upgrading existing git repositories should reinitialize their git
   submodules with ``git submodule update --init``
-  CMake no longer complains when it finds a partial MKL installation.

v2.1.5 (2017-03-09)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fixed a compile error on Mac

v2.1.4 (2017-03-09)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fixed a bug re-enabling disabled integration methods
-  Fixed a bug where adding particle types to the system failed for
   anisotropic pair potentials
-  scipy is no longer required to execute DEM component unit tests
-  Issue a warning when a subsequent call to context.initialize is given
   different arguments
-  DPD now uses the seed from rank 0 to avoid incorrect simulations when
   users provide different seeds on different ranks
-  Miscellaneous documentation updates
-  Defer initialization message until context.initialize
-  Fixed a problem where a momentary dip in TPS would cause walltime
   limited jobs to exit prematurely
-  HPMC and DEM components now correctly print citation notices

v2.1.3 (2017-02-07)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fixed a bug where the WalltimeLimitReached was ignored

v2.1.2 (2017-01-11)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  (HPMC) Implicit depletants with spheres and faceted spheres now
   produces correct ensembles
-  (HPMC) Implicit depletants with ntrial > 0 now produces correct
   ensembles
-  (HPMC) NPT ensemble in HPMC (``hpmc.update.boxmc``) now produces
   correct ensembles
-  Fix a bug where multiple nvt/npt integrators caused warnings from
   analyze.log.
-  update.balance() is properly ignored when only one rank is available
-  Add missing headers to plugin install build
-  Fix a bug where charge.pppm calculated an incorrect pressure

-  Other changes \*

-  Drop support for compute 2.0 GPU devices
-  Support cusolver with CUDA 8.0

v2.1.1 (2016-10-23)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix ``force.active`` memory allocation bug
-  Quiet Python.h warnigns when building (python 2.7)
-  Allow multi-character particle types in HPMC (python 2.7)
-  Enable ``dump.getar.writeJSON`` in MPI
-  Allow the flow to change directions in
   ``md.update.mueller_plathe_flow``
-  Fix critical bug in MPI communication when using HPMC integrators

v2.1.0 (2016-10-04)
^^^^^^^^^^^^^^^^^^^

*New features*

-  enable/disable overlap checks between pairs of constituent particles
   for ``hpmc.integrate.sphere_union()``
-  Support for non-additive mixtures in HPMC, overlap checks can now be
   enabled/disabled per type-pair
-  Add ``md.constrain.oned`` to constrain particles to move in one
   dimension
-  ``hpmc.integrate.sphere_union()`` now takes max_members as an
   optional argument, allowing to use GPU memory more efficiently
-  Add ``md.special_pair.lj()`` to support scaled 1-4 (or other)
   exclusions in all-atom force fields
-  ``md.update.mueller_plathe_flow()``: Method to create shear flows in
   MD simulations
-  ``use_charge`` option for ``md.pair.reaction_field``
-  ``md.charge.pppm()`` takes a Debye screening length as an optional
   parameter
-  ``md.charge.pppm()`` now computes the rigid body correction to the
   PPPM energy

*Deprecated*

-  HPMC: the ``ignore_overlaps`` flag is replaced by
   ``hpmc.integrate.interaction_matrix``

*Other changes*

-  Optimized MPI simulations of mixed systems with rigid and non-rigid
   bodies
-  Removed dependency on all boost libraries. Boost is no longer needed
   to build hoomd
-  Intel compiler builds are no longer supported due to c++11 bugs
-  Shorter compile time for HPMC GPU kernels
-  Include symlinked external components in the build process
-  Add template for external components
-  Optimized dense depletant simulations with HPMC on CPU

*Bug fixes*

-  fix invalid mesh energy in non-neutral systems with
   ``md.charge.pppm()``
-  Fix invalid forces in simulations with many bond types (on GPU)
-  fix rare cases where analyze.log() would report a wrong pressure
-  fix possible illegal memory access when using
   ``md.constrain.rigid()`` in GPU MPI simulations
-  fix a bug where the potential energy is misreported on the first step
   with ``md.constrain.rigid()``
-  Fix a bug where the potential energy is misreported in MPI
   simulations with ``md.constrain.rigid()``
-  Fix a bug where the potential energy is misreported on the first step
   with ``md.constrain.rigid()``
-  ``md.charge.pppm()`` computed invalid forces
-  Fix a bug where PPPM interactions on CPU where not computed correctly
-  Match logged quantitites between MPI and non-MPI runs on first time
   step
-  Fix ``md.pair.dpd`` and ``md.pair.dpdlj`` ``set_params``
-  Fix diameter handling in DEM shifted WCA potential
-  Correctly handle particle type names in lattice.unitcell
-  Validate ``md.group.tag_list`` is consistent across MPI ranks

v2.0.3 (2016-08-30)
^^^^^^^^^^^^^^^^^^^

-  hpmc.util.tune now works with particle types as documented
-  Fix pressure computation with pair.dpd() on the GPU
-  Fix a bug where dump.dcd corrupted files on job restart
-  Fix a bug where HPMC walls did not work correctly with MPI
-  Fix a bug where stdout/stderr did not appear in MPI execution
-  HOOMD will now report an human readable error when users forget
   context.initialize()
-  Fix syntax errors in frenkel ladd field

v2.0.2 (2016-08-09)
^^^^^^^^^^^^^^^^^^^

-  Support CUDA Toolkit 8.0
-  group.rigid()/nonrigid() did not work in MPI simulations
-  Fix builds with ENABLE_DOXYGEN=on
-  Always add -std=c++11 to the compiler command line arguments
-  Fix rare infinite loops when using hpmc.integrate.faceted_sphere
-  Fix hpmc.util.tune to work with more than one tunable
-  Fix a bug where dump.gsd() would write invalid data in simulations
   with changing number of particles
-  replicate() sometimes did not work when restarting a simulation

v2.0.1 (2016-07-15)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix acceptance criterion in mu-V-T simulations with implicit
   depletants (HPMC).
-  References to disabled analyzers, computes, updaters, etc. are
   properly freed from the simulation context.
-  Fix a bug where ``init.read_gsd`` ignored the ``restart`` argument.
-  Report an error when HPMC kernels run out of memory.
-  Fix ghost layer when using rigid constraints in MPI runs.
-  Clarify definition of the dihedral angle.

v2.0.0 (2016-06-22)
^^^^^^^^^^^^^^^^^^^

HOOMD-blue v2.0 is released under a clean BSD 3-clause license.

*New packages*

-  ``dem`` - simulate faceted shapes with dynamics
-  ``hpmc`` - hard particle Monte Carlo of a variety of shape classes.

*Bug fixes*

-  Angles, dihedrals, and impropers no longer initialize with one
   default type.
-  Fixed a bug where integrate.brownian gave the same x,y, and z
   velocity components.
-  Data proxies verify input types and vector lengths.
-  dump.dcd no longer generates excessive metadata traffic on lustre
   file systems

*New features*

-  Distance constraints ``constrain.distance`` - constrain pairs of
   particles to a fixed separation distance
-  Rigid body constraints ``constrain.rigid`` - rigid bodies now have
   central particles, and support MPI and replication
-  Multi-GPU electrostatics ``charge.pppm`` - the long range
   electrostatic forces are now supported in MPI runs
-  ``context.initialize()`` can now be called multiple times - useful in
   jupyter notebooks
-  Manage multiple simulations in a single job script with
   ``SimulationContext`` as a python context manager.
-  ``util.quiet_status() / util.unquiet_status()`` allow users to
   control if line status messages are output.
-  Support executing hoomd in Jupyter (ipython) notebooks. Notice,
   warning, and error messages now show up in the notebook output
   blocks.
-  ``analyze.log`` can now register python callback functions as sources
   for logged quantities.
-  The GSD file format (http://gsd.readthedocs.io) is fully implemented
   in hoomd

   -  ``dump.gsd`` writes GSD trajectories and restart files (use
      ``truncate=true`` for restarts).
   -  ``init.read_gsd`` reads GSD file and initializes the system, and
      can start the simulation from any frame in the GSD file.
   -  ``data.gsd_snapshot`` reads a GSD file into a snapshot which can
      be modified before system initialization with
      ``init.read_snapshot``.
   -  The GSD file format is capable of storing all particle and
      topology data fields in hoomd, either static at frame 0, or
      varying over the course of the trajectory. The number of
      particles, types, bonds, etc. can also vary over the trajectory.

-  ``force.active`` applies an active force (optionally with rotational
   diffusion) to a group of particles
-  ``update.constrain_ellipsoid`` constrains particles to an ellipsoid
-  ``integrate.langevin`` and ``integrate.brownian`` now apply
   rotational noise and damping to anisotropic particles
-  Support dynamically updating groups. ``group.force_update()`` forces
   the group to rebuild according to the original selection criteria.
   For example, this can be used to periodically update a cuboid group
   to include particles only in the specified region.
-  ``pair.reaction_field`` implements a pair force for a screened
   electrostatic interaction of a charge pair in a dielectric medium.
-  ``force.get_energy`` allows querying the potential energy of a
   particle group for a specific force
-  ``init.create_lattice`` initializes particles on a lattice.

   -  ``lattice.unitcell`` provides a generic unit cell definition for
      ``create_lattice``
   -  Convenience functions for common lattices: sq, hex, sc, bcc, fcc.

-  Dump and initialize commands for the GTAR file format
   (http://libgetar.readthedocs.io).

   -  GTAR can store trajectory data in zip, tar, sqlite, or bare
      directories
   -  The current version stores system properties, later versions will
      be able to capture log, metadata, and other output to reduce the
      number of files that a job script produces.

-  ``integrate.npt`` can now apply a constant stress tensor to the
   simulation box.
-  Faceted shapes can now be simulated through the ``dem`` component.

*Changes that require job script modifications*

-  ``context.initialize()`` is now required before any other hoomd
   script command.
-  ``init.reset()`` no longer exists. Use ``context.initialize()`` or
   activate a ``SimulationContext``.
-  Any scripts that relied on undocumented members of the ``globals``
   module will fail. These variables have been moved to the ``context``
   module and members of the currently active ``SimulationContext``.
-  bonds, angles, dihedrals, and impropers no longer use the
   ``set_coeff`` syntax. Use ``bond_coeff.set``, ``angle_coeff.set``,
   ``dihedral_coeff.set``, and ``improper_coeff.set`` instead.
-  ``hoomd_script`` no longer exists, python commands are now spread
   across ``hoomd``, ``hoomd.md``, and other sub packages.
-  ``integrate.\*_rigid()`` no longer exists. Use a standard integrator
   on ``group.rigid_center()``, and define rigid bodies using
   ``constrain.rigid()``
-  All neighbor lists must be explicitly created using ``nlist.\*``, and
   each pair potential must be attached explicitly to a neighbor list. A
   default global neighbor list is no longer created.
-  Moved cgcmm into its own package.
-  Moved eam into the metal package.
-  Integrators now take ``kT`` arguments for temperature instead of
   ``T`` to avoid confusion on the units of temperature.
-  phase defaults to 0 for updaters and analyzers so that restartable
   jobs are more easily enabled by default.
-  ``dump.xml`` (deprecated) requires a particle group, and can dump
   subsets of particles.

*Other changes*

-  CMake minimum version is now 2.8
-  Convert particle type names to ``str`` to allow unicode type name
   input
-  ``__version__`` is now available in the top level package
-  ``boost::iostreams`` is no longer a build dependency
-  ``boost::filesystem`` is no longer a build dependency
-  New concepts page explaining the different styles of neighbor lists
-  Default neighbor list buffer radius is more clearly shown to be
   r_buff = 0.4
-  Memory usage of ``nlist.stencil`` is significantly reduced
-  A C++11 compliant compiler is now required to build HOOMD-blue

*Removed*

-  Removed ``integrate.bdnvt``: use ``integrate.langevin``
-  Removed ``mtk=False`` option from ``integrate.nvt`` - The MTK NVT
   integrator is now the only implementation.
-  Removed ``integrate.\*_rigid()``: rigid body functionality is now
   contained in the standard integration methods
-  Removed the global neighbor list, and thin wrappers to the neighbor
   list in ``nlist``.
-  Removed PDB and MOL2 dump writers.
-  Removed init.create_empty

*Deprecated*

-  Deprecated analyze.msd.
-  Deprecated dump.xml.
-  Deprecated dump.pos.
-  Deprecated init.read_xml.
-  Deprecated init.create_random.
-  Deprecated init.create_random_polymers.

v1.x
----

v1.3.3 (2016-03-06)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix problem incluing ``hoomd.h`` in plugins
-  Fix random memory errors when using walls

v1.3.2 (2016-02-08)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix wrong access to system.box
-  Fix kinetic energy logging in MPI
-  Fix particle out of box error if particles are initialized on the
   boundary in MPI
-  Add integrate.brownian to the documentation index
-  Fix misc doc typos
-  Fix runtime errors with boost 1.60.0
-  Fix corrupt metadata dumps in MPI runs

v1.3.1 (2016-1-14)
^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix invalid MPI communicator error with Intel MPI
-  Fix python 3.5.1 seg fault

v1.3.0 (2015-12-8)
^^^^^^^^^^^^^^^^^^

*New features*

-  Automatically load balanced domain decomposition simulations.
-  Anisotropic particle integrators.
-  Gay-Berne pair potential.
-  Dipole pair potential.
-  Brownian dynamics ``integrate.brownian``
-  Langevin dynamics ``integrate.langevin`` (formerly ``bdnvt``)
-  ``nlist.stencil`` to compute neighbor lists using stencilled cell
   lists.
-  Add single value scale, ``min_image``, and ``make_fraction`` to
   ``data.boxdim``
-  ``analyze.log`` can optionally not write a file and now supports
   querying current quantity values.
-  Rewritten wall potentials.

   -  Walls are now sums of planar, cylindrical, and spherical
      half-spaces.
   -  Walls are defined and can be modified in job scripts.
   -  Walls execute on the GPU.
   -  Walls support per type interaction parameters.
   -  Implemented for: lj, gauss, slj, yukawa, morse, force_shifted_lj,
      and mie potentials.

-  External electric field potential: ``external.e_field``

*Bug fixes*

-  Fixed a bug where NVT integration hung when there were 0 particles in
   some domains.
-  Check SLURM environment variables for local MPI rank identification
-  Fixed a typo in the box math documentation
-  Fixed a bug where exceptions weren’t properly passed up to the user
   script
-  Fixed a bug in the velocity initialization example
-  Fixed an openmpi fork() warning on some systems
-  Fixed segfaults in PPPM
-  Fixed a bug where compute.thermo failed after reinitializing a system
-  Support list and dict-like objects in init.create_random_polymers.
-  Fall back to global rank to assign GPUs if local rank is not
   available

*Deprecated commands*

-  ``integrate.bdnvt`` is deprecated. Use ``integrate.langevin``
   instead.
-  ``dump.bin`` and ``init.bin`` are now removed. Use XML files for
   restartable jobs.

*Changes that may break existing scripts*

-  ``boxdim.wrap`` now returns the position and image in a tuple, where
   it used to return just the position.
-  ``wall.lj`` has a new API
-  ``dump.bin`` and ``init.bin`` have been removed.

v1.2.1 (2015-10-22)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix a crash when adding or removing particles and reinitializing
-  Fix a bug where simulations hung on sm 5.x GPUs with CUDA 7.5
-  Fix compile error with long tests enabled
-  Issue a warning instead of an error for memory allocations greater
   than 4 GiB.
-  Fix invalid RPATH when building inside ``zsh``.
-  Fix incorrect simulations with ``integrate.npt_rigid``
-  Label mie potential correctly in user documentation

v1.2.0 (2015-09-30)
^^^^^^^^^^^^^^^^^^^

*New features*

-  Performance improvements for systems with large particle size
   disparity
-  Bounding volume hierarchy (tree) neighbor list computation
-  Neighbor lists have separate ``r_cut`` values for each pair of types
-  addInfo callback for dump.pos allows user specified information in
   pos files

*Bug fixes*

-  Fix ``test_pair_set_energy`` unit test, which failed on numpy < 1.9.0
-  Analyze.log now accepts unicode strings.
-  Fixed a bug where calling ``restore_snapshot()`` during a run zeroed
   potential parameters.
-  Fix segfault on exit with python 3.4
-  Add ``cite.save()`` to documentation
-  Fix a problem were bond forces are computed incorrectly in some MPI
   configurations
-  Fix bug in pair.zbl
-  Add pair.zbl to the documentation
-  Use ``HOOMD_PYTHON_LIBRARY`` to avoid problems with modified CMake
   builds that preset ``PYTHON_LIBRARY``

v1.1.1 (2015-07-21)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  ``dump.xml(restart=True)`` now works with MPI execution
-  Added missing documentation for ``meta.dump_metadata``
-  Build all unit tests by default
-  Run all script unit tests through ``mpirun -n 1``

v1.1.0 (2015-07-14)
^^^^^^^^^^^^^^^^^^^

*New features*

-  Allow builds with ninja.
-  Allow K=0 FENE bonds.
-  Allow number of particles types to change after initialization.

   .. code::

       system.particles.types.add('newtype')

-  Allow number of particles to change after initialization.

   .. code::

       system.particles.add(‘A’)
       del system.particles[0]

-  OPLS dihedral
-  Add ``phase`` keyword to analyzers and dumps to make restartable jobs easier.
-  ``HOOMD_WALLTIME_STOP`` environment variable to stop simulation runs before they hit a wall clock limit.
-  ``init.read_xml()`` Now accepts an initialization and restart file.
-  ``dump.xml()`` can now write restart files.
-   Added documentation concepts page on writing restartable jobs.
-   New citation management infrastructure. ``cite.save()`` writes ``.bib`` files with a list of references to
    features actively used in the current job script.
-   Snapshots expose data as numpy arrays for high performance access to particle properties.
-  ``data.make_snapshot()`` makes a new empty snapshot.
-  ``analyze.callback()`` allows multiple python callbacks to operate at different periods.
-  ``comm.barrier()``and`` comm.barrier_all()``allow users to insert barriers into their scripts.
-   Mie pair potential.
-  ``meta.dump_metadata()`` writes job metadata information out to a json file.
-  ``context.initialize()`` initializes the execution context.
-  Restart option for ``dump.xml()``

*Bug fixes*

-  Fix slow performance when initializing ``pair.slj()``\ in MPI runs.
-  Properly update particle image when setting position from python.
-  PYTHON_SITEDIR hoomd shell launcher now calls the python interpreter
   used at build time.
-  Fix compile error on older gcc versions.
-  Fix a bug where rigid bodies had 0 velocity when restarting jobs.
-  Enable ``-march=native`` builds in OS X clang builds.
-  Fix ``group.rigid()`` and ``group.nonrigid()``.
-  Fix image access from the python data access proxies.
-  Gracefully exit when launching MPI jobs with mixed execution
   configurations.

*Changes that may require updated job scripts*

-  ``context.initialize()`` **must** be called before any ``comm``
   method that queries the MPI rank. Call it as early as possible in
   your job script (right after importing ``hoomd_script``) to avoid
   problems.

*Deprecated*

-  ``init.create_empty()`` is deprecated and will be removed in a future
   version. Use ``data.make_snapshot()`` and ``init.read_snapshot()``
   instead.
-  Job scripts that do not call ``context.initialize()`` will result in
   a warning message. A future version of HOOMD will require that you
   call ``context.initialize()``.

*Removed*

-  Several ``option`` commands for controlling the execution
   configuration. Replaced with ``context.initialize``.

v1.0.5 (2015-05-19)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix segfault when changing integrators
-  Fix system.box to indicate the correct number of dimensions
-  Fix syntax error in comm.get_rank with –nrank
-  Enable CUDA enabled builds with the intel compiler
-  Use CMake builtin FindCUDA on recent versions of CMake
-  GCC_ARCH env var sets the -march command line option to gcc at
   configure time
-  Auto-assign GPU-ids on non-compute exclusive systems even with
   –mode=gpu
-  Support python 3.5 alpha
-  Fix a bug where particle types were doubled with boost 1.58.0
-  Fix a bug where angle_z=true dcd output was inaccurate near 0 angles
-  Properly handle lj.wall potentials with epsilon=0.0 and particles on
   top of the walls

v1.0.4 (2015-04-07)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix invalid virials computed in rigid body simulations when
   multi-particle bodies crossed box boundaries
-  Fix invalid forces/torques for rigid body simulations caused by race
   conditions
-  Fix compile errors on Mac OS X 10.10
-  Fix invalid pair force computations caused by race conditions
-  Fix invalid neighbour list computations caused by race conditions on
   Fermi generation GPUs

*Other*

-  Extremely long running unit tests are now off by default. Enable with
   -DHOOMD_SKIP_LONG_TESTS=OFF
-  Add additional tests to detect race conditions and memory errors in
   kernels

v1.0.3 (2015-03-18)
^^^^^^^^^^^^^^^^^^^

**Bug fixes**

-  Enable builds with intel MPI
-  Silence warnings coming from boost and python headers

v1.0.2 (2015-01-21)
^^^^^^^^^^^^^^^^^^^

**Bug fixes**

-  Fixed a bug where ``linear_interp`` would not take a floating point
   value for *zero*
-  Provide more useful error messages when cuda drivers are not present
-  Assume device count is 0 when ``cudaGetDeviceCount()`` returns an
   error
-  Link to python statically when ``ENABLE_STATIC=on``
-  Misc documentation updates

v1.0.1 (2014-09-09)
^^^^^^^^^^^^^^^^^^^

**Bug fixes**

1.  Fixed bug where error messages were truncated and HOOMD exited with
    a segmentation fault instead (e.g. on Blue Waters)
2.  Fixed bug where plug-ins did not load on Blue Waters
3.  Fixed compile error with gcc4.4 and cuda5.0
4.  Fixed syntax error in ``read_snapshot()``
5.  Fixed a bug where ``init.read_xml throwing`` an error (or any other
    command outside of ``run()``) would hang in MPI runs
6.  Search the install path for hoomd_script - enable the hoomd
    executable to be outside of the install tree (useful with cray
    aprun)
7.  Fixed CMake 3.0 warnings
8.  Removed dependancy on tr1/random
9.  Fixed a bug where ``analyze.msd`` ignored images in the r0_file
10. Fixed typos in ``pair.gauss`` documentation
11. Fixed compile errors on Ubuntu 12.10
12. Fix failure of ``integrate.nvt`` to reach target temperature in
    analyze.log. The fix is a new symplectic MTK integrate.nvt
    integrator. Simulation results in hoomd v1.0.0 are correct, just the
    temperature and velocity outputs are off slightly.
13. Remove MPI from Mac OS X dmg build.
14. Enable ``import hoomd_script as ...``

*Other changes*

1. Added default compile flag -march=native
2. Support CUDA 6.5
3. Binary builds for CentOS/RHEL 6, Fedora 20, Ubuntu 14.04 LTS, and
   Ubuntu 12.04 LTS.

Version 1.0.0 (2014-05-25)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*New features*

-  Support for python 3
-  New NPT integrator capable of flexible coupling schemes
-  Triclinic unit cell support
-  MPI domain decomposition
-  Snapshot save/restore
-  Autotune block sizes at run time
-  Improve performance in small simulation boxes
-  Improve performance with smaller numbers of particles per GPU
-  Full double precision computations on the GPU (compile time option
   must be enabled, binary builds provided on the download page are
   single precision)
-  Tabulated bond potential ``bond.table``
-  Tabulated angle potential ``angle.table``
-  Tabulated dihedral potental ``dihedral.table``
-  ``update.box_resize`` now accepts ``period=None`` to trigger an
   immediate update of the box without creating a periodic updater
-  ``update.box_resize`` now replaces *None* arguments with the current
   box parameters
-  ``init.create_random`` and ``init.create_random_polymers`` can now
   create random configurations in triclinc and 2D boxes
-  ``init.create_empty`` can now create triclinic boxes
-  particle, bond, angle, dihedral, and impropers types can now be named
   in ``init.create_empty``
-  ``system.replicate`` command replicates the simulation box

*Bug fixes*

-  Fixed a bug where init.create_random_polymers failed when lx,ly,lz
   were not equal.
-  Fixed a bug in init.create_random_polymers and init.create_random
   where the separation radius was not accounted for correctly
-  Fixed a bug in bond.\* where random crashes would occur when more
   than one bond type was defined
-  Fixed a bug where dump.dcd did not write the period to the file

*Changes that may require updated job scripts*

-  ``integrate.nph``: A time scale ``tau_p`` for the relaxation of the
   barostat is now required instead of the barostat mass *W* of the
   previous release. The time scale is the relaxation time the barostat
   would have at an average temperature ``T_0 = 1``, and it is related
   to the internally used (Andersen) Barostat mass *W* via
   ``W = d N T_0 tau_p^2``, where *d* is the dimensionsality and *N* the
   number of particles.
-  ``sorter`` and ``nlist`` are now modules, not variables in the
   ``__main__`` namespace.
-  Data proxies function correctly in MPI simulations, but are extremely
   slow. If you use ``init.create_empty``, consider separating the
   generation step out to a single rank short execution that writes an
   XML file for the main run.
-  ``update.box_resize(Lx=...)`` no longer makes cubic box updates,
   instead it will keep the current **Ly** and **Lz**. Use the ``L=...``
   shorthand for cubic box updates.
-  All ``init.*`` commands now take ``data.boxdim`` objects, instead of
   ``hoomd.boxdim`` (or *3-tuples*). We strongly encourage the use of
   explicit argument names for ``data.boxdim()``. In particular, if
   ``hoomd.boxdim(123)`` was previously used to create a cubic box, it
   is now required to use ``data.boxdim(L=123)`` (CORRECT) instead of
   ``data.boxdim(123)`` (INCORRECT), otherwise a box with unit
   dimensions along the y and z axes will be created.
-  ``system.dimensions`` can no longer be set after initialization.
   System dimensions are now set during initialization via the
   ``data.boxdim`` interface. The dimensionality of the system can now
   be queried through ``system.box``.
-  ``system.box`` no longer accepts 3-tuples. It takes ``data.boxdim``
   objects.
-  ``system.dimensions`` no longer exists. Query the dimensionality of
   the system from ``system.box``. Set the dimensionality of the system
   by passing an appropriate ``data.boxdim`` to an ``init`` method.
-  ``init.create_empty`` no longer accepts ``n_*_types``. Instead, it
   now takes a list of strings to name the types.

*Deprecated*

-  Support for G80, G200 GPUs.
-  ``dump.bin`` and ``read.bin``. These will be removed in v1.1 and
   replaced with a new binary format.

*Removed*

-  OpenMP mult-core execution (replaced with MPI domain decomposition)
-  ``tune.find_optimal_block_size`` (replaced by Autotuner)

v0.x
----

Version 0.11.3 (2013-05-10)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fixed a bug where charge.pppm could not be used after init.reset()
-  Data proxies can now set body angular momentum before the first run()
-  Fixed a bug where PPPM forces were incorrect on the GPU

Version 0.11.2 (2012-12-19)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*New features*

-  Block sizes tuned for K20

*Bug fixes*

-  Warn user that PPPM ignores rigid body exclusions
-  Document that proxy iterators need to be deleted before init.reset()
-  Fixed a bug where body angular momentum could not be set
-  Fixed a bug where analyze.log would report nan for the pressure
   tensor in nve and nvt simulations

Version 0.11.1 (2012-11-2)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*New features*

-  Support for CUDA 5.0
-  Binary builds for Fedora 16 and OpenSUSE 12.1
-  Automatically specify /usr/bin/gcc to nvcc when the configured gcc is
   not supported

*Bug fixes*

-  Fixed a compile error with gcc 4.7
-  Fixed a bug where PPPM forces were incorrect with neighborlist
   exclusions
-  Fixed an issue where boost 1.50 and newer were not detected properly
   when BOOST_ROOT is set
-  Fixed a bug where accessing force data in python prevented
   init.reset() from working
-  Fixed a bug that prevented pair.external from logging energy
-  Fixed a unit test that failed randomly

Version 0.11.0 (2012-07-27)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*New features*

1.  Support for Kepler GPUs (GTX 680)
2.  NPH integration (*integrate.nph*)
3.  Compute full pressure tensor
4.  Example plugin for new bond potentials
5.  New syntax for bond coefficients: *bond.bond_coeff.set(‘type’,
    params)*
6.  New external potential: *external.periodic* applies a periodic
    potential along one direction (uses include inducing lamellar phases
    in copolymer systems)
7.  Significant performance increases when running *analyze.log*,
    *analyze.msd*, *update.box_resize*, *update.rescale_temp*, or
    *update.zero_momentum* with a small period
8.  Command line options may now be overwritten by scripts, ex:
    *options.set_gpu(2)*
9.  Added *–user* command line option to allow user defined options to
    be passed into job scripts, ex: *–user=“-N=5 -phi=0.56”*
10. Added *table.set_from_file* method to enable reading table based
    pair potentials from a file
11. Added *–notice-level* command line option to control how much extra
    information is printed during a run. Set to 0 to disable, or any
    value up to 10. At 10, verbose debugging information is printed.
12. Added *–msg-file* command line option which redirects the message
    output to a file
13. New pair potential *pair.force_shifted_lj* : Implements
    http://dx.doi.org/10.1063/1.3558787

*Bug fixes*

1. Fixed a bug where FENE bonds were sometimes computed incorrectly
2. Fixed a bug where pressure was computed incorrectly when using
   pair.dpd or pair.dpdlj
3. Fixed a bug where using OpenMP and CUDA at the same time caused
   invalid memory accesses
4. Fixed a bug where RPM packages did not work on systems where the CUDA
   toolkit was not installed
5. Fixed a bug where rigid body velocities were not set from python
6. Disabled OpenMP builds on Mac OS X. HOOMD-blue w/ openmp enabled
   crashes due to bugs in Apple’s OpenMP implementation.
7. Fixed a bug that allowed users to provide invalid rigid body data and
   cause a seg fault.
8. Fixed a bug where using PPPM resulted in error messages on program
   exit.

*API changes*

1.  Bond potentials rewritten with template evaluators
2.  External potentials use template evaluators
3.  Complete rewrite of ParticleData - may break existing plugins
4.  Bond/Angle/Dihedral data structures rewritten

    -  The GPU specific data structures are now generated on the GPU

5.  DPDThermo and DPDLJThermo are now processed by the same template
    class
6.  Headers that cannot be included by nvcc now throw an error when they
    are
7.  CUDA 4.0 is the new minimum requirement
8.  Rewrote BoxDim to internally handle minimum image conventions
9.  HOOMD now only compiles ptx code for the newest architecture, this
    halves the executable file size
10. New Messenger class for global control of messages printed to the
    screen / directed to a file.

*Testing changes*

1. Automated test suite now performs tests on OpenMPI + CUDA builds
2. Valgrind tests added back into automated test suite
3. Added CPU test in bd_ridid_updater_tests
4. ctest -S scripts can now set parallel makes (with cmake > 2.8.2)

Version 0.10.1 (2012-02-10)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Add missing entries to credits page
2. Add ``dist_check`` option to neighbor list. Can be used to force
   neighbor list builds at a specified frequency (useful in profiling
   runs with nvvp).
3. Fix typos in ubuntu compile documentation
4. Add missing header files to hoomd.h
5. Add torque to the python particle data access API
6. Support boost::filesystem API v3
7. Expose name of executing gpu, n_cpu, hoomd version, git sha1, cuda
   version, and compiler version to python
8. Fix a bug where multiple ``nvt_rigid`` or ``npt_rigid`` integrators
   didn’t work correctly
9. Fix missing pages in developer documentation

Version 0.10.0 (2011-12-14)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*New features*

1.  Added *pair.dpdlj* which uses the DPD thermostat and the
    Lennard-Jones potential. In previous versions, this could be
    accomplished by using two pair commands but at the cost of reduced
    performance.
2.  Additional example scripts are now present in the documentation. The
    example scripts are cross-linked to the commands that are used in
    them.
3.  Most dump commands now accept the form:
    *dump.ext(filename=“filename.ext”)* which immediately writes out
    filename.ext.
4.  Added *vis* parameter to dump.xml which enables output options
    commonly used in files written for the purposes of visulization.
    dump.xml also now accepts parameters on the instantiation line.
    Combined with the previous feature, *dump.xml(filename=“file.xml”,
    vis=True)* is now a convenient short hand for what was previously

    .. code::

       xml = dump.xml()
       xml.set_params(position = True, mass = True, diameter = True,
                             type = True, bond = True, angle = True,
                             dihedral = True, improper = True, charge = True)
       xml.write(filename="file.xml")

5.  Specify rigid bodies in XML input files
6.  Simulations that contain rigid body constraints applied to groups of
    particles in BDNVT, NVE, NVT, and NPT ensembles.

    -  *integrate.bdnvt_rigid*
    -  *integrate.nve_rigid*
    -  *integrate.nvt_rigid*
    -  *integrate.npt_rigid*

7.  Energy minimization of rigid bodies
    (*integrate.mode_minimize_rigid_fire*)
8.  Existing commands are now rigid-body aware

    -  update.rescale_temp
    -  update.box_resize
    -  update.enforce2d
    -  update.zero_momentum

9.  NVT integration using the Berendsen thermostat
    (*integrate.berendsen*)
10. Bonds, angles, dihedrals, and impropers can now be created and
    deleted with the python data access API.
11. Attribution clauses added to the HOOMD-blue license.

*Changes that may break existing job scripts*

1. The *wrap* option to *dump.dcd* has been changed to *unwrap_full* and
   its meaning inverted. *dump.dcd* now offers two options for
   unwrapping particles, *unwrap_full* fully unwraps particles into
   their box image and *unwrap_rigid* unwraps particles in rigid bodies
   so that bodies are not broken up across a box boundary.

*Bug/fixes small enhancements*

1.  Fixed a bug where launching hoomd on mac os X 10.5 always resulted
    in a bus error.
2.  Fixed a bug where DCD output restricted to a group saved incorrect
    data.
3.  force.constant may now be applied to a group of particles, not just
    all particles
4.  Added C++ plugin example that demonstrates how to add a pair
    potential in a plugin
5.  Fixed a bug where box.resize would always transfer particle data
    even in a flat portion of the variant
6.  OpenMP builds re-enabled on Mac OS X
7.  Initial state of integrate.nvt and integrate.npt changed to decrease
    oscillations at startup.
8.  Fixed a bug where the polymer generator would fail to initialize
    very long polymers
9.  Fixed a bug where images were passed to python as unsigned ints.
10. Fixed a bug where dump.pdb wrote coordinates in the wrong order.
11. Fixed a rare problem where a file written by dump.xml would not be
    read by init.read_xml due to round-off errors.
12. Increased the number of significant digits written out to dump.xml
    to make them more useful for ad-hoc restart files.
13. Potential energy and pressure computations that slow performance are
    now only performed on those steps where the values are actually
    needed.
14. Fixed a typo in the example C++ plugin
15. Mac build instructions updated to work with the latest version of
    macports
16. Fixed a bug where set_period on any dump was ineffective.
17. print_status_line now handles multiple lines
18. Fixed a bug where using bdnvt tally with per type gammas resulted in
    a race condition.
19. Fix an issue where ENABLE_CUDA=off builds gave nonsense errors when
    –mode=gpu was requested.
20. Fixed a bug where dumpl.xml could produce files that init.xml would
    not read
21. Fixed a typo in the example plugin
22. Fix example that uses hoomd as a library so that it compiles.
23. Update maintainer lines
24. Added message to nlist exclusions that notifies if diameter or body
    exclusions are set.
25. HOOMD-blue is now hosted in a git repository
26. Added bibtex bibliography to the user documentation
27. Converted user documentation examples to use doxygen auto
    cross-referencing ``\example`` commands
28. Fix a bug where particle data is not released in dump.binary
29. ENABLE_OPENMP can now be set in the ctest builds
30. Tuned block sizes for CUDA 4.0
31. Removed unsupported GPUS from CUDA_ARCH_LIST

Version 0.9.2 (2011-04-04)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note:* only major changes are listed here.

*New features*

1. *New exclusion option:* Particles can now be excluded from the
   neighbor list based on diameter consistent with pair.slj.
2. *New pair coeff syntax:* Coefficients for multiple type pairs can be
   specified conveniently on a single line.

   .. code::

      coeff.set(['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D'], epsilon=1.0)

3. *New documentation:* HOOMD-blue’s system of units is now fully
   documented, and every coefficient in the documentation is labeled
   with the appropriate unit.
4. *Performance improvements:* Performance has been significantly
   boosted for simulations of medium sized systems (5,000-20,000
   particles). Smaller performance boosts were made to larger runs.
5. *CUDA 3.2 support:* HOOMD-blue is now fully tested and performance
   tuned for use with CUDA 3.2.
6. *CUDA 4.0 support:* HOOMD-blue compiles with CUDA 4.0 and passes
   initial tests.
7. *New command:* tune.r_buff performs detailed auto-tuning of the
   r_buff neighborlist parameter.
8. *New installation method:* RPM, DEB, and app bundle packages are now
   built for easier installation
9. *New command:* charge.pppm computes the full long range electrostatic
   interaction using the PPPM method

*Bug/fixes small enhancements*

1.  Fixed a bug where the python library was linked statically.
2.  Added the PYTHON_SITEDIR setting to allow hoomd builds to install
    into the native python site directory.
3.  FIRE energy minimization convergence criteria changed to require
    both energy *and* force to converge
4.  Clarified that groups are static in the documentation
5.  Updated doc comments for compatibility with Doxygen#7.3
6.  system.particles.types now lists the particle types in the
    simulation
7.  Creating a group of a non-existant type is no longer an error
8.  Mention XML file format for walls in wall.lj documentation
9.  Analyzers now profile themselves
10. Use ``\n`` for newlines in dump.xml - improves
    performance when writing many XML files on a NFS file system
11. Fixed a bug where the neighbor list build could take an
    exceptionally long time (several seconds) to complete the first
    build.
12. Fixed a bug where certain logged quantities always reported as 0 on
    the first step of the simulation.
13. system.box can now be used to read and set the simulation box size
    from python
14. Numerous internal API updates
15. Fixed a bug the resulted in incorrect behavior when using
    integrate.npt on the GPU.
16. Removed hoomd launcher shell script. In non-sitedir installs,
    ${HOOMD_ROOT}/bin/hoomd is now the executable itself
17. Creating unions of groups of non-existent types no longer produces a
    seg fault
18. hoomd now builds on all cuda architectures. Modify CUDA_ARCH_LIST in
    cmake to add or remove architectures from the build
19. hoomd now builds with boost#46.0
20. Updated hoomd icons to maize/blue color scheme
21. hoomd xml file format bumped to#3, adds support for charge.
22. FENE and harmonic bonds now handle 0 interaction parameters and 0
    length bonds more gracefully
23. The packaged plugin template now actually builds and installs into a
    recent build of hoomd

Version 0.9.1 (2010-10-08)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note:* only major changes are listed here.

*New features*

1. *New constraint*: constrain.sphere constrains a group of particles to
   the surface of a sphere
2. *New pair potential/thermostat*: pair.dpd implements the standard DPD
   conservative, random, and dissipative forces
3. *New pair potential*: pair.dpd_conservative applies just the
   conservative DPD potential
4. *New pair potential*: pair.eam implements the Embedded Atom Method
   (EAM) and supports both *alloy* and *FS* type computations.
5. *Faster performance*: Cell list and neighbor list code has been
   rewritten for performance.

   -  In our benchmarks, *performance increases* ranged from *10-50%*
      over HOOMD-blue 0.9.0. Simulations with shorter cutoffs tend to
      attain a higher performance boost than those with longer cutoffs.
   -  We recommended that you *re-tune r_buff* values for optimal
      performance with 0.9.1.
   -  Due to the nature of the changes, *identical runs* may produce
      *different trajectories*.

6. *Removed limitation*: The limit on the number of neighbor list
   exclusions per particle has been removed. Any number of exclusions
   can now be added per particle. Expect reduced performance when adding
   excessive numbers of exclusions.

*Bug/fixes small enhancements*

1.  Pressure computation is now correct when constraints are applied.
2.  Removed missing files from hoomd.h
3.  pair.yukawa is no longer referred to by “gaussian” in the
    documentation
4.  Fermi GPUs are now prioritized over per-Fermi GPUs in systems where
    both are present
5.  HOOMD now compiles against CUDA 3.1
6.  Momentum conservation significantly improved on compute#x hardware
7.  hoomd plugins can now be installed into user specified directories
8.  Setting r_buff=0 no longer triggers exclusion list updates on every
    step
9.  CUDA 2.2 and older are no longer supported
10. Workaround for compiler bug in 3.1 that produces extremely high
    register usage
11. Disabled OpenMP compile checks on Mac OS X
12. Support for compute 2.1 devices (such as the GTX 460)

Version 0.9.0 (2010-05-18)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note:* only major changes are listed here.

*New features*

1.  *New pair potential*: Shifted LJ potential for particles of varying
    diameters (pair.slj)
2.  *New pair potential*: Tabulated pair potential (pair.table)
3.  *New pair potential*: Yukawa potential (pair.yukawa)
4.  *Update to pair potentials*: Most pair potentials can now accept
    different values of r_cut for different type pairs. The r_cut
    specified in the initial pair.**\* command is now treated as the
    default r_cut, so no changes to scripts are necessary.
5.  *Update to pair potentials*: Default pair coeff values are now
    supported. The parameter alpha for lj now defaults to#0, so there is
    no longer a need to specify it for a majority of simulations.
6.  *Update to pair potentials*: The maximum r_cut needed for the
    neighbor list is now determined at the start of each run(). In
    simulations where r_cut may decrease over time, increased
    performance will result.
7.  *Update to pair potentials*: Pair potentials are now specified via
    template evaluator classes. Adding a new pair potential to hoomd now
    only requires a small amount of additional code.
8.  *Plugin API* : Advanced users/developers can now write, install, and
    use plugins for hoomd without needing to modify core hoomd source
    code
9.  *Particle data access*: User-level hoomd scripts can now directly
    access the particle data. For example, one can change all particles
    in the top half of the box to be type B:

    .. code::

       top = group.cuboid(name="top", zmin=0)
       for p in top:
           p.type = 'B'

    . *All* particle data including position, velocity, type, ‘’et
    cetera’’, can be read and written in this manner. Computed forces
    and energies can also be accessed in a similar way.
10. *New script command*: init.create_empty() can be used in conjunction
    with the particle data access above to completely initialize a
    system within the hoomd script.
11. *New script command*: dump.bin() writes full binary restart files
    with the entire system state, including the internal state of
    integrators.

    -  File output can be gzip compressed (if zlib is available) to save
       space
    -  Output can alternate between two different output files for safe
       crash recovery

12. *New script command*: init.read_bin() reads restart files written by
    dump.bin()
13. *New option*: run() now accepts a quiet option. When True, it
    eliminates the status information printouts that go to stdout.
14. *New example script*: Example 6 demonstrates the use of the particle
    data access routines to initialize a system. It also demonstrates
    how to initialize velocities from a gaussian distribution
15. *New example script*: Example 7 plots the pair.lj potential energy
    and force as evaluated by hoomd. It can trivially be modified to
    plot any potential in hoomd.
16. *New feature*: Two dimensional simulations can now be run in hoomd:
    #259
17. *New pair potential*: Morse potential for particles of varying
    diameters (pair.morse)
18. *New command*: run_upto will run a simulation up to a given time
    step number (handy for breaking long simulations up into many
    independent jobs)
19. *New feature*: HOOMD on the CPU is now accelerated with OpenMP.
20. *New feature*: integrate.mode_minimize_fire performs energy
    minimization using the FIRE algorithm
21. *New feature*: analyze.msd can now accept an xml file specifying the
    initial particle positions (for restarting jobs)
22. *Improved feature*: analyze.imd now supports all IMD commands that
    VMD sends (pause, kill, change trate, etc.)
23. *New feature*: Pair potentials can now be given names, allowing
    multiple potentials of the same type to be logged separately.
    Additionally, potentials that are disabled and not applied to the
    system dynamics can be optionally logged.
24. *Performance improvements*: Simulation performance has been
    increased across the board, but especially when running systems with
    very low particle number densities.
25. *New hardware support*: 0.9.0 and newer support Fermi GPUs
26. *Deprecated hardware support*: 0.9.x might continue run on compute#1
    GPUs but that hardware is no longer officially supported
27. *New script command*: group.tag_list() takes a python list of
    particle tags and creates a group
28. *New script command*: compute.thermo() computes thermodynamic
    properties of a group of particles for logging
29. *New feature*: dump.dcd can now optionally write out only those
    particles that belong to a specified group

*Changes that will break jobs scripts written for 0.8.x*

1. Integration routines have changed significantly to enable new use
   cases. Where scripts previously had commands like:

   .. code::

      integrate.nve(dt=0.005)

   they now need

   .. code::

      all = group.all()
      integrate.mode_standard(dt=0.005)
      integrate.nve(group=all)

   . Integrating only specific groups of particles enables simulations
   to fix certain particles in place or integrate different parts of the
   system at different temperatures, among many other possibilities.
2. sorter.set_params no longer takes the ‘’bin_width’’ argument. It is
   replaced by a new ‘’grid’’ argument, see the documentation for
   details.
3. conserved_quantity is no longer a quantity available for logging.
   Instead log the nvt reservoir energy and compute the total conserved
   quantity in post processing.

*Bug/fixes small enhancements*

1.  Fixed a bug where boost#38 is not found on some machines
2.  dump.xml now has an option to write particle accelerations
3.  Fixed a bug where periods like 1e6 were not accepted by updaters
4.  Fixed a bug where bond.fene forces were calculated incorrectly
    between particles of differing diameters
5.  Fixed a bug where bond.fene energies were computed incorrectly when
    running on the GPU
6.  Fixed a bug where comments in hoomd xml files were not ignored as
    they aught to be: #331
7.  It is now possible to prevent bond exclusions from ever being added
    to the neighbor list: #338
8.  init.create_random_polymers can now generate extremely dense systems
    and will warn the user about large memory usage
9.  variant.linear_interp now accepts a user-defined zero (handy for
    breaking long simulations up into many independent jobs)
10. Improved installation and compilation documentation
11. Integration methods now silently ignore when they are given an empty
    group
12. Fixed a bug where disabling all forces resulted in some forces still
    being applied
13. Integrators now behave in a reasonable way when given empty groups
14. Analyzers now accept a floating point period
15. run() now aborts immediately if limit_hours=0 is specified.
16. Pair potentials that diverge at r=0 will no longer result in invalid
    simulations when the leading coefficients are set to zero.
17. integrate.bdnvt can now tally the energy transferred into/out of the
    “reservoir”, allowing energy conservation to be monitored during bd
    simulation runs.
18. Most potentials now prevent NaN results when computed for
    overlapping particles
19. Stopping a simulation from a callback or time limit no longer
    produces invalid simulations when continued
20. run() commands limited with limit_hours can now be set to only stop
    on given timestep multiples
21. Worked around a compiler bug where pair.morse would crash on Fermi
    GPUs
22. ULF stability improvements for G200 GPUs.

Version 0.8.2 (2009-09-10)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note:* only major changes are listed here.

*New features*

1.  Quantities that vary over time can now be specified easily in
    scripts with the variant.linear_interp command.
2.  Box resizing updater (update.box_resize) command that uses the time
    varying quantity command to grow or shrink the simulation box.
3.  Individual run() commands can be limited by wall-clock time
4.  Angle forces can now be specified
5.  Dihedral forces can now be specified
6.  Improper forces can now be specified
7.  1-3 and 1-4 exclusions from the cutoff pair force can now be chosen
8.  New command line option: –minimize-cpu-usage cuts the CPU usage of
    HOOMD down to 10% of one CPU core while only decreasing overall
    performance by 10%
9.  Major changes have been made in the way HOOMD chooses the device on
    which to run (all require CUDA 2.2 or newer)

    -  there are now checks that an appropriate NVIDIA drivers is
       installed
    -  running without any command line options will now correctly
       revert to running on the CPU if no capable GPUs are installed
    -  when no gpu is explicitly specified, the default choice is now
       prioritized to choose the fastest GPU and one that is not
       attached to a display first
    -  new command line option: –ignore-display-gpu will prevent HOOMD
       from executing on any GPU attached to a display
    -  HOOMD now prints out a short description of the GPU(s) it is
       running on
    -  on linux, devices can be set to compute-exclusive mode and HOOMD
       will then automatically choose the first free GPU (see the
       documentation for details)

10. nlist.reset_exclusions command to control the particles that are
    excluded from the neighbor list

*Bug/fixes small enhancements*

1.  Default block size change to improve stability on compute#3 devices
2.  ULF workaround on GTX 280 now works with CUDA 2.2
3.  Standalone benchmark executables have been removed and replaced by
    in script benchmarking commands
4.  Block size tuning runs can now be performed automatically using the
    python API and results can be saved on the local machine
5.  Fixed a bug where GTX 280 bug workarounds were not properly applied
    in CUDA 2.2
6.  The time step read in from the XML file can now be optionally
    overwritten with a user-chosen one
7.  Added support for CUDA 2.2
8.  Fixed a bug where the WCA forces included in bond.fene had an
    improper cutoff
9.  Added support for a python callback to be executed periodically
    during a run()
10. Removed demos from the hoomd downloads. These will be offered
    separately on the webpage now to keep the required download size
    small.
11. documentation improvements
12. Significantly increased performance of dual-GPU runs when build with
    CUDA 2.2 or newer
13. Numerous stability and performance improvements
14. Temperatures are now calculated based on 3N-3 degrees of freedom.
    See #283 for a more flexible system that is coming in the future.
15. Emulation mode builds now work on systems without an NVIDIA card
    (CUDA 2.2 or newer)
16. HOOMD now compiles with CUDA 2.3
17. Fixed a bug where uninitialized memory was written to dcd files
18. Fixed a bug that prevented the neighbor list on the CPU from working
    properly with non-cubic boxes
19. There is now a compile time hack to allow for more than 4 exclusions
    per particle
20. Documentation added to aid users in migrating from LAMMPS
21. hoomd_script now has an internal version number useful for third
    party scripts interfacing with it
22. VMD#8.7 is now found by the live demo scripts
23. live demos now run in vista 64-bit
24. init.create_random_polymers can now create polymers with more than
    one type of bond

Version 0.8.1 (2009-03-24)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note:* only major changes are listed here.

*New features*

1.  Significant performance enhancements
2.  New build option for compiling on UMich CAC clusters:
    ENABLE_CAC_GPU_ID compiles HOOMD to read in the *$CAC_GPU_ID*
    environment variable and use it to determine which GPUs to execute
    on. No –gpu command line required in job scripts any more.
3.  Particles can now be assigned a *non-unit mass*
4.  *init.reset()* command added to allow for the creation of a looped
    series of simulations all in python
5.  *dump.pdb()* command for writing PDB files
6.  pair.lj now comes with an option to *shift* the potential energy to
    0 at the cutoff
7.  pair.lj now comes with an opiton to *smoothly switch* both the
    *potential* and *force* to 0 at the cutoff with the XPLOR smoothing
    function
8.  *Gaussian pair potential* computation added (pair.gauss)
9.  update and analyze commands can now be given a function to determine
    a non-linear rate to run at
10. analyze.log, and dump.dcd can now append to existing files

*Changes that will break scripts from 0.8.0*

1. *dump.mol2()* has been changed to be more consistent with other dump
   commands. In order to get the same result as the previous behavior,
   replace

   .. code::

       dump.mol2(filename="file.mol2")

   with

   .. code::

       mol2 = dump.mol2()
       mol2.write(filename="file.mol2")

2. Grouping commands have been moved to their own package for
   organizational purposes. *group_all()* must now be called as
   *group.all()* and similarly for tags and type.

*Bug/fixes small enhancements*

1.  Documentation updates
2.  DCD file writing no longer crashes HOOMD in windows
3.  !FindBoost.cmake is patched upstream. Use CMake 2.6.3 if you need
    BOOST_ROOT to work correctly
4.  Validation tests now run with –gpu_error_checking
5.  ULF bug workarounds are now enabled only on hardware where they are
    needed. This boosts performance on C1060 and newer GPUs.
6.  !FindPythonLibs now always finds the shared python libraries, if
    they exist
7.  “make package” now works fine on mac os x
8.  Fixed erroneously reported dangerous neighbor list builds when using
    –mode=cpu
9.  Small tweaks to the XML file format.
10. Numerous performance enhancements
11. Workaround for ULF on compute#1 devices in place
12. dump.xml can now be given the option “all=true” to write all fields
13. total momentum can now be logged by analyze.log
14. HOOMD now compiles with boost#38 (and hopefully future versions)
15. Updaters can now be given floating point periods such as 1e5
16. Additional warnings are now printed when HOOMD is about to allocate
    a large amount of memory due to the specification of an extremely
    large box size
17. run() now shows up in the documentation index
18. Default sorter period is now 100 on CPUs to improve performance on
    chips with small caches

Version 0.8.0 (2008-12-22)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note:* only major changes are listed here.

*New features*

1. Addition of FENE bond potential
2. Addition of update.zero_momentum command to zero a system’s linear
   momentum
3. Brownian dynamics integration implemented
4. Multi-GPU simulations
5. Particle image flags are now tracked. analyze.msd command added to
   calculate the mean squared displacement.

*Changes that will break scripts from 0.7.x*

1. analyze.log quantity names have changed

*Bug/fixes small enhancements*

1.  Performance of the neighbor list has been increased significantly on
    the GPU (overall performance improvements are approximately 10%)
2.  Profile option added to the run() command
3.  Warnings are now correctly printed when negative coefficients are
    given to bond forces
4.  Simulations no longer fail on G200 cards
5.  Mac OS X binaries will be provided for download: new documentation
    for installing on Mac OS x has been written
6.  Two new demos showcasing large systems
7.  Particles leaving the simulation box due to bad initial conditions
    now generate an error
8.  win64 installers will no longer attempt to install on win32 and
    vice-versa
9.  neighborlist check_period now defaults to 1
10. The elapsed time counter in run() now continues counting time over
    multiple runs.
11. init.create_random_polymers now throws an error if the bond length
    is too small given the specified separation radii
12. Fixed a bug where a floating point value for the count field in
    init.create_random_polymers produced an error
13. Additional error checking to test if particles go NaN
14. Much improved status line printing for identifying hoomd_script
    commands
15. Numerous documentation updates
16. The VS redistributable package no longer needs to be installed to
    run HOOMD on windows (these files are distributed with HOOMD)
17. Now using new features in doxygen#5.7 to build pdf user
    documentation for download.
18. Performance enhancements of the Lennard-Jones pair force
    computation, thanks to David Tarjan
19. A header prefix can be added to log files to make them more gnuplot
    friendly
20. Log quantities completely revamped. Common quantities (i.e. kinetic
    energy, potential energy can now be logged in any simulation)
21. Particle groups can now be created. Currently only analyze.msd makes
    use of them.
22. The CUDA toolkit no longer needs to be installed to run a packaged
    HOOMD binary in windows.
23. User documentation can now be downloaded as a pdf.
24. Analyzers and updaters now count time 0 as being the time they were
    created, instead of time step 0.
25. Added job test scripts to aid in validating HOOMD
26. HOOMD will now build with default settings on a linux/unix-like OS
    where the boost static libraries are not installed, but the dynamic
    ones are.

Version 0.7.1 (2008-09-12)
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Fixed bug where extremely large box dimensions resulted in an
   argument error - ticket:118
2. Fixed bug where simulations ran incorrectly with extremely small box
   dimensions - ticket:138

Version 0.7.0 (2008-08-12)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note:* only major changes are listed here.

1.  Stability and performance improvements.
2.  Cleaned up the hoomd_xml file format.
3.  Improved detection of errors in hoomd_xml files significantly.
4.  Users no longer need to manually specify HOOMD_ROOT, unless their
    installation is non-standard
5.  Particle charge can now be read in from a hoomd_xml file
6.  Consistency changes in the hoomd_xml file format: HOOMD 0.6.0 XML
    files are not compatible. No more compatibility breaking changes are
    planned after 0.7.0
7.  Enabled parallel builds in MSVC for faster compilation times on
    multicore systems
8.  Numerous small bug fixes
9.  New force compute for implementing walls
10. Documentation updates
11. Support for CUDA 2.0
12. Bug fixed allowing simulations with no integrator
13. Support for boost#35.0
14. Cleaned up GPU code interface
15. NVT integrator now uses tau (period) instead of Q (the mass of the
    extra degree of freedom).
16. Added option to NVE integration to limit the distance a particle
    moves in a single time step
17. Added code to dump system snapshots in the DCD file format
18. Particle types can be named by strings
19. A snapshot of the initial configuration can now be written in the
    .mol2 file format
20. The default build settings now enable most of the optional features
21. Separated the user and developer documentation
22. Mixed polymer systems can now be generated inside HOOMD
23. Support for CMake 2.6.0
24. Wrote the user documentation
25. GPU selection from the command line
26. Implementation of the job scripting system
27. GPU can now handle neighbor lists that overflow
28. Energies are now calculated
29. Added a logger for logging energies during a simulation run
30. Code now actually compiles on Mac OS X
31. Benchmark and demo scripts now use the new scripting system
32. Consistent error message format that is more visible.
33. Multiple types of bonds each with the own coefficients are now
    supported
34. Added python scripts to convert from HOOMD’s XML file format to
    LAMMPS input and dump files
35. Fixed a bug where empty xml nodes in input files resulted in an
    error message
36. Fixed a bug where HOOMD seg faulted when a particle left the
    simulation , vis=True)\* is now a convenient short hand for what was
    previously box now works fine on mac os x
37. Fixed erroneously reported dangerous neighbor list builds when using
    –mode=cpu
38. Small tweaks to the XML file format.
39. Numerous performance enhancements
40. Workaround for ULF on compute#1 devices in place
41. dump.xml can now be given the option
