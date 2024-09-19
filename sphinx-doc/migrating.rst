.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Migrating to the latest version
===============================

Migrating to HOOMD v5
---------------------

Removed functionalities
^^^^^^^^^^^^^^^^^^^^^^^

HOOMD-blue v5 removes functionalities deprecated in v4.x releases:

* ``_InternalCustomUpdater.update``
* ``_InternalCustomTuner.tune``
* ``_InternalCustomWriter.write``
* ``HDF5Log.write``
* ``hoomd.snapshot.from_gsd_snapshot``

  * Use `hoomd.Snapshot.from_gsd_frame`.
* ``hoomd.util.GPUNotAvailableError``

  * use ``hoomd.error.GPUNotAvailableError``.


Migrating to HOOMD v4
---------------------

Breaking changes to existing functionalities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For some functionalities, you will need to update your scripts to use a new API:

* ``hoomd.md.dihedral.Harmonic``

  * Use `hoomd.md.dihedral.Periodic`.

* ``charges`` key in `Rigid.body <hoomd.md.constrain.Rigid.body>`.

  * Pass charges to `Rigid.create_bodies <hoomd.md.constrain.Rigid.create_bodies>` or set in
    the system state.

* ``diameters`` key in `Rigid.body <hoomd.md.constrain.Rigid.body>`.

  * Set diameters in system state.

* ``hoomd.md.methods.NVE``.

  * Use `hoomd.md.methods.ConstantVolume` with ``thermostat=None``.

* ``hoomd.md.methods.NVT``.

  * Use `hoomd.md.methods.ConstantVolume` with a `hoomd.md.methods.thermostats.MTTK` thermostat.

* ``hoomd.md.methods.Berendsen``.

  * Use `hoomd.md.methods.ConstantVolume` with a `hoomd.md.methods.thermostats.Berendsen`
    thermostat.

* ``hoomd.md.methods.NPH``.

  * Use `hoomd.md.methods.ConstantPressure` with ``thermostat=None``.

* ``hoomd.md.methods.NPT``.

  * Use `hoomd.md.methods.ConstantPressure` with a `hoomd.md.methods.thermostats.MTTK` thermostat.

* ``hoomd.write.GSD.log``.

  * Use `hoomd.write.GSD.logger`.

* ``hoomd.mesh.Mesh().triangles``.

  * Use ``hoomd.mesh.Mesh().triangulation`` in `hoomd.mesh.Mesh` to define the mesh triangulation.

* ``hoomd.md.pair.Gauss``.

  * Use `hoomd.md.pair.Gaussian`.

* ``hoomd.md.external.wall.Gauss``.

  * Use `hoomd.md.external.wall.Gaussian`.

* ``msg_file`` property and argument in `hoomd.device.Device`.

  * Use `message_filename <hoomd.device.Device.message_filename>`.

* ``sdf`` property of `hoomd.hpmc.compute.SDF`.

  * Use `sdf_compression <hoomd.hpmc.compute.SDF.sdf_compression>`.

* `hoomd.write.GSD` no longer writes ``particles/diameter`` by default.

  * Set `write_diameter <hoomd.write.GSD.write_diameter>` as needed.

* ``alpha`` property of `hoomd.md.methods.Langevin`, `hoomd.md.methods.Brownian`,
  `hoomd.md.methods.OverdampedViscous`, `hoomd.md.methods.rattle.Langevin`,
  `hoomd.md.methods.rattle.Brownian`, and `hoomd.md.methods.rattle.OverdampedViscous`.

  * Use the ``gamma`` property.

* The ``dynamic`` property and argument of `hoomd.write.GSD` no longer enforces ``'property'`` as
  an always dynamic quantity. Users must include ``'property'``, ``'particles/position'`` and/or
  ``'particles/orientation'`` as needed in ``dynamic`` lists that contain other fields.

* `hoomd.write.GSD` aggressively buffers output. Call `hoomd.write.GSD.flush` to write the buffer
  to disk when opening a file for reading that is still open for writing. There is **no need** to
  call ``flush`` in normal workflows when files are closed and then opened later for reading.

Removed functionalities
^^^^^^^^^^^^^^^^^^^^^^^

HOOMD-blue v4 removes functionalities deprecated in v3.x releases:

* ``hoomd.md.pair.aniso.ALJ.mode`` parameter
* ``hoomd.md.pair.aniso.Dipole.mode`` parameter
* ``hoomd.device.GPU.memory_traceback`` parameter

Reintroducing `hoomd.mpcd`
^^^^^^^^^^^^^^^^^^^^^^^^^^

`hoomd.mpcd` was previously available in HOOMD 2, but it was not available in HOOMD 3 because its
API needed to be significantly rewritten. It is reintroduced in HOOMD 4 with all classes and methods
renamed to be consistent with the rest of HOOMD's API. The most significant changes for users are:

* The way MPCD particle data is stored and initialized has changed. MPCD particles are now part of
  `hoomd.State`, so HOOMD and MPCD particles need to be initialized together rather than separately.
* After initialization, most objects now need to be attached to the :class:`hoomd.mpcd.Integrator`,
  similarly to other features migrated from HOOMD 2 to HOOMD 3.
* The `hoomd.mpcd.geometry.ParallelPlates` and `hoomd.mpcd.geometry.PlanarPore` streaming geometries
  have been rotated to the *xy* plane from the *xz* plane.
* MPCD particle sorting is not enabled by default but is still highly recommended for performance.
  Users should explicitly create a `hoomd.mpcd.tune.ParticleSorter` and attach it to the
  :class:`hoomd.mpcd.Integrator`.

Please refer to the module-level documentation and examples for full details of the new API. Some
common changes that you may need to make to your HOOMD 2 scripts are:

.. list-table::
    :header-rows: 1

    * - Feature
      - Change
    * - Create snapshots using ``mpcd.data``
      - Use `hoomd.Snapshot.mpcd`
    * - Specify cell size using ``mpcd.data``
      - The cell size is fixed at 1.0.
    * - Initialize MPCD particles with ``mpcd.init.read_snapshot``
      - Use `hoomd.Simulation.create_state_from_snapshot`
    * - Initialize MPCD particles randomly with ``mpcd.init.make_random``
      - Not currently supported
    * - Initialize HOOMD particles from a file, then add MPCD particles through ``mpcd.init``.
      - Use `hoomd.Snapshot.from_gsd_frame`, add the MPCD particles, then initialize as above
    * - Bounce-back integration of HOOMD particles using ``mpcd.integrate``
      - Use `hoomd.mpcd.methods.BounceBack` with a geometry from `hoomd.mpcd.geometry`
    * - Bounce-back streaming of MPCD particles using ``mpcd.stream``
      - Use `hoomd.mpcd.stream.BounceBack` with a geometry from `hoomd.mpcd.geometry`
    * - Fill geometry with virtual particles using ``mpcd.fill``
      - Use `hoomd.mpcd.fill.GeometryFiller` with a geometry from `hoomd.mpcd.geometry`
    * - Change sorting period of automatically created ``system.sorter``
      - Explicitly create a `hoomd.mpcd.tune.ParticleSorter` with desired period
    * - Have HOOMD automatically validate my streaming geometry fits inside my box
      - No longer performed. Users should make sure their geometries make sense
    * - Have HOOMD automatically validate my particles are inside my streaming geometry
      - Call `hoomd.mpcd.stream.BounceBack.check_mpcd_particles` directly

For developers, the following are the most significant changes to be aware of:

* The MPCD namespace is ``hoomd::mpcd``.
* ``hoomd::mpcd::SystemData`` has been removed. Classes should accept ``hoomd::SystemDefinition``
  instead and use ``SystemDefinition::getMPCDParticleData()``.
* Force and geometry files have been renamed.
* Bounce-back streaming methods are now templated on both geometries and forces, rather than using
  polymorphism for the forces. This means that combinations of geometries and forces need to be
  compiled when new classes are added. CMake can automatically generate the necessary files if new
  geometries and forces are added to the appropriate lists. Python will automatically deduce the
  right C++ class names if standard naming conventions are followed; otherwise, explicit
  registration is required.
* The virtual particle filler design has been refactored to enable other methods for virtual
  particle filling. Fillers that derived from the previous ``hoomd::mpcd::VirtualParticleFiller``
  should inherit from ``hoomd::mpcd::ManualVirtualParticleFiller`` instead.

Compiling
^^^^^^^^^

* HOOMD-blue v4 no longer builds on macOS with ``ENABLE_GPU=on``.
* Use the CMake options ``HOOMD_LONGREAL_SIZE`` and ``HOOMD_SHORTREAL_SIZE`` to control the floating
  point precision of the calculations. These replace the ``SINGLE_PRECISION`` and
  ``HPMC_MIXED_PRECISION`` options from v3.

Components
^^^^^^^^^^

* Remove ``fix_cudart_rpath(_${COMPONENT_NAME})`` from your components ``CMakeLists.txt``
* Use ``LongReal`` and ``ShortReal`` types in new code. ``Scalar`` will be removed in a future
  release (v5 or later).
* Replace any use of ``hpmc::OverlapReal`` with ``ShortReal``.
* Remove ``needsDiameter`` and ``setDiameter`` methods in potential evaluator classes.

Migrating to HOOMD v3
---------------------

HOOMD v3 introduces many breaking changes for both users and developers
in order to provide a cleaner Python interface, enable new functionalities, and
move away from unsupported tools. This guide highlights those changes.

Overview of API changes
^^^^^^^^^^^^^^^^^^^^^^^

HOOMD v3 introduces a completely new API. All classes have been renamed to match
PEP8 naming guidelines and have new or renamed parameters, methods, and
properties. See the tutorials and the Python module documentation for full
class-level details.

Here is a module level overview of features that have been moved or removed:

.. list-table::
   :header-rows: 1

   * - v2 module, class, or method
     - Replaced with
   * - ``hoomd.analyze.log``
     - `hoomd.logging`
   * - ``hoomd.benchmark``
     - *Removed.* Use Python standard libraries for timing.
   * - ``hoomd.cite``
     - *Removed.* See `citing`.
   * - ``hoomd.dump``
     - `hoomd.write`
   * - ``hoomd.compute.thermo``
     - `hoomd.md.compute.ThermodynamicQuantities`
   * - ``hoomd.context.initialize``
     - `hoomd.device.CPU` and `hoomd.device.GPU`
   * - ``hoomd.data``
     - `hoomd.State`
   * - ``hoomd.group``
     - `hoomd.filter`
   * - ``hoomd.init``
     - `hoomd.Simulation` ``create_state_from_`` factory methods
   * - ``hoomd.lattice``
     - *Removed.* Use an external tool.
   * - ``hoomd.meta``
     - `hoomd.logging.Logger`.
   * - ``hoomd.option``
     - *Removed.* Use Python standard libraries for option parsing.
   * - ``hoomd.update``
     - Some classes have been moved to `hoomd.tune`.
   * - ``hoomd.util``
     -  Enable GPU profiling with `hoomd.device.GPU.enable_profiling`.
   * - ``hoomd.hpmc.analyze.sdf``
     - `hoomd.hpmc.compute.SDF`
   * - ``hoomd.hpmc.data``
     - `hoomd.hpmc.integrate.HPMCIntegrator` properties.
   * - ``hoomd.hpmc.util``
     - `hoomd.hpmc.tune`
   * - ``hoomd.md.integrate.mode_standard``
     - `hoomd.md.Integrator`
   * - ``hoomd.md.update.rescale_temp``
     - `hoomd.State.thermalize_particle_momenta`
   * - ``hoomd.md.update.enforce2d``
     - *Removed.* This is not needed.
   * - ``hoomd.md.constrain.sphere``
     - `hoomd.md.manifold.Sphere`
   * - ``hoomd.md.constrain.oneD``
     - *Removed.*
   * - ``hoomd.md.update.constraint_ellipsoid``
     - `hoomd.md.manifold.Ellipsoid`
   * - ``hoomd.jit.patch``
     - `hoomd.hpmc.pair.user`
   * - ``hoomd.jit.external``
     - `hoomd.hpmc.external.user`

Removed functionality
^^^^^^^^^^^^^^^^^^^^^

HOOMD v3 removes old APIs, unused functionality, and features better served by other codes:

:py:mod:`hoomd`:

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
   * - Python 2.7
     - Python >= 3.6
   * - Compute < 6.0 GPUs
     - Compute >= 6.0 GPUs
   * - ``static`` parameter in ``hoomd.dump.gsd``
     - ``dynamic`` parameter
   * - ``set_params`` and other ``set_*`` methods
     - Parameters and type parameters accessed by properties.
   * - ``context.initialize``
     - `device.CPU` / `device.GPU`
   * - ``util.quiet_status`` and ``util.unquiet_status``
     - No longer needed.

``hoomd.deprecated``:

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
   * - ``deprecated.analyze.msd``
     - Offline analysis: e.g. `Freud's msd module <https://freud.readthedocs.io>`_.
   * - ``deprecated.dump.xml``
     - `hoomd.write.GSD`
   * - ``deprecated.dump.pos``
     - `hoomd.write.GSD` with on-demand conversion to ``.pos``.
   * - ``deprecated.init.read_xml``
     - `Simulation.create_state_from_gsd`
   * - ``deprecated.init.create_random``
     - `mBuild <https://github.com/mosdef-hub/mbuild/>`_, `packmol <https://www.ime.unicamp.br/~martinez/packmol/userguide.shtml>`_, or user script.
   * - ``deprecated.init.create_random_polymers``
     - `mBuild <https://github.com/mosdef-hub/mbuild/>`_, `packmol <https://www.ime.unicamp.br/~martinez/packmol/userguide.shtml>`_, or user script.

:py:mod:`hoomd.hpmc`:

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
   * - ``sphere_union::max_members`` parameter
     - no longer needed
   * - ``convex_polyhedron_union``
     - :py:class:`ConvexSpheropolyhedronUnion <hoomd.hpmc.integrate.ConvexSpheropolyhedronUnion>`, ``sweep_radius=0``
   * - ``setup_pos_writer`` member
     - n/a
   * - ``depletant_mode='circumsphere'``
     - no longer needed
   * - ``max_verts`` parameter
     - no longer needed
   * - ``depletant_mode`` parameter
     - no longer needed
   * - ``ntrial`` parameter
     - no longer needed
   * - ``implicit`` boolean parameter
     - set ``fugacity`` non-zero

:py:mod:`hoomd.md`:

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
   * - ``group`` parameter to ``integrate.mode_minimize_fire``
     - Pass group to integration method.
   * - ``alpha`` parameter to ``pair.lj`` and related classes
     - n/a
   * - ``f_list`` and ``t_list`` parameters to ``md.force.active``
     - Per-type ``active_force`` and ``active_torque``
   * - ``md.pair.SLJ``
     - `md.pair.ExpandedLJ` with `hoomd.md.pair.Pair.r_cut` set to ``r_cut(for delta=0) + delta``

``hoomd.cgcmm``:

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
   * - ``cgcmm.angle.cgcmm``
     - no longer needed
   * - ``cgcmm.pair.cgcmm``
     - no longer needed

``hoomd.dem``:

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
   * - DEM pair potentials
     - ALJ pair potential in `hoomd.md.pair.aniso`.

Not yet ported
^^^^^^^^^^^^^^

The following v2 functionalities have not yet been ported to the v3 API. They may be added in a
future 3.x release:

- HPMC box volume move size tuner.

These contributed functionalities rely on the community for support. Please
contact the developers if you have an interest in porting these in a future release:

- ``hoomd.hdf5``
- ``hoomd.metal``
- ``hoomd.mpcd``


Compiling
^^^^^^^^^

* CMake 3.8 or newer is required to build HOOMD v3.0.
* To compile with GPU support, use the option ``ENABLE_GPU=ON``.
* ``UPDATE_SUBMODULES`` no longer exists. Users and developers should use
  ``git clone --recursive``, ``git submodule update`` and ``git submodule sync``
  as appropriate.
* ``COPY_HEADERS`` no longer exists. HOOMD will pull headers from the source directory when needed.
* ``CMAKE_INSTALL_PREFIX`` is set to the Python ``site-packages`` directory (if
  not explicitly set by the user).
* **cereal**, **eigen**, and **pybind11** headers must be provided to build
  HOOMD. See :doc:`installation` for details.
* ``BUILD_JIT`` is replaced with ``ENABLE_LLVM``.

Components
^^^^^^^^^^

* HOOMD now uses native CUDA support in CMake. Use ``CMAKE_CUDA_COMPILER`` to
  specify a specific ``nvcc`` or ``hipcc``. Plugins will require updates to
  ``CMakeLists.txt`` to compile ``.cu`` files.

  - Remove ``CUDA_COMPILE``.
  - Pass ``.cu`` sources directly to ``pybind11_add_module``.
  - Add ``NVCC`` as a compile definition to ``.cu`` sources.

* External components require additional updates to work with v3. See
  ``example_plugin`` for details:

  - Remove ``FindHOOMD.cmake``.
  - Replace ``include(FindHOOMD.cmake)`` with
    ``find_package(HOOMD 3.Y REQUIRED)`` (where 3.Y is the minor version this
    plugin is compatible with).
  - Always force set ``CMAKE_INSTALL_PREFIX`` to ``${HOOMD_INSTALL_PREFIX}``.
  - Replace ``PYTHON_MODULE_BASE_DIR`` with ``PYTHON_SITE_INSTALL_DIR``.
  - Replace all ``target_link_libraries`` and ``set_target_properties`` with
    ``target_link_libraries(_${COMPONENT_NAME} PUBLIC HOOMD::_hoomd)`` (can link
    ``HOOMD::_md``, ``HOOMD::_hpmc``, etc. if necessary).

* Numerous C++ class APIs have changed, been removed, or renamed. Review the
  header files to see new class signatures. These changes may require you to
  update your component accordingly. Some of the more notable changes include:

  - ``Variant`` has been completely rewritten.
  - ``Trigger`` replaces periodic and variable period scheduling.
  - ``NeighborList`` has a ``addRCutMatrix`` method clients must use to specify
    the maximum cutoff radii per type pair.
  - ``timestep`` is now of type ``uint64_t``.
  - ``Saru`` has been removed. Use ``RandomGenerator``.
  - ``RandomGenerator`` is now constructed with a ``Seed`` and ``Counter``
    object that support 64-bit timesteps.
  - ``m_seed`` is no longer present in individual operation objects. Use the
    global seed provided by ``SystemDefinition``.
  - The HPMC integrators have been heavily refactored.
  - HPMC GPU kernels are now instantiated by template .cu files that are generated by CMake at
    configure time.
  - ``ParticleGroup`` instances are now constructed from immutable, reusable,
    and user-customizable ``ParticleFilter`` instances.
  - All GPU code is now written with HIP to support NVIDIA and AMD GPUs.
  - ``ActiveForceCompute`` always uses particle orientation in combination with
    per-type active forces and torques.
  - ``getProvidedLogQuantities`` and ``getLogQuantities`` have been removed. Provide loggable
    properties instead.
  - Removed the Sphere, Ellipsoid, and oneD constraints. Replaced with the more general RATTLE
    integration methods and Manifold classes.
  - Removed the Enforce2D and TempRescale Updaters. Enforce2D is not needed for 2D simulations,
    and TempRescale has been replaced by ``thermalize_`` methods.
  - Removed Doxygen configuration scripts. View the document for classes in the source files.
  - Particle types may no longer be added after a Simulation is initialized. Classes no longer
    need to subscribe to the types added signal and reallocate data structures when the number of
    types changes.
