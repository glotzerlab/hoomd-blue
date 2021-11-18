Migrating to HOOMD v3
=====================

HOOMD v3 introduces many breaking changes for both users and developers
in order to provide a cleaner Python interface, enable new functionalities, and
move away from unsupported tools. This guide highlights those changes.

Overview of API changes
-----------------------

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
   * - ``hoomd.compute.thermo``
     - ``hoomd.md.compute.ThermodynamicQuantities``
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
     - ``hoomd.hpmc.compute.SDF``
   * - ``hoomd.hpmc.data``
     - HPMC integrator properties.
   * - ``hoomd.hpmc.util``
     - ``hoomd.hpmc.tune``
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
     - ``hoomd.hpmc.pair.user``
   * - ``hoomd.jit.external``
     - ``hoomd.hpmc.external.user``

Removed functionality
---------------------

HOOMD v3 removes old APIs, unused functionality, and features better served by other codes.

Commands and features deprecated in v2.x are removed in v3.0.

:py:mod:`hoomd`:

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
   * - Python 2.7
     - Python >= 3.6
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
     - `mBuild <https://mosdef-hub.github.io/mbuild/>`_, `packmol <https://www.ime.unicamp.br/~martinez/packmol/userguide.shtml>`_, or user script.
   * - ``deprecated.init.create_random_polymers``
     - `mBuild <https://mosdef-hub.github.io/mbuild/>`_, `packmol <https://www.ime.unicamp.br/~martinez/packmol/userguide.shtml>`_, or user script.

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

``hoomd.cgcmm``:

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
   * - ``cgcmm.angle.cgcmm``
     - no longer needed
   * - ``cgcmm.pair.cgcmm``
     - no longer needed


Not yet ported
--------------

The following v2 functionalities have not yet been ported to the v3 API. They may be added in a
future 3.x release. These contributed functionalities rely on the community for support. Please
contact the developers if you have an interest in porting these:

- ``hoomd.hdf5``
- ``hoomd.metal``
- ``hoomd.mpcd``
- getar file format support


Compiling
---------

* CMake 3.8 or newer is required to build HOOMD.
* To compile with GPU support, use the option ``ENABLE_GPU=ON``.
* ``UPDATE_SUBMODULES`` no longer exists. Users and developers should use
  ``git clone --recursive``, ``git submodule update`` and ``git submodule sync``
  as appropriate.
* ``COPY_HEADERS`` no longer exists. Users must install HOOMD for use
  with external components.
* ``CMAKE_INSTALL_PREFIX`` is set to the Python ``site-packages`` directory (if
  not explicitly set by the user).
* **cereal**, **eigen**, and **pybind11** headers must be provided to build
  HOOMD. See :doc:`installation` for details.
* ``BUILD_JIT`` is replaced with ``ENABLE_LLVM``.

Components
----------

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
