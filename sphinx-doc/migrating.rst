Migrating to HOOMD v3
=====================

HOOMD v3 introduces many breaking changes for both users and developers
in order to provide a cleaner python interface, enable new functionalities, and
move away from unsupported tools. This guide highlights those changes.

Removed functionality
---------------------

HOOMD v3 removes old APIs and unused functionality. See :doc:`deprecated` for a
full list.

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
     - `hoomd.State` ``create_from_`` factory methods
   * - ``hoomd.lattice``
     - *Removed.* Use an external tool.
   * - ``hoomd.meta``
     - `hoomd.logging.Logger` logs operation's ``state`` dictionaries.
   * - ``hoomd.option``
     - *Removed.* Use Python standard libraries for option parsing.
   * - ``hoomd.update``
     - Some classes have been moved to `hoomd.tune`.
   * - ``hoomd.util``
     -  Enable GPU profiling with `hoomd.device.GPU.enable_profiling`.
   * - ``hoomd.hdf5``
     - *Removed.* A future release may re-implement HDF5 logging.
   * - ``hoomd.hpmc.analyze.sdf``
     - ``hoomd.hpmc.compute.SDF``
   * - ``hoomd.hpmc.data``
     - HPMC integrator properties.
   * - ``hoomd.hpmc.util``
     - ``hoomd.hpmc.tune``
   * - ``hoomd.md.integrate.mode_standard``
     - `hoomd.md.Integrator`

Compiling
---------

* CMake 3.8 or newer is required to build HOOMD.
* To compile with GPU support, use the option ``ENABLE_GPU=ON``.
* ``UPDATE_SUBMODULES`` no longer exists. Users and developers should use
  ``git clone --recursive``, ``git submodule update`` and ``git submodule sync``
  as appropriate.
* ``COPY_HEADERS`` no longer exists. Users must ``make install`` HOOMD for use
  with external components.
* ``CMAKE_INSTALL_PREFIX`` is set to the python ``site-packages`` directory (if
  not explicitly set by the user).
* **cereal**, **eigen**, and **pybind11** headers must be provided to build
  HOOMD. See :doc:`installation` for details.

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
