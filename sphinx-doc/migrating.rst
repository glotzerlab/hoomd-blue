Migrating to HOOMD v3
=====================

HOOMD v3 introduces a number of breaking changes for both users and developers in order to provide a cleaner
python interface, enable new functionalities, and move away from unsupported tools. This guide highlights
those changes.

Removed functionality
---------------------

HOOMD v3 removes old APIs and unused functionality. See :doc:`deprecated` for a full list.

Compiling
---------

* CMake 3.8 or newer is required to build HOOMD.
* To compile with GPU support, use the option ``ENABLE_GPU=ON``.
* ``UPDATE_SUBMODULES`` no longer exists. Users and developers should use ``git clone --recursive``,
  ``git submodule update`` and ``git submodule sync`` as appropriate.
* ``COPY_HEADERS`` no longer exists. Users must ``make install`` HOOMD for use with external components.
* ``CMAKE_INSTALL_PREFIX`` is set to the python ``site-packages`` directory (if not explicitly set by the user).
* **cereal**, **eigen**, and **pybind11** headers must be provided to build HOOMD. See :doc:`installation` for details.

Components
----------

* HOOMD now uses native CUDA support in CMake. Use ``CMAKE_CUDA_COMPILER`` to specify a specific ``nvcc`` or ``hipcc``. Plugins
  will require updates to ``CMakeLists.txt`` to compile ``.cu`` files.

  - Remove ``CUDA_COMPILE``.
  - Pass ``.cu`` sources directly to ``pybind11_add_module``.
  - Add ``NVCC`` as a compile definition to ``.cu`` sources.

* External components require additional updates to work with v3. See ``example_plugin`` for details:

  - Remove ``FindHOOMD.cmake``.
  - Replace ``include(FindHOOMD.cmake)`` with ``find_package(HOOMD 3.Y REQUIRED)`` (where 3.Y is the minor version this
    plugin is compatible with).
  - Always force set ``CMAKE_INSTALL_PREFIX`` to ``${HOOMD_INSTALL_PREFIX}``.
  - Replace ``PYTHON_MODULE_BASE_DIR`` with ``PYTHON_SITE_INSTALL_DIR``.
  - Replace all ``target_link_libraries`` and ``set_target_properties`` with
    ``target_link_libraries(_${COMPONENT_NAME} PUBLIC HOOMD::_hoomd)`` (can link ``HOOMD::_md``, ``HOOMD::_hpmc``,
    etc. if necessary).
